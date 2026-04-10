"""ComfyUI HTTP API tools.

Wraps the ComfyUI REST API as Agent SDK tools so the agent can
query the running ComfyUI instance for node info, system stats,
queue status, and execution history.
"""

import threading

import httpx

from ..circuit_breaker import COMFYUI_BREAKER
from ..config import COMFYUI_URL
from ._util import to_json

# ---------------------------------------------------------------------------
# Shared HTTP client (connection pool)
# ---------------------------------------------------------------------------

_client_lock = threading.Lock()
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    """Return a shared httpx client for ComfyUI API calls."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(
                    base_url=COMFYUI_URL,
                    timeout=10.0,
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                )
    return _client

# ---------------------------------------------------------------------------
# Tool schemas (Anthropic tool-use format)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "is_comfyui_running",
        "description": (
            "Check if ComfyUI is running and reachable. "
            "Call this first before using other ComfyUI tools."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_all_nodes",
        "description": (
            "Get registered node types in ComfyUI via GET /object_info. "
            "Use format='names_only' (just class_type names) or "
            "'summary' (name+category) to control response size. "
            "Use 'full' for complete input/output schemas. "
            "Prefer get_node_info for a specific node."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category_filter": {
                    "type": "string",
                    "description": (
                        "Optional substring to filter nodes by category "
                        "(e.g. 'sampling', 'loaders', 'image'). "
                        "Case-insensitive."
                    ),
                },
                "name_filter": {
                    "type": "string",
                    "description": (
                        "Optional substring to filter nodes by class_type name "
                        "(e.g. 'KSampler', 'ControlNet'). Case-insensitive."
                    ),
                },
                "format": {
                    "type": "string",
                    "enum": ["names_only", "summary", "full"],
                    "description": (
                        "Response format. 'names_only': just class_type names (smallest). "
                        "'summary': name + category + display_name (default). "
                        "'full': includes input types and output types."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_node_info",
        "description": (
            "Get detailed info for a specific ComfyUI node type: "
            "its required/optional inputs with types and defaults, "
            "output types, category, and description. "
            "Use the exact class_type name (e.g. 'KSampler', 'CLIPTextEncode')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_type": {
                    "type": "string",
                    "description": "Exact class_type name of the node.",
                },
            },
            "required": ["node_type"],
        },
    },
    {
        "name": "get_system_stats",
        "description": (
            "Get ComfyUI system stats: GPU info, VRAM usage, "
            "Python version, and currently loaded models."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_queue_status",
        "description": (
            "Get the current ComfyUI queue: running prompts and pending items."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_history",
        "description": (
            "Get execution history from ComfyUI. Optionally filter by prompt_id. "
            "Returns outputs (images, videos) and execution status."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt_id": {
                    "type": "string",
                    "description": "Specific prompt ID to look up. Omit for recent history.",
                },
                "max_items": {
                    "type": "integer",
                    "description": "Max number of history items to return (default 5).",
                },
            },
            "required": [],
        },
    },
]

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_TIMEOUT = 15.0


def _get(path: str, timeout: float = _TIMEOUT) -> dict:
    """GET request to ComfyUI, returns parsed JSON.

    Raises httpx.ConnectError or httpx.HTTPStatusError on failure.
    Circuit breaker tracks failures and fast-fails when ComfyUI is down.
    """
    breaker = COMFYUI_BREAKER()
    if not breaker.allow_request():
        raise httpx.ConnectError(
            f"ComfyUI has been unreachable. Waiting {breaker.recovery_timeout:.0f}s before retrying. Is ComfyUI still running?"
        )
    try:
        resp = _get_client().get(f"{COMFYUI_URL}{path}", timeout=timeout)
        resp.raise_for_status()
        breaker.record_success()
        try:
            return resp.json()
        except ValueError as e:  # Cycle 43: guard against non-JSON response body
            raise httpx.ConnectError(
                f"ComfyUI returned non-JSON on {path} (HTML error page?): {e}"
            ) from e
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        breaker.record_failure()
        raise httpx.ConnectError(str(e)) from e


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_is_running() -> str:
    try:
        stats = _get("/system_stats", timeout=5.0)
        devices = stats.get("devices", [])
        # Cycle 67: guard non-dict element (malformed API shape)
        gpu = (
            devices[0].get("name", "unknown")
            if devices and isinstance(devices[0], dict)
            else "no GPU detected"
        )
        py_ver = stats.get("system", {}).get("python_version", "unknown")
        return to_json({
            "running": True,
            "url": COMFYUI_URL,
            "gpu": gpu,
            "python": py_ver,
        })
    except Exception as e:
        return to_json({
            "running": False,
            "url": COMFYUI_URL,
            "error": str(e),
        })


def _handle_get_all_nodes(tool_input: dict) -> str:
    cat_filter = (tool_input.get("category_filter") or "").lower()
    name_filter = (tool_input.get("name_filter") or "").lower()
    fmt = tool_input.get("format", "summary")

    all_info = _get("/object_info", timeout=30.0)

    # Apply filters
    filtered_names = []
    for name in sorted(all_info.keys()):
        info = all_info[name]
        if not isinstance(info, dict):  # Cycle 42: guard malformed node entries
            continue
        cat = (info.get("category") or "").lower()
        if cat_filter and cat_filter not in cat:
            continue
        if name_filter and name_filter not in name.lower():
            continue
        filtered_names.append(name)

    # Format output based on requested detail level
    if fmt == "names_only":
        return to_json({
            "count": len(filtered_names),
            "nodes": filtered_names,
        })

    if fmt == "full":
        nodes = {}
        for name in filtered_names:
            info = all_info[name]
            nodes[name] = {
                "category": info.get("category", ""),
                "display_name": info.get("display_name", name),
                "description": info.get("description", ""),
                "input_types": sorted((info.get("input", {}).get("required", {})).keys()),
                "output_types": info.get("output", []),
            }
        return to_json({"count": len(nodes), "nodes": nodes})

    # Default: summary (name + category + display_name)
    nodes = {}
    for name in filtered_names:
        info = all_info[name]
        nodes[name] = {
            "category": info.get("category", ""),
            "display_name": info.get("display_name", name),
        }
    return to_json({
        "count": len(nodes),
        "nodes": nodes,
        "hint": "Use format='names_only' for smaller response, or get_node_info for details on a specific node.",
    })


def _handle_get_node_info(tool_input: dict) -> str:
    node_type = tool_input.get("node_type")  # Cycle 46: guard required field
    if not node_type or not isinstance(node_type, str):
        return to_json({"error": "node_type is required and must be a non-empty string."})
    all_info = _get(f"/object_info/{node_type}", timeout=10.0)

    info = all_info.get(node_type)
    if not info:
        # Try case-insensitive search
        all_nodes = _get("/object_info", timeout=30.0)
        matches = [n for n in all_nodes if n.lower() == node_type.lower()]
        if matches:
            info = all_nodes[matches[0]]
            node_type = matches[0]
        else:
            # Suggest similar names
            # He2025: sort for deterministic suggestion order
            similar = sorted(n for n in all_nodes if node_type.lower() in n.lower())[:10]
            return to_json({
                "error": f"Node type '{node_type}' not found.",
                "similar_nodes": similar,
            })

    result = {
        "class_type": node_type,
        "display_name": info.get("display_name", node_type),
        "category": info.get("category", ""),
        "description": info.get("description", ""),
        "input": info.get("input", {}),
        "output": info.get("output", []),
        "output_name": info.get("output_name", []),
        "output_is_list": info.get("output_is_list", []),
    }

    # Annotate COMFY_AUTOGROW_V3 inputs with dotted-name hints so the
    # agent knows to use "group.sub" notation (e.g. "values.a") when
    # setting inputs or making connections on these nodes.
    autogrow_hints = {}
    for section in ("required", "optional"):
        for inp_name, spec in info.get("input", {}).get(section, {}).items():
            if (
                isinstance(spec, (list, tuple))
                and len(spec) > 0
                and spec[0] == "COMFY_AUTOGROW_V3"
            ):
                tmpl = spec[1] if len(spec) > 1 else {}
                template_info = tmpl.get("template", {})
                names = tmpl.get("names", [])
                min_count = tmpl.get("min", 0)
                # Extract sub-input type from template
                tmpl_inputs = template_info.get("input", {}).get("required", {})
                sub_type = None
                if tmpl_inputs:
                    first_spec = next(iter(tmpl_inputs.values()))
                    if isinstance(first_spec, (list, tuple)) and first_spec:
                        sub_type = first_spec[0]
                autogrow_hints[inp_name] = {
                    "type": "COMFY_AUTOGROW_V3",
                    "sub_input_type": sub_type,
                    "template_names": names[:10],  # cap for readability
                    "min": min_count,
                    "usage": (
                        f"Use dotted names: '{inp_name}.{names[0]}', "
                        f"'{inp_name}.{names[1]}', etc."
                        if len(names) >= 2
                        else f"Use dotted names: '{inp_name}.<name>'"
                    ),
                }

    if autogrow_hints:
        result["autogrow_inputs"] = autogrow_hints

    return to_json(result)


def _handle_get_system_stats() -> str:
    stats = _get("/system_stats", timeout=5.0)
    return to_json(stats)


def _handle_get_queue() -> str:
    queue = _get("/queue", timeout=5.0)
    running = queue.get("queue_running") or []  # Cycle 54: guard against explicit null from API
    pending = queue.get("queue_pending") or []
    return to_json({
        "running_count": len(running),
        "pending_count": len(pending),
        "running": [{"prompt_id": r[1]} for r in running] if running else [],
        "pending": [{"prompt_id": p[1]} for p in pending] if pending else [],
    })


def _handle_get_history(tool_input: dict) -> str:
    prompt_id = tool_input.get("prompt_id")
    try:
        max_items = int(tool_input.get("max_items", 5))  # Cycle 67: guard string input
    except (TypeError, ValueError):
        return to_json({"error": "max_items must be an integer."})
    if prompt_id is not None and not isinstance(prompt_id, str):  # Cycle 54: type guard optional field
        return to_json({"error": "prompt_id must be a string if provided."})

    if prompt_id:
        history = _get(f"/history/{prompt_id}", timeout=10.0)
    else:
        history = _get("/history", timeout=10.0)

    # Trim to max_items
    if not prompt_id and len(history) > max_items:
        # History keys are prompt IDs; take most recent
        keys = list(history.keys())[-max_items:]
        history = {k: history[k] for k in keys}

    # Summarize outputs
    result = {}
    # He2025: sort for deterministic output order
    for pid, entry in sorted(history.items()):
        outputs_summary = []
        for node_id, node_out in sorted(entry.get("outputs", {}).items()):
            for img in node_out.get("images", []):
                outputs_summary.append({
                    "type": "image",
                    "filename": img.get("filename"),
                })
            for vid in node_out.get("gifs", []):
                outputs_summary.append({
                    "type": "video",
                    "filename": vid.get("filename"),
                })
        status_info = entry.get("status", {})
        result[pid] = {
            "status": status_info.get("status_str", "unknown"),
            "completed": status_info.get("completed", False),
            "outputs": outputs_summary,
        }

    return to_json(result)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute a comfy_api tool call."""
    try:
        if name == "is_comfyui_running":
            return _handle_is_running()
        elif name == "get_all_nodes":
            return _handle_get_all_nodes(tool_input)
        elif name == "get_node_info":
            return _handle_get_node_info(tool_input)
        elif name == "get_system_stats":
            return _handle_get_system_stats()
        elif name == "get_queue_status":
            return _handle_get_queue()
        elif name == "get_history":
            return _handle_get_history(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except httpx.ConnectError:
        return to_json({
            "error": f"Could not connect to ComfyUI at {COMFYUI_URL}. Is it running?",
        })
    except httpx.HTTPStatusError as e:
        return to_json({
            "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
        })
    except Exception as e:
        return to_json({"error": str(e)})
