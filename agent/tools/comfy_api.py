"""ComfyUI HTTP API tools.

Wraps the ComfyUI REST API as Agent SDK tools so the agent can
query the running ComfyUI instance for node info, system stats,
queue status, and execution history.
"""

import json

import httpx

from ..config import COMFYUI_URL
from ._util import to_json

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
            "Get the complete list of all registered node types in ComfyUI "
            "via GET /object_info. Returns every node's class_type, inputs, "
            "outputs, and category. This can be large — prefer get_node_info "
            "for a specific node."
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
    """GET request to ComfyUI, returns parsed JSON."""
    with httpx.Client() as client:
        resp = client.get(f"{COMFYUI_URL}{path}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_is_running() -> str:
    try:
        stats = _get("/system_stats", timeout=5.0)
        devices = stats.get("devices", [])
        gpu = devices[0].get("name", "unknown") if devices else "no GPU detected"
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

    all_info = _get("/object_info", timeout=30.0)

    # Filter if requested
    if cat_filter or name_filter:
        filtered = {}
        for name, info in sorted(all_info.items()):
            cat = (info.get("category") or "").lower()
            if cat_filter and cat_filter not in cat:
                continue
            if name_filter and name_filter not in name.lower():
                continue
            filtered[name] = {
                "category": info.get("category", ""),
                "display_name": info.get("display_name", name),
                "description": info.get("description", ""),
                "input_types": list((info.get("input", {}).get("required", {})).keys()),
                "output_types": info.get("output", []),
            }
        return to_json({
            "count": len(filtered),
            "nodes": filtered,
        })

    # No filter — return summary (full object_info is huge)
    summary = {}
    for name, info in sorted(all_info.items()):
        summary[name] = {
            "category": info.get("category", ""),
            "display_name": info.get("display_name", name),
        }
    return to_json({
        "count": len(summary),
        "nodes": summary,
        "hint": "Use name_filter or category_filter to narrow results, or get_node_info for details on a specific node.",
    })


def _handle_get_node_info(tool_input: dict) -> str:
    node_type = tool_input["node_type"]
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
            similar = [n for n in all_nodes if node_type.lower() in n.lower()][:10]
            return to_json({
                "error": f"Node type '{node_type}' not found.",
                "similar_nodes": similar,
            })

    return to_json({
        "class_type": node_type,
        "display_name": info.get("display_name", node_type),
        "category": info.get("category", ""),
        "description": info.get("description", ""),
        "input": info.get("input", {}),
        "output": info.get("output", []),
        "output_name": info.get("output_name", []),
        "output_is_list": info.get("output_is_list", []),
    })


def _handle_get_system_stats() -> str:
    stats = _get("/system_stats", timeout=5.0)
    return to_json(stats)


def _handle_get_queue() -> str:
    queue = _get("/queue", timeout=5.0)
    running = queue.get("queue_running", [])
    pending = queue.get("queue_pending", [])
    return to_json({
        "running_count": len(running),
        "pending_count": len(pending),
        "running": [{"prompt_id": r[1]} for r in running] if running else [],
        "pending": [{"prompt_id": p[1]} for p in pending] if pending else [],
    })


def _handle_get_history(tool_input: dict) -> str:
    prompt_id = tool_input.get("prompt_id")
    max_items = tool_input.get("max_items", 5)

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
    for pid, entry in history.items():
        outputs_summary = []
        for node_id, node_out in entry.get("outputs", {}).items():
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
