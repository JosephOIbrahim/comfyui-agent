"""Workflow execution tools.

Queue workflows to ComfyUI, track execution, retrieve outputs.
Supports both HTTP polling (reliable) and WebSocket monitoring
(real-time progress for long renders and demo mode).
"""

import json
import logging
import time
import uuid

import httpx

from ..config import COMFYUI_HOST, COMFYUI_PORT, COMFYUI_URL
from ._util import to_json

log = logging.getLogger(__name__)

try:
    import websockets
    import websockets.sync.client
    _HAS_WS = True
except ImportError:
    _HAS_WS = False

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "validate_before_execute",
        "description": (
            "Pre-flight check before executing a workflow. Validates that: "
            "all node class_types exist in ComfyUI, required inputs have values, "
            "connections have compatible types, and referenced model files exist "
            "on disk. Much faster than getting errors at execution time. "
            "Uses the currently loaded workflow or a file path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path to workflow JSON file. If omitted, validates the "
                        "currently loaded/patched workflow."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "execute_workflow",
        "description": (
            "Execute a workflow on ComfyUI. Can run the currently loaded "
            "(and possibly patched) workflow, or a workflow from a file. "
            "Queues the prompt, waits for completion, and returns the "
            "output filenames. If execution takes longer than the timeout, "
            "returns the prompt_id so you can check later with "
            "get_execution_status."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path to a workflow JSON file. If omitted, uses the "
                        "currently loaded/patched workflow from workflow_patch."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Max seconds to wait for completion (default 120). "
                        "Set higher for video generation or complex workflows."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "execute_with_progress",
        "description": (
            "Execute a workflow with real-time WebSocket progress monitoring. "
            "Returns node-by-node execution updates, progress percentages, and "
            "estimated time. Best for long renders (video, multi-pass, Flux). "
            "Falls back to HTTP polling if WebSocket is unavailable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to workflow JSON. If omitted, uses loaded workflow.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to wait (default 300 for progress mode).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_execution_status",
        "description": (
            "Check the status of a previously queued prompt. "
            "Returns whether it's still running, completed, or errored, "
            "and lists output files if complete."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt_id": {
                    "type": "string",
                    "description": "The prompt_id returned from execute_workflow.",
                },
            },
            "required": ["prompt_id"],
        },
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLIENT_ID = str(uuid.uuid4())


def _load_workflow_from_file(path_str: str) -> tuple[dict | None, str | None]:
    """Load API-format workflow from file. Returns (workflow, error)."""
    from pathlib import Path

    path = Path(path_str)
    if not path.exists():
        return None, f"File not found: {path_str}"

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"

    # Extract API format
    if "nodes" in data and isinstance(data["nodes"], list):
        prompt_data = data.get("extra", {}).get("prompt")
        if prompt_data and isinstance(prompt_data, dict):
            nodes = {
                k: v for k, v in prompt_data.items()
                if isinstance(v, dict) and "class_type" in v
            }
            return nodes, None
        return None, "UI-only format -- can't execute without API data."

    nodes = {
        k: v for k, v in data.items()
        if isinstance(v, dict) and "class_type" in v
    }
    if not nodes:
        return None, "No nodes found in workflow."
    return nodes, None


def _queue_prompt(workflow: dict) -> tuple[str | None, str | None]:
    """Queue a workflow. Returns (prompt_id, error)."""
    payload = {
        "prompt": workflow,
        "client_id": _CLIENT_ID,
    }
    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{COMFYUI_URL}/prompt",
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            prompt_id = data.get("prompt_id")
            if not prompt_id:
                return None, f"No prompt_id in response: {data}"
            return prompt_id, None
    except httpx.ConnectError:
        return None, f"ComfyUI not reachable at {COMFYUI_URL}. Is it running?"
    except httpx.HTTPStatusError as e:
        # ComfyUI returns errors in the response body
        try:
            err_data = e.response.json()
            # Format common errors nicely
            node_errors = err_data.get("node_errors", {})
            if node_errors:
                msgs = []
                for nid, nerr in node_errors.items():
                    class_type = nerr.get("class_type", "?")
                    for exc in nerr.get("errors", []):
                        msgs.append(f"Node [{nid}] {class_type}: {exc.get('message', str(exc))}")
                return None, "Validation errors:\n" + "\n".join(msgs)
            return None, err_data.get("error", str(err_data))
        except Exception:
            return None, f"HTTP {e.response.status_code}: {e.response.text[:300]}"
    except Exception as e:
        return None, str(e)


def _poll_completion(
    prompt_id: str,
    timeout: float,
    poll_interval: float = 1.0,
) -> dict:
    """Poll /history until the prompt completes or times out."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            with httpx.Client() as client:
                resp = client.get(
                    f"{COMFYUI_URL}/history/{prompt_id}",
                    timeout=10.0,
                )
                resp.raise_for_status()
                history = resp.json()
        except Exception:
            time.sleep(poll_interval)
            continue

        if prompt_id in history:
            entry = history[prompt_id]
            status_info = entry.get("status", {})
            status_str = status_info.get("status_str", "unknown")
            completed = status_info.get("completed", False)

            if completed or status_str in ("success", "error"):
                # Extract outputs
                outputs = []
                for node_id, node_out in entry.get("outputs", {}).items():
                    for img in node_out.get("images", []):
                        outputs.append({
                            "type": "image",
                            "filename": img.get("filename"),
                            "subfolder": img.get("subfolder", ""),
                        })
                    for vid in node_out.get("gifs", []):
                        outputs.append({
                            "type": "video",
                            "filename": vid.get("filename"),
                            "subfolder": vid.get("subfolder", ""),
                        })

                return {
                    "status": "complete" if status_str == "success" else status_str,
                    "prompt_id": prompt_id,
                    "outputs": outputs,
                }

        time.sleep(poll_interval)

    return {
        "status": "timeout",
        "prompt_id": prompt_id,
        "message": f"Execution did not complete within {timeout}s. "
                   f"Use get_execution_status to check later.",
    }


# ---------------------------------------------------------------------------
# WebSocket progress monitoring
# ---------------------------------------------------------------------------

def _execute_with_websocket(
    workflow: dict,
    timeout: float = 300,
) -> dict:
    """Execute workflow with real-time WebSocket progress updates."""
    if not _HAS_WS:
        # Fallback to polling
        prompt_id, err = _queue_prompt(workflow)
        if err:
            return {"error": err}
        result = _poll_completion(prompt_id, timeout=timeout)
        result["monitoring"] = "polling_fallback"
        return result

    ws_url = f"ws://{COMFYUI_HOST}:{COMFYUI_PORT}/ws?clientId={_CLIENT_ID}"

    # Queue the prompt first
    prompt_id, err = _queue_prompt(workflow)
    if err:
        return {"error": err}

    progress_log = []
    node_times = {}
    current_node = None
    start_time = time.monotonic()

    try:
        with websockets.sync.client.connect(ws_url, close_timeout=5) as ws:
            ws.recv_bufsize = 16 * 1024 * 1024  # 16MB for preview images
            deadline = time.monotonic() + timeout

            while time.monotonic() < deadline:
                try:
                    raw = ws.recv(timeout=2.0)
                except TimeoutError:
                    continue

                # Skip binary messages (preview images)
                if isinstance(raw, bytes):
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type", "")
                data = msg.get("data", {})

                if msg_type == "status":
                    queue_remaining = data.get("status", {}).get("exec_info", {}).get("queue_remaining", 0)
                    if queue_remaining == 0 and progress_log:
                        # Execution finished
                        break

                elif msg_type == "execution_start":
                    if data.get("prompt_id") == prompt_id:
                        progress_log.append({
                            "event": "start",
                            "elapsed_s": round(time.monotonic() - start_time, 2),
                        })

                elif msg_type == "executing":
                    if data.get("prompt_id") != prompt_id:
                        continue
                    node_id = data.get("node")
                    if node_id is None:
                        # Execution complete
                        progress_log.append({
                            "event": "complete",
                            "elapsed_s": round(time.monotonic() - start_time, 2),
                        })
                        break
                    # Track node timing
                    now = time.monotonic()
                    if current_node and current_node in node_times:
                        node_times[current_node]["end"] = now
                        node_times[current_node]["duration_s"] = round(
                            now - node_times[current_node]["start"], 2
                        )
                    current_node = node_id
                    node_times[node_id] = {"start": now}
                    # Get class_type from workflow
                    class_type = workflow.get(node_id, {}).get("class_type", "Unknown")
                    progress_log.append({
                        "event": "executing_node",
                        "node_id": node_id,
                        "class_type": class_type,
                        "elapsed_s": round(now - start_time, 2),
                    })

                elif msg_type == "progress":
                    if data.get("prompt_id", prompt_id) == prompt_id:
                        value = data.get("value", 0)
                        max_val = data.get("max", 1)
                        pct = round(value / max_val * 100, 1) if max_val else 0
                        progress_log.append({
                            "event": "progress",
                            "node_id": current_node,
                            "value": value,
                            "max": max_val,
                            "pct": pct,
                            "elapsed_s": round(time.monotonic() - start_time, 2),
                        })

                elif msg_type == "execution_error":
                    if data.get("prompt_id") == prompt_id:
                        return {
                            "status": "error",
                            "prompt_id": prompt_id,
                            "error": data.get("exception_message", "Unknown error"),
                            "node_id": data.get("node_id"),
                            "class_type": data.get("node_type"),
                            "progress_log": progress_log,
                            "monitoring": "websocket",
                        }

    except Exception as e:
        log.warning("WebSocket monitoring failed, falling back to polling: %s", e)
        result = _poll_completion(prompt_id, timeout=max(timeout - (time.monotonic() - start_time), 10))
        result["monitoring"] = "polling_fallback"
        result["ws_error"] = str(e)
        return result

    # Finalize node timing
    if current_node and current_node in node_times and "end" not in node_times[current_node]:
        node_times[current_node]["end"] = time.monotonic()
        node_times[current_node]["duration_s"] = round(
            node_times[current_node]["end"] - node_times[current_node]["start"], 2
        )

    total_time = round(time.monotonic() - start_time, 2)

    # Fetch outputs from history
    outputs = []
    try:
        with httpx.Client() as client:
            resp = client.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10.0)
            resp.raise_for_status()
            history = resp.json()
        if prompt_id in history:
            for node_out in history[prompt_id].get("outputs", {}).values():
                for img in node_out.get("images", []):
                    outputs.append({"type": "image", "filename": img.get("filename"), "subfolder": img.get("subfolder", "")})
                for vid in node_out.get("gifs", []):
                    outputs.append({"type": "video", "filename": vid.get("filename"), "subfolder": vid.get("subfolder", "")})
    except Exception:
        pass

    # Build node timing summary
    timing = []
    for nid, t in sorted(node_times.items()):
        class_type = workflow.get(nid, {}).get("class_type", "Unknown")
        timing.append({
            "node_id": nid,
            "class_type": class_type,
            "duration_s": t.get("duration_s", 0),
        })
    timing.sort(key=lambda x: x["duration_s"], reverse=True)

    return {
        "status": "complete",
        "prompt_id": prompt_id,
        "total_time_s": total_time,
        "outputs": outputs,
        "node_timing": timing[:10],
        "slowest_node": timing[0] if timing else None,
        "progress_events": len(progress_log),
        "monitoring": "websocket",
    }


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_validate_before_execute(tool_input: dict) -> str:
    """Pre-flight validation of a workflow before queueing."""
    path_str = tool_input.get("path")

    # Get workflow
    if path_str:
        workflow, err = _load_workflow_from_file(path_str)
        if err:
            return to_json({"error": err})
    else:
        from .workflow_patch import get_current_workflow
        workflow = get_current_workflow()
        if workflow is None:
            return to_json({
                "error": "No workflow loaded. Provide a 'path' or load one first.",
            })

    errors = []
    warnings = []

    # Fetch node registry from ComfyUI
    try:
        with httpx.Client() as client:
            resp = client.get(f"{COMFYUI_URL}/object_info", timeout=30.0)
            resp.raise_for_status()
            object_info = resp.json()
    except httpx.ConnectError:
        return to_json({"error": f"ComfyUI not reachable at {COMFYUI_URL}. Is it running?"})
    except Exception as e:
        return to_json({"error": f"Could not fetch node info: {e}"})

    available_nodes = set(object_info.keys())

    for nid, node in sorted(workflow.items()):
        if not isinstance(node, dict) or "class_type" not in node:
            continue
        class_type = node["class_type"]
        inputs = node.get("inputs", {})

        # Check node exists
        if class_type not in available_nodes:
            errors.append(f"Node [{nid}] {class_type}: not installed in ComfyUI.")
            continue

        node_schema = object_info[class_type]
        required_inputs = node_schema.get("input", {}).get("required", {})

        # Check required inputs are provided
        for req_name, req_spec in required_inputs.items():
            if req_name not in inputs:
                errors.append(
                    f"Node [{nid}] {class_type}: missing required input '{req_name}'."
                )
                continue

            value = inputs[req_name]

            # If it's a connection, skip value checks
            if isinstance(value, list) and len(value) == 2:
                # Validate the source node exists
                src_id = str(value[0])
                if src_id not in workflow:
                    errors.append(
                        f"Node [{nid}] {class_type}.{req_name}: "
                        f"connected to node [{src_id}] which doesn't exist."
                    )
                continue

            # Check model file references
            if isinstance(req_spec, (list, tuple)) and len(req_spec) > 0:
                input_type = req_spec[0]
                if isinstance(input_type, list) and isinstance(value, str):
                    # COMBO type — value must be in the allowed list
                    if value not in input_type:
                        warnings.append(
                            f"Node [{nid}] {class_type}.{req_name}: "
                            f"value '{value}' not in known options. "
                            f"May be valid if models changed since last restart."
                        )

    if errors:
        return to_json({
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "message": "Fix errors before executing.",
        })

    return to_json({
        "valid": True,
        "node_count": len(workflow),
        "errors": [],
        "warnings": warnings,
        "message": "Workflow looks ready to execute.",
    })


def _handle_execute_workflow(tool_input: dict) -> str:
    path_str = tool_input.get("path")
    timeout = tool_input.get("timeout", 120)

    # Get workflow
    if path_str:
        workflow, err = _load_workflow_from_file(path_str)
        if err:
            return to_json({"error": err})
    else:
        # Use currently loaded/patched workflow
        from .workflow_patch import get_current_workflow
        workflow = get_current_workflow()
        if workflow is None:
            return to_json({
                "error": (
                    "No workflow loaded. Either provide a 'path' to a workflow "
                    "file, or load one first with apply_workflow_patch."
                ),
            })

    # Queue it
    prompt_id, err = _queue_prompt(workflow)
    if err:
        return to_json({"error": err})

    # Wait for completion
    result = _poll_completion(prompt_id, timeout=timeout)
    return to_json(result)


def _handle_execute_with_progress(tool_input: dict) -> str:
    path_str = tool_input.get("path")
    timeout = tool_input.get("timeout", 300)

    if path_str:
        workflow, err = _load_workflow_from_file(path_str)
        if err:
            return to_json({"error": err})
    else:
        from .workflow_patch import get_current_workflow
        workflow = get_current_workflow()
        if workflow is None:
            return to_json({
                "error": "No workflow loaded. Provide a 'path' or load one first.",
            })

    result = _execute_with_websocket(workflow, timeout=timeout)
    return to_json(result)


def _handle_get_execution_status(tool_input: dict) -> str:
    prompt_id = tool_input["prompt_id"]

    try:
        with httpx.Client() as client:
            resp = client.get(
                f"{COMFYUI_URL}/history/{prompt_id}",
                timeout=10.0,
            )
            resp.raise_for_status()
            history = resp.json()
    except httpx.ConnectError:
        return to_json({"error": f"ComfyUI not reachable at {COMFYUI_URL}."})
    except Exception as e:
        return to_json({"error": str(e)})

    if prompt_id not in history:
        # Check if it's in the queue
        try:
            with httpx.Client() as client:
                resp = client.get(f"{COMFYUI_URL}/queue", timeout=5.0)
                resp.raise_for_status()
                queue = resp.json()

            running = queue.get("queue_running", [])
            pending = queue.get("queue_pending", [])
            in_running = any(r[1] == prompt_id for r in running if len(r) > 1)
            in_pending = any(p[1] == prompt_id for p in pending if len(p) > 1)

            if in_running:
                return to_json({"status": "running", "prompt_id": prompt_id})
            elif in_pending:
                return to_json({"status": "pending", "prompt_id": prompt_id})
            else:
                return to_json({
                    "status": "not_found",
                    "prompt_id": prompt_id,
                    "message": "Prompt not found in history or queue.",
                })
        except Exception:
            return to_json({
                "status": "unknown",
                "prompt_id": prompt_id,
                "message": "Could not determine status.",
            })

    # Found in history
    entry = history[prompt_id]
    status_info = entry.get("status", {})
    outputs = []
    for node_id, node_out in entry.get("outputs", {}).items():
        for img in node_out.get("images", []):
            outputs.append({
                "type": "image",
                "filename": img.get("filename"),
                "subfolder": img.get("subfolder", ""),
            })
        for vid in node_out.get("gifs", []):
            outputs.append({
                "type": "video",
                "filename": vid.get("filename"),
                "subfolder": vid.get("subfolder", ""),
            })

    return to_json({
        "status": status_info.get("status_str", "unknown"),
        "completed": status_info.get("completed", False),
        "prompt_id": prompt_id,
        "outputs": outputs,
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute a comfy_execute tool call."""
    try:
        if name == "validate_before_execute":
            return _handle_validate_before_execute(tool_input)
        elif name == "execute_workflow":
            return _handle_execute_workflow(tool_input)
        elif name == "execute_with_progress":
            return _handle_execute_with_progress(tool_input)
        elif name == "get_execution_status":
            return _handle_get_execution_status(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return to_json({"error": str(e)})
