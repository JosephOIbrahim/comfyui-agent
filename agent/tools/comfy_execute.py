"""Workflow execution tools.

Queue workflows to ComfyUI, track execution, retrieve outputs.
Uses HTTP polling of /history for completion detection (simpler
and more reliable than WebSocket in a tool context).
"""

import json
import time
import uuid

import httpx

from ..config import COMFYUI_URL
from ._util import to_json

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
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
# Handlers
# ---------------------------------------------------------------------------

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
        if name == "execute_workflow":
            return _handle_execute_workflow(tool_input)
        elif name == "get_execution_status":
            return _handle_get_execution_status(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return to_json({"error": str(e)})
