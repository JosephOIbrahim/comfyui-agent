"""Workflow patching tools.

Apply RFC6902 JSON patches to ComfyUI workflows with full undo support.
The agent reasons about what changes to make, then calls these tools
with concrete patch operations.

State is module-level: load a workflow once, patch it multiple times,
undo as needed, then save or execute.
"""

import copy
import json
from pathlib import Path

import jsonpatch

from ._util import to_json

# ---------------------------------------------------------------------------
# Module-level state (one active workflow at a time)
# ---------------------------------------------------------------------------
# WorkflowSession is dict-like: _state["key"] and _state.get("key") work.
# The lock is per-session, so the existing `with _state_lock:` pattern
# continues to work for the default session.

from ..workflow_session import get_session

_state = get_session("default")
_state_lock = _state._lock


def _ensure_loaded() -> str | None:
    """Return error message if no workflow is loaded, else None."""
    if _state["current_workflow"] is None:
        return "No workflow loaded. Use apply_workflow_patch with a file path first."
    return None


def _load_workflow(path_str: str) -> str | None:
    """Load a workflow into state. Returns error or None."""
    from ._util import validate_path
    path_err = validate_path(path_str, must_exist=True)
    if path_err:
        return path_err
    path = Path(path_str)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    # Extract API format
    if "nodes" in data and isinstance(data["nodes"], list):
        prompt_data = data.get("extra", {}).get("prompt")
        if prompt_data and isinstance(prompt_data, dict):
            api_nodes = {
                k: v for k, v in prompt_data.items()
                if isinstance(v, dict) and "class_type" in v
            }
            _state["format"] = "ui_with_api"
        else:
            return (
                "UI-only workflow format -- can't patch without API data. "
                "Re-export using 'Save (API Format)' in ComfyUI."
            )
    else:
        api_nodes = {
            k: v for k, v in data.items()
            if isinstance(v, dict) and "class_type" in v
        }
        _state["format"] = "api"

    if not api_nodes:
        return "No nodes found in workflow."

    _state["loaded_path"] = path_str
    _state["base_workflow"] = copy.deepcopy(api_nodes)
    _state["current_workflow"] = copy.deepcopy(api_nodes)
    _state["history"] = []
    return None


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "apply_workflow_patch",
        "description": (
            "Apply RFC6902 JSON patches to a workflow. If no workflow is "
            "loaded yet, provide the file path to load it first. "
            "Patches use the format: "
            '[{"op": "replace", "path": "/node_id/inputs/field", "value": new_value}]. '
            "Returns the changes made and the current state of affected fields. "
            "Supports undo via undo_workflow_patch."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path to workflow JSON file. Required on first call, "
                        "optional after (reuses loaded workflow)."
                    ),
                },
                "patches": {
                    "type": "array",
                    "description": (
                        "RFC6902 patch operations. Each has 'op' (usually 'replace'), "
                        "'path' (e.g. '/5/inputs/text'), and 'value'."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string"},
                            "path": {"type": "string"},
                            "value": {},
                        },
                        "required": ["op", "path", "value"],
                    },
                },
            },
            "required": ["patches"],
        },
    },
    {
        "name": "preview_workflow_patch",
        "description": (
            "Preview what a patch would change WITHOUT applying it. "
            "Shows before/after values for each affected field."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patches": {
                    "type": "array",
                    "description": "RFC6902 patch operations to preview.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string"},
                            "path": {"type": "string"},
                            "value": {},
                        },
                        "required": ["op", "path", "value"],
                    },
                },
            },
            "required": ["patches"],
        },
    },
    {
        "name": "undo_workflow_patch",
        "description": (
            "Undo the last patch operation. Can be called multiple times "
            "to step back through history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_workflow_diff",
        "description": (
            "Show all changes from the original workflow as a JSON patch diff. "
            "Useful for reviewing everything that's been modified."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "save_workflow",
        "description": (
            "Save the current (patched) workflow to a file. "
            "If no output path is given, overwrites the original."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "output_path": {
                    "type": "string",
                    "description": (
                        "Path to save the workflow. If omitted, overwrites the original file."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "reset_workflow",
        "description": (
            "Reset the workflow back to its original state, discarding all patches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    # --- Semantic composition tools ---
    {
        "name": "add_node",
        "description": (
            "Add a new node to the loaded workflow. Assigns a unique node ID "
            "and sets up the node with the given class_type and optional input values. "
            "Returns the assigned node_id so you can connect it to other nodes. "
            "A workflow must be loaded first (via apply_workflow_patch or load from template)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "class_type": {
                    "type": "string",
                    "description": "Node class_type (e.g. 'KSampler', 'CLIPTextEncode', 'CheckpointLoaderSimple').",
                },
                "inputs": {
                    "type": "object",
                    "description": (
                        "Optional initial input values. Keys are input names, "
                        "values are literals (int, float, str). Connections should "
                        "be set separately with connect_nodes."
                    ),
                },
            },
            "required": ["class_type"],
        },
    },
    {
        "name": "connect_nodes",
        "description": (
            "Connect one node's output to another node's input. "
            "This sets the connection in the workflow graph. "
            "Example: connect KSampler output 0 to VAEDecode input 'samples'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "from_node": {
                    "type": "string",
                    "description": "Source node ID (the node producing the output).",
                },
                "from_output": {
                    "type": "integer",
                    "description": "Output index on the source node (usually 0).",
                },
                "to_node": {
                    "type": "string",
                    "description": "Target node ID (the node receiving the input).",
                },
                "to_input": {
                    "type": "string",
                    "description": "Input name on the target node (e.g. 'model', 'samples', 'clip').",
                },
            },
            "required": ["from_node", "from_output", "to_node", "to_input"],
        },
    },
    {
        "name": "set_input",
        "description": (
            "Set a literal input value on a node. For connections between nodes, "
            "use connect_nodes instead. This is for values like seed, steps, "
            "cfg, sampler_name, prompt text, model filenames, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Target node ID.",
                },
                "input_name": {
                    "type": "string",
                    "description": "Input field name (e.g. 'seed', 'steps', 'text', 'ckpt_name').",
                },
                "value": {
                    "description": "The value to set (string, int, float, bool).",
                },
            },
            "required": ["node_id", "input_name", "value"],
        },
    },
]

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _get_value_at_path(workflow: dict, path: str):
    """Navigate a JSON path and return the value, or None."""
    parts = path.strip("/").split("/")
    current = workflow
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            idx = int(part)
            current = current[idx] if idx < len(current) else None
        else:
            return None
        if current is None:
            return None
    return current


def _handle_apply_patch(tool_input: dict) -> str:
    path_str = tool_input.get("path")
    patches = tool_input["patches"]

    # Load workflow if path provided or not loaded yet
    if path_str:
        err = _load_workflow(path_str)
        if err:
            return to_json({"error": err})
    elif _state["current_workflow"] is None:
        return to_json({
            "error": "No workflow loaded. Provide a 'path' to load one first."
        })

    # Record before values for the diff report
    before_values = {}
    for patch in patches:
        p = patch.get("path", "")
        before_values[p] = _get_value_at_path(_state["current_workflow"], p)

    # Save current state for undo
    _state["history"].append(copy.deepcopy(_state["current_workflow"]))

    # Apply patches
    try:
        jp = jsonpatch.JsonPatch(patches)
        _state["current_workflow"] = jp.apply(_state["current_workflow"])
    except Exception as e:
        # Rollback on failure
        _state["current_workflow"] = _state["history"].pop()
        return to_json({"error": f"Patch failed: {e}"})

    # Build change report
    changes = []
    for patch in patches:
        p = patch.get("path", "")
        changes.append({
            "path": p,
            "op": patch.get("op"),
            "before": before_values.get(p),
            "after": patch.get("value"),
        })

    return to_json({
        "applied": len(patches),
        "changes": changes,
        "undo_available": True,
        "total_changes_from_base": len(
            jsonpatch.make_patch(_state["base_workflow"], _state["current_workflow"]).patch
        ),
    })


def _handle_preview_patch(tool_input: dict) -> str:
    err = _ensure_loaded()
    if err:
        return to_json({"error": err})

    patches = tool_input["patches"]

    # Preview without modifying state
    preview = []
    try:
        jp = jsonpatch.JsonPatch(patches)
        jp.apply(copy.deepcopy(_state["current_workflow"]))
    except Exception as e:
        return to_json({"error": f"Patch would fail: {e}"})

    for patch in patches:
        p = patch.get("path", "")
        preview.append({
            "path": p,
            "op": patch.get("op"),
            "current_value": _get_value_at_path(_state["current_workflow"], p),
            "new_value": patch.get("value"),
        })

    return to_json({"preview": preview, "would_succeed": True})


def _handle_undo() -> str:
    err = _ensure_loaded()
    if err:
        return to_json({"error": err})

    if not _state["history"]:
        return to_json({"error": "Nothing to undo."})

    _state["current_workflow"] = _state["history"].pop()

    remaining = len(
        jsonpatch.make_patch(_state["base_workflow"], _state["current_workflow"]).patch
    )
    return to_json({
        "undone": True,
        "remaining_changes_from_base": remaining,
        "undo_steps_remaining": len(_state["history"]),
    })


def _handle_get_diff() -> str:
    err = _ensure_loaded()
    if err:
        return to_json({"error": err})

    diff = jsonpatch.make_patch(_state["base_workflow"], _state["current_workflow"]).patch

    if not diff:
        return to_json({"changes": 0, "message": "No changes from original."})

    return to_json({
        "changes": len(diff),
        "diff": diff,
        "loaded_from": _state["loaded_path"],
    })


def _handle_save(tool_input: dict) -> str:
    err = _ensure_loaded()
    if err:
        return to_json({"error": err})

    output_path = tool_input.get("output_path") or _state["loaded_path"]
    if not output_path:
        return to_json({"error": "No output path specified."})

    from ._util import validate_path
    path_err = validate_path(output_path)
    if path_err:
        return to_json({"error": path_err})

    try:
        Path(output_path).write_text(
            to_json(_state["current_workflow"], indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        return to_json({"error": f"Failed to save: {e}"})

    changes = len(
        jsonpatch.make_patch(_state["base_workflow"], _state["current_workflow"]).patch
    )
    return to_json({
        "saved": output_path,
        "changes_from_base": changes,
    })


def _handle_reset() -> str:
    err = _ensure_loaded()
    if err:
        return to_json({"error": err})

    _state["current_workflow"] = copy.deepcopy(_state["base_workflow"])
    _state["history"] = []

    return to_json({
        "reset": True,
        "loaded_from": _state["loaded_path"],
    })


# ---------------------------------------------------------------------------
# Semantic composition handlers
# ---------------------------------------------------------------------------

def _next_node_id() -> str:
    """Find the next available numeric node ID."""
    workflow = _state["current_workflow"]
    if not workflow:
        return "1"
    existing = set()
    for key in workflow:
        try:
            existing.add(int(key))
        except ValueError:
            pass
    next_id = max(existing, default=0) + 1
    return str(next_id)


def _handle_add_node(tool_input: dict) -> str:
    err = _ensure_loaded()
    if err:
        return to_json({"error": err})

    class_type = tool_input["class_type"]
    inputs = tool_input.get("inputs", {})

    # Save state for undo
    _state["history"].append(copy.deepcopy(_state["current_workflow"]))

    node_id = _next_node_id()
    _state["current_workflow"][node_id] = {
        "class_type": class_type,
        "inputs": inputs,
    }

    return to_json({
        "added": True,
        "node_id": node_id,
        "class_type": class_type,
        "inputs": inputs,
        "total_nodes": len(_state["current_workflow"]),
    })


def _handle_connect_nodes(tool_input: dict) -> str:
    err = _ensure_loaded()
    if err:
        return to_json({"error": err})

    from_node = tool_input["from_node"]
    from_output = tool_input["from_output"]
    to_node = tool_input["to_node"]
    to_input = tool_input["to_input"]

    workflow = _state["current_workflow"]

    # Validate nodes exist
    if from_node not in workflow:
        return to_json({"error": f"Source node '{from_node}' not found in workflow."})
    if to_node not in workflow:
        return to_json({"error": f"Target node '{to_node}' not found in workflow."})

    # Save state for undo
    _state["history"].append(copy.deepcopy(workflow))

    # Set connection (ComfyUI format: [source_node_id_str, output_index_int])
    old_value = workflow[to_node].get("inputs", {}).get(to_input)
    workflow[to_node].setdefault("inputs", {})[to_input] = [from_node, from_output]

    from_class = workflow[from_node].get("class_type", "?")
    to_class = workflow[to_node].get("class_type", "?")

    return to_json({
        "connected": True,
        "from": f"{from_class} [{from_node}] output {from_output}",
        "to": f"{to_class} [{to_node}].{to_input}",
        "previous_value": old_value,
    })


def _handle_set_input(tool_input: dict) -> str:
    err = _ensure_loaded()
    if err:
        return to_json({"error": err})

    node_id = tool_input["node_id"]
    input_name = tool_input["input_name"]
    value = tool_input["value"]

    workflow = _state["current_workflow"]

    if node_id not in workflow:
        return to_json({"error": f"Node '{node_id}' not found in workflow."})

    # Save state for undo
    _state["history"].append(copy.deepcopy(workflow))

    old_value = workflow[node_id].get("inputs", {}).get(input_name)
    workflow[node_id].setdefault("inputs", {})[input_name] = value

    class_type = workflow[node_id].get("class_type", "?")

    return to_json({
        "set": True,
        "node": f"{class_type} [{node_id}]",
        "input": input_name,
        "old_value": old_value,
        "new_value": value,
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute a workflow_patch tool call."""
    with _state_lock:
        try:
            if name == "apply_workflow_patch":
                return _handle_apply_patch(tool_input)
            elif name == "preview_workflow_patch":
                return _handle_preview_patch(tool_input)
            elif name == "undo_workflow_patch":
                return _handle_undo()
            elif name == "get_workflow_diff":
                return _handle_get_diff()
            elif name == "save_workflow":
                return _handle_save(tool_input)
            elif name == "reset_workflow":
                return _handle_reset()
            elif name == "add_node":
                return _handle_add_node(tool_input)
            elif name == "connect_nodes":
                return _handle_connect_nodes(tool_input)
            elif name == "set_input":
                return _handle_set_input(tool_input)
            else:
                return to_json({"error": f"Unknown tool: {name}"})
        except Exception as e:
            return to_json({"error": str(e)})


def load_workflow_from_data(data: dict, source: str = "<sidebar>") -> str | None:
    """Load a workflow from raw dict data (no filesystem I/O).

    Called by the sidebar backend to inject the live ComfyUI graph.
    Returns None on success, error string on failure.
    """
    from .workflow_parse import _extract_api_format

    nodes, fmt = _extract_api_format(data)

    if fmt == "ui_only":
        return (
            "UI-only workflow format -- can't patch without API data. "
            "Re-export using 'Save (API Format)' in ComfyUI."
        )

    if not nodes:
        return "No nodes found in workflow data."

    with _state_lock:
        _state["loaded_path"] = source
        _state["format"] = fmt
        _state["base_workflow"] = copy.deepcopy(nodes)
        _state["current_workflow"] = copy.deepcopy(nodes)
        _state["history"] = []

    return None


def get_current_workflow() -> dict | None:
    """Get the current workflow dict (used by comfy_execute)."""
    return _state["current_workflow"]
