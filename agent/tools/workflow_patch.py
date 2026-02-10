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

_state = {
    "loaded_path": None,       # str: path to the original file
    "base_workflow": None,     # dict: original workflow (immutable)
    "current_workflow": None,  # dict: current working copy
    "history": [],             # list[dict]: previous states for undo
    "format": None,            # str: "api" | "ui_with_api" | "ui_only"
}


def _ensure_loaded() -> str | None:
    """Return error message if no workflow is loaded, else None."""
    if _state["current_workflow"] is None:
        return "No workflow loaded. Use apply_workflow_patch with a file path first."
    return None


def _load_workflow(path_str: str) -> str | None:
    """Load a workflow into state. Returns error or None."""
    path = Path(path_str)
    if not path.exists():
        return f"File not found: {path_str}"

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
        result = jp.apply(copy.deepcopy(_state["current_workflow"]))
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
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute a workflow_patch tool call."""
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
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return to_json({"error": str(e)})


def get_current_workflow() -> dict | None:
    """Get the current workflow dict (used by comfy_execute)."""
    return _state["current_workflow"]
