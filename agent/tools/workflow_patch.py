"""Workflow patching tools.

Apply RFC6902 JSON patches to ComfyUI workflows with full undo support.
The agent reasons about what changes to make, then calls these tools
with concrete patch operations.

State is session-scoped: load a workflow once, patch it multiple times,
undo as needed, then save or execute. The CognitiveGraphEngine is stored
in the session alongside the workflow data, enabling multi-session isolation.

CognitiveGraphEngine integration: When the cognitive module is available,
all mutations are tracked as LIVRPS delta layers for non-destructive
composition. The engine runs alongside the existing state mechanism —
_get_state()["current_workflow"] is always kept in sync.
"""

import copy
import json
import logging
import os
import shutil
import tempfile
from collections import deque
from pathlib import Path

import jsonpatch

try:
    from cognitive.core.graph import CognitiveGraphEngine
    _HAS_COGNITIVE = True
except (ImportError, AttributeError, ModuleNotFoundError):
    CognitiveGraphEngine = None  # type: ignore[assignment, misc]
    _HAS_COGNITIVE = False

from ._util import to_json
from ..workflow_session import get_session
from .._conn_ctx import current_conn_session

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session-scoped state (per-connection workflow isolation)
# ---------------------------------------------------------------------------

# _get_state() is called per-operation instead of binding at import time.
# This prevents stale-reference bugs when the session registry is cleared
# (e.g., in tests or after a server restart). The session ID is read from
# the _conn_session ContextVar, which is set per-connection by:
#   - mcp_server.py    (one ID per MCP client connection)
#   - routes.py        (one ID per WebSocket / HTTP conversation)
#   - cli.py / tests   (no contextvar set → falls back to "default")
# This means the sidebar and MCP server each get their own isolated workflow
# state instead of trampling each other through a shared "default" slot.
def _get_state():
    """Return the live WorkflowSession for the current connection."""
    return get_session(current_conn_session())

_MAX_HISTORY = 50


def _create_engine(workflow_data: dict):
    """Create a CognitiveGraphEngine for the given workflow, or raise if unavailable."""
    if not _HAS_COGNITIVE or CognitiveGraphEngine is None:
        raise RuntimeError("CognitiveGraphEngine not available (cognitive module not installed)")
    return CognitiveGraphEngine(workflow_data)


def _get_engine():
    """Get the engine from session state. Thread-safe via _get_state()._lock."""
    return _get_state()["_engine"]


def _set_engine(engine):
    """Set the engine in session state. Thread-safe via _get_state()._lock."""
    _get_state()["_engine"] = engine


def _sync_state_from_engine():
    """Update _get_state()['current_workflow'] from engine's resolved graph.

    MUST be called inside `with _get_state()._lock:`.
    """
    engine = _get_engine()
    if engine is not None:
        _get_state()["current_workflow"] = engine.to_api_json()


def _ensure_loaded() -> str | None:
    """Return error message if no workflow is loaded, else None."""
    if _get_state()["current_workflow"] is None:
        return "No workflow is open. Load a workflow first with load_workflow, then make changes."
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
    except json.JSONDecodeError:
        return "This workflow file appears to be corrupted or incomplete. Try re-exporting it from ComfyUI using Save (API Format)."

    # Extract API format
    if "nodes" in data and isinstance(data["nodes"], list):
        prompt_data = data.get("extra", {}).get("prompt")
        if prompt_data and isinstance(prompt_data, dict):
            api_nodes = {
                k: v for k, v in prompt_data.items()
                if isinstance(v, dict) and "class_type" in v
            }
            _get_state()["format"] = "ui_with_api"
        else:
            return (
                "This workflow was saved without editable data. "
                "Re-export using 'Save (API Format)' in ComfyUI's menu."
            )
    else:
        api_nodes = {
            k: v for k, v in data.items()
            if isinstance(v, dict) and "class_type" in v
        }
        _get_state()["format"] = "api"

    if not api_nodes:
        return "No nodes found in workflow."

    _get_state()["loaded_path"] = path_str
    _get_state()["base_workflow"] = copy.deepcopy(api_nodes)
    _get_state()["current_workflow"] = copy.deepcopy(api_nodes)
    # MoE-R2: deque(maxlen=N) auto-trims undo history; replaces a list +
    # manual `if len > MAX: history = history[-MAX:]` at 4 sites.
    _get_state()["history"] = deque(maxlen=_MAX_HISTORY)

    # Create engine from the loaded workflow (session-scoped)
    _set_engine(_create_engine(api_nodes))

    return None


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "apply_workflow_patch",
        "description": (
            "Modify specific values in a workflow. Load a workflow file first "
            "if needed. Returns what changed and supports undo."
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
                        "Changes to apply. Each needs an operation ('replace'), "
                        "the field path (e.g. '/5/inputs/text'), and the new value."
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
            "Preview what changes would look like WITHOUT applying them. "
            "Shows before/after values for each affected field."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patches": {
                    "type": "array",
                    "description": "Changes to preview (same format as apply_workflow_patch).",
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
                    "description": (
                        "Input name on the target node (e.g. 'model', 'samples', 'clip'). "
                        "For COMFY_AUTOGROW_V3 inputs, use dotted names like 'values.a'."
                    ),
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
                    "description": (
                        "Input field name (e.g. 'seed', 'steps', 'text', 'ckpt_name'). "
                        "For COMFY_AUTOGROW_V3 inputs, use dotted names like 'values.a'."
                    ),
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


def _patches_to_mutations(patches: list[dict]) -> dict[str, dict[str, object]] | None:
    """Try to convert RFC6902 patches to engine mutation format.

    Returns {node_id: {param: value}} or None if patches can't be cleanly
    converted (e.g. remove ops, non-input paths).
    """
    mutations: dict[str, dict[str, object]] = {}
    for patch in patches:
        op = patch.get("op")
        path = patch.get("path", "")
        value = patch.get("value")

        # We can convert replace/add ops targeting /node_id/inputs/param
        if op not in ("replace", "add"):
            return None

        parts = path.strip("/").split("/")
        if len(parts) < 3 or parts[1] != "inputs":
            return None

        node_id = parts[0]
        param_name = "/".join(parts[2:])  # Handle nested paths
        mutations.setdefault(node_id, {})[param_name] = value

    return mutations if mutations else None


def _handle_apply_patch(tool_input: dict) -> str:
    path_str = tool_input.get("path")
    patches = tool_input.get("patches")  # Cycle 48: guard required field
    if patches is None:
        return to_json({"error": "patches is required."})
    if not isinstance(patches, list):
        return to_json({"error": "patches must be a list."})

    # Load workflow if path provided or not loaded yet
    if path_str:
        err = _load_workflow(path_str)
        if err:
            return to_json({"error": err})
    elif _get_state()["current_workflow"] is None:
        return to_json({
            "error": "No workflow loaded. Provide a 'path' to load one first."
        })

    # Validate patches list — each element must be a dict with op and path.
    # A non-dict element (e.g. a string) would cause AttributeError on .get().
    # Catch this early with a clear error rather than an opaque crash. (Cycle 30 fix)
    for i, patch in enumerate(patches):
        if not isinstance(patch, dict):
            return to_json({
                "error": f"patches[{i}] must be a dict (RFC6902 patch object), got {type(patch).__name__}.",
            })
        if "op" not in patch or "path" not in patch:
            return to_json({
                "error": f"patches[{i}] is missing required 'op' or 'path' field.",
            })

    # Record before values for the diff report
    before_values = {}
    for patch in patches:
        p = patch.get("path", "")
        before_values[p] = _get_value_at_path(_get_state()["current_workflow"], p)

    # Save current state for undo (history list kept for backward compat)
    _get_state()["history"].append(copy.deepcopy(_get_state()["current_workflow"]))

    # Try engine-based mutation first
    engine = _get_engine()
    engine_used = False
    if engine is not None:
        mutations = _patches_to_mutations(patches)
        if mutations is not None:
            try:
                engine.mutate_workflow(
                    mutations,
                    opinion="L",
                    description=f"apply_workflow_patch: {len(patches)} patches",
                )
                _sync_state_from_engine()
                engine_used = True
            except Exception as exc:
                log.debug("Engine mutation failed, falling back to jsonpatch: %s", exc)

    # Fallback: apply via jsonpatch (original path)
    if not engine_used:
        try:
            jp = jsonpatch.JsonPatch(patches)
            _get_state()["current_workflow"] = jp.apply(_get_state()["current_workflow"])
        except Exception as e:
            _get_state()["current_workflow"] = _get_state()["history"].pop()
            return to_json({"error": f"Patch failed: {e}"})

        # Rebuild engine from current state to keep it in sync
        if engine is not None:
            _set_engine(_create_engine(_get_state()["current_workflow"]))

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
            jsonpatch.make_patch(_get_state()["base_workflow"], _get_state()["current_workflow"]).patch
        ),
    })


def _handle_preview_patch(tool_input: dict) -> str:
    err = _ensure_loaded()
    if err:
        return to_json({"error": err})

    patches = tool_input.get("patches")
    if not patches:
        return to_json({"error": "'patches' is required and must be a non-empty list."})

    # Preview without modifying state. jsonpatch.JsonPatch.apply defaults to
    # in_place=False so it returns a new document without mutating the input;
    # the previous explicit deepcopy was double-copying. We discard the
    # returned doc — the only thing we need from this call is the
    # exception-raise behavior on invalid patches.
    try:
        jp = jsonpatch.JsonPatch(patches)
        jp.apply(_get_state()["current_workflow"])
    except Exception as e:
        return to_json({"error": f"Patch would fail: {e}"})

    preview = []
    for patch in patches:
        p = patch.get("path", "")
        preview.append({
            "path": p,
            "op": patch.get("op"),
            "current_value": _get_value_at_path(_get_state()["current_workflow"], p),
            "new_value": patch.get("value"),
        })

    return to_json({"preview": preview, "would_succeed": True})


def _handle_undo() -> str:
    err = _ensure_loaded()
    if err:
        return to_json({"error": err})

    if not _get_state()["history"]:
        return to_json({"error": "Nothing to undo."})

    # Pop from history (backward compat)
    _get_state()["current_workflow"] = _get_state()["history"].pop()

    # Pop from engine delta stack too
    engine = _get_engine()
    if engine is not None:
        popped = engine.pop_delta()
        if popped is not None:
            _sync_state_from_engine()
        else:
            # Engine stack empty but history had entries — rebuild engine from restored state
            try:
                _set_engine(_create_engine(_get_state()["current_workflow"]))
            except Exception as exc:  # Cycle 45: guard — undo already succeeded, engine is optional
                log.warning("Could not rebuild engine after undo: %s. Engine disabled.", exc)
                _set_engine(None)

    remaining = len(
        jsonpatch.make_patch(_get_state()["base_workflow"], _get_state()["current_workflow"]).patch
    )
    return to_json({
        "undone": True,
        "remaining_changes_from_base": remaining,
        "undo_steps_remaining": len(_get_state()["history"]),
    })


def _handle_get_diff() -> str:
    err = _ensure_loaded()
    if err:
        return to_json({"error": err})

    diff = jsonpatch.make_patch(_get_state()["base_workflow"], _get_state()["current_workflow"]).patch

    if not diff:
        return to_json({"changes": 0, "message": "No changes from original."})

    return to_json({
        "changes": len(diff),
        "diff": diff,
        "loaded_from": _get_state()["loaded_path"],
    })


def _handle_save(tool_input: dict) -> str:
    err = _ensure_loaded()
    if err:
        return to_json({"error": err})

    output_path = tool_input.get("output_path") or _get_state()["loaded_path"]
    if not output_path:
        return to_json({"error": "No output path specified."})

    from ._util import validate_path
    path_err = validate_path(output_path)
    if path_err:
        return to_json({"error": path_err})

    try:
        content = to_json(_get_state()["current_workflow"], indent=2)
        dest = Path(output_path)
        fd = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=dest.parent,
            suffix=".tmp",
            delete=False,
        )
        try:
            fd.write(content)
            fd.flush()
            os.fsync(fd.fileno())
            fd.close()
            shutil.move(fd.name, str(dest))
        except Exception:
            try:
                Path(fd.name).unlink(missing_ok=True)
            except Exception:
                pass
            raise
    except Exception as e:
        return to_json({"error": f"Failed to save: {e}"})

    changes = len(
        jsonpatch.make_patch(_get_state()["base_workflow"], _get_state()["current_workflow"]).patch
    )
    return to_json({
        "saved": output_path,
        "changes_from_base": changes,
    })


def _handle_reset() -> str:
    err = _ensure_loaded()
    if err:
        return to_json({"error": err})

    _get_state()["current_workflow"] = copy.deepcopy(_get_state()["base_workflow"])
    # MoE-R2: deque(maxlen=N) auto-trims undo history.
    _get_state()["history"] = deque(maxlen=_MAX_HISTORY)

    # Reset engine from base workflow
    if _get_engine() is not None:
        _set_engine(_create_engine(_get_state()["base_workflow"]))

    return to_json({
        "reset": True,
        "loaded_from": _get_state()["loaded_path"],
    })


# ---------------------------------------------------------------------------
# Semantic composition handlers
# ---------------------------------------------------------------------------

def _next_node_id() -> str:
    """Find the next available numeric node ID."""
    workflow = _get_state()["current_workflow"]
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
    class_type = tool_input.get("class_type")  # Cycle 48: guard required field before _ensure_loaded
    if not class_type or not isinstance(class_type, str):
        return to_json({"error": "class_type is required and must be a non-empty string."})

    err = _ensure_loaded()
    if err:
        return to_json({"error": err})
    inputs = tool_input.get("inputs", {})

    # Save state for undo
    _get_state()["history"].append(copy.deepcopy(_get_state()["current_workflow"]))

    node_id = _next_node_id()

    # Use engine if available
    engine = _get_engine()
    if engine is not None:
        try:
            mutations = {node_id: {"class_type": class_type, **inputs}}
            engine.mutate_workflow(
                mutations,
                opinion="L",
                description=f"add_node: {class_type} as {node_id}",
            )
            _sync_state_from_engine()
        except Exception as exc:  # Cycle 45: guard undo stack — engine failure falls back to direct write
            log.debug("Engine mutation failed in add_node, using direct write: %s", exc)
            _get_state()["current_workflow"][node_id] = {
                "class_type": class_type,
                "inputs": inputs,
            }
    else:
        _get_state()["current_workflow"][node_id] = {
            "class_type": class_type,
            "inputs": inputs,
        }

    return to_json({
        "added": True,
        "node_id": node_id,
        "class_type": class_type,
        "inputs": inputs,
        "total_nodes": len(_get_state()["current_workflow"]),
    })


def _handle_connect_nodes(tool_input: dict) -> str:
    for _f in ("from_node", "from_output", "to_node", "to_input"):  # Cycle 48: guard required fields before _ensure_loaded
        if _f not in tool_input:
            return to_json({"error": f"{_f} is required."})
    # Cycle 51: validate from_output format/range before _ensure_loaded so format errors
    # fire regardless of whether a workflow is loaded.
    try:
        _fo_check = int(tool_input["from_output"])
        if _fo_check < 0:
            raise ValueError(f"from_output must be >= 0, got {_fo_check}")
        if _fo_check > 100:
            raise ValueError(f"from_output={_fo_check} is unreasonably large (max 100)")
    except (TypeError, ValueError) as e:
        return to_json({"error": f"Invalid from_output: {e}"})
    err = _ensure_loaded()
    if err:
        return to_json({"error": err})
    from_node = tool_input["from_node"]
    from_output = _fo_check  # already validated above
    to_node = tool_input["to_node"]
    to_input = tool_input["to_input"]

    # Validate from_output is a non-negative integer (ComfyUI output slot index)
    # (range already checked above; keep local alias for clarity downstream)

    workflow = _get_state()["current_workflow"]

    # Validate nodes exist and are dicts (malformed workflows map IDs to non-dict values)
    if from_node not in workflow:
        return to_json({"error": f"Source node '{from_node}' not found in workflow."})
    if to_node not in workflow:
        return to_json({"error": f"Target node '{to_node}' not found in workflow."})
    if not isinstance(workflow[from_node], dict):
        return to_json({"error": f"Malformed workflow: node '{from_node}' is not a dict (got {type(workflow[from_node]).__name__})."})
    if not isinstance(workflow[to_node], dict):
        return to_json({"error": f"Malformed workflow: node '{to_node}' is not a dict (got {type(workflow[to_node]).__name__})."})
    # (Cycle 33 fix)

    # Save state for undo
    _get_state()["history"].append(copy.deepcopy(workflow))

    # Connection value in ComfyUI format
    connection = [from_node, from_output]

    engine = _get_engine()

    # COMFY_AUTOGROW_V3 support: dotted names like "values.a"
    if "." in to_input and not to_input.startswith("."):
        group, _, sub = to_input.partition(".")
        inputs = workflow[to_node].setdefault("inputs", {})
        autogrow_dict = inputs.get(group, {})
        if not isinstance(autogrow_dict, dict):
            autogrow_dict = {}
        old_value = autogrow_dict.get(sub)
        autogrow_dict[sub] = connection
        inputs[group] = autogrow_dict

        if engine is not None:
            try:
                engine.mutate_workflow(
                    {to_node: {group: copy.deepcopy(inputs[group])}},
                    opinion="L",
                    description=f"connect_nodes: {from_node}[{from_output}] -> {to_node}.{to_input}",
                )
                _sync_state_from_engine()
            except Exception as exc:  # Cycle 45: guard undo stack — autogrow direct write already applied
                log.debug("Engine mutation failed in connect_nodes (autogrow), rebuilding: %s", exc)
                _set_engine(_create_engine(_get_state()["current_workflow"]))
    else:
        old_value = workflow[to_node].get("inputs", {}).get(to_input)

        if engine is not None:
            try:
                engine.mutate_workflow(
                    {to_node: {to_input: connection}},
                    opinion="L",
                    description=f"connect_nodes: {from_node}[{from_output}] -> {to_node}.{to_input}",
                )
                _sync_state_from_engine()
            except Exception as exc:  # Cycle 45: guard undo stack — fall back to direct write
                log.debug("Engine mutation failed in connect_nodes, using direct write: %s", exc)
                workflow[to_node].setdefault("inputs", {})[to_input] = connection
        else:
            workflow[to_node].setdefault("inputs", {})[to_input] = connection

    from_class = workflow[from_node].get("class_type", "?")
    to_class = workflow[to_node].get("class_type", "?")

    return to_json({
        "connected": True,
        "from": f"{from_class} [{from_node}] output {from_output}",
        "to": f"{to_class} [{to_node}].{to_input}",
        "previous_value": old_value,
    })


def _handle_set_input(tool_input: dict) -> str:
    for _f in ("node_id", "input_name", "value"):  # Cycle 48: guard required fields before _ensure_loaded
        if _f not in tool_input:
            return to_json({"error": f"{_f} is required."})
    err = _ensure_loaded()
    if err:
        return to_json({"error": err})

    node_id = tool_input["node_id"]
    input_name = tool_input["input_name"]
    value = tool_input["value"]

    workflow = _get_state()["current_workflow"]

    if node_id not in workflow:
        return to_json({"error": f"Node '{node_id}' not found in workflow."})
    if not isinstance(workflow[node_id], dict):
        return to_json({"error": f"Malformed workflow: node '{node_id}' is not a dict (got {type(workflow[node_id]).__name__})."})
    # (Cycle 33 fix)

    # Save state for undo
    _get_state()["history"].append(copy.deepcopy(workflow))

    engine = _get_engine()

    # COMFY_AUTOGROW_V3 support: dotted names like "values.a"
    if "." in input_name and not input_name.startswith("."):
        group, _, sub = input_name.partition(".")
        inputs = workflow[node_id].setdefault("inputs", {})
        autogrow_dict = inputs.get(group, {})
        if not isinstance(autogrow_dict, dict):
            autogrow_dict = {}
        old_value = autogrow_dict.get(sub)
        autogrow_dict[sub] = value
        inputs[group] = autogrow_dict

        if engine is not None:
            try:
                engine.mutate_workflow(
                    {node_id: {group: copy.deepcopy(inputs[group])}},
                    opinion="L",
                    description=f"set_input: {node_id}.{input_name} = {value!r}",
                )
                _sync_state_from_engine()
            except Exception as exc:  # Cycle 45: guard undo stack — autogrow direct write already applied
                log.debug("Engine mutation failed in set_input (autogrow), rebuilding: %s", exc)
                _set_engine(_create_engine(_get_state()["current_workflow"]))
    else:
        old_value = workflow[node_id].get("inputs", {}).get(input_name)

        if engine is not None:
            try:
                engine.mutate_workflow(
                    {node_id: {input_name: value}},
                    opinion="L",
                    description=f"set_input: {node_id}.{input_name} = {value!r}",
                )
                _sync_state_from_engine()
            except Exception as exc:  # Cycle 45: guard undo stack — fall back to direct write
                log.debug("Engine mutation failed in set_input, using direct write: %s", exc)
                workflow[node_id].setdefault("inputs", {})[input_name] = value
        else:
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
    # Capture session once before entering the lock — prevents a race where
    # clear_sessions() (tests only) could swap the registry while we hold the
    # lock, causing _get_state() inside handlers to return a different session.
    _session = _get_state()
    with _session._lock:
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
            "This workflow was saved without editable data. "
            "Re-export using 'Save (API Format)' in ComfyUI's menu."
        )

    if not nodes:
        return "No nodes found in workflow data."

    with _get_state()._lock:
        _get_state()["loaded_path"] = source
        _get_state()["format"] = fmt
        _get_state()["base_workflow"] = copy.deepcopy(nodes)
        _get_state()["current_workflow"] = copy.deepcopy(nodes)
        # MoE-R2: deque(maxlen=N) auto-trims undo history.
        _get_state()["history"] = deque(maxlen=_MAX_HISTORY)

        # Create engine from loaded workflow (session-scoped)
        _set_engine(_create_engine(nodes))

    return None


def get_current_workflow() -> dict | None:
    """Get the current workflow dict (used by comfy_execute)."""
    return _get_state()["current_workflow"]


def get_engine():
    """Get the current CognitiveGraphEngine instance, or None.

    Returns the session-scoped engine. Each WorkflowSession has
    its own engine, preventing multi-session collision.
    """
    return _get_state()["_engine"]
