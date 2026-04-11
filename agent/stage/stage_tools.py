"""Stage tools — expose CognitiveWorkflowStage operations as MCP tools.

Six tools for working with the USD-backed workflow stage:
  stage_read               — read a prim attribute (composed value)
  stage_write              — write an attribute to the base layer
  stage_add_delta          — add an agent delta sublayer
  stage_rollback           — remove the N most-recent delta sublayers
  stage_reconstruct_clean  — read base layer content without agent deltas
  stage_list_deltas        — list delta sublayer identifiers

Tool pattern: TOOLS list[dict] + handle(name, tool_input) -> str.
Registered in agent/tools/__init__.py via the stage module import.
"""

from __future__ import annotations

from ..tools._util import to_json

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "stage_read",
        "description": (
            "Read a composed attribute value from the workflow stage. "
            "If attr_name is omitted, returns True/False for prim existence. "
            "Agent delta overrides are included in the resolved value."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prim_path": {
                    "type": "string",
                    "description": "USD prim path (e.g. /workflows/w1/nodes/KSampler).",
                },
                "attr_name": {
                    "type": "string",
                    "description": "Attribute name to read. Omit to check prim existence.",
                },
            },
            "required": ["prim_path"],
        },
    },
    {
        "name": "stage_write",
        "description": (
            "Write an attribute value to the base layer of the workflow stage. "
            "Agent deltas override base-layer values, so use stage_add_delta "
            "to write temporary overrides. Raises if the attribute is anchor-protected."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prim_path": {
                    "type": "string",
                    "description": "USD prim path.",
                },
                "attr_name": {
                    "type": "string",
                    "description": "Attribute name to write.",
                },
                "value": {
                    "description": "Value to write (bool, int, float, or string).",
                },
                "node_type": {
                    "type": "string",
                    "description": (
                        "ComfyUI node class name — enables anchor protection check "
                        "before writing. Optional."
                    ),
                },
            },
            "required": ["prim_path", "attr_name", "value"],
        },
    },
    {
        "name": "stage_add_delta",
        "description": (
            "Add an agent delta sublayer to the workflow stage. Deltas have the "
            "strongest opinion and override base-layer values. Each delta is an "
            "anonymous sublayer that can be rolled back with stage_rollback."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": (
                        "Agent identifier used to label the delta sublayer "
                        "(e.g. 'forge', 'scout')."
                    ),
                },
                "delta": {
                    "type": "object",
                    "description": (
                        "Map of 'prim_path:attr_name' to value. "
                        "Example: {'/workflows/w1/nodes/KSampler:steps': 30}. "
                        "Values must be bool, int, float, or string."
                    ),
                },
            },
            "required": ["agent_name", "delta"],
        },
    },
    {
        "name": "stage_rollback",
        "description": (
            "Remove the N most-recent agent delta sublayers from the stage. "
            "Returns the number of deltas actually removed (may be less than n "
            "if there were fewer active deltas)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_deltas": {
                    "type": "integer",
                    "description": "Number of most-recent deltas to remove.",
                    "minimum": 1,
                },
            },
            "required": ["n_deltas"],
        },
    },
    {
        "name": "stage_reconstruct_clean",
        "description": (
            "Read the base layer of the stage without any agent delta overrides. "
            "Returns a nested dict of {prim_path: {attr_name: value}} showing "
            "only the base-layer opinions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "stage_list_deltas",
        "description": (
            "List the identifiers of all active agent delta sublayers on the stage, "
            "oldest first. Useful for understanding composition stack depth before "
            "calling stage_rollback."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Stage accessor
# ---------------------------------------------------------------------------

def _get_stage(session_id: str | None = None):
    """Return the CognitiveWorkflowStage for this session, or None.

    If session_id is not provided, reads the _conn_session ContextVar so
    each MCP connection / sidebar conversation operates on its own stage.
    Falls back to "default" in CLI / test contexts where no contextvar is
    set.  This mirrors workflow_patch._get_state() and keeps the gate's
    session lookup aligned with what the handler actually mutates.
    """
    from ..session_context import get_session_context
    if session_id is None:
        from .._conn_ctx import current_conn_session
        session_id = current_conn_session()
    ctx = get_session_context(session_id)
    return ctx.ensure_stage()


_NO_STAGE = to_json({
    "error": (
        "CognitiveWorkflowStage is not available. "
        "usd-core may not be installed in this environment."
    )
})


# ---------------------------------------------------------------------------
# Value serialisation helper
# ---------------------------------------------------------------------------

def _to_python(value: object) -> object:
    """Convert a USD attribute value to a JSON-serializable Python type."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_stage_read(tool_input: dict) -> str:
    prim_path = tool_input.get("prim_path")  # Cycle 55: guard required field
    if not prim_path or not isinstance(prim_path, str):
        return to_json({"error": "prim_path is required and must be a non-empty string."})
    attr_name: str | None = tool_input.get("attr_name")

    stage = _get_stage()
    if stage is None:
        return _NO_STAGE

    value = stage.read(prim_path, attr_name)
    return to_json({
        "prim_path": prim_path,
        "attr_name": attr_name,
        "value": value,
    })


def _handle_stage_write(tool_input: dict) -> str:
    prim_path = tool_input.get("prim_path")  # Cycle 55: guard required fields
    if not prim_path or not isinstance(prim_path, str):
        return to_json({"error": "prim_path is required and must be a non-empty string."})
    attr_name = tool_input.get("attr_name")
    if not attr_name or not isinstance(attr_name, str):
        return to_json({"error": "attr_name is required and must be a non-empty string."})
    if "value" not in tool_input:
        return to_json({"error": "value is required."})
    value = tool_input["value"]
    node_type: str | None = tool_input.get("node_type")

    stage = _get_stage()
    if stage is None:
        return _NO_STAGE

    try:
        stage.write(prim_path, attr_name, value, node_type=node_type)
    except Exception as exc:  # noqa: BLE001
        return to_json({"error": str(exc), "prim_path": prim_path, "attr_name": attr_name})

    return to_json({
        "prim_path": prim_path,
        "attr_name": attr_name,
        "value": value,
        "written": True,
    })


def _handle_stage_add_delta(tool_input: dict) -> str:
    agent_name = tool_input.get("agent_name")  # Cycle 55: guard required fields
    if not agent_name or not isinstance(agent_name, str):
        return to_json({"error": "agent_name is required and must be a non-empty string."})
    delta = tool_input.get("delta")
    if delta is None or not isinstance(delta, dict):
        return to_json({"error": "delta is required and must be a dict."})

    stage = _get_stage()
    if stage is None:
        return _NO_STAGE

    try:
        layer_id = stage.add_agent_delta(agent_name, delta)
    except Exception as exc:  # noqa: BLE001
        return to_json({"error": str(exc), "agent_name": agent_name})

    return to_json({
        "agent_name": agent_name,
        "layer_id": layer_id,
        "delta_count": stage.delta_count,
        "keys_applied": len(delta),
    })


def _handle_stage_rollback(tool_input: dict) -> str:
    n_deltas = tool_input.get("n_deltas")  # Cycle 55: guard required field
    if n_deltas is None:
        return to_json({"error": "n_deltas is required."})
    if not isinstance(n_deltas, int) or isinstance(n_deltas, bool):
        return to_json({"error": "n_deltas must be an integer."})

    stage = _get_stage()
    if stage is None:
        return _NO_STAGE

    removed = stage.rollback_to(n_deltas)
    return to_json({
        "requested": n_deltas,
        "removed": removed,
        "remaining_deltas": stage.delta_count,
    })


def _handle_stage_reconstruct_clean(tool_input: dict) -> str:  # noqa: ARG001
    stage = _get_stage()
    if stage is None:
        return _NO_STAGE

    try:
        result = stage.reconstruct_clean()
    except Exception as exc:  # noqa: BLE001
        return to_json({"error": str(exc)})

    serializable: dict = {}
    for prim_path, attrs in result.items():
        serializable[prim_path] = {k: _to_python(v) for k, v in attrs.items()}

    return to_json({
        "prim_count": len(serializable),
        "prims": serializable,
    })


def _handle_stage_list_deltas(tool_input: dict) -> str:  # noqa: ARG001
    stage = _get_stage()
    if stage is None:
        return _NO_STAGE

    deltas = stage.list_deltas()
    return to_json({
        "delta_count": len(deltas),
        "deltas": deltas,
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_DISPATCH = {
    "stage_read": _handle_stage_read,
    "stage_write": _handle_stage_write,
    "stage_add_delta": _handle_stage_add_delta,
    "stage_rollback": _handle_stage_rollback,
    "stage_reconstruct_clean": _handle_stage_reconstruct_clean,
    "stage_list_deltas": _handle_stage_list_deltas,
}


def handle(name: str, tool_input: dict) -> str:
    """Execute a stage tool call. Returns JSON string."""
    try:
        handler = _DISPATCH.get(name)
        if handler is None:
            return to_json({"error": f"Unknown tool: {name}"})
        return handler(tool_input)
    except Exception as exc:  # noqa: BLE001
        return to_json({"error": str(exc)})
