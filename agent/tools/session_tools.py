"""Session management tools — persist and restore agent state.

Enables the agent to save its working context (loaded workflows,
notes, metadata) and resume later. Sessions are named JSON files
in the sessions/ directory.
"""

import copy

from ..memory.session import (
    save_session, load_session, list_sessions, add_note,
    restore_workflow_state, save_stage, load_stage,
    save_ratchet, load_ratchet,
)
from ._util import to_json

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "save_session",
        "description": (
            "Save the current working state to a named session. Captures "
            "the loaded workflow (with patches), any notes you've recorded, "
            "and metadata. Use this before ending a conversation so the "
            "user can resume later."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Session name (alphanumeric + hyphens). "
                        "Examples: 'flux-portrait', 'sdxl-video-project'."
                    ),
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "load_session",
        "description": (
            "Load a previously saved session by name. Restores the "
            "workflow state (loaded file, applied patches) and surfaces "
            "any notes from the previous session. Use at the start of "
            "a conversation to continue where you left off."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Session name to load.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_sessions",
        "description": (
            "List all saved sessions with their metadata — name, "
            "save date, whether a workflow is loaded, and note count."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "add_note",
        "description": (
            "Record a note in the current session for future reference. "
            "Use this to remember observations, user preferences, "
            "model compatibility findings, or workflow-specific tips. "
            "Notes persist across conversations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_name": {
                    "type": "string",
                    "description": "Session name to add the note to.",
                },
                "note": {
                    "type": "string",
                    "description": (
                        "The note text. Be specific and actionable."
                    ),
                },
                "note_type": {
                    "type": "string",
                    "enum": ["preference", "observation", "decision", "tip"],
                    "description": (
                        "Type of note: 'preference' (artist taste), "
                        "'observation' (what worked), 'decision' (explicit choice), "
                        "'tip' (workflow advice). Defaults to 'observation'."
                    ),
                },
            },
            "required": ["session_name", "note"],
        },
    },
]

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_save_session(tool_input: dict) -> str:
    name = tool_input["name"]

    # Capture current workflow state from session
    from ..workflow_session import get_session
    wf_state = get_session("default")
    workflow_state = copy.deepcopy(dict(wf_state.items())) if wf_state.get("current_workflow") else None

    # Preserve existing notes from prior add_note calls
    existing = load_session(name)
    notes = existing.get("notes", []) if "error" not in existing else None

    result = save_session(name, workflow_state=workflow_state, notes=notes)

    # Save CognitiveWorkflowStage alongside JSON if available
    from ..session_context import get_session_context
    ctx = get_session_context("default")
    if ctx.stage is not None:
        stage_result = save_stage(name, ctx.stage)
        if "saved_stage" in stage_result:
            result["stage_saved"] = True

    # Save Ratchet decision history alongside session if available
    if ctx.ratchet is not None and ctx.ratchet.history:
        ratchet_result = save_ratchet(name, ctx.ratchet)
        if "saved_ratchet" in ratchet_result:
            result["ratchet_saved"] = True
            result["ratchet_decisions"] = ratchet_result["decisions"]

    return to_json(result)


def _handle_load_session(tool_input: dict) -> str:
    name = tool_input["name"]
    session = load_session(name)

    if "error" in session:
        return to_json(session)

    # Restore workflow state if present
    wf = restore_workflow_state(session)
    restored_workflow = False

    if wf and wf.get("base_workflow") and wf.get("current_workflow"):
        from ..workflow_session import get_session
        wf_state = get_session("default")
        wf_state["loaded_path"] = wf.get("loaded_path")
        wf_state["base_workflow"] = copy.deepcopy(wf["base_workflow"])
        wf_state["current_workflow"] = copy.deepcopy(wf["current_workflow"])
        wf_state["format"] = wf.get("format")
        wf_state["history"] = []  # Fresh undo history on restore
        restored_workflow = True

    # Load CognitiveWorkflowStage if .usda exists alongside session
    stage = load_stage(name)
    stage_restored = False
    if stage is not None:
        from ..session_context import get_session_context
        ctx = get_session_context("default")
        ctx.stage = stage
        stage_restored = True

    # Load Ratchet decision history if .ratchet.json exists
    ratchet = load_ratchet(name)
    ratchet_restored = False
    if ratchet is not None:
        from ..session_context import get_session_context
        ctx = get_session_context("default")
        ctx._ratchet = ratchet
        ratchet_restored = True

    notes = session.get("notes", [])

    return to_json({
        "loaded": name,
        "saved_at": session.get("saved_at", ""),
        "workflow_restored": restored_workflow,
        "workflow_path": wf.get("loaded_path") if wf else None,
        "stage_restored": stage_restored,
        "ratchet_restored": ratchet_restored,
        "notes": notes,
        "notes_count": len(notes),
    })


def _handle_list_sessions(tool_input: dict) -> str:
    result = list_sessions()
    return to_json(result)


def _handle_add_note(tool_input: dict) -> str:
    session_name = tool_input["session_name"]
    note = tool_input["note"]
    note_type = tool_input.get("note_type", "observation")
    result = add_note(session_name, note, note_type=note_type)
    return to_json(result)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute a session tool call."""
    try:
        if name == "save_session":
            return _handle_save_session(tool_input)
        elif name == "load_session":
            return _handle_load_session(tool_input)
        elif name == "list_sessions":
            return _handle_list_sessions(tool_input)
        elif name == "add_note":
            return _handle_add_note(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return to_json({"error": str(e)})
