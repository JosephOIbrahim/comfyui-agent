"""Session persistence — save and restore agent state across conversations.

Sessions capture:
- Loaded workflow state (path, base, current, patch history)
- Agent notes (observations, preferences, learnings)
- Metadata (timestamps, tool call counts)

File format: JSON in sessions/{name}.json with sort_keys=True
for deterministic serialization (He2025 alignment).
"""

import copy
import json
import time
from pathlib import Path

from ..config import SESSIONS_DIR


def _sessions_dir() -> Path:
    """Ensure sessions directory exists."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR


def save_session(
    name: str,
    *,
    workflow_state: dict | None = None,
    notes: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Save session state to a named JSON file.

    Returns {"saved": path, "size_bytes": n} or {"error": msg}.
    """
    path = _sessions_dir() / f"{name}.json"

    session_data = {
        "name": name,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "workflow": _serialize_workflow_state(workflow_state),
        "notes": notes or [],
        "metadata": metadata or {},
    }

    try:
        content = json.dumps(session_data, sort_keys=True, indent=2)
        path.write_text(content, encoding="utf-8")
        return {"saved": str(path), "size_bytes": len(content)}
    except Exception as e:
        return {"error": f"Failed to save session: {e}"}


def load_session(name: str) -> dict:
    """Load a session from disk.

    Returns the full session dict or {"error": msg}.
    """
    path = _sessions_dir() / f"{name}.json"

    if not path.exists():
        # Suggest available sessions
        available = list_sessions()
        names = [s["name"] for s in available.get("sessions", [])]
        if names:
            return {
                "error": f"Session '{name}' not found.",
                "available": names,
            }
        return {"error": f"Session '{name}' not found. No saved sessions exist."}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except json.JSONDecodeError as e:
        return {"error": f"Corrupt session file: {e}"}
    except Exception as e:
        return {"error": f"Failed to load session: {e}"}


def list_sessions() -> dict:
    """List all saved sessions with metadata.

    Returns {"sessions": [...], "count": n, "directory": path}.
    """
    sessions_dir = _sessions_dir()
    sessions = []

    for path in sorted(sessions_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sessions.append({
                "name": data.get("name", path.stem),
                "saved_at": data.get("saved_at", ""),
                "notes_count": len(data.get("notes", [])),
                "has_workflow": data.get("workflow", {}).get("loaded_path") is not None,
                "file": str(path),
            })
        except Exception:
            sessions.append({
                "name": path.stem,
                "saved_at": "",
                "notes_count": 0,
                "has_workflow": False,
                "file": str(path),
                "error": "corrupt",
            })

    return {
        "sessions": sessions,
        "count": len(sessions),
        "directory": str(sessions_dir),
    }


def add_note(name: str, note: str) -> dict:
    """Add a note to a session (create session if it doesn't exist).

    Returns {"added": True, "total_notes": n} or {"error": msg}.
    """
    path = _sessions_dir() / f"{name}.json"

    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = _empty_session(name)
    else:
        data = _empty_session(name)

    if "notes" not in data:
        data["notes"] = []

    data["notes"].append({
        "text": note,
        "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    data["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        path.write_text(
            json.dumps(data, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return {"added": True, "total_notes": len(data["notes"])}
    except Exception as e:
        return {"error": f"Failed to save note: {e}"}


def restore_workflow_state(session_data: dict) -> dict | None:
    """Extract workflow state from session data for re-loading.

    Returns the workflow state dict, or None if no workflow was saved.
    """
    wf = session_data.get("workflow")
    if not wf or not wf.get("loaded_path"):
        return None
    return wf


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_session(name: str) -> dict:
    return {
        "name": name,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "workflow": {"loaded_path": None, "format": None},
        "notes": [],
        "metadata": {},
    }


def _serialize_workflow_state(state: dict | None) -> dict:
    """Serialize workflow_patch._state for disk storage."""
    if state is None:
        return {"loaded_path": None, "format": None}

    return {
        "loaded_path": state.get("loaded_path"),
        "format": state.get("format"),
        "base_workflow": state.get("base_workflow"),
        "current_workflow": state.get("current_workflow"),
        "history_depth": len(state.get("history", [])),
        # Don't serialize full history — can be large.
        # User can undo from current_workflow vs base.
    }
