"""Session persistence — save and restore agent state across conversations.

Sessions capture:
- Loaded workflow state (path, base, current, patch history)
- Agent notes (observations, preferences, learnings)
- Metadata (timestamps, tool call counts)

File format: JSON in sessions/{name}.json with sort_keys=True
for deterministic serialization (He2025 alignment).
"""

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path

from ..config import SESSIONS_DIR

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2

NOTE_TYPES = ("preference", "observation", "decision", "tip")


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
        "schema_version": SCHEMA_VERSION,
        "workflow": _serialize_workflow_state(workflow_state),
        "notes": notes or [],
        "metadata": metadata or {},
    }

    try:
        content = json.dumps(session_data, sort_keys=True, indent=2)
        _atomic_write(path, content)
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
        data = _migrate_session(data)
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


def add_note(name: str, note: str, *, note_type: str = "observation") -> dict:
    """Add a typed note to a session (create session if it doesn't exist).

    Returns {"added": True, "total_notes": n} or {"error": msg}.
    """
    if note_type not in NOTE_TYPES:
        return {
            "error": f"Unknown note type: {note_type}",
            "hint": f"Use one of: {', '.join(NOTE_TYPES)}",
        }

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
        "type": note_type,
        "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    data["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        content = json.dumps(data, sort_keys=True, indent=2)
        _atomic_write(path, content)
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

def _migrate_session(data: dict) -> dict:
    """Upgrade session data from older schema versions to current.

    v0 -> v1: adds schema_version field.
    v1 -> v2: typed notes (preference/observation/decision/tip).
    """
    version = data.get("schema_version", 0)
    if version < 1:
        data["schema_version"] = 1
        log.debug("Migrated session '%s' from v0 to v1", data.get("name", "?"))
    if version < 2:
        notes = data.get("notes", [])
        migrated = []
        for note in notes:
            if isinstance(note, str):
                migrated.append({
                    "text": note,
                    "type": "observation",
                    "added_at": data.get("saved_at", ""),
                })
            elif isinstance(note, dict) and "type" not in note:
                note["type"] = "observation"
                migrated.append(note)
            else:
                migrated.append(note)
        data["notes"] = migrated
        data["schema_version"] = 2
        log.debug(
            "Migrated session '%s' from v1 to v2 (typed notes)",
            data.get("name", "?"),
        )
    return data


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically using temp-file-then-rename.

    Prevents corrupt session files from interrupted writes.
    """
    fd = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    )
    try:
        fd.write(content)
        fd.close()
        shutil.move(fd.name, str(path))
    except Exception:
        # Clean up temp file on failure
        try:
            Path(fd.name).unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _empty_session(name: str) -> dict:
    return {
        "name": name,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "workflow": {"loaded_path": None, "format": None},
        "notes": [],
        "metadata": {},
    }


def save_stage(name: str, stage: "object") -> dict:
    """Save a CognitiveWorkflowStage as a flattened .usda file alongside the session JSON.

    Args:
        name: Session name (matches the .json session file).
        stage: CognitiveWorkflowStage instance.

    Returns:
        {"saved_stage": path} or {"error": msg}.
    """
    path = _sessions_dir() / f"{name}.usda"
    try:
        stage.flush(path)
        return {"saved_stage": str(path)}
    except Exception as e:
        log.warning("Failed to save stage for session '%s': %s", name, e)
        return {"error": f"Failed to save stage: {e}"}


def load_stage(name: str) -> "object | None":
    """Load a CognitiveWorkflowStage from a .usda file if it exists.

    Args:
        name: Session name to look up.

    Returns:
        CognitiveWorkflowStage instance, or None if not available.
    """
    path = _sessions_dir() / f"{name}.usda"
    if not path.exists():
        return None
    try:
        from ..stage import CognitiveWorkflowStage, HAS_USD
        if not HAS_USD:
            return None
        return CognitiveWorkflowStage(str(path))
    except Exception as e:
        log.warning("Failed to load stage for session '%s': %s", name, e)
        return None


def save_ratchet(name: str, ratchet: "object") -> dict:
    """Save Ratchet decision history as a JSON file alongside the session.

    Args:
        name: Session name (matches the .json session file).
        ratchet: Ratchet instance with history to persist.

    Returns:
        {"saved_ratchet": path, "decisions": count} or {"error": msg}.
    """
    path = _sessions_dir() / f"{name}.ratchet.json"
    try:
        history = []
        for d in ratchet.history:
            history.append({
                "delta_id": d.delta_id,
                "kept": d.kept,
                "axis_scores": d.axis_scores,
                "composite": d.composite,
                "timestamp": d.timestamp,
            })
        data = {
            "threshold": ratchet.threshold,
            "weights": ratchet.weights,
            "history": history,
        }
        _atomic_write(path, json.dumps(data, sort_keys=True, indent=2))
        return {"saved_ratchet": str(path), "decisions": len(history)}
    except Exception as e:
        log.warning("Failed to save ratchet for session '%s': %s", name, e)
        return {"error": f"Failed to save ratchet: {e}"}


def load_ratchet(name: str) -> "object | None":
    """Load a Ratchet from a .ratchet.json file if it exists.

    Restores weights, threshold, and replays the decision history.

    Args:
        name: Session name to look up.

    Returns:
        Ratchet instance with restored history, or None.
    """
    path = _sessions_dir() / f"{name}.ratchet.json"
    if not path.exists():
        return None
    try:
        from ..stage.ratchet import Ratchet, RatchetDecision
        data = json.loads(path.read_text(encoding="utf-8"))
        r = Ratchet(
            weights=data.get("weights"),
            threshold=data.get("threshold", 0.5),
        )
        # Replay history
        for d in data.get("history", []):
            r._history.append(RatchetDecision(
                delta_id=d["delta_id"],
                kept=d["kept"],
                axis_scores=d.get("axis_scores", {}),
                composite=d.get("composite", 0.0),
                timestamp=d.get("timestamp", 0.0),
            ))
        return r
    except Exception as e:
        log.warning("Failed to load ratchet for session '%s': %s", name, e)
        return None


def save_experience(name: str, stage: "object") -> dict:
    """Save experience data from a CognitiveWorkflowStage.

    Experience prims live under /experience/ in the USD stage. Since the
    stage is already saved via save_stage(), this extracts a lightweight
    JSON summary of experiences for quick loading without USD.

    Args:
        name: Session name.
        stage: CognitiveWorkflowStage instance.

    Returns:
        {"saved_experience": path, "count": n} or {"error": msg}.
    """
    path = _sessions_dir() / f"{name}.experience.json"
    try:
        from ..stage.experience import query_experience
        chunks = query_experience(stage, limit=10000)
        data = {
            "count": len(chunks),
            "experiences": [c.to_dict() for c in chunks],
        }
        _atomic_write(path, json.dumps(data, sort_keys=True, indent=2))
        return {"saved_experience": str(path), "count": len(chunks)}
    except Exception as e:
        log.warning("Failed to save experience for session '%s': %s", name, e)
        return {"error": f"Failed to save experience: {e}"}


def load_experience(name: str, stage: "object") -> int:
    """Replay saved experiences into a CognitiveWorkflowStage.

    Reads the .experience.json file and re-records each experience
    into the stage's /experience/ prims.

    Args:
        name: Session name.
        stage: CognitiveWorkflowStage to record into.

    Returns:
        Number of experiences replayed, or 0 if file not found.
    """
    path = _sessions_dir() / f"{name}.experience.json"
    if not path.exists():
        return 0
    try:
        from ..stage.experience import record_experience
        data = json.loads(path.read_text(encoding="utf-8"))
        count = 0
        for exp in data.get("experiences", []):
            record_experience(
                stage,
                initial_state=exp.get("initial_state", {}),
                decisions=exp.get("decisions", []),
                outcome=exp.get("outcome", {}),
                context_signature_hash=exp.get("context_signature_hash", ""),
                predicted_outcome=exp.get("predicted_outcome") or None,
                timestamp=exp.get("timestamp", 0.0),
            )
            count += 1
        return count
    except Exception as e:
        log.warning("Failed to load experience for session '%s': %s", name, e)
        return 0


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
