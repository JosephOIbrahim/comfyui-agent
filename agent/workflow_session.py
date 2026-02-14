"""Workflow session state container with per-session isolation.

Provides a dict-like WorkflowSession class that replaces the module-level
_state dict in workflow_patch.py. Enables future multi-session support
via a session registry keyed by session_id.

Each session holds: loaded_path, base_workflow, current_workflow, history,
and format — the same keys as the original _state dict.
"""

import copy
import threading


class WorkflowSession:
    """Dict-like workflow state container with per-session locking.

    Implements __getitem__, __setitem__, and .get() so existing code that
    does _state["key"] or _state.get("key") continues to work unchanged.
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self._data: dict = {
            "loaded_path": None,
            "base_workflow": None,
            "current_workflow": None,
            "history": [],
            "format": None,
        }
        self._lock = threading.Lock()

    def __getitem__(self, key: str):
        return self._data[key]

    def __setitem__(self, key: str, value):
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def update(self, other):
        """Update session data from a dict or another WorkflowSession."""
        if isinstance(other, WorkflowSession):
            self._data.update(other._data)
        else:
            self._data.update(other)

    def __deepcopy__(self, memo):
        """Support copy.deepcopy() — copies data but creates a fresh lock."""
        new = WorkflowSession(self.session_id)
        new._data = copy.deepcopy(self._data, memo)
        return new

    def __repr__(self) -> str:
        return f"WorkflowSession(session_id={self.session_id!r}, keys={list(self._data.keys())})"


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

_sessions: dict[str, WorkflowSession] = {}
_registry_lock = threading.Lock()


def get_session(session_id: str = "default") -> WorkflowSession:
    """Get or create a workflow session by ID.

    Thread-safe: concurrent calls with the same session_id return
    the same WorkflowSession instance.
    """
    with _registry_lock:
        if session_id not in _sessions:
            _sessions[session_id] = WorkflowSession(session_id)
        return _sessions[session_id]


def clear_sessions() -> None:
    """Remove all sessions from the registry. Used in tests."""
    with _registry_lock:
        _sessions.clear()
