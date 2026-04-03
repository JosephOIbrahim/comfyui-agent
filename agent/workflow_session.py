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
            "_engine": None,
        }
        self._lock = threading.RLock()
        self._observation_log = None  # Lazy — created on first observe() call
        self._observation_log_checked = False

    def __getitem__(self, key: str):
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: str, value):
        with self._lock:
            self._data[key] = value

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def get(self, key: str, default=None):
        with self._lock:
            return self._data.get(key, default)

    def keys(self):
        with self._lock:
            return list(self._data.keys())

    def values(self):
        with self._lock:
            return list(self._data.values())

    def items(self):
        with self._lock:
            return list(self._data.items())

    def update(self, other):
        """Update session data from a dict or another WorkflowSession."""
        with self._lock:
            if isinstance(other, WorkflowSession):
                self._data.update(other._data)
            else:
                self._data.update(other)

    def __deepcopy__(self, memo):
        """Support copy.deepcopy() — copies data but creates a fresh lock."""
        with self._lock:
            new = WorkflowSession(self.session_id)
            new._data = copy.deepcopy(self._data, memo)
            return new

    def _ensure_observation_log(self):
        """Lazily create observation log on first use. Never raises."""
        if self._observation_log_checked:
            return
        self._observation_log_checked = True
        try:
            from .config import OBSERVATION_ENABLED
            if not OBSERVATION_ENABLED:
                return
            from .workflow_observation_log import WorkflowObservationLog
            self._observation_log = WorkflowObservationLog(self.session_id)
        except Exception:
            pass

    def observe(self, tool_name: str, tool_input: dict) -> "int | None":
        """Record a workflow observation after a tool call. Returns step_index or None."""
        self._ensure_observation_log()
        if self._observation_log is None:
            return None
        try:
            import hashlib
            import json
            import time as _time
            from .workflow_observation import (
                WorkflowObservation, ActionBlock, WorkflowStateBlock,
                WorkflowPhase,
            )
            # Determine phase from workflow state
            phase = WorkflowPhase.EMPTY
            wf = self._data.get("current_workflow")
            if wf:
                phase = WorkflowPhase.LOADED

            # Build observation
            obs = WorkflowObservation(
                session_id=self.session_id,
                timestamp=_time.time(),
                state=WorkflowStateBlock(
                    phase=phase,
                    node_count=len(wf) if wf else 0,
                ),
                action=ActionBlock(
                    tool_name=tool_name,
                    tool_input_hash=hashlib.sha256(
                        json.dumps(tool_input, sort_keys=True, default=str).encode()
                    ).hexdigest()[:16],
                    action_type=tool_name.split("_")[0] if tool_name else "",
                ),
            )
            return self._observation_log.author(obs)
        except Exception:
            return None

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"WorkflowSession(session_id={self.session_id!r},"
                f" keys={list(self._data.keys())})"
            )


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
