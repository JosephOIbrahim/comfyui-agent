"""Session-scoped state container for the ComfyUI Agent.

Replaces all module-level mutable state with per-session containers.
Each MCP connection (or CLI session) gets its own SessionContext with
isolated workflow state, brain config, and circuit breakers.

The SessionRegistry manages lifecycle: create, get, destroy, GC.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .workflow_session import WorkflowSession


@dataclass
class SessionContext:
    """Per-session state container.

    Holds all mutable state that was previously module-level singletons.
    Each field corresponds to a specific module's state that has been
    migrated to session scope.
    """

    session_id: str
    workflow: WorkflowSession = field(default=None)  # type: ignore[assignment]
    intent_state: dict[str, Any] = field(default_factory=dict)
    iteration_state: dict[str, Any] = field(default_factory=dict)
    demo_state: dict[str, Any] = field(default_factory=dict)
    orchestrator_tasks: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.workflow is None:
            self.workflow = WorkflowSession(self.session_id)
        self._stage = None  # Optional CognitiveWorkflowStage (lazy)
        self._ratchet = None  # Optional Ratchet (lazy, requires stage)
        self._cwm = None  # Optional CWM predict function (lazy)
        self._arbiter = None  # Optional Arbiter instance (lazy)
        self._workflow_signature = None  # Optional WorkflowSignature

    @property
    def stage(self):
        """CognitiveWorkflowStage for this session, or None."""
        return self._stage

    @stage.setter
    def stage(self, value):
        """Set the stage (e.g., when loading from disk or creating fresh)."""
        self._stage = value

    def ensure_stage(self):
        """Get or create the CognitiveWorkflowStage for this session.

        Lazy-initialized: only creates the stage when first requested.
        Returns None if usd-core is not installed.
        """
        if self._stage is None:
            try:
                from .stage import CognitiveWorkflowStage, HAS_USD
                if HAS_USD:
                    self._stage = CognitiveWorkflowStage()
            except ImportError:
                pass
        return self._stage

    @property
    def ratchet(self):
        """Ratchet for this session, or None."""
        return self._ratchet

    def ensure_ratchet(self, **kwargs):
        """Get or create the Ratchet for this session.

        Lazy-initialized: creates a Ratchet on first request. Requires
        the stage to be available (calls ensure_stage internally).
        If FORESIGHT components are available, wires them into the Ratchet.
        Returns None if the stage cannot be initialized.

        Args:
            **kwargs: Forwarded to Ratchet constructor (weights, threshold).
        """
        if self._ratchet is None:
            stage = self.ensure_stage()
            if stage is not None:
                try:
                    from .stage.ratchet import Ratchet
                    # Wire FORESIGHT if available (degradation cascade)
                    cwm = self.ensure_cwm()
                    arbiter = self.ensure_arbiter()
                    self._ratchet = Ratchet(
                        cws=stage,
                        cwm=cwm,
                        arbiter=arbiter,
                        workflow_signature=self._workflow_signature,
                        **kwargs,
                    )
                except ImportError:
                    pass
        return self._ratchet

    @property
    def cwm(self):
        """CWM predict function for this session, or None."""
        return self._cwm

    def ensure_cwm(self):
        """Get or initialize CWM predict function.

        Lazy-initialized: imports cwm.predict on first request.
        Returns None if the module is not available.
        """
        if self._cwm is None:
            try:
                from .stage.cwm import predict
                self._cwm = predict
            except ImportError:
                pass
        return self._cwm

    @property
    def arbiter(self):
        """Arbiter instance for this session, or None."""
        return self._arbiter

    def ensure_arbiter(self):
        """Get or create the Arbiter for this session.

        Lazy-initialized on first request. Returns None if not available.
        """
        if self._arbiter is None:
            try:
                from .stage.arbiter import Arbiter
                self._arbiter = Arbiter()
            except ImportError:
                pass
        return self._arbiter

    @property
    def workflow_signature(self):
        """Current WorkflowSignature, or None."""
        return self._workflow_signature

    @workflow_signature.setter
    def workflow_signature(self, value):
        """Set the workflow signature (e.g., after loading a workflow)."""
        self._workflow_signature = value

    def touch(self):
        """Update last_activity timestamp."""
        self.last_activity = time.time()

    def age_seconds(self) -> float:
        """Seconds since last activity."""
        return time.time() - self.last_activity


class SessionRegistry:
    """Thread-safe registry of active sessions with GC support."""

    def __init__(self):
        self._sessions: dict[str, SessionContext] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str = "default") -> SessionContext:
        """Get existing session or create a new one. Thread-safe."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionContext(session_id=session_id)
            ctx = self._sessions[session_id]
            ctx.touch()
            return ctx

    def get(self, session_id: str) -> SessionContext | None:
        """Get existing session or None."""
        with self._lock:
            ctx = self._sessions.get(session_id)
            if ctx is not None:
                ctx.touch()
            return ctx

    def destroy(self, session_id: str) -> bool:
        """Remove a session. Returns True if it existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        with self._lock:
            return list(self._sessions.keys())

    def gc_stale(self, max_age_seconds: float = 3600.0) -> int:
        """Remove sessions idle longer than max_age_seconds. Returns count removed."""
        now = time.time()
        to_remove = []
        with self._lock:
            for sid, ctx in self._sessions.items():
                if sid == "default":
                    continue  # never GC the default session
                if now - ctx.last_activity > max_age_seconds:
                    to_remove.append(sid)
            for sid in to_remove:
                del self._sessions[sid]
        return len(to_remove)

    @property
    def count(self) -> int:
        """Number of active sessions."""
        with self._lock:
            return len(self._sessions)

    def clear(self) -> None:
        """Remove all sessions. For testing only."""
        with self._lock:
            self._sessions.clear()


# ---------------------------------------------------------------------------
# Global registry singleton (process-level, not session-level)
# ---------------------------------------------------------------------------
_registry = SessionRegistry()


def get_session_context(session_id: str = "default") -> SessionContext:
    """Get or create a session context. Convenience wrapper."""
    return _registry.get_or_create(session_id)


def get_registry() -> SessionRegistry:
    """Get the global session registry."""
    return _registry
