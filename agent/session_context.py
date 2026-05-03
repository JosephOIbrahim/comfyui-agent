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
        self._dag_state = None  # Optional DAG engine state (lazy)
        self._degradation = None  # Optional DegradationManager (lazy)
        self._autosave_timer = None  # threading.Timer (W1.3)
        self._autosave_shutdown = threading.Event()  # signals timer to stop
        self._experience_accumulator = None  # cognitive ExperienceAccumulator (W1.4)
        self._pipeline = None  # cognitive AutonomousPipeline (W1.4)
        # RLock (not Lock) so nested ensure_*() calls can re-enter — e.g.
        # ensure_ratchet() holds the lock and calls ensure_stage() which
        # also acquires it. With plain Lock this deadlocks; the deadlock
        # was latent until usd-core was installed and the FORESIGHT tests
        # actually exercised the chain.
        self._init_lock = threading.RLock()  # Guards all lazy-init ensure_*() methods
        try:
            import logging as _logging
            from .circuit_breaker import CircuitBreaker
            from .degradation import DegradationManager
            _dm = DegradationManager()
            _log = _logging.getLogger(__name__)

            # Brain layer — graceful degradation if brain module fails to load
            _dm.register(
                "brain",
                fallback=lambda *_a, **_kw: _log.warning(
                    "Brain subsystem unavailable — skipping"
                ),
                breaker=CircuitBreaker(
                    name="brain", failure_threshold=3, recovery_timeout=30.0
                ),
            )

            # ComfyUI HTTP — graceful degradation if ComfyUI is unreachable
            _dm.register(
                "comfyui_http",
                fallback=lambda *_a, **_kw: {
                    "error": (
                        "ComfyUI is not reachable. "
                        "Make sure it's running on the configured host/port."
                    )
                },
                breaker=CircuitBreaker(
                    name="comfyui_http", failure_threshold=5, recovery_timeout=60.0
                ),
            )

            self._degradation = _dm
        except ImportError:
            pass

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

        Honors STAGE_DEFAULT_PATH from .env: when set, the stage loads from
        that .usda file on cold start (existing logic in CognitiveWorkflowStage
        constructor at cognitive_stage.py:109-112) and uses it as the default
        flush target.

        Honors STAGE_AUTOSAVE_SECONDS from .env: when > 0, a daemon
        threading.Timer flushes the stage on a periodic interval.

        Honors STAGE_AUTOLOAD_EXPERIENCE from .env: when true, the cognitive
        ExperienceAccumulator is loaded from EXPERIENCE_FILE so that
        accumulated experience persists across sessions (closes the README's
        "Sessions 100+ driven by personal history" claim).
        """
        if self._stage is None:
            with self._init_lock:
                if self._stage is None:
                    try:
                        from .config import STAGE_DEFAULT_PATH
                        from .stage import CognitiveWorkflowStage, HAS_USD
                        if HAS_USD:
                            root = STAGE_DEFAULT_PATH or None
                            self._stage = CognitiveWorkflowStage(
                                root_path=root,
                            )
                            self._start_autosave_timer()
                            self._maybe_autoload_experience()
                    except ImportError:
                        pass
        return self._stage

    # ------------------------------------------------------------------
    # W1.3 — autosave daemon timer
    # ------------------------------------------------------------------

    def _start_autosave_timer(self) -> None:
        """Start a daemon Timer that flushes the stage every N seconds.

        No-op if STAGE_AUTOSAVE_SECONDS <= 0 or the stage has no _root_path
        (no flush target). Uses a self-rescheduling threading.Timer so we
        don't introduce asyncio into a thread-only codebase.
        """
        try:
            from .config import STAGE_AUTOSAVE_SECONDS
        except ImportError:
            return
        if STAGE_AUTOSAVE_SECONDS <= 0:
            return
        if self._stage is None or self._stage._root_path is None:
            return  # nothing to flush to

        import logging as _logging
        _log = _logging.getLogger(__name__)

        def _tick():
            if self._autosave_shutdown.is_set():
                return
            try:
                if self._stage is not None and self._stage._root_path is not None:
                    self._stage.flush()
            except Exception as exc:
                _log.warning("Autosave flush failed: %s", exc)
            finally:
                # Reschedule
                if not self._autosave_shutdown.is_set():
                    t = threading.Timer(STAGE_AUTOSAVE_SECONDS, _tick)
                    t.daemon = True
                    self._autosave_timer = t
                    t.start()

        t = threading.Timer(STAGE_AUTOSAVE_SECONDS, _tick)
        t.daemon = True
        self._autosave_timer = t
        t.start()

    def stop_autosave(self) -> None:
        """Cancel the autosave timer (used by atexit / shutdown)."""
        self._autosave_shutdown.set()
        if self._autosave_timer is not None:
            try:
                self._autosave_timer.cancel()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # W1.4 — lazy experience accumulator
    # ------------------------------------------------------------------

    def _maybe_autoload_experience(self) -> None:
        """If STAGE_AUTOLOAD_EXPERIENCE=true, load the cognitive accumulator.

        Wires the dormant create_default_pipeline() into the live runtime so
        comfy-cozy-experience.jsonl is read on first stage init and saved
        after each generation (handled inside cognitive/pipeline/autonomous.py).
        """
        try:
            from .config import STAGE_AUTOLOAD_EXPERIENCE
        except ImportError:
            return
        if not STAGE_AUTOLOAD_EXPERIENCE:
            return
        try:
            from cognitive.pipeline import create_default_pipeline
            self._pipeline = create_default_pipeline()
            self._experience_accumulator = self._pipeline._accumulator
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Experience autoload failed: %s", exc
            )

    @property
    def experience_accumulator(self):
        """Cognitive ExperienceAccumulator for this session, or None."""
        return self._experience_accumulator

    @property
    def pipeline(self):
        """Cognitive AutonomousPipeline for this session, or None."""
        return self._pipeline

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
            with self._init_lock:
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
            with self._init_lock:
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
            with self._init_lock:
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

    def ensure_dag(self):
        """Get or create the DAG engine state for this session.

        Lazy-initialized. Returns None if networkx unavailable or DAG_ENABLED=False.
        """
        if self._dag_state is None:
            with self._init_lock:
                if self._dag_state is None:
                    try:
                        from .config import DAG_ENABLED
                        if not DAG_ENABLED:
                            return None
                        from .stage.dag import build_dag
                        self._dag_state = {"dag": build_dag()}
                    except ImportError:
                        pass
        return self._dag_state

    @property
    def observation_log(self):
        """WorkflowObservationLog for this session, or None."""
        return getattr(self.workflow, '_observation_log', None)

    @property
    def degradation(self):
        """DegradationManager for this session, or None."""
        return self._degradation

    @property
    def graph_engine(self):
        """CognitiveGraphEngine for this session, or None.

        Engine is stored in the WorkflowSession (self.workflow["_engine"]),
        not on SessionContext. This property delegates for convenience.
        """
        return self.workflow.get("_engine")

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

    def get_or_create_with_is_new(self, session_id: str = "default") -> tuple[SessionContext, bool]:
        """Atomically get-or-create a session, returning (ctx, is_new).

        Both the existence check and the creation happen under the same lock
        acquisition, eliminating the TOCTOU race that exists when callers call
        get() then get_or_create() separately. (Cycle 30 fix)
        """
        with self._lock:
            is_new = session_id not in self._sessions
            if is_new:
                self._sessions[session_id] = SessionContext(session_id=session_id)
            ctx = self._sessions[session_id]
            ctx.touch()
            return ctx, is_new

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
    """Get or create a session context. Convenience wrapper.

    On first creation of the default session, runs auto-initialization
    (model scan, workflow load, session restore) if configured via env vars.
    """
    _registry.gc_stale()  # Clean up sessions idle > 1 hour
    # Use the atomic get_or_create_with_is_new() so is_new is determined inside
    # the same lock acquisition as the creation. The old two-step pattern
    # (get() → get_or_create()) had a TOCTOU race: another thread could create
    # the session between the two calls, making is_new a false-positive and
    # triggering a duplicate run_auto_init(). (Cycle 30 fix)
    ctx, is_new = _registry.get_or_create_with_is_new(session_id)

    # Auto-init on first default session creation
    if is_new and session_id == "default":
        try:
            from .startup import run_auto_init
            run_auto_init(ctx)
        except Exception:
            import logging
            logging.getLogger(__name__).debug(
                "Auto-init skipped or failed", exc_info=True,
            )

    return ctx


def iter_sessions() -> list[SessionContext]:
    """Snapshot every live SessionContext.

    Returns a list (not a generator) so the caller can iterate without holding
    the registry lock — useful from atexit handlers where the registry may be
    torn down during iteration.
    """
    with _registry._lock:
        return list(_registry._sessions.values())


def get_registry() -> SessionRegistry:
    """Get the global session registry."""
    return _registry
