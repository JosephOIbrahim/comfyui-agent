"""Tool registry for the ComfyUI Comfy Cozy Agent.

Each tool module exports:
  TOOLS: list[dict]    -- Anthropic tool schemas
  handle(name, input)  -- Execute a tool call, return result string

Intelligence layers (53 tools) + Brain layer (27 tools) + Stage layer (22 tools) = 103 tools total.
Brain tools are lazily imported to avoid circular dependencies
(brain modules import _util from this package).
"""

import importlib
import logging
import threading
import time

log = logging.getLogger(__name__)

# Intelligence-layer tool module names.  Imported individually so a single
# broken module (missing dependency, syntax error) degrades gracefully instead
# of crashing the entire tool registry.
_INTELLIGENCE_MODULE_NAMES = [
    "comfy_api", "comfy_inspect", "workflow_parse", "workflow_patch",
    "comfy_execute", "comfy_discover", "session_tools", "workflow_templates",
    "civitai_api", "model_compat", "verify_execution", "github_releases",
    "pipeline", "image_metadata", "node_replacement", "comfy_provision",
    "auto_wire", "provision_pipeline",
]
_STAGE_MODULE_NAMES = [
    "provision_tools", "stage_tools", "foresight_tools",
    "compositor_tools", "hyperagent_tools",
]

_MODULES: list = []
for _mod_name in _INTELLIGENCE_MODULE_NAMES:
    try:
        _MODULES.append(importlib.import_module(f".{_mod_name}", package=__name__))
    except Exception as _ie:
        log.warning("Tool module %r failed to import — its tools are unavailable: %s", _mod_name, _ie)

for _mod_name in _STAGE_MODULE_NAMES:
    try:
        _MODULES.append(importlib.import_module(f"..stage.{_mod_name}", package=__name__))
    except Exception as _ie:
        log.warning("Stage module %r failed to import — its tools are unavailable: %s", _mod_name, _ie)

# Intelligence layer tool schemas
_LAYER_TOOLS: list[dict] = []
for _mod in _MODULES:
    _LAYER_TOOLS.extend(_mod.TOOLS)

# Map tool name -> handler module (intelligence layers).
# MoE-R7: detect duplicate registrations at import time. Pre-fix, a tool
# defined in two modules would silently have one handler overwrite the
# other, with the order-of-import determining which won. With the
# warning, registration drift is visible in cold-start logs.
_HANDLERS = {}
for _mod in _MODULES:
    for _tool in _mod.TOOLS:
        _name = _tool["name"]
        if _name in _HANDLERS:
            log.warning(
                "tool registration collision: %r registered by %s, "
                "overwriting prior registration from %s",
                _name, _mod.__name__, _HANDLERS[_name].__name__,
            )
        _HANDLERS[_name] = _mod

# MoE-R7: per-layer count diagnostic. Emits at INFO so ops dashboards can
# alert on drift. Brain layer counts are added by `_ensure_brain()` when
# it lazy-loads; the line below is the stage+intelligence subtotal.
log.info(
    "tool dispatch: %d intelligence/stage tools registered (brain lazy)",
    len(_HANDLERS),
)

# Brain tools are loaded lazily to break the circular import
_brain_loaded = False
_brain_lock = threading.Lock()
_BRAIN_TOOL_NAMES: set[str] = set()


def _ensure_brain():
    """Lazily load brain layer tools (thread-safe)."""
    from ..config import BRAIN_ENABLED
    if not BRAIN_ENABLED:
        return
    global _brain_loaded, _BRAIN_TOOL_NAMES
    if _brain_loaded:
        return
    with _brain_lock:
        if _brain_loaded:  # double-check after acquiring lock
            return
        try:
            from ..brain import ALL_BRAIN_TOOLS
            _BRAIN_TOOL_NAMES.update(t["name"] for t in ALL_BRAIN_TOOLS)
            _brain_loaded = True
        except Exception as _be:
            log.warning(
                "Brain layer unavailable — brain tools will not be registered: %s", _be
            )


# Capability registry (parallel index for smart routing)
_CAPABILITY_REGISTRY = None
_cap_lock = threading.Lock()


def _ensure_capabilities():
    """Lazily build capability registry (thread-safe)."""
    global _CAPABILITY_REGISTRY
    if _CAPABILITY_REGISTRY is not None:
        return _CAPABILITY_REGISTRY
    with _cap_lock:
        if _CAPABILITY_REGISTRY is not None:
            return _CAPABILITY_REGISTRY
        try:
            from .capability_registry import ToolCapabilityRegistry
            from .capability_defaults import build_default_capabilities
            reg = ToolCapabilityRegistry()
            reg.register_batch(build_default_capabilities())
            _CAPABILITY_REGISTRY = reg
        except Exception:
            log.debug("Capability registry not available", exc_info=True)
    return _CAPABILITY_REGISTRY


def select_tools(requirements: dict) -> list[str]:
    """Select tools matching capability requirements. Returns tool names."""
    reg = _ensure_capabilities()
    if reg is None:
        return []
    try:
        caps = reg.select(requirements)
        return [c.tool_name for c in caps]
    except Exception:
        return []


def _record_metric(name: str, status: str, elapsed: float) -> None:
    """Record tool call metrics. Lazy import so metrics failure doesn't break tools."""
    try:
        from ..metrics import tool_call_total, tool_call_duration_seconds
        tool_call_total.inc(tool_name=name, status=status)
        tool_call_duration_seconds.observe(elapsed, tool_name=name)
    except Exception:
        pass  # Metrics failure must never break tool dispatch


def _observe(name: str, tool_input: dict, ctx: "object | None") -> None:
    """Record observation after tool dispatch. Never raises."""
    try:
        if ctx is not None and hasattr(ctx, 'workflow'):
            ctx.workflow.observe(name, tool_input)
    except Exception:
        log.debug("Observation failed for tool %s", name, exc_info=True)


def _get_all_tools() -> list[dict]:
    """Get all tool schemas (intelligence + brain layers)."""
    _ensure_brain()
    if not _brain_loaded:
        return list(_LAYER_TOOLS)
    from ..brain import ALL_BRAIN_TOOLS
    return _LAYER_TOOLS + list(ALL_BRAIN_TOOLS)


# Public API — ALL_TOOLS is a property-like accessor
class _ToolList(list):
    """Lazy tool list that includes brain tools on first access.

    Uses instance-level lock and initialized flag (not class-level) so that
    multiple instances don't share state. Class-level attributes would leave
    the class attribute False after the first instance sets its own instance
    attribute True, causing any second instance to re-initialize. (Cycle 30 fix)
    """

    def __init__(self):
        super().__init__()
        self._initialized = False
        self._lock = threading.Lock()

    def _init_once(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:  # Double-check after acquiring lock
                return
            self.extend(_LAYER_TOOLS)
            _ensure_brain()
            if _brain_loaded:
                try:
                    from ..brain import ALL_BRAIN_TOOLS
                    self.extend(ALL_BRAIN_TOOLS)
                except ImportError:
                    log.warning("Brain layer not available")
            self._initialized = True

    def __iter__(self):
        self._init_once()
        return super().__iter__()

    def __len__(self):
        self._init_once()
        return super().__len__()

    def __getitem__(self, idx):
        self._init_once()
        return super().__getitem__(idx)


ALL_TOOLS = _ToolList()


def handle(
    name: str,
    tool_input: dict,
    *,
    session_id: str | None = None,
    progress: "object | None" = None,
    ctx: "object | None" = None,
) -> str:
    """Dispatch a tool call to the right handler.

    Args:
        name: Tool name to dispatch.
        tool_input: Tool arguments dict.
        session_id: Optional session ID for workflow state isolation.
                    When ctx is not provided, this is used to look up
                    the SessionContext from the global registry.
        progress: Optional progress reporter for long-running tools.
                  Passed through to handlers that support it.
        ctx: Optional SessionContext for session-scoped state.
             When None, falls back to the default session (v1 behavior).
    """
    # Resolve session context if not provided
    if ctx is None and session_id:
        from ..session_context import get_session_context
        ctx = get_session_context(session_id)

    # Ensure brain is loaded BEFORE the gate check — _BRAIN_TOOL_NAMES is
    # populated by _ensure_brain(). Without this, brain tools on their first
    # call are not present in _BRAIN_TOOL_NAMES, so _is_known=False and the
    # gate is silently skipped for high-risk brain tools. (Cycle 28 fix)
    _ensure_brain()

    # Pre-dispatch gate (guarded by kill switch, only for known tools)
    _is_known = name in _HANDLERS or name in _BRAIN_TOOL_NAMES
    try:
        from ..config import GATE_ENABLED
        if GATE_ENABLED and _is_known:
            from ..gate import pre_dispatch_check, GateDecision

            # Determine session_active: either explicit ctx (MCP path with
            # SessionContext), or a workflow loaded in this connection's
            # WorkflowSession.  _get_state() reads the _conn_session
            # ContextVar, which is set per-connection by routes.py and
            # mcp_server.py — so the sidebar's injected workflow lives in
            # its own session and the gate sees it correctly.
            _session_active = ctx is not None
            _has_undo = bool(
                ctx and hasattr(ctx, 'workflow')
                and ctx.workflow.get("history")
            )
            if not _session_active:
                try:
                    from .workflow_patch import _get_state
                    _wf = _get_state().get("current_workflow")
                    if _wf is not None:
                        _session_active = True
                        _has_undo = bool(_get_state().get("history"))
                except Exception:
                    pass

            # Stage-state fallback for stage_* tools.  The workflow-state
            # fallback above misses the case where a CognitiveWorkflowStage
            # exists but no workflow is loaded — stage tools operate on USD
            # stage prims, which can exist independently of workflow_patch
            # state.  Without this, a REVERSIBLE stage tool like stage_write
            # would be incorrectly DENIED by check_consent (no session) or
            # check_reversibility (no workflow_patch undo history) even
            # though the stage itself has its own delta-rollback mechanism.
            if name.startswith("stage_") and not _session_active:
                try:
                    from ..session_context import get_session_context
                    from .._conn_ctx import current_conn_session
                    _stage_ctx = get_session_context(current_conn_session())
                    if _stage_ctx.stage is not None:
                        _session_active = True
                        # Stage delta sublayers provide undo capability
                        _has_undo = True
                except Exception:
                    pass

            gate_result = pre_dispatch_check(
                name, tool_input,
                session_active=_session_active,
                has_undo=_has_undo,
            )
            if gate_result.decision == GateDecision.DENY:
                from ..errors import error_json
                return error_json(
                    f"Gate denied '{name}': {gate_result.reason}",
                    hint="Check prerequisites or try a different approach.",
                )
            elif gate_result.decision == GateDecision.LOCKED:
                from ..errors import error_json
                return error_json(
                    f"'{name}' is a destructive operation and requires "
                    f"explicit confirmation.",
                    hint="This tool cannot be auto-executed.",
                )
            elif gate_result.decision == GateDecision.ESCALATE:
                log.info("Gate escalated '%s' (risk level %d)",
                         name, gate_result.risk_level)
                # Escalation logged but allowed through — the MCP client
                # (Claude) decides whether to confirm with the user.
    except ImportError:
        pass  # Gate not available — degrade silently
    except Exception:
        log.warning("Gate check failed for '%s' — denying for safety", name,
                    exc_info=True)
        from ..errors import error_json
        return error_json(
            f"Gate unavailable for '{name}' — denied for safety. Check logs."
        )

    # Check brain tools (_ensure_brain already called above before gate check)
    if name in _BRAIN_TOOL_NAMES:
        from ..brain import handle as handle_brain
        _t0 = time.monotonic()
        try:
            result = handle_brain(name, tool_input)
            _observe(name, tool_input, ctx)
            _record_metric(name, "ok", time.monotonic() - _t0)
            return result
        except Exception:
            _record_metric(name, "error", time.monotonic() - _t0)
            log.error("Unhandled error in brain tool %s", name, exc_info=True)
            from ..errors import error_json
            return error_json(
                f"Something went wrong with {name}.",
                hint="Check the logs or try again.",
            )

    # Intelligence layer tools
    mod = _HANDLERS.get(name)
    if mod is None:
        log.warning("Unknown tool called: %s", name)
        from ..errors import error_json
        return error_json(f"Unknown tool: {name}", hint="Check the tool name and try again.")
    _t0 = time.monotonic()
    try:
        # Forward progress to any module that accepts it; fall back gracefully
        try:
            result = mod.handle(name, tool_input, progress=progress)
        except TypeError:
            result = mod.handle(name, tool_input)
        _observe(name, tool_input, ctx)
        _record_metric(name, "ok", time.monotonic() - _t0)
        return result
    except Exception:
        _record_metric(name, "error", time.monotonic() - _t0)
        log.error("Unhandled error in tool %s", name, exc_info=True)
        from ..errors import error_json
        return error_json(
            f"Something went wrong with {name}.",
            hint="Check the logs or try again.",
        )
