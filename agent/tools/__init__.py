"""Tool registry for the ComfyUI SUPER DUPER Agent.

Each tool module exports:
  TOOLS: list[dict]    -- Anthropic tool schemas
  handle(name, input)  -- Execute a tool call, return result string

Intelligence layers (53 tools) + Brain layer (27 tools) + Stage layer (22 tools) = 103 tools total.
Brain tools are lazily imported to avoid circular dependencies
(brain modules import _util from this package).
"""

import logging
import threading

from . import comfy_api, comfy_inspect, workflow_parse, workflow_patch, comfy_execute, comfy_discover, session_tools, workflow_templates, civitai_api, model_compat, verify_execution, github_releases, pipeline, image_metadata, node_replacement, comfy_provision
from ..stage import provision_tools, stage_tools, foresight_tools, compositor_tools, hyperagent_tools

log = logging.getLogger(__name__)

_MODULES = (comfy_api, comfy_inspect, workflow_parse, workflow_patch, comfy_execute, comfy_discover, session_tools, workflow_templates, civitai_api, model_compat, verify_execution, github_releases, pipeline, image_metadata, node_replacement, comfy_provision, provision_tools, stage_tools, foresight_tools, compositor_tools, hyperagent_tools)

# Intelligence layer tool schemas
_LAYER_TOOLS: list[dict] = []
for _mod in _MODULES:
    _LAYER_TOOLS.extend(_mod.TOOLS)

# Map tool name -> handler module (intelligence layers)
_HANDLERS = {}
for _mod in _MODULES:
    for _tool in _mod.TOOLS:
        _HANDLERS[_tool["name"]] = _mod

# Brain tools are loaded lazily to break the circular import
_brain_loaded = False
_brain_lock = threading.Lock()
_BRAIN_TOOL_NAMES: set[str] = set()


def _ensure_brain():
    """Lazily load brain layer tools (thread-safe)."""
    global _brain_loaded, _BRAIN_TOOL_NAMES
    if _brain_loaded:
        return
    with _brain_lock:
        if _brain_loaded:  # double-check after acquiring lock
            return
        from ..brain import ALL_BRAIN_TOOLS
        _BRAIN_TOOL_NAMES.update(t["name"] for t in ALL_BRAIN_TOOLS)
        _brain_loaded = True


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


def _observe(name: str, tool_input: dict, ctx: "object | None") -> None:
    """Record observation after tool dispatch. Never raises."""
    try:
        if ctx is not None and hasattr(ctx, 'workflow'):
            ctx.workflow.observe(name, tool_input)
    except Exception:
        pass


def _get_all_tools() -> list[dict]:
    """Get all tool schemas (intelligence + brain layers)."""
    _ensure_brain()
    from ..brain import ALL_BRAIN_TOOLS
    return _LAYER_TOOLS + list(ALL_BRAIN_TOOLS)


# Public API — ALL_TOOLS is a property-like accessor
class _ToolList(list):
    """Lazy tool list that includes brain tools on first access."""
    _initialized = False
    _lock = threading.Lock()

    def _init_once(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:  # Double-check after acquiring lock
                return
            self.extend(_LAYER_TOOLS)
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

    # Pre-dispatch gate (guarded by kill switch, only for known tools)
    _is_known = name in _HANDLERS or name in _BRAIN_TOOL_NAMES
    try:
        from ..config import GATE_ENABLED
        if GATE_ENABLED and _is_known:
            from ..gate import pre_dispatch_check, GateDecision
            gate_result = pre_dispatch_check(
                name, tool_input,
                session_active=ctx is not None,
                has_undo=bool(
                    ctx and hasattr(ctx, 'workflow')
                    and ctx.workflow.get("history")
                ),
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
        log.debug("Gate check failed for %s, proceeding anyway", name,
                   exc_info=True)

    # Check brain tools (lazy loaded)
    _ensure_brain()
    if name in _BRAIN_TOOL_NAMES:
        from ..brain import handle as handle_brain
        try:
            result = handle_brain(name, tool_input)
            _observe(name, tool_input, ctx)
            return result
        except Exception:
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
        return f"Unknown tool: {name}"
    try:
        # Forward progress to any module that accepts it; fall back gracefully
        try:
            result = mod.handle(name, tool_input, progress=progress)
        except TypeError:
            result = mod.handle(name, tool_input)
        _observe(name, tool_input, ctx)
        return result
    except Exception:
        log.error("Unhandled error in tool %s", name, exc_info=True)
        from ..errors import error_json
        return error_json(
            f"Something went wrong with {name}.",
            hint="Check the logs or try again.",
        )
