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


def _get_all_tools() -> list[dict]:
    """Get all tool schemas (intelligence + brain layers)."""
    _ensure_brain()
    from ..brain import ALL_BRAIN_TOOLS
    return _LAYER_TOOLS + list(ALL_BRAIN_TOOLS)


# Public API — ALL_TOOLS is a property-like accessor
class _ToolList(list):
    """Lazy tool list that includes brain tools on first access."""
    _initialized = False

    def _init_once(self):
        if not self._initialized:
            self._initialized = True
            self.extend(_LAYER_TOOLS)
            try:
                from ..brain import ALL_BRAIN_TOOLS
                self.extend(ALL_BRAIN_TOOLS)
            except ImportError:
                log.warning("Brain layer not available")

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

    # Check brain tools (lazy loaded)
    _ensure_brain()
    if name in _BRAIN_TOOL_NAMES:
        from ..brain import handle as handle_brain
        try:
            return handle_brain(name, tool_input)
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
        # Forward progress to modules that accept it
        if mod is comfy_execute:
            return mod.handle(name, tool_input, progress=progress)
        return mod.handle(name, tool_input)
    except Exception:
        log.error("Unhandled error in tool %s", name, exc_info=True)
        from ..errors import error_json
        return error_json(
            f"Something went wrong with {name}.",
            hint="Check the logs or try again.",
        )
