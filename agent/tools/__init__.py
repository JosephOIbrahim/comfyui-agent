"""Tool registry for the ComfyUI SUPER DUPER Agent.

Each tool module exports:
  TOOLS: list[dict]    -- Anthropic tool schemas
  handle(name, input)  -- Execute a tool call, return result string

Intelligence layers (47 tools) + Brain layer (21 tools) = 68 tools total.
Brain tools are lazily imported to avoid circular dependencies
(brain modules import _util from this package).
"""

import logging

from . import comfy_api, comfy_inspect, workflow_parse, workflow_patch, comfy_execute, comfy_discover, session_tools, workflow_templates, civitai_api, model_compat, verify_execution, github_releases, pipeline, image_metadata

log = logging.getLogger(__name__)

_MODULES = (comfy_api, comfy_inspect, workflow_parse, workflow_patch, comfy_execute, comfy_discover, session_tools, workflow_templates, civitai_api, model_compat, verify_execution, github_releases, pipeline, image_metadata)

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
_BRAIN_TOOL_NAMES: set[str] = set()


def _ensure_brain():
    """Lazily load brain layer tools."""
    global _brain_loaded, _BRAIN_TOOL_NAMES
    if _brain_loaded:
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


def handle(name: str, tool_input: dict, *, session_id: str | None = None) -> str:
    """Dispatch a tool call to the right handler.

    Args:
        name: Tool name to dispatch.
        tool_input: Tool arguments dict.
        session_id: Optional session ID for workflow state isolation.
                    Currently unused (default session), but enables future
                    multi-session MCP usage.
    """
    # Check brain tools (lazy loaded)
    _ensure_brain()
    if name in _BRAIN_TOOL_NAMES:
        from ..brain import handle as handle_brain
        return handle_brain(name, tool_input)

    # Intelligence layer tools
    mod = _HANDLERS.get(name)
    if mod is None:
        log.warning("Unknown tool called: %s", name)
        return f"Unknown tool: {name}"
    try:
        return mod.handle(name, tool_input)
    except Exception as e:
        log.error("Unhandled error in tool %s", name, exc_info=True)
        from ._util import to_json
        return to_json({"error": f"Internal error in {name}: {type(e).__name__}: {e}"})
