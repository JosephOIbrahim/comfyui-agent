"""Tool registry for the ComfyUI Agent.

Each tool module exports:
  TOOLS: list[dict]    — Anthropic tool schemas
  handle(name, input)  — Execute a tool call, return result string
"""

from . import comfy_api, comfy_inspect, workflow_parse, workflow_patch, comfy_execute, comfy_discover, session_tools

_MODULES = (comfy_api, comfy_inspect, workflow_parse, workflow_patch, comfy_execute, comfy_discover, session_tools)

# Collect all tool schemas
ALL_TOOLS: list[dict] = []
for _mod in _MODULES:
    ALL_TOOLS.extend(_mod.TOOLS)

# Map tool name → handler module
_HANDLERS = {}
for _mod in _MODULES:
    for _tool in _mod.TOOLS:
        _HANDLERS[_tool["name"]] = _mod


def handle(name: str, tool_input: dict) -> str:
    """Dispatch a tool call to the right handler."""
    mod = _HANDLERS.get(name)
    if mod is None:
        return f"Unknown tool: {name}"
    return mod.handle(name, tool_input)
