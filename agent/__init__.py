"""ComfyUI Agent — AI co-pilot for ComfyUI workflows."""

__version__ = "0.4.0"


def tool_count() -> tuple[int, int, int]:
    """Return (intelligence_tools, brain_tools, total) from live registry."""
    from .tools import _LAYER_TOOLS
    from .brain import ALL_BRAIN_TOOLS
    intel = len(_LAYER_TOOLS)
    brain = len(ALL_BRAIN_TOOLS)
    return intel, brain, intel + brain
