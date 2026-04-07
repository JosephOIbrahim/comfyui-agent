"""Comfy Cozy Panel — Cognitive UI for ComfyUI.

Registers web directory for the panel extension and mounts
server routes on PromptServer for agent communication.
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

try:
    # The `agent` package is a sibling of `panel/` inside G:/Comfy-Cozy,
    # but ComfyUI only puts `panel/`'s parent (custom_nodes/) on sys.path.
    # Resolve the real Comfy-Cozy root via the symlinked file location and
    # prepend it so `from agent...` imports work.
    import os, sys
    _cozy_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if _cozy_root not in sys.path:
        sys.path.insert(0, _cozy_root)

    from .server.routes import setup_routes
    setup_routes()
except Exception as e:
    import logging
    logging.getLogger("superduper-panel").warning("Route setup skipped: %s", e)
