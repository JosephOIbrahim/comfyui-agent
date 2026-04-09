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
    #
    # NOTE on intentional duplication: panel/server/chat.py and ui/server/routes.py
    # also inject project root into sys.path, but they do so lazily inside
    # _ensure_brain() (called at WebSocket/HTTP request time, not import time).
    # This call runs at ComfyUI node-load time — before any request arrives — so
    # all three guards are needed. The `if x not in sys.path` check prevents
    # multiple insertions within the same process.
    import os, sys
    _cozy_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if _cozy_root not in sys.path:
        sys.path.insert(0, _cozy_root)

    from .server.routes import setup_routes
    setup_routes()
except Exception as e:
    import logging
    logging.getLogger("comfy-cozy").warning("Route setup skipped: %s", e)
