"""SUPER DUPER UI -- ComfyUI Sidebar Extension.

Registers as a ComfyUI custom node (for WEB_DIRECTORY serving)
and mounts aiohttp routes on PromptServer for the agent backend.
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Mount server routes when ComfyUI loads this extension
try:
    from .server.routes import setup_routes
    setup_routes()
except Exception as e:
    import logging
    logging.getLogger("superduper").error("Route setup failed: %s", e)
