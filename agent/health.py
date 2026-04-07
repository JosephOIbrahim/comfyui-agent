"""Health check for Comfy Cozy subsystems."""

import logging
import time

import httpx

from .config import COMFYUI_URL, LLM_PROVIDER

log = logging.getLogger(__name__)

_start_time = time.monotonic()


def check_health() -> dict:
    """Check health of all subsystems. Returns structured status dict."""
    result = {
        "status": "ok",
        "uptime_seconds": round(time.monotonic() - _start_time),
        "llm_provider": LLM_PROVIDER,
        "comfyui": _check_comfyui(),
        "llm": _check_llm(),
    }

    # Determine overall status
    if result["comfyui"]["status"] == "error" or result["llm"]["status"] == "error":
        result["status"] = "degraded"

    return result


def _check_comfyui() -> dict:
    """Check ComfyUI reachability.

    The panel runs *inside* ComfyUI's aiohttp server, so HTTP-calling
    127.0.0.1:8188 from within a route handler deadlocks the single-worker
    event loop. Instead, check the in-process PromptServer instance and read
    GPU stats directly via comfy.model_management.
    """
    try:
        # Look up the running PromptServer via sys.modules — avoids re-importing
        # `server` (which can collide with other top-level `utils` packages in
        # standalone test contexts).
        import sys
        server_mod = sys.modules.get("server")
        if server_mod is None or getattr(server_mod, "PromptServer", None) is None \
                or server_mod.PromptServer.instance is None:
            return {"status": "error", "url": COMFYUI_URL, "error": "PromptServer not initialized"}

        gpu_info = {}
        try:
            import comfy.model_management as mm  # type: ignore
            import torch
            dev = mm.get_torch_device()
            if dev.type == "cuda":
                idx = dev.index if dev.index is not None else torch.cuda.current_device()
                props = torch.cuda.get_device_properties(idx)
                free, total = torch.cuda.mem_get_info(idx)
                gpu_info = {
                    "name": props.name,
                    "vram_total_gb": round(total / 1e9, 1),
                    "vram_free_gb": round(free / 1e9, 1),
                }
        except Exception as e:
            gpu_info = {"error": f"gpu probe failed: {e}"}

        return {"status": "ok", "url": COMFYUI_URL, "gpu": gpu_info}
    except Exception as e:
        return {"status": "error", "url": COMFYUI_URL, "error": str(e)}


def _check_llm() -> dict:
    """Check LLM provider can be constructed (no API call)."""
    try:
        from .llm import get_provider

        provider = get_provider()
        return {"status": "ok", "provider": LLM_PROVIDER, "class": type(provider).__name__}
    except Exception as e:
        return {"status": "error", "provider": LLM_PROVIDER, "error": str(e)}
