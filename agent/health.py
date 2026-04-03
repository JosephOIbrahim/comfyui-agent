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
    """Check ComfyUI reachability."""
    try:
        with httpx.Client() as client:
            resp = client.get(f"{COMFYUI_URL}/system_stats", timeout=3.0)
            resp.raise_for_status()
            data = resp.json()
            # Extract GPU info if available
            gpu_info = {}
            devices = data.get("devices", [])
            if devices:
                dev = devices[0]
                gpu_info = {
                    "name": dev.get("name", "unknown"),
                    "vram_total_gb": round(dev.get("vram_total", 0) / 1e9, 1),
                    "vram_free_gb": round(dev.get("vram_free", 0) / 1e9, 1),
                }
            return {"status": "ok", "url": COMFYUI_URL, "gpu": gpu_info}
    except httpx.ConnectError:
        return {"status": "error", "url": COMFYUI_URL, "error": "Connection refused"}
    except httpx.TimeoutException:
        return {"status": "error", "url": COMFYUI_URL, "error": "Timeout (3s)"}
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
