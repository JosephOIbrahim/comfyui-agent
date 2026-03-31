"""Interrupt endpoint for mid-execution abort.

Sends POST /interrupt to ComfyUI to abort the current execution.
Used by the prediction system when it detects a failure path.
"""

from __future__ import annotations

import httpx


def interrupt_execution(
    base_url: str = "http://127.0.0.1:8188",
    timeout: float = 5.0,
) -> tuple[bool, str]:
    """Send interrupt signal to ComfyUI.

    Args:
        base_url: ComfyUI base URL.
        timeout: HTTP timeout in seconds.

    Returns:
        (success, message)
    """
    try:
        resp = httpx.post(f"{base_url}/interrupt", timeout=timeout)
        if resp.status_code == 200:
            return (True, "Execution interrupted successfully.")
        return (False, f"Interrupt returned HTTP {resp.status_code}")
    except httpx.ConnectError:
        return (False, f"Could not connect to ComfyUI at {base_url}")
    except httpx.TimeoutException:
        return (False, "Interrupt request timed out")
    except Exception as e:
        return (False, f"Interrupt failed: {e}")


def get_system_stats(
    base_url: str = "http://127.0.0.1:8188",
    timeout: float = 5.0,
) -> dict:
    """Get system stats from ComfyUI for resource-aware scheduling.

    Returns parsed JSON from GET /system_stats, or error dict.
    """
    try:
        resp = httpx.get(f"{base_url}/system_stats", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}
