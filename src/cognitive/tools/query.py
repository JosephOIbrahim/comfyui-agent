"""query_environment — Unified environment snapshot.

Absorbs UNDERSTAND scan + DISCOVER into a single environment query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvironmentSnapshot:
    """Snapshot of the ComfyUI environment."""

    comfyui_running: bool = False
    gpu_name: str = ""
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    installed_node_packs: list[str] = field(default_factory=list)
    available_models: dict[str, list[str]] = field(default_factory=dict)
    node_count: int = 0
    queue_running: int = 0
    queue_pending: int = 0
    schema_cached: bool = False
    error: str = ""


def query_environment(
    system_stats: dict[str, Any] | None = None,
    queue_info: dict[str, Any] | None = None,
    node_packs: list[str] | None = None,
    models: dict[str, list[str]] | None = None,
    schema_cache: Any | None = None,
) -> EnvironmentSnapshot:
    """Build an environment snapshot from available data sources.

    This function composes data from multiple sources into a unified view.
    Pass whatever data is available — missing sources produce empty fields.

    Args:
        system_stats: Response from GET /system_stats.
        queue_info: Response from GET /queue.
        node_packs: List of installed custom node pack names.
        models: Dict of {model_type: [filenames]}.
        schema_cache: Optional SchemaCache instance.
    """
    snap = EnvironmentSnapshot()

    if system_stats is not None:
        snap.comfyui_running = "error" not in system_stats
        devices = system_stats.get("devices", [])
        if devices:
            dev = devices[0]
            snap.gpu_name = dev.get("name", "")
            snap.vram_total_mb = dev.get("vram_total", 0) // (1024 * 1024)
            snap.vram_free_mb = dev.get("vram_free", 0) // (1024 * 1024)

    if queue_info is not None:
        snap.queue_running = len(queue_info.get("queue_running", []))
        snap.queue_pending = len(queue_info.get("queue_pending", []))

    if node_packs is not None:
        snap.installed_node_packs = sorted(node_packs)

    if models is not None:
        snap.available_models = {k: sorted(v) for k, v in sorted(models.items())}

    if schema_cache is not None:
        snap.schema_cached = getattr(schema_cache, "is_populated", False)
        snap.node_count = getattr(schema_cache, "node_count", 0)

    return snap
