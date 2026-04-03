"""Compute optimization opportunities for a workflow.

Pure function — no side effects.  Examines model requirements and
system stats to identify what can be sped up (TensorRT, batch size
headroom, resolution scaling).
"""

from __future__ import annotations

from .schemas import ComplexityLevel, ModelRequirements, OptimizationVector


# Families with proven TensorRT support in ComfyUI
_TENSORRT_FAMILIES: frozenset[str] = frozenset({"sd15", "sdxl"})

# Exotic node class_types that break TensorRT compilation
_TENSORRT_BLOCKERS: frozenset[str] = frozenset({
    "IPAdapter",
    "IPAdapterAdvanced",
    "IPAdapterApply",
    "ControlNetApplySD3",
    "SamplerCustomAdvanced",
    "Reroute",
    "FluxGuidance",
})

# Resolution bands where downscaling for iteration makes sense
_DOWNSCALABLE: frozenset[str] = frozenset({"1024", "2048"})

# VRAM headroom needed for batch_size > 1 (GB per extra batch item)
_BATCH_VRAM_PER_ITEM: dict[str, float] = {
    "sd15": 2.0,
    "sdxl": 4.0,
    "flux": 8.0,
    "sd3": 6.0,
    "unknown": 3.0,
}


def compute_optimization(
    workflow: dict,
    complexity: ComplexityLevel,
    model_reqs: ModelRequirements,
    *,
    system_stats: dict | None = None,
) -> OptimizationVector:
    """Identify optimization opportunities.

    Args:
        workflow: ComfyUI API-format workflow JSON.
        complexity: Already-computed complexity level.
        model_reqs: Already-computed model requirements.
        system_stats: Optional dict from ``get_system_stats`` with
            ``vram_total_gb`` and ``vram_free_gb`` keys.

    Returns:
        Immutable optimization vector.
    """
    if not workflow:
        return OptimizationVector()

    # ---------------------------------------------------------------
    # TensorRT eligibility
    # ---------------------------------------------------------------
    tensorrt_ok = model_reqs.checkpoint_family in _TENSORRT_FAMILIES
    if tensorrt_ok:
        # Check for blocker nodes
        for node_data in workflow.values():
            ct = node_data.get("class_type", "")
            if ct in _TENSORRT_BLOCKERS:
                tensorrt_ok = False
                break
    # Only single-checkpoint workflows are TensorRT-safe
    if tensorrt_ok:
        ckpt_count = sum(
            1
            for nd in workflow.values()
            if nd.get("class_type", "")
            in ("CheckpointLoaderSimple", "CheckpointLoader", "UNETLoader")
        )
        if ckpt_count > 1:
            tensorrt_ok = False

    # ---------------------------------------------------------------
    # Batch size headroom
    # ---------------------------------------------------------------
    batch_headroom = 0
    if system_stats:
        vram_free = float(system_stats.get("vram_free_gb", 0.0))
        per_item = _BATCH_VRAM_PER_ITEM.get(
            model_reqs.checkpoint_family,
            _BATCH_VRAM_PER_ITEM["unknown"],
        )
        if per_item > 0:
            batch_headroom = int(vram_free / per_item)
            # Cap at reasonable maximum
            batch_headroom = min(batch_headroom, 8)

    # ---------------------------------------------------------------
    # Resolution scaling
    # ---------------------------------------------------------------
    res_scale = model_reqs.resolution_band in _DOWNSCALABLE

    # ---------------------------------------------------------------
    # Estimated time reduction
    # ---------------------------------------------------------------
    reduction = 0.0
    if tensorrt_ok:
        reduction += 40.0  # TensorRT typically 30-50% faster
    if res_scale and model_reqs.resolution_band == "2048":
        reduction += 20.0  # Dropping to 1024 saves ~60% but partial
    if batch_headroom > 0:
        reduction += 5.0   # Marginal unless doing batch renders
    reduction = min(reduction, 70.0)  # Don't over-promise

    return OptimizationVector(
        tensorrt_eligible=tensorrt_ok,
        batch_size_headroom=batch_headroom,
        resolution_scale_possible=res_scale,
        estimated_time_reduction_pct=round(reduction, 1),
    )
