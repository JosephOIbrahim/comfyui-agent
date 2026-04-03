"""Compute execution risk level for a workflow.

Pure function — examines node registry availability, model family
consistency, and VRAM headroom to determine risk.
"""

from __future__ import annotations

from .schemas import ModelRequirements, RiskLevel


# Families that must not be mixed in a single workflow
_INCOMPATIBLE_FAMILIES: frozenset[frozenset[str]] = frozenset({
    frozenset({"sd15", "sdxl"}),
    frozenset({"sd15", "flux"}),
    frozenset({"sd15", "sd3"}),
    frozenset({"sdxl", "flux"}),
    frozenset({"sdxl", "sd3"}),
    frozenset({"flux", "sd3"}),
})

# LoRA loader types that should match the checkpoint family
_LORA_LOADERS: frozenset[str] = frozenset({
    "LoraLoader",
    "LoraLoaderModelOnly",
})

# Node class_types that strongly imply a specific family
_FAMILY_INDICATORS: dict[str, str] = {
    "EmptySD3LatentImage": "sd3",
    "FluxGuidance": "flux",
    "TripleCLIPLoader": "sd3",
    "DualCLIPLoader": "flux",
    "CLIPTextEncodeFlux": "flux",
}


def _detect_family_conflicts(workflow: dict) -> bool:
    """True if the workflow mixes incompatible model families."""
    families_seen: set[str] = set()
    for node_data in workflow.values():
        ct = node_data.get("class_type", "")
        if ct in _FAMILY_INDICATORS:
            families_seen.add(_FAMILY_INDICATORS[ct])
    if len(families_seen) <= 1:
        return False
    for pair in _INCOMPATIBLE_FAMILIES:
        if pair.issubset(families_seen):
            return True
    return False


def _find_missing_nodes(
    workflow: dict,
    node_registry: dict | None,
) -> tuple[list[str], bool]:
    """Find nodes not in the registry.

    Returns (missing_class_types, any_critical).
    Critical means the node is a loader or sampler — without it nothing runs.
    """
    if node_registry is None:
        return [], False

    critical_prefixes = (
        "Checkpoint", "UNET", "CLIP", "VAE", "KSampler", "Sampler",
    )
    missing: list[str] = []
    any_critical = False
    seen: set[str] = set()

    for node_data in workflow.values():
        ct = node_data.get("class_type", "")
        if ct and ct not in seen:
            seen.add(ct)
            if ct not in node_registry:
                missing.append(ct)
                if any(ct.startswith(p) for p in critical_prefixes):
                    any_critical = True

    return missing, any_critical


def compute_risk(
    workflow: dict,
    model_reqs: ModelRequirements,
    *,
    node_registry: dict | None = None,
    system_stats: dict | None = None,
) -> RiskLevel:
    """Assess execution risk.

    Args:
        workflow: ComfyUI API-format workflow JSON.
        model_reqs: Already-computed model requirements.
        node_registry: Optional dict mapping class_type -> node info.
            If ``None``, node availability checks are skipped.
        system_stats: Optional dict with ``vram_total_gb``.

    Returns:
        Risk level enum value.
    """
    if not workflow:
        return RiskLevel.SAFE

    risk = RiskLevel.SAFE

    # --- Family conflict (BLOCKED) ---
    if _detect_family_conflicts(workflow):
        return RiskLevel.BLOCKED

    # --- Missing nodes ---
    missing, critical_missing = _find_missing_nodes(workflow, node_registry)
    if critical_missing:
        return RiskLevel.BLOCKED
    if missing:
        risk = max(risk, RiskLevel.RISKY)

    # --- VRAM overflow ---
    if system_stats:
        vram_total = float(system_stats.get("vram_total_gb", 0.0))
        if vram_total > 0 and model_reqs.vram_estimate_gb > vram_total:
            return RiskLevel.BLOCKED
        if vram_total > 0 and model_reqs.vram_estimate_gb > vram_total * 0.85:
            risk = max(risk, RiskLevel.CAUTION)

    # --- High LoRA count warning ---
    if model_reqs.lora_count > 3:
        risk = max(risk, RiskLevel.CAUTION)

    return risk
