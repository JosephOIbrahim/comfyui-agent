"""Compute model resource requirements from workflow JSON.

Pure function — examines node class_types and input values to derive
what checkpoint family is in use, how many LoRAs, whether ControlNet
is present, sampler class, resolution band, and a VRAM estimate.
"""

from __future__ import annotations

from .schemas import ModelRequirements


# ---------------------------------------------------------------------------
# Checkpoint family detection
# ---------------------------------------------------------------------------

# Maps checkpoint loader class_type → heuristic for family detection.
# The actual family is determined by the checkpoint filename when possible.
_CHECKPOINT_LOADERS: frozenset[str] = frozenset({
    "CheckpointLoaderSimple",
    "CheckpointLoader",
    "UNETLoader",
    "DiffusersLoader",
})

# Filename substring → family
_FAMILY_HINTS: list[tuple[str, str]] = [
    ("flux", "flux"),
    ("sd3", "sd3"),
    ("sdxl", "sdxl"),
    ("xl", "sdxl"),
    ("sd_xl", "sdxl"),
    ("v1-5", "sd15"),
    ("v1_5", "sd15"),
    ("sd15", "sd15"),
    ("sd1.", "sd15"),
    ("1.5", "sd15"),
]

# Node class_types that strongly imply a family
_FAMILY_NODES: dict[str, str] = {
    "EmptySD3LatentImage": "sd3",
    "FluxGuidance": "flux",
    "TripleCLIPLoader": "sd3",
    "DualCLIPLoader": "flux",
    "CLIPTextEncodeFlux": "flux",
    "SD3LatentFormat": "sd3",
}

# LoRA loader class_types
_LORA_LOADERS: frozenset[str] = frozenset({
    "LoraLoader",
    "LoraLoaderModelOnly",
    "LoRAStacker",
    "CR LoRA Stack",
})

# ControlNet class_types
_CONTROLNET_NODES: frozenset[str] = frozenset({
    "ControlNetLoader",
    "ControlNetApply",
    "ControlNetApplyAdvanced",
    "ControlNetApplySD3",
    "ACN_AdvancedControlNetApply",
    "IPAdapterApply",
    "IPAdapter",
    "IPAdapterAdvanced",
})

# Sampler class_types
_SAMPLER_NODES: dict[str, str] = {
    "KSampler": "standard",
    "KSamplerAdvanced": "advanced",
    "SamplerCustom": "custom",
    "SamplerCustomAdvanced": "custom_advanced",
}

# Resolution source nodes (priority order)
_RESOLUTION_NODES: tuple[str, ...] = (
    "EmptyLatentImage",
    "EmptySD3LatentImage",
)

# Family → base VRAM estimate (GB) for a single inference
_BASE_VRAM: dict[str, float] = {
    "sd15": 4.0,
    "sdxl": 6.5,
    "flux": 10.0,
    "sd3": 8.0,
    "unknown": 4.0,
}

# Per-LoRA VRAM overhead (GB)
_LORA_VRAM: float = 0.3

# ControlNet VRAM overhead (GB)
_CONTROLNET_VRAM: float = 1.5


def _detect_family(workflow: dict) -> str:
    """Detect the checkpoint model family from workflow content."""
    # Phase 1: strong node signals
    for node_data in workflow.values():
        ct = node_data.get("class_type", "")
        if ct in _FAMILY_NODES:
            return _FAMILY_NODES[ct]

    # Phase 2: checkpoint filename hints
    for node_data in workflow.values():
        ct = node_data.get("class_type", "")
        if ct in _CHECKPOINT_LOADERS:
            inputs = node_data.get("inputs", {})
            for key in ("ckpt_name", "unet_name", "model_path"):
                name = inputs.get(key, "")
                if isinstance(name, str):
                    lower = name.lower()
                    for hint, family in _FAMILY_HINTS:
                        if hint in lower:
                            return family

    return "unknown"


def _count_loras(workflow: dict) -> int:
    """Count LoRA loader nodes."""
    count = 0
    for node_data in workflow.values():
        ct = node_data.get("class_type", "")
        if ct in _LORA_LOADERS:
            count += 1
    return count


def _has_controlnet(workflow: dict) -> bool:
    """True if any ControlNet / IP-Adapter node is present."""
    for node_data in workflow.values():
        ct = node_data.get("class_type", "")
        if ct in _CONTROLNET_NODES:
            return True
    return False


def _detect_sampler(workflow: dict) -> str:
    """Detect the sampler class in use."""
    for node_data in workflow.values():
        ct = node_data.get("class_type", "")
        if ct in _SAMPLER_NODES:
            return _SAMPLER_NODES[ct]
    return "unknown"


def _detect_resolution(workflow: dict) -> str:
    """Detect the target resolution band from latent image nodes."""
    for node_data in workflow.values():
        ct = node_data.get("class_type", "")
        if ct in _RESOLUTION_NODES:
            inputs = node_data.get("inputs", {})
            width = inputs.get("width", 0)
            height = inputs.get("height", 0)
            if isinstance(width, int) and isinstance(height, int):
                max_dim = max(width, height)
                if max_dim <= 576:
                    return "512"
                if max_dim <= 832:
                    return "768"
                if max_dim <= 1152:
                    return "1024"
                return "2048"
    return "unknown"


def _estimate_vram(
    family: str,
    lora_count: int,
    controlnet: bool,
    resolution: str,
) -> float:
    """Rough VRAM estimate in GB."""
    base = _BASE_VRAM.get(family, _BASE_VRAM["unknown"])
    total = base + (lora_count * _LORA_VRAM)
    if controlnet:
        total += _CONTROLNET_VRAM
    # High resolution multiplier
    if resolution == "2048":
        total *= 1.6
    elif resolution == "1024" and family in ("sd15",):
        total *= 1.3
    return round(total, 1)


def compute_model_reqs(workflow: dict) -> ModelRequirements:
    """Derive ``ModelRequirements`` from workflow JSON.

    Args:
        workflow: ComfyUI API-format workflow JSON.

    Returns:
        Immutable model requirements dataclass.
    """
    if not workflow:
        return ModelRequirements()

    family = _detect_family(workflow)
    lora_count = _count_loras(workflow)
    controlnet = _has_controlnet(workflow)
    sampler = _detect_sampler(workflow)
    resolution = _detect_resolution(workflow)
    vram = _estimate_vram(family, lora_count, controlnet, resolution)

    return ModelRequirements(
        vram_estimate_gb=vram,
        checkpoint_family=family,
        lora_count=lora_count,
        controlnet_present=controlnet,
        sampler_class=sampler,
        resolution_band=resolution,
    )
