"""WorkflowSignature — context encoding for experience matching.

Encodes the key characteristics of a workflow into a compact, hashable
signature used to match similar workflows when querying the experience base.

Fields:
  model_family     SD15 | SDXL | Flux | SD3 | unknown
  resolution_band  512 | 768 | 1024 | 2048 | other
  style_target     photorealistic | artistic | anime | abstract | unknown
  sampler_class    euler | dpm | uni | lcm | other
  controlnet       bool — whether ControlNet nodes are present
  lora_count       int — number of LoRA loader nodes

Deterministic signature_hash() via SHA-256 over sorted fields.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

try:
    from pxr import Usd  # noqa: F401 — HAS_USD guard
    HAS_USD = True
except ImportError:
    HAS_USD = False


# Known model families (lowercase for matching)
_MODEL_FAMILIES = {"sd15", "sdxl", "flux", "sd3"}

# Resolution bands: map resolution to band name
_RESOLUTION_BANDS = {512: "512", 768: "768", 1024: "1024", 2048: "2048"}

# Known sampler prefixes (lowercase matching)
_SAMPLER_MAP = {
    "euler": "euler",
    "dpm": "dpm",
    "uni": "uni",
    "lcm": "lcm",
    "ddim": "ddim",
    "heun": "euler",  # Heun is Euler-family
}

# Node class_type patterns that indicate ControlNet usage
_CONTROLNET_TYPES = {
    "ControlNetLoader",
    "ControlNetApply",
    "ControlNetApplyAdvanced",
    "DiffControlNetLoader",
    "ControlNetApplySD3",
}

# Node class_type patterns that indicate LoRA usage
_LORA_TYPES = {
    "LoraLoader",
    "LoraLoaderModelOnly",
    "LoRALoader",
}

# Style keywords in checkpoint/model names
_STYLE_KEYWORDS = {
    "photorealistic": ["realistic", "photo", "real"],
    "anime": ["anime", "manga", "waifu"],
    "artistic": ["artistic", "art", "paint", "illustration"],
    "abstract": ["abstract", "surreal"],
}


@dataclass(frozen=True)
class WorkflowSignature:
    """Immutable signature encoding workflow characteristics."""

    model_family: str = "unknown"
    resolution_band: str = "other"
    style_target: str = "unknown"
    sampler_class: str = "other"
    controlnet: bool = False
    lora_count: int = 0

    def signature_hash(self) -> str:
        """Deterministic SHA-256 hash of sorted fields."""
        data = {
            "controlnet": self.controlnet,
            "lora_count": self.lora_count,
            "model_family": self.model_family,
            "resolution_band": self.resolution_band,
            "sampler_class": self.sampler_class,
            "style_target": self.style_target,
        }
        raw = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict."""
        return {
            "model_family": self.model_family,
            "resolution_band": self.resolution_band,
            "style_target": self.style_target,
            "sampler_class": self.sampler_class,
            "controlnet": self.controlnet,
            "lora_count": self.lora_count,
        }

    def match_score(self, other: WorkflowSignature) -> float:
        """Similarity score [0.0, 1.0] against another signature.

        Each matching field contributes equally. lora_count uses
        inverse distance (1 / (1 + abs(diff))).
        """
        score = 0.0
        total = 6.0

        if self.model_family == other.model_family:
            score += 1.0
        if self.resolution_band == other.resolution_band:
            score += 1.0
        if self.style_target == other.style_target:
            score += 1.0
        if self.sampler_class == other.sampler_class:
            score += 1.0
        if self.controlnet == other.controlnet:
            score += 1.0
        # lora_count: continuous similarity
        score += 1.0 / (1.0 + abs(self.lora_count - other.lora_count))

        return score / total


def _classify_resolution(width: int, height: int) -> str:
    """Map width/height to a resolution band."""
    max_dim = max(width, height)
    if max_dim <= 576:
        return "512"
    if max_dim <= 896:
        return "768"
    if max_dim <= 1280:
        return "1024"
    if max_dim <= 2560:
        return "2048"
    return "other"


def _classify_sampler(sampler_name: str) -> str:
    """Map a sampler name to a sampler class."""
    lower = sampler_name.lower()
    for prefix, cls in _SAMPLER_MAP.items():
        if prefix in lower:
            return cls
    return "other"


def _classify_style(checkpoint_name: str) -> str:
    """Guess style target from checkpoint/model filename."""
    lower = checkpoint_name.lower()
    for style, keywords in _STYLE_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return style
    return "unknown"


def _classify_model_family(class_types: set[str], checkpoint_name: str) -> str:
    """Detect model family from node types and checkpoint name."""
    lower = checkpoint_name.lower()

    # Flux detection
    flux_nodes = {"FluxGuidance", "ModelSamplingFlux", "DualCLIPLoader"}
    if flux_nodes & class_types or "flux" in lower:
        return "flux"

    # SD3 detection
    sd3_nodes = {"ControlNetApplySD3", "SD3LatentImage"}
    if sd3_nodes & class_types or "sd3" in lower:
        return "sd3"

    # SDXL detection
    sdxl_nodes = {"SDXLPromptStyler", "SDXLRefiner"}
    if sdxl_nodes & class_types or "sdxl" in lower or "xl" in lower:
        return "sdxl"

    # SD 1.5 detection
    if "sd15" in lower or "sd1.5" in lower or "v1-5" in lower:
        return "sd15"

    return "unknown"


def from_workflow_json(workflow_json: dict[str, dict]) -> WorkflowSignature:
    """Extract a WorkflowSignature from ComfyUI API-format workflow JSON.

    Args:
        workflow_json: ComfyUI API-format JSON (node_id -> node_data).

    Returns:
        WorkflowSignature with fields populated from workflow analysis.
    """
    class_types: set[str] = set()
    checkpoint_name = ""
    sampler_name = ""
    width = 0
    height = 0
    has_controlnet = False
    lora_count = 0

    for _node_id, node_data in workflow_json.items():
        ct = node_data.get("class_type", "")
        class_types.add(ct)
        inputs = node_data.get("inputs", {})

        # Detect ControlNet
        if ct in _CONTROLNET_TYPES:
            has_controlnet = True

        # Count LoRAs
        if ct in _LORA_TYPES:
            lora_count += 1

        # Extract checkpoint name
        if ct in ("CheckpointLoaderSimple", "CheckpointLoader"):
            ckpt = inputs.get("ckpt_name", "")
            if isinstance(ckpt, str) and ckpt:
                checkpoint_name = ckpt

        # Extract sampler name
        if ct == "KSampler" or ct == "KSamplerAdvanced":
            s = inputs.get("sampler_name", "")
            if isinstance(s, str) and s:
                sampler_name = s

        # Extract resolution
        if ct == "EmptyLatentImage":
            w = inputs.get("width", 0)
            h = inputs.get("height", 0)
            if isinstance(w, int) and isinstance(h, int):
                width, height = w, h

    return WorkflowSignature(
        model_family=_classify_model_family(class_types, checkpoint_name),
        resolution_band=_classify_resolution(width, height) if width > 0 else "other",
        style_target=_classify_style(checkpoint_name),
        sampler_class=_classify_sampler(sampler_name) if sampler_name else "other",
        controlnet=has_controlnet,
        lora_count=lora_count,
    )


def from_stage(
    cws: Any,
    workflow_name: str,
) -> WorkflowSignature:
    """Extract a WorkflowSignature from USD prims in a CognitiveWorkflowStage.

    Reads the composed stage (all LIVRPS resolved) and extracts signature
    fields from node prims.

    Args:
        cws: CognitiveWorkflowStage instance.
        workflow_name: Name of the workflow to analyze.

    Returns:
        WorkflowSignature with fields populated from USD prim analysis.
    """
    nodes_base = f"/workflows/{workflow_name}/nodes"
    nodes_prim = cws.stage.GetPrimAtPath(nodes_base)

    if not nodes_prim.IsValid():
        return WorkflowSignature()

    class_types: set[str] = set()
    checkpoint_name = ""
    sampler_name = ""
    width = 0
    height = 0
    has_controlnet = False
    lora_count = 0

    for child in nodes_prim.GetChildren():
        ct_attr = child.GetAttribute("class_type")
        if not ct_attr.IsValid():
            continue
        ct = str(ct_attr.Get())
        class_types.add(ct)

        if ct in _CONTROLNET_TYPES:
            has_controlnet = True
        if ct in _LORA_TYPES:
            lora_count += 1

        if ct in ("CheckpointLoaderSimple", "CheckpointLoader"):
            a = child.GetAttribute("input:ckpt_name")
            if a.IsValid():
                v = a.Get()
                if isinstance(v, str) and v:
                    checkpoint_name = v

        if ct in ("KSampler", "KSamplerAdvanced"):
            a = child.GetAttribute("input:sampler_name")
            if a.IsValid():
                v = a.Get()
                if isinstance(v, str) and v:
                    sampler_name = v

        if ct == "EmptyLatentImage":
            wa = child.GetAttribute("input:width")
            ha = child.GetAttribute("input:height")
            if wa.IsValid() and ha.IsValid():
                w_val = wa.Get()
                h_val = ha.Get()
                if isinstance(w_val, int) and isinstance(h_val, int):
                    width, height = w_val, h_val

    return WorkflowSignature(
        model_family=_classify_model_family(class_types, checkpoint_name),
        resolution_band=_classify_resolution(width, height) if width > 0 else "other",
        style_target=_classify_style(checkpoint_name),
        sampler_class=_classify_sampler(sampler_name) if sampler_name else "other",
        controlnet=has_controlnet,
        lora_count=lora_count,
    )
