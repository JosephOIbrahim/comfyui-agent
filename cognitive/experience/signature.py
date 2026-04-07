"""GenerationContextSignature — discretized parameter space for fast matching.

Converts continuous workflow parameters into a discrete signature
that can be quickly compared for experience lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationContextSignature:
    """Discretized parameter signature for fast experience matching.

    Continuous values are bucketed into discrete ranges to enable
    efficient similarity search without exact floating-point comparison.
    """

    model_family: str = ""
    checkpoint_hash: str = ""  # First 8 chars of checkpoint name hash
    resolution_bucket: str = ""  # e.g. "512x512", "1024x1024"
    cfg_bucket: str = ""  # "low" (<5), "medium" (5-9), "high" (>9)
    steps_bucket: str = ""  # "few" (<15), "normal" (15-30), "many" (>30)
    sampler: str = ""
    scheduler: str = ""
    denoise_bucket: str = ""  # "low" (<0.5), "medium" (0.5-0.8), "high" (>0.8)
    has_controlnet: bool = False
    has_lora: bool = False
    has_ipadapter: bool = False

    @classmethod
    def from_workflow(cls, workflow_data: dict[str, Any]) -> GenerationContextSignature:
        """Build a signature from a ComfyUI API format workflow."""
        sig = cls()

        nodes = {
            k: v for k, v in workflow_data.items()
            if isinstance(v, dict) and "class_type" in v
        }
        class_types = {v["class_type"] for v in nodes.values()}

        # Detect features
        sig.has_controlnet = any("controlnet" in ct.lower() for ct in class_types)
        sig.has_lora = any("lora" in ct.lower() for ct in class_types)
        sig.has_ipadapter = any("ipadapter" in ct.lower() for ct in class_types)

        # Extract parameters from sampler node
        for node_data in nodes.values():
            ct = node_data["class_type"]
            inputs = node_data.get("inputs", {})

            if "sampler" in ct.lower() or ct == "KSampler":
                cfg = inputs.get("cfg")
                if isinstance(cfg, (int, float)):
                    sig.cfg_bucket = _bucket_cfg(cfg)

                steps = inputs.get("steps")
                if isinstance(steps, int):
                    sig.steps_bucket = _bucket_steps(steps)

                sig.sampler = str(inputs.get("sampler_name", ""))
                sig.scheduler = str(inputs.get("scheduler", ""))

                denoise = inputs.get("denoise")
                if isinstance(denoise, (int, float)):
                    sig.denoise_bucket = _bucket_denoise(denoise)

            if "checkpoint" in ct.lower() or "loader" in ct.lower():
                ckpt = inputs.get("ckpt_name", "")
                if isinstance(ckpt, str) and ckpt:
                    sig.checkpoint_hash = ckpt[:8]
                    sig.model_family = _detect_family(ckpt)

            if "emptylatent" in ct.lower().replace("_", ""):
                w = inputs.get("width", 0)
                h = inputs.get("height", 0)
                if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                    sig.resolution_bucket = f"{w}x{h}"

        return sig

    def similarity(self, other: GenerationContextSignature) -> float:
        """Compute similarity score (0.0 - 1.0) against another signature."""
        matches = 0
        total = 0

        fields = [
            ("model_family", self.model_family, other.model_family),
            ("resolution_bucket", self.resolution_bucket, other.resolution_bucket),
            ("cfg_bucket", self.cfg_bucket, other.cfg_bucket),
            ("steps_bucket", self.steps_bucket, other.steps_bucket),
            ("sampler", self.sampler, other.sampler),
            ("scheduler", self.scheduler, other.scheduler),
            ("denoise_bucket", self.denoise_bucket, other.denoise_bucket),
        ]

        for _, a, b in fields:
            if a and b:
                total += 1
                if a == b:
                    matches += 1

        if total == 0:
            return 0.0
        return matches / total


def _bucket_cfg(cfg: float) -> str:
    if cfg < 5:
        return "low"
    if cfg <= 9:
        return "medium"
    return "high"


def _bucket_steps(steps: int) -> str:
    if steps < 15:
        return "few"
    if steps <= 30:
        return "normal"
    return "many"


def _bucket_denoise(denoise: float) -> str:
    if denoise < 0.5:
        return "low"
    if denoise <= 0.8:
        return "medium"
    return "high"


def _detect_family(ckpt_name: str) -> str:
    lower = ckpt_name.lower()
    if "flux" in lower:
        return "Flux"
    if "sdxl" in lower or "sd_xl" in lower:
        return "SDXL"
    if "sd3" in lower:
        return "SD3"
    return "SD1.5"
