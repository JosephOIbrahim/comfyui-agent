"""Data types for the Workflow Intelligence DAG.

All enums, dataclasses, and type definitions used by the DAG engine
and its compute functions live here. Everything is immutable (frozen
dataclasses, IntEnums) so the DAG evaluation pipeline stays pure.

Enum conventions follow the IntEnum pattern used elsewhere in
``agent/stage/`` — ordinal values increase with severity/complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ComplexityLevel(IntEnum):
    """Workflow topology complexity band.

    Thresholds are calibrated to ComfyUI workflows observed in the wild:
      TRIVIAL  — 1-5 nodes, linear chain (txt2img, single upscale)
      SIMPLE   — 6-15 nodes, minimal branching
      MODERATE — 16-30 nodes, some branching (img2img + ControlNet)
      COMPLEX  — 31-50 nodes, multiple branches (multi-ControlNet, IP-Adapter)
      EXTREME  — 50+ nodes, deep branching or sub-graphs
    """

    TRIVIAL = 0
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    EXTREME = 4


class RiskLevel(IntEnum):
    """Execution risk assessment.

    SAFE     — All models present, family matches, VRAM sufficient.
    CAUTION  — Minor warnings (high VRAM usage, untested combo).
    RISKY    — Missing optional models, untested combinations.
    BLOCKED  — Missing required models, family mismatch, VRAM overflow.
    """

    SAFE = 0
    CAUTION = 1
    RISKY = 2
    BLOCKED = 3


class ReadinessGrade(IntEnum):
    """Go/no-go gate combining all upstream signals.

    READY           — All clear, execute freely.
    NEEDS_PROVISION — Missing models need download first.
    NEEDS_FIX       — Workflow errors need repair.
    BLOCKED         — Cannot proceed until resolved.
    """

    READY = 0
    NEEDS_PROVISION = 1
    NEEDS_FIX = 2
    BLOCKED = 3


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRequirements:
    """Derived model resource requirements for a workflow."""

    vram_estimate_gb: float = 0.0
    checkpoint_family: str = "unknown"
    lora_count: int = 0
    controlnet_present: bool = False
    sampler_class: str = "unknown"
    resolution_band: str = "unknown"

    def describe(self) -> str:
        """Human-readable one-liner for logging / display."""
        parts = [f"family={self.checkpoint_family}"]
        if self.lora_count:
            parts.append(f"loras={self.lora_count}")
        if self.controlnet_present:
            parts.append("controlnet")
        parts.append(f"vram~{self.vram_estimate_gb:.1f}GB")
        parts.append(f"res={self.resolution_band}")
        return ", ".join(parts)


@dataclass(frozen=True)
class OptimizationVector:
    """Optimization opportunities identified for a workflow."""

    tensorrt_eligible: bool = False
    batch_size_headroom: int = 0
    resolution_scale_possible: bool = False
    estimated_time_reduction_pct: float = 0.0

    def has_opportunities(self) -> bool:
        """True if any optimization is possible."""
        return (
            self.tensorrt_eligible
            or self.batch_size_headroom > 0
            or self.resolution_scale_possible
        )


@dataclass(frozen=True)
class WorkflowIntelligence:
    """Full resolved state after one DAG evaluation pass.

    Constructed exclusively by ``evaluate_dag`` — never mutated afterward.
    Each field corresponds to one compute node in the DAG.
    """

    complexity: ComplexityLevel = ComplexityLevel.TRIVIAL
    model_requirements: ModelRequirements = field(
        default_factory=ModelRequirements,
    )
    optimization: OptimizationVector = field(
        default_factory=OptimizationVector,
    )
    risk: RiskLevel = RiskLevel.SAFE
    readiness: ReadinessGrade = ReadinessGrade.READY
    tool_scope: frozenset[str] = frozenset()
    evaluated: bool = False

    def to_dict(self) -> dict:
        """Serializable snapshot for session persistence."""
        return {
            "complexity": self.complexity.name,
            "model_requirements": {
                "vram_estimate_gb": self.model_requirements.vram_estimate_gb,
                "checkpoint_family": self.model_requirements.checkpoint_family,
                "lora_count": self.model_requirements.lora_count,
                "controlnet_present": self.model_requirements.controlnet_present,
                "sampler_class": self.model_requirements.sampler_class,
                "resolution_band": self.model_requirements.resolution_band,
            },
            "optimization": {
                "tensorrt_eligible": self.optimization.tensorrt_eligible,
                "batch_size_headroom": self.optimization.batch_size_headroom,
                "resolution_scale_possible": (
                    self.optimization.resolution_scale_possible
                ),
                "estimated_time_reduction_pct": (
                    self.optimization.estimated_time_reduction_pct
                ),
            },
            "risk": self.risk.name,
            "readiness": self.readiness.name,
            "tool_scope": sorted(self.tool_scope),
            "evaluated": self.evaluated,
        }
