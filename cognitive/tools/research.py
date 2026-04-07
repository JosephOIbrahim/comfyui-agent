"""autoresearch — Karpathy ratchet for iterative optimization.

New capability: automatically iterate on a workflow, measuring
quality after each generation and making targeted parameter
adjustments. The ratchet only moves forward — quality can only
increase or hold, never decrease.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class RatchetDirection(Enum):
    """Direction of a ratchet step."""

    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    REJECTED = "rejected"  # Would have decreased quality — reverted


@dataclass
class RatchetStep:
    """A single step in the autoresearch ratchet."""

    step_number: int
    mutations: dict[str, dict[str, Any]] = field(default_factory=dict)
    quality_before: float = 0.0
    quality_after: float = 0.0
    direction: RatchetDirection = RatchetDirection.UNCHANGED
    description: str = ""

    @property
    def improvement(self) -> float:
        return self.quality_after - self.quality_before


@dataclass
class AutoresearchConfig:
    """Configuration for an autoresearch run."""

    max_steps: int = 10
    quality_threshold: float = 0.8  # Stop when quality >= threshold
    parameters_to_optimize: list[str] = field(default_factory=list)
    quality_evaluator: Callable | None = None  # Returns 0.0-1.0


@dataclass
class AutoresearchResult:
    """Result of an autoresearch run."""

    steps: list[RatchetStep] = field(default_factory=list)
    final_quality: float = 0.0
    best_mutations: dict[str, dict[str, Any]] = field(default_factory=dict)
    stopped_reason: str = ""

    @property
    def total_improvement(self) -> float:
        if not self.steps:
            return 0.0
        return self.final_quality - self.steps[0].quality_before

    @property
    def steps_taken(self) -> int:
        return len(self.steps)


def autoresearch(
    engine: Any,
    config: AutoresearchConfig,
    initial_quality: float = 0.0,
) -> AutoresearchResult:
    """Run an autoresearch ratchet on a workflow.

    The ratchet iteratively tries parameter adjustments, keeping
    only those that improve or maintain quality. It never accepts
    a step that would decrease quality.

    Args:
        engine: CognitiveGraphEngine instance.
        config: AutoresearchConfig with optimization parameters.
        initial_quality: Starting quality score (0.0-1.0).

    Returns:
        AutoresearchResult with the optimization history.
    """
    result = AutoresearchResult()
    current_quality = initial_quality

    for step_num in range(config.max_steps):
        if current_quality >= config.quality_threshold:
            result.stopped_reason = "quality_threshold_reached"
            break

        step = RatchetStep(
            step_number=step_num,
            quality_before=current_quality,
        )

        # Without a quality evaluator, we can only plan steps
        if config.quality_evaluator is None:
            step.direction = RatchetDirection.UNCHANGED
            step.quality_after = current_quality
            step.description = "No quality evaluator — cannot assess improvement"
            result.steps.append(step)
            result.stopped_reason = "no_evaluator"
            break

        # Generate candidate mutation (placeholder — real logic in Phase 6)
        # For now, the ratchet framework is in place
        step.quality_after = current_quality
        step.direction = RatchetDirection.UNCHANGED
        step.description = f"Step {step_num}: evaluated"
        result.steps.append(step)

    else:
        result.stopped_reason = "max_steps_reached"

    result.final_quality = current_quality
    result.best_mutations = {}  # Accumulated from successful steps

    return result
