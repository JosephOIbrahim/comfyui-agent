"""Cognitive World Model — predicts outcomes of proposed workflow changes.

Queries the experience base, applies prior rules, and considers
counterfactuals to predict the outcome of a proposed change.

Three learning phases:
  Prior Only     (<30 experiences)  — rely on hardcoded priors
  Prior+Experience (30-100)        — blend priors with experience
  Experience Dominant (100+)       — experience weights dominate

Composes predictions via LIVRPS: prior is the weakest opinion,
experience is the strongest when sufficient data exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .experience import (
    OUTCOME_AXES,
    ExperienceChunk,
    query_experience,
)
from .workflow_signature import WorkflowSignature

# Learning phase thresholds
PHASE_PRIOR_ONLY = 30
PHASE_BLENDED = 100


@dataclass
class PredictedOutcome:
    """Result of a CWM prediction."""

    axis_scores: dict[str, float]      # predicted score per axis [0,1]
    confidence: float                   # [0,1] overall confidence
    phase: str                          # "prior_only" | "blended" | "experience_dominant"
    experience_count: int               # how many experiences informed this
    similar_count: int                  # how many had matching signatures
    reasoning: str = ""                 # human-readable explanation

    def composite(self) -> float:
        """Weighted average of all axis scores (equal weights)."""
        if not self.axis_scores:
            return 0.0
        return sum(self.axis_scores.values()) / len(self.axis_scores)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "axis_scores": dict(self.axis_scores),
            "confidence": self.confidence,
            "composite": self.composite(),
            "phase": self.phase,
            "experience_count": self.experience_count,
            "similar_count": self.similar_count,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# Prior rules (hardcoded domain knowledge)
# ---------------------------------------------------------------------------

# Default prior scores when no experience exists
_DEFAULT_PRIOR: dict[str, float] = {
    axis: 0.5 for axis in OUTCOME_AXES
}

# Known parameter change effects (direction heuristics)
_PARAM_PRIORS: dict[str, dict[str, float]] = {
    "increase_steps": {
        "aesthetic": 0.65,
        "depth": 0.55,
        "normals": 0.55,
        "lighting": 0.6,
    },
    "decrease_steps": {
        "aesthetic": 0.4,
        "depth": 0.45,
        "normals": 0.45,
        "lighting": 0.45,
    },
    "increase_cfg": {
        "aesthetic": 0.55,
        "depth": 0.6,
        "normals": 0.6,
        "lighting": 0.55,
    },
    "decrease_cfg": {
        "aesthetic": 0.5,
        "depth": 0.45,
        "normals": 0.45,
        "lighting": 0.5,
    },
    "add_controlnet": {
        "depth": 0.7,
        "normals": 0.7,
        "camera": 0.65,
    },
    "add_lora": {
        "aesthetic": 0.6,
    },
    "increase_resolution": {
        "aesthetic": 0.6,
        "depth": 0.55,
    },
}


def _classify_change(proposed_change: dict[str, Any]) -> str | None:
    """Classify a proposed change into a known prior category.

    Args:
        proposed_change: Dict describing the change. Expected keys:
            "param" (str), "direction" (str: "increase"|"decrease"),
            or "action" (str: "add_controlnet"|"add_lora"|etc.)

    Returns:
        Prior category key or None if unrecognized.
    """
    action = proposed_change.get("action", "")
    if action in _PARAM_PRIORS:
        return action

    param = proposed_change.get("param", "")
    direction = proposed_change.get("direction", "")

    if param in ("steps", "cfg", "resolution") and direction in ("increase", "decrease"):
        return f"{direction}_{param}"

    return None


def _get_prior_scores(proposed_change: dict[str, Any]) -> dict[str, float]:
    """Get prior scores for a proposed change.

    Returns axis scores from hardcoded priors, falling back to defaults.
    """
    category = _classify_change(proposed_change)
    if category and category in _PARAM_PRIORS:
        # Merge with defaults (prior fills any missing axes)
        result = dict(_DEFAULT_PRIOR)
        result.update(_PARAM_PRIORS[category])
        return result
    return dict(_DEFAULT_PRIOR)


def _compute_experience_scores(
    chunks: list[ExperienceChunk],
    current_signature: WorkflowSignature | None = None,
) -> dict[str, float]:
    """Compute weighted average outcome scores from experience.

    Uses decayed experience_weight. If current_signature is provided,
    similar experiences get a match_score bonus.
    """
    if not chunks:
        return {}

    axis_sums: dict[str, float] = {}
    axis_weights: dict[str, float] = {}

    for chunk in chunks:
        base_weight = chunk.decayed_weight()

        # Bonus for similar signatures
        if current_signature and chunk.context_signature_hash:
            # We can't reconstruct the full signature from just a hash,
            # so we use a simpler heuristic: hash match = full bonus
            bonus = 1.5 if (
                current_signature.signature_hash() == chunk.context_signature_hash
            ) else 1.0
            weight = base_weight * bonus
        else:
            weight = base_weight

        for axis, score in chunk.outcome.items():
            axis_sums[axis] = axis_sums.get(axis, 0.0) + score * weight
            axis_weights[axis] = axis_weights.get(axis, 0.0) + weight

    return {
        axis: axis_sums[axis] / axis_weights[axis]
        for axis in axis_sums
        if axis_weights.get(axis, 0.0) > 0.0
    }


def _blend_scores(
    prior: dict[str, float],
    experience: dict[str, float],
    experience_count: int,
) -> tuple[dict[str, float], str]:
    """Blend prior and experience scores based on learning phase.

    Returns (blended_scores, phase_name).
    """
    if experience_count < PHASE_PRIOR_ONLY:
        phase = "prior_only"
        # Prior dominates, small experience nudge
        alpha = experience_count / PHASE_PRIOR_ONLY * 0.3  # max 0.3
    elif experience_count < PHASE_BLENDED:
        phase = "blended"
        # Linear blend from 0.3 to 0.8
        progress = (experience_count - PHASE_PRIOR_ONLY) / (
            PHASE_BLENDED - PHASE_PRIOR_ONLY
        )
        alpha = 0.3 + progress * 0.5
    else:
        phase = "experience_dominant"
        alpha = 0.9  # experience almost fully dominates

    result: dict[str, float] = {}
    all_axes = set(prior) | set(experience)

    for axis in all_axes:
        p = prior.get(axis, 0.5)
        e = experience.get(axis, p)  # fall back to prior if no experience
        result[axis] = (1.0 - alpha) * p + alpha * e

    return result, phase


def _compute_confidence(
    experience_count: int,
    similar_count: int,
    phase: str,
) -> float:
    """Compute prediction confidence [0,1].

    Higher with more experience, especially similar experience.
    """
    if experience_count == 0:
        return 0.1  # pure prior = low confidence

    # Base confidence from experience count
    base = min(0.5, experience_count / 200.0)

    # Bonus for similar experiences
    similar_bonus = min(0.3, similar_count / 50.0)

    # Phase bonus
    phase_bonus = {
        "prior_only": 0.0,
        "blended": 0.1,
        "experience_dominant": 0.2,
    }.get(phase, 0.0)

    return min(1.0, base + similar_bonus + phase_bonus)


def predict(
    cws: Any,  # CognitiveWorkflowStage
    proposed_change: dict[str, Any],
    current_signature: WorkflowSignature | None = None,
) -> PredictedOutcome:
    """Predict the outcome of a proposed workflow change.

    Queries the experience base, applies prior rules, and blends
    based on the current learning phase.

    Args:
        cws: CognitiveWorkflowStage instance (for experience access).
        proposed_change: Dict describing the proposed change.
            Common keys: "param", "direction", "action", "value".
        current_signature: WorkflowSignature of the current workflow.

    Returns:
        PredictedOutcome with axis scores, confidence, and reasoning.
    """
    # Get prior scores
    prior = _get_prior_scores(proposed_change)

    # Query experience base
    sig_hash = (
        current_signature.signature_hash() if current_signature else None
    )

    all_experiences = query_experience(cws, limit=200)
    similar_experiences = (
        query_experience(cws, context_signature_hash=sig_hash, limit=100)
        if sig_hash else []
    )

    experience_count = len(all_experiences)
    similar_count = len(similar_experiences)

    # Compute experience scores (prefer similar, fall back to all)
    experience_pool = similar_experiences if similar_experiences else all_experiences
    experience_scores = _compute_experience_scores(
        experience_pool, current_signature,
    )

    # Blend
    blended, phase = _blend_scores(prior, experience_scores, experience_count)

    # Confidence
    confidence = _compute_confidence(experience_count, similar_count, phase)

    # Reasoning
    change_desc = _classify_change(proposed_change) or "unknown change"
    reasoning_parts = [
        f"Change type: {change_desc}",
        f"Phase: {phase} ({experience_count} total, {similar_count} similar)",
    ]
    if phase == "prior_only":
        reasoning_parts.append("Prediction driven by prior knowledge")
    elif phase == "experience_dominant":
        reasoning_parts.append("Prediction driven by accumulated experience")
    else:
        reasoning_parts.append("Prediction blends prior knowledge with experience")

    return PredictedOutcome(
        axis_scores=blended,
        confidence=confidence,
        phase=phase,
        experience_count=experience_count,
        similar_count=similar_count,
        reasoning=". ".join(reasoning_parts),
    )
