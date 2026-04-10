"""Simulation Arbiter — prioritizes which predictions to experiment with.

Given a CWM prediction, determines the surfacing mode:
  silent         Log internally, don't show user
  soft_surface   Mention briefly ("I think X would improve Y")
  explicit       Ask user directly ("Should we try X?")

Decision tree (from patent):
  1. High confidence + small improvement → silent
  2. High confidence + large improvement → soft_surface
  3. Low confidence + large improvement  → explicit (capped: max 1/session)
  4. Low confidence + small improvement  → silent
  5. Medium confidence                   → soft_surface

Self-calibrating: tracks how often suggestions are accepted/rejected
and adjusts thresholds accordingly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .cwm import PredictedOutcome


# Threshold defaults
DEFAULT_CONFIDENCE_HIGH = 0.7
DEFAULT_CONFIDENCE_LOW = 0.3
DEFAULT_IMPROVEMENT_LARGE = 0.15  # composite improvement threshold

# Max explicit suggestions per session
MAX_EXPLICIT_PER_SESSION = 1

# Calibration parameters
CALIBRATION_STEP = 0.02  # How much to adjust thresholds per feedback

# History caps — FIFO eviction prevents unbounded growth on long sessions (Cycle 39)
_MAX_DECISIONS = 10_000
_MAX_FEEDBACK = 10_000


@dataclass
class ArbiterDecision:
    """Result of arbiter prioritization."""

    mode: str                  # "silent" | "soft_surface" | "explicit"
    prediction: PredictedOutcome
    improvement_estimate: float  # predicted composite improvement
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "mode": self.mode,
            "improvement_estimate": self.improvement_estimate,
            "confidence": self.prediction.confidence,
            "phase": self.prediction.phase,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
        }


@dataclass
class CalibrationFeedback:
    """User feedback on a surfaced suggestion."""

    decision_mode: str         # mode that was surfaced
    accepted: bool             # did user accept the suggestion?
    timestamp: float = field(default_factory=time.time)


class Arbiter:
    """Prioritizes predictions and self-calibrates from feedback.

    Args:
        confidence_high: Threshold for "high confidence". Default 0.7.
        confidence_low: Threshold for "low confidence". Default 0.3.
        improvement_large: Threshold for "large improvement". Default 0.15.
    """

    def __init__(
        self,
        *,
        confidence_high: float = DEFAULT_CONFIDENCE_HIGH,
        confidence_low: float = DEFAULT_CONFIDENCE_LOW,
        improvement_large: float = DEFAULT_IMPROVEMENT_LARGE,
    ) -> None:
        self._confidence_high = confidence_high
        self._confidence_low = confidence_low
        self._improvement_large = improvement_large
        self._explicit_count = 0
        self._decisions: list[ArbiterDecision] = []
        self._feedback: list[CalibrationFeedback] = []
        self._max_decisions = _MAX_DECISIONS
        self._max_feedback = _MAX_FEEDBACK

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def confidence_high(self) -> float:
        return self._confidence_high

    @property
    def confidence_low(self) -> float:
        return self._confidence_low

    @property
    def improvement_large(self) -> float:
        return self._improvement_large

    @property
    def explicit_count(self) -> int:
        """Number of explicit suggestions made this session."""
        return self._explicit_count

    @property
    def decisions(self) -> list[ArbiterDecision]:
        """All decisions made, oldest first (copy)."""
        return list(self._decisions)

    @property
    def feedback_history(self) -> list[CalibrationFeedback]:
        """All feedback received (copy)."""
        return list(self._feedback)

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def prioritize_experiment(
        self,
        prediction: PredictedOutcome,
        current_composite: float = 0.5,
    ) -> ArbiterDecision:
        """Determine how to surface a prediction to the user.

        Args:
            prediction: CWM prediction result.
            current_composite: Current workflow's composite score.

        Returns:
            ArbiterDecision with mode, reasoning, and improvement estimate.
        """
        predicted_composite = prediction.composite()
        improvement = predicted_composite - current_composite
        confidence = prediction.confidence

        mode = self._decide_mode(confidence, improvement)

        reasoning_parts = [
            f"Confidence: {confidence:.2f}",
            f"Improvement: {improvement:+.3f}",
        ]

        if mode == "explicit":
            if self._explicit_count >= MAX_EXPLICIT_PER_SESSION:
                mode = "soft_surface"
                reasoning_parts.append(
                    f"Downgraded from explicit (limit {MAX_EXPLICIT_PER_SESSION}/session)"
                )
            else:
                self._explicit_count += 1
                reasoning_parts.append("Explicit suggestion (low confidence, high potential)")

        if mode == "silent":
            reasoning_parts.append("Silent: not worth surfacing")
        elif mode == "soft_surface":
            reasoning_parts.append("Soft surface: worth mentioning")

        decision = ArbiterDecision(
            mode=mode,
            prediction=prediction,
            improvement_estimate=improvement,
            reasoning=". ".join(reasoning_parts),
        )
        self._decisions.append(decision)
        if len(self._decisions) > self._max_decisions:  # Cycle 39: FIFO eviction
            self._decisions.pop(0)
        return decision

    def _decide_mode(self, confidence: float, improvement: float) -> str:
        """Apply decision tree to determine surfacing mode."""
        high = confidence >= self._confidence_high
        low = confidence <= self._confidence_low
        large = improvement >= self._improvement_large

        if high and not large:
            return "silent"
        if high and large:
            return "soft_surface"
        if low and large:
            return "explicit"
        if low and not large:
            return "silent"

        # Medium confidence
        return "soft_surface"

    # ------------------------------------------------------------------
    # Feedback & calibration
    # ------------------------------------------------------------------

    def record_feedback(self, feedback: CalibrationFeedback) -> None:
        """Record user feedback and calibrate thresholds.

        If user consistently rejects soft_surface suggestions, raise the
        improvement_large threshold (be more conservative).
        If user accepts, lower it (be more aggressive).
        """
        self._feedback.append(feedback)
        if len(self._feedback) > self._max_feedback:  # Cycle 39: FIFO eviction
            self._feedback.pop(0)
        self._calibrate(feedback)

    def _calibrate(self, fb: CalibrationFeedback) -> None:
        """Adjust thresholds based on feedback."""
        if fb.decision_mode == "soft_surface":
            if fb.accepted:
                # User liked it — be slightly more aggressive
                self._improvement_large = max(
                    0.05, self._improvement_large - CALIBRATION_STEP
                )
            else:
                # User rejected — be more conservative
                self._improvement_large = min(
                    0.5, self._improvement_large + CALIBRATION_STEP
                )

        elif fb.decision_mode == "explicit":
            if fb.accepted:
                # Explicit was useful — lower confidence threshold
                self._confidence_low = max(
                    0.1, self._confidence_low - CALIBRATION_STEP
                )
            else:
                # Explicit was annoying — raise confidence threshold
                self._confidence_low = min(
                    0.6, self._confidence_low + CALIBRATION_STEP
                )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """Reset per-session counters (explicit count).

        Call at session start. Does NOT reset calibration.
        """
        self._explicit_count = 0

    def summary(self) -> dict[str, Any]:
        """Return a summary of arbiter state."""
        total = len(self._decisions)
        modes = {"silent": 0, "soft_surface": 0, "explicit": 0}
        for d in self._decisions:
            modes[d.mode] = modes.get(d.mode, 0) + 1

        fb_total = len(self._feedback)
        fb_accepted = sum(1 for f in self._feedback if f.accepted)

        return {
            "total_decisions": total,
            "mode_counts": modes,
            "explicit_this_session": self._explicit_count,
            "feedback_total": fb_total,
            "feedback_accepted": fb_accepted,
            "feedback_acceptance_rate": (
                fb_accepted / fb_total if fb_total > 0 else 0.0
            ),
            "current_thresholds": {
                "confidence_high": self._confidence_high,
                "confidence_low": self._confidence_low,
                "improvement_large": self._improvement_large,
            },
        }
