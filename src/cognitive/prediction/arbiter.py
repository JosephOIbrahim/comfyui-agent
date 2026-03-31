"""Simulation Arbiter — controls prediction delivery.

Three delivery modes:
- Silent (80%): Log prediction, don't surface to user
- Soft Surface (15%): Mention prediction in passing
- Explicit (5%): Actively warn or recommend changes

The mode is selected based on prediction confidence and risk level.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DeliveryMode(Enum):
    """How to deliver a prediction to the user."""

    SILENT = "silent"  # Log only, don't show
    SOFT = "soft"  # Mention briefly
    EXPLICIT = "explicit"  # Active warning/recommendation


@dataclass
class ArbiterDecision:
    """Arbiter's decision on how to deliver a prediction."""

    mode: DeliveryMode
    message: str = ""
    should_interrupt: bool = False  # True = stop execution
    reasoning: str = ""


class SimulationArbiter:
    """Controls when and how predictions are surfaced.

    Thresholds:
    - High confidence + high risk → EXPLICIT
    - Medium confidence or medium risk → SOFT
    - Low confidence or low risk → SILENT
    """

    def __init__(
        self,
        explicit_threshold: float = 0.7,
        soft_threshold: float = 0.4,
        interrupt_quality_floor: float = 0.2,
    ):
        self._explicit_threshold = explicit_threshold
        self._soft_threshold = soft_threshold
        self._interrupt_floor = interrupt_quality_floor

    def decide(
        self,
        quality_estimate: float,
        confidence: float,
        risk_factors: list[str],
    ) -> ArbiterDecision:
        """Decide how to deliver a prediction.

        Args:
            quality_estimate: Predicted quality (0.0-1.0).
            confidence: Prediction confidence (0.0-1.0).
            risk_factors: List of identified risks.

        Returns:
            ArbiterDecision with delivery mode and optional message.
        """
        risk_level = len(risk_factors) / max(1, 5)  # Normalize to 0-1
        urgency = confidence * (1.0 - quality_estimate + risk_level)

        # Interrupt: very confident that quality will be very low
        if confidence >= self._explicit_threshold and quality_estimate < self._interrupt_floor:
            return ArbiterDecision(
                mode=DeliveryMode.EXPLICIT,
                message=self._format_warning(quality_estimate, risk_factors),
                should_interrupt=True,
                reasoning=f"High confidence ({confidence:.0%}) that quality will be very low ({quality_estimate:.0%})",
            )

        # Explicit: confident prediction of problems
        if urgency >= self._explicit_threshold:
            return ArbiterDecision(
                mode=DeliveryMode.EXPLICIT,
                message=self._format_warning(quality_estimate, risk_factors),
                reasoning=f"Urgency {urgency:.2f} >= {self._explicit_threshold}",
            )

        # Soft: moderate concern
        if urgency >= self._soft_threshold:
            return ArbiterDecision(
                mode=DeliveryMode.SOFT,
                message=self._format_note(quality_estimate, risk_factors),
                reasoning=f"Urgency {urgency:.2f} >= {self._soft_threshold}",
            )

        # Silent: low concern
        return ArbiterDecision(
            mode=DeliveryMode.SILENT,
            reasoning=f"Urgency {urgency:.2f} < {self._soft_threshold}",
        )

    def _format_warning(self, quality: float, risks: list[str]) -> str:
        parts = [f"Predicted quality is low ({quality:.0%})."]
        if risks:
            parts.append(f"Risks: {', '.join(risks[:3])}")
        parts.append("Consider adjusting parameters before generating.")
        return " ".join(parts)

    def _format_note(self, quality: float, risks: list[str]) -> str:
        if risks:
            return f"Note: {risks[0]}"
        return f"Predicted quality: {quality:.0%}"
