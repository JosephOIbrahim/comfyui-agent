"""Counterfactual Generator — "what if" experiments.

Generates one counterfactual per generation: "what if we had
used different parameters?" Validates counterfactuals against
future actual data to update prediction confidence.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Counterfactual:
    """A "what if" alternative parameter set."""

    cf_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    original_params: dict[str, Any] = field(default_factory=dict)
    alternative_params: dict[str, Any] = field(default_factory=dict)
    changed_parameter: str = ""
    original_value: Any = None
    alternative_value: Any = None
    predicted_quality_delta: float = 0.0  # +/- from original prediction
    actual_quality_delta: float | None = None  # Set when validated
    validated: bool = False

    @property
    def prediction_error(self) -> float | None:
        """Error between predicted and actual quality delta."""
        if self.actual_quality_delta is None:
            return None
        return abs(self.predicted_quality_delta - self.actual_quality_delta)

    @property
    def was_correct(self) -> bool | None:
        """True if predicted direction matched actual direction."""
        if self.actual_quality_delta is None:
            return None
        if self.predicted_quality_delta == 0 and self.actual_quality_delta == 0:
            return True
        return (self.predicted_quality_delta > 0) == (self.actual_quality_delta > 0)


class CounterfactualGenerator:
    """Generates and tracks counterfactual experiments.

    For each generation, produces one "what if" with a single
    parameter change, and tracks whether the prediction was correct.
    """

    def __init__(self):
        self._counterfactuals: list[Counterfactual] = []
        self._parameter_ranges: dict[str, tuple[float, float]] = {
            "cfg": (1.0, 15.0),
            "steps": (10, 50),
            "denoise": (0.3, 1.0),
        }

    @property
    def total_generated(self) -> int:
        return len(self._counterfactuals)

    @property
    def total_validated(self) -> int:
        return sum(1 for cf in self._counterfactuals if cf.validated)

    @property
    def accuracy(self) -> float:
        """Fraction of validated counterfactuals where direction was correct."""
        validated = [cf for cf in self._counterfactuals if cf.validated]
        if not validated:
            return 0.0
        correct = sum(1 for cf in validated if cf.was_correct)
        return correct / len(validated)

    def generate(
        self,
        current_params: dict[str, Any],
        predicted_quality: float,
    ) -> Counterfactual | None:
        """Generate a counterfactual for the current generation.

        Picks a single parameter to vary and predicts the quality delta.
        Returns None if no suitable parameter found.

        Args:
            current_params: Current workflow parameters.
            predicted_quality: Predicted quality for current params.

        Returns:
            Counterfactual or None.
        """
        # Find a parameter we can vary
        for param_name, (low, high) in self._parameter_ranges.items():
            current_val = current_params.get(param_name)
            if current_val is None:
                continue

            try:
                val = float(current_val)
            except (ValueError, TypeError):
                continue

            # Generate alternative: move toward optimal middle of range
            midpoint = (low + high) / 2
            if val < midpoint:
                alt_val = min(val * 1.2, high)
            else:
                alt_val = max(val * 0.8, low)

            if abs(alt_val - val) < 0.01:
                continue

            # Predict quality delta (heuristic: closer to midpoint = better)
            original_distance = abs(val - midpoint) / (high - low)
            alt_distance = abs(alt_val - midpoint) / (high - low)
            quality_delta = (original_distance - alt_distance) * 0.2

            cf = Counterfactual(
                original_params=dict(current_params),
                alternative_params={**current_params, param_name: alt_val},
                changed_parameter=param_name,
                original_value=val,
                alternative_value=round(alt_val, 2),
                predicted_quality_delta=round(quality_delta, 3),
            )
            self._counterfactuals.append(cf)
            return cf

        return None

    def validate(self, cf_id: str, actual_quality_delta: float) -> bool:
        """Validate a counterfactual with actual quality data.

        Args:
            cf_id: ID of the counterfactual to validate.
            actual_quality_delta: Actual quality difference observed.

        Returns:
            True if found and validated, False if not found.
        """
        for cf in self._counterfactuals:
            if cf.cf_id == cf_id:
                cf.actual_quality_delta = actual_quality_delta
                cf.validated = True
                return True
        return False

    def get_adjustment(self) -> float:
        """Get a calibration adjustment based on validation history.

        Returns a correction factor to apply to future predictions.
        Positive = predictions are too low, negative = too high.
        """
        validated = [cf for cf in self._counterfactuals if cf.validated]
        if len(validated) < 5:
            return 0.0

        # Average bias in predictions
        biases = [
            cf.actual_quality_delta - cf.predicted_quality_delta
            for cf in validated
            if cf.actual_quality_delta is not None
        ]
        if not biases:
            return 0.0

        return round(sum(biases) / len(biases), 3)
