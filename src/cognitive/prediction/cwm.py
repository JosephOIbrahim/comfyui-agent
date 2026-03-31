"""Cognitive World Model — prediction composition via LIVRPS.

Composes predictions from three sources:
1. Prior rules (model family defaults, known good ranges)
2. Accumulated experience (what worked before in similar contexts)
3. Counterfactual corrections (learned from prediction errors)

Each source maps to a LIVRPS opinion tier:
- R (References): Prior rules
- I (Inherits): Experience-derived predictions
- S (Safety): Safety constraints on predictions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.delta import LIVRPS_PRIORITY


@dataclass
class Prediction:
    """A quality/outcome prediction for a proposed generation."""

    quality_estimate: float = 0.0  # 0.0 - 1.0
    confidence: float = 0.0  # 0.0 - 1.0
    reasoning: str = ""
    risk_factors: list[str] = field(default_factory=list)
    suggested_changes: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, float] = field(default_factory=dict)
    # sources: {"prior": weight, "experience": weight, "counterfactual": weight}

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.5

    @property
    def is_good(self) -> bool:
        return self.quality_estimate >= 0.6

    @property
    def should_proceed(self) -> bool:
        """Recommend proceeding if prediction is positive OR confidence is low."""
        return self.is_good or not self.is_confident


class CognitiveWorldModel:
    """Prediction engine using LIVRPS composition.

    Composes predictions from prior rules, experience, and
    counterfactual corrections. The LIVRPS priority ordering
    determines which prediction source wins on conflict:
    Safety > Experience > Prior.
    """

    def __init__(self):
        self._prior_rules: dict[str, dict[str, Any]] = {}
        self._confidence_history: list[tuple[float, float]] = []
        # List of (predicted, actual) pairs for calibration

    def add_prior_rule(
        self,
        model_family: str,
        parameter: str,
        good_range: tuple[float, float],
        optimal: float,
    ) -> None:
        """Register a prior rule for a model family + parameter.

        Args:
            model_family: e.g. "SD1.5", "SDXL", "Flux"
            parameter: e.g. "cfg", "steps", "denoise"
            good_range: (min, max) of the known good range
            optimal: The optimal value within the range
        """
        key = f"{model_family}:{parameter}"
        self._prior_rules[key] = {
            "range": good_range,
            "optimal": optimal,
        }

    def predict(
        self,
        model_family: str,
        parameters: dict[str, Any],
        experience_quality: float | None = None,
        experience_weight: float = 0.0,
        counterfactual_adjustment: float = 0.0,
    ) -> Prediction:
        """Predict the outcome quality for a proposed generation.

        Uses LIVRPS composition to combine prediction sources:
        - R (prior): Known good ranges and model family defaults
        - I (experience): Historical quality for similar parameters
        - S (safety): Hard constraints (e.g., degenerate parameter combos)

        Args:
            model_family: The model family being used.
            parameters: Proposed parameter values.
            experience_quality: Average quality from similar experience chunks.
            experience_weight: How much to weight experience (from accumulator).
            counterfactual_adjustment: Correction from counterfactual learning.

        Returns:
            Prediction with quality estimate and confidence.
        """
        prediction = Prediction()

        # Layer 1: Prior rules (opinion R, priority 2)
        prior_score, prior_risks = self._evaluate_priors(model_family, parameters)

        # Layer 2: Experience (opinion I, priority 4)
        exp_score = experience_quality if experience_quality is not None else prior_score

        # Layer 3: Safety check (opinion S, priority 6)
        safety_score, safety_risks = self._check_safety(model_family, parameters)

        # LIVRPS composition: apply weakest first, strongest last
        # R < I < S in our inverted LIVRPS
        layers = [
            (LIVRPS_PRIORITY["R"], prior_score, 1.0 - experience_weight),
            (LIVRPS_PRIORITY["I"], exp_score, experience_weight),
        ]
        layers.sort(key=lambda x: x[0])

        # Weighted blend of prior and experience
        total_weight = sum(w for _, _, w in layers)
        if total_weight > 0:
            blended = sum(s * w for _, s, w in layers) / total_weight
        else:
            blended = prior_score

        # Apply counterfactual correction
        blended += counterfactual_adjustment
        blended = max(0.0, min(1.0, blended))

        # Safety overrides (S is strongest)
        if safety_score < blended:
            blended = safety_score

        prediction.quality_estimate = round(blended, 3)
        prediction.risk_factors = prior_risks + safety_risks
        prediction.sources = {
            "prior": round(prior_score, 3),
            "experience": round(exp_score, 3) if experience_quality is not None else 0.0,
            "safety": round(safety_score, 3),
        }

        # Confidence based on available data
        prediction.confidence = self._compute_confidence(
            has_experience=experience_quality is not None,
            experience_weight=experience_weight,
            risk_count=len(prediction.risk_factors),
        )

        prediction.reasoning = self._build_reasoning(prediction, model_family)

        return prediction

    def record_accuracy(self, predicted: float, actual: float) -> None:
        """Record a prediction vs actual pair for calibration.

        Args:
            predicted: The predicted quality (0.0-1.0).
            actual: The actual quality (0.0-1.0).
        """
        self._confidence_history.append((predicted, actual))

    def get_calibration(self) -> dict[str, float]:
        """Get prediction calibration statistics."""
        if not self._confidence_history:
            return {"samples": 0, "mean_error": 0.0, "bias": 0.0}

        errors = [abs(p - a) for p, a in self._confidence_history]
        biases = [p - a for p, a in self._confidence_history]
        return {
            "samples": len(self._confidence_history),
            "mean_error": round(sum(errors) / len(errors), 3),
            "bias": round(sum(biases) / len(biases), 3),
        }

    def _evaluate_priors(
        self,
        model_family: str,
        parameters: dict[str, Any],
    ) -> tuple[float, list[str]]:
        """Score parameters against prior rules. Returns (score, risks)."""
        if not self._prior_rules:
            return (0.5, [])  # No priors = uncertain

        scores = []
        risks = []

        for param_name, param_value in parameters.items():
            key = f"{model_family}:{param_name}"
            rule = self._prior_rules.get(key)
            if rule is None:
                continue

            try:
                val = float(param_value)
            except (ValueError, TypeError):
                continue

            low, high = rule["range"]
            optimal = rule["optimal"]

            if low <= val <= high:
                # Within range — score based on distance from optimal
                range_width = high - low
                if range_width > 0:
                    distance = abs(val - optimal) / range_width
                    scores.append(1.0 - distance * 0.5)
                else:
                    scores.append(1.0)
            else:
                scores.append(0.3)
                risks.append(f"{param_name}={val} is outside good range [{low}-{high}]")

        if not scores:
            return (0.5, risks)
        return (sum(scores) / len(scores), risks)

    def _check_safety(
        self,
        model_family: str,
        parameters: dict[str, Any],
    ) -> tuple[float, list[str]]:
        """Check for safety issues. Returns (score, risks)."""
        risks = []
        score = 1.0

        # Detect degenerate parameter combinations
        cfg = parameters.get("cfg")
        if cfg is not None:
            try:
                cfg_val = float(cfg)
                if cfg_val <= 0:
                    risks.append("cfg=0 produces pure noise")
                    score = 0.1
                elif cfg_val > 30:
                    risks.append(f"cfg={cfg_val} is extremely high — expect artifacts")
                    score = min(score, 0.3)
            except (ValueError, TypeError):
                pass

        steps = parameters.get("steps")
        if steps is not None:
            try:
                steps_val = int(steps)
                if steps_val < 1:
                    risks.append("steps < 1 is invalid")
                    score = 0.0
            except (ValueError, TypeError):
                pass

        denoise = parameters.get("denoise")
        if denoise is not None:
            try:
                d = float(denoise)
                if d <= 0:
                    risks.append("denoise=0 produces unchanged input")
                    score = min(score, 0.2)
            except (ValueError, TypeError):
                pass

        return (score, risks)

    def _compute_confidence(
        self,
        has_experience: bool,
        experience_weight: float,
        risk_count: int,
    ) -> float:
        """Compute prediction confidence."""
        base = 0.3  # Minimum confidence from priors

        if has_experience:
            base += experience_weight * 0.5

        if self._prior_rules:
            base += 0.1

        # Risks reduce confidence
        base -= risk_count * 0.1

        # Calibration data increases confidence
        if len(self._confidence_history) > 10:
            base += 0.1

        return round(max(0.0, min(1.0, base)), 3)

    def _build_reasoning(self, prediction: Prediction, model_family: str) -> str:
        """Build human-readable reasoning."""
        parts = [f"Predicted quality: {prediction.quality_estimate:.1%}"]
        parts.append(f"Confidence: {prediction.confidence:.1%}")
        parts.append(f"Model family: {model_family}")

        if prediction.risk_factors:
            parts.append(f"Risks: {', '.join(prediction.risk_factors)}")

        if prediction.should_proceed:
            parts.append("Recommendation: proceed")
        else:
            parts.append("Recommendation: adjust parameters before generating")

        return " | ".join(parts)
