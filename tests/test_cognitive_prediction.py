"""Tests for the Cognitive World Model.

[PREDICTION x CRUCIBLE] — Tests for CWM, Arbiter, and Counterfactuals.
"""

import pytest

from cognitive.prediction.cwm import CognitiveWorldModel
from cognitive.prediction.arbiter import SimulationArbiter, DeliveryMode
from cognitive.prediction.counterfactual import CounterfactualGenerator, Counterfactual


# ---------------------------------------------------------------------------
# CWM
# ---------------------------------------------------------------------------

class TestCWM:

    @pytest.fixture
    def cwm(self):
        model = CognitiveWorldModel()
        model.add_prior_rule("SD1.5", "cfg", (5.0, 12.0), 7.0)
        model.add_prior_rule("SD1.5", "steps", (10, 50), 20)
        model.add_prior_rule("SD1.5", "denoise", (0.3, 1.0), 0.7)
        return model

    def test_prior_only_good_params(self, cwm):
        pred = cwm.predict("SD1.5", {"cfg": 7.0, "steps": 20})
        assert pred.quality_estimate > 0.5
        assert pred.confidence > 0

    def test_prior_only_bad_params(self, cwm):
        pred = cwm.predict("SD1.5", {"cfg": 0.5, "steps": 1})
        assert pred.quality_estimate < 0.5

    def test_experience_weighted(self, cwm):
        pred = cwm.predict(
            "SD1.5", {"cfg": 7.0}, experience_quality=0.9, experience_weight=0.7,
        )
        # With high experience quality and weight, estimate should be high
        assert pred.quality_estimate > 0.7

    def test_safety_overrides(self, cwm):
        """S-opinion (safety) limits quality even with good experience."""
        pred = cwm.predict(
            "SD1.5",
            {"cfg": 0.0},  # cfg=0 is degenerate
            experience_quality=0.9,
            experience_weight=0.8,
        )
        assert pred.quality_estimate < 0.5
        assert len(pred.risk_factors) > 0

    def test_safety_predictions_override_all(self, cwm):
        """S-opinion predictions override both R and I predictions."""
        # cfg=0 should trigger safety floor regardless of experience
        pred = cwm.predict(
            "SD1.5",
            {"cfg": 0.0, "steps": 20},
            experience_quality=1.0,
            experience_weight=1.0,
        )
        assert pred.quality_estimate <= 0.1

    def test_counterfactual_adjustment(self, cwm):
        # Use params that score below 1.0 so the adjustment is visible
        pred_base = cwm.predict("SD1.5", {"cfg": 11.0, "steps": 45})
        pred_adj = cwm.predict(
            "SD1.5", {"cfg": 11.0, "steps": 45}, counterfactual_adjustment=0.1,
        )
        assert pred_adj.quality_estimate > pred_base.quality_estimate

    def test_calibration_tracking(self, cwm):
        cwm.record_accuracy(0.7, 0.6)
        cwm.record_accuracy(0.8, 0.7)
        stats = cwm.get_calibration()
        assert stats["samples"] == 2
        assert stats["mean_error"] == 0.1
        assert stats["bias"] == 0.1  # Predictions are 0.1 too high

    def test_empty_calibration(self, cwm):
        stats = cwm.get_calibration()
        assert stats["samples"] == 0

    def test_no_prior_rules(self):
        cwm = CognitiveWorldModel()
        pred = cwm.predict("Unknown", {"cfg": 7.0})
        assert pred.quality_estimate == 0.5  # Uncertain

    def test_should_proceed_good(self, cwm):
        pred = cwm.predict("SD1.5", {"cfg": 7.0, "steps": 20})
        assert pred.should_proceed is True

    def test_prediction_sources_populated(self, cwm):
        pred = cwm.predict(
            "SD1.5", {"cfg": 7.0}, experience_quality=0.8, experience_weight=0.5,
        )
        assert "prior" in pred.sources
        assert "experience" in pred.sources
        assert "safety" in pred.sources

    def test_livrps_priority_ordering(self, cwm):
        """Verify LIVRPS ordering: R < V < I < L < S for predictions."""
        from cognitive.core.delta import LIVRPS_PRIORITY
        assert LIVRPS_PRIORITY["R"] < LIVRPS_PRIORITY["I"]
        assert LIVRPS_PRIORITY["I"] < LIVRPS_PRIORITY["L"]
        assert LIVRPS_PRIORITY["L"] < LIVRPS_PRIORITY["S"]

    def test_high_cfg_warning(self, cwm):
        pred = cwm.predict("SD1.5", {"cfg": 35.0})
        assert any("extremely high" in r for r in pred.risk_factors)


# ---------------------------------------------------------------------------
# Arbiter
# ---------------------------------------------------------------------------

class TestArbiter:

    @pytest.fixture
    def arbiter(self):
        return SimulationArbiter()

    def test_silent_for_good_prediction(self, arbiter):
        decision = arbiter.decide(
            quality_estimate=0.8, confidence=0.3, risk_factors=[],
        )
        assert decision.mode == DeliveryMode.SILENT
        assert decision.should_interrupt is False

    def test_explicit_for_high_risk(self, arbiter):
        decision = arbiter.decide(
            quality_estimate=0.1,
            confidence=0.9,
            risk_factors=["cfg=0 produces noise", "steps too low", "bad sampler"],
        )
        assert decision.mode == DeliveryMode.EXPLICIT

    def test_interrupt_for_very_low_quality(self, arbiter):
        decision = arbiter.decide(
            quality_estimate=0.1,
            confidence=0.8,
            risk_factors=["degenerate"],
        )
        assert decision.should_interrupt is True

    def test_soft_for_moderate_concern(self, arbiter):
        decision = arbiter.decide(
            quality_estimate=0.4,
            confidence=0.6,
            risk_factors=["cfg outside range"],
        )
        assert decision.mode in (DeliveryMode.SOFT, DeliveryMode.EXPLICIT)

    def test_no_interrupt_low_confidence(self, arbiter):
        decision = arbiter.decide(
            quality_estimate=0.1,
            confidence=0.2,
            risk_factors=[],
        )
        assert decision.should_interrupt is False

    def test_decision_has_reasoning(self, arbiter):
        decision = arbiter.decide(0.5, 0.5, [])
        assert decision.reasoning != ""

    def test_explicit_has_message(self, arbiter):
        decision = arbiter.decide(0.1, 0.9, ["risk1", "risk2"])
        if decision.mode == DeliveryMode.EXPLICIT:
            assert decision.message != ""


# ---------------------------------------------------------------------------
# Counterfactuals
# ---------------------------------------------------------------------------

class TestCounterfactuals:

    @pytest.fixture
    def generator(self):
        return CounterfactualGenerator()

    def test_generate_counterfactual(self, generator):
        cf = generator.generate({"cfg": 7.0, "steps": 20}, predicted_quality=0.7)
        assert cf is not None
        assert cf.changed_parameter != ""
        assert cf.original_value is not None
        assert cf.alternative_value is not None
        assert cf.alternative_value != cf.original_value

    def test_validate_counterfactual(self, generator):
        cf = generator.generate({"cfg": 7.0}, predicted_quality=0.7)
        assert cf is not None
        ok = generator.validate(cf.cf_id, actual_quality_delta=0.05)
        assert ok is True
        assert cf.validated is True
        assert cf.actual_quality_delta == 0.05

    def test_validate_unknown_id(self, generator):
        ok = generator.validate("nonexistent", 0.0)
        assert ok is False

    def test_prediction_error(self):
        cf = Counterfactual(predicted_quality_delta=0.1, actual_quality_delta=0.05, validated=True)
        assert cf.prediction_error == pytest.approx(0.05)

    def test_was_correct_same_direction(self):
        cf = Counterfactual(predicted_quality_delta=0.1, actual_quality_delta=0.03, validated=True)
        assert cf.was_correct is True

    def test_was_correct_wrong_direction(self):
        cf = Counterfactual(predicted_quality_delta=0.1, actual_quality_delta=-0.05, validated=True)
        assert cf.was_correct is False

    def test_accuracy_tracking(self, generator):
        for _ in range(5):
            cf = generator.generate({"cfg": 7.0, "steps": 20}, 0.7)
            if cf:
                generator.validate(cf.cf_id, cf.predicted_quality_delta)
        assert generator.accuracy > 0

    def test_adjustment_needs_samples(self, generator):
        assert generator.get_adjustment() == 0.0

    def test_adjustment_with_data(self, generator):
        for i in range(10):
            cf = generator.generate({"cfg": 7.0 + i * 0.1, "steps": 20}, 0.7)
            if cf:
                generator.validate(cf.cf_id, cf.predicted_quality_delta + 0.05)
        adj = generator.get_adjustment()
        assert adj != 0.0

    def test_no_params_returns_none(self, generator):
        cf = generator.generate({}, 0.7)
        assert cf is None

    def test_total_counts(self, generator):
        generator.generate({"cfg": 7.0}, 0.7)
        assert generator.total_generated == 1
        assert generator.total_validated == 0
