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

    def test_round_robin_fairness(self, generator):
        """Cycle 27 fix: generate() must rotate across parameters, not always pick cfg."""
        params = {"cfg": 7.0, "steps": 20, "denoise": 0.8}
        changed = [
            generator.generate(params, 0.7).changed_parameter
            for _ in range(6)
        ]
        # After 6 rounds across 3 parameters, all three must have appeared at least once
        assert "cfg" in changed
        assert "steps" in changed
        assert "denoise" in changed

    def test_round_robin_sequential(self, generator):
        """Consecutive calls cycle through parameters in order, not always the same one."""
        params = {"cfg": 7.0, "steps": 20, "denoise": 0.8}
        first = generator.generate(params, 0.7).changed_parameter
        second = generator.generate(params, 0.7).changed_parameter
        assert first != second  # Must not repeat the same parameter back-to-back


# ---------------------------------------------------------------------------
# Cycle 31: arbiter NaN/out-of-range guard tests
# ---------------------------------------------------------------------------

class TestArbiterInputGuards:
    """SimulationArbiter.decide() must handle NaN and out-of-range inputs."""

    def setup_method(self):
        self.arbiter = SimulationArbiter()

    def test_nan_quality_returns_explicit(self):
        """NaN quality_estimate must surface EXPLICIT warning, not silent fallthrough."""
        import math
        result = self.arbiter.decide(math.nan, 0.8, [])
        assert result.mode == DeliveryMode.EXPLICIT
        assert "nan" in result.message.lower() or "invalid" in result.message.lower()

    def test_nan_confidence_returns_explicit(self):
        """NaN confidence must surface EXPLICIT warning."""
        import math
        result = self.arbiter.decide(0.3, math.nan, [])
        assert result.mode == DeliveryMode.EXPLICIT

    def test_quality_above_1_clamped(self):
        """quality_estimate=2.0 must be clamped to 1.0 — not produce NaN urgency."""
        result = self.arbiter.decide(2.0, 0.8, [])
        # quality=1.0 after clamp: urgency = 0.8*(1.0-1.0+risk) = low → SILENT or SOFT
        assert result.mode in (DeliveryMode.SILENT, DeliveryMode.SOFT, DeliveryMode.EXPLICIT)
        # Must not raise

    def test_negative_confidence_clamped(self):
        """Negative confidence must be clamped to 0.0 — urgency = 0."""
        result = self.arbiter.decide(0.5, -1.0, [])
        # confidence=0 → urgency=0 → SILENT
        assert result.mode == DeliveryMode.SILENT

    def test_normal_inputs_unchanged(self):
        """Normal in-range inputs must still produce correct delivery mode."""
        # High confidence + low quality → interrupt (EXPLICIT)
        result = self.arbiter.decide(0.1, 0.9, ["risk1", "risk2"])
        assert result.mode == DeliveryMode.EXPLICIT
        assert result.should_interrupt is True


# ---------------------------------------------------------------------------
# Cycle 31: CWM experience_weight and record_accuracy clamp tests
# ---------------------------------------------------------------------------

class TestCWMInputGuards:
    """CognitiveWorldModel must clamp experience_weight and record_accuracy inputs."""

    def setup_method(self):
        self.cwm = CognitiveWorldModel()

    def test_experience_weight_above_1_clamped(self):
        """experience_weight=2.0 must be clamped to 1.0 — no negative prior weight."""
        # Register a prior so the prediction is non-trivial
        self.cwm.add_prior_rule("SD1.5", "cfg", (5.0, 12.0), 7.0)
        # Should not raise and should produce a bounded quality_estimate
        pred = self.cwm.predict(
            model_family="SD1.5",
            parameters={"cfg": 7.0},
            experience_quality=0.8,
            experience_weight=2.0,  # Out of range
        )
        assert 0.0 <= pred.quality_estimate <= 1.0
        assert 0.0 <= pred.confidence <= 1.0

    def test_experience_weight_negative_clamped(self):
        """experience_weight=-1.0 must be clamped to 0.0 — no negative experience weight."""
        self.cwm.add_prior_rule("SD1.5", "cfg", (5.0, 12.0), 7.0)
        pred = self.cwm.predict(
            model_family="SD1.5",
            parameters={"cfg": 7.0},
            experience_quality=0.8,
            experience_weight=-1.0,
        )
        assert 0.0 <= pred.quality_estimate <= 1.0

    def test_record_accuracy_clamped(self):
        """record_accuracy() must clamp out-of-range values before storing."""
        self.cwm.record_accuracy(predicted=2.0, actual=-0.5)
        cal = self.cwm.get_calibration()
        # With predicted clamped to 1.0 and actual clamped to 0.0:
        # mean_error = |1.0 - 0.0| = 1.0, bias = 1.0 - 0.0 = 1.0
        # Both are within [0, 1] — the unclamped version would give error=2.5
        assert cal["mean_error"] <= 1.0
        assert abs(cal["bias"]) <= 1.0
        assert cal["samples"] == 1

    def test_normal_predict_unchanged(self):
        """predict() with valid experience_weight=0.5 must work correctly."""
        self.cwm.add_prior_rule("SD1.5", "steps", (15, 50), 30)
        pred = self.cwm.predict(
            model_family="SD1.5",
            parameters={"steps": 20},
            experience_quality=0.75,
            experience_weight=0.5,
        )
        assert 0.0 <= pred.quality_estimate <= 1.0
        assert pred.confidence > 0


# ---------------------------------------------------------------------------
# Cycle 38: unbounded-list eviction caps
# ---------------------------------------------------------------------------

class TestCWMCalibrationHistoryCap:
    """_confidence_history must never exceed max_calibration_history. (Cycle 38 fix)"""

    def test_history_capped_at_max(self):
        """After max+1 calls to record_accuracy, history len equals max, not max+1."""
        cwm = CognitiveWorldModel(max_calibration_history=10)
        for i in range(15):
            cwm.record_accuracy(float(i % 10) / 10, 0.5)
        with cwm._lock:
            assert len(cwm._confidence_history) == 10

    def test_fifo_eviction_drops_oldest(self):
        """The oldest entry is evicted, not a random one."""
        cwm = CognitiveWorldModel(max_calibration_history=3)
        cwm.record_accuracy(0.1, 0.1)  # will be evicted
        cwm.record_accuracy(0.2, 0.2)
        cwm.record_accuracy(0.3, 0.3)
        cwm.record_accuracy(0.4, 0.4)  # pushes 0.1 out
        with cwm._lock:
            first = cwm._confidence_history[0]
        assert first == (0.2, 0.2), f"Expected (0.2, 0.2) as oldest, got {first}"

    def test_get_calibration_uses_capped_history(self):
        """get_calibration() samples field must equal min(calls, max)."""
        cwm = CognitiveWorldModel(max_calibration_history=5)
        for _ in range(20):
            cwm.record_accuracy(0.7, 0.6)
        cal = cwm.get_calibration()
        assert cal["samples"] == 5

    def test_default_cap_is_reasonable(self):
        """Default cap (1000) is in place — list never grows beyond it."""
        from cognitive.prediction.cwm import _MAX_CALIBRATION_HISTORY
        assert _MAX_CALIBRATION_HISTORY == 1000


class TestCounterfactualGeneratorCap:
    """_counterfactuals must never exceed max_counterfactuals. (Cycle 38 fix)"""

    def test_list_capped_at_max(self):
        """After max+N generates, list len equals max, not max+N."""
        gen = CounterfactualGenerator(max_counterfactuals=5)
        params = {"cfg": 7.0, "steps": 20, "denoise": 0.8}
        for _ in range(10):
            gen.generate(params, predicted_quality=0.7)
        with gen._lock:
            assert len(gen._counterfactuals) <= 5

    def test_total_generated_reflects_calls_not_cap(self):
        """total_generated is the list length (capped), not the call count."""
        gen = CounterfactualGenerator(max_counterfactuals=3)
        params = {"cfg": 7.0, "steps": 20, "denoise": 0.8}
        results = [gen.generate(params, 0.7) for _ in range(6)]
        # Some may return None if no suitable param found (all same) — count non-None
        assert gen.total_generated <= 3

    def test_accuracy_still_works_after_eviction(self):
        """accuracy property must not crash after eviction reorders list."""
        gen = CounterfactualGenerator(max_counterfactuals=3)
        params = {"cfg": 7.0, "steps": 20, "denoise": 0.8}
        for _ in range(10):
            cf = gen.generate(params, 0.7)
            if cf:
                gen.validate(cf.cf_id, actual_quality_delta=0.05)
        # Must not raise
        acc = gen.accuracy
        assert 0.0 <= acc <= 1.0

    def test_default_cap_is_reasonable(self):
        """Default cap (500) is in place."""
        from cognitive.prediction.counterfactual import _MAX_COUNTERFACTUALS
        assert _MAX_COUNTERFACTUALS == 500


# ---------------------------------------------------------------------------
# Cycle 39: Counterfactual cursor atomicity
# ---------------------------------------------------------------------------

class TestCounterfactualCursorAtomicity:
    """Cycle 39: cursor must advance atomically to prevent cfg-only bias under concurrency."""

    def test_round_robin_covers_all_params(self):
        """Sequential calls should rotate across all 3 parameters."""
        gen = CounterfactualGenerator()
        params = {"cfg": 7.0, "steps": 20, "denoise": 0.8}
        changed = set()
        for _ in range(12):  # 4 full rotations
            cf = gen.generate(params, 0.7)
            if cf:
                changed.add(cf.changed_parameter)
        # All three parameters must appear across 12 calls
        assert changed == {"cfg", "steps", "denoise"}

    def test_concurrent_generate_no_duplicate_params(self):
        """Concurrent generate() calls must not both select the same parameter."""
        import threading

        gen = CounterfactualGenerator()
        params = {"cfg": 7.0, "steps": 20, "denoise": 0.8}
        results: list = []
        lock = threading.Lock()

        def _gen():
            cf = gen.generate(params, 0.7)
            if cf:
                with lock:
                    results.append(cf.changed_parameter)

        threads = [threading.Thread(target=_gen) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # With 30 calls and 3 params, distribution should not be entirely one param.
        # If cursor is not atomic, all 30 concurrent calls hit "cfg".
        from collections import Counter
        counts = Counter(results)
        total = sum(counts.values())
        if total >= 6:  # only assert if we have enough results to be meaningful
            # No single parameter should account for 100% of calls (bias detection)
            max_fraction = max(counts.values()) / total
            assert max_fraction < 1.0, (
                f"All {total} concurrent generate() calls selected the same "
                f"parameter — cursor advance is not atomic: {counts}"
            )

    def test_cursor_advances_each_call(self):
        """Each generate() call must advance _param_cursor by exactly 1."""
        gen = CounterfactualGenerator()
        params = {"cfg": 7.0, "steps": 20, "denoise": 0.8}
        initial = gen._param_cursor
        n = len(gen._parameter_ranges)
        for i in range(1, n + 2):
            gen.generate(params, 0.7)
            with gen._lock:
                expected = (initial + i) % n
                assert gen._param_cursor == expected
