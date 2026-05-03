"""Tests for agent/stage/counterfactuals.py — no real I/O."""

from __future__ import annotations

import pytest

from agent.stage.counterfactuals import (
    DEFAULT_CONFIDENCE,
    Counterfactual,
    CounterfactualError,
    _generate_cf_id,
    generate_counterfactual,
    list_pending,
    list_validated,
    promote_validated,
    validate_counterfactual,
)
from agent.stage.experience import ExperienceChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def usd_stage():
    pytest.importorskip("pxr", reason="usd-core not installed")
    from agent.stage.cognitive_stage import CognitiveWorkflowStage
    return CognitiveWorkflowStage()


@pytest.fixture
def source_chunk():
    return ExperienceChunk(
        chunk_id="src_001",
        initial_state={"steps": 20},
        decisions=[{"set": "steps", "to": 20}],
        outcome={"aesthetic": 0.7, "lighting": 0.8},
        timestamp=1000.0,
    )


@pytest.fixture
def hypothesis():
    return {"alternative": "increase_steps", "value": 40}


@pytest.fixture
def predicted():
    return {"aesthetic": 0.85, "lighting": 0.9}


# ---------------------------------------------------------------------------
# Counterfactual dataclass
# ---------------------------------------------------------------------------

class TestCounterfactualDataclass:
    def test_defaults(self):
        cf = Counterfactual(
            cf_id="x", source_chunk_id="s",
            hypothesis={}, predicted_outcome={},
        )
        assert cf.confidence == DEFAULT_CONFIDENCE
        assert cf.status == "pending"
        assert cf.validation_outcome == {}

    def test_to_dict_contains_all_fields(self):
        cf = Counterfactual(
            cf_id="x", source_chunk_id="s",
            hypothesis={"a": 1}, predicted_outcome={"aesthetic": 0.8},
        )
        d = cf.to_dict()
        assert d["cf_id"] == "x"
        assert d["status"] == "pending"
        assert len(d) == 9


# ---------------------------------------------------------------------------
# _generate_cf_id
# ---------------------------------------------------------------------------

class TestGenerateCfId:
    def test_deterministic(self):
        a = _generate_cf_id("src", {"a": 1}, 100.0)
        b = _generate_cf_id("src", {"a": 1}, 100.0)
        assert a == b

    def test_different_inputs_different_ids(self):
        a = _generate_cf_id("src1", {}, 100.0)
        b = _generate_cf_id("src2", {}, 100.0)
        assert a != b

    def test_length_16(self):
        assert len(_generate_cf_id("x", {}, 0.0)) == 16


# ---------------------------------------------------------------------------
# generate_counterfactual (requires usd-core)
# ---------------------------------------------------------------------------

class TestGenerateCounterfactual:
    def test_basic_creation(self, usd_stage, source_chunk, hypothesis, predicted):
        cf = generate_counterfactual(
            usd_stage, source_chunk, hypothesis, predicted, timestamp=2000.0,
        )
        assert isinstance(cf, Counterfactual)
        assert cf.status == "pending"
        assert cf.confidence == DEFAULT_CONFIDENCE

    def test_prim_created(self, usd_stage, source_chunk, hypothesis, predicted):
        cf = generate_counterfactual(
            usd_stage, source_chunk, hypothesis, predicted, timestamp=3000.0,
        )
        prim_path = f"/counterfactuals/pending/cf_{cf.cf_id}"
        assert usd_stage.read(prim_path) is True

    def test_predicted_stored(self, usd_stage, source_chunk, hypothesis, predicted):
        cf = generate_counterfactual(
            usd_stage, source_chunk, hypothesis, predicted, timestamp=4000.0,
        )
        prim_path = f"/counterfactuals/pending/cf_{cf.cf_id}"
        stored = usd_stage.read(prim_path, "predicted:aesthetic")
        assert abs(stored - 0.85) < 1e-9

    def test_invalid_score_raises(self, usd_stage, source_chunk, hypothesis):
        with pytest.raises(CounterfactualError, match="Predicted score"):
            generate_counterfactual(
                usd_stage, source_chunk, hypothesis,
                {"aesthetic": 1.5}, timestamp=5000.0,
            )

    def test_invalid_confidence_raises(self, usd_stage, source_chunk, hypothesis, predicted):
        with pytest.raises(CounterfactualError, match="Confidence"):
            generate_counterfactual(
                usd_stage, source_chunk, hypothesis, predicted,
                confidence=1.5, timestamp=6000.0,
            )

    def test_custom_confidence(self, usd_stage, source_chunk, hypothesis, predicted):
        cf = generate_counterfactual(
            usd_stage, source_chunk, hypothesis, predicted,
            confidence=0.5, timestamp=7000.0,
        )
        assert cf.confidence == 0.5

    def test_source_chunk_id_stored(self, usd_stage, source_chunk, hypothesis, predicted):
        cf = generate_counterfactual(
            usd_stage, source_chunk, hypothesis, predicted, timestamp=8000.0,
        )
        prim_path = f"/counterfactuals/pending/cf_{cf.cf_id}"
        assert usd_stage.read(prim_path, "source_chunk_id") == "src_001"


# ---------------------------------------------------------------------------
# validate_counterfactual (requires usd-core)
# ---------------------------------------------------------------------------

class TestValidateCounterfactual:
    def test_basic_validation(self, usd_stage, source_chunk, hypothesis, predicted):
        cf = generate_counterfactual(
            usd_stage, source_chunk, hypothesis, predicted, timestamp=1000.0,
        )
        updated = validate_counterfactual(
            usd_stage, cf.cf_id,
            actual_outcome={"aesthetic": 0.8, "lighting": 0.85},
            timestamp=2000.0,
        )
        assert updated.validation_outcome == {"aesthetic": 0.8, "lighting": 0.85}
        assert updated.validation_timestamp == 2000.0

    def test_confidence_adjusts_upward(self, usd_stage, source_chunk, hypothesis, predicted):
        cf = generate_counterfactual(
            usd_stage, source_chunk, hypothesis, predicted, timestamp=1000.0,
        )
        # Actual matches predicted closely
        updated = validate_counterfactual(
            usd_stage, cf.cf_id,
            actual_outcome={"aesthetic": 0.85, "lighting": 0.9},
            timestamp=2000.0,
        )
        assert updated.confidence > DEFAULT_CONFIDENCE

    def test_confidence_adjusts_downward(self, usd_stage, source_chunk, hypothesis, predicted):
        cf = generate_counterfactual(
            usd_stage, source_chunk, hypothesis, predicted,
            confidence=0.8, timestamp=1000.0,
        )
        # Actual is very different from predicted
        updated = validate_counterfactual(
            usd_stage, cf.cf_id,
            actual_outcome={"aesthetic": 0.1, "lighting": 0.1},
            timestamp=2000.0,
        )
        assert updated.confidence < 0.8

    def test_not_found_raises(self, usd_stage):
        with pytest.raises(CounterfactualError, match="not found"):
            validate_counterfactual(
                usd_stage, "nonexistent",
                actual_outcome={"aesthetic": 0.5},
            )

    def test_invalid_actual_raises(self, usd_stage, source_chunk, hypothesis, predicted):
        cf = generate_counterfactual(
            usd_stage, source_chunk, hypothesis, predicted, timestamp=1000.0,
        )
        with pytest.raises(CounterfactualError, match="Actual score"):
            validate_counterfactual(
                usd_stage, cf.cf_id,
                actual_outcome={"aesthetic": -0.1},
            )

    def test_validated_status_on_high_accuracy(self, usd_stage, source_chunk, hypothesis):
        # Predict exactly what we'll validate with
        pred = {"aesthetic": 0.9, "lighting": 0.9}
        cf = generate_counterfactual(
            usd_stage, source_chunk, hypothesis, pred,
            confidence=0.8, timestamp=1000.0,
        )
        updated = validate_counterfactual(
            usd_stage, cf.cf_id,
            actual_outcome={"aesthetic": 0.9, "lighting": 0.9},
            timestamp=2000.0,
        )
        # accuracy = 1.0, new_conf = 0.5*0.8 + 0.5*1.0 = 0.9 >= 0.7
        assert updated.status == "validated"


# ---------------------------------------------------------------------------
# promote_validated (requires usd-core)
# ---------------------------------------------------------------------------

class TestPromoteValidated:
    def test_no_pending(self, usd_stage):
        result = promote_validated(usd_stage)
        assert result == []

    def test_promotes_validated(self, usd_stage, source_chunk, hypothesis):
        pred = {"aesthetic": 0.9}
        cf = generate_counterfactual(
            usd_stage, source_chunk, hypothesis, pred,
            confidence=0.8, timestamp=1000.0,
        )
        # Validate with perfect match
        validate_counterfactual(
            usd_stage, cf.cf_id,
            actual_outcome={"aesthetic": 0.9},
            timestamp=2000.0,
        )
        promoted = promote_validated(usd_stage)
        assert cf.cf_id in promoted

        # Check it exists in validated
        val_path = f"/counterfactuals/validated/cf_{cf.cf_id}"
        assert usd_stage.read(val_path) is True

    def test_does_not_promote_pending(self, usd_stage, source_chunk, hypothesis, predicted):
        cf = generate_counterfactual(
            usd_stage, source_chunk, hypothesis, predicted, timestamp=1000.0,
        )
        promoted = promote_validated(usd_stage)
        assert cf.cf_id not in promoted

    def test_pending_marked_promoted(self, usd_stage, source_chunk, hypothesis):
        pred = {"aesthetic": 0.9}
        cf = generate_counterfactual(
            usd_stage, source_chunk, hypothesis, pred,
            confidence=0.8, timestamp=1000.0,
        )
        validate_counterfactual(
            usd_stage, cf.cf_id,
            actual_outcome={"aesthetic": 0.9},
            timestamp=2000.0,
        )
        promote_validated(usd_stage)
        pend_path = f"/counterfactuals/pending/cf_{cf.cf_id}"
        assert usd_stage.read(pend_path, "status") == "promoted"


# ---------------------------------------------------------------------------
# list_pending / list_validated (requires usd-core)
# ---------------------------------------------------------------------------

class TestListFunctions:
    def test_list_pending_empty(self, usd_stage):
        assert list_pending(usd_stage) == []

    def test_list_pending_returns_pending(self, usd_stage, source_chunk, hypothesis, predicted):
        generate_counterfactual(
            usd_stage, source_chunk, hypothesis, predicted, timestamp=1000.0,
        )
        results = list_pending(usd_stage)
        assert len(results) == 1
        assert results[0].status == "pending"

    def test_list_validated_empty(self, usd_stage):
        assert list_validated(usd_stage) == []

    def test_list_validated_after_promote(self, usd_stage, source_chunk, hypothesis):
        pred = {"aesthetic": 0.9}
        cf = generate_counterfactual(
            usd_stage, source_chunk, hypothesis, pred,
            confidence=0.8, timestamp=1000.0,
        )
        validate_counterfactual(
            usd_stage, cf.cf_id,
            actual_outcome={"aesthetic": 0.9},
            timestamp=2000.0,
        )
        promote_validated(usd_stage)
        results = list_validated(usd_stage)
        assert len(results) == 1

    def test_promoted_not_in_pending(self, usd_stage, source_chunk, hypothesis):
        pred = {"aesthetic": 0.9}
        cf = generate_counterfactual(
            usd_stage, source_chunk, hypothesis, pred,
            confidence=0.8, timestamp=1000.0,
        )
        validate_counterfactual(
            usd_stage, cf.cf_id,
            actual_outcome={"aesthetic": 0.9},
            timestamp=2000.0,
        )
        promote_validated(usd_stage)
        pending = list_pending(usd_stage)
        assert all(p.cf_id != cf.cf_id for p in pending)


# ---------------------------------------------------------------------------
# Cycle 65: json.loads guards in _prim_to_cf and validate_counterfactual
# ---------------------------------------------------------------------------

class TestPrimToCfJsonDecodeGuard:
    """Cycle 65: _prim_to_cf must not crash on corrupted JSON in hypothesis attribute."""

    def _make_mock_prim(self, cf_id="cf_test", hypothesis_json="{}"):
        """Build a MagicMock prim that returns the given hypothesis JSON string."""
        from unittest.mock import MagicMock
        prim = MagicMock()

        def make_valid_attr(value):
            attr = MagicMock()
            attr.IsValid.return_value = True
            attr.Get.return_value = value
            return attr

        def make_invalid_attr():
            attr = MagicMock()
            attr.IsValid.return_value = False
            return attr

        def get_attribute(name):
            if name == "cf_id":
                return make_valid_attr(cf_id)
            elif name == "hypothesis":
                return make_valid_attr(hypothesis_json)
            else:
                return make_invalid_attr()

        prim.GetAttribute.side_effect = get_attribute
        return prim

    def test_corrupted_hypothesis_json_returns_cf_with_empty_dict(self):
        """_prim_to_cf with corrupted hypothesis JSON must return Counterfactual with {} hypothesis."""
        from agent.stage.counterfactuals import _prim_to_cf
        prim = self._make_mock_prim(hypothesis_json="{corrupted: json{{")
        cf = _prim_to_cf(prim)
        assert cf is not None, "_prim_to_cf returned None on corrupted JSON"
        assert cf.hypothesis == {}

    def test_valid_hypothesis_json_parsed_correctly(self):
        """_prim_to_cf with valid JSON hypothesis must parse it correctly."""
        import json
        from agent.stage.counterfactuals import _prim_to_cf
        hypothesis = {"sampler": "dpm++", "cfg": 7.5}
        prim = self._make_mock_prim(hypothesis_json=json.dumps(hypothesis))
        cf = _prim_to_cf(prim)
        assert cf is not None
        assert cf.hypothesis == hypothesis

    def test_empty_string_hypothesis_falls_back_to_empty_dict(self):
        """_prim_to_cf with empty string hypothesis must not crash."""
        from agent.stage.counterfactuals import _prim_to_cf
        prim = self._make_mock_prim(hypothesis_json="")
        # json.loads("") raises JSONDecodeError — guard must catch it
        cf = _prim_to_cf(prim)
        # Either returns None (invalid prim) or a CF with empty hypothesis
        if cf is not None:
            assert isinstance(cf.hypothesis, dict)


class TestValidateCfHypothesisJsonGuard:
    """Cycle 65: validate_counterfactual must not crash on corrupted USD hypothesis."""

    def test_validate_reads_hypothesis_without_crash(self, usd_stage, source_chunk, hypothesis):
        """validate_counterfactual must complete even if the stored hypothesis JSON is somehow corrupted."""
        cf = generate_counterfactual(
            usd_stage, source_chunk, hypothesis, {"aesthetic": 0.8},
            confidence=0.9, timestamp=1000.0,
        )
        # Overwrite the hypothesis attribute with corrupted JSON.
        # Storage sites prepend `cf_` (counterfactuals.py:132,179) so prim
        # names begin with a letter; mirror that here.
        prim_path = f"/counterfactuals/pending/cf_{cf.cf_id}"
        usd_stage._stage.GetPrimAtPath(prim_path).CreateAttribute(
            "hypothesis", __import__("pxr").Sdf.ValueTypeNames.String
        ).Set("{{broken json")

        # validate_counterfactual must not crash
        result = validate_counterfactual(
            usd_stage, cf.cf_id,
            actual_outcome={"aesthetic": 0.85},
            timestamp=2000.0,
        )
        assert result is not None
        assert result.cf_id == cf.cf_id
