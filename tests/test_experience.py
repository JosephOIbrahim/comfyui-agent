"""Tests for agent/stage/experience.py — no real I/O."""

from __future__ import annotations

import time

import pytest

from agent.stage.experience import (
    DEFAULT_DECAY_RATE,
    DEFAULT_INITIAL_WEIGHT,
    ExperienceChunk,
    ExperienceError,
    _compute_prediction_accuracy,
    _generate_chunk_id,
    get_statistics,
    query_experience,
    record_experience,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def usd_stage():
    pytest.importorskip("pxr", reason="usd-core not installed")
    from agent.stage.cognitive_stage import CognitiveWorkflowStage
    return CognitiveWorkflowStage()


@pytest.fixture
def sample_outcome():
    return {"aesthetic": 0.8, "depth": 0.6, "lighting": 0.9}


@pytest.fixture
def sample_predicted():
    return {"aesthetic": 0.7, "depth": 0.5, "lighting": 0.85}


# ---------------------------------------------------------------------------
# ExperienceChunk dataclass
# ---------------------------------------------------------------------------

class TestExperienceChunk:
    def test_defaults(self):
        chunk = ExperienceChunk(
            chunk_id="abc",
            initial_state={},
            decisions=[],
            outcome={"aesthetic": 0.5},
        )
        assert chunk.predicted_outcome == {}
        assert chunk.prediction_accuracy == 0.0
        assert chunk.experience_weight == DEFAULT_INITIAL_WEIGHT

    def test_to_dict_contains_all_fields(self):
        chunk = ExperienceChunk(
            chunk_id="x",
            initial_state={"key": "val"},
            decisions=[{"action": "set_steps"}],
            outcome={"aesthetic": 0.9},
            context_signature_hash="abc123",
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == "x"
        assert d["initial_state"] == {"key": "val"}
        assert d["context_signature_hash"] == "abc123"
        assert len(d) == 9

    def test_decayed_weight_at_creation(self):
        now = time.time()
        chunk = ExperienceChunk(
            chunk_id="a", initial_state={}, decisions=[],
            outcome={}, timestamp=now,
        )
        # At creation, weight should be very close to initial
        assert abs(chunk.decayed_weight(now) - DEFAULT_INITIAL_WEIGHT) < 1e-9

    def test_decayed_weight_after_one_day(self):
        now = time.time()
        chunk = ExperienceChunk(
            chunk_id="a", initial_state={}, decisions=[],
            outcome={}, timestamp=now - 86400,  # 1 day ago
        )
        expected = DEFAULT_INITIAL_WEIGHT * DEFAULT_DECAY_RATE
        assert abs(chunk.decayed_weight(now) - expected) < 1e-9

    def test_decayed_weight_after_30_days(self):
        now = time.time()
        chunk = ExperienceChunk(
            chunk_id="a", initial_state={}, decisions=[],
            outcome={}, timestamp=now - (30 * 86400),
        )
        weight = chunk.decayed_weight(now)
        assert weight < DEFAULT_INITIAL_WEIGHT
        assert weight > 0.0

    def test_decayed_weight_future_timestamp(self):
        now = time.time()
        chunk = ExperienceChunk(
            chunk_id="a", initial_state={}, decisions=[],
            outcome={}, timestamp=now + 86400,  # future
        )
        # age_days clamped to 0, so weight = initial
        assert abs(chunk.decayed_weight(now) - DEFAULT_INITIAL_WEIGHT) < 1e-9

    def test_custom_decay_rate(self):
        now = time.time()
        chunk = ExperienceChunk(
            chunk_id="a", initial_state={}, decisions=[],
            outcome={}, timestamp=now - 86400,
        )
        weight = chunk.decayed_weight(now, decay_rate=0.5)
        assert abs(weight - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestComputePredictionAccuracy:
    def test_perfect_prediction(self):
        outcome = {"aesthetic": 0.8, "depth": 0.6}
        predicted = {"aesthetic": 0.8, "depth": 0.6}
        assert abs(_compute_prediction_accuracy(outcome, predicted) - 1.0) < 1e-9

    def test_worst_prediction(self):
        outcome = {"aesthetic": 1.0}
        predicted = {"aesthetic": 0.0}
        assert abs(_compute_prediction_accuracy(outcome, predicted) - 0.0) < 1e-9

    def test_partial_overlap(self):
        outcome = {"aesthetic": 0.8, "depth": 0.6}
        predicted = {"aesthetic": 0.7}  # only aesthetic overlaps
        acc = _compute_prediction_accuracy(outcome, predicted)
        # MAE = |0.8 - 0.7| = 0.1; accuracy = 0.9
        assert abs(acc - 0.9) < 1e-9

    def test_no_overlap(self):
        outcome = {"aesthetic": 0.8}
        predicted = {"depth": 0.6}
        assert _compute_prediction_accuracy(outcome, predicted) == 0.0

    def test_empty_predicted(self):
        assert _compute_prediction_accuracy({"aesthetic": 0.5}, {}) == 0.0

    def test_empty_outcome(self):
        assert _compute_prediction_accuracy({}, {"aesthetic": 0.5}) == 0.0


class TestGenerateChunkId:
    def test_deterministic(self):
        a = _generate_chunk_id("hash1", 1000.0)
        b = _generate_chunk_id("hash1", 1000.0)
        assert a == b

    def test_different_inputs_different_ids(self):
        a = _generate_chunk_id("hash1", 1000.0)
        b = _generate_chunk_id("hash2", 1000.0)
        assert a != b

    def test_length_is_16(self):
        cid = _generate_chunk_id("x", 0.0)
        assert len(cid) == 16


# ---------------------------------------------------------------------------
# record_experience (requires usd-core)
# ---------------------------------------------------------------------------

class TestRecordExperience:
    def test_basic_record(self, usd_stage, sample_outcome):
        chunk = record_experience(
            usd_stage,
            initial_state={"steps": 20},
            decisions=[{"set": "steps", "to": 30}],
            outcome=sample_outcome,
            context_signature_hash="abc123",
            timestamp=1000.0,
        )
        assert isinstance(chunk, ExperienceChunk)
        assert chunk.context_signature_hash == "abc123"

    def test_prim_created(self, usd_stage, sample_outcome):
        chunk = record_experience(
            usd_stage,
            initial_state={},
            decisions=[],
            outcome=sample_outcome,
            timestamp=2000.0,
        )
        prim_path = f"/experience/exp_{chunk.chunk_id}"
        assert usd_stage.read(prim_path) is True

    def test_outcome_scores_stored(self, usd_stage, sample_outcome):
        chunk = record_experience(
            usd_stage,
            initial_state={},
            decisions=[],
            outcome=sample_outcome,
            timestamp=3000.0,
        )
        prim_path = f"/experience/exp_{chunk.chunk_id}"
        stored = usd_stage.read(prim_path, "outcome:aesthetic")
        assert abs(stored - 0.8) < 1e-9

    def test_predicted_outcome_stored(self, usd_stage, sample_outcome, sample_predicted):
        chunk = record_experience(
            usd_stage,
            initial_state={},
            decisions=[],
            outcome=sample_outcome,
            predicted_outcome=sample_predicted,
            timestamp=4000.0,
        )
        prim_path = f"/experience/exp_{chunk.chunk_id}"
        stored = usd_stage.read(prim_path, "predicted:aesthetic")
        assert abs(stored - 0.7) < 1e-9

    def test_prediction_accuracy_computed(self, usd_stage, sample_outcome, sample_predicted):
        chunk = record_experience(
            usd_stage,
            initial_state={},
            decisions=[],
            outcome=sample_outcome,
            predicted_outcome=sample_predicted,
            timestamp=5000.0,
        )
        assert chunk.prediction_accuracy > 0.0

    def test_invalid_outcome_raises(self, usd_stage):
        with pytest.raises(ExperienceError, match="Outcome score"):
            record_experience(
                usd_stage,
                initial_state={},
                decisions=[],
                outcome={"aesthetic": 1.5},
                timestamp=6000.0,
            )

    def test_invalid_predicted_raises(self, usd_stage):
        with pytest.raises(ExperienceError, match="Predicted score"):
            record_experience(
                usd_stage,
                initial_state={},
                decisions=[],
                outcome={"aesthetic": 0.5},
                predicted_outcome={"aesthetic": -0.1},
                timestamp=7000.0,
            )

    def test_negative_outcome_raises(self, usd_stage):
        with pytest.raises(ExperienceError):
            record_experience(
                usd_stage,
                initial_state={},
                decisions=[],
                outcome={"depth": -0.01},
                timestamp=8000.0,
            )

    def test_empty_outcome_ok(self, usd_stage):
        chunk = record_experience(
            usd_stage,
            initial_state={},
            decisions=[],
            outcome={},
            timestamp=9000.0,
        )
        assert chunk.outcome == {}

    def test_initial_state_stored_as_json(self, usd_stage):
        chunk = record_experience(
            usd_stage,
            initial_state={"model": "sdxl", "steps": 25},
            decisions=[],
            outcome={"aesthetic": 0.5},
            timestamp=10000.0,
        )
        prim_path = f"/experience/exp_{chunk.chunk_id}"
        raw = usd_stage.read(prim_path, "initial_state")
        import json
        parsed = json.loads(raw)
        assert parsed["model"] == "sdxl"


# ---------------------------------------------------------------------------
# query_experience (requires usd-core)
# ---------------------------------------------------------------------------

class TestQueryExperience:
    def test_empty_stage(self, usd_stage):
        results = query_experience(usd_stage)
        assert results == []

    def test_returns_recorded(self, usd_stage):
        record_experience(
            usd_stage,
            initial_state={},
            decisions=[],
            outcome={"aesthetic": 0.5},
            timestamp=1000.0,
        )
        results = query_experience(usd_stage)
        assert len(results) == 1

    def test_filter_by_signature_hash(self, usd_stage):
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.5},
            context_signature_hash="aaa",
            timestamp=1000.0,
        )
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.6},
            context_signature_hash="bbb",
            timestamp=2000.0,
        )
        results = query_experience(usd_stage, context_signature_hash="aaa")
        assert len(results) == 1
        assert results[0].context_signature_hash == "aaa"

    def test_sorted_by_timestamp_desc(self, usd_stage):
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.5}, timestamp=1000.0,
        )
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.6}, timestamp=3000.0,
        )
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.7}, timestamp=2000.0,
        )
        results = query_experience(usd_stage)
        timestamps = [c.timestamp for c in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_limit(self, usd_stage):
        for i in range(5):
            record_experience(
                usd_stage, initial_state={}, decisions=[],
                outcome={"aesthetic": 0.5}, timestamp=float(i * 1000),
            )
        results = query_experience(usd_stage, limit=3)
        assert len(results) == 3

    def test_min_weight_filters(self, usd_stage):
        # Record with old timestamp so weight decays below threshold
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.5},
            timestamp=1.0,  # very old
        )
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.9},
            timestamp=time.time(),  # fresh
        )
        results = query_experience(usd_stage, min_weight=0.9)
        assert len(results) == 1

    def test_all_none_filter_returns_all(self, usd_stage):
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.5}, timestamp=1000.0,
        )
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.6}, timestamp=2000.0,
        )
        results = query_experience(usd_stage, context_signature_hash=None)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# get_statistics (requires usd-core)
# ---------------------------------------------------------------------------

class TestGetStatistics:
    def test_empty_stage(self, usd_stage):
        stats = get_statistics(usd_stage)
        assert stats["total_count"] == 0
        assert stats["avg_outcome"] == {}

    def test_single_experience(self, usd_stage):
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.8, "depth": 0.6},
            timestamp=time.time(),
        )
        stats = get_statistics(usd_stage)
        assert stats["total_count"] == 1
        assert abs(stats["avg_outcome"]["aesthetic"] - 0.8) < 1e-9

    def test_multiple_experiences_averaged(self, usd_stage):
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.8}, timestamp=1000.0,
        )
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.4}, timestamp=2000.0,
        )
        stats = get_statistics(usd_stage)
        assert stats["total_count"] == 2
        assert abs(stats["avg_outcome"]["aesthetic"] - 0.6) < 1e-9

    def test_unique_signatures_counted(self, usd_stage):
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.5},
            context_signature_hash="aaa",
            timestamp=1000.0,
        )
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.5},
            context_signature_hash="aaa",
            timestamp=2000.0,
        )
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.5},
            context_signature_hash="bbb",
            timestamp=3000.0,
        )
        stats = get_statistics(usd_stage)
        assert stats["unique_signatures"] == 2

    def test_filter_by_signature(self, usd_stage):
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.8},
            context_signature_hash="target",
            timestamp=1000.0,
        )
        record_experience(
            usd_stage, initial_state={}, decisions=[],
            outcome={"aesthetic": 0.2},
            context_signature_hash="other",
            timestamp=2000.0,
        )
        stats = get_statistics(usd_stage, context_signature_hash="target")
        assert stats["total_count"] == 1
        assert abs(stats["avg_outcome"]["aesthetic"] - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# Cycle 61 — allow_nan=False coverage for experience serialization
# ---------------------------------------------------------------------------

class TestNaNSafety:
    """Cycle 61: record_experience must reject NaN in initial_state/decisions."""

    def test_nan_in_initial_state_raises(self):
        """NaN in initial_state must raise ValueError (allow_nan=False guard)."""
        from unittest.mock import MagicMock
        cws = MagicMock()
        with pytest.raises(ValueError):
            record_experience(
                cws,
                initial_state={"cfg": float("nan")},
                decisions=[],
                outcome={"aesthetic": 0.8},
                context_signature_hash="test_c61_nan_init",
            )

    def test_inf_in_initial_state_raises(self):
        """Infinity in initial_state must raise ValueError (allow_nan=False guard)."""
        from unittest.mock import MagicMock
        cws = MagicMock()
        with pytest.raises(ValueError):
            record_experience(
                cws,
                initial_state={"steps": float("inf")},
                decisions=[],
                outcome={"aesthetic": 0.8},
                context_signature_hash="test_c61_inf_init",
            )

    def test_nan_in_decisions_raises(self):
        """NaN in decisions list must raise ValueError (allow_nan=False guard)."""
        from unittest.mock import MagicMock
        cws = MagicMock()
        with pytest.raises(ValueError):
            record_experience(
                cws,
                initial_state={},
                decisions=[{"cfg": float("nan"), "action": "lower"}],
                outcome={"aesthetic": 0.8},
                context_signature_hash="test_c61_nan_dec",
            )

    def test_valid_data_does_not_raise(self):
        """Normal finite data must not raise (NaN guard must not block valid calls)."""
        from unittest.mock import MagicMock
        cws = MagicMock()
        # Should complete without raising
        chunk = record_experience(
            cws,
            initial_state={"cfg": 7.0, "steps": 20},
            decisions=[{"action": "lower_cfg", "delta": -1.5}],
            outcome={"aesthetic": 0.85},
            context_signature_hash="test_c61_valid",
        )
        assert chunk.context_signature_hash == "test_c61_valid"


# ---------------------------------------------------------------------------
# Cycle 65: json.loads guards in _prim_to_chunk
# ---------------------------------------------------------------------------

class TestPrimToChunkJsonDecodeGuard:
    """Cycle 65: _prim_to_chunk must not crash on corrupted JSON in initial_state/decisions."""

    def _make_mock_prim(self, initial_state_json="{}", decisions_json="[]"):
        """Build a MagicMock prim returning the given JSON strings."""
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
            if name == "chunk_id":
                return make_valid_attr("chunk_001")
            elif name == "initial_state":
                return make_valid_attr(initial_state_json)
            elif name == "decisions":
                return make_valid_attr(decisions_json)
            else:
                return make_invalid_attr()

        prim.GetAttribute.side_effect = get_attribute
        return prim

    def test_corrupted_initial_state_json_returns_chunk_with_empty_dict(self):
        """_prim_to_chunk with corrupted initial_state JSON must return chunk with {} not crash."""
        from agent.stage.experience import _prim_to_chunk
        prim = self._make_mock_prim(initial_state_json="{broken: json")
        chunk = _prim_to_chunk(prim)
        assert chunk is not None, "_prim_to_chunk returned None on corrupted initial_state"
        assert chunk.initial_state == {}

    def test_corrupted_decisions_json_returns_chunk_with_empty_list(self):
        """_prim_to_chunk with corrupted decisions JSON must return chunk with [] not crash."""
        from agent.stage.experience import _prim_to_chunk
        prim = self._make_mock_prim(decisions_json="[not valid json}")
        chunk = _prim_to_chunk(prim)
        assert chunk is not None, "_prim_to_chunk returned None on corrupted decisions"
        assert chunk.decisions == []

    def test_both_corrupted_returns_chunk_with_empty_fields(self):
        """Both fields corrupted must return chunk with {} initial_state and [] decisions."""
        from agent.stage.experience import _prim_to_chunk
        prim = self._make_mock_prim(
            initial_state_json="{{invalid",
            decisions_json="}}invalid",
        )
        chunk = _prim_to_chunk(prim)
        assert chunk is not None
        assert chunk.initial_state == {}
        assert chunk.decisions == []

    def test_valid_json_parsed_correctly(self):
        """Valid JSON in both fields must be parsed and included in the chunk."""
        import json
        from agent.stage.experience import _prim_to_chunk
        state = {"cfg": 7.0, "steps": 25}
        decisions = [{"action": "adjust_cfg", "delta": -1.0}]
        prim = self._make_mock_prim(
            initial_state_json=json.dumps(state),
            decisions_json=json.dumps(decisions),
        )
        chunk = _prim_to_chunk(prim)
        assert chunk is not None
        assert chunk.initial_state == state
        assert chunk.decisions == decisions
