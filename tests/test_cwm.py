"""Tests for agent/stage/cwm.py — Cognitive World Model, no real I/O."""

from __future__ import annotations

import pytest

from agent.stage.cwm import (
    PHASE_BLENDED,
    PHASE_PRIOR_ONLY,
    PredictedOutcome,
    _blend_scores,
    _classify_change,
    _compute_confidence,
    _compute_experience_scores,
    _get_prior_scores,
    predict,
)
from agent.stage.experience import ExperienceChunk, record_experience
from agent.stage.workflow_signature import WorkflowSignature


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def usd_stage():
    pytest.importorskip("pxr", reason="usd-core not installed")
    from agent.stage.cognitive_stage import CognitiveWorkflowStage
    return CognitiveWorkflowStage()


def _record_n(cws, n, sig_hash="test_hash", base_ts=1000.0):
    """Record n experiences into a stage."""
    for i in range(n):
        record_experience(
            cws,
            initial_state={"step": i},
            decisions=[],
            outcome={"aesthetic": 0.7, "lighting": 0.8},
            context_signature_hash=sig_hash,
            timestamp=base_ts + i,
        )


# ---------------------------------------------------------------------------
# PredictedOutcome
# ---------------------------------------------------------------------------

class TestPredictedOutcome:
    def test_composite_empty(self):
        po = PredictedOutcome(
            axis_scores={}, confidence=0.5,
            phase="prior_only", experience_count=0, similar_count=0,
        )
        assert po.composite() == 0.0

    def test_composite_single_axis(self):
        po = PredictedOutcome(
            axis_scores={"aesthetic": 0.8}, confidence=0.5,
            phase="prior_only", experience_count=0, similar_count=0,
        )
        assert abs(po.composite() - 0.8) < 1e-9

    def test_composite_multiple_axes(self):
        po = PredictedOutcome(
            axis_scores={"aesthetic": 0.8, "depth": 0.4},
            confidence=0.5, phase="blended",
            experience_count=50, similar_count=10,
        )
        assert abs(po.composite() - 0.6) < 1e-9

    def test_to_dict_contains_all_fields(self):
        po = PredictedOutcome(
            axis_scores={"aesthetic": 0.8},
            confidence=0.7, phase="blended",
            experience_count=50, similar_count=10,
            reasoning="test",
        )
        d = po.to_dict()
        assert "axis_scores" in d
        assert "confidence" in d
        assert "composite" in d
        assert "phase" in d
        assert "reasoning" in d

    def test_to_dict_composite_matches(self):
        po = PredictedOutcome(
            axis_scores={"aesthetic": 0.6}, confidence=0.5,
            phase="prior_only", experience_count=0, similar_count=0,
        )
        assert abs(po.to_dict()["composite"] - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# _classify_change
# ---------------------------------------------------------------------------

class TestClassifyChange:
    def test_increase_steps(self):
        assert _classify_change({"param": "steps", "direction": "increase"}) == "increase_steps"

    def test_decrease_cfg(self):
        assert _classify_change({"param": "cfg", "direction": "decrease"}) == "decrease_cfg"

    def test_add_controlnet(self):
        assert _classify_change({"action": "add_controlnet"}) == "add_controlnet"

    def test_add_lora(self):
        assert _classify_change({"action": "add_lora"}) == "add_lora"

    def test_unknown(self):
        assert _classify_change({"param": "weird", "direction": "up"}) is None

    def test_empty(self):
        assert _classify_change({}) is None

    def test_action_takes_priority(self):
        result = _classify_change({"action": "add_lora", "param": "steps"})
        assert result == "add_lora"


# ---------------------------------------------------------------------------
# _get_prior_scores
# ---------------------------------------------------------------------------

class TestGetPriorScores:
    def test_known_change(self):
        scores = _get_prior_scores({"param": "steps", "direction": "increase"})
        assert scores["aesthetic"] == 0.65

    def test_unknown_change_returns_defaults(self):
        scores = _get_prior_scores({"param": "unknown"})
        assert all(v == 0.5 for v in scores.values())

    def test_prior_fills_missing_axes(self):
        scores = _get_prior_scores({"action": "add_lora"})
        # add_lora only specifies aesthetic, rest should be 0.5
        assert scores["depth"] == 0.5
        assert scores["aesthetic"] == 0.6

    def test_all_outcome_axes_present(self):
        from agent.stage.experience import OUTCOME_AXES
        scores = _get_prior_scores({"param": "steps", "direction": "increase"})
        for axis in OUTCOME_AXES:
            assert axis in scores


# ---------------------------------------------------------------------------
# _compute_experience_scores
# ---------------------------------------------------------------------------

class TestComputeExperienceScores:
    def test_empty_chunks(self):
        assert _compute_experience_scores([]) == {}

    def test_single_chunk(self):
        chunk = ExperienceChunk(
            chunk_id="a", initial_state={}, decisions=[],
            outcome={"aesthetic": 0.9},
            timestamp=1000.0,
        )
        scores = _compute_experience_scores([chunk])
        assert "aesthetic" in scores
        assert scores["aesthetic"] > 0.0

    def test_multiple_chunks_averaged(self):
        chunks = [
            ExperienceChunk(
                chunk_id="a", initial_state={}, decisions=[],
                outcome={"aesthetic": 0.8}, timestamp=1000.0,
            ),
            ExperienceChunk(
                chunk_id="b", initial_state={}, decisions=[],
                outcome={"aesthetic": 0.4}, timestamp=1000.0,
            ),
        ]
        scores = _compute_experience_scores(chunks)
        # Equal timestamps = equal weights, so average
        assert abs(scores["aesthetic"] - 0.6) < 0.05

    def test_signature_match_bonus(self):
        sig = WorkflowSignature(model_family="sdxl")
        chunks = [
            ExperienceChunk(
                chunk_id="a", initial_state={}, decisions=[],
                outcome={"aesthetic": 0.9},
                context_signature_hash=sig.signature_hash(),
                timestamp=1000.0,
            ),
        ]
        scores = _compute_experience_scores(chunks, sig)
        assert scores["aesthetic"] > 0.0


# ---------------------------------------------------------------------------
# _blend_scores
# ---------------------------------------------------------------------------

class TestBlendScores:
    def test_prior_only_phase(self):
        prior = {"aesthetic": 0.5}
        exp = {"aesthetic": 0.9}
        blended, phase = _blend_scores(prior, exp, 10)
        assert phase == "prior_only"
        # With 10 exp, alpha = 10/30 * 0.3 = 0.1
        # blended = 0.9 * 0.5 + 0.1 * 0.9 = 0.54
        assert blended["aesthetic"] > 0.5
        assert blended["aesthetic"] < 0.9

    def test_blended_phase(self):
        prior = {"aesthetic": 0.5}
        exp = {"aesthetic": 0.9}
        _, phase = _blend_scores(prior, exp, 60)
        assert phase == "blended"

    def test_experience_dominant_phase(self):
        prior = {"aesthetic": 0.5}
        exp = {"aesthetic": 0.9}
        blended, phase = _blend_scores(prior, exp, 150)
        assert phase == "experience_dominant"
        # alpha = 0.9, so blended ≈ 0.1*0.5 + 0.9*0.9 = 0.86
        assert blended["aesthetic"] > 0.8

    def test_zero_experience(self):
        prior = {"aesthetic": 0.5}
        blended, phase = _blend_scores(prior, {}, 0)
        assert phase == "prior_only"
        assert abs(blended["aesthetic"] - 0.5) < 1e-9

    def test_missing_axis_in_experience_uses_prior(self):
        prior = {"aesthetic": 0.5, "depth": 0.6}
        exp = {"aesthetic": 0.9}  # depth missing
        blended, _ = _blend_scores(prior, exp, 50)
        assert "depth" in blended

    def test_extra_axis_in_experience(self):
        prior = {"aesthetic": 0.5}
        exp = {"aesthetic": 0.9, "depth": 0.7}
        blended, _ = _blend_scores(prior, exp, 50)
        assert "depth" in blended


# ---------------------------------------------------------------------------
# _compute_confidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def test_zero_experience(self):
        assert _compute_confidence(0, 0, "prior_only") == 0.1

    def test_increases_with_experience(self):
        c_low = _compute_confidence(10, 0, "prior_only")
        c_high = _compute_confidence(100, 0, "experience_dominant")
        assert c_high > c_low

    def test_similar_bonus(self):
        c_none = _compute_confidence(50, 0, "blended")
        c_some = _compute_confidence(50, 20, "blended")
        assert c_some > c_none

    def test_capped_at_one(self):
        c = _compute_confidence(1000, 1000, "experience_dominant")
        assert c <= 1.0

    def test_in_range(self):
        c = _compute_confidence(50, 10, "blended")
        assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# predict (requires usd-core)
# ---------------------------------------------------------------------------

class TestPredict:
    def test_prior_only_with_no_experience(self, usd_stage):
        result = predict(
            usd_stage,
            proposed_change={"param": "steps", "direction": "increase"},
        )
        assert result.phase == "prior_only"
        assert result.experience_count == 0
        assert result.confidence > 0.0
        assert "aesthetic" in result.axis_scores

    def test_prior_only_with_few_experiences(self, usd_stage):
        _record_n(usd_stage, 5)
        result = predict(
            usd_stage,
            proposed_change={"param": "steps", "direction": "increase"},
        )
        assert result.phase == "prior_only"
        assert result.experience_count == 5

    def test_blended_phase(self, usd_stage):
        _record_n(usd_stage, 50)
        result = predict(
            usd_stage,
            proposed_change={"param": "steps", "direction": "increase"},
        )
        assert result.phase == "blended"

    def test_experience_dominant_phase(self, usd_stage):
        _record_n(usd_stage, 120)
        result = predict(
            usd_stage,
            proposed_change={"param": "steps", "direction": "increase"},
        )
        assert result.phase == "experience_dominant"

    def test_with_signature(self, usd_stage):
        sig = WorkflowSignature(model_family="sdxl")
        _record_n(usd_stage, 5, sig_hash=sig.signature_hash())
        result = predict(
            usd_stage,
            proposed_change={"param": "cfg", "direction": "increase"},
            current_signature=sig,
        )
        assert result.similar_count > 0

    def test_unknown_change(self, usd_stage):
        result = predict(
            usd_stage,
            proposed_change={"param": "weird_thing"},
        )
        assert result.phase == "prior_only"
        assert "unknown change" in result.reasoning

    def test_result_scores_in_range(self, usd_stage):
        _record_n(usd_stage, 10)
        result = predict(
            usd_stage,
            proposed_change={"action": "add_controlnet"},
        )
        for score in result.axis_scores.values():
            assert 0.0 <= score <= 1.0

    def test_confidence_increases_with_experience(self, usd_stage):
        r1 = predict(
            usd_stage,
            proposed_change={"param": "steps", "direction": "increase"},
        )
        _record_n(usd_stage, 50)
        r2 = predict(
            usd_stage,
            proposed_change={"param": "steps", "direction": "increase"},
        )
        assert r2.confidence > r1.confidence

    def test_reasoning_not_empty(self, usd_stage):
        result = predict(
            usd_stage,
            proposed_change={"param": "steps", "direction": "increase"},
        )
        assert len(result.reasoning) > 0
