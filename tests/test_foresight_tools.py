"""Tests for agent/stage/foresight_tools.py — all mocked, no real I/O."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.stage.foresight_tools import TOOLS, handle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def usd_stage():
    pytest.importorskip("pxr", reason="usd-core not installed")
    from agent.stage.cognitive_stage import CognitiveWorkflowStage
    return CognitiveWorkflowStage()


@pytest.fixture(autouse=True)
def _reset_registry():
    """Clean session registry between tests."""
    from agent.session_context import get_registry
    yield
    get_registry().clear()


# ---------------------------------------------------------------------------
# TOOLS schema validation
# ---------------------------------------------------------------------------

class TestToolSchemas:
    def test_tools_is_list(self):
        assert isinstance(TOOLS, list)

    def test_five_tools(self):
        assert len(TOOLS) == 5

    def test_all_have_name(self):
        for t in TOOLS:
            assert "name" in t

    def test_all_have_description(self):
        for t in TOOLS:
            assert "description" in t
            assert len(t["description"]) > 10

    def test_all_have_input_schema(self):
        for t in TOOLS:
            assert "input_schema" in t
            assert t["input_schema"]["type"] == "object"

    def test_expected_tool_names(self):
        names = {t["name"] for t in TOOLS}
        assert names == {
            "predict_experiment",
            "record_experience",
            "get_experience_stats",
            "list_counterfactuals",
            "get_prediction_accuracy",
        }


# ---------------------------------------------------------------------------
# handle() dispatch
# ---------------------------------------------------------------------------

class TestHandleDispatch:
    def test_unknown_tool(self):
        result = json.loads(handle("nonexistent_tool", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]


# ---------------------------------------------------------------------------
# predict_experiment
# ---------------------------------------------------------------------------

class TestPredictExperiment:
    def test_no_stage_returns_error(self):
        with patch(
            "agent.stage.foresight_tools._get_stage", return_value=None,
        ):
            result = json.loads(handle("predict_experiment", {
                "proposed_change": {"param": "steps", "direction": "increase"},
            }))
        assert "error" in result

    def test_with_stage(self, usd_stage):
        with patch(
            "agent.stage.foresight_tools._get_stage",
            return_value=usd_stage,
        ), patch(
            "agent.stage.foresight_tools._get_ctx",
        ) as mock_ctx:
            mock_ctx.return_value = MagicMock(
                workflow_signature=None, arbiter=None,
            )
            result = json.loads(handle("predict_experiment", {
                "proposed_change": {"param": "steps", "direction": "increase"},
            }))
        assert "axis_scores" in result
        assert "confidence" in result
        assert "phase" in result

    def test_with_arbiter(self, usd_stage):
        from agent.stage.arbiter import Arbiter
        arbiter = Arbiter()

        with patch(
            "agent.stage.foresight_tools._get_stage",
            return_value=usd_stage,
        ), patch(
            "agent.stage.foresight_tools._get_ctx",
        ) as mock_ctx:
            mock_ctx.return_value = MagicMock(
                workflow_signature=None,
                arbiter=arbiter,
                ratchet=None,
            )
            result = json.loads(handle("predict_experiment", {
                "proposed_change": {"param": "steps", "direction": "increase"},
            }))
        assert "arbiter_mode" in result


# ---------------------------------------------------------------------------
# record_experience
# ---------------------------------------------------------------------------

class TestRecordExperience:
    def test_no_stage_returns_error(self):
        with patch(
            "agent.stage.foresight_tools._get_stage", return_value=None,
        ):
            result = json.loads(handle("record_experience", {
                "initial_state": {},
                "decisions": [],
                "outcome": {"aesthetic": 0.8},
            }))
        assert "error" in result

    def test_records_successfully(self, usd_stage):
        with patch(
            "agent.stage.foresight_tools._get_stage",
            return_value=usd_stage,
        ), patch(
            "agent.stage.foresight_tools._get_ctx",
        ) as mock_ctx:
            mock_ctx.return_value = MagicMock(workflow_signature=None)
            result = json.loads(handle("record_experience", {
                "initial_state": {"steps": 20},
                "decisions": [{"action": "set_steps"}],
                "outcome": {"aesthetic": 0.8, "lighting": 0.7},
            }))
        assert result["recorded"] is True
        assert "chunk_id" in result

    def test_invalid_score_returns_error(self, usd_stage):
        with patch(
            "agent.stage.foresight_tools._get_stage",
            return_value=usd_stage,
        ), patch(
            "agent.stage.foresight_tools._get_ctx",
        ) as mock_ctx:
            mock_ctx.return_value = MagicMock(workflow_signature=None)
            result = json.loads(handle("record_experience", {
                "initial_state": {},
                "decisions": [],
                "outcome": {"aesthetic": 1.5},
            }))
        assert "error" in result


# ---------------------------------------------------------------------------
# get_experience_stats
# ---------------------------------------------------------------------------

class TestGetExperienceStats:
    def test_no_stage_returns_error(self):
        with patch(
            "agent.stage.foresight_tools._get_stage", return_value=None,
        ):
            result = json.loads(handle("get_experience_stats", {}))
        assert "error" in result

    def test_empty_stats(self, usd_stage):
        with patch(
            "agent.stage.foresight_tools._get_stage",
            return_value=usd_stage,
        ):
            result = json.loads(handle("get_experience_stats", {}))
        assert result["total_count"] == 0

    def test_with_filter(self, usd_stage):
        from agent.stage.experience import record_experience
        record_experience(
            usd_stage,
            initial_state={}, decisions=[],
            outcome={"aesthetic": 0.8},
            context_signature_hash="target_hash",
            timestamp=1000.0,
        )
        with patch(
            "agent.stage.foresight_tools._get_stage",
            return_value=usd_stage,
        ):
            result = json.loads(handle("get_experience_stats", {
                "context_signature_hash": "target_hash",
            }))
        assert result["total_count"] == 1


# ---------------------------------------------------------------------------
# list_counterfactuals
# ---------------------------------------------------------------------------

class TestListCounterfactuals:
    def test_no_stage_returns_error(self):
        with patch(
            "agent.stage.foresight_tools._get_stage", return_value=None,
        ):
            result = json.loads(handle("list_counterfactuals", {}))
        assert "error" in result

    def test_empty(self, usd_stage):
        with patch(
            "agent.stage.foresight_tools._get_stage",
            return_value=usd_stage,
        ):
            result = json.loads(handle("list_counterfactuals", {}))
        assert result["pending_count"] == 0
        assert result["validated_count"] == 0

    def test_filter_pending_only(self, usd_stage):
        with patch(
            "agent.stage.foresight_tools._get_stage",
            return_value=usd_stage,
        ):
            result = json.loads(handle("list_counterfactuals", {
                "status": "pending",
            }))
        assert "pending" in result
        assert "validated" not in result

    def test_filter_validated_only(self, usd_stage):
        with patch(
            "agent.stage.foresight_tools._get_stage",
            return_value=usd_stage,
        ):
            result = json.loads(handle("list_counterfactuals", {
                "status": "validated",
            }))
        assert "validated" in result
        assert "pending" not in result


# ---------------------------------------------------------------------------
# get_prediction_accuracy
# ---------------------------------------------------------------------------

class TestGetPredictionAccuracy:
    def test_no_ratchet_returns_error(self):
        with patch(
            "agent.stage.foresight_tools._get_ratchet", return_value=None,
        ):
            result = json.loads(handle("get_prediction_accuracy", {}))
        assert "error" in result

    def test_no_predictions(self):
        from agent.stage.ratchet import Ratchet
        r = Ratchet()
        with patch(
            "agent.stage.foresight_tools._get_ratchet", return_value=r,
        ):
            result = json.loads(handle("get_prediction_accuracy", {}))
        assert result["predictions_made"] == 0

    def test_with_predictions(self, usd_stage):
        from agent.stage.cwm import predict
        from agent.stage.ratchet import Ratchet
        r = Ratchet(cws=usd_stage, cwm=predict)
        r.decide(
            "d1", {"aesthetic": 0.8},
            change_context={"param": "steps", "direction": "increase"},
        )
        with patch(
            "agent.stage.foresight_tools._get_ratchet", return_value=r,
        ):
            result = json.loads(handle("get_prediction_accuracy", {}))
        assert result["predictions_made"] >= 1
        assert "avg_accuracy" in result


# ---------------------------------------------------------------------------
# Cycle 55 — required field guards for foresight_tools handlers
# ---------------------------------------------------------------------------

class TestPredictExperimentRequiredField:
    """predict_experiment must return error when proposed_change is missing."""

    def test_missing_proposed_change_returns_error(self):
        result = json.loads(handle("predict_experiment", {}))
        assert "error" in result
        assert "proposed_change" in result["error"].lower()

    def test_none_proposed_change_returns_error(self):
        result = json.loads(handle("predict_experiment", {"proposed_change": None}))
        assert "error" in result


class TestRecordExperienceRequiredFields:
    """record_experience must return errors when required fields are missing."""

    def test_missing_initial_state_returns_error(self):
        result = json.loads(handle("record_experience", {
            "decisions": [], "outcome": {},
        }))
        assert "error" in result
        assert "initial_state" in result["error"].lower()

    def test_missing_decisions_returns_error(self):
        result = json.loads(handle("record_experience", {
            "initial_state": {}, "outcome": {},
        }))
        assert "error" in result
        assert "decisions" in result["error"].lower()

    def test_missing_outcome_returns_error(self):
        result = json.loads(handle("record_experience", {
            "initial_state": {}, "decisions": [],
        }))
        assert "error" in result
        assert "outcome" in result["error"].lower()
