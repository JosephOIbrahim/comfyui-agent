"""CRUCIBLE tests for inter-module adapters (agent/brain/adapters/).

Adversarial: tests missing fields, empty inputs, boundary scores,
adapter registry routing, purity, and the verify↔intent round-trip.
"""

from __future__ import annotations

import copy

import pytest

from agent.brain.adapters import adapt, get_adapter
from agent.brain.adapters.vision_memory import (
    vision_to_outcome,
    patterns_to_vision_context,
)
from agent.brain.adapters.planner_orchestrator import (
    plan_step_to_subtask,
    subtask_result_to_completion,
)
from agent.brain.adapters.intent_verify import (
    intent_to_criteria,
    verify_against_intent,
)


# ---------------------------------------------------------------------------
# vision_to_outcome
# ---------------------------------------------------------------------------


class TestVisionToOutcome:
    def test_high_quality_scores_success(self):
        data = {
            "analysis": "Sharp image with good composition.",
            "scores": {"quality": 0.9, "composition": 0.85, "detail": 0.95},
            "suggestions": [],
        }
        result = vision_to_outcome(data)
        assert result["result"] == "success"
        assert result["details"]["average_score"] >= 0.8
        assert result["action"] == "vision_analysis"

    def test_low_quality_scores_needs_improvement(self):
        data = {
            "analysis": "Blurry image.",
            "scores": {"quality": 0.2, "composition": 0.3},
            "suggestions": ["Increase sharpness"],
        }
        result = vision_to_outcome(data)
        assert result["result"] == "needs_improvement"
        assert result["details"]["average_score"] < 0.5

    def test_medium_quality_partial(self):
        data = {
            "scores": {"quality": 0.6, "composition": 0.55},
        }
        result = vision_to_outcome(data)
        assert result["result"] == "partial"

    def test_missing_fields_graceful(self):
        result = vision_to_outcome({})
        assert result["result"] == "needs_improvement"
        assert result["details"]["average_score"] == 0.0
        assert result["details"]["suggestion_count"] == 0
        assert result["session"] == "default"

    def test_empty_scores_dict(self):
        result = vision_to_outcome({"scores": {}})
        assert result["details"]["average_score"] == 0.0

    def test_suggestions_truncated_to_10(self):
        data = {
            "scores": {"quality": 0.5},
            "suggestions": [f"suggestion_{i}" for i in range(20)],
        }
        result = vision_to_outcome(data)
        assert len(result["details"]["suggestions"]) <= 10

    def test_analysis_truncated_to_500(self):
        data = {
            "analysis": "x" * 1000,
            "scores": {"quality": 0.5},
        }
        result = vision_to_outcome(data)
        assert len(result["details"]["analysis_summary"]) <= 500

    def test_scores_sorted_deterministically(self):
        data = {
            "scores": {"z_score": 0.5, "a_score": 0.5, "m_score": 0.5},
        }
        result = vision_to_outcome(data)
        keys = list(result["details"]["scores"].keys())
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# patterns_to_vision_context
# ---------------------------------------------------------------------------


class TestPatternsToVisionContext:
    def test_empty_patterns_safe_defaults(self):
        result = patterns_to_vision_context({})
        assert result["known_issues"] == []
        assert result["expected_quality"] == 0.5

    def test_negative_outcomes_become_known_issues(self):
        data = {
            "patterns": [
                {"pattern": "blurry edges", "outcome": "needs_improvement"},
                {"pattern": "good lighting", "outcome": "success"},
                {"pattern": "noise artifacts", "outcome": "failure"},
            ],
        }
        result = patterns_to_vision_context(data)
        assert "blurry edges" in result["known_issues"]
        assert "noise artifacts" in result["known_issues"]
        assert "good lighting" not in result["known_issues"]

    def test_model_combos_to_expected_quality(self):
        data = {
            "model_combos": {
                "sdxl+detail_lora": {"success_rate": 0.9},
                "sd15+anime_lora": {"success_rate": 0.7},
            },
        }
        result = patterns_to_vision_context(data)
        assert result["expected_quality"] == 0.8  # (0.9+0.7)/2

    def test_known_issues_capped_at_20(self):
        patterns = [
            {"pattern": f"issue_{i}", "outcome": "failure"}
            for i in range(30)
        ]
        result = patterns_to_vision_context({"patterns": patterns})
        assert len(result["known_issues"]) <= 20

    def test_empty_pattern_string_excluded(self):
        data = {
            "patterns": [
                {"pattern": "", "outcome": "failure"},
            ],
        }
        result = patterns_to_vision_context(data)
        assert result["known_issues"] == []


# ---------------------------------------------------------------------------
# plan_step_to_subtask
# ---------------------------------------------------------------------------


class TestPlanStepToSubtask:
    def test_format_conversion(self):
        step = {
            "step_id": 3,
            "action": "load workflow",
            "tool": "load_workflow",
            "params": {"path": "/test.json"},
        }
        result = plan_step_to_subtask(step)
        assert result["name"] == "step_3_load_workflow"
        assert result["profile"] == "builder"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["tool"] == "load_workflow"

    def test_empty_step(self):
        result = plan_step_to_subtask({})
        assert result["name"] == "step_0_unnamed_action"
        assert result["steps"] == []

    def test_no_tool_means_no_steps(self):
        step = {"step_id": 1, "action": "think", "tool": "", "params": {}}
        result = plan_step_to_subtask(step)
        assert result["steps"] == []

    def test_params_sorted_deterministically(self):
        step = {
            "step_id": 1,
            "action": "test",
            "tool": "set_input",
            "params": {"z_param": 1, "a_param": 2, "m_param": 3},
        }
        result = plan_step_to_subtask(step)
        input_keys = list(result["steps"][0]["input"].keys())
        assert input_keys == sorted(input_keys)

    def test_custom_profile(self):
        step = {"step_id": 1, "action": "verify", "tool": "analyze_image", "params": {}}
        result = plan_step_to_subtask(step, profile="verifier")
        assert result["profile"] == "verifier"


# ---------------------------------------------------------------------------
# subtask_result_to_completion
# ---------------------------------------------------------------------------


class TestSubtaskResultToCompletion:
    def test_success_conversion(self):
        result = subtask_result_to_completion({
            "subtask_name": "step_3_load_workflow",
            "status": "success",
            "outputs": ["Loaded 7 nodes"],
        })
        assert result["step_id"] == 3
        assert result["success"] is True
        assert "Loaded 7 nodes" in result["output_summary"]

    def test_failure_conversion(self):
        result = subtask_result_to_completion({
            "subtask_name": "step_1_execute",
            "status": "failed",
            "outputs": [],
        })
        assert result["success"] is False
        assert "failed" in result["output_summary"]

    def test_timeout_is_failure(self):
        result = subtask_result_to_completion({
            "subtask_name": "step_2_run",
            "status": "timeout",
            "outputs": [],
        })
        assert result["success"] is False

    def test_step_id_extraction_from_name(self):
        result = subtask_result_to_completion({
            "subtask_name": "step_99_complex_action",
            "status": "success",
            "outputs": [],
        })
        assert result["step_id"] == 99

    def test_step_id_zero_for_unparseable(self):
        result = subtask_result_to_completion({
            "subtask_name": "custom_name_no_id",
            "status": "success",
            "outputs": [],
        })
        assert result["step_id"] == 0

    def test_output_summary_truncated(self):
        long_output = "x" * 1000
        result = subtask_result_to_completion({
            "subtask_name": "step_1_test",
            "status": "success",
            "outputs": [long_output],
        })
        assert len(result["output_summary"]) <= 500

    def test_dict_outputs_use_summary_key(self):
        result = subtask_result_to_completion({
            "subtask_name": "step_1_test",
            "status": "success",
            "outputs": [{"summary": "All good", "extra": "ignored"}],
        })
        assert "All good" in result["output_summary"]


# ---------------------------------------------------------------------------
# intent_to_criteria
# ---------------------------------------------------------------------------


class TestIntentToCriteria:
    def test_basic_intent(self):
        intent = {
            "description": "A photorealistic portrait",
            "parameters": {"cfg": 8, "steps": 30},
            "style": "photorealistic",
        }
        criteria = intent_to_criteria(intent)
        assert "expected_attributes" in criteria
        assert "quality_threshold" in criteria
        assert "style_match" in criteria
        assert criteria["style_match"] == "photorealistic"

    def test_high_cfg_maps_to_sharp(self):
        intent = {"parameters": {"cfg": 10}}
        criteria = intent_to_criteria(intent)
        attrs = criteria["expected_attributes"]
        assert any("sharp" in a for a in attrs)

    def test_low_cfg_maps_to_creative(self):
        intent = {"parameters": {"cfg": 3}}
        criteria = intent_to_criteria(intent)
        attrs = criteria["expected_attributes"]
        assert any("creative" in a for a in attrs)

    def test_high_steps_maps_to_detail(self):
        intent = {"parameters": {"steps": 50}}
        criteria = intent_to_criteria(intent)
        attrs = criteria["expected_attributes"]
        assert any("detail" in a.lower() for a in attrs)

    def test_low_denoise_maps_to_structure(self):
        intent = {"parameters": {"denoise": 0.2}}
        criteria = intent_to_criteria(intent)
        attrs = criteria["expected_attributes"]
        assert any("structure" in a for a in attrs)

    def test_empty_intent(self):
        criteria = intent_to_criteria({})
        assert criteria["quality_threshold"] == 0.6
        assert criteria["style_match"] == "unspecified"
        assert isinstance(criteria["expected_attributes"], list)


# ---------------------------------------------------------------------------
# verify_against_intent
# ---------------------------------------------------------------------------


class TestVerifyAgainstIntent:
    def test_intent_preserved_matching(self):
        verify_result = {
            "analysis": "Photorealistic portrait with sharp details",
            "scores": {"quality": 0.9, "composition": 0.85},
            "suggestions": [],
        }
        criteria = {
            "expected_attributes": ["style: photorealistic"],
            "quality_threshold": 0.7,
            "style_match": "photorealistic",
        }
        result = verify_against_intent(verify_result, criteria)
        assert result["intent_preserved"] is True
        assert result["deviations"] == []
        assert result["score"] >= 0.7

    def test_intent_deviated_low_quality(self):
        verify_result = {
            "analysis": "Blurry output",
            "scores": {"quality": 0.3},
            "suggestions": [],
        }
        criteria = {
            "expected_attributes": [],
            "quality_threshold": 0.7,
            "style_match": "unspecified",
        }
        result = verify_against_intent(verify_result, criteria)
        assert result["intent_preserved"] is False
        assert len(result["deviations"]) > 0
        assert any("threshold" in d.lower() for d in result["deviations"])

    def test_style_mismatch_deviation(self):
        verify_result = {
            "analysis": "Cartoon style output",
            "scores": {"quality": 0.9},
            "suggestions": [],
        }
        criteria = {
            "expected_attributes": [],
            "quality_threshold": 0.5,
            "style_match": "photorealistic",
        }
        result = verify_against_intent(verify_result, criteria)
        assert any("photorealistic" in d for d in result["deviations"])

    def test_empty_inputs(self):
        result = verify_against_intent({}, {})
        assert "intent_preserved" in result
        assert "deviations" in result
        assert "score" in result

    def test_deviations_capped_at_10(self):
        verify_result = {
            "scores": {"quality": 0.1},
            "suggestions": [f"Need more detail {i}" for i in range(20)],
            "analysis": "",
        }
        criteria = {
            "expected_attributes": ["high_detail" for _ in range(20)],
            "quality_threshold": 0.99,
            "style_match": "nonexistent_style",
        }
        result = verify_against_intent(verify_result, criteria)
        assert len(result["deviations"]) <= 10


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------


class TestAdapterRegistry:
    def test_vision_to_memory_registered(self):
        adapter = get_adapter("vision", "memory")
        assert adapter is not None

    def test_memory_to_vision_registered(self):
        adapter = get_adapter("memory", "vision")
        assert adapter is not None

    def test_planner_to_orchestrator_registered(self):
        adapter = get_adapter("planner", "orchestrator")
        assert adapter is not None

    def test_orchestrator_to_planner_registered(self):
        adapter = get_adapter("orchestrator", "planner")
        assert adapter is not None

    def test_intent_to_verify_registered(self):
        adapter = get_adapter("intent", "verify")
        assert adapter is not None

    def test_verify_to_intent_registered(self):
        adapter = get_adapter("verify", "intent")
        assert adapter is not None

    def test_unknown_route_returns_none(self):
        assert get_adapter("nonexistent", "fake") is None

    def test_adapt_unknown_route_raises_key_error(self):
        with pytest.raises(KeyError, match="No adapter registered"):
            adapt("nonexistent", "fake", {})

    def test_adapt_routes_correctly(self):
        """adapt("vision", "memory", data) should produce valid outcome."""
        data = {"scores": {"quality": 0.9}, "analysis": "good"}
        result = adapt("vision", "memory", data)
        assert "result" in result
        assert result["result"] == "success"

    def test_verify_intent_wrapper(self):
        """The verify->intent wrapper should unpack the compound dict."""
        data = {
            "verify_result": {
                "analysis": "good",
                "scores": {"quality": 0.9},
                "suggestions": [],
            },
            "criteria": {
                "expected_attributes": [],
                "quality_threshold": 0.5,
                "style_match": "unspecified",
            },
        }
        result = adapt("verify", "intent", data)
        assert "intent_preserved" in result


# ---------------------------------------------------------------------------
# Purity tests
# ---------------------------------------------------------------------------


class TestAdapterPurity:
    def test_vision_to_outcome_pure(self):
        data = {"scores": {"quality": 0.7}, "analysis": "test"}
        data_copy = copy.deepcopy(data)
        r1 = vision_to_outcome(data)
        r2 = vision_to_outcome(data_copy)
        assert r1 == r2
        # Input not modified
        assert data == data_copy

    def test_patterns_to_vision_context_pure(self):
        data = {"patterns": [{"pattern": "x", "outcome": "failure"}]}
        data_copy = copy.deepcopy(data)
        r1 = patterns_to_vision_context(data)
        r2 = patterns_to_vision_context(data_copy)
        assert r1 == r2

    def test_plan_step_to_subtask_pure(self):
        step = {"step_id": 1, "action": "load", "tool": "load_workflow", "params": {"a": 1}}
        step_copy = copy.deepcopy(step)
        r1 = plan_step_to_subtask(step)
        r2 = plan_step_to_subtask(step_copy)
        assert r1 == r2

    def test_intent_to_criteria_pure(self):
        intent = {"description": "test", "parameters": {"cfg": 7}, "style": "real"}
        intent_copy = copy.deepcopy(intent)
        r1 = intent_to_criteria(intent)
        r2 = intent_to_criteria(intent_copy)
        assert r1 == r2
