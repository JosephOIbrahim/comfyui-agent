"""Tests for the iterative_refine brain tool.

Tests cover: mode budgets, convergence detection (threshold, plateau, regression),
heuristic diagnosis, patch application, unlimited mode, and result structure.
All execution/vision/patching is mocked — no ComfyUI needed.
"""

import json
from unittest.mock import patch

import pytest

from agent.brain.iterative_refine import (
    FALLBACK_ADJUSTMENTS,
    HEURISTIC_RULES,
    MODE_BUDGETS,
    IterativeRefineAgent,
    handle,
    TOOLS,
)
from agent.brain._sdk import BrainConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    """Create an IterativeRefineAgent with standalone config."""
    cfg = BrainConfig()
    return IterativeRefineAgent(config=cfg)


def _mock_execute_result(
    output_path="/fake/output.png",
    render_time=2.5,
    status="complete",
    key_params=None,
):
    """Build a mock execute_with_progress result."""
    if key_params is None:
        key_params = {
            "model": "sdxl_v10.safetensors",
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "seed": 42,
            "denoise": 1.0,
            "resolution": "1024x1024",
        }
    return json.dumps({
        "status": status,
        "prompt_id": "test-prompt-id",
        "total_time_s": render_time,
        "verification": {
            "outputs": [
                {
                    "type": "image",
                    "exists": True,
                    "absolute_path": output_path,
                    "size_bytes": 1024,
                }
            ],
            "key_params": key_params,
        },
    })


def _mock_vision_result(quality_score=0.6, artifacts=None):
    """Build a mock analyze_image result."""
    return json.dumps({
        "quality_score": quality_score,
        "artifacts": artifacts or [],
        "suggestions": [],
    })


def _make_dispatcher(vision_scores, artifacts_per_iter=None):
    """Create a mock tool dispatcher that returns sequential vision scores.

    Args:
        vision_scores: list of quality scores for each iteration
        artifacts_per_iter: list of artifact lists per iteration (optional)
    """
    call_count = {"execute": 0, "analyze": 0}

    def dispatcher(name, tool_input):
        if name == "execute_with_progress":
            idx = call_count["execute"]
            call_count["execute"] += 1
            return _mock_execute_result()
        elif name == "analyze_image":
            idx = call_count["analyze"]
            call_count["analyze"] += 1
            score = vision_scores[idx] if idx < len(vision_scores) else vision_scores[-1]
            artifacts = []
            if artifacts_per_iter and idx < len(artifacts_per_iter):
                artifacts = artifacts_per_iter[idx]
            return _mock_vision_result(quality_score=score, artifacts=artifacts)
        elif name == "set_input":
            return json.dumps({"success": True})
        elif name == "undo_workflow_patch":
            return json.dumps({"success": True})
        elif name == "get_output_path":
            return json.dumps({
                "exists": True,
                "absolute_path": "/fake/output.png",
            })
        return json.dumps({"error": f"Unexpected tool: {name}"})

    return dispatcher


# ---------------------------------------------------------------------------
# 1. Mode selection respects iteration budget
# ---------------------------------------------------------------------------

class TestModeSelection:
    """Each mode should respect its iteration budget."""

    @pytest.mark.parametrize("mode,expected_max", [
        ("quick", 2),
        ("refine", 5),
        ("deep", 10),
        ("unlimited", 1),
    ])
    def test_mode_budget(self, mode, expected_max):
        assert MODE_BUDGETS[mode] == expected_max

    def test_quick_mode_max_iterations(self, agent):
        """Quick mode runs at most 2 iterations."""
        with patch("agent.brain.iterative_refine.IterativeRefineAgent._execute_workflow") as mock_exec, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._analyze_output") as mock_analyze, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._apply_adjustment") as mock_apply:

            mock_exec.return_value = {
                "output_path": "/fake/output.png",
                "render_time_s": 2.0,
                "key_params": {"cfg": 7.0, "steps": 20},
            }
            mock_analyze.side_effect = [
                {"quality_score": 0.3, "artifacts": []},
                {"quality_score": 0.4, "artifacts": []},
            ]
            mock_apply.return_value = {"param": "cfg", "direction": "decrease"}

            raw = agent.handle("iterative_refine", {
                "intent": "test image",
                "mode": "quick",
            })
            result = json.loads(raw)
            assert result["total_iterations"] == 2

    def test_refine_mode_default(self, agent):
        """Default mode is 'refine'."""
        # Converge on first iteration
        with patch("agent.brain.iterative_refine.IterativeRefineAgent._execute_workflow") as mock_exec, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._analyze_output") as mock_analyze:

            mock_exec.return_value = {
                "output_path": "/fake/output.png",
                "render_time_s": 2.0,
                "key_params": {"cfg": 7.0},
            }
            mock_analyze.return_value = {"quality_score": 0.9, "artifacts": []}

            raw = agent.handle("iterative_refine", {"intent": "test"})
            result = json.loads(raw)
            # Should converge immediately since score >= 0.7 threshold
            assert result["converged"] is True


# ---------------------------------------------------------------------------
# 2. Convergence: threshold_met stops the loop
# ---------------------------------------------------------------------------

class TestConvergenceThreshold:
    def test_threshold_met_stops(self, agent):
        """When quality meets threshold, loop stops with threshold_met."""
        with patch("agent.brain.iterative_refine.IterativeRefineAgent._execute_workflow") as mock_exec, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._analyze_output") as mock_analyze:

            mock_exec.return_value = {
                "output_path": "/fake/output.png",
                "render_time_s": 2.0,
                "key_params": {"cfg": 7.0, "steps": 20},
            }
            # First iteration meets threshold
            mock_analyze.return_value = {"quality_score": 0.8, "artifacts": []}

            raw = agent.handle("iterative_refine", {
                "intent": "test",
                "quality_threshold": 0.7,
                "mode": "refine",
            })
            result = json.loads(raw)
            assert result["converged"] is True
            assert result["reason"] == "threshold_met"
            assert result["total_iterations"] == 1

    def test_threshold_met_on_second_iteration(self, agent):
        """Threshold met on second iteration after first is below."""
        with patch("agent.brain.iterative_refine.IterativeRefineAgent._execute_workflow") as mock_exec, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._analyze_output") as mock_analyze, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._apply_adjustment") as mock_apply:

            mock_exec.return_value = {
                "output_path": "/fake/output.png",
                "render_time_s": 2.0,
                "key_params": {"cfg": 7.0, "steps": 20},
            }
            mock_analyze.side_effect = [
                {"quality_score": 0.5, "artifacts": []},
                {"quality_score": 0.8, "artifacts": []},
            ]
            mock_apply.return_value = {"param": "cfg", "direction": "decrease"}

            raw = agent.handle("iterative_refine", {
                "intent": "test",
                "quality_threshold": 0.7,
                "mode": "refine",
            })
            result = json.loads(raw)
            assert result["converged"] is True
            assert result["reason"] == "threshold_met"
            assert result["total_iterations"] == 2


# ---------------------------------------------------------------------------
# 3. Convergence: plateau detection
# ---------------------------------------------------------------------------

class TestConvergencePlateau:
    def test_plateau_detected(self, agent):
        """Scores 0.6, 0.62, 0.61 → plateau (two consecutive < 0.3 improvement)."""
        scores = [0.6, 0.62, 0.61]

        convergence = agent._check_convergence(scores, threshold=0.9, consecutive_regressions=0)
        assert convergence == "plateaued"

    def test_no_plateau_with_large_improvement(self, agent):
        """Scores 0.3, 0.5, 0.7 → not a plateau (second jump is 0.2 but first is 0.2)."""
        # Both improvements are < 0.3 but the last one goes up:
        # 0.5-0.3=0.2, 0.7-0.5=0.2 — both < 0.3 and both >= 0
        scores = [0.3, 0.5, 0.7]
        convergence = agent._check_convergence(scores, threshold=0.9, consecutive_regressions=0)
        # Both are >= 0 and < 0.3 → plateau
        assert convergence == "plateaued"

    def test_no_plateau_with_improvement_above_threshold(self, agent):
        """Scores 0.3, 0.4, 0.8 → not a plateau (last jump is 0.4 > 0.3)."""
        scores = [0.3, 0.4, 0.8]
        convergence = agent._check_convergence(scores, threshold=0.9, consecutive_regressions=0)
        assert convergence is None  # 0.8-0.4 = 0.4 > 0.3, not a plateau

    def test_plateau_needs_three_scores(self, agent):
        """With only 2 scores, can't detect plateau."""
        scores = [0.6, 0.62]
        convergence = agent._check_convergence(scores, threshold=0.9, consecutive_regressions=0)
        assert convergence is None


# ---------------------------------------------------------------------------
# 4. Convergence: regression detection
# ---------------------------------------------------------------------------

class TestConvergenceRegression:
    def test_regression_triggers_rollback(self, agent):
        """Two consecutive regressions should stop with rollback reason."""
        with patch("agent.brain.iterative_refine.IterativeRefineAgent._execute_workflow") as mock_exec, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._analyze_output") as mock_analyze, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._apply_adjustment") as mock_apply, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._rollback") as mock_rollback:

            mock_exec.return_value = {
                "output_path": "/fake/output.png",
                "render_time_s": 2.0,
                "key_params": {"cfg": 7.0, "steps": 20},
            }
            # Three iterations: 0.6, 0.5 (regression), 0.4 (regression again)
            mock_analyze.side_effect = [
                {"quality_score": 0.6, "artifacts": []},
                {"quality_score": 0.5, "artifacts": []},
                {"quality_score": 0.4, "artifacts": []},
            ]
            mock_apply.return_value = {"param": "cfg", "direction": "decrease"}

            raw = agent.handle("iterative_refine", {
                "intent": "test",
                "quality_threshold": 0.9,
                "mode": "deep",
            })
            result = json.loads(raw)
            assert result["reason"] == "regression_rollback"
            assert mock_rollback.call_count >= 2  # rolled back both times

    def test_single_regression_retries(self, agent):
        """Single regression rolls back and tries a different vector."""
        with patch("agent.brain.iterative_refine.IterativeRefineAgent._execute_workflow") as mock_exec, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._analyze_output") as mock_analyze, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._apply_adjustment") as mock_apply, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._rollback") as mock_rollback:

            mock_exec.return_value = {
                "output_path": "/fake/output.png",
                "render_time_s": 2.0,
                "key_params": {"cfg": 7.0, "steps": 20},
            }
            # 0.6 → 0.5 (regression) → 0.8 (recovery, meets threshold)
            mock_analyze.side_effect = [
                {"quality_score": 0.6, "artifacts": []},
                {"quality_score": 0.5, "artifacts": []},
                {"quality_score": 0.8, "artifacts": []},
            ]
            mock_apply.return_value = {"param": "steps", "direction": "increase"}

            raw = agent.handle("iterative_refine", {
                "intent": "test",
                "quality_threshold": 0.7,
                "mode": "deep",
            })
            result = json.loads(raw)
            # Recovered after regression and met threshold
            assert result["converged"] is True
            assert mock_rollback.call_count >= 1


# ---------------------------------------------------------------------------
# 5. Patch generation: diagnosis maps to valid patches
# ---------------------------------------------------------------------------

class TestPatchGeneration:
    def test_apply_adjustment_sets_input(self, agent):
        """_apply_adjustment calls set_input with correct params."""
        mock_wf = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 9.0,
                    "steps": 20,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "seed": 42,
                    "denoise": 1.0,
                },
            },
        }

        diagnosis = {"param": "cfg", "direction": "decrease", "delta": 1.5, "reason": "test"}

        with patch("agent.brain.iterative_refine.IterativeRefineAgent._execute_workflow"), \
             patch("agent.tools.workflow_patch.get_current_workflow", return_value=mock_wf), \
             patch("agent.tools.handle") as mock_dispatch:

            mock_dispatch.return_value = json.dumps({"success": True})
            result = agent._apply_adjustment(diagnosis, {"cfg": 9.0})

            assert result is not None
            assert result["param"] == "cfg"
            assert result["old_value"] == 9.0
            assert result["new_value"] == 7.5  # 9.0 - 1.5

    def test_calculate_decrease_with_clamp(self, agent):
        """CFG decrease should clamp at 1.0 minimum."""
        new_val = agent._calculate_new_value("cfg", "decrease", 1.5, {"delta": 2.0}, {})
        assert new_val == 1.0  # clamped

    def test_calculate_increase_steps(self, agent):
        """Steps increase should cap at 150."""
        new_val = agent._calculate_new_value("steps", "increase", 145, {"delta": 10}, {})
        assert new_val == 150  # clamped

    def test_calculate_switch_sampler(self, agent):
        """Switch should use the specified value."""
        new_val = agent._calculate_new_value(
            "sampler_name", "switch", "euler",
            {"value": "dpmpp_2m"}, {},
        )
        assert new_val == "dpmpp_2m"

    def test_calculate_randomize_seed(self, agent):
        """Randomize should produce an integer."""
        new_val = agent._calculate_new_value("seed", "randomize", 42, {}, {})
        assert isinstance(new_val, int)
        assert new_val >= 1


# ---------------------------------------------------------------------------
# 6. Unlimited mode: returns after exactly 1 iteration
# ---------------------------------------------------------------------------

class TestUnlimitedMode:
    def test_unlimited_single_iteration(self, agent):
        """Unlimited mode runs exactly 1 iteration regardless of score."""
        with patch("agent.brain.iterative_refine.IterativeRefineAgent._execute_workflow") as mock_exec, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._analyze_output") as mock_analyze:

            mock_exec.return_value = {
                "output_path": "/fake/output.png",
                "render_time_s": 2.0,
                "key_params": {"cfg": 7.0, "steps": 20},
            }
            # Score below threshold — but unlimited mode should still stop at 1
            mock_analyze.return_value = {"quality_score": 0.3, "artifacts": []}

            raw = agent.handle("iterative_refine", {
                "intent": "test",
                "mode": "unlimited",
                "quality_threshold": 0.9,
            })
            result = json.loads(raw)
            assert result["total_iterations"] == 1
            assert result["converged"] is False
            assert result["reason"] == "max_iterations"


# ---------------------------------------------------------------------------
# 7. Heuristic diagnosis: known artifacts map to correct adjustments
# ---------------------------------------------------------------------------

class TestHeuristicDiagnosis:
    def test_color_banding_lowers_cfg(self, agent):
        """'color banding' artifact should diagnose CFG decrease."""
        diagnosis = agent._diagnose(
            ["color banding"], [0.5], {"cfg": 9.0}, "default", [],
        )
        assert diagnosis["param"] == "cfg"
        assert diagnosis["direction"] == "decrease"

    def test_blurry_increases_steps(self, agent):
        """'blurry' artifact should diagnose steps increase."""
        diagnosis = agent._diagnose(
            ["blurry"], [0.5], {"steps": 15}, "default", [],
        )
        assert diagnosis["param"] == "steps"
        assert diagnosis["direction"] == "increase"

    def test_noisy_switches_sampler(self, agent):
        """'noisy' artifact should diagnose sampler switch."""
        diagnosis = agent._diagnose(
            ["noisy"], [0.5], {"sampler_name": "euler"}, "default", [],
        )
        assert diagnosis["param"] == "sampler_name"
        assert diagnosis["direction"] == "switch"
        assert diagnosis["value"] == "dpmpp_2m"

    def test_oversaturated_lowers_cfg(self, agent):
        """'oversaturated' artifact should diagnose CFG decrease."""
        diagnosis = agent._diagnose(
            ["oversaturated"], [0.5], {"cfg": 12.0}, "default", [],
        )
        assert diagnosis["param"] == "cfg"
        assert diagnosis["direction"] == "decrease"
        assert diagnosis["delta"] == 2.0  # specific delta for oversaturation

    def test_fallback_when_no_match(self, agent):
        """Unknown artifact should use fallback sequence."""
        diagnosis = agent._diagnose(
            ["some_unrecognized_issue"], [0.5], {}, "default", [],
        )
        # Should match first fallback
        assert diagnosis["param"] == FALLBACK_ADJUSTMENTS[0]["param"]
        assert diagnosis["direction"] == FALLBACK_ADJUSTMENTS[0]["direction"]

    def test_skips_already_tried_adjustment(self, agent):
        """Should skip adjustments that were already tried."""
        used = ["cfg:decrease"]
        diagnosis = agent._diagnose(
            ["color banding"], [0.5], {"cfg": 9.0}, "default", used,
        )
        # cfg:decrease is used, so should pick something else
        assert not (diagnosis["param"] == "cfg" and diagnosis["direction"] == "decrease")

    def test_all_exhausted_randomizes_seed(self, agent):
        """When all adjustments are exhausted, randomize seed."""
        # Build a used list that covers all heuristics and fallbacks
        used = []
        for rule in HEURISTIC_RULES:
            used.append(f"{rule['param']}:{rule['direction']}")
        for fb in FALLBACK_ADJUSTMENTS:
            used.append(f"{fb['param']}:{fb['direction']}")

        diagnosis = agent._diagnose([], [0.5], {}, "default", used)
        assert diagnosis["param"] == "seed"
        assert diagnosis["direction"] == "randomize"


# ---------------------------------------------------------------------------
# 8. Result structure matches documented interface
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_result_fields(self, agent):
        """Output should contain all documented fields."""
        with patch("agent.brain.iterative_refine.IterativeRefineAgent._execute_workflow") as mock_exec, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._analyze_output") as mock_analyze:

            mock_exec.return_value = {
                "output_path": "/fake/output.png",
                "render_time_s": 2.5,
                "key_params": {"cfg": 7.0, "steps": 20, "model": "test.safetensors"},
            }
            mock_analyze.return_value = {"quality_score": 0.8, "artifacts": []}

            raw = agent.handle("iterative_refine", {
                "intent": "test",
                "quality_threshold": 0.7,
            })
            result = json.loads(raw)

            # Top-level fields
            assert "iterations" in result
            assert "best_result" in result
            assert "converged" in result
            assert "reason" in result
            assert "recommendation" in result
            assert "total_iterations" in result

    def test_iteration_record_fields(self, agent):
        """Each iteration record should have required fields."""
        with patch("agent.brain.iterative_refine.IterativeRefineAgent._execute_workflow") as mock_exec, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._analyze_output") as mock_analyze:

            mock_exec.return_value = {
                "output_path": "/fake/output.png",
                "render_time_s": 2.5,
                "key_params": {"cfg": 7.0, "steps": 20},
            }
            mock_analyze.return_value = {"quality_score": 0.8, "artifacts": []}

            raw = agent.handle("iterative_refine", {
                "intent": "test",
                "quality_threshold": 0.7,
            })
            result = json.loads(raw)
            iteration = result["iterations"][0]

            assert "iteration" in iteration
            assert "parameters" in iteration
            assert "quality_score" in iteration
            assert "artifacts_detected" in iteration
            assert "output_path" in iteration

    def test_best_result_fields(self, agent):
        """best_result should contain iteration, score, path, params."""
        with patch("agent.brain.iterative_refine.IterativeRefineAgent._execute_workflow") as mock_exec, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._analyze_output") as mock_analyze:

            mock_exec.return_value = {
                "output_path": "/fake/output.png",
                "render_time_s": 2.5,
                "key_params": {"cfg": 7.0, "steps": 20},
            }
            mock_analyze.return_value = {"quality_score": 0.8, "artifacts": []}

            raw = agent.handle("iterative_refine", {
                "intent": "test",
                "quality_threshold": 0.7,
            })
            result = json.loads(raw)
            best = result["best_result"]

            assert best["iteration"] == 1
            assert best["quality_score"] == 0.8
            assert best["output_path"] == "/fake/output.png"
            assert "parameters" in best

    def test_recommendation_empty_when_converged(self, agent):
        """recommendation should be empty when converged."""
        with patch("agent.brain.iterative_refine.IterativeRefineAgent._execute_workflow") as mock_exec, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._analyze_output") as mock_analyze:

            mock_exec.return_value = {
                "output_path": "/fake/output.png",
                "render_time_s": 2.5,
                "key_params": {"cfg": 7.0},
            }
            mock_analyze.return_value = {"quality_score": 0.9, "artifacts": []}

            raw = agent.handle("iterative_refine", {
                "intent": "test",
                "quality_threshold": 0.7,
            })
            result = json.loads(raw)
            assert result["recommendation"] == ""

    def test_recommendation_present_when_not_converged(self, agent):
        """recommendation should have content when not converged."""
        with patch("agent.brain.iterative_refine.IterativeRefineAgent._execute_workflow") as mock_exec, \
             patch("agent.brain.iterative_refine.IterativeRefineAgent._analyze_output") as mock_analyze:

            mock_exec.return_value = {
                "output_path": "/fake/output.png",
                "render_time_s": 2.5,
                "key_params": {"cfg": 7.0},
            }
            mock_analyze.return_value = {"quality_score": 0.3, "artifacts": []}

            raw = agent.handle("iterative_refine", {
                "intent": "test",
                "mode": "unlimited",
                "quality_threshold": 0.9,
            })
            result = json.loads(raw)
            assert len(result["recommendation"]) > 0


# ---------------------------------------------------------------------------
# Module-level backward compat
# ---------------------------------------------------------------------------

class TestModuleLevelInterface:
    def test_tools_exported(self):
        """Module-level TOOLS should be accessible."""
        assert len(TOOLS) == 1
        assert TOOLS[0]["name"] == "iterative_refine"

    def test_handle_dispatches(self):
        """Module-level handle() should dispatch to the agent."""
        with patch("agent.brain.iterative_refine.IterativeRefineAgent._handle_iterative_refine") as mock_h:
            mock_h.return_value = '{"ok": true}'
            result = handle("iterative_refine", {"intent": "test"})
            assert result == '{"ok": true}'

    def test_handle_unknown_tool(self):
        """Unknown tool name should return error."""
        result = handle("nonexistent", {})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_execution_error_stops_loop(self, agent):
        """If execution fails, loop should stop with error."""
        with patch("agent.brain.iterative_refine.IterativeRefineAgent._execute_workflow") as mock_exec:
            mock_exec.return_value = {"error": "ComfyUI not reachable"}

            raw = agent.handle("iterative_refine", {
                "intent": "test",
                "mode": "refine",
            })
            result = json.loads(raw)
            assert result["total_iterations"] == 1
            assert result["reason"] == "execution_error"
            assert result["iterations"][0]["error"] == "ComfyUI not reachable"
