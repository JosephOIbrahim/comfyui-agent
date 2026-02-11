"""Integration tests — verify multi-tool workflows chain correctly.

These tests simulate end-to-end scenarios where multiple brain and
intelligence layer tools are chained in realistic sequences.
"""

import json

import pytest

from agent.brain import planner, memory, optimizer, demo
from agent.tools import workflow_patch, model_compat
from agent.config import SESSIONS_DIR


@pytest.fixture(autouse=True)
def clean_sessions():
    """Clean up test session files."""
    yield
    for f in SESSIONS_DIR.glob("integ_*"):
        f.unlink(missing_ok=True)


@pytest.fixture
def loaded_workflow(tmp_path):
    """Load a sample workflow into workflow_patch state."""
    wf = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sdxl_base_1.0.safetensors"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "a beautiful landscape", "clip": ["1", 1]},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["4", 0],
                "seed": 42,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "ugly, blurry", "clip": ["1", 1]},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "test"},
        },
    }
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(wf), encoding="utf-8")
    workflow_patch.handle("apply_workflow_patch", {
        "path": str(path),
        "patches": [],
    })
    yield wf
    workflow_patch._state["current_workflow"] = None


class TestPlanExecuteLearnLoop:
    """Test the full plan -> execute -> record -> recommend cycle."""

    def test_plan_then_record_outcomes(self):
        # 1. Plan a goal
        plan = json.loads(planner.handle("plan_goal", {
            "session": "integ_plan",
            "goal": "build an SDXL txt2img workflow",
        }))
        assert plan["pattern"] == "build_workflow"
        assert len(plan["steps"]) >= 3

        # 2. Complete steps
        for step in plan["steps"]:
            planner.handle("complete_step", {
                "session": "integ_plan",
                "step_id": step["id"],
                "result": f"Completed {step['id']}",
            })

        # 3. Record outcome
        outcome = json.loads(memory.handle("record_outcome", {
            "session": "integ_plan",
            "key_params": {"model": "sdxl_base", "steps": 20, "cfg": 7.0},
            "model_combo": ["sdxl_base"],
            "quality_score": 0.85,
            "render_time_s": 12.0,
            "user_feedback": "positive",
        }))
        assert outcome["recorded"] is True

        # 4. Get recommendations (should have data now)
        recs = json.loads(memory.handle("get_recommendations", {
            "session": "integ_plan",
            "current_model": "sdxl_base",
            "current_params": {"steps": 20, "cfg": 7.0},
        }))
        assert recs["based_on"] == 1

    def test_multiple_outcomes_improve_recommendations(self):
        # Record multiple outcomes with varying quality
        configs = [
            ({"model": "sdxl", "steps": 30, "sampler": "dpm++2m"}, 0.9, ["sdxl"]),
            ({"model": "sdxl", "steps": 20, "sampler": "euler"}, 0.7, ["sdxl"]),
            ({"model": "sdxl", "steps": 30, "sampler": "dpm++2m"}, 0.88, ["sdxl"]),
            ({"model": "sdxl", "steps": 10, "sampler": "euler"}, 0.4, ["sdxl"]),
            ({"model": "sdxl", "steps": 30, "sampler": "dpm++2m"}, 0.92, ["sdxl"]),
        ]
        for params, score, combo in configs:
            memory.handle("record_outcome", {
                "session": "integ_multi",
                "key_params": params,
                "model_combo": combo,
                "quality_score": score,
            })

        # Ask for recommendations with suboptimal config
        recs = json.loads(memory.handle("get_recommendations", {
            "session": "integ_multi",
            "current_model": "sdxl",
            "current_params": {"steps": 10, "sampler": "euler", "model": "sdxl"},
        }))
        assert recs["based_on"] == 5
        assert len(recs["recommendations"]) > 0

        # Should also have patterns
        patterns = json.loads(memory.handle("get_learned_patterns", {
            "session": "integ_multi",
            "query": "all",
        }))
        assert patterns["outcomes_count"] == 5


class TestOptimizationWorkflow:
    """Test optimize -> profile -> apply -> compare cycle."""

    def test_profile_then_optimize(self, loaded_workflow):
        # 1. Profile the workflow
        profile = json.loads(optimizer.handle("profile_workflow", {}))
        assert "workflow" in profile

        # 2. Get optimization suggestions
        suggestions = json.loads(optimizer.handle("suggest_optimizations", {}))
        assert "optimizations" in suggestions

        # 3. Apply an optimization (vae_tiling)
        result = json.loads(optimizer.handle("apply_optimization", {
            "optimization_id": "vae_tiling",
        }))
        assert "applied" in result


class TestCompatibilityCheck:
    """Test model compatibility in workflow context."""

    def test_workflow_compat_check(self, loaded_workflow):
        # Check the loaded workflow's models
        result = json.loads(model_compat.handle("check_model_compatibility", {
            "models": ["sdxl_base_1.0.safetensors"],
        }))
        assert result["compatible"] is True
        assert result["family"] == "sdxl"

    def test_detect_mismatch_before_swap(self):
        # Before swapping, check if new model is compatible
        result = json.loads(model_compat.handle("check_model_compatibility", {
            "models": [
                "sdxl_base_1.0.safetensors",
                "control_v11p_sd15_depth.pth",
            ],
        }))
        assert result["compatible"] is False


class TestDemoScenario:
    """Test a complete demo scenario flow."""

    @pytest.fixture(autouse=True)
    def reset_demo(self):
        demo._demo_state.update({
            "active": False, "scenario": None,
            "current_step_idx": 0, "started_at": None, "checkpoints": [],
        })
        yield

    def test_full_model_swap_demo(self):
        # Start demo
        start = json.loads(demo.handle("start_demo", {"scenario": "model_swap"}))
        assert start["demo_started"] is True
        assert start["total_steps"] == 4

        # Walk through all steps
        steps = ["analyze", "find_upgrade", "apply_swap", "compare"]
        for step_id in steps:
            result = json.loads(demo.handle("demo_checkpoint", {
                "step_completed": step_id,
                "notes": f"Did {step_id}",
            }))

        assert result["demo_complete"] is True
        assert result["steps_completed"] == 4


class TestReplanAfterFailure:
    """Test replanning when a step fails."""

    def test_replan_preserves_progress(self):
        # Create initial plan
        planner.handle("plan_goal", {
            "session": "integ_replan",
            "goal": "optimize workflow for speed",
        })

        # Complete first step
        plan = json.loads(planner.handle("get_plan", {"session": "integ_replan"}))
        first_step = plan["steps"][0]["id"]
        planner.handle("complete_step", {
            "session": "integ_replan",
            "step_id": first_step,
            "result": "Done",
        })

        # Replan remaining steps
        replan = json.loads(planner.handle("replan", {
            "session": "integ_replan",
            "reason": "Found that TensorRT is already installed",
            "new_remaining_steps": [
                {"id": "convert_engine", "action": "Convert model to TRT engine"},
                {"id": "benchmark", "action": "Run A/B benchmark"},
            ],
        }))

        assert replan["completed_preserved"] == 1
        assert replan["new_steps"] == 2

        # Original completed step should still be complete
        plan = json.loads(planner.handle("get_plan", {"session": "integ_replan"}))
        completed = [s for s in plan["steps"] if s["status"] == "done"]
        assert len(completed) == 1
