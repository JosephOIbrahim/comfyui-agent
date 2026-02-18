"""Tests for brain/planner.py — goal decomposition and progress tracking."""

import json

import pytest

from agent.brain import planner
from agent.config import SESSIONS_DIR


@pytest.fixture(autouse=True)
def clean_session_goals():
    """Remove test goal files after each test."""
    yield
    for f in SESSIONS_DIR.glob("test_*_goals.json"):
        f.unlink(missing_ok=True)


class TestPlanGoal:
    def test_build_workflow_pattern(self):
        result = json.loads(planner.handle("plan_goal", {
            "goal": "Build me a Flux portrait pipeline",
            "session": "test_planner",
        }))
        assert result["planned"] is True
        assert result["pattern"] == "build_workflow"
        assert result["total_steps"] >= 5
        assert result["current_step"] is not None

    def test_optimize_pattern(self):
        result = json.loads(planner.handle("plan_goal", {
            "goal": "Speed up this workflow with TensorRT",
            "session": "test_optimize",
        }))
        assert result["pattern"] == "optimize_workflow"

    def test_debug_pattern(self):
        result = json.loads(planner.handle("plan_goal", {
            "goal": "Fix this broken workflow",
            "session": "test_debug",
        }))
        assert result["pattern"] == "debug_workflow"

    def test_swap_model_pattern(self):
        result = json.loads(planner.handle("plan_goal", {
            "goal": "Switch to SDXL lightning model",
            "session": "test_swap",
        }))
        assert result["pattern"] == "swap_model"

    def test_controlnet_pattern(self):
        result = json.loads(planner.handle("plan_goal", {
            "goal": "Add ControlNet depth guidance",
            "session": "test_cn",
        }))
        assert result["pattern"] == "add_controlnet"

    def test_chain_workflows_pattern(self):
        result = json.loads(planner.handle("plan_goal", {
            "goal": "Chain txt2img then upscale in a pipeline",
            "session": "test_chain",
        }))
        assert result["pattern"] == "chain_workflows"
        assert result["total_steps"] >= 4

    def test_pipeline_trigger(self):
        result = json.loads(planner.handle("plan_goal", {
            "goal": "Create a multi-stage pipeline for 3D generation",
            "session": "test_pipeline",
        }))
        assert result["pattern"] == "chain_workflows"

    def test_generate_3d_pattern(self):
        result = json.loads(planner.handle("plan_goal", {
            "goal": "Generate a 3D model of a stone column",
            "session": "test_3d",
        }))
        assert result["pattern"] == "generate_3d"
        assert result["total_steps"] >= 4

    def test_hunyuan3d_trigger(self):
        result = json.loads(planner.handle("plan_goal", {
            "goal": "Use hunyuan3d to create a mesh",
            "session": "test_hy3d",
        }))
        assert result["pattern"] == "generate_3d"

    def test_generate_audio_pattern(self):
        result = json.loads(planner.handle("plan_goal", {
            "goal": "Generate TTS narration for the scene",
            "session": "test_audio",
        }))
        assert result["pattern"] == "generate_audio"
        assert result["total_steps"] >= 4

    def test_cosyvoice_trigger(self):
        result = json.loads(planner.handle("plan_goal", {
            "goal": "Use cosyvoice to generate a voiceover",
            "session": "test_cosy",
        }))
        assert result["pattern"] == "generate_audio"

    def test_generic_fallback(self):
        result = json.loads(planner.handle("plan_goal", {
            "goal": "Do something unusual with the workflow",
            "session": "test_generic",
        }))
        assert result["pattern"] == "generic"
        assert result["total_steps"] == 4

    def test_first_step_is_active(self):
        result = json.loads(planner.handle("plan_goal", {
            "goal": "Build a workflow",
            "session": "test_active",
        }))
        steps = result["steps"]
        assert steps[0]["status"] == "active"
        assert all(s["status"] == "pending" for s in steps[1:])


class TestGetPlan:
    def test_no_plan(self):
        result = json.loads(planner.handle("get_plan", {"session": "test_noplan"}))
        assert "error" in result

    def test_get_existing_plan(self):
        planner.handle("plan_goal", {
            "goal": "Build a workflow",
            "session": "test_getplan",
        })
        result = json.loads(planner.handle("get_plan", {"session": "test_getplan"}))
        assert result["goal"] == "Build a workflow"
        assert result["progress"].startswith("0/")
        assert result["current_step"] is not None


class TestCompleteStep:
    def test_complete_advances(self):
        planner.handle("plan_goal", {
            "goal": "Build a workflow",
            "session": "test_complete",
        })
        plan = json.loads(planner.handle("get_plan", {"session": "test_complete"}))
        first_step = plan["current_step"]

        result = json.loads(planner.handle("complete_step", {
            "step_id": first_step,
            "result": "Found the model",
            "session": "test_complete",
        }))
        assert result["completed"] == first_step
        assert "next_step" in result
        assert result["next_step"] != first_step

    def test_complete_all_steps(self):
        planner.handle("plan_goal", {
            "goal": "Do something unusual",
            "session": "test_all",
        })
        plan = json.loads(planner.handle("get_plan", {"session": "test_all"}))
        for step in plan["steps"]:
            planner.handle("complete_step", {
                "step_id": step["id"],
                "result": "Done",
                "session": "test_all",
            })
        final = json.loads(planner.handle("get_plan", {"session": "test_all"}))
        assert final["status"] == "completed"

    def test_complete_nonexistent_step(self):
        planner.handle("plan_goal", {
            "goal": "Build a workflow",
            "session": "test_bad_step",
        })
        result = json.loads(planner.handle("complete_step", {
            "step_id": "nonexistent",
            "result": "Done",
            "session": "test_bad_step",
        }))
        assert "error" in result


class TestReplan:
    def test_replan_preserves_completed(self):
        planner.handle("plan_goal", {
            "goal": "Do something unusual",
            "session": "test_replan",
        })
        # Complete first step
        plan = json.loads(planner.handle("get_plan", {"session": "test_replan"}))
        planner.handle("complete_step", {
            "step_id": plan["steps"][0]["id"],
            "result": "Done",
            "session": "test_replan",
        })

        # Replan
        result = json.loads(planner.handle("replan", {
            "reason": "Changing approach",
            "new_remaining_steps": [
                {"id": "new_step_1", "action": "New approach step 1"},
                {"id": "new_step_2", "action": "New approach step 2"},
            ],
            "session": "test_replan",
        }))
        assert result["replanned"] is True
        assert result["completed_preserved"] == 1
        assert result["new_steps"] == 2

    def test_replan_no_plan(self):
        result = json.loads(planner.handle("replan", {
            "reason": "No plan exists",
            "session": "test_replan_none",
        }))
        assert "error" in result


class TestGoalId:
    def test_plan_goal_returns_goal_id(self):
        """plan_goal should generate and return a unique goal_id."""
        result = json.loads(planner.handle("plan_goal", {
            "goal": "Build a test workflow",
            "session": "test_goal_id",
        }))
        assert "goal_id" in result
        assert isinstance(result["goal_id"], str)
        assert len(result["goal_id"]) == 12  # make_id() returns 12-char hex

    def test_goal_id_persisted_to_disk(self):
        """goal_id should be saved in the plan file on disk."""
        planner.handle("plan_goal", {
            "goal": "Build a test workflow",
            "session": "test_goal_persist",
        })
        plan = planner._load_plan("test_goal_persist")
        assert "goal_id" in plan
        assert len(plan["goal_id"]) == 12

    def test_goal_id_in_get_plan_response(self):
        """get_plan should include the goal_id."""
        planner.handle("plan_goal", {
            "goal": "Debug a broken workflow",
            "session": "test_goal_get",
        })
        result = json.loads(planner.handle("get_plan", {"session": "test_goal_get"}))
        assert "goal_id" in result
        assert result["goal_id"] is not None

    def test_unique_goal_ids(self):
        """Each plan should get a unique goal_id."""
        r1 = json.loads(planner.handle("plan_goal", {
            "goal": "Goal one",
            "session": "test_goal_unique1",
        }))
        r2 = json.loads(planner.handle("plan_goal", {
            "goal": "Goal two",
            "session": "test_goal_unique2",
        }))
        assert r1["goal_id"] != r2["goal_id"]
