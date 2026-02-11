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
