"""Tests for brain/demo.py — guided demo walkthroughs."""

import json

import pytest

from agent.brain import demo


@pytest.fixture(autouse=True)
def reset_demo_state():
    """Reset demo state between tests."""
    demo._demo_state.update({
        "active": False,
        "scenario": None,
        "current_step_idx": 0,
        "started_at": None,
        "checkpoints": [],
    })
    yield


class TestStartDemo:
    def test_list_scenarios(self):
        result = json.loads(demo.handle("start_demo", {"scenario": "list"}))
        assert result["count"] >= 4
        names = [s["name"] for s in result["available_scenarios"]]
        assert "model_swap" in names
        assert "speed_run" in names
        assert "controlnet_add" in names
        assert "full_pipeline" in names

    def test_start_model_swap(self):
        result = json.loads(demo.handle("start_demo", {"scenario": "model_swap"}))
        assert result["demo_started"] is True
        assert result["scenario"] == "model_swap"
        assert result["total_steps"] == 4
        assert result["first_step"]["id"] == "analyze"

    def test_start_speed_run(self):
        result = json.loads(demo.handle("start_demo", {"scenario": "speed_run"}))
        assert result["demo_started"] is True
        assert result["title"] == "Making It Fast"

    def test_start_unknown(self):
        result = json.loads(demo.handle("start_demo", {"scenario": "nonexistent"}))
        assert "error" in result
        assert "available" in result

    def test_demo_state_activated(self):
        demo.handle("start_demo", {"scenario": "model_swap"})
        assert demo._demo_state["active"] is True
        assert demo._demo_state["scenario"] == "model_swap"


class TestDemoCheckpoint:
    def test_checkpoint_no_demo(self):
        result = json.loads(demo.handle("demo_checkpoint", {"step_completed": "test"}))
        assert "error" in result

    def test_checkpoint_advances(self):
        demo.handle("start_demo", {"scenario": "model_swap"})
        result = json.loads(demo.handle("demo_checkpoint", {
            "step_completed": "analyze",
            "notes": "Found SD 1.5 workflow with 30 steps",
        }))
        assert result["checkpoint"] == "analyze"
        assert result["next_step"]["id"] == "find_upgrade"
        assert "1/4" in result["progress"]

    def test_checkpoint_completes_demo(self):
        demo.handle("start_demo", {"scenario": "model_swap"})
        # Complete all 4 steps
        steps = ["analyze", "find_upgrade", "apply_swap", "compare"]
        for i, step in enumerate(steps):
            result = json.loads(demo.handle("demo_checkpoint", {
                "step_completed": step,
            }))

        assert result["demo_complete"] is True
        assert result["steps_completed"] == 4
        assert "elapsed_human" in result

    def test_checkpoint_records_history(self):
        demo.handle("start_demo", {"scenario": "model_swap"})
        demo.handle("demo_checkpoint", {
            "step_completed": "analyze",
            "notes": "test note",
        })
        assert len(demo._demo_state["checkpoints"]) == 1
        assert demo._demo_state["checkpoints"][0]["notes"] == "test note"


class TestDemoScenarios:
    """Verify all scenarios have valid structure."""

    def test_all_scenarios_have_required_fields(self):
        for name, scenario in demo.DEMO_SCENARIOS.items():
            assert "title" in scenario, f"{name} missing title"
            assert "description" in scenario, f"{name} missing description"
            assert "steps" in scenario, f"{name} missing steps"
            assert len(scenario["steps"]) >= 3, f"{name} has too few steps"

    def test_all_steps_have_required_fields(self):
        for name, scenario in demo.DEMO_SCENARIOS.items():
            for step in scenario["steps"]:
                assert "id" in step, f"{name}/{step} missing id"
                assert "label" in step, f"{name}/{step} missing label"
                assert "narration" in step, f"{name}/{step} missing narration"
                assert "suggested_tools" in step, f"{name}/{step} missing suggested_tools"

    def test_scenario_durations_present(self):
        for name, scenario in demo.DEMO_SCENARIOS.items():
            assert "duration_estimate" in scenario, f"{name} missing duration_estimate"
