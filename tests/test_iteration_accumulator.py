"""Tests for the iteration_accumulator brain module."""

import json
import threading

import pytest

from agent.brain.iteration_accumulator import (
    IterationAccumulatorAgent,
    TOOLS,
)
from agent.brain._sdk import BrainConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    """Fresh IterationAccumulatorAgent for each test."""
    return IterationAccumulatorAgent(config=BrainConfig())


@pytest.fixture(autouse=True)
def reset_singleton(monkeypatch):
    """Reset module-level singleton between tests."""
    import agent.brain.iteration_accumulator as mod
    monkeypatch.setattr(mod, "_singleton", None)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

class TestToolSchemas:
    def test_tool_count(self):
        assert len(TOOLS) == 3

    def test_tool_names(self):
        names = {t["name"] for t in TOOLS}
        assert names == {
            "start_iteration_tracking",
            "record_iteration_step",
            "finalize_iterations",
        }


# ---------------------------------------------------------------------------
# Direct API
# ---------------------------------------------------------------------------

class TestDirectAPI:
    def test_start_tracking(self, agent):
        result = agent.start("make it dreamier")
        assert result["status"] == "tracking"
        assert result["intent_summary"] == "make it dreamier"
        assert agent.is_tracking()

    def test_record_step(self, agent):
        agent.start("test goal")
        result = agent.record_step(
            iteration=1,
            step_type="initial",
            trigger="first generation",
            params={"cfg": 7.0, "steps": 20},
        )
        assert result["status"] == "recorded"
        assert result["iteration"] == 1
        assert result["total_steps"] == 1

    def test_multiple_steps(self, agent):
        agent.start("test goal")
        agent.record_step(1, "initial", "first gen")
        agent.record_step(2, "refinement", "user asked for changes",
                          patches=[{"op": "replace", "path": "/2/inputs/cfg", "value": 5.0}])
        agent.record_step(3, "variation", "trying different seed")

        steps = agent.get_steps()
        assert len(steps) == 3
        assert steps[0]["type"] == "initial"
        assert steps[1]["type"] == "refinement"
        assert len(steps[1]["patches"]) == 1
        assert steps[2]["type"] == "variation"

    def test_finalize(self, agent):
        agent.start("test goal")
        agent.record_step(1, "initial", "first gen")
        agent.record_step(2, "refinement", "tweak")

        history = agent.finalize(accepted_iteration=2)
        assert history["intent_summary"] == "test goal"
        assert history["accepted_iteration"] == 2
        assert history["total_steps"] == 2
        assert len(history["iterations"]) == 2
        assert "started_at" in history
        assert "finalized_at" in history
        assert not agent.is_tracking()

    def test_start_clears_previous(self, agent):
        agent.start("first goal")
        agent.record_step(1, "initial", "gen 1")
        agent.start("second goal")

        assert agent.get_steps() == []
        assert agent.is_tracking()

    def test_not_tracking_before_start(self, agent):
        assert not agent.is_tracking()

    def test_default_patches_and_params(self, agent):
        agent.start("test")
        result = agent.record_step(1, "initial", "gen")
        steps = agent.get_steps()
        assert steps[0]["patches"] == []
        assert steps[0]["params"] == {}

    def test_feedback_and_observation(self, agent):
        agent.start("test")
        agent.record_step(
            1, "initial", "gen",
            feedback="looks great!",
            observation="sharp details, good composition",
        )
        steps = agent.get_steps()
        assert steps[0]["feedback"] == "looks great!"
        assert steps[0]["observation"] == "sharp details, good composition"


# ---------------------------------------------------------------------------
# Tool handle()
# ---------------------------------------------------------------------------

class TestToolHandle:
    def test_start_tool(self, agent):
        result = json.loads(agent.handle("start_iteration_tracking", {
            "intent_summary": "anime portrait refinement",
        }))
        assert result["status"] == "tracking"

    def test_record_step_tool(self, agent):
        agent.handle("start_iteration_tracking", {"intent_summary": "test"})
        result = json.loads(agent.handle("record_iteration_step", {
            "iteration": 1,
            "type": "initial",
            "trigger": "first generation",
            "params": {"cfg": 7.0},
        }))
        assert result["status"] == "recorded"
        assert result["iteration"] == 1

    def test_finalize_tool(self, agent):
        agent.handle("start_iteration_tracking", {"intent_summary": "test"})
        agent.handle("record_iteration_step", {
            "iteration": 1, "type": "initial", "trigger": "gen",
        })
        result = json.loads(agent.handle("finalize_iterations", {
            "accepted_iteration": 1,
        }))
        assert result["accepted_iteration"] == 1
        assert result["total_steps"] == 1

    def test_unknown_tool(self, agent):
        result = json.loads(agent.handle("nonexistent", {}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Thread Safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_record_steps(self, agent):
        """Multiple threads recording steps should not corrupt state."""
        agent.start("concurrent test")
        errors = []

        def record_n(thread_id):
            try:
                for i in range(10):
                    agent.record_step(
                        iteration=thread_id * 10 + i,
                        step_type="refinement",
                        trigger=f"thread-{thread_id}-step-{i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_n, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(agent.get_steps()) == 40


# ---------------------------------------------------------------------------
# Module-level dispatch
# ---------------------------------------------------------------------------

class TestModuleDispatch:
    def test_module_handle(self):
        from agent.brain.iteration_accumulator import handle
        result = json.loads(handle("start_iteration_tracking", {
            "intent_summary": "module-level test",
        }))
        assert result["status"] == "tracking"
