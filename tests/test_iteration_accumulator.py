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
def reset_singleton():
    """Reset registered IterationAccumulatorAgent state between tests.

    Cycle 52: state is now per-session dicts; reset by clearing _sessions.
    """
    from agent.brain._sdk import BrainAgent
    BrainAgent._register_all()
    instance = BrainAgent._registry.get("start_iteration_tracking")
    if instance is not None:
        with instance._sessions_mutex:
            instance._sessions.clear()
            instance._session_locks.clear()
    yield


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
        agent.record_step(1, "initial", "gen")
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


# ---------------------------------------------------------------------------
# Cycle 41: guards for record_step and finalize before start
# ---------------------------------------------------------------------------

class TestPreStartGuards:
    """Cycle 41: record_step and finalize must fail gracefully before start."""

    def test_record_step_before_start_returns_error(self):
        """record_iteration_step before start_iteration_tracking must return error JSON."""
        from agent.brain.iteration_accumulator import IterationAccumulatorAgent, BrainConfig
        agent = IterationAccumulatorAgent(BrainConfig())
        result = json.loads(agent.handle("record_iteration_step", {
            "iteration": 1,
            "type": "parameter_change",
            "trigger": "user",
        }))
        assert "error" in result
        assert "start" in result["error"].lower()

    def test_finalize_before_start_returns_error(self):
        """finalize_iterations before start_iteration_tracking must return error JSON."""
        from agent.brain.iteration_accumulator import IterationAccumulatorAgent, BrainConfig
        agent = IterationAccumulatorAgent(BrainConfig())
        result = json.loads(agent.handle("finalize_iterations", {
            "accepted_iteration": 1,
        }))
        assert "error" in result
        assert "start" in result["error"].lower()

    def test_finalize_with_no_steps_returns_error(self):
        """finalize_iterations with zero recorded steps must return error JSON."""
        from agent.brain.iteration_accumulator import IterationAccumulatorAgent, BrainConfig
        agent = IterationAccumulatorAgent(BrainConfig())
        agent.start(intent_summary="empty run")
        result = json.loads(agent.handle("finalize_iterations", {
            "accepted_iteration": 1,
        }))
        assert "error" in result
        assert "step" in result["error"].lower() or "record" in result["error"].lower()

    def test_normal_flow_still_works(self):
        """Start → record → finalize must still succeed after adding guards."""
        from agent.brain.iteration_accumulator import IterationAccumulatorAgent, BrainConfig
        agent = IterationAccumulatorAgent(BrainConfig())
        agent.start(intent_summary="normal run")
        rec = json.loads(agent.handle("record_iteration_step", {
            "iteration": 1,
            "type": "parameter_change",
            "trigger": "user",
        }))
        assert rec.get("status") == "recorded"
        fin = json.loads(agent.handle("finalize_iterations", {
            "accepted_iteration": 1,
        }))
        assert "error" not in fin
        assert fin["total_steps"] == 1

    def test_compositor_scenes_lock_exists(self):
        """compositor_tools must have a _scenes_lock threading.Lock.

        Iter 13 cycle 12 renamed _scene_lock → _scenes_lock when refactoring
        the module-global _current_scene into a per-session _scenes dict.
        """
        import threading
        from agent.stage import compositor_tools
        assert hasattr(compositor_tools, "_scenes_lock")
        assert isinstance(compositor_tools._scenes_lock, type(threading.Lock()))


# ---------------------------------------------------------------------------
# Cycle 47 — iteration_accumulator handle() required field guards
# ---------------------------------------------------------------------------

class TestIterationAccumulatorRequiredFields:
    """handle() must return structured errors when required fields are missing."""

    @pytest.fixture(autouse=True)
    def fresh_agent(self):
        from agent.brain.iteration_accumulator import IterationAccumulatorAgent
        self.agent = IterationAccumulatorAgent()

    def test_start_missing_intent_summary_returns_error(self):
        result = json.loads(self.agent.handle("start_iteration_tracking", {}))
        assert "error" in result
        assert "intent_summary" in result["error"].lower()

    def test_start_empty_intent_summary_returns_error(self):
        result = json.loads(self.agent.handle("start_iteration_tracking", {"intent_summary": ""}))
        assert "error" in result

    def test_start_none_intent_summary_returns_error(self):
        result = json.loads(self.agent.handle("start_iteration_tracking", {"intent_summary": None}))
        assert "error" in result

    def test_record_missing_iteration_returns_error(self):
        # Start first so we have context
        self.agent.handle("start_iteration_tracking", {"intent_summary": "test"})
        result = json.loads(self.agent.handle("record_iteration_step", {
            "type": "param_change",
            "trigger": "manual",
        }))
        assert "error" in result
        assert "iteration" in result["error"].lower()

    def test_record_missing_type_returns_error(self):
        self.agent.handle("start_iteration_tracking", {"intent_summary": "test"})
        result = json.loads(self.agent.handle("record_iteration_step", {
            "iteration": 1,
            "trigger": "manual",
        }))
        assert "error" in result
        assert "type" in result["error"].lower()

    def test_record_missing_trigger_returns_error(self):
        self.agent.handle("start_iteration_tracking", {"intent_summary": "test"})
        result = json.loads(self.agent.handle("record_iteration_step", {
            "iteration": 1,
            "type": "param_change",
        }))
        assert "error" in result
        assert "trigger" in result["error"].lower()

    def test_finalize_missing_accepted_iteration_returns_error(self):
        self.agent.handle("start_iteration_tracking", {"intent_summary": "test"})
        result = json.loads(self.agent.handle("finalize_iterations", {}))
        assert "error" in result
        assert "accepted_iteration" in result["error"].lower()

    def test_finalize_none_accepted_iteration_returns_error(self):
        self.agent.handle("start_iteration_tracking", {"intent_summary": "test"})
        result = json.loads(self.agent.handle("finalize_iterations", {"accepted_iteration": None}))
        assert "error" in result

    def test_valid_calls_not_blocked_by_guards(self):
        """Guards must not block well-formed calls."""
        start = json.loads(self.agent.handle("start_iteration_tracking", {
            "intent_summary": "make image dreamier",
        }))
        assert "error" not in start

        rec = json.loads(self.agent.handle("record_iteration_step", {
            "iteration": 1,
            "type": "param_change",
            "trigger": "manual",
        }))
        assert "error" not in rec


# ---------------------------------------------------------------------------
# Cycle 52 — per-session isolation (singleton-state bug fix)
# ---------------------------------------------------------------------------

class TestIterationAccumulatorSessionIsolation:
    """Two sessions must not interfere with each other's iteration state."""

    def _agent(self):
        from agent.brain.iteration_accumulator import IterationAccumulatorAgent
        return IterationAccumulatorAgent()

    def test_separate_sessions_isolated(self):
        """Starting session A must not clear session B's state."""
        agent = self._agent()
        # Start session A
        agent.start(intent_summary="Portrait pipeline", session="conn-A")
        agent.record_step(1, "initial", "user asked for portrait", session="conn-A")
        # Start session B — must not wipe A's steps
        agent.start(intent_summary="Landscape pipeline", session="conn-B")
        # A's steps must still be intact
        assert len(agent.get_steps(session="conn-A")) == 1

    def test_finalize_session_a_does_not_affect_b(self):
        """Finalizing session A must not touch session B."""
        agent = self._agent()
        agent.start(intent_summary="A goal", session="sess-A")
        agent.record_step(1, "initial", "trigger A", session="sess-A")
        agent.start(intent_summary="B goal", session="sess-B")
        agent.record_step(1, "initial", "trigger B", session="sess-B")
        # Finalize A
        result_a = agent.finalize(accepted_iteration=1, session="sess-A")
        assert "error" not in result_a
        # B must still be tracking
        assert agent.is_tracking(session="sess-B") is True

    def test_handle_routes_to_session(self):
        """handle() must pass session from tool_input to state isolation."""
        import json
        from agent.brain.iteration_accumulator import IterationAccumulatorAgent
        agent = IterationAccumulatorAgent()
        # Start two sessions via handle()
        r1 = json.loads(agent.handle("start_iteration_tracking", {
            "intent_summary": "Session alpha",
            "session": "alpha",
        }))
        r2 = json.loads(agent.handle("start_iteration_tracking", {
            "intent_summary": "Session beta",
            "session": "beta",
        }))
        assert r1["session"] == "alpha"
        assert r2["session"] == "beta"
        # Record into alpha
        json.loads(agent.handle("record_iteration_step", {
            "iteration": 1, "type": "initial", "trigger": "start",
            "session": "alpha",
        }))
        # Beta must have 0 steps
        assert len(agent.get_steps(session="beta")) == 0
        assert len(agent.get_steps(session="alpha")) == 1

    def test_default_session_backward_compat(self):
        """Callers that omit session must use 'default' and work correctly."""
        import json
        from agent.brain.iteration_accumulator import IterationAccumulatorAgent
        agent = IterationAccumulatorAgent()
        r = json.loads(agent.handle("start_iteration_tracking", {
            "intent_summary": "No session key",
        }))
        assert r.get("session") == "default"
        assert r.get("status") == "tracking"
