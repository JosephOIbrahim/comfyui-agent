"""Tests for agent/brain/iteration_accumulator.py — Cycle 57 coverage.

IterationAccumulatorAgent: per-session isolation, required field guards,
step FIFO eviction cap, finalize behavior, and thread safety.
"""

from __future__ import annotations

import json
import threading

import pytest

from agent.brain.iteration_accumulator import (
    IterationAccumulatorAgent,
    TOOLS,
    _MAX_STEPS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    """Fresh IterationAccumulatorAgent for each test."""
    return IterationAccumulatorAgent()


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_tools_exported(self):
        names = {t["name"] for t in TOOLS}
        assert "start_iteration_tracking" in names
        assert "record_iteration_step" in names
        assert "finalize_iterations" in names

    def test_start_required_fields(self):
        schema = next(t for t in TOOLS if t["name"] == "start_iteration_tracking")
        assert "intent_summary" in schema["input_schema"]["required"]

    def test_record_required_fields(self):
        schema = next(t for t in TOOLS if t["name"] == "record_iteration_step")
        required = schema["input_schema"]["required"]
        assert "iteration" in required
        assert "type" in required
        assert "trigger" in required

    def test_finalize_required_fields(self):
        schema = next(t for t in TOOLS if t["name"] == "finalize_iterations")
        assert "accepted_iteration" in schema["input_schema"]["required"]


# ---------------------------------------------------------------------------
# Required field guards
# ---------------------------------------------------------------------------

class TestRequiredFieldGuards:
    def test_missing_intent_summary_returns_error(self, agent):
        result = json.loads(agent.handle("start_iteration_tracking", {}))
        assert "error" in result
        assert "intent_summary" in result["error"]

    def test_empty_intent_summary_returns_error(self, agent):
        result = json.loads(agent.handle("start_iteration_tracking", {"intent_summary": ""}))
        assert "error" in result

    def test_missing_iteration_returns_error(self, agent):
        result = json.loads(agent.handle("record_iteration_step", {
            "type": "initial", "trigger": "user asked",
        }))
        assert "error" in result
        assert "iteration" in result["error"]

    def test_missing_type_returns_error(self, agent):
        result = json.loads(agent.handle("record_iteration_step", {
            "iteration": 1, "trigger": "user asked",
        }))
        assert "error" in result
        assert "type" in result["error"]

    def test_missing_trigger_returns_error(self, agent):
        result = json.loads(agent.handle("record_iteration_step", {
            "iteration": 1, "type": "initial",
        }))
        assert "error" in result
        assert "trigger" in result["error"]

    def test_missing_accepted_iteration_returns_error(self, agent):
        result = json.loads(agent.handle("finalize_iterations", {}))
        assert "error" in result
        assert "accepted_iteration" in result["error"]


# ---------------------------------------------------------------------------
# start → record → finalize happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    SESSION = "test_happy"

    def _call(self, agent, name, extra=None):
        payload = {"session": self.SESSION, **(extra or {})}
        return json.loads(agent.handle(name, payload))

    def test_start_returns_tracking_status(self, agent):
        r = self._call(agent, "start_iteration_tracking", {"intent_summary": "make it pop"})
        assert r["status"] == "tracking"
        assert r["intent_summary"] == "make it pop"

    def test_record_after_start_succeeds(self, agent):
        self._call(agent, "start_iteration_tracking", {"intent_summary": "test"})
        r = self._call(agent, "record_iteration_step", {
            "iteration": 1, "type": "initial", "trigger": "user request",
        })
        assert r["status"] == "recorded"
        assert r["iteration"] == 1
        assert r["total_steps"] == 1

    def test_finalize_returns_full_history(self, agent):
        self._call(agent, "start_iteration_tracking", {"intent_summary": "test"})
        self._call(agent, "record_iteration_step", {
            "iteration": 1, "type": "initial", "trigger": "start",
        })
        self._call(agent, "record_iteration_step", {
            "iteration": 2, "type": "refinement", "trigger": "warmer tones",
        })
        r = self._call(agent, "finalize_iterations", {"accepted_iteration": 2})

        assert r["intent_summary"] == "test"
        assert r["accepted_iteration"] == 2
        assert r["total_steps"] == 2
        assert len(r["iterations"]) == 2

    def test_second_start_resets_state(self, agent):
        self._call(agent, "start_iteration_tracking", {"intent_summary": "first"})
        self._call(agent, "record_iteration_step", {
            "iteration": 1, "type": "initial", "trigger": "start",
        })
        # Second start should clear previous steps
        self._call(agent, "start_iteration_tracking", {"intent_summary": "second"})
        steps = agent.get_steps(self.SESSION)
        assert steps == []

    def test_all_step_fields_stored(self, agent):
        self._call(agent, "start_iteration_tracking", {"intent_summary": "fields test"})
        self._call(agent, "record_iteration_step", {
            "iteration": 1,
            "type": "refinement",
            "trigger": "user asked for warmer",
            "patches": [{"op": "replace", "path": "/1/inputs/cfg", "value": 5}],
            "params": {"cfg": 5},
            "feedback": "looks better",
            "observation": "warmer now",
        })
        steps = agent.get_steps(self.SESSION)
        assert len(steps) == 1
        s = steps[0]
        assert s["trigger"] == "user asked for warmer"
        assert s["feedback"] == "looks better"
        assert s["observation"] == "warmer now"
        assert s["params"] == {"cfg": 5}

    def test_record_before_start_returns_error(self, agent):
        """record_step without start_iteration_tracking must return error."""
        r = json.loads(agent.handle("record_iteration_step", {
            "session": "fresh_session_c57",
            "iteration": 1, "type": "initial", "trigger": "oops",
        }))
        assert "error" in r

    def test_finalize_before_steps_returns_error(self, agent):
        """finalize without any recorded steps must return error."""
        self._call(agent, "start_iteration_tracking", {"intent_summary": "empty"})
        r = self._call(agent, "finalize_iterations", {"accepted_iteration": 1})
        assert "error" in r


# ---------------------------------------------------------------------------
# FIFO eviction cap
# ---------------------------------------------------------------------------

class TestStepCap:
    SESSION = "cap_test"

    def test_steps_capped_at_max(self, agent):
        """Recording more than _MAX_STEPS must trim oldest steps."""
        agent.handle("start_iteration_tracking", {
            "session": self.SESSION, "intent_summary": "cap test",
        })
        for i in range(_MAX_STEPS + 10):
            agent.handle("record_iteration_step", {
                "session": self.SESSION,
                "iteration": i + 1, "type": "refinement", "trigger": f"step {i}",
            })

        steps = agent.get_steps(self.SESSION)
        assert len(steps) == _MAX_STEPS

    def test_cap_retains_newest_steps(self, agent):
        """After eviction, the retained steps are the most recent ones."""
        agent.handle("start_iteration_tracking", {
            "session": self.SESSION, "intent_summary": "cap retain",
        })
        for i in range(_MAX_STEPS + 5):
            agent.handle("record_iteration_step", {
                "session": self.SESSION,
                "iteration": i + 1, "type": "initial", "trigger": f"t{i}",
            })

        steps = agent.get_steps(self.SESSION)
        # First step retained should be index 5 (oldest 5 evicted)
        assert steps[0]["trigger"] == "t5"
        assert steps[-1]["trigger"] == f"t{_MAX_STEPS + 4}"


# ---------------------------------------------------------------------------
# Per-session isolation
# ---------------------------------------------------------------------------

class TestSessionIsolation:
    def test_two_sessions_independent(self, agent):
        for sess in ("alpha", "beta"):
            agent.handle("start_iteration_tracking", {
                "session": sess, "intent_summary": f"goal {sess}",
            })
            agent.handle("record_iteration_step", {
                "session": sess, "iteration": 1, "type": "initial", "trigger": sess,
            })

        alpha_steps = agent.get_steps("alpha")
        beta_steps = agent.get_steps("beta")

        assert len(alpha_steps) == 1
        assert len(beta_steps) == 1
        assert alpha_steps[0]["trigger"] == "alpha"
        assert beta_steps[0]["trigger"] == "beta"

    def test_session_default_fallback(self, agent):
        """Invalid session string falls back to 'default'."""
        r = json.loads(agent.handle("start_iteration_tracking", {
            "session": "",
            "intent_summary": "fallback test",
        }))
        assert r["session"] == "default"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_sessions_no_exception(self, agent):
        """10 sessions recording steps simultaneously must not crash or corrupt."""
        errors = []

        def _work(i):
            try:
                sess = f"thread_{i}"
                agent.handle("start_iteration_tracking", {
                    "session": sess, "intent_summary": f"goal {i}",
                })
                for j in range(5):
                    agent.handle("record_iteration_step", {
                        "session": sess, "iteration": j + 1,
                        "type": "initial", "trigger": f"t{j}",
                    })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_work, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_each_session_has_correct_step_count(self, agent):
        """After concurrent recording, each session must have exactly its own steps."""
        errors = []

        def _work(i):
            try:
                sess = f"count_{i}"
                agent.handle("start_iteration_tracking", {
                    "session": sess, "intent_summary": "count",
                })
                for j in range(3):
                    agent.handle("record_iteration_step", {
                        "session": sess, "iteration": j + 1,
                        "type": "initial", "trigger": f"s{j}",
                    })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_work, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        for i in range(5):
            steps = agent.get_steps(f"count_{i}")
            assert len(steps) == 3, f"Session count_{i} has {len(steps)} steps, expected 3"


# ---------------------------------------------------------------------------
# Cycle 63: _get_session_lock — WeakValueDictionary prevents eviction race
# ---------------------------------------------------------------------------

class TestIterationLockWeakRef:
    """_get_session_lock() must return same lock while caller holds reference (Cycle 63)."""

    def test_same_session_same_lock(self):
        """Two calls for the same session return the SAME lock object."""
        from agent.brain.iteration_accumulator import IterationAccumulatorAgent

        agent = IterationAccumulatorAgent()
        lock_a = agent._get_session_lock("iter-same-63")
        lock_b = agent._get_session_lock("iter-same-63")
        assert lock_a is lock_b, "Same session must return the same lock object"

    def test_concurrent_same_session_same_lock(self):
        """Concurrent _get_session_lock() for the same session yields the same object."""
        import threading
        from agent.brain.iteration_accumulator import IterationAccumulatorAgent

        agent = IterationAccumulatorAgent()
        results = []

        def grab():
            results.append(agent._get_session_lock("iter-concurrent-63"))

        threads = [threading.Thread(target=grab) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(id(lk) for lk in results)) == 1, \
            "All concurrent callers must receive the same lock object"
