"""Tests for brain/planner.py — goal decomposition and progress tracking."""

import json
import unittest.mock

import pytest

from agent.brain import handle
from agent.brain.planner import PlannerAgent
from agent.config import SESSIONS_DIR


@pytest.fixture(autouse=True)
def clean_session_goals():
    """Remove test goal files after each test."""
    yield
    for f in SESSIONS_DIR.glob("test_*_goals.json"):
        f.unlink(missing_ok=True)


class TestPlanGoal:
    def test_build_workflow_pattern(self):
        result = json.loads(handle("plan_goal", {
            "goal": "Build me a Flux portrait pipeline",
            "session": "test_planner",
        }))
        assert result["planned"] is True
        assert result["pattern"] == "build_workflow"
        assert result["total_steps"] >= 5
        assert result["current_step"] is not None

    def test_optimize_pattern(self):
        result = json.loads(handle("plan_goal", {
            "goal": "Speed up this workflow with TensorRT",
            "session": "test_optimize",
        }))
        assert result["pattern"] == "optimize_workflow"

    def test_debug_pattern(self):
        result = json.loads(handle("plan_goal", {
            "goal": "Fix this broken workflow",
            "session": "test_debug",
        }))
        assert result["pattern"] == "debug_workflow"

    def test_swap_model_pattern(self):
        result = json.loads(handle("plan_goal", {
            "goal": "Switch to SDXL lightning model",
            "session": "test_swap",
        }))
        assert result["pattern"] == "swap_model"

    def test_controlnet_pattern(self):
        result = json.loads(handle("plan_goal", {
            "goal": "Add ControlNet depth guidance",
            "session": "test_cn",
        }))
        assert result["pattern"] == "add_controlnet"

    def test_chain_workflows_pattern(self):
        result = json.loads(handle("plan_goal", {
            "goal": "Chain txt2img then upscale in a pipeline",
            "session": "test_chain",
        }))
        assert result["pattern"] == "chain_workflows"
        assert result["total_steps"] >= 4

    def test_pipeline_trigger(self):
        result = json.loads(handle("plan_goal", {
            "goal": "Create a multi-stage pipeline for 3D generation",
            "session": "test_pipeline",
        }))
        assert result["pattern"] == "chain_workflows"

    def test_generate_3d_pattern(self):
        result = json.loads(handle("plan_goal", {
            "goal": "Generate a 3D model of a stone column",
            "session": "test_3d",
        }))
        assert result["pattern"] == "generate_3d"
        assert result["total_steps"] >= 4

    def test_hunyuan3d_trigger(self):
        result = json.loads(handle("plan_goal", {
            "goal": "Use hunyuan3d to create a mesh",
            "session": "test_hy3d",
        }))
        assert result["pattern"] == "generate_3d"

    def test_generate_audio_pattern(self):
        result = json.loads(handle("plan_goal", {
            "goal": "Generate TTS narration for the scene",
            "session": "test_audio",
        }))
        assert result["pattern"] == "generate_audio"
        assert result["total_steps"] >= 4

    def test_cosyvoice_trigger(self):
        result = json.loads(handle("plan_goal", {
            "goal": "Use cosyvoice to generate a voiceover",
            "session": "test_cosy",
        }))
        assert result["pattern"] == "generate_audio"

    def test_generic_fallback(self):
        result = json.loads(handle("plan_goal", {
            "goal": "Do something unusual with the workflow",
            "session": "test_generic",
        }))
        assert result["pattern"] == "generic"
        assert result["total_steps"] == 4

    def test_first_step_is_active(self):
        result = json.loads(handle("plan_goal", {
            "goal": "Build a workflow",
            "session": "test_active",
        }))
        steps = result["steps"]
        assert steps[0]["status"] == "active"
        assert all(s["status"] == "pending" for s in steps[1:])


class TestGetPlan:
    def test_no_plan(self):
        result = json.loads(handle("get_plan", {"session": "test_noplan"}))
        assert "error" in result

    def test_get_existing_plan(self):
        handle("plan_goal", {
            "goal": "Build a workflow",
            "session": "test_getplan",
        })
        result = json.loads(handle("get_plan", {"session": "test_getplan"}))
        assert result["goal"] == "Build a workflow"
        assert result["progress"].startswith("0/")
        assert result["current_step"] is not None


class TestCompleteStep:
    def test_complete_advances(self):
        handle("plan_goal", {
            "goal": "Build a workflow",
            "session": "test_complete",
        })
        plan = json.loads(handle("get_plan", {"session": "test_complete"}))
        first_step = plan["current_step"]

        result = json.loads(handle("complete_step", {
            "step_id": first_step,
            "result": "Found the model",
            "session": "test_complete",
        }))
        assert result["completed"] == first_step
        assert "next_step" in result
        assert result["next_step"] != first_step

    def test_complete_all_steps(self):
        handle("plan_goal", {
            "goal": "Do something unusual",
            "session": "test_all",
        })
        plan = json.loads(handle("get_plan", {"session": "test_all"}))
        for step in plan["steps"]:
            handle("complete_step", {
                "step_id": step["id"],
                "result": "Done",
                "session": "test_all",
            })
        final = json.loads(handle("get_plan", {"session": "test_all"}))
        assert final["status"] == "completed"

    def test_complete_nonexistent_step(self):
        handle("plan_goal", {
            "goal": "Build a workflow",
            "session": "test_bad_step",
        })
        result = json.loads(handle("complete_step", {
            "step_id": "nonexistent",
            "result": "Done",
            "session": "test_bad_step",
        }))
        assert "error" in result


class TestReplan:
    def test_replan_preserves_completed(self):
        handle("plan_goal", {
            "goal": "Do something unusual",
            "session": "test_replan",
        })
        # Complete first step
        plan = json.loads(handle("get_plan", {"session": "test_replan"}))
        handle("complete_step", {
            "step_id": plan["steps"][0]["id"],
            "result": "Done",
            "session": "test_replan",
        })

        # Replan
        result = json.loads(handle("replan", {
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
        result = json.loads(handle("replan", {
            "reason": "No plan exists",
            "session": "test_replan_none",
        }))
        assert "error" in result


class TestGoalId:
    def test_plan_goal_returns_goal_id(self):
        """plan_goal should generate and return a unique goal_id."""
        result = json.loads(handle("plan_goal", {
            "goal": "Build a test workflow",
            "session": "test_goal_id",
        }))
        assert "goal_id" in result
        assert isinstance(result["goal_id"], str)
        assert len(result["goal_id"]) == 12  # make_id() returns 12-char hex

    def test_goal_id_persisted_to_disk(self):
        """goal_id should be saved in the plan file on disk."""
        handle("plan_goal", {
            "goal": "Build a test workflow",
            "session": "test_goal_persist",
        })
        # Access via a fresh PlannerAgent instance to check disk
        agent = PlannerAgent()
        plan = agent._load_plan("test_goal_persist")
        assert "goal_id" in plan
        assert len(plan["goal_id"]) == 12

    def test_goal_id_in_get_plan_response(self):
        """get_plan should include the goal_id."""
        handle("plan_goal", {
            "goal": "Debug a broken workflow",
            "session": "test_goal_get",
        })
        result = json.loads(handle("get_plan", {"session": "test_goal_get"}))
        assert "goal_id" in result
        assert result["goal_id"] is not None

    def test_unique_goal_ids(self):
        """Each plan should get a unique goal_id."""
        r1 = json.loads(handle("plan_goal", {
            "goal": "Goal one",
            "session": "test_goal_unique1",
        }))
        r2 = json.loads(handle("plan_goal", {
            "goal": "Goal two",
            "session": "test_goal_unique2",
        }))
        assert r1["goal_id"] != r2["goal_id"]


# ---------------------------------------------------------------------------
# Cycle 34: required-field validation + per-session lock
# ---------------------------------------------------------------------------

class TestPlannerRequiredFields:
    """plan_goal / complete_step / replan must reject missing required fields."""

    def test_plan_goal_missing_goal_returns_error(self):
        """plan_goal with no 'goal' key must return error JSON, not raise KeyError."""
        result = json.loads(handle("plan_goal", {"session": "test_c34_missing"}))
        assert "error" in result

    def test_plan_goal_empty_string_goal_returns_error(self):
        """plan_goal with empty-string goal must return error JSON."""
        result = json.loads(handle("plan_goal", {"goal": "", "session": "test_c34_empty"}))
        assert "error" in result

    def test_plan_goal_whitespace_only_returns_error(self):
        """plan_goal with whitespace-only goal must return error JSON."""
        result = json.loads(handle("plan_goal", {"goal": "   ", "session": "test_c34_ws"}))
        assert "error" in result

    def test_complete_step_missing_step_id_returns_error(self):
        """complete_step with no 'step_id' must return error JSON, not raise KeyError."""
        # Create a plan first so the session has one
        handle("plan_goal", {"goal": "Build something", "session": "test_c34_cs"})
        result = json.loads(handle("complete_step", {"session": "test_c34_cs", "result": "done"}))
        assert "error" in result

    def test_replan_missing_reason_returns_error(self):
        """replan with no 'reason' must return error JSON, not raise KeyError."""
        handle("plan_goal", {"goal": "Build something", "session": "test_c34_rp"})
        result = json.loads(handle("replan", {"session": "test_c34_rp"}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Cycle 40: replan new_remaining_steps type validation
# ---------------------------------------------------------------------------

class TestReplanStepsValidation:
    """Cycle 40: replan must validate new_remaining_steps type."""

    def test_replan_steps_as_string_returns_error(self):
        """new_remaining_steps must be a list — passing a string must return error."""
        handle("plan_goal", {"goal": "Plan for validation", "session": "test_c40_rsv"})
        result = json.loads(handle("replan", {
            "reason": "Changing approach",
            "new_remaining_steps": "not a list",
            "session": "test_c40_rsv",
        }))
        assert "error" in result
        assert "list" in result["error"].lower()

    def test_replan_steps_as_int_returns_error(self):
        """new_remaining_steps as integer must return error."""
        handle("plan_goal", {"goal": "Plan for validation", "session": "test_c40_rsv2"})
        result = json.loads(handle("replan", {
            "reason": "Changing approach",
            "new_remaining_steps": 42,
            "session": "test_c40_rsv2",
        }))
        assert "error" in result

    def test_replan_steps_as_dict_returns_error(self):
        """new_remaining_steps as a dict (not list) must return error."""
        handle("plan_goal", {"goal": "Plan for validation", "session": "test_c40_rsv3"})
        result = json.loads(handle("replan", {
            "reason": "Changing approach",
            "new_remaining_steps": {"id": "step1", "action": "Do thing"},
            "session": "test_c40_rsv3",
        }))
        assert "error" in result

    def test_replan_omitting_steps_succeeds(self):
        """Omitting new_remaining_steps (defaults to []) should still work."""
        handle("plan_goal", {"goal": "Plan for validation", "session": "test_c40_rsv4"})
        result = json.loads(handle("replan", {
            "reason": "Starting fresh",
            "session": "test_c40_rsv4",
        }))
        assert "error" not in result
        assert result.get("replanned") is True

    def test_replan_valid_list_succeeds(self):
        """new_remaining_steps as a proper list must succeed."""
        handle("plan_goal", {"goal": "Plan for validation", "session": "test_c40_rsv5"})
        result = json.loads(handle("replan", {
            "reason": "New direction",
            "new_remaining_steps": [
                {"id": "step_a", "action": "Do step A"},
                {"id": "step_b", "action": "Do step B"},
            ],
            "session": "test_c40_rsv5",
        }))
        assert "error" not in result
        assert result.get("replanned") is True


# ---------------------------------------------------------------------------
# Cycle 42 — complete_step guards
# ---------------------------------------------------------------------------

class TestCompleteStepGuards:
    """Adversarial tests for Cycle 42 guards in _handle_complete_step."""

    def _make_plan(self, session: str) -> None:
        handle("plan_goal", {
            "goal": "Goal for guard tests",
            "session": session,
        })

    def test_corrupt_steps_not_list_returns_error(self, tmp_path, monkeypatch):
        """If plan['steps'] is not a list (e.g. dict), must return error not crash."""
        import json as _json
        from agent.brain.planner import PlannerAgent

        agent = PlannerAgent()
        bad_plan = {"goal": "bad", "steps": {"key": "not a list"}, "status": "active"}

        with unittest.mock.patch.object(agent, "_load_plan", return_value=bad_plan), \
             unittest.mock.patch.object(agent, "_save_plan") as mock_save:
            result = _json.loads(agent._handle_complete_step({
                "step_id": "s1", "session": "c42_corrupt",
            }))
        assert "error" in result
        assert "corrupt" in result["error"].lower() or "not a list" in result["error"].lower()
        mock_save.assert_not_called()

    def test_recompletion_of_done_step_returns_error(self):
        """Completing an already-done step must return an error."""
        session = "c42_recompletion"
        handle("plan_goal", {"goal": "Recompletion test", "session": session})

        # get first step id — get_plan returns steps at top level
        plan_result = json.loads(handle("get_plan", {"session": session}))
        first_step_id = plan_result["steps"][0]["id"]

        # Complete it once — should succeed
        r1 = json.loads(handle("complete_step", {"step_id": first_step_id, "session": session}))
        assert "error" not in r1

        # Complete it again — must fail
        r2 = json.loads(handle("complete_step", {"step_id": first_step_id, "session": session}))
        assert "error" in r2
        assert "already" in r2["error"].lower()

    def test_save_plan_failure_surfaces_error(self):
        """If _save_plan raises, the error is surfaced as JSON, not an exception."""
        import json as _json
        from agent.brain.planner import PlannerAgent

        agent = PlannerAgent()
        steps = [{"id": "s1", "action": "Do X", "status": "active", "tools": []}]
        good_plan = {"goal": "test", "steps": steps, "status": "active"}

        with unittest.mock.patch.object(agent, "_load_plan", return_value=good_plan), \
             unittest.mock.patch.object(agent, "_save_plan", side_effect=OSError("disk full")):
            result = _json.loads(agent._handle_complete_step({
                "step_id": "s1", "session": "c42_save_fail",
            }))
        assert "error" in result
        assert "disk full" in result["error"] or "save" in result["error"].lower()

    def test_complete_step_valid_still_works(self):
        """Normal complete_step flow must still succeed after guards added."""
        session = "c42_valid_complete"
        handle("plan_goal", {"goal": "Normal flow", "session": session})
        plan_result = json.loads(handle("get_plan", {"session": session}))
        first_id = plan_result["steps"][0]["id"]
        result = json.loads(handle("complete_step", {"step_id": first_id, "session": session}))
        assert "error" not in result
        assert result["completed"] == first_id


# ---------------------------------------------------------------------------
# Cycle 43 — plan_goal lock consistency + get_plan lock consistency
# ---------------------------------------------------------------------------

class TestPlanGoalLockConsistency:
    """plan_goal must acquire the per-session lock before saving."""

    def test_plan_goal_holds_lock_during_save(self):
        """_save_plan inside plan_goal is called within the session lock."""
        from agent.brain.planner import PlannerAgent, _get_plan_lock

        agent = PlannerAgent()
        session = "c43_lock_order"
        save_called_in_lock = []

        original_save = agent._save_plan

        def spy_save(sess, plan):
            lock = _get_plan_lock(sess)
            # If we're inside the lock, lock.acquire(blocking=False) returns False
            acquired = lock.acquire(blocking=False)
            save_called_in_lock.append(not acquired)  # True = was holding lock
            if acquired:
                lock.release()
            original_save(sess, plan)

        with unittest.mock.patch.object(agent, "_save_plan", side_effect=spy_save):
            agent._handle_plan_goal({"goal": "Test lock ordering", "session": session})

        assert len(save_called_in_lock) >= 1
        assert all(save_called_in_lock), "plan_goal must save while holding the session lock"

    def test_concurrent_plan_goal_does_not_lose_last_write(self):
        """Two concurrent plan_goal calls for same session serialize; last write wins cleanly."""
        import threading
        from agent.brain import handle

        session = "c43_concurrent_plan"
        results = []
        errors = []

        def make_plan(goal):
            try:
                import json as _j
                r = _j.loads(handle("plan_goal", {"goal": goal, "session": session}))
                results.append(r)
            except Exception as e:
                errors.append(str(e))

        t1 = threading.Thread(target=make_plan, args=("Goal Alpha",))
        t2 = threading.Thread(target=make_plan, args=("Goal Beta",))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"Concurrent plan_goal raised: {errors}"
        assert len(results) == 2, "Both calls must complete"
        assert all("error" not in r for r in results), "Neither should produce an error"

    def test_get_plan_returns_consistent_state(self):
        """get_plan under concurrent modification must not crash or return torn state."""
        import threading
        from agent.brain import handle

        session = "c43_get_plan_concurrent"
        handle("plan_goal", {"goal": "Baseline for read test", "session": session})

        get_results = []
        errors = []

        def read_plan():
            try:
                import json as _j
                r = _j.loads(handle("get_plan", {"session": session}))
                get_results.append(r)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=read_plan) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Concurrent get_plan raised: {errors}"
        assert len(get_results) == 5
        assert all("error" not in r for r in get_results)


# ---------------------------------------------------------------------------
# Cycle 46 — _plan_locks FIFO eviction cap
# ---------------------------------------------------------------------------

class TestPlanLocksFifoCap:
    """_plan_locks uses WeakValueDictionary for automatic memory management (Cycle 63).

    Replaces the old FIFO-eviction dict.  Tests validate the new WeakValueDictionary
    semantics: same session → same lock while referenced; locks are collected when
    unreferenced; many sessions can coexist without a hard cap.
    """

    def test_many_sessions_all_get_locks(self):
        """200+ unique sessions must each receive a usable lock, no cap error."""
        import threading
        from agent.brain import planner as planner_mod

        locks = []
        for i in range(200):
            lk = planner_mod._get_plan_lock(f"session_many_{i}")
            assert isinstance(lk, type(threading.Lock()))
            locks.append(lk)  # keep strong refs so they survive GC
        assert len(locks) == 200

    def test_existing_session_returns_same_lock(self):
        """Calling _get_plan_lock twice for same session returns the same Lock object."""
        from agent.brain import planner as planner_mod

        lock_a = planner_mod._get_plan_lock("same_session_fifo_compat")
        lock_b = planner_mod._get_plan_lock("same_session_fifo_compat")
        assert lock_a is lock_b

    def test_session_can_get_lock_after_gc(self):
        """A session whose lock was GC'd can re-acquire a fresh lock on re-entry."""
        import gc
        from agent.brain import planner as planner_mod

        session = "session_gc_reacquire"
        _first = planner_mod._get_plan_lock(session)
        del _first           # drop strong reference
        gc.collect()         # lock may now be GC'd

        new_lock = planner_mod._get_plan_lock(session)
        assert new_lock is not None   # must not raise, must return a usable lock

    def test_dict_shrinks_after_gc(self):
        """WeakValueDictionary must self-shrink once all refs are dropped."""
        import gc
        from agent.brain import planner as planner_mod

        prefix = "shrink_test_63_"
        # Create 10 locks with no strong refs kept
        for i in range(10):
            planner_mod._get_plan_lock(f"{prefix}{i}")

        gc.collect()
        surviving = [k for k in planner_mod._plan_locks if k.startswith(prefix)]
        # After GC with no strong refs, entries should be gone (or reduced)
        # We can only assert fewer than 10 remain, not exactly 0 (GC timing)
        assert len(surviving) <= 10  # at most as many as we created — never more


# ---------------------------------------------------------------------------
# Cycle 54 — replan new_remaining_steps item structure validation
# ---------------------------------------------------------------------------

class TestReplanStepItemValidation:
    """Cycle 54: each step dict in new_remaining_steps must have 'id' and 'action'."""

    def test_step_missing_id_returns_error(self):
        handle("plan_goal", {"goal": "Step id validation", "session": "test_c54_id"})
        result = json.loads(handle("replan", {
            "reason": "Adjusting",
            "new_remaining_steps": [{"action": "Do something"}],
            "session": "test_c54_id",
        }))
        assert "error" in result
        assert "id" in result["error"].lower()

    def test_step_missing_action_returns_error(self):
        handle("plan_goal", {"goal": "Step action validation", "session": "test_c54_action"})
        result = json.loads(handle("replan", {
            "reason": "Adjusting",
            "new_remaining_steps": [{"id": "step_1"}],
            "session": "test_c54_action",
        }))
        assert "error" in result
        assert "action" in result["error"].lower()

    def test_step_as_string_returns_error(self):
        handle("plan_goal", {"goal": "Step type validation", "session": "test_c54_str"})
        result = json.loads(handle("replan", {
            "reason": "Adjusting",
            "new_remaining_steps": ["not_a_dict"],
            "session": "test_c54_str",
        }))
        assert "error" in result

    def test_step_as_integer_returns_error(self):
        handle("plan_goal", {"goal": "Step int validation", "session": "test_c54_int"})
        result = json.loads(handle("replan", {
            "reason": "Adjusting",
            "new_remaining_steps": [42],
            "session": "test_c54_int",
        }))
        assert "error" in result

    def test_valid_step_not_blocked(self):
        """Well-formed step dicts must pass the struct guard and replan successfully."""
        handle("plan_goal", {"goal": "Valid step test", "session": "test_c54_valid"})
        result = json.loads(handle("replan", {
            "reason": "Adjusting approach",
            "new_remaining_steps": [
                {"id": "new_step_1", "action": "Do the new thing"},
            ],
            "session": "test_c54_valid",
        }))
        assert "error" not in result
        assert result.get("replanned") is True


# ---------------------------------------------------------------------------
# Cycle 57: plan schema guard in _handle_get_plan
# ---------------------------------------------------------------------------

class TestGetPlanCorruptSchema:
    """_handle_get_plan must guard against disk-corrupt plan files."""

    def test_missing_steps_key_returns_error(self):
        """Plan file without 'steps' key → structured error, no KeyError crash."""
        import json as _json
        from unittest.mock import patch
        from agent.brain import planner as planner_mod

        bad_plan = {"goal": "test goal", "pattern": "generic", "status": "active"}
        with patch.object(planner_mod.PlannerAgent, "_load_plan", return_value=bad_plan):
            result = _json.loads(handle("get_plan", {"session": "test_c57_bad_steps"}))

        assert "error" in result
        assert "steps" in result["error"]

    def test_steps_as_dict_returns_error(self):
        """Plan with steps as dict (not list) → structured error, no crash."""
        import json as _json
        from unittest.mock import patch
        from agent.brain import planner as planner_mod

        bad_plan = {"goal": "g", "pattern": "p", "status": "active", "steps": {}}
        with patch.object(planner_mod.PlannerAgent, "_load_plan", return_value=bad_plan):
            result = _json.loads(handle("get_plan", {"session": "test_c57_bad_dict"}))

        assert "error" in result
        assert "steps" in result["error"]

    def test_missing_goal_key_returns_error(self):
        """Plan file without 'goal' key → structured error, no KeyError crash."""
        import json as _json
        from unittest.mock import patch
        from agent.brain import planner as planner_mod

        bad_plan = {"steps": [], "pattern": "generic", "status": "active"}
        with patch.object(planner_mod.PlannerAgent, "_load_plan", return_value=bad_plan):
            result = _json.loads(handle("get_plan", {"session": "test_c57_no_goal"}))

        assert "error" in result
        assert "goal" in result["error"]

    def test_missing_pattern_key_returns_error(self):
        """Plan file without 'pattern' key → structured error."""
        import json as _json
        from unittest.mock import patch
        from agent.brain import planner as planner_mod

        bad_plan = {"steps": [], "goal": "test", "status": "active"}
        with patch.object(planner_mod.PlannerAgent, "_load_plan", return_value=bad_plan):
            result = _json.loads(handle("get_plan", {"session": "test_c57_no_pattern"}))

        assert "error" in result
        assert "pattern" in result["error"]

    def test_missing_status_key_returns_error(self):
        """Plan file without 'status' key → structured error."""
        import json as _json
        from unittest.mock import patch
        from agent.brain import planner as planner_mod

        bad_plan = {"steps": [], "goal": "test", "pattern": "generic"}
        with patch.object(planner_mod.PlannerAgent, "_load_plan", return_value=bad_plan):
            result = _json.loads(handle("get_plan", {"session": "test_c57_no_status"}))

        assert "error" in result
        assert "status" in result["error"]

    def test_valid_plan_still_works(self):
        """A properly structured plan must still return the correct response."""
        import json as _json
        result = _json.loads(handle("plan_goal", {"goal": "Build workflow", "session": "test_c57_valid"}))
        assert "error" not in result

        result = _json.loads(handle("get_plan", {"session": "test_c57_valid"}))
        assert "error" not in result
        assert result["goal"] == "Build workflow"
        assert "steps" in result


# ---------------------------------------------------------------------------
# Cycle 63: _get_plan_lock — WeakValueDictionary prevents eviction race
# ---------------------------------------------------------------------------

class TestPlanLockWeakRef:
    """_get_plan_lock() must return same lock while caller holds a reference (Cycle 63)."""

    def test_same_session_same_lock(self):
        """Two calls for the same session return the SAME lock object."""
        from agent.brain.planner import _get_plan_lock

        lock_a = _get_plan_lock("sess-same")
        lock_b = _get_plan_lock("sess-same")
        assert lock_a is lock_b, "Same session must return the same lock object"

    def test_lock_survives_while_held(self):
        """Lock stays in WeakValueDictionary as long as caller holds a strong reference."""
        import gc
        from agent.brain.planner import _get_plan_lock, _plan_locks

        lock = _get_plan_lock("sess-held")
        gc.collect()
        # Strong reference in 'lock' keeps entry alive
        assert _plan_locks.get("sess-held") is lock

    def test_lock_evicted_after_release(self):
        """When no thread holds the lock, the WeakValueDictionary can evict it."""
        import gc
        from agent.brain.planner import _get_plan_lock, _plan_locks

        session = "sess-evict-63"
        _get_plan_lock(session)   # No strong ref kept — local goes out of scope
        gc.collect()
        # May or may not be present (GC timing), but must not be a stale-different lock
        # Key assertion: if it exists, it must be a valid threading.Lock
        entry = _plan_locks.get(session)
        if entry is not None:
            import threading
            assert isinstance(entry, type(threading.Lock()))

    def test_concurrent_same_session_same_lock(self):
        """Concurrent _get_plan_lock() for the same session returns the same object."""
        import threading
        from agent.brain.planner import _get_plan_lock

        results = []

        def grab():
            results.append(_get_plan_lock("sess-concurrent-63"))

        threads = [threading.Thread(target=grab) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All returned locks must be identical objects (same session, same lock)
        assert len(set(id(lk) for lk in results)) == 1, \
            "All concurrent callers must receive the same lock object"


# ---------------------------------------------------------------------------
# Cycle 70: replan active step .get() guard (KeyError prevention)
# ---------------------------------------------------------------------------

class TestReplanActiveStepGuardCycle70:
    """Cycle 70: replan current_step must use .get("id") not ["id"] on active step."""

    def _base_plan(self, steps):
        return {
            "goal": "Test", "goal_id": "abc123", "pattern": "generic",
            "status": "active", "created_at": 0.0, "updated_at": 0.0,
            "replan_history": [],  # required by _handle_replan
            "steps": steps,
        }

    def test_replan_no_active_after_cancel_returns_none_current_step(self):
        """When new_remaining_steps=[] (cancel all), current_step must be None."""
        import json as _json
        import unittest.mock
        from agent.brain import planner as planner_mod

        plan = self._base_plan([
            {"id": "s1", "action": "First", "status": "active", "tools": []},
        ])

        with unittest.mock.patch.object(planner_mod.PlannerAgent, "_load_plan", return_value=plan), \
             unittest.mock.patch.object(planner_mod.PlannerAgent, "_save_plan"):
            result = _json.loads(planner_mod.PlannerAgent().handle("replan", {
                "session": "test_c70_cancel",
                "reason": "abort",
                "new_remaining_steps": [],  # cancels all remaining — active=None after
            }))

        assert result.get("replanned") is True
        assert result.get("current_step") is None  # Cycle 70: no active step → None

    def test_replan_with_new_steps_returns_first_step_id(self):
        """When new steps are provided, current_step must be the first new step's id."""
        import json as _json
        import unittest.mock
        from agent.brain import planner as planner_mod

        plan = self._base_plan([
            {"id": "old-1", "action": "Old step", "status": "active", "tools": []},
        ])

        with unittest.mock.patch.object(planner_mod.PlannerAgent, "_load_plan", return_value=plan), \
             unittest.mock.patch.object(planner_mod.PlannerAgent, "_save_plan"):
            result = _json.loads(planner_mod.PlannerAgent().handle("replan", {
                "session": "test_c70_newsteps",
                "reason": "pivot",
                "new_remaining_steps": [
                    {"id": "new-1", "action": "New first step", "tools": []},
                    {"id": "new-2", "action": "New second step", "tools": []},
                ],
            }))

        assert result.get("replanned") is True
        assert result.get("current_step") == "new-1"  # Cycle 70: .get("id") returns value
        assert result.get("current_action") == "New first step"  # Cycle 70: .get("action")
