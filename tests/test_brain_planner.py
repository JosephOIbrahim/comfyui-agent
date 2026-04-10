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
        import threading
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
        t1.start(); t2.start()
        t1.join(timeout=5); t2.join(timeout=5)

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
        for t in threads: t.start()
        for t in threads: t.join(timeout=5)

        assert not errors, f"Concurrent get_plan raised: {errors}"
        assert len(get_results) == 5
        assert all("error" not in r for r in get_results)
