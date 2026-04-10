"""Tests for brain/orchestrator.py — parallel sub-task coordination."""

import json
import time
from unittest.mock import patch

import pytest

from agent.brain import handle
from agent.brain._sdk import BrainAgent
from agent.brain.orchestrator import OrchestratorAgent, _TOOL_PROFILES


# Get the auto-registered OrchestratorAgent instance from the registry
def _get_orch_instance() -> OrchestratorAgent:
    BrainAgent._register_all()
    agent = BrainAgent._registry.get("spawn_subtask")
    assert isinstance(agent, OrchestratorAgent)
    return agent


@pytest.fixture(autouse=True)
def clean_active_tasks():
    """Reset active tasks between tests."""
    agent = _get_orch_instance()
    agent._active_tasks.clear()
    yield
    agent._active_tasks.clear()


class TestSpawnSubtask:
    def test_spawn_researcher(self):
        agent = _get_orch_instance()
        with patch.object(agent, "_run_subtask", return_value={
            "task_id": "test", "status": "completed", "results": [],
            "completed_at": time.time(),
        }):
            result = json.loads(handle("spawn_subtask", {
                "task_description": "Find SDXL models",
                "profile": "researcher",
                "tool_calls": [
                    {"tool": "list_models", "input": {"model_type": "checkpoints"}},
                ],
            }))
        assert result["spawned"] is True
        assert result["profile"] == "researcher"
        assert result["tool_count"] == 1

    def test_spawn_rejects_invalid_profile(self):
        result = json.loads(handle("spawn_subtask", {
            "task_description": "Test",
            "profile": "invalid_profile",
            "tool_calls": [],
        }))
        assert "error" in result

    def test_spawn_rejects_unauthorized_tool(self):
        result = json.loads(handle("spawn_subtask", {
            "task_description": "Try to modify workflow",
            "profile": "researcher",
            "tool_calls": [
                {"tool": "apply_workflow_patch", "input": {"patches": []}},
            ],
        }))
        assert "error" in result
        assert "not allowed" in result["error"]

    def test_spawn_max_concurrent(self):
        agent = _get_orch_instance()
        # Fill up the active tasks
        for i in range(3):
            agent._active_tasks[f"task_{i}"] = {"status": "running"}

        result = json.loads(handle("spawn_subtask", {
            "task_description": "One too many",
            "profile": "researcher",
            "tool_calls": [{"tool": "list_models", "input": {}}],
        }))
        assert "error" in result
        assert "Max" in result["error"]


class TestCheckSubtasks:
    def test_no_tasks(self):
        result = json.loads(handle("check_subtasks", {}))
        assert result["tasks"] == []

    def test_check_completed(self):
        agent = _get_orch_instance()
        agent._active_tasks["abc123"] = {
            "status": "completed",
            "description": "Find models",
            "profile": "researcher",
            "started_at": time.time() - 5,
            "results": [
                {"tool": "list_models", "result": '{"count": 5}', "elapsed_s": 0.5},
            ],
            "completed_at": time.time(),
        }
        result = json.loads(handle("check_subtasks", {}))
        assert result["summary"]["completed"] == 1
        assert result["tasks"][0]["tool_results"] == 1

    def test_check_running(self):
        agent = _get_orch_instance()
        agent._active_tasks["run123"] = {
            "status": "running",
            "description": "Searching",
            "profile": "researcher",
            "started_at": time.time() - 10,
            "tool_count": 3,
        }
        result = json.loads(handle("check_subtasks", {}))
        assert result["summary"]["running"] == 1
        assert result["tasks"][0]["elapsed_s"] >= 9

    def test_check_specific_ids(self):
        agent = _get_orch_instance()
        agent._active_tasks["a"] = {"status": "completed", "description": "A", "profile": "researcher", "results": []}
        agent._active_tasks["b"] = {"status": "running", "description": "B", "profile": "builder", "started_at": time.time(), "tool_count": 1}

        result = json.loads(handle("check_subtasks", {"task_ids": ["a"]}))
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["task_id"] == "a"


class TestToolProfiles:
    def test_researcher_has_read_tools(self):
        allowed = _TOOL_PROFILES["researcher"]["allowed_tools"]
        assert "list_models" in allowed
        assert "discover" in allowed
        assert "get_all_nodes" in allowed
        # Should NOT have write tools
        assert "apply_workflow_patch" not in allowed
        assert "execute_workflow" not in allowed

    def test_builder_has_write_tools(self):
        allowed = _TOOL_PROFILES["builder"]["allowed_tools"]
        assert "apply_workflow_patch" in allowed
        assert "add_node" in allowed
        assert "connect_nodes" in allowed

    def test_validator_has_execute_tools(self):
        allowed = _TOOL_PROFILES["validator"]["allowed_tools"]
        assert "validate_before_execute" in allowed
        assert "execute_workflow" in allowed
        assert "analyze_image" in allowed


# ---------------------------------------------------------------------------
# Cycle 36: daemon thread fix — spawned worker thread must not block exit
# ---------------------------------------------------------------------------

class TestSubtaskDaemonThread:
    """spawn_subtask worker thread must be a daemon thread (Cycle 36 fix)."""

    def test_spawned_worker_is_daemon(self):
        """After spawn, the worker thread running _run_subtask must be daemon=True."""
        import threading
        agent = _get_orch_instance()

        captured_threads: list[threading.Thread] = []
        original_start = threading.Thread.start

        def _recording_start(self_thread, *args, **kwargs):
            # Capture the thread BEFORE it starts (daemon attribute is set before start)
            if "subtask-" in (self_thread.name or ""):
                captured_threads.append(self_thread)
            original_start(self_thread, *args, **kwargs)

        with patch.object(threading.Thread, "start", _recording_start), \
             patch.object(agent, "_run_subtask", return_value={
                 "task_id": "t1", "status": "completed", "results": [],
                 "completed_at": time.time(),
             }):
            agent.handle("spawn_subtask", {
                "task_description": "Daemon check",
                "profile": "researcher",
                "tool_calls": [{"tool": "list_models", "input": {}}],
            })
            time.sleep(0.05)  # Let the thread start

        assert len(captured_threads) >= 1, "No subtask thread was recorded"
        for t in captured_threads:
            assert t.daemon is True, f"Thread {t.name!r} is not a daemon thread"


# ---------------------------------------------------------------------------
# Cycle 46 — spawn_subtask required field guards
# ---------------------------------------------------------------------------

class TestSpawnSubtaskRequiredFields:
    """spawn_subtask must return structured errors when required fields are missing."""

    def test_missing_task_description_returns_error(self):
        result = json.loads(handle("spawn_subtask", {
            "profile": "researcher",
            "tool_calls": [],
        }))
        assert "error" in result
        assert "task_description" in result["error"].lower()

    def test_missing_profile_returns_error(self):
        result = json.loads(handle("spawn_subtask", {
            "task_description": "do something",
            "tool_calls": [],
        }))
        assert "error" in result
        assert "profile" in result["error"].lower()

    def test_missing_tool_calls_returns_error(self):
        result = json.loads(handle("spawn_subtask", {
            "task_description": "do something",
            "profile": "researcher",
        }))
        assert "error" in result
        assert "tool_calls" in result["error"].lower()

    def test_none_task_description_returns_error(self):
        result = json.loads(handle("spawn_subtask", {
            "task_description": None,
            "profile": "researcher",
            "tool_calls": [],
        }))
        assert "error" in result

    def test_non_list_tool_calls_returns_error(self):
        result = json.loads(handle("spawn_subtask", {
            "task_description": "do something",
            "profile": "researcher",
            "tool_calls": "not_a_list",
        }))
        assert "error" in result
        assert "tool_calls" in result["error"].lower()

    def test_empty_tool_calls_list_returns_error(self):
        """Cycle 54: Empty tool_calls list must be rejected — spawning a task with no work is pointless."""
        result = json.loads(handle("spawn_subtask", {
            "task_description": "analyze models",
            "profile": "researcher",
            "tool_calls": [],
        }))
        assert "error" in result
        assert "tool_calls" in result["error"].lower()


# ---------------------------------------------------------------------------
# Cycle 54 — malformed tool_calls item guard
# ---------------------------------------------------------------------------

class TestSpawnSubtaskMalformedCalls:
    """spawn_subtask must reject malformed items in tool_calls list."""

    def test_string_item_returns_error(self):
        result = json.loads(handle("spawn_subtask", {
            "task_description": "do something",
            "profile": "researcher",
            "tool_calls": ["not_a_dict"],
        }))
        assert "error" in result
        assert "tool" in result["error"].lower()

    def test_dict_missing_tool_key_returns_error(self):
        result = json.loads(handle("spawn_subtask", {
            "task_description": "do something",
            "profile": "researcher",
            "tool_calls": [{"input": {}}],
        }))
        assert "error" in result
        assert "tool" in result["error"].lower()

    def test_dict_empty_tool_value_returns_error(self):
        result = json.loads(handle("spawn_subtask", {
            "task_description": "do something",
            "profile": "researcher",
            "tool_calls": [{"tool": "", "input": {}}],
        }))
        assert "error" in result

    def test_integer_item_returns_error(self):
        result = json.loads(handle("spawn_subtask", {
            "task_description": "do something",
            "profile": "researcher",
            "tool_calls": [42],
        }))
        assert "error" in result

    def test_valid_call_not_blocked_by_struct_guard(self):
        """Well-formed call dict must pass the struct guard (may fail on profile/tool-allowed check)."""
        result = json.loads(handle("spawn_subtask", {
            "task_description": "do something",
            "profile": "researcher",
            "tool_calls": [{"tool": "list_models", "input": {}}],
        }))
        # Must not be a struct-guard error
        assert "tool_calls must be" not in result.get("error", "")
        assert "must be a dict" not in result.get("error", "")
