"""Tests for brain/orchestrator.py — parallel sub-task coordination."""

import json
import time
from unittest.mock import patch

import pytest

from agent.brain import orchestrator


@pytest.fixture(autouse=True)
def clean_active_tasks():
    """Reset active tasks between tests."""
    orchestrator._active_tasks.clear()
    yield
    orchestrator._active_tasks.clear()


class TestSpawnSubtask:
    def test_spawn_researcher(self):
        with patch("agent.brain.orchestrator._run_subtask", return_value={
            "task_id": "test", "status": "completed", "results": [],
            "completed_at": time.time(),
        }):
            result = json.loads(orchestrator.handle("spawn_subtask", {
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
        result = json.loads(orchestrator.handle("spawn_subtask", {
            "task_description": "Test",
            "profile": "invalid_profile",
            "tool_calls": [],
        }))
        assert "error" in result

    def test_spawn_rejects_unauthorized_tool(self):
        result = json.loads(orchestrator.handle("spawn_subtask", {
            "task_description": "Try to modify workflow",
            "profile": "researcher",
            "tool_calls": [
                {"tool": "apply_workflow_patch", "input": {"patches": []}},
            ],
        }))
        assert "error" in result
        assert "not allowed" in result["error"]

    def test_spawn_max_concurrent(self):
        # Fill up the active tasks
        for i in range(3):
            orchestrator._active_tasks[f"task_{i}"] = {"status": "running"}

        result = json.loads(orchestrator.handle("spawn_subtask", {
            "task_description": "One too many",
            "profile": "researcher",
            "tool_calls": [{"tool": "list_models", "input": {}}],
        }))
        assert "error" in result
        assert "Max" in result["error"]


class TestCheckSubtasks:
    def test_no_tasks(self):
        result = json.loads(orchestrator.handle("check_subtasks", {}))
        assert result["tasks"] == []

    def test_check_completed(self):
        orchestrator._active_tasks["abc123"] = {
            "status": "completed",
            "description": "Find models",
            "profile": "researcher",
            "started_at": time.time() - 5,
            "results": [
                {"tool": "list_models", "result": '{"count": 5}', "elapsed_s": 0.5},
            ],
            "completed_at": time.time(),
        }
        result = json.loads(orchestrator.handle("check_subtasks", {}))
        assert result["summary"]["completed"] == 1
        assert result["tasks"][0]["tool_results"] == 1

    def test_check_running(self):
        orchestrator._active_tasks["run123"] = {
            "status": "running",
            "description": "Searching",
            "profile": "researcher",
            "started_at": time.time() - 10,
            "tool_count": 3,
        }
        result = json.loads(orchestrator.handle("check_subtasks", {}))
        assert result["summary"]["running"] == 1
        assert result["tasks"][0]["elapsed_s"] >= 9

    def test_check_specific_ids(self):
        orchestrator._active_tasks["a"] = {"status": "completed", "description": "A", "profile": "researcher", "results": []}
        orchestrator._active_tasks["b"] = {"status": "running", "description": "B", "profile": "builder", "started_at": time.time(), "tool_count": 1}

        result = json.loads(orchestrator.handle("check_subtasks", {"task_ids": ["a"]}))
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["task_id"] == "a"


class TestToolProfiles:
    def test_researcher_has_read_tools(self):
        allowed = orchestrator._TOOL_PROFILES["researcher"]["allowed_tools"]
        assert "list_models" in allowed
        assert "discover" in allowed
        assert "get_all_nodes" in allowed
        # Should NOT have write tools
        assert "apply_workflow_patch" not in allowed
        assert "execute_workflow" not in allowed

    def test_builder_has_write_tools(self):
        allowed = orchestrator._TOOL_PROFILES["builder"]["allowed_tools"]
        assert "apply_workflow_patch" in allowed
        assert "add_node" in allowed
        assert "connect_nodes" in allowed

    def test_validator_has_execute_tools(self):
        allowed = orchestrator._TOOL_PROFILES["validator"]["allowed_tools"]
        assert "validate_before_execute" in allowed
        assert "execute_workflow" in allowed
        assert "analyze_image" in allowed
