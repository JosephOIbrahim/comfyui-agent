"""Tests for comfy_execute tool — mocked HTTP, no real ComfyUI needed."""

import json
import pytest
from unittest.mock import patch, MagicMock
from agent.tools import comfy_execute


@pytest.fixture
def sample_workflow(tmp_path):
    """Create a minimal API-format workflow."""
    data = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd15.safetensors"},
        },
        "2": {
            "class_type": "KSampler",
            "inputs": {"model": ["1", 0], "seed": 42, "steps": 5},
        },
    }
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestQueuePrompt:
    def test_queue_success(self):
        """Mock a successful prompt queue."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"prompt_id": "abc123", "number": 1}
        mock_resp.raise_for_status = MagicMock()

        with patch("agent.tools.comfy_execute.httpx.Client") as mock_cls:
            mock_client = mock_cls.return_value.__enter__.return_value
            mock_client.post.return_value = mock_resp

            prompt_id, err = comfy_execute._queue_prompt({"1": {"class_type": "Test", "inputs": {}}})
            assert prompt_id == "abc123"
            assert err is None

    def test_queue_connection_error(self):
        import httpx
        with patch("agent.tools.comfy_execute.httpx.Client") as mock_cls:
            mock_client = mock_cls.return_value.__enter__.return_value
            mock_client.post.side_effect = httpx.ConnectError("refused")

            prompt_id, err = comfy_execute._queue_prompt({"1": {"class_type": "Test", "inputs": {}}})
            assert prompt_id is None
            assert "running" in err.lower()

    def test_queue_validation_error(self):
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {
            "error": "prompt has no outputs",
            "node_errors": {},
        }
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400", request=MagicMock(), response=mock_resp,
        )

        with patch("agent.tools.comfy_execute.httpx.Client") as mock_cls:
            mock_client = mock_cls.return_value.__enter__.return_value
            mock_client.post.return_value = mock_resp

            prompt_id, err = comfy_execute._queue_prompt({"1": {"class_type": "Test", "inputs": {}}})
            assert prompt_id is None
            assert "no outputs" in err


class TestExecuteWorkflow:
    def test_execute_from_file(self, sample_workflow):
        """Mock queue + immediate completion."""
        queue_resp = MagicMock()
        queue_resp.json.return_value = {"prompt_id": "test123"}
        queue_resp.raise_for_status = MagicMock()

        history_resp = MagicMock()
        history_resp.json.return_value = {
            "test123": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "2": {
                        "images": [{"filename": "out_00001.png", "subfolder": ""}],
                    },
                },
            },
        }
        history_resp.raise_for_status = MagicMock()

        with patch("agent.tools.comfy_execute.httpx.Client") as mock_cls:
            mock_client = mock_cls.return_value.__enter__.return_value
            mock_client.post.return_value = queue_resp
            mock_client.get.return_value = history_resp

            result = json.loads(comfy_execute.handle("execute_workflow", {
                "path": str(sample_workflow),
                "timeout": 5,
            }))
            assert result["status"] == "complete"
            assert result["prompt_id"] == "test123"
            assert len(result["outputs"]) == 1
            assert result["outputs"][0]["filename"] == "out_00001.png"

    def test_execute_no_workflow(self):
        """No file path and no loaded workflow."""
        # Clear any loaded workflow state
        from agent.tools import workflow_patch
        workflow_patch._state["current_workflow"] = None

        result = json.loads(comfy_execute.handle("execute_workflow", {}))
        assert "error" in result

    def test_execute_file_not_found(self):
        result = json.loads(comfy_execute.handle("execute_workflow", {
            "path": "/nonexistent/wf.json",
        }))
        assert "error" in result

    def test_execute_ui_only_rejected(self, tmp_path):
        data = {
            "nodes": [{"id": 1, "type": "Test"}],
            "extra": {"ds": {}},
        }
        path = tmp_path / "ui.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        result = json.loads(comfy_execute.handle("execute_workflow", {
            "path": str(path),
        }))
        assert "error" in result


class TestGetExecutionStatus:
    def test_completed(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "prompt123": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "5": {"images": [{"filename": "result.png", "subfolder": ""}]},
                },
            },
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("agent.tools.comfy_execute.httpx.Client") as mock_cls:
            mock_client = mock_cls.return_value.__enter__.return_value
            mock_client.get.return_value = mock_resp

            result = json.loads(comfy_execute.handle("get_execution_status", {
                "prompt_id": "prompt123",
            }))
            assert result["status"] == "success"
            assert result["completed"] is True
            assert result["outputs"][0]["filename"] == "result.png"

    def test_not_found_checks_queue(self):
        """If not in history, check queue for running/pending."""
        history_resp = MagicMock()
        history_resp.json.return_value = {}  # not in history
        history_resp.raise_for_status = MagicMock()

        queue_resp = MagicMock()
        queue_resp.json.return_value = {
            "queue_running": [[0, "running123", {}]],
            "queue_pending": [],
        }
        queue_resp.raise_for_status = MagicMock()

        with patch("agent.tools.comfy_execute.httpx.Client") as mock_cls:
            mock_client = mock_cls.return_value.__enter__.return_value
            mock_client.get.side_effect = [history_resp, queue_resp]

            result = json.loads(comfy_execute.handle("get_execution_status", {
                "prompt_id": "running123",
            }))
            assert result["status"] == "running"


class TestExecuteWithProgress:
    def test_fallback_when_no_websocket(self, sample_workflow):
        """Falls back to polling when websockets unavailable."""
        queue_resp = MagicMock()
        queue_resp.json.return_value = {"prompt_id": "ws_test"}
        queue_resp.raise_for_status = MagicMock()

        history_resp = MagicMock()
        history_resp.json.return_value = {
            "ws_test": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {"2": {"images": [{"filename": "ws_out.png", "subfolder": ""}]}},
            },
        }
        history_resp.raise_for_status = MagicMock()

        with patch.object(comfy_execute, "_HAS_WS", False), \
             patch("agent.tools.comfy_execute.httpx.Client") as mock_cls:
            mock_client = mock_cls.return_value.__enter__.return_value
            mock_client.post.return_value = queue_resp
            mock_client.get.return_value = history_resp

            result = json.loads(comfy_execute.handle("execute_with_progress", {
                "path": str(sample_workflow),
                "timeout": 5,
            }))
            assert result["status"] == "complete"
            assert result["monitoring"] == "polling_fallback"

    def test_no_workflow_error(self):
        from agent.tools import workflow_patch
        workflow_patch._state["current_workflow"] = None
        result = json.loads(comfy_execute.handle("execute_with_progress", {}))
        assert "error" in result

    def test_ws_execute_with_mock(self, sample_workflow):
        """Simulate WebSocket execution with mocked connection."""
        queue_resp = MagicMock()
        queue_resp.json.return_value = {"prompt_id": "ws_full"}
        queue_resp.raise_for_status = MagicMock()

        history_resp = MagicMock()
        history_resp.json.return_value = {
            "ws_full": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {"2": {"images": [{"filename": "ws_result.png", "subfolder": ""}]}},
            },
        }
        history_resp.raise_for_status = MagicMock()

        # Build WS message sequence
        ws_messages = [
            json.dumps({"type": "execution_start", "data": {"prompt_id": "ws_full"}}),
            json.dumps({"type": "executing", "data": {"node": "1", "prompt_id": "ws_full"}}),
            json.dumps({"type": "executing", "data": {"node": "2", "prompt_id": "ws_full"}}),
            json.dumps({"type": "progress", "data": {"value": 3, "max": 5, "prompt_id": "ws_full"}}),
            json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "ws_full"}}),
        ]

        mock_ws = MagicMock()
        msg_iter = iter(ws_messages)
        mock_ws.recv.side_effect = lambda timeout=None: next(msg_iter)
        mock_ws.__enter__ = MagicMock(return_value=mock_ws)
        mock_ws.__exit__ = MagicMock(return_value=False)

        with patch.object(comfy_execute, "_HAS_WS", True), \
             patch("agent.tools.comfy_execute.httpx.Client") as mock_http, \
             patch("agent.tools.comfy_execute.websockets.sync.client.connect", return_value=mock_ws):
            mock_client = mock_http.return_value.__enter__.return_value
            mock_client.post.return_value = queue_resp
            mock_client.get.return_value = history_resp

            result = json.loads(comfy_execute.handle("execute_with_progress", {
                "path": str(sample_workflow),
                "timeout": 30,
            }))
            assert result["status"] == "complete"
            assert result["monitoring"] == "websocket"
            assert result["progress_events"] >= 3
            assert len(result["node_timing"]) >= 1


class TestRegistration:
    def test_tools_registered(self):
        from agent.tools import ALL_TOOLS
        names = {t["name"] for t in ALL_TOOLS}
        assert "execute_workflow" in names
        assert "execute_with_progress" in names
        assert "get_execution_status" in names
        assert "apply_workflow_patch" in names
        assert "undo_workflow_patch" in names
        assert "preview_workflow_patch" in names
        assert "get_workflow_diff" in names
        assert "save_workflow" in names
        assert "reset_workflow" in names
