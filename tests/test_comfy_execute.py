"""Tests for comfy_execute tool — mocked HTTP, no real ComfyUI needed."""

import json
import pytest
from unittest.mock import patch, MagicMock
from agent.tools import comfy_execute


def _mock_httpx_client():
    """Create a properly chained httpx.Client context manager mock.

    Returns (patcher, mock_client) — mock_client has .post, .get, etc.
    """
    mock_client = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_client)
    mock_cm.__exit__ = MagicMock(return_value=False)
    patcher = patch("agent.tools.comfy_execute.httpx.Client", return_value=mock_cm)
    return patcher, mock_client


@pytest.fixture(autouse=True)
def _allow_breaker():
    """Ensure circuit breaker allows requests in tests."""
    with patch("agent.tools.comfy_execute.COMFYUI_BREAKER", create=True):
        breaker = MagicMock()
        breaker.allow_request.return_value = True
        # Patch at the import location used by _queue_prompt
        with patch("agent.circuit_breaker.get_breaker", return_value=breaker):
            yield


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

        patcher, mock_client = _mock_httpx_client()
        mock_client.post.return_value = mock_resp
        with patcher:
            prompt_id, err = comfy_execute._queue_prompt({"1": {"class_type": "Test", "inputs": {}}})
            assert prompt_id == "abc123"
            assert err is None

    def test_queue_connection_error(self):
        import httpx
        patcher, mock_client = _mock_httpx_client()
        mock_client.post.side_effect = httpx.ConnectError("refused")
        with patcher:
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

        patcher, mock_client = _mock_httpx_client()
        mock_client.post.return_value = mock_resp
        with patcher:
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

        patcher, mock_client = _mock_httpx_client()
        with patcher:
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
        workflow_patch._get_state()["current_workflow"] = None

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

        patcher, mock_client = _mock_httpx_client()
        with patcher:
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

        patcher, mock_client = _mock_httpx_client()
        with patcher:
            mock_client.get.side_effect = [history_resp, queue_resp]

            result = json.loads(comfy_execute.handle("get_execution_status", {
                "prompt_id": "running123",
            }))
            assert result["status"] == "running"


class TestPollCompletion:
    """Unit tests for _poll_completion — polling HTTP fallback path."""

    def _make_history_resp(self, prompt_id, status_str="success", outputs_dict=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            prompt_id: {
                "status": {"status_str": status_str, "completed": True},
                "outputs": outputs_dict or {},
            }
        }
        return resp

    def test_poll_success_with_outputs(self):
        """Successful poll with outputs returns no warning."""
        resp = self._make_history_resp(
            "p1",
            outputs_dict={"5": {"images": [{"filename": "out.png", "subfolder": ""}]}},
        )
        patcher, mock_client = _mock_httpx_client()
        with patcher:
            mock_client.get.return_value = resp
            with patch("agent.circuit_breaker.get_breaker") as gb:
                gb.return_value.allow_request.return_value = True
                result = comfy_execute._poll_completion("p1", timeout=5)

        assert result["status"] == "complete"
        assert result["outputs"][0]["filename"] == "out.png"
        assert "outputs_warning" not in result

    def test_poll_success_empty_outputs_adds_warning(self):
        """Success with no outputs in history must include outputs_warning (parity with WS path)."""
        resp = self._make_history_resp("p2", outputs_dict={})
        patcher, mock_client = _mock_httpx_client()
        with patcher:
            mock_client.get.return_value = resp
            with patch("agent.circuit_breaker.get_breaker") as gb:
                gb.return_value.allow_request.return_value = True
                result = comfy_execute._poll_completion("p2", timeout=5)

        assert result["status"] == "complete"
        assert result["outputs"] == []
        assert "outputs_warning" in result
        assert "p2" in result["outputs_warning"]

    def test_poll_error_status_no_warning(self):
        """Error-status result should not get an outputs_warning."""
        resp = self._make_history_resp("p3", status_str="error", outputs_dict={})
        patcher, mock_client = _mock_httpx_client()
        with patcher:
            mock_client.get.return_value = resp
            with patch("agent.circuit_breaker.get_breaker") as gb:
                gb.return_value.allow_request.return_value = True
                result = comfy_execute._poll_completion("p3", timeout=5)

        assert result["status"] == "error"
        assert "outputs_warning" not in result


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
        workflow_patch._get_state()["current_workflow"] = None
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

        patcher, mock_client = _mock_httpx_client()
        mock_client.post.return_value = queue_resp
        mock_client.get.return_value = history_resp
        with patch.object(comfy_execute, "_HAS_WS", True), \
             patcher, \
             patch("agent.tools.comfy_execute.websockets.sync.client.connect", return_value=mock_ws):

            result = json.loads(comfy_execute.handle("execute_with_progress", {
                "path": str(sample_workflow),
                "timeout": 30,
            }))
            assert result["status"] == "complete"
            assert result["monitoring"] == "websocket"
            assert result["progress_events"] >= 3
            assert len(result["node_timing"]) >= 1


class TestPathTraversal:
    def test_path_traversal_blocked_execute_workflow(self):
        """Path traversal attempts must be rejected before any file read."""
        result = json.loads(comfy_execute.handle("execute_workflow", {
            "path": "../../../../etc/passwd",
        }))
        assert "error" in result

    def test_path_traversal_blocked_validate_before_execute(self):
        """validate_before_execute must reject traversal paths."""
        result = json.loads(comfy_execute.handle("validate_before_execute", {
            "path": "../../../../etc/passwd",
        }))
        assert "error" in result

    def test_path_traversal_blocked_execute_with_progress(self):
        """execute_with_progress must reject traversal paths."""
        result = json.loads(comfy_execute.handle("execute_with_progress", {
            "path": "../../../../etc/passwd",
        }))
        assert "error" in result

    def test_windows_traversal_blocked(self):
        """Windows-style traversal must be rejected."""
        result = json.loads(comfy_execute.handle("execute_workflow", {
            "path": "..\\..\\..\\Windows\\System32\\drivers\\etc\\hosts",
        }))
        assert "error" in result


class TestEmptyRequiredInputValidation:
    """Cycle 29: validate_before_execute must flag empty/None required inputs."""

    def _make_object_info(self, required_fields: dict) -> dict:
        """Build a minimal object_info response with the given required fields."""
        return {
            "CLIPTextEncode": {
                "input": {
                    "required": required_fields,
                    "optional": {},
                },
                "output": ["CONDITIONING"],
            }
        }

    def _setup_workflow_and_mock(self, text_value, tmp_path):
        """Write a workflow with CLIPTextEncode.text set to text_value."""
        data = {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": text_value, "clip": ["2", 1]},
            },
        }
        path = tmp_path / "wf.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_empty_string_required_input_fails(self, tmp_path):
        """validate_before_execute must report error for empty-string required input."""
        path = self._setup_workflow_and_mock("", tmp_path)
        object_info = self._make_object_info({
            "text": ["STRING", {}],
            "clip": ["CLIP", {}],
        })
        mock_resp = MagicMock()
        mock_resp.json.return_value = object_info
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = mock_resp
            result = json.loads(comfy_execute.handle("validate_before_execute", {
                "path": str(path),
            }))

        assert result.get("valid") is False
        errors = result.get("errors", [])
        assert any("empty" in e.lower() or "none" in e.lower() for e in errors), (
            f"Expected empty/None error in {errors}"
        )

    def test_none_required_input_fails(self, tmp_path):
        """validate_before_execute must report error for None required input."""
        # Write workflow with None explicitly (JSON null)
        data = {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": None, "clip": ["2", 1]},
            },
        }
        path = tmp_path / "wf_null.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        object_info = self._make_object_info({
            "text": ["STRING", {}],
            "clip": ["CLIP", {}],
        })
        mock_resp = MagicMock()
        mock_resp.json.return_value = object_info
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = mock_resp
            result = json.loads(comfy_execute.handle("validate_before_execute", {
                "path": str(path),
            }))

        assert result.get("valid") is False
        errors = result.get("errors", [])
        assert any("empty" in e.lower() or "none" in e.lower() for e in errors), (
            f"Expected empty/None error in {errors}"
        )

    def test_whitespace_only_required_input_fails(self, tmp_path):
        """validate_before_execute must reject whitespace-only string required inputs."""
        data = {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "   ", "clip": ["2", 1]},
            },
        }
        path = tmp_path / "wf_ws.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        object_info = self._make_object_info({
            "text": ["STRING", {}],
            "clip": ["CLIP", {}],
        })
        mock_resp = MagicMock()
        mock_resp.json.return_value = object_info
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = mock_resp
            result = json.loads(comfy_execute.handle("validate_before_execute", {
                "path": str(path),
            }))

        assert result.get("valid") is False

    def test_valid_text_passes(self, tmp_path):
        """validate_before_execute must not flag non-empty required string inputs."""
        data = {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "a beautiful sunset", "clip": ["2", 1]},
            },
        }
        path = tmp_path / "wf_ok.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        object_info = self._make_object_info({
            "text": ["STRING", {}],
            "clip": ["CLIP", {}],
        })
        mock_resp = MagicMock()
        mock_resp.json.return_value = object_info
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = mock_resp
            result = json.loads(comfy_execute.handle("validate_before_execute", {
                "path": str(path),
            }))

        # "clip" is a link so it won't trigger the empty-input error.
        # "text" is valid. No empty/None errors.
        errors = result.get("errors", [])
        assert not any("empty" in e.lower() or "none" in e.lower() for e in errors)


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


# ---------------------------------------------------------------------------
# Cycle 40: validate_before_execute rejects empty workflow
# ---------------------------------------------------------------------------

class TestValidateEmptyWorkflow:
    """Cycle 40: validate_before_execute must reject empty workflows."""

    def test_empty_workflow_from_patch_state_returns_error(self):
        """If current workflow is empty dict, validate_before_execute must error."""
        from agent.tools.workflow_patch import _get_state
        state = _get_state()
        old_wf = state.get("current_workflow")
        try:
            # Inject empty workflow into patch state
            state["current_workflow"] = {}
            state["loaded_path"] = "test.json"
            result = json.loads(comfy_execute.handle("validate_before_execute", {}))
            assert "error" in result
            assert "empty" in result["error"].lower()
        finally:
            state["current_workflow"] = old_wf

    def test_empty_workflow_from_file_returns_error(self, tmp_path):
        """A workflow file with zero nodes must cause validate_before_execute to error."""
        wf_path = tmp_path / "empty.json"
        wf_path.write_text("{}", encoding="utf-8")
        result = json.loads(comfy_execute.handle("validate_before_execute", {
            "path": str(wf_path),
        }))
        assert "error" in result
        msg = result["error"].lower()
        assert "empty" in msg or "no nodes" in msg or "workflow" in msg


# ---------------------------------------------------------------------------
# Cycle 47 — get_execution_status required field guard
# ---------------------------------------------------------------------------

class TestGetExecutionStatusRequiredField:
    """get_execution_status must return structured error when prompt_id is missing."""

    def test_missing_prompt_id_returns_error(self):
        import json
        from agent.tools import comfy_execute
        result = json.loads(comfy_execute.handle("get_execution_status", {}))
        assert "error" in result
        assert "prompt_id" in result["error"].lower()

    def test_empty_prompt_id_returns_error(self):
        import json
        from agent.tools import comfy_execute
        result = json.loads(comfy_execute.handle("get_execution_status", {"prompt_id": ""}))
        assert "error" in result

    def test_none_prompt_id_returns_error(self):
        import json
        from agent.tools import comfy_execute
        result = json.loads(comfy_execute.handle("get_execution_status", {"prompt_id": None}))
        assert "error" in result

    def test_integer_prompt_id_returns_error(self):
        import json
        from agent.tools import comfy_execute
        result = json.loads(comfy_execute.handle("get_execution_status", {"prompt_id": 42}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Cycle 51 — prompt_id format/length validation
# ---------------------------------------------------------------------------

class TestGetExecutionStatusPromptIdFormat:
    """get_execution_status must reject malformed or oversized prompt_ids."""

    def test_path_traversal_prompt_id_returns_error(self):
        import json
        from agent.tools import comfy_execute
        result = json.loads(comfy_execute.handle("get_execution_status", {
            "prompt_id": "../../admin",
        }))
        assert "error" in result

    def test_prompt_id_too_long_returns_error(self):
        import json
        from agent.tools import comfy_execute
        result = json.loads(comfy_execute.handle("get_execution_status", {
            "prompt_id": "a" * 200,
        }))
        assert "error" in result

    def test_valid_uuid_prompt_id_passes_format(self):
        import json
        from unittest.mock import patch
        from agent.tools import comfy_execute
        # Only check that format guard doesn't block a valid UUID
        with patch("httpx.Client") as mock_client:
            mock_resp = mock_client.return_value.__enter__.return_value.get.return_value
            mock_resp.status_code = 200
            mock_resp.json.return_value = {}
            mock_resp.raise_for_status = lambda: None
            result = json.loads(comfy_execute.handle("get_execution_status", {
                "prompt_id": "3d2f5b1a-e4c7-4f8d-a2b3-c1d4e5f60789",
            }))
        # Should NOT error on format — may error on empty history but not format
        assert "alphanumeric" not in result.get("error", "")
        assert "too long" not in result.get("error", "")


# ---------------------------------------------------------------------------
# Cycle 63: timeout type coercion guard
# ---------------------------------------------------------------------------

class TestTimeoutTypeGuard:
    """execute_workflow/execute_with_progress must reject non-numeric timeout (Cycle 63)."""

    def test_execute_workflow_string_timeout_returns_error(self):
        """String timeout must return a JSON error, not crash with TypeError."""
        import json
        from agent.tools import comfy_execute
        result = json.loads(comfy_execute.handle("execute_workflow", {"timeout": "notanumber"}))
        assert "error" in result
        assert "timeout" in result["error"].lower()

    def test_execute_with_progress_string_timeout_returns_error(self):
        """String timeout in execute_with_progress must return a JSON error."""
        import json
        from agent.tools import comfy_execute
        result = json.loads(comfy_execute.handle("execute_with_progress", {"timeout": "bad"}))
        assert "error" in result
        assert "timeout" in result["error"].lower()

    def test_execute_workflow_numeric_string_timeout_returns_error(self):
        """Even '120' (numeric-looking string) must be rejected without coercion."""
        import json
        from agent.tools import comfy_execute
        # We intentionally do NOT silently coerce — the schema says number
        result = json.loads(comfy_execute.handle("execute_workflow", {"timeout": "120"}))
        assert "error" in result

    def test_execute_workflow_float_timeout_accepted(self):
        """A float timeout must NOT trigger the type guard (it's a valid number)."""
        import json
        from unittest.mock import patch
        from agent.tools import comfy_execute
        # Patch the actual execution so we don't need ComfyUI running
        with patch.object(comfy_execute, "_queue_prompt", return_value=(None, "ComfyUI not reachable")):
            result = json.loads(comfy_execute.handle("execute_workflow", {"timeout": 60.0}))
        # Error must not be about timeout type
        assert result.get("error", "") != "timeout must be a number (seconds)"
