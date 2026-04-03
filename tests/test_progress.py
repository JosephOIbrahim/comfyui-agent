"""Tests for progress reporting — MCP progress notifications for long-running tools."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.progress import ProgressReporter, _NoopReporter


def _mock_httpx_client():
    """Create a properly chained httpx.Client context manager mock."""
    mock_client = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_client)
    mock_cm.__exit__ = MagicMock(return_value=False)
    patcher = patch("agent.tools.comfy_execute.httpx.Client", return_value=mock_cm)
    return patcher, mock_client


class TestNoopReporter:
    def test_noop_is_silent(self):
        """Noop reporter accepts any arguments without error."""
        reporter = ProgressReporter.noop()
        assert isinstance(reporter, _NoopReporter)
        # Should not raise
        reporter.report(0)
        reporter.report(5, 10)
        reporter.report(5, 10, "hello")

    def test_noop_matches_protocol(self):
        """Noop satisfies the ProgressCallback protocol."""
        reporter = ProgressReporter.noop()
        assert hasattr(reporter, "report")
        assert callable(reporter.report)


class TestProgressReporter:
    def test_no_token_is_silent(self):
        """When progress_token is None, report() is a no-op."""
        loop = MagicMock()
        session = MagicMock()
        reporter = ProgressReporter(loop, session, progress_token=None)
        # Should not call run_coroutine_threadsafe
        reporter.report(5, 10, "test")
        # asyncio.run_coroutine_threadsafe should NOT have been called
        # (we'd get an error if it tried to use the mock loop)

    def test_sends_notification_with_token(self):
        """When progress_token is set, report() fires a notification."""
        loop = asyncio.new_event_loop()
        session = MagicMock()
        session.send_progress_notification = AsyncMock()
        reporter = ProgressReporter(loop, session, progress_token="tok-1")

        try:
            reporter.report(5, 10, "Step 5/10")
            # Give the event loop a moment to process
            loop.run_until_complete(asyncio.sleep(0.05))
        finally:
            loop.close()

        session.send_progress_notification.assert_called_once_with(
            progress_token="tok-1",
            progress=5,
            total=10,
            message="Step 5/10",
        )

    def test_sends_without_total(self):
        """Progress without total (indeterminate) works."""
        loop = asyncio.new_event_loop()
        session = MagicMock()
        session.send_progress_notification = AsyncMock()
        reporter = ProgressReporter(loop, session, progress_token=42)

        try:
            reporter.report(3, message="Loading...")
            loop.run_until_complete(asyncio.sleep(0.05))
        finally:
            loop.close()

        session.send_progress_notification.assert_called_once_with(
            progress_token=42,
            progress=3,
            total=None,
            message="Loading...",
        )

    def test_exception_suppressed(self):
        """Errors in progress notification never crash the handler."""
        loop = asyncio.new_event_loop()
        session = MagicMock()
        session.send_progress_notification = AsyncMock(
            side_effect=RuntimeError("transport closed")
        )
        reporter = ProgressReporter(loop, session, progress_token="tok-err")

        try:
            # Should not raise
            reporter.report(1, 10, "test")
            loop.run_until_complete(asyncio.sleep(0.05))
        finally:
            loop.close()

    def test_multiple_reports(self):
        """Multiple progress reports are sent independently."""
        loop = asyncio.new_event_loop()
        session = MagicMock()
        session.send_progress_notification = AsyncMock()
        reporter = ProgressReporter(loop, session, progress_token="tok-multi")

        try:
            reporter.report(0, 20, "Starting")
            reporter.report(10, 20, "Halfway")
            reporter.report(20, 20, "Done")
            loop.run_until_complete(asyncio.sleep(0.05))
        finally:
            loop.close()

        assert session.send_progress_notification.call_count == 3


class TestProgressInExecution:
    """Integration: verify progress reporter is called during execution."""

    @pytest.fixture(autouse=True)
    def _allow_breaker(self):
        """Ensure circuit breaker allows requests in tests."""
        with patch("agent.tools.comfy_execute.COMFYUI_BREAKER", create=True):
            breaker = MagicMock()
            breaker.allow_request.return_value = True
            with patch("agent.circuit_breaker.get_breaker", return_value=breaker):
                yield

    @pytest.fixture
    def sample_workflow(self, tmp_path):
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

    def test_execute_workflow_reports_progress(self, sample_workflow):
        """execute_workflow sends progress reports via reporter."""
        from agent.tools import comfy_execute

        queue_resp = MagicMock()
        queue_resp.json.return_value = {"prompt_id": "prog1"}
        queue_resp.raise_for_status = MagicMock()

        history_resp = MagicMock()
        history_resp.json.return_value = {
            "prog1": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "2": {"images": [{"filename": "out.png", "subfolder": ""}]},
                },
            },
        }
        history_resp.raise_for_status = MagicMock()

        progress = MagicMock()
        progress.report = MagicMock()

        patcher, mock_client = _mock_httpx_client()
        with patcher:
            mock_client.post.return_value = queue_resp
            mock_client.get.return_value = history_resp

            result = json.loads(comfy_execute.handle(
                "execute_workflow",
                {"path": str(sample_workflow), "timeout": 5},
                progress=progress,
            ))

        assert result["status"] == "complete"
        # Should have reported at least: "Queuing" + "Queued" + "Complete"
        assert progress.report.call_count >= 3
        # Last call should be the completion message
        last_msg = progress.report.call_args_list[-1]
        assert "Complete" in (last_msg.args[2] if len(last_msg.args) > 2
                              else last_msg.kwargs.get("message", ""))

    def test_ws_execute_reports_node_progress(self, sample_workflow):
        """execute_with_progress sends per-node progress via reporter."""
        from agent.tools import comfy_execute

        queue_resp = MagicMock()
        queue_resp.json.return_value = {"prompt_id": "ws_prog"}
        queue_resp.raise_for_status = MagicMock()

        history_resp = MagicMock()
        history_resp.json.return_value = {
            "ws_prog": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "2": {"images": [{"filename": "ws_out.png", "subfolder": ""}]},
                },
            },
        }
        history_resp.raise_for_status = MagicMock()

        ws_messages = [
            json.dumps({"type": "execution_start", "data": {"prompt_id": "ws_prog"}}),
            json.dumps({"type": "executing", "data": {"node": "1", "prompt_id": "ws_prog"}}),
            json.dumps({"type": "executing", "data": {"node": "2", "prompt_id": "ws_prog"}}),
            json.dumps({"type": "progress", "data": {"value": 2, "max": 5, "prompt_id": "ws_prog"}}),
            json.dumps({"type": "progress", "data": {"value": 5, "max": 5, "prompt_id": "ws_prog"}}),
            json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "ws_prog"}}),
        ]
        mock_ws = MagicMock()
        msg_iter = iter(ws_messages)
        mock_ws.recv.side_effect = lambda timeout=None: next(msg_iter)
        mock_ws.__enter__ = MagicMock(return_value=mock_ws)
        mock_ws.__exit__ = MagicMock(return_value=False)

        progress = MagicMock()
        progress.report = MagicMock()

        patcher, mock_client = _mock_httpx_client()
        mock_client.post.return_value = queue_resp
        mock_client.get.return_value = history_resp
        with patch.object(comfy_execute, "_HAS_WS", True), \
             patcher, \
             patch("agent.tools.comfy_execute.websockets.sync.client.connect", return_value=mock_ws):

            result = json.loads(comfy_execute.handle(
                "execute_with_progress",
                {"path": str(sample_workflow), "timeout": 30},
                progress=progress,
            ))

        assert result["status"] == "complete"
        # Expect: Queued + start + node1 + node2 + 2 sampler steps + complete
        assert progress.report.call_count >= 5

        # Verify node names appear in messages
        messages = [
            call.args[2] if len(call.args) > 2 else call.kwargs.get("message", "")
            for call in progress.report.call_args_list
        ]
        assert any("CheckpointLoaderSimple" in m for m in messages)
        assert any("KSampler" in m for m in messages)
        assert any("step 2/5" in m for m in messages)
        assert any("Complete" in m for m in messages)

    def test_execute_no_progress_still_works(self, sample_workflow):
        """Execution works fine when no progress reporter is passed."""
        from agent.tools import comfy_execute

        queue_resp = MagicMock()
        queue_resp.json.return_value = {"prompt_id": "noprog"}
        queue_resp.raise_for_status = MagicMock()

        history_resp = MagicMock()
        history_resp.json.return_value = {
            "noprog": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "2": {"images": [{"filename": "out.png", "subfolder": ""}]},
                },
            },
        }
        history_resp.raise_for_status = MagicMock()

        patcher, mock_client = _mock_httpx_client()
        with patcher:
            mock_client.post.return_value = queue_resp
            mock_client.get.return_value = history_resp

            # No progress kwarg — should use noop internally
            result = json.loads(comfy_execute.handle(
                "execute_workflow",
                {"path": str(sample_workflow), "timeout": 5},
            ))

        assert result["status"] == "complete"

    def test_ws_error_reports_progress(self, sample_workflow):
        """Execution error is reported via progress."""
        from agent.tools import comfy_execute

        queue_resp = MagicMock()
        queue_resp.json.return_value = {"prompt_id": "ws_err"}
        queue_resp.raise_for_status = MagicMock()

        ws_messages = [
            json.dumps({"type": "execution_start", "data": {"prompt_id": "ws_err"}}),
            json.dumps({"type": "executing", "data": {"node": "1", "prompt_id": "ws_err"}}),
            json.dumps({
                "type": "execution_error",
                "data": {
                    "prompt_id": "ws_err",
                    "node_id": "1",
                    "node_type": "CheckpointLoaderSimple",
                    "exception_message": "Model not found",
                },
            }),
        ]
        mock_ws = MagicMock()
        msg_iter = iter(ws_messages)
        mock_ws.recv.side_effect = lambda timeout=None: next(msg_iter)
        mock_ws.__enter__ = MagicMock(return_value=mock_ws)
        mock_ws.__exit__ = MagicMock(return_value=False)

        progress = MagicMock()
        progress.report = MagicMock()

        with patch.object(comfy_execute, "_HAS_WS", True), \
             patch("agent.tools.comfy_execute.httpx.Client") as mock_http, \
             patch("agent.tools.comfy_execute.websockets.sync.client.connect", return_value=mock_ws):
            mock_client = mock_http.return_value.__enter__.return_value
            mock_client.post.return_value = queue_resp

            result = json.loads(comfy_execute.handle(
                "execute_with_progress",
                {"path": str(sample_workflow), "timeout": 30},
                progress=progress,
            ))

        assert result["status"] == "error"
        assert "Model not found" in result["error"]
        # Should have reported the error node
        messages = [
            call.args[2] if len(call.args) > 2 else call.kwargs.get("message", "")
            for call in progress.report.call_args_list
        ]
        assert any("Error" in m for m in messages)
