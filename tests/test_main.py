"""Tests for the agent loop in main.py.

Covers: token estimation, context compaction, observation masking,
streaming with retry, and run_agent_turn.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.main import (
    _estimate_tokens,
    _summarize_dropped,
    _compact_messages,
    _mask_processed_results,
    _stream_with_retry,
    run_agent_turn,
)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty(self):
        assert _estimate_tokens([]) == 0

    def test_string_content(self):
        msgs = [{"content": "hello world"}]
        assert _estimate_tokens(msgs) == len("hello world") // 4

    def test_list_content_with_dicts(self):
        msgs = [{"content": [{"content": "x" * 100}]}]
        assert _estimate_tokens(msgs) == 25

    def test_list_content_with_text_attr(self):
        block = SimpleNamespace(text="y" * 80)
        msgs = [{"content": [block]}]
        assert _estimate_tokens(msgs) == 20

    def test_list_content_with_input_attr(self):
        block = SimpleNamespace(input={"key": "value"})
        msgs = [{"content": [block]}]
        # str({"key": "value"}) -> "{'key': 'value'}" = 16 chars -> 4 tokens
        assert _estimate_tokens(msgs) >= 3

    def test_missing_content_key(self):
        msgs = [{"role": "user"}]
        assert _estimate_tokens(msgs) == 0

    def test_multiple_messages(self):
        msgs = [
            {"content": "a" * 40},
            {"content": "b" * 80},
        ]
        assert _estimate_tokens(msgs) == 10 + 20


# ---------------------------------------------------------------------------
# Summarize dropped messages
# ---------------------------------------------------------------------------


class TestSummarizeDropped:
    def test_empty(self):
        result = _summarize_dropped([])
        assert "[Context Summary" in result
        assert "Recent conversation follows" in result

    def test_user_requests(self):
        msgs = [
            {"role": "user", "content": "Build me an SDXL workflow"},
            {"role": "user", "content": "Now add ControlNet"},
        ]
        result = _summarize_dropped(msgs)
        assert "SDXL" in result
        assert "ControlNet" in result

    def test_tool_calls_extracted(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    SimpleNamespace(type="tool_use", name="load_workflow"),
                    SimpleNamespace(type="tool_use", name="set_input"),
                ],
            },
        ]
        result = _summarize_dropped(msgs)
        assert "load_workflow" in result
        assert "set_input" in result

    def test_tool_calls_deduped(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    SimpleNamespace(type="tool_use", name="set_input"),
                    SimpleNamespace(type="tool_use", name="set_input"),
                ],
            },
        ]
        result = _summarize_dropped(msgs)
        # Should only appear once
        assert result.count("set_input") == 1

    def test_workflow_context_extracted(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": json.dumps({"loaded_path": "/test/wf.json"}),
                    },
                ],
            },
        ]
        result = _summarize_dropped(msgs)
        assert "wf.json" in result

    def test_skips_system_summaries(self):
        msgs = [{"role": "user", "content": "[Context Summary - earlier]"}]
        result = _summarize_dropped(msgs)
        assert "Context Summary - earlier" not in result.split("\n")[1:]


# ---------------------------------------------------------------------------
# Context compaction
# ---------------------------------------------------------------------------


class TestCompactMessages:
    def test_under_threshold(self):
        msgs = [{"role": "user", "content": "short"}]
        result = _compact_messages(msgs, 100_000)
        assert result == msgs

    def test_truncates_large_tool_results(self):
        big = "x" * 5000
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": big},
                ],
            },
        ]
        result = _compact_messages(msgs, 1)  # Very low threshold
        content = result[0]["content"][0]["content"]
        assert len(content) < len(big)
        assert "[...truncated]" in content

    def test_drops_old_messages_with_summary(self):
        msgs = [
            {"role": "user", "content": f"msg {i}" * 100}
            for i in range(20)
        ]
        result = _compact_messages(msgs, 1)
        assert len(result) <= 7  # summary + 6 recent
        assert "[Context Summary" in result[0]["content"]


# ---------------------------------------------------------------------------
# Observation masking
# ---------------------------------------------------------------------------


class TestMaskProcessedResults:
    def test_short_messages_untouched(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert _mask_processed_results(msgs) == msgs

    def test_recent_results_not_masked(self):
        """Most recent tool results should be preserved intact."""
        big = "x" * 5000
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "content": big}]},
        ]
        result = _mask_processed_results(msgs)
        assert result[0]["content"][0]["content"] == big

    def test_old_results_masked(self):
        """Results from earlier turns (followed by assistant) get masked."""
        big = "x" * 5000
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "content": big}]},
            {"role": "assistant", "content": "I see the results."},
            {"role": "user", "content": [{"type": "tool_result", "content": "recent"}]},
        ]
        result = _mask_processed_results(msgs)
        # First result should be masked
        assert "[Processed result" in result[0]["content"][0]["content"]
        # Last result should be intact
        assert result[2]["content"][0]["content"] == "recent"

    def test_small_results_not_masked(self):
        """Results under threshold are kept even in old turns."""
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "content": "short"}]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "tool_result", "content": "recent"}]},
        ]
        result = _mask_processed_results(msgs)
        assert result[0]["content"][0]["content"] == "short"


# ---------------------------------------------------------------------------
# Streaming with retry
# ---------------------------------------------------------------------------


class TestStreamWithRetry:
    def test_success_first_try(self):
        mock_msg = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hello")],
            stop_reason="end_turn",
        )
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.__iter__ = MagicMock(return_value=iter([]))
        mock_stream.get_final_message.return_value = mock_msg

        client = MagicMock()
        client.messages.stream.return_value = mock_stream

        result = _stream_with_retry(
            client,
            model="test",
            max_tokens=100,
            system="sys",
            tools=[],
            messages=[],
        )
        assert result == mock_msg

    def test_rate_limit_retry(self):
        import anthropic

        mock_msg = SimpleNamespace(content=[], stop_reason="end_turn")
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.__iter__ = MagicMock(return_value=iter([]))
        mock_stream.get_final_message.return_value = mock_msg

        client = MagicMock()
        # Fail first, succeed second
        rate_err = anthropic.RateLimitError(
            message="Rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
        client.messages.stream.side_effect = [rate_err, mock_stream]

        with patch("agent.main.API_RETRY_DELAY", 0.01):
            result = _stream_with_retry(
                client,
                model="test",
                max_tokens=100,
                system="sys",
                tools=[],
                messages=[],
            )
        assert result == mock_msg
        assert client.messages.stream.call_count == 2


# ---------------------------------------------------------------------------
# run_agent_turn
# ---------------------------------------------------------------------------


class TestRunAgentTurn:
    def _make_response(self, *, text_blocks=None, tool_blocks=None):
        """Helper to create a mock API response."""
        content = []
        if text_blocks:
            for t in text_blocks:
                content.append(SimpleNamespace(type="text", text=t))
        if tool_blocks:
            for name, inp, tid in tool_blocks:
                content.append(SimpleNamespace(
                    type="tool_use", name=name, input=inp, id=tid,
                ))
        return SimpleNamespace(
            content=content,
            stop_reason="end_turn" if not tool_blocks else "tool_use",
        )

    @patch("agent.main._stream_with_retry")
    def test_text_response_marks_done(self, mock_stream):
        mock_stream.return_value = self._make_response(text_blocks=["Hello!"])

        client = MagicMock()
        messages = [{"role": "user", "content": "Hi"}]
        msgs, done = run_agent_turn(client, messages, "system")
        assert done is True
        assert msgs[-1]["role"] == "assistant"

    @patch("agent.main.handle_tool")
    @patch("agent.main._stream_with_retry")
    def test_single_tool_call(self, mock_stream, mock_handle):
        mock_stream.return_value = self._make_response(
            tool_blocks=[("is_comfyui_running", {}, "tool_1")],
        )
        mock_handle.return_value = '{"running": true}'

        client = MagicMock()
        messages = [{"role": "user", "content": "Check ComfyUI"}]
        msgs, done = run_agent_turn(client, messages, "system")
        assert done is False
        # Should have assistant message + tool result
        assert msgs[-2]["role"] == "assistant"
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"][0]["type"] == "tool_result"
        mock_handle.assert_called_once_with("is_comfyui_running", {})

    @patch("agent.main.handle_tool")
    @patch("agent.main._stream_with_retry")
    def test_parallel_tool_calls(self, mock_stream, mock_handle):
        mock_stream.return_value = self._make_response(
            tool_blocks=[
                ("get_all_nodes", {}, "t1"),
                ("get_system_stats", {}, "t2"),
            ],
        )
        mock_handle.side_effect = [
            '{"nodes": []}',
            '{"stats": {}}',
        ]

        client = MagicMock()
        messages = [{"role": "user", "content": "Get info"}]
        msgs, done = run_agent_turn(client, messages, "system")
        assert done is False
        results = msgs[-1]["content"]
        assert len(results) == 2
        # Results should be in original order (t1 then t2)
        assert results[0]["tool_use_id"] == "t1"
        assert results[1]["tool_use_id"] == "t2"

    @patch("agent.main._stream_with_retry")
    def test_callbacks_called(self, mock_stream):
        mock_stream.return_value = self._make_response(text_blocks=["Done"])

        on_stream_end = MagicMock()
        client = MagicMock()
        messages = [{"role": "user", "content": "Hi"}]
        run_agent_turn(
            client, messages, "system",
            on_stream_end=on_stream_end,
        )
        on_stream_end.assert_called_once()

    @patch("agent.main.handle_tool")
    @patch("agent.main._stream_with_retry")
    def test_tool_call_callback(self, mock_stream, mock_handle):
        mock_stream.return_value = self._make_response(
            tool_blocks=[("plan_goal", {"goal": "test"}, "t1")],
        )
        mock_handle.return_value = '{"planned": true}'

        on_tool_call = MagicMock()
        client = MagicMock()
        messages = [{"role": "user", "content": "Plan"}]
        run_agent_turn(
            client, messages, "system",
            on_tool_call=on_tool_call,
        )
        on_tool_call.assert_called_once_with("plan_goal", {"goal": "test"})
