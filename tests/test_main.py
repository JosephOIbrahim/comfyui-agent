"""Tests for the agent loop in main.py.

Covers: token estimation, context compaction, observation masking,
streaming with retry, and run_agent_turn.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.context import (
    estimate_tokens,
    summarize_dropped,
    compact,
    mask_processed_results,
)
from agent.llm import LLMResponse, LLMRateLimitError, TextBlock, ToolUseBlock, ToolResultBlock
from agent.main import (
    _shutdown,
    _stream_with_retry,
    run_agent_turn,
)
from agent.streaming import NullHandler


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens([]) == 0

    def test_string_content(self):
        msgs = [{"content": "hello world"}]
        assert estimate_tokens(msgs) == len("hello world") // 4

    def test_list_content_with_dicts(self):
        msgs = [{"content": [{"content": "x" * 100}]}]
        assert estimate_tokens(msgs) == 25

    def test_list_content_with_text_attr(self):
        block = SimpleNamespace(text="y" * 80)
        msgs = [{"content": [block]}]
        assert estimate_tokens(msgs) == 20

    def test_list_content_with_input_attr(self):
        block = SimpleNamespace(input={"key": "value"})
        msgs = [{"content": [block]}]
        # str({"key": "value"}) -> "{'key': 'value'}" = 16 chars -> 4 tokens
        assert estimate_tokens(msgs) >= 3

    def test_missing_content_key(self):
        msgs = [{"role": "user"}]
        assert estimate_tokens(msgs) == 0

    def test_multiple_messages(self):
        msgs = [
            {"content": "a" * 40},
            {"content": "b" * 80},
        ]
        assert estimate_tokens(msgs) == 10 + 20


# ---------------------------------------------------------------------------
# Summarize dropped messages
# ---------------------------------------------------------------------------


class TestSummarizeDropped:
    def test_empty(self):
        result = summarize_dropped([])
        assert "[Context Summary" in result
        assert "Recent conversation follows" in result

    def test_user_requests(self):
        msgs = [
            {"role": "user", "content": "Build me an SDXL workflow"},
            {"role": "user", "content": "Now add ControlNet"},
        ]
        result = summarize_dropped(msgs)
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
        result = summarize_dropped(msgs)
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
        result = summarize_dropped(msgs)
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
        result = summarize_dropped(msgs)
        assert "wf.json" in result

    def test_skips_system_summaries(self):
        msgs = [{"role": "user", "content": "[Context Summary - earlier]"}]
        result = summarize_dropped(msgs)
        assert "Context Summary - earlier" not in result.split("\n")[1:]


# ---------------------------------------------------------------------------
# Context compaction
# ---------------------------------------------------------------------------


class TestCompactMessages:
    def test_under_threshold(self):
        msgs = [{"role": "user", "content": "short"}]
        result = compact(msgs, 100_000)
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
        result = compact(msgs, 1)  # Very low threshold
        content = result[0]["content"][0]["content"]
        assert len(content) < len(big)
        assert "[...truncated]" in content

    def test_drops_old_messages_with_summary(self):
        msgs = [
            {"role": "user", "content": f"msg {i}" * 100}
            for i in range(20)
        ]
        result = compact(msgs, 1)
        assert len(result) <= 7  # summary + 6 recent
        assert "[Context Summary" in result[0]["content"]


# ---------------------------------------------------------------------------
# Observation masking
# ---------------------------------------------------------------------------


class TestMaskProcessedResults:
    def test_short_messages_untouched(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert mask_processed_results(msgs) == msgs

    def test_recent_results_not_masked(self):
        """Most recent tool results should be preserved intact."""
        big = "x" * 5000
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "content": big}]},
        ]
        result = mask_processed_results(msgs)
        assert result[0]["content"][0]["content"] == big

    def test_old_results_masked(self):
        """Results from earlier turns (followed by assistant) get masked."""
        big = "x" * 5000
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "content": big}]},
            {"role": "assistant", "content": "I see the results."},
            {"role": "user", "content": [{"type": "tool_result", "content": "recent"}]},
        ]
        result = mask_processed_results(msgs)
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
        result = mask_processed_results(msgs)
        assert result[0]["content"][0]["content"] == "short"


# ---------------------------------------------------------------------------
# Streaming with retry
# ---------------------------------------------------------------------------


class TestStreamWithRetry:
    def test_success_first_try(self):
        expected = LLMResponse(
            content=[TextBlock(text="hello")],
            stop_reason="end_turn",
        )
        provider = MagicMock()
        provider.stream.return_value = expected

        result = _stream_with_retry(
            provider,
            model="test",
            max_tokens=100,
            system="sys",
            tools=[],
            messages=[],
        )
        assert result == expected

    def test_rate_limit_retry(self):
        expected = LLMResponse(content=[], stop_reason="end_turn")
        provider = MagicMock()
        # Fail first, succeed second
        provider.stream.side_effect = [
            LLMRateLimitError("Rate limited"),
            expected,
        ]

        with patch("agent.main.API_RETRY_DELAY", 0.01):
            result = _stream_with_retry(
                provider,
                model="test",
                max_tokens=100,
                system="sys",
                tools=[],
                messages=[],
            )
        assert result == expected
        assert provider.stream.call_count == 2


# ---------------------------------------------------------------------------
# run_agent_turn
# ---------------------------------------------------------------------------


class TestRunAgentTurn:
    def _make_response(self, *, text_blocks=None, tool_blocks=None):
        """Helper to create a mock LLMResponse."""
        content = []
        if text_blocks:
            for t in text_blocks:
                content.append(TextBlock(text=t))
        if tool_blocks:
            for name, inp, tid in tool_blocks:
                content.append(ToolUseBlock(id=tid, name=name, input=inp))
        return LLMResponse(
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
        assert isinstance(msgs[-1]["content"][0], ToolResultBlock)
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
        assert results[0].tool_use_id == "t1"
        assert results[1].tool_use_id == "t2"

    @patch("agent.main._stream_with_retry")
    def test_handler_on_stream_end_called(self, mock_stream):
        mock_stream.return_value = self._make_response(text_blocks=["Done"])

        handler = MagicMock(spec=NullHandler)
        client = MagicMock()
        messages = [{"role": "user", "content": "Hi"}]
        run_agent_turn(client, messages, "system", handler=handler)
        handler.on_stream_end.assert_called_once()

    @patch("agent.main.handle_tool")
    @patch("agent.main._stream_with_retry")
    def test_handler_on_tool_call(self, mock_stream, mock_handle):
        mock_stream.return_value = self._make_response(
            tool_blocks=[("plan_goal", {"goal": "test"}, "t1")],
        )
        mock_handle.return_value = '{"planned": true}'

        handler = MagicMock(spec=NullHandler)
        client = MagicMock()
        messages = [{"role": "user", "content": "Plan"}]
        run_agent_turn(client, messages, "system", handler=handler)
        handler.on_tool_call.assert_called_once_with("plan_goal", {"goal": "test"})

    @patch("agent.main._stream_with_retry")
    def test_shutdown_flag_stops_turn(self, mock_stream):
        """When shutdown is requested, run_agent_turn returns done=True immediately."""
        _shutdown.set()
        try:
            client = MagicMock()
            messages = [{"role": "user", "content": "Hi"}]
            msgs, done = run_agent_turn(client, messages, "system")
            assert done is True
            # Stream should NOT have been called
            mock_stream.assert_not_called()
        finally:
            _shutdown.clear()
