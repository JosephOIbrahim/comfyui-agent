"""Tests for context management — token estimation and message compaction."""

from agent.main import _estimate_tokens, _compact_messages


class TestTokenEstimation:
    def test_simple_string(self):
        messages = [{"role": "user", "content": "hello world"}]
        tokens = _estimate_tokens(messages)
        assert tokens == len("hello world") // 4

    def test_empty_messages(self):
        assert _estimate_tokens([]) == 0

    def test_tool_result(self):
        messages = [{
            "role": "user",
            "content": [{"type": "tool_result", "content": "x" * 1000}],
        }]
        tokens = _estimate_tokens(messages)
        assert tokens == 250  # 1000 / 4

    def test_mixed_content(self):
        messages = [
            {"role": "user", "content": "a" * 400},
            {"role": "user", "content": [{"type": "tool_result", "content": "b" * 800}]},
        ]
        tokens = _estimate_tokens(messages)
        assert tokens == 300  # 100 + 200

    def test_multiple_blocks(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "tool_result", "content": "a" * 400},
                {"type": "tool_result", "content": "b" * 400},
            ],
        }]
        tokens = _estimate_tokens(messages)
        assert tokens == 200  # 100 + 100


class TestCompaction:
    def test_no_compaction_needed(self):
        messages = [{"role": "user", "content": "short"}]
        result = _compact_messages(messages, threshold=1000)
        assert result is messages  # Same object — no copy

    def test_truncates_large_tool_results(self):
        large_content = "x" * 10000
        messages = [{
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "1", "content": large_content}],
        }]
        result = _compact_messages(messages, threshold=100)
        block = result[0]["content"][0]
        assert len(block["content"]) < len(large_content)
        assert "[...truncated]" in block["content"]

    def test_preserves_small_tool_results(self):
        messages = [{
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "1", "content": "small"}],
        }]
        result = _compact_messages(messages, threshold=0)  # Force compaction
        block = result[0]["content"][0]
        assert block["content"] == "small"  # Not truncated

    def test_drops_old_messages(self):
        # Create many messages that exceed threshold
        messages = [
            {"role": "user", "content": "x" * 4000}  # ~1000 tokens each
            for _ in range(20)
        ]
        result = _compact_messages(messages, threshold=2000)
        # Should keep summary + recent 6
        assert len(result) <= 7
        assert "Context Summary" in result[0]["content"]

    def test_preserves_recent_messages(self):
        messages = [
            {"role": "user", "content": f"msg_{i}" + "x" * 4000}
            for i in range(20)
        ]
        result = _compact_messages(messages, threshold=2000)
        # Last message should be preserved
        last = result[-1]["content"]
        assert "msg_19" in last

    def test_summary_message_is_user_role(self):
        messages = [
            {"role": "user", "content": "x" * 4000}
            for _ in range(20)
        ]
        result = _compact_messages(messages, threshold=2000)
        assert result[0]["role"] == "user"

    def test_pass1_sufficient(self):
        """If truncating tool results is enough, don't drop messages."""
        messages = [
            {"role": "user", "content": "short question"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "1", "content": "x" * 8000}],
            },
            {"role": "user", "content": "another question"},
        ]
        # Threshold that pass1 (truncation) can satisfy
        result = _compact_messages(messages, threshold=1500)
        # All 3 messages preserved (no dropping)
        assert len(result) == 3
        # But tool result was truncated
        block = result[1]["content"][0]
        assert "[...truncated]" in block["content"]

    def test_does_not_mutate_original(self):
        large_content = "x" * 10000
        original_block = {"type": "tool_result", "tool_use_id": "1", "content": large_content}
        messages = [{"role": "user", "content": [original_block]}]
        _compact_messages(messages, threshold=100)
        # Original block should be unchanged
        assert original_block["content"] == large_content
