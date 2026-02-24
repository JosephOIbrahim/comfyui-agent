"""Tests for brain-to-brain communication protocol."""

import json
import time
from unittest.mock import patch, MagicMock

from agent.brain._protocol import (
    make_id,
    brain_message,
    serialize,
    dispatch_brain_message,
)


class TestMakeId:
    def test_returns_12_char_hex(self):
        result = make_id()
        assert len(result) == 12
        # Must be valid hex characters
        int(result, 16)

    def test_unique_ids(self):
        ids = {make_id() for _ in range(100)}
        assert len(ids) == 100


class TestBrainMessage:
    def test_creates_message_with_all_fields(self):
        msg = brain_message(
            source="vision",
            target="memory",
            msg_type="result",
            payload={"score": 0.9},
            correlation_id="abc123def456",
        )
        assert msg["source"] == "vision"
        assert msg["target"] == "memory"
        assert msg["msg_type"] == "result"
        assert msg["payload"] == {"score": 0.9}
        assert msg["correlation_id"] == "abc123def456"
        assert "timestamp" in msg

    def test_auto_generates_correlation_id(self):
        msg = brain_message("a", "b", "request", {})
        cid = msg["correlation_id"]
        assert len(cid) == 12
        int(cid, 16)  # valid hex

    def test_uses_provided_correlation_id(self):
        msg = brain_message("a", "b", "request", {}, correlation_id="my-custom-id")
        assert msg["correlation_id"] == "my-custom-id"

    def test_timestamp_is_recent(self):
        before = time.time()
        msg = brain_message("a", "b", "request", {})
        after = time.time()
        assert before <= msg["timestamp"] <= after


class TestSerialize:
    def test_deterministic_json(self):
        msg = brain_message("vision", "memory", "result", {"z": 1, "a": 2})
        s1 = serialize(msg)
        s2 = serialize(msg)
        assert s1 == s2

    def test_sort_keys(self):
        msg = {"z_field": 1, "a_field": 2, "m_field": 3}
        result = serialize(msg)
        parsed = json.loads(result)
        keys = list(parsed.keys())
        assert keys == sorted(keys)


class TestDispatchBrainMessage:
    @patch("agent.brain._protocol.time.sleep")
    def test_vision_to_memory_success(self, mock_sleep):
        msg = brain_message("vision", "memory", "result", {
            "action": "analyze_image",
            "quality_score": 0.85,
        })
        with patch("agent.tools.handle") as mock_handle:
            result = dispatch_brain_message(msg)
        assert result is True
        mock_handle.assert_called_once()
        call_args = mock_handle.call_args
        assert call_args[0][0] == "record_outcome"
        outcome = call_args[0][1]
        assert outcome["action"] == "analyze_image"
        assert outcome["session"] == "default"
        assert outcome["result"] == "success"
        mock_sleep.assert_not_called()

    def test_vision_to_memory_returns_true(self):
        msg = brain_message("vision", "memory", "result", {"action": "test"})
        with patch("agent.tools.handle"):
            assert dispatch_brain_message(msg) is True

    def test_unknown_route_returns_true(self):
        msg = brain_message("planner", "optimizer", "request", {"goal": "speed"})
        result = dispatch_brain_message(msg)
        assert result is True

    @patch("agent.brain._protocol.time.sleep")
    def test_retry_on_failure(self, mock_sleep):
        msg = brain_message("vision", "memory", "result", {"action": "retry_test"})
        mock_handle = MagicMock(side_effect=[RuntimeError("fail"), RuntimeError("fail"), "ok"])
        with patch("agent.tools.handle", mock_handle):
            result = dispatch_brain_message(msg)
        assert result is True
        assert mock_handle.call_count == 3
        # Should have slept twice (before retry 2 and 3)
        assert mock_sleep.call_count == 2

    @patch("agent.brain._protocol.time.sleep")
    def test_retry_exhausted_returns_false(self, mock_sleep):
        msg = brain_message("vision", "memory", "result", {"action": "always_fail"})
        mock_handle = MagicMock(side_effect=RuntimeError("persistent failure"))
        with patch("agent.tools.handle", mock_handle):
            result = dispatch_brain_message(msg)
        assert result is False
        assert mock_handle.call_count == 3

    @patch("agent.brain._protocol.time.sleep")
    def test_max_retries_parameter(self, mock_sleep):
        msg = brain_message("vision", "memory", "result", {"action": "one_try"})
        mock_handle = MagicMock(side_effect=RuntimeError("fail"))
        with patch("agent.tools.handle", mock_handle):
            result = dispatch_brain_message(msg, max_retries=1)
        assert result is False
        assert mock_handle.call_count == 1
        mock_sleep.assert_not_called()

    def test_payload_forwarded_correctly(self):
        msg = brain_message("vision", "memory", "result", {
            "action": "compare_outputs",
            "similarity": 0.92,
            "diff_regions": ["top-left"],
        })
        with patch("agent.tools.handle") as mock_handle:
            dispatch_brain_message(msg)
        outcome = mock_handle.call_args[0][1]
        assert outcome["action"] == "compare_outputs"
        details = outcome["details"]
        # "action" excluded from details, rest present with sorted keys
        assert "action" not in details
        assert details["diff_regions"] == ["top-left"]
        assert details["similarity"] == 0.92

    def test_empty_message_handled(self):
        # Empty dict should not crash — falls through to no-route path
        result = dispatch_brain_message({})
        assert result is True

    @patch("agent.brain._protocol.time.sleep")
    def test_backoff_delays(self, mock_sleep):
        """Verify exponential backoff timing: 0.1s, 0.2s."""
        msg = brain_message("vision", "memory", "result", {"action": "backoff_test"})
        mock_handle = MagicMock(
            side_effect=[RuntimeError("1"), RuntimeError("2"), "ok"],
        )
        with patch("agent.tools.handle", mock_handle):
            dispatch_brain_message(msg)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0.1)
        mock_sleep.assert_any_call(0.2)
