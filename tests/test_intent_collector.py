"""Tests for the intent_collector brain module."""

import json
import threading

import pytest

from agent.brain.intent_collector import (
    IntentCollectorAgent,
    TOOLS,
)
from agent.brain._sdk import BrainConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    """Fresh IntentCollectorAgent for each test."""
    return IntentCollectorAgent(config=BrainConfig())


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset registered IntentCollectorAgent state between tests."""
    from agent.brain._sdk import BrainAgent
    BrainAgent._register_all()
    instance = BrainAgent._registry.get("capture_intent")
    if instance is not None:
        # Cycle 35: storage is now per-session dicts
        instance._intents.clear()
        instance._histories.clear()
    yield


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

class TestToolSchemas:
    def test_tool_count(self):
        assert len(TOOLS) == 2

    def test_tool_names(self):
        names = {t["name"] for t in TOOLS}
        assert names == {"capture_intent", "get_current_intent"}


# ---------------------------------------------------------------------------
# Direct API
# ---------------------------------------------------------------------------

class TestDirectAPI:
    def test_capture_and_get(self, agent):
        intent = agent.capture(
            user_request="Make it dreamier",
            interpretation="Lower CFG to 5",
            style_references=["Studio Ghibli"],
            session_context="anime portraits session",
        )
        assert intent["user_request"] == "Make it dreamier"
        assert intent["interpretation"] == "Lower CFG to 5"
        assert "captured_at" in intent

        current = agent.get_current()
        assert current is not None
        assert current["user_request"] == "Make it dreamier"

    def test_get_empty(self, agent):
        assert agent.get_current() is None

    def test_clear(self, agent):
        agent.capture("test", "test")
        assert agent.get_current() is not None
        agent.clear()
        assert agent.get_current() is None

    def test_history_accumulates(self, agent):
        agent.capture("first", "first interp")
        agent.capture("second", "second interp")
        agent.capture("third", "third interp")

        history = agent.get_history()
        assert len(history) == 3
        assert history[0]["user_request"] == "first"
        assert history[2]["user_request"] == "third"

    def test_latest_overwrites_current(self, agent):
        agent.capture("first", "first interp")
        agent.capture("second", "second interp")

        current = agent.get_current()
        assert current["user_request"] == "second"

    def test_default_style_refs(self, agent):
        intent = agent.capture("test", "test")
        assert intent["style_references"] == []

    def test_default_session_context(self, agent):
        intent = agent.capture("test", "test")
        assert intent["session_context"] == ""


# ---------------------------------------------------------------------------
# Tool handle()
# ---------------------------------------------------------------------------

class TestToolHandle:
    def test_capture_intent_tool(self, agent):
        result = json.loads(agent.handle("capture_intent", {
            "user_request": "sharper edges",
            "interpretation": "increase CFG to 10",
            "style_references": ["cyberpunk"],
            "session_context": "sci-fi series",
        }))
        assert result["status"] == "captured"
        assert result["intent"]["user_request"] == "sharper edges"
        assert result["history_count"] == 1

    def test_get_current_intent_empty(self, agent):
        result = json.loads(agent.handle("get_current_intent", {}))
        assert result["status"] == "empty"
        assert result["intent"] is None

    def test_get_current_intent_after_capture(self, agent):
        agent.handle("capture_intent", {
            "user_request": "warm tones",
            "interpretation": "shift color temperature",
        })
        result = json.loads(agent.handle("get_current_intent", {}))
        assert result["status"] == "ok"
        assert result["intent"]["user_request"] == "warm tones"

    def test_unknown_tool(self, agent):
        result = json.loads(agent.handle("nonexistent", {}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Thread Safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_captures(self, agent):
        """Multiple threads capturing concurrently should not corrupt state."""
        errors = []

        def capture_n(n):
            try:
                for i in range(10):
                    agent.capture(f"thread-{n}-req-{i}", f"interp-{n}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=capture_n, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(agent.get_history()) == 40


# ---------------------------------------------------------------------------
# Module-level dispatch
# ---------------------------------------------------------------------------

class TestModuleDispatch:
    def test_module_handle(self):
        from agent.brain.intent_collector import handle
        result = json.loads(handle("get_current_intent", {}))
        assert result["status"] == "empty"


# ---------------------------------------------------------------------------
# Cycle 35: per-session isolation — two sessions must not overwrite each other
# ---------------------------------------------------------------------------

class TestSessionIsolation:
    """capture_intent must isolate intent per MCP session; no cross-contamination."""

    def test_two_sessions_isolated(self):
        """Intents captured under different session keys must not overwrite each other."""
        from agent._conn_ctx import _conn_session
        from agent.brain._sdk import BrainAgent

        BrainAgent._register_all()
        instance = BrainAgent._registry.get("capture_intent")
        assert instance is not None

        # Simulate two concurrent connections using the ContextVar

        token_a = _conn_session.set("conn_session_a")
        instance.capture("Session A request", "Lower CFG", session_context="a")
        _conn_session.reset(token_a)

        token_b = _conn_session.set("conn_session_b")
        instance.capture("Session B request", "Higher steps", session_context="b")
        _conn_session.reset(token_b)

        # Now verify each session sees only its own intent
        token_a = _conn_session.set("conn_session_a")
        intent_a = instance.get_current()
        _conn_session.reset(token_a)

        token_b = _conn_session.set("conn_session_b")
        intent_b = instance.get_current()
        _conn_session.reset(token_b)

        assert intent_a is not None
        assert intent_b is not None
        assert intent_a["user_request"] == "Session A request"
        assert intent_b["user_request"] == "Session B request"
        assert intent_a["user_request"] != intent_b["user_request"]

    def test_clear_only_clears_current_session(self):
        """clear() must only clear the intent for the calling session."""
        from agent._conn_ctx import _conn_session
        from agent.brain._sdk import BrainAgent

        BrainAgent._register_all()
        instance = BrainAgent._registry.get("capture_intent")

        token_a = _conn_session.set("conn_clear_a")
        instance.capture("A request", "A interp")
        _conn_session.reset(token_a)

        token_b = _conn_session.set("conn_clear_b")
        instance.capture("B request", "B interp")
        instance.clear()  # Only clears session B's intent
        assert instance.get_current() is None  # B is cleared
        _conn_session.reset(token_b)

        # Session A should be untouched
        token_a = _conn_session.set("conn_clear_a")
        assert instance.get_current() is not None
        assert instance.get_current()["user_request"] == "A request"
        _conn_session.reset(token_a)


# ---------------------------------------------------------------------------
# Cycle 70: style_references isinstance(list) guard
# ---------------------------------------------------------------------------

class TestStyleReferencesGuardCycle70:
    """Cycle 70: style_references must be validated as list before passing to capture()."""

    def test_string_style_references_returns_error(self):
        """style_references='some_style' (string) must return structured error."""
        from agent.brain import handle
        result = json.loads(handle("capture_intent", {
            "user_request": "make it dreamier",
            "interpretation": "lower CFG, dreamier palette",
            "style_references": "baroque",  # string instead of list — Cycle 70 guard
        }))
        assert "error" in result
        assert "style_references" in result["error"].lower()

    def test_dict_style_references_returns_error(self):
        """style_references={} (dict) must return structured error, not silent corruption."""
        from agent.brain import handle
        result = json.loads(handle("capture_intent", {
            "user_request": "make it dreamier",
            "interpretation": "lower CFG",
            "style_references": {"key": "value"},  # dict instead of list — Cycle 70 guard
        }))
        assert "error" in result
        assert "style_references" in result["error"].lower()

    def test_valid_list_style_references_captured(self):
        """Well-formed list style_references must pass through and be stored."""
        from agent.brain import handle
        result = json.loads(handle("capture_intent", {
            "user_request": "make it dreamier",
            "interpretation": "lower CFG",
            "style_references": ["baroque", "impressionist"],  # valid list — Cycle 70
        }))
        assert result.get("status") == "captured"
        assert result["intent"]["style_references"] == ["baroque", "impressionist"]

    def test_empty_list_style_references_captured(self):
        """Empty list style_references must be accepted (optional field)."""
        from agent.brain import handle
        result = json.loads(handle("capture_intent", {
            "user_request": "make it dreamier",
            "interpretation": "lower CFG",
            "style_references": [],  # empty list — valid
        }))
        assert result.get("status") == "captured"
