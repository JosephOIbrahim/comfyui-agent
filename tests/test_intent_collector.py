"""Tests for the intent_collector brain module."""

import json
import threading

import pytest

from agent.brain.intent_collector import (
    IntentCollectorAgent,
    TOOLS,
    _singleton_lock,
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
def reset_singleton(monkeypatch):
    """Reset module-level singleton between tests."""
    import agent.brain.intent_collector as mod
    monkeypatch.setattr(mod, "_singleton", None)


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
