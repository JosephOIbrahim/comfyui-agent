"""Tests for agent/brain/intent_collector.py — Cycle 57 coverage.

IntentCollectorAgent: per-session isolation, required field guards,
history eviction cap, and thread safety.
"""

from __future__ import annotations

import json
import threading
from unittest.mock import patch

import pytest

from agent.brain.intent_collector import IntentCollectorAgent, TOOLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    """Fresh IntentCollectorAgent for each test."""
    return IntentCollectorAgent()


@pytest.fixture
def patch_session():
    """Patch _session_key to return a predictable value."""
    def _patch(agent_instance, session_id: str):
        return patch.object(agent_instance, "_session_key", return_value=session_id)
    return _patch


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_tools_exported(self):
        assert any(t["name"] == "capture_intent" for t in TOOLS)
        assert any(t["name"] == "get_current_intent" for t in TOOLS)

    def test_capture_intent_required_fields(self):
        schema = next(t for t in TOOLS if t["name"] == "capture_intent")
        required = schema["input_schema"]["required"]
        assert "user_request" in required
        assert "interpretation" in required


# ---------------------------------------------------------------------------
# Required field guards
# ---------------------------------------------------------------------------

class TestRequiredFieldGuards:
    """capture_intent must reject missing / empty required fields."""

    def test_missing_user_request_returns_error(self, agent, patch_session):
        with patch_session(agent, "s1"):
            result = json.loads(agent.handle("capture_intent", {"interpretation": "lower CFG"}))
        assert "error" in result
        assert "user_request" in result["error"]

    def test_empty_user_request_returns_error(self, agent, patch_session):
        with patch_session(agent, "s1"):
            result = json.loads(agent.handle("capture_intent", {
                "user_request": "", "interpretation": "lower CFG",
            }))
        assert "error" in result

    def test_missing_interpretation_returns_error(self, agent, patch_session):
        with patch_session(agent, "s1"):
            result = json.loads(agent.handle("capture_intent", {"user_request": "make dreamier"}))
        assert "error" in result
        assert "interpretation" in result["error"]

    def test_empty_interpretation_returns_error(self, agent, patch_session):
        with patch_session(agent, "s1"):
            result = json.loads(agent.handle("capture_intent", {
                "user_request": "make dreamier", "interpretation": "",
            }))
        assert "error" in result


# ---------------------------------------------------------------------------
# capture + retrieve round-trip
# ---------------------------------------------------------------------------

class TestCaptureRetrieve:
    def test_capture_then_get_returns_same_intent(self, agent, patch_session):
        with patch_session(agent, "roundtrip"):
            agent.handle("capture_intent", {
                "user_request": "warmer tones",
                "interpretation": "raise warmth slider to 0.7",
            })
            result = json.loads(agent.handle("get_current_intent", {}))

        assert result["status"] == "ok"
        assert result["intent"]["user_request"] == "warmer tones"
        assert result["intent"]["interpretation"] == "raise warmth slider to 0.7"

    def test_get_before_capture_returns_empty(self, agent, patch_session):
        with patch_session(agent, "empty"):
            result = json.loads(agent.handle("get_current_intent", {}))
        assert result["status"] == "empty"
        assert result["intent"] is None

    def test_second_capture_overwrites_first(self, agent, patch_session):
        with patch_session(agent, "overwrite"):
            agent.handle("capture_intent", {
                "user_request": "first", "interpretation": "a",
            })
            agent.handle("capture_intent", {
                "user_request": "second", "interpretation": "b",
            })
            result = json.loads(agent.handle("get_current_intent", {}))

        assert result["intent"]["user_request"] == "second"

    def test_style_references_stored(self, agent, patch_session):
        with patch_session(agent, "styles"):
            agent.handle("capture_intent", {
                "user_request": "cinematic",
                "interpretation": "use film look",
                "style_references": ["Blade Runner", "Dune"],
            })
            result = json.loads(agent.handle("get_current_intent", {}))

        assert result["intent"]["style_references"] == ["Blade Runner", "Dune"]

    def test_history_count_increments(self, agent, patch_session):
        with patch_session(agent, "history"):
            for i in range(5):
                result = json.loads(agent.handle("capture_intent", {
                    "user_request": f"req {i}", "interpretation": f"interp {i}",
                }))
        assert result["history_count"] == 5


# ---------------------------------------------------------------------------
# Per-session isolation
# ---------------------------------------------------------------------------

class TestSessionIsolation:
    def test_different_sessions_isolated(self, agent):
        with patch.object(agent, "_session_key", return_value="session_A"):
            agent.handle("capture_intent", {
                "user_request": "session A request", "interpretation": "A interp",
            })

        with patch.object(agent, "_session_key", return_value="session_B"):
            result = json.loads(agent.handle("get_current_intent", {}))

        # Session B should have no intent
        assert result["status"] == "empty"

    def test_session_a_does_not_see_session_b_intent(self, agent):
        with patch.object(agent, "_session_key", return_value="A"):
            agent.handle("capture_intent", {"user_request": "A req", "interpretation": "A"})

        with patch.object(agent, "_session_key", return_value="B"):
            agent.handle("capture_intent", {"user_request": "B req", "interpretation": "B"})

        with patch.object(agent, "_session_key", return_value="A"):
            result = json.loads(agent.handle("get_current_intent", {}))

        assert result["intent"]["user_request"] == "A req"


# ---------------------------------------------------------------------------
# History eviction cap
# ---------------------------------------------------------------------------

class TestHistoryCap:
    """History must not grow beyond _MAX_INTENT_HISTORY=100."""

    def test_history_cap_enforced(self, agent, patch_session):
        from agent.brain.intent_collector import _MAX_INTENT_HISTORY
        with patch_session(agent, "cap"):
            for i in range(_MAX_INTENT_HISTORY + 20):
                agent.handle("capture_intent", {
                    "user_request": f"req {i}", "interpretation": f"interp {i}",
                })
            history = agent.get_history()

        assert len(history) == _MAX_INTENT_HISTORY
        # Most recent entries are retained
        assert history[-1]["user_request"] == f"req {_MAX_INTENT_HISTORY + 19}"

    def test_history_retains_most_recent(self, agent, patch_session):
        """After cap eviction, the latest N entries must be the ones kept."""
        from agent.brain.intent_collector import _MAX_INTENT_HISTORY
        with patch_session(agent, "recent"):
            for i in range(_MAX_INTENT_HISTORY + 10):
                agent.handle("capture_intent", {
                    "user_request": f"r{i}", "interpretation": f"i{i}",
                })
            history = agent.get_history()

        # Oldest evicted entries should not be present
        assert history[0]["user_request"] == "r10"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_captures_no_exception(self, agent):
        """10 threads each capturing intent simultaneously must not crash."""
        errors = []

        def _capture(i):
            try:
                with patch.object(agent, "_session_key", return_value=f"t{i}"):
                    agent.handle("capture_intent", {
                        "user_request": f"req {i}", "interpretation": f"interp {i}",
                    })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_capture, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
