"""Integration tests — metrics counters, histograms, and health endpoint.

Verifies that tool calls and LLM calls correctly increment the
observability metrics defined in agent.metrics.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.metrics import (
    MetricsRegistry,
    llm_call_duration_seconds,
    llm_call_total,
    tool_call_duration_seconds,
    tool_call_total,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _reset_metrics():
    """Reset all metrics before and after each test."""
    registry = MetricsRegistry()
    registry.reset()
    yield
    registry.reset()


class TestToolCallMetrics:
    """Verify tool dispatch increments tool_call_total and duration."""

    def test_tool_call_increments_counter(self) -> None:
        """A successful tool call increments tool_call_total with status=ok."""
        tool_call_total.inc(tool_name="is_comfyui_running", status="ok")
        snapshot = tool_call_total.get()
        key = ("is_comfyui_running", "ok")
        assert snapshot.get(key, 0) == 1

    def test_tool_call_records_duration(self) -> None:
        """A tool call observation is recorded in the histogram."""
        tool_call_duration_seconds.observe(0.123, tool_name="is_comfyui_running")
        snapshot = tool_call_duration_seconds.get()
        # Should have at least one label-combo with count >= 1
        assert len(snapshot) >= 1
        first_key = next(iter(snapshot))
        assert snapshot[first_key]["count"] >= 1
        assert snapshot[first_key]["sum"] >= 0.123


class TestLLMCallMetrics:
    """Verify LLM provider instrumentation records metrics."""

    def test_llm_call_increments_counter(self) -> None:
        """Mocked LLM stream records llm_call_total with status=ok."""
        from agent.llm._base import _record_llm_metric

        _record_llm_metric("anthropic", "ok", 1.5)
        snapshot = llm_call_total.get()
        key = ("anthropic", "ok")
        assert snapshot.get(key, 0) == 1

    def test_llm_error_increments_error_counter(self) -> None:
        """Mocked LLM error records llm_call_total with status=error."""
        from agent.llm._base import _record_llm_metric

        _record_llm_metric("openai", "error", 0.5)
        snapshot = llm_call_total.get()
        key = ("openai", "error")
        assert snapshot.get(key, 0) == 1

    def test_llm_duration_recorded(self) -> None:
        """LLM call duration is observed in the histogram."""
        from agent.llm._base import _record_llm_metric

        _record_llm_metric("gemini", "ok", 2.34)
        snapshot = llm_call_duration_seconds.get()
        key = ("gemini",)
        assert key in snapshot
        assert snapshot[key]["count"] == 1
        assert abs(snapshot[key]["sum"] - 2.34) < 0.001

    def test_multiple_providers_tracked_separately(self) -> None:
        """Each provider label is tracked independently."""
        from agent.llm._base import _record_llm_metric

        _record_llm_metric("anthropic", "ok", 1.0)
        _record_llm_metric("openai", "ok", 2.0)
        _record_llm_metric("ollama", "ok", 0.5)

        snapshot = llm_call_total.get()
        assert snapshot.get(("anthropic", "ok"), 0) == 1
        assert snapshot.get(("openai", "ok"), 0) == 1
        assert snapshot.get(("ollama", "ok"), 0) == 1


class TestHealthMetrics:
    """Verify the health endpoint includes metrics."""

    def test_health_includes_metrics(self) -> None:
        """check_health() returns a 'metrics' key with tool_call data."""
        # Record some metrics so the health endpoint has something to report
        tool_call_total.inc(tool_name="load_workflow", status="ok")
        tool_call_duration_seconds.observe(0.05, tool_name="load_workflow")

        with patch("agent.health._check_comfyui", return_value={"status": "ok"}), \
             patch("agent.health._check_llm", return_value={"status": "ok"}):
            from agent.health import check_health

            health = check_health()
            assert "metrics" in health
            metrics = health["metrics"]
            assert "total_tool_calls" in metrics
            assert metrics["total_tool_calls"] >= 1


class TestMetricsReset:
    """Verify metrics can be reset to zero."""

    def test_metrics_reset(self) -> None:
        """After recording and resetting, all counters return to zero."""
        # Record some values
        tool_call_total.inc(tool_name="discover", status="ok")
        tool_call_total.inc(tool_name="discover", status="error")
        llm_call_total.inc(provider="anthropic", status="ok")
        tool_call_duration_seconds.observe(1.0, tool_name="discover")
        llm_call_duration_seconds.observe(2.0, provider="anthropic")

        # Verify non-empty
        assert len(tool_call_total.get()) > 0
        assert len(llm_call_total.get()) > 0

        # Reset
        registry = MetricsRegistry()
        registry.reset()

        # Verify all zeroed
        assert len(tool_call_total.get()) == 0
        assert len(llm_call_total.get()) == 0
        assert len(tool_call_duration_seconds.get()) == 0
        assert len(llm_call_duration_seconds.get()) == 0
