"""Integration tests — trigger/callback dispatch patterns.

Tests event-driven callback patterns using a minimal in-test trigger
registry. Verifies that callbacks fire correctly on simulated events,
multiple triggers can coexist, and cleanup works.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Minimal trigger registry for testing callback dispatch patterns
# ---------------------------------------------------------------------------


class _TriggerRegistry:
    """Simple event-trigger registry for integration testing."""

    def __init__(self) -> None:
        self._triggers: dict[str, list] = {}
        self._lock = threading.Lock()

    def register(self, event: str, callback) -> None:
        with self._lock:
            self._triggers.setdefault(event, []).append(callback)

    def dispatch(self, event: str, **kwargs) -> int:
        with self._lock:
            callbacks = list(self._triggers.get(event, []))
        fired = 0
        for cb in callbacks:
            cb(**kwargs)
            fired += 1
        return fired

    def clear(self) -> None:
        with self._lock:
            self._triggers.clear()

    def count(self, event: str | None = None) -> int:
        with self._lock:
            if event:
                return len(self._triggers.get(event, []))
            return sum(len(v) for v in self._triggers.values())


@pytest.fixture()
def trigger_registry():
    """Provide a fresh trigger registry per test."""
    reg = _TriggerRegistry()
    yield reg
    reg.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTriggerFiresOnExecution:
    """Verify a registered trigger fires when its event is dispatched."""

    def test_trigger_fires_on_mock_execution(
        self, trigger_registry: _TriggerRegistry
    ) -> None:
        callback = MagicMock()
        trigger_registry.register("execution_complete", callback)

        fired = trigger_registry.dispatch(
            "execution_complete", prompt_id="abc123", status="success"
        )

        assert fired == 1
        callback.assert_called_once_with(prompt_id="abc123", status="success")


class TestWebhookCalledOnEvent:
    """Verify a webhook-style callback is invoked on dispatch."""

    def test_webhook_called_on_event(
        self, trigger_registry: _TriggerRegistry
    ) -> None:
        webhook_calls: list[dict] = []

        def mock_webhook(**kwargs) -> None:
            webhook_calls.append(kwargs)

        trigger_registry.register("workflow_saved", mock_webhook)
        trigger_registry.dispatch("workflow_saved", path="/tmp/wf.json")

        assert len(webhook_calls) == 1
        assert webhook_calls[0]["path"] == "/tmp/wf.json"


class TestMultipleTriggersFire:
    """Verify all registered triggers for the same event fire."""

    def test_multiple_triggers_fire(
        self, trigger_registry: _TriggerRegistry
    ) -> None:
        cb1 = MagicMock()
        cb2 = MagicMock()
        cb3 = MagicMock()

        trigger_registry.register("node_added", cb1)
        trigger_registry.register("node_added", cb2)
        trigger_registry.register("node_added", cb3)

        fired = trigger_registry.dispatch("node_added", node_type="KSampler")

        assert fired == 3
        cb1.assert_called_once_with(node_type="KSampler")
        cb2.assert_called_once_with(node_type="KSampler")
        cb3.assert_called_once_with(node_type="KSampler")


class TestTriggerCleanup:
    """Verify trigger cleanup removes all registered callbacks."""

    def test_trigger_cleanup_after_test(
        self, trigger_registry: _TriggerRegistry
    ) -> None:
        trigger_registry.register("event_a", MagicMock())
        trigger_registry.register("event_b", MagicMock())
        trigger_registry.register("event_a", MagicMock())

        assert trigger_registry.count() == 3

        trigger_registry.clear()

        assert trigger_registry.count() == 0
        # Dispatch after clear fires nothing
        fired = trigger_registry.dispatch("event_a")
        assert fired == 0
