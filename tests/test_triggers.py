"""Unit tests for the event trigger system (cognitive/transport/triggers.py)."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from cognitive.transport.events import EventType, ExecutionEvent
from cognitive.transport.triggers import (
    TriggerRegistry,
    _default_registry,
    clear,
    dispatch,
    on_execution_complete,
    on_execution_error,
    on_progress,
    register,
    register_webhook,
    unregister,
)


@pytest.fixture(autouse=True)
def _reset_default_registry():
    """Clear the module-level default registry between tests."""
    _default_registry.clear()
    yield
    _default_registry.clear()


def _make_event(
    event_type: EventType = EventType.PROGRESS,
    prompt_id: str = "p1",
    node_id: str | None = "5",
    progress_value: int = 0,
    progress_max: int = 0,
) -> ExecutionEvent:
    return ExecutionEvent(
        event_type=event_type,
        prompt_id=prompt_id,
        node_id=node_id,
        progress_value=progress_value,
        progress_max=progress_max,
        started_at=time.time() - 1.0,
    )


class TestTriggerRegistry:
    """Core registry behaviour."""

    def test_register_and_dispatch(self):
        reg = TriggerRegistry()
        cb = MagicMock()
        reg.register(EventType.PROGRESS, cb)
        event = _make_event(EventType.PROGRESS)
        fired = reg.dispatch(event)
        assert fired == 1
        cb.assert_called_once_with(event)

    def test_filter_matching(self):
        reg = TriggerRegistry()
        cb = MagicMock()
        reg.register(EventType.PROGRESS, cb, filter={"node_id": "5"})

        # Matching event
        event_match = _make_event(EventType.PROGRESS, node_id="5")
        assert reg.dispatch(event_match) == 1
        cb.assert_called_once()

        cb.reset_mock()

        # Non-matching event
        event_miss = _make_event(EventType.PROGRESS, node_id="3")
        assert reg.dispatch(event_miss) == 0
        cb.assert_not_called()

    def test_once_trigger(self):
        reg = TriggerRegistry()
        cb = MagicMock()
        reg.register(EventType.PROGRESS, cb, once=True)

        event = _make_event(EventType.PROGRESS)
        assert reg.dispatch(event) == 1
        cb.assert_called_once()

        # Second dispatch — trigger was auto-removed
        cb.reset_mock()
        assert reg.dispatch(event) == 0
        cb.assert_not_called()
        assert reg.count() == 0

    def test_unregister(self):
        reg = TriggerRegistry()
        cb = MagicMock()
        tid = reg.register(EventType.PROGRESS, cb)
        assert reg.unregister(tid) is True

        event = _make_event(EventType.PROGRESS)
        assert reg.dispatch(event) == 0
        cb.assert_not_called()

    def test_unregister_unknown_id(self):
        reg = TriggerRegistry()
        assert reg.unregister("nonexistent") is False

    def test_clear(self):
        reg = TriggerRegistry()
        for _ in range(3):
            reg.register(EventType.PROGRESS, MagicMock())
        assert reg.count() == 3
        reg.clear()
        assert reg.count() == 0

    def test_callback_exception_contained(self):
        reg = TriggerRegistry()

        def bad_callback(event: ExecutionEvent) -> None:
            raise RuntimeError("boom")

        reg.register(EventType.PROGRESS, bad_callback)
        event = _make_event(EventType.PROGRESS)
        # Should NOT propagate
        fired = reg.dispatch(event)
        assert fired == 1

    def test_dispatch_returns_count(self):
        reg = TriggerRegistry()
        reg.register(EventType.PROGRESS, MagicMock())
        reg.register(EventType.PROGRESS, MagicMock())
        reg.register(EventType.EXECUTION_ERROR, MagicMock())  # non-matching

        event = _make_event(EventType.PROGRESS)
        assert reg.dispatch(event) == 2


class TestFactoryFunctions:
    """Built-in trigger factories."""

    def test_on_execution_complete_factory(self):
        cb = MagicMock()
        on_execution_complete(cb)
        event = _make_event(EventType.EXECUTION_COMPLETE)
        assert dispatch(event) == 1
        cb.assert_called_once_with(event)

    def test_on_execution_error_factory(self):
        cb = MagicMock()
        on_execution_error(cb)
        event = _make_event(EventType.EXECUTION_ERROR)
        assert dispatch(event) == 1
        cb.assert_called_once_with(event)

    def test_on_progress_with_node_filter(self):
        cb = MagicMock()
        on_progress(cb, node_id="5")

        # Matching node
        assert dispatch(_make_event(EventType.PROGRESS, node_id="5")) == 1
        cb.assert_called_once()

        cb.reset_mock()

        # Non-matching node
        assert dispatch(_make_event(EventType.PROGRESS, node_id="9")) == 0
        cb.assert_not_called()

    def test_on_progress_without_filter(self):
        cb = MagicMock()
        on_progress(cb)
        assert dispatch(_make_event(EventType.PROGRESS, node_id="99")) == 1
        cb.assert_called_once()


class TestWebhook:
    """Webhook trigger support."""

    @patch("httpx.post")
    def test_register_webhook(self, mock_post):
        ids = register_webhook(
            "https://example.com/hook",
            [EventType.EXECUTION_COMPLETE],
        )
        assert len(ids) == 1

        event = _make_event(
            EventType.EXECUTION_COMPLETE,
            prompt_id="abc",
            node_id=None,
        )
        assert dispatch(event) == 1
        mock_post.assert_called_once()

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["event_type"] == "execution_complete"
        assert body["prompt_id"] == "abc"
        assert body["node_id"] is None

    @patch("httpx.post", side_effect=ConnectionError("fail"))
    def test_webhook_failure_contained(self, mock_post):
        register_webhook(
            "https://example.com/hook",
            [EventType.EXECUTION_COMPLETE],
        )
        event = _make_event(EventType.EXECUTION_COMPLETE)
        # Should NOT propagate
        fired = dispatch(event)
        assert fired == 1


class TestThreadSafety:
    """Concurrent access to the registry."""

    def test_thread_safety(self):
        reg = TriggerRegistry()
        errors: list[Exception] = []
        barrier = threading.Barrier(8)

        def worker(idx: int) -> None:
            try:
                barrier.wait(timeout=5)
                # Register
                tid = reg.register(
                    EventType.PROGRESS, lambda e: None, once=(idx % 2 == 0)
                )
                # Dispatch
                event = _make_event(EventType.PROGRESS)
                reg.dispatch(event)
                # Unregister (may already be removed if once=True fired)
                reg.unregister(tid)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"


class TestModuleLevelFunctions:
    """Module-level register/dispatch/clear delegate to singleton."""

    def test_default_registry_module_functions(self):
        cb = MagicMock()
        tid = register(EventType.PROGRESS, cb)
        assert _default_registry.count() == 1

        event = _make_event(EventType.PROGRESS)
        assert dispatch(event) == 1
        cb.assert_called_once()

        assert unregister(tid) is True
        assert _default_registry.count() == 0

    def test_clear_delegates(self):
        register(EventType.PROGRESS, MagicMock())
        register(EventType.PROGRESS, MagicMock())
        assert _default_registry.count() == 2
        clear()
        assert _default_registry.count() == 0
