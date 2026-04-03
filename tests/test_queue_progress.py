"""Tests for agent.queue_progress — queue-backed progress reporter."""

import queue

from agent.queue_progress import QueueProgressReporter, _NoopQueueReporter


def test_report_puts_event():
    """QueueProgressReporter.report() enqueues a progress dict."""
    q = queue.Queue()
    reporter = QueueProgressReporter(q)
    reporter.report(5, message="Loading model")

    event = q.get_nowait()
    assert event["type"] == "progress"
    assert event["progress"] == 5
    assert event["total"] is None
    assert event["message"] == "Loading model"


def test_report_with_total():
    """Event includes progress, total, and message when all provided."""
    q = queue.Queue()
    reporter = QueueProgressReporter(q)
    reporter.report(10, total=20, message="KSampler step 10/20")

    event = q.get_nowait()
    assert event["type"] == "progress"
    assert event["progress"] == 10
    assert event["total"] == 20
    assert event["message"] == "KSampler step 10/20"


def test_report_default_message():
    """Message defaults to empty string when omitted."""
    q = queue.Queue()
    reporter = QueueProgressReporter(q)
    reporter.report(3, total=10)

    event = q.get_nowait()
    assert event["message"] == ""


def test_noop_does_nothing():
    """Noop reporter doesn't raise and doesn't put anything."""
    reporter = QueueProgressReporter.noop()
    assert isinstance(reporter, _NoopQueueReporter)
    # Should not raise
    reporter.report(1)
    reporter.report(5, total=10, message="test")


def test_multiple_reports_ordered():
    """Multiple reports arrive in FIFO order."""
    q = queue.Queue()
    reporter = QueueProgressReporter(q)
    reporter.report(1, total=3, message="step 1")
    reporter.report(2, total=3, message="step 2")
    reporter.report(3, total=3, message="step 3")

    events = []
    while not q.empty():
        events.append(q.get_nowait())

    assert len(events) == 3
    assert [e["progress"] for e in events] == [1, 2, 3]
    assert [e["message"] for e in events] == ["step 1", "step 2", "step 3"]
