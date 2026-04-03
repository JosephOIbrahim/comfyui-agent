"""Queue-backed progress reporter for panel WebSocket streaming."""

import queue
import logging

log = logging.getLogger(__name__)


class QueueProgressReporter:
    """Pushes progress events to a thread-safe queue.

    Used by panel/server/chat.py to bridge execution progress
    from comfy_execute tools back to the WebSocket client.
    """

    def __init__(self, msg_queue: queue.Queue):
        self._q = msg_queue

    def report(self, progress, total=None, message=None):
        """Push a progress event to the queue."""
        self._q.put({
            "type": "progress",
            "progress": progress,
            "total": total,
            "message": message or "",
        })

    @staticmethod
    def noop():
        """Return a no-op reporter (for when no queue is available)."""
        return _NoopQueueReporter()


class _NoopQueueReporter:
    def report(self, progress, total=None, message=None):
        pass
