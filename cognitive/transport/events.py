"""Structured execution event types.

Parses ComfyUI WebSocket messages into typed models with
computed fields (progress percentage, elapsed time).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    """ComfyUI execution event types."""

    EXECUTION_START = "execution_start"
    EXECUTION_CACHED = "execution_cached"
    EXECUTING = "executing"
    PROGRESS = "progress"
    EXECUTED = "executed"
    EXECUTION_ERROR = "execution_error"
    EXECUTION_INTERRUPTED = "execution_interrupted"
    EXECUTION_COMPLETE = "execution_complete"  # Synthetic — generated when node=None
    UNKNOWN = "unknown"


@dataclass
class ExecutionEvent:
    """A single parsed execution event from the ComfyUI WebSocket.

    Attributes:
        event_type: Classified event type.
        prompt_id: The prompt this event belongs to.
        node_id: The node currently executing (if applicable).
        data: Raw event data from WebSocket.
        timestamp: When this event was received.
        progress_value: Current step progress (0-max).
        progress_max: Total steps for current node.
        started_at: Timestamp of the execution_start event (for elapsed_ms).
    """

    event_type: EventType
    prompt_id: str = ""
    node_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    progress_value: int = 0
    progress_max: int = 0
    started_at: float = 0.0

    @property
    def progress_pct(self) -> float:
        """Progress as a percentage (0.0 - 100.0)."""
        if self.progress_max <= 0:
            return 0.0
        return min(100.0, (self.progress_value / self.progress_max) * 100.0)

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since execution started."""
        if self.started_at <= 0:
            return 0.0
        return (self.timestamp - self.started_at) * 1000.0

    @property
    def is_terminal(self) -> bool:
        """True if this event signals execution end."""
        return self.event_type in (
            EventType.EXECUTION_COMPLETE,
            EventType.EXECUTION_ERROR,
            EventType.EXECUTION_INTERRUPTED,
        )

    @property
    def is_error(self) -> bool:
        """True if this event signals an error."""
        return self.event_type == EventType.EXECUTION_ERROR

    @classmethod
    def from_ws_message(
        cls,
        msg: dict[str, Any],
        started_at: float = 0.0,
    ) -> ExecutionEvent:
        """Parse a raw ComfyUI WebSocket message into an ExecutionEvent.

        ComfyUI WS messages have format: {"type": "...", "data": {...}}
        """
        msg_type = msg.get("type", "unknown")
        data = msg.get("data", {})

        try:
            event_type = EventType(msg_type)
        except ValueError:
            event_type = EventType.UNKNOWN

        prompt_id = data.get("prompt_id", "")
        node_id = data.get("node")

        # "executing" with node=None means execution complete
        if event_type == EventType.EXECUTING and node_id is None:
            event_type = EventType.EXECUTION_COMPLETE

        progress_value = 0
        progress_max = 0
        if event_type == EventType.PROGRESS:
            progress_value = data.get("value", 0)
            progress_max = data.get("max", 0)

        now = time.time()
        actual_started_at = started_at
        if event_type == EventType.EXECUTION_START:
            actual_started_at = now

        return cls(
            event_type=event_type,
            prompt_id=prompt_id,
            node_id=node_id,
            data=data,
            timestamp=now,
            progress_value=progress_value,
            progress_max=progress_max,
            started_at=actual_started_at,
        )
