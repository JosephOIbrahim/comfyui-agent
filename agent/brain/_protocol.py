"""Shared protocol for brain-to-brain communication.

Defines the BrainMessage format used for all inter-module communication.
Today: dicts passed between functions in the same process.
Tomorrow: serialized JSON between Agent SDK agents.
"""

import time
import uuid

from ..tools._util import to_json


def make_id() -> str:
    """Generate a deterministic-format correlation ID."""
    return uuid.uuid4().hex[:12]


def brain_message(
    source: str,
    target: str,
    msg_type: str,
    payload: dict,
    correlation_id: str | None = None,
) -> dict:
    """Create a BrainMessage dict.

    Args:
        source: originating module ("planner", "vision", "memory", etc.)
        target: destination module
        msg_type: "request", "result", "status", "error"
        payload: the actual data
        correlation_id: links request -> response (auto-generated if None)
    """
    return {
        "source": source,
        "target": target,
        "msg_type": msg_type,
        "payload": payload,
        "correlation_id": correlation_id or make_id(),
        "timestamp": time.time(),
    }


def serialize(msg: dict) -> str:
    """Serialize a BrainMessage to JSON (He2025 deterministic)."""
    return to_json(msg)
