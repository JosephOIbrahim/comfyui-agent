"""Shared protocol for brain-to-brain communication.

Defines the BrainMessage format used for all inter-module communication.
Today: dicts passed between functions in the same process.
Tomorrow: serialized JSON between Agent SDK agents.
"""

import time
import uuid

from ._sdk import _default_to_json as to_json


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


def dispatch_brain_message(msg: dict) -> None:
    """Route a BrainMessage to its target module via the tools dispatcher.

    Fire-and-forget: catches all exceptions, logs warning on failure.
    Currently supports vision->memory routing (record_outcome).
    """
    import logging

    _log = logging.getLogger(__name__)

    try:
        source = msg.get("source", "")
        target = msg.get("target", "")
        payload = msg.get("payload", {})

        if source == "vision" and target == "memory":
            from ..tools import handle as dispatch_tool

            action = payload.get("action", "")
            outcome_input = {
                "session": "default",
                "action": action,
                "result": "success",
                "details": {
                    k: v for k, v in sorted(payload.items())
                    if k != "action"
                },
            }
            dispatch_tool("record_outcome", outcome_input)
            _log.debug("Dispatched brain message %s->%s (%s)", source, target, action)
        else:
            _log.debug("No dispatch route for %s->%s, skipping", source, target)
    except Exception as e:
        _log.warning("dispatch_brain_message failed (non-fatal): %s", e)
