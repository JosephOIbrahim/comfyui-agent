"""Intent Collector — captures artistic intent for metadata embedding.

Stores the artist's request, the agent's interpretation, style references,
and session context. Intent is captured when the user issues a generation
request, then consumed by image_metadata.write_image_metadata after
successful execution.

Thread-safe module-level state (matches orchestrator/demo pattern).
"""

import logging
import threading
import time

from ._sdk import BrainAgent, BrainConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "capture_intent",
        "description": (
            "Capture the artist's creative intent for the current generation. "
            "Call this before executing a workflow to record what the artist "
            "wants and how the agent interprets it. The intent is embedded "
            "into the output image's metadata after successful execution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_request": {
                    "type": "string",
                    "description": (
                        "The artist's original request in their own words "
                        "(e.g. 'make it dreamier' or 'add dramatic lighting')."
                    ),
                },
                "interpretation": {
                    "type": "string",
                    "description": (
                        "How the agent interpreted the request in technical terms "
                        "(e.g. 'Lower CFG to 5, switch sampler to DPM++ 2M Karras')."
                    ),
                },
                "style_references": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Style references mentioned by the artist (image URLs, "
                        "artist names, aesthetic descriptions)."
                    ),
                },
                "session_context": {
                    "type": "string",
                    "description": (
                        "Relevant session context (e.g. 'iteration 3 of anime "
                        "portrait series, artist prefers warm tones')."
                    ),
                },
            },
            "required": ["user_request", "interpretation"],
        },
    },
    {
        "name": "get_current_intent",
        "description": (
            "Retrieve the most recently captured artistic intent. "
            "Used by the metadata writer after execution to embed "
            "intent into the output image."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ---------------------------------------------------------------------------
# SDK Agent class
# ---------------------------------------------------------------------------

class IntentCollectorAgent(BrainAgent):
    """Captures and serves artistic intent for metadata embedding."""

    TOOLS = TOOLS

    def __init__(self, config: BrainConfig | None = None):
        super().__init__(config)
        self._current_intent: dict | None = None
        self._intent_history: list[dict] = []
        self._lock = threading.Lock()

    def capture(
        self,
        user_request: str,
        interpretation: str,
        style_references: list[str] | None = None,
        session_context: str = "",
    ) -> dict:
        """Store intent for current generation."""
        intent = {
            "user_request": user_request,
            "interpretation": interpretation,
            "style_references": style_references or [],
            "session_context": session_context,
            "captured_at": time.time(),
        }
        with self._lock:
            self._current_intent = intent
            self._intent_history.append(intent)
        return intent

    def get_current(self) -> dict | None:
        """Return the most recent intent, or None."""
        with self._lock:
            return self._current_intent

    def clear(self) -> None:
        """Clear current intent (after it's been consumed)."""
        with self._lock:
            self._current_intent = None

    def get_history(self) -> list[dict]:
        """Return all captured intents this session."""
        with self._lock:
            return list(self._intent_history)

    def handle(self, name: str, tool_input: dict) -> str:
        if name == "capture_intent":
            intent = self.capture(
                user_request=tool_input["user_request"],
                interpretation=tool_input["interpretation"],
                style_references=tool_input.get("style_references", []),
                session_context=tool_input.get("session_context", ""),
            )
            return self.to_json({
                "status": "captured",
                "intent": intent,
                "history_count": len(self._intent_history),
            })
        elif name == "get_current_intent":
            current = self.get_current()
            if current is None:
                return self.to_json({
                    "status": "empty",
                    "intent": None,
                    "message": "No intent captured yet. Use capture_intent first.",
                })
            return self.to_json({
                "status": "ok",
                "intent": current,
            })
        else:
            return self.to_json({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Module-level singleton (lazy, for backward compat with tools registry)
# ---------------------------------------------------------------------------

_singleton: IntentCollectorAgent | None = None
_singleton_lock = threading.Lock()


def _get_agent() -> IntentCollectorAgent:
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = IntentCollectorAgent()
        return _singleton


def handle(name: str, tool_input: dict) -> str:
    """Module-level dispatch."""
    return _get_agent().handle(name, tool_input)
