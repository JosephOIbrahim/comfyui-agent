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

_MAX_INTENT_HISTORY = 100

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_TOOLS: list[dict] = [
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
    """Captures and serves artistic intent for metadata embedding.

    Intent is stored per-session so concurrent MCP connections do not
    overwrite each other's creative context. The session key is derived
    from current_conn_session() — each MCP connection gets a unique,
    stable name (e.g. "conn_3f2a1b4c"); tests and CLI default to "default".
    (Cycle 35 fix)
    """

    TOOLS = _TOOLS

    def __init__(self, config: BrainConfig | None = None):
        super().__init__(config)
        # Per-session storage: session_key -> intent dict / history list
        self._intents: dict[str, dict | None] = {}
        self._histories: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def _session_key(self) -> str:
        """Return the current MCP connection's session name (or 'default')."""
        from .._conn_ctx import current_conn_session
        return current_conn_session()

    def capture(
        self,
        user_request: str,
        interpretation: str,
        style_references: list[str] | None = None,
        session_context: str = "",
    ) -> dict:
        """Store intent for the current generation (isolated per MCP session)."""
        intent = {
            "user_request": user_request,
            "interpretation": interpretation,
            "style_references": style_references or [],
            "session_context": session_context,
            "captured_at": time.time(),
        }
        key = self._session_key()
        with self._lock:
            self._intents[key] = intent
            history = self._histories.setdefault(key, [])
            history.append(intent)
            if len(history) > _MAX_INTENT_HISTORY:
                self._histories[key] = history[-_MAX_INTENT_HISTORY:]
        return intent

    def get_current(self) -> dict | None:
        """Return the most recent intent for this session, or None."""
        key = self._session_key()
        with self._lock:
            return self._intents.get(key)

    def clear(self) -> None:
        """Clear current intent for this session (after it's been consumed)."""
        key = self._session_key()
        with self._lock:
            self._intents[key] = None

    def get_history(self) -> list[dict]:
        """Return all captured intents for this session."""
        key = self._session_key()
        with self._lock:
            return list(self._histories.get(key, []))

    def handle(self, name: str, tool_input: dict) -> str:
        if name == "capture_intent":
            user_request = tool_input.get("user_request")
            interpretation = tool_input.get("interpretation")
            if not user_request or not isinstance(user_request, str):  # Cycle 43: guard required fields
                return self.to_json({"error": "user_request is required and must be a non-empty string."})
            if not interpretation or not isinstance(interpretation, str):
                return self.to_json({"error": "interpretation is required and must be a non-empty string."})
            intent = self.capture(
                user_request=user_request,
                interpretation=interpretation,
                style_references=tool_input.get("style_references", []),
                session_context=tool_input.get("session_context", ""),
            )
            return self.to_json({
                "status": "captured",
                "intent": intent,
                "history_count": len(self.get_history()),
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
# Backward-compat re-exports (consumed by tests outside test_brain_*.py)
# ---------------------------------------------------------------------------

TOOLS = _TOOLS


def handle(name: str, tool_input: dict) -> str:
    """Module-level dispatch shim — routes to the registered instance."""
    BrainAgent._register_all()
    agent = BrainAgent._registry.get(name)
    if agent is not None:
        return agent.handle(name, tool_input)
    import json as _json  # Cycle 43: return error directly; avoid circular import via brain package
    return _json.dumps({"error": f"Unknown tool: {name}"}, sort_keys=True, allow_nan=False)  # Cycle 59
