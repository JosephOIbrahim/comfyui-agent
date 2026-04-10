"""Unified error vocabulary for all tools.

Every tool error follows one shape. No exceptions.

Exception hierarchy:
    AgentError          Base for all agent-specific exceptions.
      ToolError         A tool call failed in a recoverable way.
      TransportError    Network / HTTP communication failure.
      ValidationError   Input or workflow validation failure.
"""

import json as _json


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class AgentError(Exception):
    """Base exception for all ComfyUI Agent errors."""


class ToolError(AgentError):
    """Raised when a tool call fails in a recoverable, expected way.

    Use for user-actionable conditions (missing node, bad input, etc.)
    rather than for unexpected internal failures.
    """


class TransportError(AgentError):
    """Raised when network or HTTP communication with ComfyUI fails.

    Wraps connection errors, timeouts, and non-2xx responses so callers
    can distinguish network problems from logic errors.
    """


class ValidationError(AgentError):
    """Raised when workflow or input validation fails.

    Wraps schema mismatches, missing required fields, and structural
    errors that prevent safe execution.
    """


# ---------------------------------------------------------------------------
# JSON error helper (tool-layer response format)
# ---------------------------------------------------------------------------


def error_json(message: str, *, hint: str | None = None, **context) -> str:
    """Return a standardized error JSON string.

    Args:
        message: What went wrong (human-readable, no jargon).
        hint: Optional suggestion for what to do next.
        **context: Optional structured context (e.g., available=["a","b"]).
    """
    result: dict = {"error": message}
    if hint:
        result["hint"] = hint
    if context:
        result["context"] = context
    return _json.dumps(result, sort_keys=True, allow_nan=False)  # Cycle 61: NaN-safe
