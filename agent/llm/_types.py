"""Common types for multi-provider LLM abstraction.

These types decouple the agent loop from any specific LLM SDK.
Providers translate between these types and their native formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Content blocks — provider-agnostic message content
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    """A text content block in an LLM response."""
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    """A tool-use request from the LLM."""
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class ToolResultBlock:
    """A tool result fed back to the LLM."""
    tool_use_id: str
    content: str
    type: str = "tool_result"


@dataclass
class ImageBlock:
    """An image content block (base64-encoded)."""
    data: str          # base64-encoded image bytes
    media_type: str    # "image/png", "image/jpeg", etc.
    type: str = "image"


# ---------------------------------------------------------------------------
# LLM response
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    content: list             # list[TextBlock | ToolUseBlock]
    stop_reason: str          # "end_turn" | "tool_use" | "length"
    model: str = ""
    usage: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Error hierarchy — providers catch native errors and re-raise as these
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base error for all LLM provider errors."""


class LLMRateLimitError(LLMError):
    """Rate limit exceeded — caller should retry with backoff."""


class LLMConnectionError(LLMError):
    """Network / connection failure — transient, retry may help."""


class LLMServerError(LLMError):
    """Server-side error (5xx) — transient, retry may help."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class LLMAuthError(LLMError):
    """Authentication failure — bad or missing API key."""
