"""Multi-provider LLM abstraction layer.

Usage:
    from agent.llm import get_provider
    provider = get_provider()  # Uses LLM_PROVIDER env var
    response = provider.stream(model=..., ...)

Supported providers: anthropic, openai, gemini, ollama.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ._base import LLMProvider
from ._types import (
    ImageBlock,
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    LLMServerError,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

__all__ = [
    "get_provider",
    "LLMProvider",
    "LLMResponse",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ImageBlock",
    "LLMError",
    "LLMRateLimitError",
    "LLMConnectionError",
    "LLMServerError",
    "LLMAuthError",
]

# Default models per provider — used when AGENT_MODEL is not explicitly set
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.1",
}

_provider_cache: dict[str, LLMProvider] = {}


def get_provider(name: str | None = None) -> LLMProvider:
    """Get an LLM provider instance (cached per name).

    Args:
        name: Provider name. If None, reads LLM_PROVIDER env var (default: "anthropic").

    Returns:
        An LLMProvider instance ready to use.

    Raises:
        ValueError: Unknown provider name.
        LLMAuthError: Missing API key for the requested provider.
    """
    if name is None:
        from ..config import LLM_PROVIDER
        name = LLM_PROVIDER

    name = name.lower().strip()

    if name in _provider_cache:
        return _provider_cache[name]

    provider = _create_provider(name)
    _provider_cache[name] = provider
    log.info("LLM provider initialized: %s", name)
    return provider


def _create_provider(name: str) -> LLMProvider:
    """Lazy-import and instantiate a provider."""
    if name == "anthropic":
        from ._anthropic import AnthropicProvider
        return AnthropicProvider()

    elif name == "openai":
        from ._openai import OpenAIProvider
        return OpenAIProvider()

    elif name == "gemini":
        from ._gemini import GeminiProvider
        return GeminiProvider()

    elif name == "ollama":
        from ._ollama import OllamaProvider
        return OllamaProvider()

    else:
        raise ValueError(
            f"Unknown LLM provider: {name!r}. "
            f"Supported: anthropic, openai, gemini, ollama"
        )
