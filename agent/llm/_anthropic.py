"""Anthropic Claude provider — reference implementation.

Wraps the anthropic SDK to implement the LLMProvider protocol.
Handles prompt caching, streaming, tool format, and error translation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import anthropic

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
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

log = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Claude via the Anthropic SDK."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        tools: list[dict],
        messages: list[dict],
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        native_tools = self.convert_tools(tools)
        native_messages = self.convert_messages(messages)
        cached_system = _cached_system(system)

        try:
            with self._client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=cached_system,
                tools=native_tools,
                messages=native_messages,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta
                        # Cycle 18: filter empty deltas. Anthropic emits
                        # zero-width content_block_delta events at content
                        # block boundaries; without the truthy check, those
                        # would fire on_text("") / on_thinking("") and (since
                        # cycle 7) set content_emitted=True, suppressing
                        # legitimate retries on transient errors.
                        if hasattr(delta, "text") and delta.text and on_text:
                            on_text(delta.text)
                        elif (
                            hasattr(delta, "thinking")
                            and delta.thinking
                            and on_thinking
                        ):
                            on_thinking(delta.thinking)
                final = stream.get_final_message()

        except anthropic.AuthenticationError as e:
            raise LLMAuthError(str(e)) from e
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except anthropic.APIConnectionError as e:
            raise LLMConnectionError(str(e)) from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise LLMServerError(str(e), status_code=e.status_code) from e
            raise LLMError(str(e)) from e
        except anthropic.APIError as e:
            raise LLMError(str(e)) from e

        return _to_response(final)

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict],
        timeout: float | None = None,
    ) -> LLMResponse:
        native_messages = self._convert_vision_messages(messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": native_messages,
        }

        try:
            if timeout:
                client = anthropic.Anthropic(timeout=timeout)
            else:
                client = self._client
            response = client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise LLMAuthError(str(e)) from e
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except anthropic.APIConnectionError as e:
            raise LLMConnectionError(str(e)) from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise LLMServerError(str(e), status_code=e.status_code) from e
            raise LLMError(str(e)) from e
        except anthropic.APIError as e:
            raise LLMError(str(e)) from e

        return _to_response(response)

    def convert_tools(self, tools: list[dict]) -> list[dict]:
        """Anthropic tools use MCP format natively. Add prompt caching."""
        if not tools:
            return tools
        cached = [dict(t) for t in tools]
        cached[-1] = {**cached[-1], "cache_control": {"type": "ephemeral"}}
        return cached

    def convert_messages(self, messages: list[dict]) -> list[dict]:
        """Convert common types back to Anthropic-native dicts."""
        result = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                result.append(msg)
                continue

            if isinstance(content, list):
                native_content = []
                for block in content:
                    if isinstance(block, TextBlock):
                        native_content.append({"type": "text", "text": block.text})
                    elif isinstance(block, ToolUseBlock):
                        native_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                    elif isinstance(block, ToolResultBlock):
                        native_content.append({
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id,
                            "content": block.content,
                        })
                    elif isinstance(block, ThinkingBlock):
                        # Cycle 20: skip thinking blocks in multi-turn messages.
                        # The Anthropic API requires a signature field we don't
                        # capture, so sending it back would cause a 400 error.
                        # Thinking content is already streamed via on_thinking.
                        continue
                    elif isinstance(block, dict):
                        native_content.append(block)
                    else:
                        native_content.append(block)
                result.append({"role": role, "content": native_content})
            else:
                result.append(msg)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _convert_vision_messages(self, messages: list[dict]) -> list[dict]:
        """Convert messages containing ImageBlock to Anthropic vision format."""
        result = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                result.append(msg)
                continue
            native_content = []
            for block in content:
                if isinstance(block, ImageBlock):
                    native_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": block.media_type,
                            "data": block.data,
                        },
                    })
                elif isinstance(block, TextBlock):
                    native_content.append({"type": "text", "text": block.text})
                elif isinstance(block, dict):
                    native_content.append(block)
                else:
                    native_content.append(block)
            result.append({"role": msg["role"], "content": native_content})
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cached_system(system: str) -> list[dict]:
    """Wrap system prompt in a cacheable text block."""
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def _to_response(msg) -> LLMResponse:
    """Convert Anthropic Message to LLMResponse.

    Cycle 18: handle thinking blocks. Claude 3.7+ extended-thinking and
    Claude 4 reasoning return content blocks with type=\"thinking\" alongside
    text/tool_use blocks. Without the elif branch below, those blocks were
    silently dropped — causing the model's reasoning to disappear from the
    final response object for any caller using provider.create() (vision
    pipeline, programmatic API).
    """
    content = []
    for block in msg.content:
        if block.type == "text":
            content.append(TextBlock(text=block.text))
        elif block.type == "tool_use":
            content.append(ToolUseBlock(id=block.id, name=block.name, input=block.input))
        elif block.type == "thinking":
            # `block.thinking` holds the reasoning text for Anthropic
            # extended-thinking. Defensive getattr in case the SDK ever
            # renames the field.
            content.append(ThinkingBlock(
                thinking=getattr(block, "thinking", "") or "",
            ))
    return LLMResponse(
        content=content,
        stop_reason=msg.stop_reason,
        model=msg.model,
        usage={
            "input_tokens": getattr(msg.usage, "input_tokens", 0),
            "output_tokens": getattr(msg.usage, "output_tokens", 0),
        },
    )
