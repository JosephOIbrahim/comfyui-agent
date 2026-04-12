"""Ollama provider — local LLM via OpenAI-compatible API.

Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1.
This provider wraps the openai Python SDK with a custom base_url and a
dummy API key (Ollama doesn't require authentication).

Requires: pip install openai
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable

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

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

if openai is None:  # Cycle 58: surface unavailability at import time, not first use
    log.debug("openai package not installed; OllamaProvider unavailable (pip install openai)")


def _require_openai() -> None:
    """Raise a clear error if the openai package is not installed."""
    if openai is None:
        raise LLMError(
            "The 'openai' package is required for the Ollama provider. "
            "Install it with: pip install openai"
        )


class OllamaProvider(LLMProvider):
    """Local LLM via Ollama's OpenAI-compatible API."""

    def __init__(self) -> None:
        _require_openai()
        from ..config import OLLAMA_BASE_URL

        self._client = openai.OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  # Dummy key — Ollama ignores it but the SDK requires one
        )
        log.info("Ollama provider ready (base_url=%s)", OLLAMA_BASE_URL)

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

        # Prepend system message (OpenAI format)
        if system:
            native_messages = [{"role": "system", "content": system}] + native_messages

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": native_messages,
            "stream": True,
            # Cycle 7: ask Ollama's OpenAI-compat endpoint to emit a final
            # chunk containing token usage so compact() can trigger context
            # compaction correctly. Without this, usage stays empty and
            # compact() falls back to a len(content)//4 heuristic that
            # drifts vs. real token counts, causing silent context overflow.
            "stream_options": {"include_usage": True},
        }
        if native_tools:
            kwargs["tools"] = native_tools

        try:
            stream = self._client.chat.completions.create(**kwargs)
            return self._consume_stream(stream, on_text=on_text)
        except openai.AuthenticationError as e:
            raise LLMAuthError(str(e)) from e
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except openai.APIConnectionError as e:
            raise LLMConnectionError(str(e)) from e
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                raise LLMServerError(str(e), status_code=e.status_code) from e
            raise LLMError(str(e)) from e
        except openai.APIError as e:
            raise LLMError(str(e)) from e

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

        if system:
            native_messages = [{"role": "system", "content": system}] + native_messages

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": native_messages,
        }
        if timeout:
            kwargs["timeout"] = timeout

        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as e:
            raise LLMAuthError(str(e)) from e
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except openai.APIConnectionError as e:
            raise LLMConnectionError(str(e)) from e
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                raise LLMServerError(str(e), status_code=e.status_code) from e
            raise LLMError(str(e)) from e
        except openai.APIError as e:
            raise LLMError(str(e)) from e

        return _to_response(response)

    def convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert MCP-format tool defs to OpenAI function-calling format."""
        if not tools:
            return []
        result = []
        for tool in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        return result

    def convert_messages(self, messages: list[dict]) -> list[dict]:
        """Convert common types to OpenAI chat message format."""
        result = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Simple string content — pass through
            if isinstance(content, str):
                result.append({"role": role, "content": content})
                continue

            if isinstance(content, list):
                # Cycle 20: filter out ThinkingBlocks — Ollama has no
                # thinking concept; raw objects would corrupt the request.
                content = [b for b in content if not isinstance(b, ThinkingBlock)]

                # Check if this is a tool-result message
                tool_results = [b for b in content if isinstance(b, ToolResultBlock)]
                if tool_results:
                    for tr in tool_results:
                        result.append({
                            "role": "tool",
                            "tool_call_id": tr.tool_use_id,
                            "content": tr.content,
                        })
                    continue

                # Check if this contains tool-use blocks (assistant message)
                tool_uses = [b for b in content if isinstance(b, ToolUseBlock)]
                text_blocks = [b for b in content if isinstance(b, TextBlock)]
                if tool_uses:
                    text_content = " ".join(b.text for b in text_blocks) if text_blocks else None
                    tool_calls = []
                    for tu in tool_uses:
                        import json
                        tool_calls.append({
                            "id": tu.id,
                            "type": "function",
                            "function": {
                                "name": tu.name,
                                "arguments": json.dumps(tu.input, sort_keys=True, allow_nan=False),  # Cycle 61
                            },
                        })
                    result.append({
                        "role": "assistant",
                        "content": text_content,
                        "tool_calls": tool_calls,
                    })
                    continue

                # Plain text blocks only
                if text_blocks:
                    combined = "\n".join(b.text for b in text_blocks)
                    result.append({"role": role, "content": combined})
                    continue

                # Dict blocks or other — pass through as-is
                result.append({"role": role, "content": content})
            else:
                result.append(msg)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _consume_stream(
        self,
        stream,
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Consume an OpenAI streaming response and build an LLMResponse."""
        import json

        text_parts: list[str] = []
        # tool_calls accumulator: {index: {id, name, arguments_parts}}
        tool_acc: dict[int, dict] = {}
        finish_reason = ""
        model_name = ""
        # Cycle 7: capture usage from the final chunk so compact() can
        # correctly trigger context compaction. OpenAI-compat streams emit
        # a terminal chunk with a populated `usage` object (and empty
        # `choices`) when stream_options.include_usage is requested.
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        for chunk in stream:
            # Usage may arrive on the terminal chunk even when choices is empty.
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                pt = getattr(chunk_usage, "prompt_tokens", None)
                ct = getattr(chunk_usage, "completion_tokens", None)
                if isinstance(pt, int):
                    prompt_tokens = pt
                if isinstance(ct, int):
                    completion_tokens = ct

            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            model_name = chunk.model or model_name

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            delta = choice.delta
            if delta is None:
                continue

            # Text content
            if delta.content:
                text_parts.append(delta.content)
                if on_text:
                    on_text(delta.content)

            # Tool calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_acc:
                        tool_acc[idx] = {
                            "id": tc_delta.id or "",
                            "name": "",
                            "arguments_parts": [],
                        }
                    acc = tool_acc[idx]
                    if tc_delta.id:
                        acc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            acc["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            acc["arguments_parts"].append(tc_delta.function.arguments)

        # Build content blocks
        content: list[TextBlock | ToolUseBlock] = []
        full_text = "".join(text_parts)
        if full_text:
            content.append(TextBlock(text=full_text))

        for _idx in sorted(tool_acc):
            acc = tool_acc[_idx]
            raw_args = "".join(acc["arguments_parts"])
            try:
                parsed_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                parsed_args = {"_raw": raw_args}
            tool_id = acc["id"] or f"call_{uuid.uuid4().hex[:24]}"
            content.append(ToolUseBlock(
                id=tool_id,
                name=acc["name"],
                input=parsed_args,
            ))

        # Map finish reason
        stop_reason = _map_stop_reason(finish_reason)

        # Cycle 7: populate usage from the final streaming chunk so
        # compact() can correctly trigger context compaction for Ollama
        # users. Falls back to empty dict if the endpoint didn't emit
        # usage (older Ollama versions without stream_options support).
        usage: dict[str, int] = {}
        if prompt_tokens is not None and completion_tokens is not None:
            usage = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
            }

        return LLMResponse(
            content=content,
            stop_reason=stop_reason,
            model=model_name,
            usage=usage,
        )

    def _convert_vision_messages(self, messages: list[dict]) -> list[dict]:
        """Convert messages containing ImageBlock to OpenAI vision format."""
        result = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                result.append({"role": msg["role"], "content": content})
                continue
            parts = []
            for block in content:
                if isinstance(block, ImageBlock):
                    parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.media_type};base64,{block.data}",
                        },
                    })
                elif isinstance(block, TextBlock):
                    parts.append({"type": "text", "text": block.text})
                elif isinstance(block, dict):
                    parts.append(block)
                else:
                    parts.append({"type": "text", "text": str(block)})
            result.append({"role": msg["role"], "content": parts})
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _map_stop_reason(finish_reason: str) -> str:
    """Map OpenAI finish_reason to our common stop_reason vocabulary."""
    mapping = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "length",
        "content_filter": "end_turn",
    }
    return mapping.get(finish_reason, "end_turn")


def _to_response(completion) -> LLMResponse:
    """Convert a non-streaming OpenAI ChatCompletion to LLMResponse."""
    import json

    choice = completion.choices[0]
    message = choice.message
    content: list[TextBlock | ToolUseBlock] = []

    if message.content:
        content.append(TextBlock(text=message.content))

    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            content.append(ToolUseBlock(
                id=tc.id,
                name=tc.function.name,
                input=args,
            ))

    usage = {}
    if completion.usage:
        usage = {
            "input_tokens": getattr(completion.usage, "prompt_tokens", 0),
            "output_tokens": getattr(completion.usage, "completion_tokens", 0),
        }

    return LLMResponse(
        content=content,
        stop_reason=_map_stop_reason(choice.finish_reason or "stop"),
        model=completion.model or "",
        usage=usage,
    )
