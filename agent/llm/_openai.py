"""OpenAI provider — GPT-4o, o1, etc.

Wraps the openai SDK to implement the LLMProvider protocol.
Handles streaming with manual accumulation, tool format translation,
and error mapping to common LLM error types.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

try:
    import openai
except ImportError:
    openai = None  # type: ignore

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

log = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT models via the openai SDK."""

    def __init__(self) -> None:
        if openai is None:
            raise LLMError(
                "The 'openai' package is not installed. Install it with: pip install openai"
            )
        self._client = openai.OpenAI()

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

        # Prepend system message
        all_messages = [{"role": "system", "content": system}] + native_messages

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": all_messages,
            "stream": True,
        }
        if native_tools:
            kwargs["tools"] = native_tools

        try:
            stream = self._client.chat.completions.create(**kwargs)

            # Accumulate streaming response
            text_parts: list[str] = []
            # tool_calls_acc: dict mapping index -> {id, name, arguments_parts}
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            finish_reason = None
            resp_model = ""

            for chunk in stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                resp_model = chunk.model or resp_model

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta
                if delta is None:
                    continue

                # Text delta
                if delta.content:
                    text_parts.append(delta.content)
                    if on_text:
                        on_text(delta.content)

                # Tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": "",
                                "name": "",
                                "arguments_parts": [],
                            }
                        acc = tool_calls_acc[idx]
                        if tc_delta.id:
                            acc["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                acc["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                acc["arguments_parts"].append(tc_delta.function.arguments)

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

        return _build_response(
            text_parts=text_parts,
            tool_calls_acc=tool_calls_acc,
            finish_reason=finish_reason,
            model=resp_model,
        )

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
        all_messages = [{"role": "system", "content": system}] + native_messages

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": all_messages,
        }

        try:
            if timeout:
                client = openai.OpenAI(timeout=timeout)
            else:
                client = self._client
            response = client.chat.completions.create(**kwargs)
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
        """Convert MCP-format tool definitions to OpenAI function-calling format.

        MCP:    {"name": ..., "description": ..., "input_schema": {...}}
        OpenAI: {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
        """
        result = []
        for tool in tools:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
            )
        return result

    def convert_messages(self, messages: list[dict]) -> list[dict]:
        """Convert messages with common types to OpenAI chat format.

        Key differences from Anthropic:
        - ToolUseBlock in assistant content -> tool_calls field on assistant message
        - ToolResultBlock -> separate {"role": "tool"} messages
        - ImageBlock -> {"type": "image_url", ...} in content
        """
        result = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                result.append(msg)
                continue

            if not isinstance(content, list):
                result.append(msg)
                continue

            # Separate content into text/image parts vs tool-use vs tool-result
            content_parts: list[dict] = []
            tool_calls: list[dict] = []
            tool_results: list[dict] = []

            for block in content:
                if isinstance(block, TextBlock):
                    content_parts.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{block.media_type};base64,{block.data}",
                            },
                        }
                    )
                elif isinstance(block, ToolUseBlock):
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input, sort_keys=True),
                            },
                        }
                    )
                elif isinstance(block, ToolResultBlock):
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": block.content,
                        }
                    )
                elif isinstance(block, dict):
                    # Pass through raw dicts (already native format)
                    if block.get("type") == "tool_result":
                        tool_results.append(
                            {
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": block.get("content", ""),
                            }
                        )
                    elif block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "id": block["id"],
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(
                                        block.get("input", {}), sort_keys=True
                                    ),
                                },
                            }
                        )
                    else:
                        content_parts.append(block)

            # Build the assistant or user message
            if role == "assistant" and tool_calls:
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if content_parts:
                    # If there's only one text part, use string content
                    if len(content_parts) == 1 and content_parts[0].get("type") == "text":
                        assistant_msg["content"] = content_parts[0]["text"]
                    else:
                        assistant_msg["content"] = content_parts
                else:
                    assistant_msg["content"] = None
                assistant_msg["tool_calls"] = tool_calls
                result.append(assistant_msg)
            elif content_parts:
                result.append({"role": role, "content": content_parts})
            elif not tool_calls and not tool_results:
                # Empty content — keep the message with empty string
                result.append({"role": role, "content": ""})

            # Tool results become separate top-level messages
            for tr in tool_results:
                result.append(tr)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _convert_vision_messages(self, messages: list[dict]) -> list[dict]:
        """Convert messages containing ImageBlock to OpenAI vision format."""
        result = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                result.append(msg)
                continue
            native_content = []
            for block in content:
                if isinstance(block, ImageBlock):
                    native_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{block.media_type};base64,{block.data}",
                            },
                        }
                    )
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


def _build_response(
    *,
    text_parts: list[str],
    tool_calls_acc: dict[int, dict[str, Any]],
    finish_reason: str | None,
    model: str,
) -> LLMResponse:
    """Build LLMResponse from accumulated streaming data."""
    content: list[TextBlock | ToolUseBlock] = []

    # Add text block if any text was accumulated
    full_text = "".join(text_parts)
    if full_text:
        content.append(TextBlock(text=full_text))

    # Add tool use blocks from accumulated tool calls
    for idx in sorted(tool_calls_acc.keys()):
        acc = tool_calls_acc[idx]
        arguments_str = "".join(acc["arguments_parts"])
        try:
            parsed_input = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError:
            log.warning("Failed to parse tool arguments: %s", arguments_str)
            parsed_input = {}
        content.append(
            ToolUseBlock(
                id=acc["id"],
                name=acc["name"],
                input=parsed_input,
            )
        )

    # Map OpenAI finish reasons to common stop reasons
    stop_reason = _map_finish_reason(finish_reason)

    return LLMResponse(
        content=content,
        stop_reason=stop_reason,
        model=model,
        usage={},
    )


def _to_response(response) -> LLMResponse:
    """Convert OpenAI ChatCompletion to LLMResponse."""
    choice = response.choices[0]
    message = choice.message
    content: list[TextBlock | ToolUseBlock] = []

    if message.content:
        content.append(TextBlock(text=message.content))

    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                parsed_input = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                log.warning("Failed to parse tool arguments: %s", tc.function.arguments)
                parsed_input = {}
            content.append(
                ToolUseBlock(
                    id=tc.id,
                    name=tc.function.name,
                    input=parsed_input,
                )
            )

    stop_reason = _map_finish_reason(choice.finish_reason)
    usage = {}
    if response.usage:
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    return LLMResponse(
        content=content,
        stop_reason=stop_reason,
        model=response.model,
        usage=usage,
    )


def _map_finish_reason(finish_reason: str | None) -> str:
    """Map OpenAI finish_reason to common stop_reason."""
    mapping = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "length",
        "content_filter": "end_turn",
    }
    return mapping.get(finish_reason or "stop", "end_turn")
