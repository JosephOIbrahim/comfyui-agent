"""Google Gemini provider via the google-genai SDK.

Wraps the new google.genai client to implement the LLMProvider protocol.
Handles streaming, tool format conversion, and error translation.

Requires: pip install google-genai  (optional dependency, not in pyproject.toml)
"""

from __future__ import annotations

import base64
import logging
import os
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
    ToolResultBlock,
    ToolUseBlock,
)

try:
    import google.genai as genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
except ImportError:
    genai = None  # type: ignore[assignment]
    genai_errors = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

if genai is None:  # Cycle 58: surface unavailability at import time, not first use
    log.debug("google-genai not installed; GeminiProvider unavailable (pip install google-genai)")


def _require_sdk() -> None:
    """Raise a clear error if google-genai is not installed."""
    if genai is None:
        raise LLMError(
            "Google Gemini provider requires the 'google-genai' package. "
            "Install it with: pip install google-genai"
        )


class GeminiProvider(LLMProvider):
    """Google Gemini via the google-genai SDK."""

    def __init__(self) -> None:
        _require_sdk()
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise LLMAuthError(
                "Gemini provider requires GEMINI_API_KEY or GOOGLE_API_KEY env var."
            )
        self._client = genai.Client(api_key=api_key)

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

        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            tools=native_tools or None,
        )

        try:
            stream = self._client.models.generate_content_stream(
                model=model,
                contents=native_messages,
                config=config,
            )

            accumulated_text = ""
            tool_calls: list[ToolUseBlock] = []
            usage_meta: dict[str, Any] = {}

            for chunk in stream:
                # Extract usage from final chunk if available
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    meta = chunk.usage_metadata
                    usage_meta = {
                        "input_tokens": getattr(meta, "prompt_token_count", 0) or 0,
                        "output_tokens": (getattr(meta, "candidates_token_count", 0) or 0),
                    }

                if not chunk.candidates:
                    continue

                for candidate in chunk.candidates:
                    if not candidate.content or not candidate.content.parts:
                        continue
                    for part in candidate.content.parts:
                        # Text delta
                        if hasattr(part, "text") and part.text:
                            accumulated_text += part.text
                            if on_text:
                                on_text(part.text)

                        # Thinking delta (Gemini 2.5 models)
                        if hasattr(part, "thought") and part.thought:
                            if on_thinking:
                                on_thinking(part.text or "")

                        # Function call
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            tool_calls.append(
                                ToolUseBlock(
                                    id=f"call_{uuid.uuid4().hex[:24]}",
                                    name=fc.name,
                                    input=dict(fc.args) if fc.args else {},
                                )
                            )

        except Exception as e:
            _translate_error(e)

        # Build response
        content: list[TextBlock | ToolUseBlock] = []
        if accumulated_text:
            content.append(TextBlock(text=accumulated_text))
        content.extend(tool_calls)

        stop_reason = "tool_use" if tool_calls else "end_turn"

        return LLMResponse(
            content=content,
            stop_reason=stop_reason,
            model=model,
            usage=usage_meta,
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
        native_messages = self.convert_messages(messages)

        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )

        http_options = None
        if timeout:
            http_options = genai_types.HttpOptions(timeout=int(timeout * 1000))

        try:
            if timeout:
                client = genai.Client(
                    api_key=self._client.api_key,
                    http_options=http_options,
                )
            else:
                client = self._client

            response = client.models.generate_content(
                model=model,
                contents=native_messages,
                config=config,
            )
        except Exception as e:
            _translate_error(e)

        return _to_response(response, model)

    def convert_tools(self, tools: list[dict]) -> list[Any]:
        """Convert MCP-format tool definitions to Gemini function declarations."""
        if not tools:
            return []

        declarations = []
        for tool in tools:
            # Build OpenAPI-style parameters from input_schema
            schema = tool.get("input_schema", {})
            parameters = _convert_schema(schema) if schema else None

            decl = genai_types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=parameters,
            )
            declarations.append(decl)

        return [genai_types.Tool(function_declarations=declarations)]

    def convert_messages(self, messages: list[dict]) -> list[dict]:
        """Convert common types to Gemini contents format.

        Gemini uses "model" instead of "assistant" and structures content
        as a list of Part objects within each message.
        """
        result = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Map roles: Gemini uses "model" not "assistant"
            gemini_role = "model" if role == "assistant" else "user"

            if isinstance(content, str):
                result.append(
                    {
                        "role": gemini_role,
                        "parts": [genai_types.Part.from_text(text=content)],
                    }
                )
                continue

            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, TextBlock):
                        parts.append(genai_types.Part.from_text(text=block.text))

                    elif isinstance(block, ToolUseBlock):
                        parts.append(
                            genai_types.Part.from_function_call(
                                name=block.name,
                                args=block.input,
                            )
                        )

                    elif isinstance(block, ToolResultBlock):
                        # Find the tool name from conversation context
                        tool_name = _find_tool_name(messages, block.tool_use_id)
                        parts.append(
                            genai_types.Part.from_function_response(
                                name=tool_name,
                                response={"result": block.content},
                            )
                        )

                    elif isinstance(block, ImageBlock):
                        parts.append(
                            genai_types.Part.from_bytes(
                                data=base64.b64decode(block.data),
                                mime_type=block.media_type,
                            )
                        )

                    elif isinstance(block, dict):
                        # Pass-through for already-native dicts
                        if "text" in block:
                            parts.append(genai_types.Part.from_text(text=block["text"]))
                        else:
                            parts.append(genai_types.Part.from_text(text=str(block)))
                    else:
                        parts.append(genai_types.Part.from_text(text=str(block)))

                if parts:
                    result.append({"role": gemini_role, "parts": parts})
            else:
                result.append(
                    {
                        "role": gemini_role,
                        "parts": [genai_types.Part.from_text(text=str(content))],
                    }
                )

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_tool_name(messages: list[dict], tool_use_id: str) -> str:
    """Walk conversation to find the tool name matching a tool_use_id."""
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, ToolUseBlock) and block.id == tool_use_id:
                return block.name
    return "unknown_tool"


def _convert_schema(schema: dict) -> dict:
    """Convert JSON Schema to Gemini-compatible OpenAPI subset.

    Gemini function declarations accept an OpenAPI-style parameters object.
    We strip unsupported keys and ensure the structure is clean.
    """
    if not schema:
        return {}

    result: dict[str, Any] = {}

    schema_type = schema.get("type")
    if schema_type:
        result["type"] = schema_type.upper()

    if "description" in schema:
        result["description"] = schema["description"]

    if "enum" in schema:
        result["enum"] = schema["enum"]

    if "properties" in schema:
        result["properties"] = {k: _convert_schema(v) for k, v in schema["properties"].items()}

    if "required" in schema:
        result["required"] = schema["required"]

    if "items" in schema:
        result["items"] = _convert_schema(schema["items"])

    return result


def _to_response(response: Any, model: str) -> LLMResponse:
    """Convert Gemini GenerateContentResponse to LLMResponse."""
    content: list[TextBlock | ToolUseBlock] = []
    tool_calls_found = False

    if response.candidates:
        for candidate in response.candidates:
            if not candidate.content or not candidate.content.parts:
                continue
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    content.append(TextBlock(text=part.text))
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    content.append(
                        ToolUseBlock(
                            id=f"call_{uuid.uuid4().hex[:24]}",
                            name=fc.name,
                            input=dict(fc.args) if fc.args else {},
                        )
                    )
                    tool_calls_found = True

    # Extract usage
    usage: dict[str, int] = {}
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        meta = response.usage_metadata
        usage = {
            "input_tokens": getattr(meta, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(meta, "candidates_token_count", 0) or 0,
        }

    stop_reason = "tool_use" if tool_calls_found else "end_turn"

    return LLMResponse(
        content=content,
        stop_reason=stop_reason,
        model=model,
        usage=usage,
    )


def _translate_error(e: Exception) -> None:
    """Translate google-genai exceptions to LLM error hierarchy.

    Always raises — never returns.
    """
    if genai_errors is None:
        raise LLMError(str(e)) from e

    if isinstance(e, genai_errors.ClientError):
        msg = str(e)
        # Check for auth-related client errors
        if (
            getattr(e, "code", None) in (401, 403)
            or getattr(e, "status_code", None) in (401, 403)
            or "401" in msg
            or "403" in msg
            or "API key" in msg.lower()
        ):
            raise LLMAuthError(msg) from e
        # Check for rate limit
        if (
            getattr(e, "code", None) == 429
            or getattr(e, "status_code", None) == 429
            or "429" in msg
            or "rate" in msg.lower()
        ):
            raise LLMRateLimitError(msg) from e
        raise LLMError(msg) from e

    if isinstance(e, genai_errors.ServerError):
        msg = str(e)
        if (
            getattr(e, "code", None) == 503
            or getattr(e, "status_code", None) == 503
            or "503" in msg
        ):
            raise LLMServerError(msg, status_code=503) from e
        raise LLMServerError(msg, status_code=500) from e

    # Connection-level errors
    if isinstance(e, (ConnectionError, TimeoutError, OSError)):
        raise LLMConnectionError(str(e)) from e

    raise LLMError(str(e)) from e
