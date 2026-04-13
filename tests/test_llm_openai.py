"""Tests for the OpenAI LLM provider.

All tests mock the openai SDK — no real API calls, no network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai", reason="openai SDK not installed")

from agent.llm._types import (
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


# ---------------------------------------------------------------------------
# Helpers — build mock OpenAI SDK objects
# ---------------------------------------------------------------------------


def _make_usage(prompt=10, completion=20):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)


def _make_message(content="Hello", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _make_choice(message=None, finish_reason="stop"):
    if message is None:
        message = _make_message()
    return SimpleNamespace(message=message, finish_reason=finish_reason)


def _make_completion(choices=None, model="gpt-4o", usage=None):
    if choices is None:
        choices = [_make_choice()]
    if usage is None:
        usage = _make_usage()
    return SimpleNamespace(choices=choices, model=model, usage=usage)


def _make_stream_chunk(
    content=None,
    tool_calls=None,
    finish_reason=None,
    model="gpt-4o",
    usage=None,
):
    """Build a single streaming chunk."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(
        choices=[choice],
        model=model,
        usage=usage,
    )


def _make_usage_chunk(prompt=10, completion=20, model="gpt-4o"):
    """Build the final usage-only chunk (empty choices)."""
    return SimpleNamespace(
        choices=[],
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


def _make_tc_delta(index=0, tc_id=None, name=None, arguments=None):
    """Build a tool_call delta fragment."""
    func = None
    if name is not None or arguments is not None:
        func = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tc_id, function=func)


def _get_provider():
    """Instantiate OpenAIProvider with a mocked openai client."""
    with patch("agent.llm._openai.openai") as mock_openai:
        mock_openai.OpenAI.return_value = MagicMock()
        # Ensure the module-level `openai` reference is not None
        mock_openai.__bool__ = lambda self: True
        # Re-assign exception classes so isinstance checks work
        import openai as real_openai

        mock_openai.AuthenticationError = real_openai.AuthenticationError
        mock_openai.RateLimitError = real_openai.RateLimitError
        mock_openai.APIConnectionError = real_openai.APIConnectionError
        mock_openai.APIStatusError = real_openai.APIStatusError
        mock_openai.APIError = real_openai.APIError

        from agent.llm._openai import OpenAIProvider

        provider = OpenAIProvider()
    return provider


# ---------------------------------------------------------------------------
# Message Conversion
# ---------------------------------------------------------------------------


class TestOpenAIMessageConversion:
    """Test convert_messages for various content block types."""

    def setup_method(self):
        self.provider = _get_provider()

    def test_message_conversion_text(self):
        """Plain string content passes through unchanged."""
        msgs = [{"role": "user", "content": "Hello world"}]
        result = self.provider.convert_messages(msgs)
        assert result == [{"role": "user", "content": "Hello world"}]

    def test_message_conversion_text_block(self):
        """TextBlock in content list converts to OpenAI text dict."""
        msgs = [{"role": "user", "content": [TextBlock(text="Hi there")]}]
        result = self.provider.convert_messages(msgs)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == [{"type": "text", "text": "Hi there"}]

    def test_message_conversion_tool_use(self):
        """ToolUseBlock converts to OpenAI tool_calls on assistant message."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    ToolUseBlock(
                        id="call_123",
                        name="get_weather",
                        input={"city": "NYC"},
                    ),
                ],
            }
        ]
        result = self.provider.convert_messages(msgs)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] is None  # No text parts
        assert len(result[0]["tool_calls"]) == 1
        tc = result[0]["tool_calls"][0]
        assert tc["id"] == "call_123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed == {"city": "NYC"}

    def test_message_conversion_tool_result(self):
        """ToolResultBlock converts to a separate role=tool message."""
        msgs = [
            {
                "role": "user",
                "content": [
                    ToolResultBlock(tool_use_id="call_123", content="72F"),
                ],
            }
        ]
        result = self.provider.convert_messages(msgs)
        assert any(m["role"] == "tool" for m in result)
        tool_msg = [m for m in result if m["role"] == "tool"][0]
        assert tool_msg["tool_call_id"] == "call_123"
        assert tool_msg["content"] == "72F"

    def test_message_conversion_image(self):
        """ImageBlock converts to OpenAI image_url format."""
        msgs = [
            {
                "role": "user",
                "content": [
                    ImageBlock(data="abc123", media_type="image/png"),
                ],
            }
        ]
        result = self.provider.convert_messages(msgs)
        part = result[0]["content"][0]
        assert part["type"] == "image_url"
        assert part["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_message_conversion_system(self):
        """System messages (string content) pass through."""
        msgs = [{"role": "system", "content": "You are helpful."}]
        result = self.provider.convert_messages(msgs)
        assert result == msgs

    def test_message_conversion_thinking_block_skipped(self):
        """ThinkingBlock is silently dropped (OpenAI has no equivalent)."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    ThinkingBlock(thinking="Let me think..."),
                    TextBlock(text="Answer"),
                ],
            }
        ]
        result = self.provider.convert_messages(msgs)
        # Should have text content but no thinking block
        assert result[0]["content"] == [{"type": "text", "text": "Answer"}]

    def test_message_conversion_mixed_text_and_tool_use(self):
        """Assistant message with both text and tool_use blocks."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    TextBlock(text="Checking..."),
                    ToolUseBlock(id="call_1", name="search", input={"q": "test"}),
                ],
            }
        ]
        result = self.provider.convert_messages(msgs)
        msg = result[0]
        assert msg["role"] == "assistant"
        # Single text part -> string content
        assert msg["content"] == "Checking..."
        assert len(msg["tool_calls"]) == 1

    def test_empty_content_list(self):
        """Empty content list produces a message with empty string content."""
        msgs = [{"role": "user", "content": []}]
        result = self.provider.convert_messages(msgs)
        assert result[0]["content"] == ""


# ---------------------------------------------------------------------------
# Tool Schema Conversion
# ---------------------------------------------------------------------------


class TestOpenAIToolConversion:
    """Test convert_tools: MCP format -> OpenAI function-calling format."""

    def setup_method(self):
        self.provider = _get_provider()

    def test_tool_schema_conversion(self):
        """MCP tool schema converts to OpenAI function format."""
        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            }
        ]
        result = self.provider.convert_tools(tools)
        assert len(result) == 1
        t = result[0]
        assert t["type"] == "function"
        assert t["function"]["name"] == "get_weather"
        assert t["function"]["description"] == "Get current weather"
        assert t["function"]["parameters"]["required"] == ["city"]

    def test_tool_schema_empty_tools(self):
        """Empty tools list returns empty list."""
        assert self.provider.convert_tools([]) == []

    def test_tool_schema_missing_optional_fields(self):
        """Tool with missing optional fields still converts."""
        tools = [{"name": "ping"}]
        result = self.provider.convert_tools(tools)
        assert result[0]["function"]["name"] == "ping"
        assert result[0]["function"]["description"] == ""
        assert result[0]["function"]["parameters"] == {}


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestOpenAIStreaming:
    """Test stream() with mocked OpenAI client."""

    def setup_method(self):
        self.provider = _get_provider()

    def test_stream_text(self):
        """Streaming text chunks fire on_text callback."""
        chunks = [
            _make_stream_chunk(content="Hello"),
            _make_stream_chunk(content=" world"),
            _make_stream_chunk(finish_reason="stop"),
            _make_usage_chunk(prompt=5, completion=2),
        ]
        self.provider._client.chat.completions.create.return_value = iter(chunks)

        text_parts = []
        resp = self.provider.stream(
            model="gpt-4o",
            max_tokens=100,
            system="Be helpful",
            tools=[],
            messages=[{"role": "user", "content": "Hi"}],
            on_text=lambda t: text_parts.append(t),
        )

        assert isinstance(resp, LLMResponse)
        assert resp.content[0].text == "Hello world"
        assert text_parts == ["Hello", " world"]
        assert resp.stop_reason == "end_turn"
        assert resp.usage["input_tokens"] == 5
        assert resp.usage["output_tokens"] == 2

    def test_stream_tool_use(self):
        """Streaming tool call deltas accumulate into ToolUseBlock."""
        chunks = [
            _make_stream_chunk(
                tool_calls=[_make_tc_delta(0, tc_id="call_abc", name="search")],
            ),
            _make_stream_chunk(
                tool_calls=[_make_tc_delta(0, arguments='{"q":')],
            ),
            _make_stream_chunk(
                tool_calls=[_make_tc_delta(0, arguments='"test"}')],
            ),
            _make_stream_chunk(finish_reason="tool_calls"),
            _make_usage_chunk(),
        ]
        self.provider._client.chat.completions.create.return_value = iter(chunks)

        resp = self.provider.stream(
            model="gpt-4o",
            max_tokens=100,
            system="",
            tools=[{"name": "search", "description": "", "input_schema": {}}],
            messages=[{"role": "user", "content": "search for test"}],
        )

        assert resp.stop_reason == "tool_use"
        tool_block = resp.content[0]
        assert isinstance(tool_block, ToolUseBlock)
        assert tool_block.id == "call_abc"
        assert tool_block.name == "search"
        assert tool_block.input == {"q": "test"}

    def test_stream_empty_delta(self):
        """Chunks with no content and no tool_calls are skipped."""
        chunks = [
            _make_stream_chunk(),  # Both content and tool_calls are None
            _make_stream_chunk(content="ok"),
            _make_stream_chunk(finish_reason="stop"),
            _make_usage_chunk(),
        ]
        self.provider._client.chat.completions.create.return_value = iter(chunks)

        resp = self.provider.stream(
            model="gpt-4o",
            max_tokens=100,
            system="",
            tools=[],
            messages=[{"role": "user", "content": "test"}],
        )
        assert resp.content[0].text == "ok"


# ---------------------------------------------------------------------------
# Non-streaming (create)
# ---------------------------------------------------------------------------


class TestOpenAICreate:
    """Test create() with mocked responses."""

    def setup_method(self):
        self.provider = _get_provider()

    def test_create_non_streaming(self):
        """Non-streaming call returns text content."""
        completion = _make_completion()
        self.provider._client.chat.completions.create.return_value = completion

        resp = self.provider.create(
            model="gpt-4o",
            max_tokens=100,
            system="Be helpful",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert isinstance(resp, LLMResponse)
        assert resp.content[0].text == "Hello"
        assert resp.model == "gpt-4o"
        assert resp.usage["input_tokens"] == 10

    def test_create_with_tool_calls(self):
        """Non-streaming response with tool calls."""
        tc = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="search", arguments='{"q": "test"}'),
        )
        msg = _make_message(content=None, tool_calls=[tc])
        choice = _make_choice(message=msg, finish_reason="tool_calls")
        completion = _make_completion(choices=[choice])

        self.provider._client.chat.completions.create.return_value = completion

        resp = self.provider.create(
            model="gpt-4o",
            max_tokens=100,
            system="",
            messages=[{"role": "user", "content": "search"}],
        )

        assert resp.stop_reason == "tool_use"
        assert isinstance(resp.content[0], ToolUseBlock)
        assert resp.content[0].input == {"q": "test"}

    def test_max_tokens_passed(self):
        """max_tokens parameter is forwarded to the client."""
        completion = _make_completion()
        self.provider._client.chat.completions.create.return_value = completion

        self.provider.create(
            model="gpt-4o",
            max_tokens=4096,
            system="test",
            messages=[{"role": "user", "content": "Hi"}],
        )

        call_kwargs = self.provider._client.chat.completions.create.call_args
        assert call_kwargs.kwargs["max_tokens"] == 4096

    def test_model_parameter(self):
        """Model name is passed to the client correctly."""
        completion = _make_completion()
        self.provider._client.chat.completions.create.return_value = completion

        self.provider.create(
            model="gpt-4o-mini",
            max_tokens=100,
            system="",
            messages=[{"role": "user", "content": "Hi"}],
        )

        call_kwargs = self.provider._client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"

    def test_system_prepended(self):
        """System prompt is prepended as a system message."""
        completion = _make_completion()
        self.provider._client.chat.completions.create.return_value = completion

        self.provider.create(
            model="gpt-4o",
            max_tokens=100,
            system="You are a test bot",
            messages=[{"role": "user", "content": "Hi"}],
        )

        call_kwargs = self.provider._client.chat.completions.create.call_args
        msgs = call_kwargs.kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "You are a test bot"}

    def test_create_with_timeout(self):
        """Timeout parameter creates a new client."""
        import openai as real_openai

        with patch("agent.llm._openai.openai") as mock_mod:
            mock_mod.OpenAI.return_value = MagicMock()
            mock_mod.AuthenticationError = real_openai.AuthenticationError
            mock_mod.RateLimitError = real_openai.RateLimitError
            mock_mod.APIConnectionError = real_openai.APIConnectionError
            mock_mod.APIStatusError = real_openai.APIStatusError
            mock_mod.APIError = real_openai.APIError

            from agent.llm._openai import OpenAIProvider

            prov = OpenAIProvider()

            completion = _make_completion()
            # The timeout client is a new OpenAI() call
            mock_mod.OpenAI.return_value.chat.completions.create.return_value = completion

            prov.create(
                model="gpt-4o",
                max_tokens=100,
                system="",
                messages=[{"role": "user", "content": "Hi"}],
                timeout=30.0,
            )

            # OpenAI() called twice: once in __init__, once for timeout
            assert mock_mod.OpenAI.call_count == 2
            assert mock_mod.OpenAI.call_args.kwargs["timeout"] == 30.0


# ---------------------------------------------------------------------------
# Error Mapping
# ---------------------------------------------------------------------------


class TestOpenAIErrorMapping:
    """Test that openai exceptions map to LLM error hierarchy."""

    def setup_method(self):
        self.provider = _get_provider()

    def test_error_auth(self):
        """AuthenticationError maps to LLMAuthError."""
        import openai as real_openai

        exc = real_openai.AuthenticationError(
            message="bad key",
            response=MagicMock(status_code=401),
            body=None,
        )
        self.provider._client.chat.completions.create.side_effect = exc

        with pytest.raises(LLMAuthError):
            self.provider.create(model="gpt-4o", max_tokens=100, system="", messages=[])

    def test_error_rate_limit(self):
        """RateLimitError maps to LLMRateLimitError."""
        import openai as real_openai

        exc = real_openai.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
        self.provider._client.chat.completions.create.side_effect = exc

        with pytest.raises(LLMRateLimitError):
            self.provider.create(model="gpt-4o", max_tokens=100, system="", messages=[])

    def test_error_connection(self):
        """APIConnectionError maps to LLMConnectionError."""
        import openai as real_openai

        exc = real_openai.APIConnectionError(request=MagicMock())
        self.provider._client.chat.completions.create.side_effect = exc

        with pytest.raises(LLMConnectionError):
            self.provider.create(model="gpt-4o", max_tokens=100, system="", messages=[])

    def test_error_server_5xx(self):
        """APIStatusError with 5xx maps to LLMServerError."""
        import openai as real_openai

        resp_mock = MagicMock()
        resp_mock.status_code = 500
        exc = real_openai.APIStatusError(
            message="server error",
            response=resp_mock,
            body=None,
        )
        self.provider._client.chat.completions.create.side_effect = exc

        with pytest.raises(LLMServerError):
            self.provider.create(model="gpt-4o", max_tokens=100, system="", messages=[])

    def test_error_generic(self):
        """Generic APIError maps to LLMError."""
        import openai as real_openai

        exc = real_openai.APIError(
            message="something broke",
            request=MagicMock(),
            body=None,
        )
        self.provider._client.chat.completions.create.side_effect = exc

        with pytest.raises(LLMError):
            self.provider.create(model="gpt-4o", max_tokens=100, system="", messages=[])

    def test_error_stream_auth(self):
        """Auth error during streaming also maps correctly."""
        import openai as real_openai

        exc = real_openai.AuthenticationError(
            message="bad key",
            response=MagicMock(status_code=401),
            body=None,
        )
        self.provider._client.chat.completions.create.side_effect = exc

        with pytest.raises(LLMAuthError):
            self.provider.stream(
                model="gpt-4o",
                max_tokens=100,
                system="",
                tools=[],
                messages=[{"role": "user", "content": "hi"}],
            )
