"""Tests for the Ollama LLM provider.

All tests mock the openai SDK (Ollama uses OpenAI-compatible API) — no real
API calls, no network, no running Ollama server.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Ollama provider uses the openai SDK (OpenAI-compatible API)
openai = pytest.importorskip("openai", reason="openai SDK not installed")

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
# Helpers
# ---------------------------------------------------------------------------


def _make_usage(prompt=10, completion=20):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)


def _make_message(content="Hello", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _make_choice(message=None, finish_reason="stop"):
    if message is None:
        message = _make_message()
    return SimpleNamespace(message=message, finish_reason=finish_reason)


def _make_completion(choices=None, model="llama3.1", usage=None):
    if choices is None:
        choices = [_make_choice()]
    if usage is None:
        usage = _make_usage()
    return SimpleNamespace(choices=choices, model=model, usage=usage)


def _make_stream_chunk(
    content=None,
    tool_calls=None,
    finish_reason=None,
    model="llama3.1",
    usage=None,
):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def _make_usage_chunk(prompt=10, completion=20, model="llama3.1"):
    return SimpleNamespace(
        choices=[],
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


def _make_tc_delta(index=0, tc_id=None, name=None, arguments=None):
    func = None
    if name is not None or arguments is not None:
        func = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tc_id, function=func)


def _get_provider():
    """Instantiate OllamaProvider with mocked openai client and config."""
    with (
        patch("agent.llm._ollama.openai") as mock_openai,
        patch("agent.config.OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    ):
        mock_openai.OpenAI.return_value = MagicMock()

        import openai as real_openai

        mock_openai.AuthenticationError = real_openai.AuthenticationError
        mock_openai.RateLimitError = real_openai.RateLimitError
        mock_openai.APIConnectionError = real_openai.APIConnectionError
        mock_openai.APIStatusError = real_openai.APIStatusError
        mock_openai.APIError = real_openai.APIError

        from agent.llm._ollama import OllamaProvider

        provider = OllamaProvider()
    return provider


# ---------------------------------------------------------------------------
# Message Conversion
# ---------------------------------------------------------------------------


class TestOllamaMessageConversion:
    def setup_method(self):
        self.provider = _get_provider()

    def test_message_conversion_text(self):
        """Plain string content passes through."""
        msgs = [{"role": "user", "content": "Hello"}]
        result = self.provider.convert_messages(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_message_conversion_text_blocks(self):
        """TextBlock list joins into single string."""
        msgs = [
            {
                "role": "user",
                "content": [TextBlock(text="Hello"), TextBlock(text="World")],
            }
        ]
        result = self.provider.convert_messages(msgs)
        assert result[0]["content"] == "Hello\nWorld"

    def test_message_conversion_tool_use(self):
        """ToolUseBlock converts to OpenAI-style tool_calls."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    ToolUseBlock(id="call_1", name="search", input={"q": "test"}),
                ],
            }
        ]
        result = self.provider.convert_messages(msgs)
        assert result[0]["role"] == "assistant"
        assert len(result[0]["tool_calls"]) == 1
        tc = result[0]["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "search"
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed == {"q": "test"}

    def test_message_conversion_tool_result(self):
        """ToolResultBlock converts to role=tool message."""
        msgs = [
            {
                "role": "user",
                "content": [
                    ToolResultBlock(tool_use_id="call_1", content="result here"),
                ],
            }
        ]
        result = self.provider.convert_messages(msgs)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "result here"

    def test_message_conversion_thinking_filtered(self):
        """ThinkingBlock is filtered out before processing."""
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
        assert result[0]["content"] == "Answer"

    def test_message_conversion_image(self):
        """ImageBlock in vision messages converts to data URL."""
        msgs = [
            {
                "role": "user",
                "content": [
                    ImageBlock(data="abc123", media_type="image/png"),
                ],
            }
        ]
        result = self.provider._convert_vision_messages(msgs)
        part = result[0]["content"][0]
        assert part["type"] == "image_url"
        assert part["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_message_conversion_system_prepend(self):
        """System prompt is prepended in stream() and create()."""
        # This is tested implicitly through stream/create, but let's verify
        # the convert_messages doesn't eat system messages
        msgs = [{"role": "system", "content": "Be helpful"}]
        result = self.provider.convert_messages(msgs)
        assert result == [{"role": "system", "content": "Be helpful"}]

    def test_message_conversion_mixed_tool_and_text(self):
        """Assistant with both text and tool_use blocks."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    TextBlock(text="Searching..."),
                    ToolUseBlock(id="call_1", name="search", input={"q": "x"}),
                ],
            }
        ]
        result = self.provider.convert_messages(msgs)
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Searching..."
        assert len(msg["tool_calls"]) == 1


# ---------------------------------------------------------------------------
# Tool Schema Conversion
# ---------------------------------------------------------------------------


class TestOllamaToolConversion:
    def setup_method(self):
        self.provider = _get_provider()

    def test_tool_schema_conversion(self):
        """MCP tool schema converts to OpenAI function format."""
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]
        result = self.provider.convert_tools(tools)
        assert len(result) == 1
        t = result[0]
        assert t["type"] == "function"
        assert t["function"]["name"] == "get_weather"
        assert t["function"]["parameters"]["required"] == ["city"]

    def test_tool_schema_empty(self):
        """Empty tools list returns empty list."""
        assert self.provider.convert_tools([]) == []

    def test_tool_schema_preserves_all_fields(self):
        """All MCP fields map to correct OpenAI fields."""
        tools = [
            {
                "name": "test",
                "description": "A test tool",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
        result = self.provider.convert_tools(tools)
        func = result[0]["function"]
        assert func["name"] == "test"
        assert func["description"] == "A test tool"
        assert func["parameters"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestOllamaStreaming:
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
            model="llama3.1",
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
            _make_stream_chunk(tool_calls=[_make_tc_delta(0, tc_id="call_abc", name="search")]),
            _make_stream_chunk(tool_calls=[_make_tc_delta(0, arguments='{"q":')]),
            _make_stream_chunk(tool_calls=[_make_tc_delta(0, arguments='"test"}')]),
            _make_stream_chunk(finish_reason="tool_calls"),
            _make_usage_chunk(),
        ]
        self.provider._client.chat.completions.create.return_value = iter(chunks)

        resp = self.provider.stream(
            model="llama3.1",
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

    def test_stream_system_prepended(self):
        """System prompt is prepended as system message during stream."""
        chunks = [
            _make_stream_chunk(content="ok"),
            _make_stream_chunk(finish_reason="stop"),
            _make_usage_chunk(),
        ]
        self.provider._client.chat.completions.create.return_value = iter(chunks)

        self.provider.stream(
            model="llama3.1",
            max_tokens=100,
            system="You are a test bot",
            tools=[],
            messages=[{"role": "user", "content": "Hi"}],
        )

        call_kwargs = self.provider._client.chat.completions.create.call_args
        msgs = call_kwargs.kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "You are a test bot"}

    def test_stream_no_usage_chunk(self):
        """Stream without usage chunk returns empty usage dict."""
        chunks = [
            _make_stream_chunk(content="ok"),
            _make_stream_chunk(finish_reason="stop"),
        ]
        self.provider._client.chat.completions.create.return_value = iter(chunks)

        resp = self.provider.stream(
            model="llama3.1",
            max_tokens=100,
            system="",
            tools=[],
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert resp.usage == {}

    def test_stream_malformed_tool_args(self):
        """Malformed JSON in tool args is captured as _raw."""
        chunks = [
            _make_stream_chunk(tool_calls=[_make_tc_delta(0, tc_id="call_1", name="test")]),
            _make_stream_chunk(tool_calls=[_make_tc_delta(0, arguments="{bad json")]),
            _make_stream_chunk(finish_reason="tool_calls"),
            _make_usage_chunk(),
        ]
        self.provider._client.chat.completions.create.return_value = iter(chunks)

        resp = self.provider.stream(
            model="llama3.1",
            max_tokens=100,
            system="",
            tools=[{"name": "test", "description": "", "input_schema": {}}],
            messages=[{"role": "user", "content": "test"}],
        )

        assert resp.content[0].input == {"_raw": "{bad json"}


# ---------------------------------------------------------------------------
# Non-streaming (create)
# ---------------------------------------------------------------------------


class TestOllamaCreate:
    def setup_method(self):
        self.provider = _get_provider()

    def test_create_non_streaming(self):
        """Non-streaming call returns text content."""
        completion = _make_completion()
        self.provider._client.chat.completions.create.return_value = completion

        resp = self.provider.create(
            model="llama3.1",
            max_tokens=100,
            system="Be helpful",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert isinstance(resp, LLMResponse)
        assert resp.content[0].text == "Hello"
        assert resp.model == "llama3.1"

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
            model="llama3.1",
            max_tokens=100,
            system="",
            messages=[{"role": "user", "content": "search"}],
        )

        assert resp.stop_reason == "tool_use"
        assert isinstance(resp.content[0], ToolUseBlock)

    def test_create_system_prepended(self):
        """System prompt prepended as system message."""
        completion = _make_completion()
        self.provider._client.chat.completions.create.return_value = completion

        self.provider.create(
            model="llama3.1",
            max_tokens=100,
            system="You are a bot",
            messages=[{"role": "user", "content": "Hi"}],
        )

        call_kwargs = self.provider._client.chat.completions.create.call_args
        msgs = call_kwargs.kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "You are a bot"}

    def test_create_timeout_passed(self):
        """Timeout parameter is forwarded in kwargs."""
        completion = _make_completion()
        self.provider._client.chat.completions.create.return_value = completion

        self.provider.create(
            model="llama3.1",
            max_tokens=100,
            system="",
            messages=[{"role": "user", "content": "Hi"}],
            timeout=30.0,
        )

        call_kwargs = self.provider._client.chat.completions.create.call_args
        assert call_kwargs.kwargs["timeout"] == 30.0

    def test_create_model_parameter(self):
        """Model name forwarded correctly."""
        completion = _make_completion(model="mistral")
        self.provider._client.chat.completions.create.return_value = completion

        self.provider.create(
            model="mistral",
            max_tokens=200,
            system="",
            messages=[{"role": "user", "content": "Hi"}],
        )

        call_kwargs = self.provider._client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "mistral"


# ---------------------------------------------------------------------------
# Error Mapping
# ---------------------------------------------------------------------------


class TestOllamaErrorMapping:
    def setup_method(self):
        self.provider = _get_provider()

    def test_error_connection_refused(self):
        """APIConnectionError (Ollama not running) maps to LLMConnectionError."""
        import openai as real_openai

        exc = real_openai.APIConnectionError(request=MagicMock())
        self.provider._client.chat.completions.create.side_effect = exc

        with pytest.raises(LLMConnectionError):
            self.provider.create(model="llama3.1", max_tokens=100, system="", messages=[])

    def test_error_server_5xx(self):
        """Server error maps to LLMServerError."""
        import openai as real_openai

        resp_mock = MagicMock()
        resp_mock.status_code = 500
        exc = real_openai.APIStatusError(
            message="internal error",
            response=resp_mock,
            body=None,
        )
        self.provider._client.chat.completions.create.side_effect = exc

        with pytest.raises(LLMServerError):
            self.provider.create(model="llama3.1", max_tokens=100, system="", messages=[])

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
            self.provider.create(model="llama3.1", max_tokens=100, system="", messages=[])

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
            self.provider.create(model="llama3.1", max_tokens=100, system="", messages=[])

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
            self.provider.create(model="llama3.1", max_tokens=100, system="", messages=[])

    def test_error_stream_connection(self):
        """Connection error during streaming maps correctly."""
        import openai as real_openai

        exc = real_openai.APIConnectionError(request=MagicMock())
        self.provider._client.chat.completions.create.side_effect = exc

        with pytest.raises(LLMConnectionError):
            self.provider.stream(
                model="llama3.1",
                max_tokens=100,
                system="",
                tools=[],
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_error_4xx_non_auth(self):
        """4xx status (not 401/429) maps to generic LLMError."""
        import openai as real_openai

        resp_mock = MagicMock()
        resp_mock.status_code = 404
        exc = real_openai.APIStatusError(
            message="model not found",
            response=resp_mock,
            body=None,
        )
        self.provider._client.chat.completions.create.side_effect = exc

        with pytest.raises(LLMError):
            self.provider.create(model="nonexistent", max_tokens=100, system="", messages=[])


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestOllamaEdgeCases:
    def setup_method(self):
        self.provider = _get_provider()

    def test_empty_system_not_prepended(self):
        """Empty string system prompt is not prepended."""
        completion = _make_completion()
        self.provider._client.chat.completions.create.return_value = completion

        self.provider.create(
            model="llama3.1",
            max_tokens=100,
            system="",
            messages=[{"role": "user", "content": "Hi"}],
        )

        call_kwargs = self.provider._client.chat.completions.create.call_args
        msgs = call_kwargs.kwargs["messages"]
        # Empty string is falsy, so system not prepended
        assert msgs[0]["role"] == "user"

    def test_base_url_used(self):
        """OllamaProvider uses the configured base URL."""
        with (
            patch("agent.llm._ollama.openai") as mock_openai,
            patch(
                "agent.config.OLLAMA_BASE_URL",
                "http://custom:1234/v1",
            ),
        ):
            mock_openai.OpenAI.return_value = MagicMock()
            from agent.llm._ollama import OllamaProvider

            OllamaProvider()
            mock_openai.OpenAI.assert_called_with(
                base_url="http://custom:1234/v1",
                api_key="ollama",
            )
