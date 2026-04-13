"""Tests for the Google Gemini LLM provider.

All tests mock the google.genai SDK — no real API calls, no network.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("google.genai", reason="google-genai SDK not installed")

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
# Fake error classes (module-level so isinstance checks work)
# ---------------------------------------------------------------------------


class _ClientError(Exception):
    """Fake google.genai ClientError for isinstance checks."""

    def __init__(self, msg="", code=None, status_code=None):
        super().__init__(msg)
        self.code = code
        self.status_code = status_code


class _ServerError(Exception):
    """Fake google.genai ServerError for isinstance checks."""

    def __init__(self, msg="", code=None, status_code=None):
        super().__init__(msg)
        self.code = code
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_genai_modules():
    """Create mocked google.genai module and its sub-modules."""
    mock_genai = MagicMock()
    mock_errors = MagicMock()
    mock_types = MagicMock()

    mock_types.GenerateContentConfig = MagicMock()
    mock_types.HttpOptions = MagicMock()
    mock_types.FunctionDeclaration = MagicMock()
    mock_types.Tool = MagicMock()

    mock_types.Part.from_text = lambda text: SimpleNamespace(
        text=text, thought=False, function_call=None
    )
    mock_types.Part.from_function_call = lambda name, args: SimpleNamespace(
        text=None, thought=False, function_call=SimpleNamespace(name=name, args=args)
    )
    mock_types.Part.from_function_response = lambda name, response: SimpleNamespace(
        text=None, thought=False, function_call=None, function_response=True
    )
    mock_types.Part.from_bytes = lambda data, mime_type: SimpleNamespace(
        text=None, thought=False, function_call=None, inline_data=True
    )

    mock_errors.ClientError = _ClientError
    mock_errors.ServerError = _ServerError

    return mock_genai, mock_errors, mock_types


@pytest.fixture
def gemini_env():
    """Yield (provider, mock_genai, mock_errors, mock_types) with active patches."""
    mock_genai, mock_errors, mock_types = _mock_genai_modules()

    with (
        patch("agent.llm._gemini.genai", mock_genai),
        patch("agent.llm._gemini.genai_errors", mock_errors),
        patch("agent.llm._gemini.genai_types", mock_types),
        patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
    ):
        from agent.llm._gemini import GeminiProvider

        provider = GeminiProvider()
        yield provider, mock_genai, mock_errors, mock_types


def _make_text_part(text, thought=False):
    return SimpleNamespace(text=text, thought=thought, function_call=None)


def _make_fc_part(name, args):
    return SimpleNamespace(
        text=None,
        thought=False,
        function_call=SimpleNamespace(name=name, args=args),
    )


def _make_candidate(parts):
    return SimpleNamespace(content=SimpleNamespace(parts=parts))


def _make_chunk(parts, usage=None):
    candidates = [_make_candidate(parts)] if parts else []
    return SimpleNamespace(candidates=candidates, usage_metadata=usage)


def _make_response(parts, usage=None):
    candidates = [_make_candidate(parts)] if parts else []
    return SimpleNamespace(candidates=candidates, usage_metadata=usage)


def _make_usage(prompt=10, completion=20):
    return SimpleNamespace(prompt_token_count=prompt, candidates_token_count=completion)


# ---------------------------------------------------------------------------
# Message Conversion
# ---------------------------------------------------------------------------


class TestGeminiMessageConversion:
    def test_message_conversion_text(self, gemini_env):
        provider, _, _, _ = gemini_env
        msgs = [{"role": "user", "content": "Hello world"}]
        result = provider.convert_messages(msgs)
        assert result[0]["role"] == "user"
        assert len(result[0]["parts"]) == 1
        assert result[0]["parts"][0].text == "Hello world"

    def test_message_conversion_text_block(self, gemini_env):
        provider, _, _, _ = gemini_env
        msgs = [{"role": "user", "content": [TextBlock(text="Hi there")]}]
        result = provider.convert_messages(msgs)
        assert result[0]["parts"][0].text == "Hi there"

    def test_message_conversion_assistant_role(self, gemini_env):
        """Assistant role maps to Gemini 'model' role."""
        provider, _, _, _ = gemini_env
        msgs = [{"role": "assistant", "content": "I am a model"}]
        result = provider.convert_messages(msgs)
        assert result[0]["role"] == "model"

    def test_message_conversion_tool_use(self, gemini_env):
        provider, _, _, _ = gemini_env
        msgs = [
            {
                "role": "assistant",
                "content": [
                    ToolUseBlock(id="call_1", name="search", input={"q": "test"}),
                ],
            }
        ]
        result = provider.convert_messages(msgs)
        assert result[0]["role"] == "model"
        part = result[0]["parts"][0]
        assert part.function_call is not None
        assert part.function_call.name == "search"

    def test_message_conversion_tool_result(self, gemini_env):
        provider, _, _, _ = gemini_env
        msgs = [
            {
                "role": "assistant",
                "content": [
                    ToolUseBlock(id="call_1", name="search", input={}),
                ],
            },
            {
                "role": "user",
                "content": [
                    ToolResultBlock(tool_use_id="call_1", content="results here"),
                ],
            },
        ]
        result = provider.convert_messages(msgs)
        assert len(result) == 2

    def test_message_conversion_image(self, gemini_env):
        provider, _, _, _ = gemini_env
        valid_b64 = base64.b64encode(b"fake-png-data").decode()
        msgs = [
            {
                "role": "user",
                "content": [
                    ImageBlock(data=valid_b64, media_type="image/png"),
                ],
            }
        ]
        result = provider.convert_messages(msgs)
        part = result[0]["parts"][0]
        assert hasattr(part, "inline_data")

    def test_message_conversion_thinking_skipped(self, gemini_env):
        provider, _, _, _ = gemini_env
        msgs = [
            {
                "role": "assistant",
                "content": [
                    ThinkingBlock(thinking="Reasoning..."),
                    TextBlock(text="Answer"),
                ],
            }
        ]
        result = provider.convert_messages(msgs)
        assert len(result[0]["parts"]) == 1
        assert result[0]["parts"][0].text == "Answer"

    def test_message_conversion_empty_content_list(self, gemini_env):
        """Empty content list produces no message (parts empty -> skipped)."""
        provider, _, _, _ = gemini_env
        msgs = [{"role": "user", "content": []}]
        result = provider.convert_messages(msgs)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tool Schema Conversion
# ---------------------------------------------------------------------------


class TestGeminiToolConversion:
    def test_tool_schema_conversion(self, gemini_env):
        provider, _, _, mock_types = gemini_env
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
        result = provider.convert_tools(tools)
        assert len(result) == 1
        mock_types.FunctionDeclaration.assert_called_once()
        call_kwargs = mock_types.FunctionDeclaration.call_args
        assert call_kwargs.kwargs["name"] == "get_weather"

    def test_tool_schema_empty(self, gemini_env):
        provider, _, _, _ = gemini_env
        assert provider.convert_tools([]) == []

    def test_tool_schema_no_input_schema(self, gemini_env):
        provider, _, _, _ = gemini_env
        tools = [{"name": "ping", "description": "Ping"}]
        result = provider.convert_tools(tools)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestGeminiStreaming:
    def test_stream_text(self, gemini_env):
        provider, _, _, _ = gemini_env
        chunks = [
            _make_chunk([_make_text_part("Hello")]),
            _make_chunk([_make_text_part(" world")]),
            _make_chunk([], usage=_make_usage(5, 2)),
        ]
        provider._client.models.generate_content_stream.return_value = iter(chunks)

        text_parts = []
        resp = provider.stream(
            model="gemini-2.5-flash",
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

    def test_stream_tool_use(self, gemini_env):
        provider, _, _, _ = gemini_env
        chunks = [
            _make_chunk([_make_fc_part("search", {"q": "test"})]),
            _make_chunk([], usage=_make_usage()),
        ]
        provider._client.models.generate_content_stream.return_value = iter(chunks)

        resp = provider.stream(
            model="gemini-2.5-flash",
            max_tokens=100,
            system="",
            tools=[{"name": "search", "description": "", "input_schema": {}}],
            messages=[{"role": "user", "content": "find it"}],
        )

        assert resp.stop_reason == "tool_use"
        assert isinstance(resp.content[0], ToolUseBlock)
        assert resp.content[0].name == "search"
        assert resp.content[0].input == {"q": "test"}

    def test_stream_thinking(self, gemini_env):
        provider, _, _, _ = gemini_env
        chunks = [
            _make_chunk([_make_text_part("Thinking...", thought=True)]),
            _make_chunk([_make_text_part("Answer")]),
            _make_chunk([], usage=_make_usage()),
        ]
        provider._client.models.generate_content_stream.return_value = iter(chunks)

        text_parts = []
        thinking_parts = []
        resp = provider.stream(
            model="gemini-2.5-flash",
            max_tokens=100,
            system="",
            tools=[],
            messages=[{"role": "user", "content": "think"}],
            on_text=lambda t: text_parts.append(t),
            on_thinking=lambda t: thinking_parts.append(t),
        )

        assert text_parts == ["Answer"]
        assert thinking_parts == ["Thinking..."]
        assert resp.content[0].text == "Answer"

    def test_stream_empty_candidates(self, gemini_env):
        provider, _, _, _ = gemini_env
        chunks = [
            SimpleNamespace(candidates=[], usage_metadata=None),
            _make_chunk([_make_text_part("ok")]),
            _make_chunk([], usage=_make_usage()),
        ]
        provider._client.models.generate_content_stream.return_value = iter(chunks)

        resp = provider.stream(
            model="gemini-2.5-flash",
            max_tokens=100,
            system="",
            tools=[],
            messages=[{"role": "user", "content": "test"}],
        )
        assert resp.content[0].text == "ok"


# ---------------------------------------------------------------------------
# Non-streaming (create)
# ---------------------------------------------------------------------------


class TestGeminiCreate:
    def test_create_text(self, gemini_env):
        provider, _, _, _ = gemini_env
        response = _make_response([_make_text_part("Hello")], usage=_make_usage(8, 3))
        provider._client.models.generate_content.return_value = response

        resp = provider.create(
            model="gemini-2.5-flash",
            max_tokens=100,
            system="Be helpful",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert isinstance(resp, LLMResponse)
        assert resp.content[0].text == "Hello"
        assert resp.usage["input_tokens"] == 8

    def test_create_with_tool_calls(self, gemini_env):
        provider, _, _, _ = gemini_env
        response = _make_response([_make_fc_part("search", {"q": "test"})], usage=_make_usage())
        provider._client.models.generate_content.return_value = response

        resp = provider.create(
            model="gemini-2.5-flash",
            max_tokens=100,
            system="",
            messages=[{"role": "user", "content": "search"}],
        )

        assert resp.stop_reason == "tool_use"
        assert isinstance(resp.content[0], ToolUseBlock)

    def test_create_empty_response(self, gemini_env):
        provider, _, _, _ = gemini_env
        response = SimpleNamespace(candidates=[], usage_metadata=None)
        provider._client.models.generate_content.return_value = response

        resp = provider.create(
            model="gemini-2.5-flash",
            max_tokens=100,
            system="",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert resp.content == []
        assert resp.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# Error Mapping
# ---------------------------------------------------------------------------


class TestGeminiErrorMapping:
    def test_error_auth_by_code(self, gemini_env):
        """ClientError with code=401 maps to LLMAuthError."""
        provider, _, _, _ = gemini_env
        exc = _ClientError("unauthorized", code=401)
        provider._client.models.generate_content_stream.side_effect = exc

        with pytest.raises(LLMAuthError):
            provider.stream(
                model="gemini-2.5-flash",
                max_tokens=100,
                system="",
                tools=[],
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_error_auth_by_status_code(self, gemini_env):
        """ClientError with status_code=403 maps to LLMAuthError."""
        provider, _, _, _ = gemini_env
        exc = _ClientError("forbidden", status_code=403)
        provider._client.models.generate_content_stream.side_effect = exc

        with pytest.raises(LLMAuthError):
            provider.stream(
                model="gemini-2.5-flash",
                max_tokens=100,
                system="",
                tools=[],
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_error_auth_by_message_with_401(self, gemini_env):
        """ClientError with '401' in message triggers auth path."""
        provider, _, _, _ = gemini_env
        exc = _ClientError("Error 401: invalid credentials")
        provider._client.models.generate_content_stream.side_effect = exc

        with pytest.raises(LLMAuthError):
            provider.stream(
                model="gemini-2.5-flash",
                max_tokens=100,
                system="",
                tools=[],
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_error_auth_api_key_message_bug(self, gemini_env):
        """Document bug: 'API key' message check is case-broken.

        BUG in _gemini.py:_translate_error line ~409:
            "API key" in msg.lower()
        Since msg.lower() yields all-lowercase, the mixed-case literal
        "API key" can never match. This means ClientError messages
        containing 'api key' (in any casing) fall through to LLMError
        instead of LLMAuthError, unless they also contain '401' or '403'
        or have code/status_code set to 401/403.
        """
        provider, _, _, _ = gemini_env
        exc = _ClientError("Invalid API key provided")
        provider._client.models.generate_content_stream.side_effect = exc

        # Current (buggy) behavior: falls through to LLMError
        with pytest.raises(LLMError):
            provider.stream(
                model="gemini-2.5-flash",
                max_tokens=100,
                system="",
                tools=[],
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_error_rate_limit(self, gemini_env):
        provider, _, _, _ = gemini_env
        exc = _ClientError("rate limited", code=429)
        provider._client.models.generate_content_stream.side_effect = exc

        with pytest.raises(LLMRateLimitError):
            provider.stream(
                model="gemini-2.5-flash",
                max_tokens=100,
                system="",
                tools=[],
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_error_server(self, gemini_env):
        provider, _, _, _ = gemini_env
        exc = _ServerError("internal error", code=503)
        provider._client.models.generate_content_stream.side_effect = exc

        with pytest.raises(LLMServerError):
            provider.stream(
                model="gemini-2.5-flash",
                max_tokens=100,
                system="",
                tools=[],
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_error_connection(self, gemini_env):
        provider, _, _, _ = gemini_env
        exc = ConnectionError("network down")
        provider._client.models.generate_content_stream.side_effect = exc

        with pytest.raises(LLMConnectionError):
            provider.stream(
                model="gemini-2.5-flash",
                max_tokens=100,
                system="",
                tools=[],
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_error_generic(self, gemini_env):
        provider, _, _, _ = gemini_env
        exc = RuntimeError("something broke")
        provider._client.models.generate_content_stream.side_effect = exc

        with pytest.raises(LLMError):
            provider.stream(
                model="gemini-2.5-flash",
                max_tokens=100,
                system="",
                tools=[],
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_error_create_auth(self, gemini_env):
        """Auth error on create() also maps correctly (using '401' in msg)."""
        provider, _, _, _ = gemini_env
        exc = _ClientError("401 unauthorized")
        provider._client.models.generate_content.side_effect = exc

        with pytest.raises(LLMAuthError):
            provider.create(
                model="gemini-2.5-flash",
                max_tokens=100,
                system="",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_error_timeout(self, gemini_env):
        provider, _, _, _ = gemini_env
        exc = TimeoutError("connection timed out")
        provider._client.models.generate_content_stream.side_effect = exc

        with pytest.raises(LLMConnectionError):
            provider.stream(
                model="gemini-2.5-flash",
                max_tokens=100,
                system="",
                tools=[],
                messages=[{"role": "user", "content": "hi"}],
            )


# ---------------------------------------------------------------------------
# Schema Conversion Helper
# ---------------------------------------------------------------------------


class TestGeminiSchemaConversion:
    def test_convert_schema_basic(self):
        from agent.llm._gemini import _convert_schema

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The name"},
            },
            "required": ["name"],
        }
        result = _convert_schema(schema)
        assert result["type"] == "OBJECT"
        assert "name" in result["properties"]
        assert result["required"] == ["name"]

    def test_convert_schema_empty(self):
        from agent.llm._gemini import _convert_schema

        assert _convert_schema({}) == {}

    def test_convert_schema_array(self):
        from agent.llm._gemini import _convert_schema

        schema = {"type": "array", "items": {"type": "string"}}
        result = _convert_schema(schema)
        assert result["type"] == "ARRAY"
        assert result["items"]["type"] == "STRING"
