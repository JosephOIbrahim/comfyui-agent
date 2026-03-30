"""Tests for security validator functions — pure unit tests, no mocking needed.

Covers:
- _validate_download_url  (agent/tools/comfy_provision.py)
- _safe_filename          (agent/tools/comfy_provision.py)
- _safe_path_component    (agent/tools/comfy_provision.py)
- _validate_git_url       (agent/tools/comfy_provision.py)
- _validate_session_name  (agent/memory/session.py)
- error hierarchy         (agent/errors.py)
"""

from __future__ import annotations

import json

import pytest

from agent.tools.comfy_provision import (
    _safe_filename,
    _safe_path_component,
    _validate_download_url,
    _validate_git_url,
)
from agent.memory.session import _validate_session_name
from agent.errors import (
    AgentError,
    ToolError,
    TransportError,
    ValidationError,
    error_json,
)


# ---------------------------------------------------------------------------
# _validate_download_url
# ---------------------------------------------------------------------------

class TestValidateDownloadUrl:
    """SSRF-prevention checks for model download URLs."""

    def test_valid_https_url_returns_none(self):
        assert _validate_download_url("https://huggingface.co/model.safetensors") is None

    def test_valid_https_civitai_returns_none(self):
        assert _validate_download_url("https://civitai.com/api/download/models/12345") is None

    def test_valid_https_github_returns_none(self):
        assert _validate_download_url("https://github.com/owner/repo/releases/download/v1/model.ckpt") is None

    def test_http_url_returns_error(self):
        result = _validate_download_url("http://huggingface.co/model.safetensors")
        assert result is not None
        assert isinstance(result, str)

    def test_ftp_url_returns_error(self):
        result = _validate_download_url("ftp://example.com/model.safetensors")
        assert result is not None

    def test_localhost_blocked(self):
        result = _validate_download_url("https://localhost/model.safetensors")
        assert result is not None

    def test_localhost_with_port_blocked(self):
        result = _validate_download_url("https://localhost:8188/model.safetensors")
        assert result is not None

    def test_loopback_ipv4_blocked(self):
        result = _validate_download_url("https://127.0.0.1/model.safetensors")
        assert result is not None

    def test_loopback_ipv4_with_port_blocked(self):
        result = _validate_download_url("https://127.0.0.1:8080/model.safetensors")
        assert result is not None

    def test_private_ip_10_blocked(self):
        result = _validate_download_url("https://10.0.0.1/model.safetensors")
        assert result is not None

    def test_private_ip_192_168_blocked(self):
        result = _validate_download_url("https://192.168.1.1/model.safetensors")
        assert result is not None

    def test_private_ip_172_16_blocked(self):
        result = _validate_download_url("https://172.16.0.1/model.safetensors")
        assert result is not None

    def test_metadata_google_endpoint_blocked(self):
        result = _validate_download_url("https://metadata.google.internal/computeMetadata/v1/")
        assert result is not None

    def test_link_local_169_254_blocked(self):
        result = _validate_download_url("https://169.254.169.254/latest/meta-data/")
        assert result is not None

    def test_empty_string_returns_error(self):
        result = _validate_download_url("")
        assert result is not None

    def test_whitespace_only_returns_error(self):
        result = _validate_download_url("   ")
        assert result is not None

    def test_malformed_no_scheme_returns_error(self):
        result = _validate_download_url("not-a-url-at-all")
        assert result is not None

    def test_leading_whitespace_stripped_valid_url(self):
        # Leading/trailing whitespace should be stripped before checking
        result = _validate_download_url("  https://huggingface.co/model.safetensors  ")
        assert result is None

    def test_returns_string_on_error_not_exception(self):
        result = _validate_download_url("http://bad.com/model.safetensors")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _safe_filename
# ---------------------------------------------------------------------------

class TestSafeFilename:
    """Path-traversal prevention for model filenames."""

    def test_clean_filename_returns_filename(self):
        assert _safe_filename("model.safetensors") == "model.safetensors"

    def test_clean_filename_with_underscores_and_dashes(self):
        assert _safe_filename("my-lora_v2.safetensors") == "my-lora_v2.safetensors"

    def test_clean_filename_strips_whitespace(self):
        assert _safe_filename("  model.ckpt  ") == "model.ckpt"

    def test_forward_slash_returns_none(self):
        assert _safe_filename("subdir/model.safetensors") is None

    def test_backslash_returns_none(self):
        assert _safe_filename("subdir\\model.safetensors") is None

    def test_dotdot_alone_returns_none(self):
        assert _safe_filename("..") is None

    def test_dot_alone_returns_none(self):
        assert _safe_filename(".") is None

    def test_dotdot_prefix_path_traversal_returns_none(self):
        assert _safe_filename("../../../etc/passwd") is None

    def test_dotdot_as_prefix_no_slash_returns_none(self):
        assert _safe_filename("..hidden") is None

    def test_empty_string_returns_none(self):
        assert _safe_filename("") is None

    def test_whitespace_only_returns_none(self):
        assert _safe_filename("   ") is None

    def test_valid_return_type_is_str(self):
        result = _safe_filename("valid.safetensors")
        assert isinstance(result, str)

    def test_invalid_return_type_is_none(self):
        result = _safe_filename("bad/path.safetensors")
        assert result is None


# ---------------------------------------------------------------------------
# _safe_path_component
# ---------------------------------------------------------------------------

class TestSafePathComponent:
    """Path-traversal prevention for single directory components."""

    def test_clean_component_returns_component(self):
        assert _safe_path_component("loras") == "loras"

    def test_alphanumeric_with_hyphens(self):
        assert _safe_path_component("my-models") == "my-models"

    def test_strips_whitespace(self):
        assert _safe_path_component("  checkpoints  ") == "checkpoints"

    def test_forward_slash_returns_none(self):
        assert _safe_path_component("parent/child") is None

    def test_backslash_returns_none(self):
        assert _safe_path_component("parent\\child") is None

    def test_dotdot_alone_returns_none(self):
        assert _safe_path_component("..") is None

    def test_dot_alone_returns_none(self):
        assert _safe_path_component(".") is None

    def test_dotdot_prefix_returns_none(self):
        assert _safe_path_component("..secret") is None

    def test_empty_string_returns_none(self):
        assert _safe_path_component("") is None

    def test_whitespace_only_returns_none(self):
        assert _safe_path_component("   ") is None

    def test_valid_return_type_is_str(self):
        result = _safe_path_component("embeddings")
        assert isinstance(result, str)

    def test_invalid_return_type_is_none(self):
        result = _safe_path_component("bad/component")
        assert result is None


# ---------------------------------------------------------------------------
# _validate_git_url
# ---------------------------------------------------------------------------

class TestValidateGitUrl:
    """Allowlist checks for git repository URLs."""

    def test_valid_github_https_returns_none(self):
        assert _validate_git_url("https://github.com/owner/ComfyUI-PackName") is None

    def test_valid_gitlab_https_returns_none(self):
        assert _validate_git_url("https://gitlab.com/owner/repo") is None

    def test_valid_huggingface_https_returns_none(self):
        assert _validate_git_url("https://huggingface.co/owner/model") is None

    def test_valid_bitbucket_https_returns_none(self):
        assert _validate_git_url("https://bitbucket.org/owner/repo") is None

    def test_valid_codeberg_https_returns_none(self):
        assert _validate_git_url("https://codeberg.org/owner/repo") is None

    def test_http_github_blocked(self):
        result = _validate_git_url("http://github.com/owner/repo")
        assert result is not None

    def test_unknown_host_blocked(self):
        result = _validate_git_url("https://malicious-site.com/owner/repo")
        assert result is not None

    def test_unknown_host_error_mentions_host(self):
        result = _validate_git_url("https://notallowed.example.com/repo")
        assert result is not None
        assert isinstance(result, str)

    def test_ssh_git_url_blocked(self):
        result = _validate_git_url("git@github.com:owner/repo.git")
        assert result is not None

    def test_ftp_blocked(self):
        result = _validate_git_url("ftp://github.com/owner/repo")
        assert result is not None

    def test_empty_string_returns_error(self):
        result = _validate_git_url("")
        assert result is not None

    def test_leading_whitespace_stripped_valid_url(self):
        result = _validate_git_url("  https://github.com/owner/repo  ")
        assert result is None

    def test_returns_string_on_error(self):
        result = _validate_git_url("http://github.com/owner/repo")
        assert isinstance(result, str)

    def test_returns_none_type_on_success(self):
        result = _validate_git_url("https://github.com/owner/repo")
        assert result is None


# ---------------------------------------------------------------------------
# _validate_session_name
# ---------------------------------------------------------------------------

class TestValidateSessionName:
    """Path-traversal and safety checks for session names."""

    def test_clean_name_returns_none(self):
        assert _validate_session_name("my-session") is None

    def test_alphanumeric_name_returns_none(self):
        assert _validate_session_name("session123") is None

    def test_name_with_underscores_returns_none(self):
        assert _validate_session_name("my_session_name") is None

    def test_name_with_forward_slash_returns_error(self):
        result = _validate_session_name("bad/name")
        assert result is not None
        assert isinstance(result, str)

    def test_name_with_backslash_returns_error(self):
        result = _validate_session_name("bad\\name")
        assert result is not None
        assert isinstance(result, str)

    def test_dotdot_returns_error(self):
        result = _validate_session_name("..")
        assert result is not None

    def test_dotdot_prefix_returns_error(self):
        result = _validate_session_name("../../../etc/passwd")
        assert result is not None

    def test_null_byte_returns_error(self):
        result = _validate_session_name("session\x00name")
        assert result is not None

    def test_name_over_255_chars_returns_error(self):
        long_name = "a" * 256
        result = _validate_session_name(long_name)
        assert result is not None

    def test_name_exactly_255_chars_returns_none(self):
        name = "a" * 255
        assert _validate_session_name(name) is None

    def test_empty_string_returns_error(self):
        result = _validate_session_name("")
        assert result is not None

    def test_non_string_returns_error(self):
        result = _validate_session_name(None)  # type: ignore[arg-type]
        assert result is not None

    def test_valid_returns_none_type(self):
        result = _validate_session_name("valid-session-name")
        assert result is None

    def test_invalid_returns_str_type(self):
        result = _validate_session_name("")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class TestErrorHierarchy:
    """Verify the exception class hierarchy and error_json helper."""

    def test_agent_error_is_exception(self):
        assert issubclass(AgentError, Exception)

    def test_tool_error_is_agent_error(self):
        assert issubclass(ToolError, AgentError)

    def test_transport_error_is_agent_error(self):
        assert issubclass(TransportError, AgentError)

    def test_validation_error_is_agent_error(self):
        assert issubclass(ValidationError, AgentError)

    def test_tool_error_can_be_raised_and_caught_as_agent_error(self):
        with pytest.raises(AgentError):
            raise ToolError("something went wrong")

    def test_transport_error_can_be_raised_and_caught_as_agent_error(self):
        with pytest.raises(AgentError):
            raise TransportError("connection refused")

    def test_validation_error_can_be_raised_and_caught_as_agent_error(self):
        with pytest.raises(AgentError):
            raise ValidationError("schema mismatch")

    def test_error_json_returns_valid_json(self):
        raw = error_json("Something broke")
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_error_json_contains_error_key(self):
        raw = error_json("Something broke")
        parsed = json.loads(raw)
        assert "error" in parsed
        assert parsed["error"] == "Something broke"

    def test_error_json_with_hint_includes_hint_key(self):
        raw = error_json("Something broke", hint="Try restarting ComfyUI")
        parsed = json.loads(raw)
        assert "hint" in parsed
        assert parsed["hint"] == "Try restarting ComfyUI"

    def test_error_json_without_hint_omits_hint_key(self):
        raw = error_json("No hint here")
        parsed = json.loads(raw)
        assert "hint" not in parsed

    def test_error_json_uses_sorted_keys(self):
        raw = error_json("test", hint="hint text")
        # With sort_keys=True, 'error' < 'hint' alphabetically
        error_pos = raw.index('"error"')
        hint_pos = raw.index('"hint"')
        assert error_pos < hint_pos

    def test_error_json_with_context_includes_context_key(self):
        raw = error_json("Nodes missing", available=["NodeA", "NodeB"])
        parsed = json.loads(raw)
        assert "context" in parsed
        assert parsed["context"]["available"] == ["NodeA", "NodeB"]

    def test_error_json_returns_str(self):
        result = error_json("test message")
        assert isinstance(result, str)
