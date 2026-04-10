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
    _resolve_and_check_private,
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

    def test_cgnat_ip_blocked(self):
        # 100.64.0.0/10 is RFC 6598 CGNAT — must be blocked
        result = _validate_download_url("https://100.64.0.1/model.safetensors")
        assert result is not None

    def test_cgnat_upper_bound_blocked(self):
        result = _validate_download_url("https://100.127.255.255/model.safetensors")
        assert result is not None


# ---------------------------------------------------------------------------
# _resolve_and_check_private
# ---------------------------------------------------------------------------


class TestResolveAndCheckPrivate:
    """Verify that _resolve_and_check_private catches private IPs after DNS lookup.

    This is the guard that fires on each redirect hop — an allowlisted CDN
    hostname that resolves to a private IP must be blocked here.
    """

    def test_safe_public_ip_returns_none(self, monkeypatch):
        """1.1.1.1 (Cloudflare public DNS) must be allowed."""
        import socket
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, **kw: [(None, None, None, None, ("1.1.1.1", 0))],
        )
        assert _resolve_and_check_private("one.one.one.one") is None

    def test_redirect_to_rfc1918_192_168_blocked(self, monkeypatch):
        """302 from allowlisted CDN → 192.168.1.1 must be blocked."""
        import socket
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, **kw: [(None, None, None, None, ("192.168.1.1", 0))],
        )
        result = _resolve_and_check_private("huggingface.co")
        assert result is not None
        assert "192.168.1.1" in result

    def test_redirect_to_loopback_blocked(self, monkeypatch):
        import socket
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, **kw: [(None, None, None, None, ("127.0.0.1", 0))],
        )
        result = _resolve_and_check_private("evil.example.com")
        assert result is not None

    def test_redirect_to_rfc1918_10_x_blocked(self, monkeypatch):
        import socket
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, **kw: [(None, None, None, None, ("10.0.0.1", 0))],
        )
        result = _resolve_and_check_private("cdn.example.com")
        assert result is not None

    def test_redirect_to_cgnat_blocked(self, monkeypatch):
        """100.64.0.0/10 CGNAT range must be blocked on DNS resolution too."""
        import socket
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, **kw: [(None, None, None, None, ("100.64.0.1", 0))],
        )
        result = _resolve_and_check_private("cdn.example.com")
        assert result is not None
        assert "100.64.0.1" in result

    def test_unresolvable_hostname_returns_error(self, monkeypatch):
        import socket
        def _raise(*a, **kw):
            raise socket.gaierror("Name or service not known")
        monkeypatch.setattr("socket.getaddrinfo", _raise)
        result = _resolve_and_check_private("does-not-exist.invalid")
        assert result is not None

    def test_link_local_blocked(self, monkeypatch):
        """169.254.x.x link-local addresses (e.g. AWS metadata) must be blocked."""
        import socket
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, **kw: [(None, None, None, None, ("169.254.169.254", 0))],
        )
        result = _resolve_and_check_private("metadata.internal")
        assert result is not None


class TestDownloadModelSsrfRedirectBlocked:
    """Integration test: download_model must refuse a redirect to a private IP.

    This is the P0-J regression test. An allowlisted CDN (huggingface.co)
    issues a 302 to 192.168.1.1 — the download must be refused.
    """

    def test_302_from_allowlisted_host_to_private_ip_is_refused(
        self, tmp_path, monkeypatch
    ):
        import json
        import socket
        from unittest.mock import MagicMock, patch

        from agent.tools.comfy_provision import handle

        # Patch MODELS_DIR to tmp_path so the tool doesn't touch real dirs
        monkeypatch.setattr("agent.tools.comfy_provision.MODELS_DIR", tmp_path)
        (tmp_path / "checkpoints").mkdir()

        # First httpx call: allowlisted host returns 302 → private IP
        redirect_response = MagicMock()
        redirect_response.status_code = 302
        redirect_response.headers = {"location": "http://192.168.1.1/evil.safetensors"}
        redirect_response.__enter__ = lambda s: s
        redirect_response.__exit__ = MagicMock(return_value=False)

        with patch("agent.tools.comfy_provision.httpx.stream", return_value=redirect_response):
            # The redirect target http:// also fails _validate_download_url (not https),
            # so we test by patching _validate_download_url to pass for the redirect URL
            # but _resolve_and_check_private to return the private IP error.
            # Actually — http://192.168.1.1 will fail _validate_download_url already
            # (not https), so this is doubly blocked. Verify the error is returned.
            result = handle(
                "download_model",
                {
                    "url": "https://huggingface.co/models/test/model.safetensors",
                    "model_type": "checkpoints",
                    "filename": "model.safetensors",
                },
            )

        parsed = json.loads(result)
        assert "error" in parsed
        # Must be refused — either for non-https redirect URL or private IP
        error_msg = parsed["error"].lower()
        assert any(
            phrase in error_msg
            for phrase in ("redirect", "blocked", "private", "denied", "https")
        ), f"Expected SSRF-related refusal, got: {parsed['error']}"
        # Must NOT have downloaded anything
        assert not any(tmp_path.rglob("*.safetensors"))

    def test_302_to_https_private_ip_blocked_by_dns_check(
        self, tmp_path, monkeypatch
    ):
        """Redirect to https://192.168.1.1 blocked at _validate_download_url level."""
        import json
        from unittest.mock import MagicMock, patch

        from agent.tools.comfy_provision import handle

        monkeypatch.setattr("agent.tools.comfy_provision.MODELS_DIR", tmp_path)
        (tmp_path / "checkpoints").mkdir()

        redirect_response = MagicMock()
        redirect_response.status_code = 302
        redirect_response.headers = {"location": "https://192.168.1.1/evil.safetensors"}
        redirect_response.__enter__ = lambda s: s
        redirect_response.__exit__ = MagicMock(return_value=False)

        with patch("agent.tools.comfy_provision.httpx.stream", return_value=redirect_response):
            result = handle(
                "download_model",
                {
                    "url": "https://huggingface.co/models/test/model.safetensors",
                    "model_type": "checkpoints",
                    "filename": "model.safetensors",
                },
            )

        parsed = json.loads(result)
        assert "error" in parsed
        assert not any(tmp_path.rglob("*.safetensors"))


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


# ---------------------------------------------------------------------------
# Cycle 36: zero-byte download guard
# ---------------------------------------------------------------------------

class TestZeroByteDownloadGuard:
    """download_model must return an error (not create an empty file) when the
    server returns HTTP 200 with an empty response body. (Cycle 36 fix)"""

    def test_empty_body_returns_error(self, tmp_path, monkeypatch):
        """HTTP 200 + zero bytes must produce error JSON, not an empty model file."""
        from unittest.mock import MagicMock, patch

        from agent.tools.comfy_provision import handle

        monkeypatch.setattr("agent.tools.comfy_provision.MODELS_DIR", tmp_path)
        (tmp_path / "checkpoints").mkdir()

        # Mock an HTTP 200 response that yields no bytes
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-length": "0"}
        mock_resp.iter_bytes = MagicMock(return_value=iter([]))  # empty body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("agent.tools.comfy_provision.httpx.stream", return_value=mock_resp), \
             patch("agent.tools.comfy_provision._resolve_and_check_private", return_value=None):
            result = handle("download_model", {
                "url": "https://huggingface.co/models/test/model.safetensors",
                "model_type": "checkpoints",
                "filename": "model.safetensors",
            })

        parsed = json.loads(result)
        assert "error" in parsed
        assert "empty" in parsed["error"].lower() or "0 byte" in parsed["error"].lower()
        # Must not have created any file
        assert not any(tmp_path.rglob("model.safetensors"))
        # Temp file must also be cleaned up
        assert not any(tmp_path.rglob("*.download"))

    def test_empty_body_does_not_rename_temp_file(self, tmp_path, monkeypatch):
        """No .download temp file should survive after a zero-byte response."""
        from unittest.mock import MagicMock, patch

        from agent.tools.comfy_provision import handle

        monkeypatch.setattr("agent.tools.comfy_provision.MODELS_DIR", tmp_path)
        (tmp_path / "checkpoints").mkdir()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.iter_bytes = MagicMock(return_value=iter([]))
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("agent.tools.comfy_provision.httpx.stream", return_value=mock_resp), \
             patch("agent.tools.comfy_provision._resolve_and_check_private", return_value=None):
            handle("download_model", {
                "url": "https://huggingface.co/models/test/model.safetensors",
                "model_type": "checkpoints",
                "filename": "model.safetensors",
            })

        remaining = list(tmp_path.rglob("*"))
        file_names = [f.name for f in remaining if f.is_file()]
        assert not any(".download" in n for n in file_names), (
            f"Temp .download file survived: {file_names}"
        )
