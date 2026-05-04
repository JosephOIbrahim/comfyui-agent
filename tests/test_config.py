"""Tests for agent/config.py — API key validation, config defaults, security checks."""

import os
from unittest.mock import patch


class TestApiKeyValidation:
    def test_valid_key_no_warning(self, capsys):
        """Valid sk-ant- key should not trigger a warning."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test123"}, clear=False):
            # Re-import to refresh module-level ANTHROPIC_API_KEY constant
            import importlib
            import agent.config as config_mod
            importlib.reload(config_mod)
            # Reset the once-per-process latch so the function is allowed
            # to emit (T5 from the 5x review made the warning a deferred
            # call rather than an import-time print).
            config_mod._api_key_warn_emitted = False
            config_mod.warn_on_missing_api_key()
            captured = capsys.readouterr()
            assert "WARNING" not in captured.err

    def test_invalid_key_format_warns(self, capsys):
        """Non sk-ant- key should print a warning to stderr."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "bad-key-format"}, clear=False):
            import importlib
            import agent.config as config_mod
            importlib.reload(config_mod)
            config_mod._api_key_warn_emitted = False
            config_mod.warn_on_missing_api_key()
            captured = capsys.readouterr()
            assert "WARNING" in captured.err
            assert "sk-ant-" in captured.err

    def test_missing_key_is_none(self):
        """Missing ANTHROPIC_API_KEY defaults to None.

        load_dotenv() reads the .env file and re-injects the key even after
        popping it from os.environ.  We must mock load_dotenv to be a no-op
        so the reload sees a clean environment.
        """
        import importlib

        import agent.config as config_mod

        _real_getenv = os.getenv

        def _getenv_no_key(key, *args):
            if key == "ANTHROPIC_API_KEY":
                return None
            return _real_getenv(key, *args)

        with patch("os.getenv", side_effect=_getenv_no_key), \
             patch("dotenv.load_dotenv"):
            importlib.reload(config_mod)
            assert config_mod.ANTHROPIC_API_KEY is None


class TestConfigDefaults:
    def test_default_model(self):
        from agent.config import AGENT_MODEL
        assert "claude" in AGENT_MODEL.lower() or "opus" in AGENT_MODEL.lower()

    def test_default_comfyui_host(self):
        from agent.config import COMFYUI_HOST
        assert COMFYUI_HOST == "127.0.0.1"

    def test_default_comfyui_port(self):
        from agent.config import COMFYUI_PORT
        assert COMFYUI_PORT == 8188

    def test_log_dir_exists(self):
        from agent.config import LOG_DIR
        assert LOG_DIR is not None
        assert "logs" in str(LOG_DIR)

    def test_sessions_dir_defined(self):
        from agent.config import SESSIONS_DIR
        assert SESSIONS_DIR is not None


class TestMcpAuthToken:
    def test_default_none(self):
        """MCP_AUTH_TOKEN defaults to None when not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MCP_AUTH_TOKEN", None)
            import importlib
            import agent.config as config_mod
            importlib.reload(config_mod)
            assert config_mod.MCP_AUTH_TOKEN is None

    def test_reads_from_env(self):
        with patch.dict(os.environ, {"MCP_AUTH_TOKEN": "secret-token"}, clear=False):
            import importlib
            import agent.config as config_mod
            importlib.reload(config_mod)
            assert config_mod.MCP_AUTH_TOKEN == "secret-token"
