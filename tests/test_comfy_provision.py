import json


# ---------------------------------------------------------------------------
# Cycle 43 — _install_locks FIFO eviction cap
# ---------------------------------------------------------------------------

class TestInstallLocksCap:
    """_install_locks must be capped at _MAX_INSTALL_LOCKS entries."""

    def test_locks_capped_at_max(self):
        """After _MAX_INSTALL_LOCKS+1 unique paths, dict never exceeds the cap."""
        from agent.tools.comfy_provision import (
            _get_install_lock, _install_locks, _install_locks_mutex, _MAX_INSTALL_LOCKS,
        )

        # Clear state first
        with _install_locks_mutex:
            _install_locks.clear()

        # Register MAX + 10 unique paths
        for i in range(_MAX_INSTALL_LOCKS + 10):
            _get_install_lock(f"/fake/path/{i}")

        with _install_locks_mutex:
            actual = len(_install_locks)

        assert actual <= _MAX_INSTALL_LOCKS, (
            f"_install_locks grew to {actual}, must stay ≤ {_MAX_INSTALL_LOCKS}"
        )

    def test_max_install_locks_constant_exists(self):
        """_MAX_INSTALL_LOCKS constant must be defined and positive."""
        from agent.tools.comfy_provision import _MAX_INSTALL_LOCKS
        assert isinstance(_MAX_INSTALL_LOCKS, int)
        assert _MAX_INSTALL_LOCKS > 0

    def test_existing_path_returns_same_lock_object(self):
        """Calling _get_install_lock twice for same path returns the same lock."""
        from agent.tools.comfy_provision import _get_install_lock, _install_locks_mutex, _install_locks

        with _install_locks_mutex:
            _install_locks.clear()

        lock_a = _get_install_lock("/same/path")
        lock_b = _get_install_lock("/same/path")
        assert lock_a is lock_b, "Same path must return same lock object"


# ---------------------------------------------------------------------------
# Cycle 44 — named constants exist at module level
# ---------------------------------------------------------------------------

class TestProvisionModuleLevelConstants:
    """All timeout/size magic numbers must exist as named module-level constants."""

    def test_lock_acquire_timeout_constant_exists(self):
        from agent.tools.comfy_provision import _LOCK_ACQUIRE_TIMEOUT
        assert isinstance(_LOCK_ACQUIRE_TIMEOUT, (int, float))
        assert _LOCK_ACQUIRE_TIMEOUT > 0

    def test_git_clone_timeout_constant_exists(self):
        from agent.tools.comfy_provision import _GIT_CLONE_TIMEOUT
        assert isinstance(_GIT_CLONE_TIMEOUT, (int, float))
        assert _GIT_CLONE_TIMEOUT > 0

    def test_pip_install_timeout_constant_exists(self):
        from agent.tools.comfy_provision import _PIP_INSTALL_TIMEOUT
        assert isinstance(_PIP_INSTALL_TIMEOUT, (int, float))
        assert _PIP_INSTALL_TIMEOUT > 0

    def test_download_stream_timeout_constant_exists(self):
        from agent.tools.comfy_provision import _DOWNLOAD_STREAM_TIMEOUT
        assert isinstance(_DOWNLOAD_STREAM_TIMEOUT, (int, float))
        assert _DOWNLOAD_STREAM_TIMEOUT > 0

    def test_download_chunk_size_constant_exists(self):
        from agent.tools.comfy_provision import _DOWNLOAD_CHUNK_SIZE
        assert isinstance(_DOWNLOAD_CHUNK_SIZE, int)
        assert _DOWNLOAD_CHUNK_SIZE >= 4096, "Chunk size should be at least 4KB"

    def test_max_download_bytes_at_module_level(self):
        """_MAX_DOWNLOAD_BYTES must be a module-level constant, not inside a function."""
        import agent.tools.comfy_provision as mod
        assert hasattr(mod, "_MAX_DOWNLOAD_BYTES"), "_MAX_DOWNLOAD_BYTES must be at module level"
        assert mod._MAX_DOWNLOAD_BYTES > 0

    def test_clone_uses_git_clone_timeout(self):
        """git clone subprocess must use _GIT_CLONE_TIMEOUT, not a literal 120."""
        import inspect
        import agent.tools.comfy_provision as mod
        src = inspect.getsource(mod._handle_install_node_pack)
        # Should reference the constant name, not the raw literal inside the subprocess call
        # We verify the constant is used (not '120' after 'timeout=')
        import re
        # Find timeout= assignments in subprocess.run context
        timeouts_in_src = re.findall(r"timeout=(\S+)", src)
        # All timeouts should be constant references, not bare ints
        for t in timeouts_in_src:
            t_clean = t.rstrip(",)")
            assert not t_clean.isdigit(), (
                f"Bare numeric timeout {t_clean!r} found in install handler — use named constant"
            )


# ---------------------------------------------------------------------------
# Cycle 47 — install/download/uninstall required field guards
# ---------------------------------------------------------------------------

class TestInstallNodePackRequiredField:
    """install_node_pack must return structured error when url is missing or invalid."""

    def test_missing_url_returns_error(self):
        from agent.tools import comfy_provision
        result = json.loads(comfy_provision.handle("install_node_pack", {}))
        assert "error" in result
        assert "url" in result["error"].lower()

    def test_empty_url_returns_error(self):
        from agent.tools import comfy_provision
        result = json.loads(comfy_provision.handle("install_node_pack", {"url": ""}))
        assert "error" in result

    def test_none_url_returns_error(self):
        from agent.tools import comfy_provision
        result = json.loads(comfy_provision.handle("install_node_pack", {"url": None}))
        assert "error" in result


class TestDownloadModelRequiredFields:
    """download_model must return structured error when url or model_type is missing."""

    def test_missing_url_returns_error(self):
        from agent.tools import comfy_provision
        result = json.loads(comfy_provision.handle("download_model", {
            "model_type": "checkpoints",
        }))
        assert "error" in result
        assert "url" in result["error"].lower()

    def test_missing_model_type_returns_error(self):
        from agent.tools import comfy_provision
        result = json.loads(comfy_provision.handle("download_model", {
            "url": "https://example.com/model.safetensors",
        }))
        assert "error" in result
        assert "model_type" in result["error"].lower()

    def test_empty_url_returns_error(self):
        from agent.tools import comfy_provision
        result = json.loads(comfy_provision.handle("download_model", {
            "url": "", "model_type": "checkpoints",
        }))
        assert "error" in result

    def test_none_model_type_returns_error(self):
        from agent.tools import comfy_provision
        result = json.loads(comfy_provision.handle("download_model", {
            "url": "https://example.com/model.safetensors",
            "model_type": None,
        }))
        assert "error" in result


class TestUninstallNodePackRequiredField:
    """uninstall_node_pack must return structured error when name is missing."""

    def test_missing_name_returns_error(self):
        from agent.tools import comfy_provision
        result = json.loads(comfy_provision.handle("uninstall_node_pack", {}))
        assert "error" in result
        assert "name" in result["error"].lower()

    def test_empty_name_returns_error(self):
        from agent.tools import comfy_provision
        result = json.loads(comfy_provision.handle("uninstall_node_pack", {"name": ""}))
        assert "error" in result

    def test_none_name_returns_error(self):
        from agent.tools import comfy_provision
        result = json.loads(comfy_provision.handle("uninstall_node_pack", {"name": None}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Cycle 61 — URL validation exception logging
# ---------------------------------------------------------------------------

class TestURLValidationExceptionLogging:
    """Cycle 61: URL validation exceptions must be logged at DEBUG level."""

    def test_download_url_malformed_logs_debug(self, caplog):
        """_validate_download_url must log.debug on unexpected parse errors."""
        import logging
        from unittest.mock import patch
        from agent.tools.comfy_provision import _validate_download_url
        # Patch urlparse to raise unexpectedly (local import so patch source module)
        with patch("urllib.parse.urlparse", side_effect=RuntimeError("parse error")), \
             caplog.at_level(logging.DEBUG, logger="agent.tools.comfy_provision"):
            result = _validate_download_url("https://example.com/model.safetensors")
        assert result == "Invalid URL format."
        assert any("url" in r.message.lower() or "invalid" in r.message.lower()
                   for r in caplog.records)

    def test_git_url_malformed_logs_debug(self, caplog):
        """_validate_git_url must log.debug on unexpected parse errors."""
        import logging
        from unittest.mock import patch
        from agent.tools.comfy_provision import _validate_git_url
        with patch("urllib.parse.urlparse", side_effect=RuntimeError("parse error")), \
             caplog.at_level(logging.DEBUG, logger="agent.tools.comfy_provision"):
            result = _validate_git_url("https://github.com/org/repo.git")
        assert result == "Invalid URL format."
        assert any("url" in r.message.lower() or "invalid" in r.message.lower()
                   for r in caplog.records)
