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


# ---------------------------------------------------------------------------
# Cycle 67: boolean coercion guards (auto_install, auto_fix)
# ---------------------------------------------------------------------------

class TestBooleanCoercionCycle67:
    """Cycle 67: string 'false' must not be truthy for action-gating boolean flags."""

    def test_auto_install_string_false_does_not_install(self):
        """repair_workflow with auto_install='false' (string) must not trigger install."""
        import json as _json
        from unittest.mock import patch, MagicMock
        from agent.tools import comfy_provision

        # find_missing_nodes returns dicts with class_type, pack_url, pack_name
        missing_result = _json.dumps({
            "missing": [
                {"class_type": "SomeNode",
                 "pack_url": "https://github.com/org/pack",
                 "pack_name": "SomePack"},
            ],
            "installed": [],
        })

        # _handle_install_node_pack is called directly (not via handle()) when auto_install is truthy
        mock_install = MagicMock(return_value=_json.dumps({"installed": True}))

        # discover_handle is a local import from comfy_discover — patch at source module
        with patch("agent.tools.comfy_discover.handle", return_value=missing_result), \
             patch("agent.tools.comfy_provision._handle_install_node_pack", mock_install):
            comfy_provision._handle_repair_workflow({"auto_install": "false"})

        assert mock_install.call_count == 0, \
            "auto_install='false' (string) was truthy — install was triggered"

    def _make_refs(self, exists: bool):
        return [
            {"node_id": "1", "class_type": "CheckpointLoaderSimple",
             "field": "ckpt_name", "value": "missing.safetensors",
             "model_type": "checkpoint",
             "exists": exists, "best_match": "found.safetensors" if not exists else None},
        ]

    def test_auto_fix_string_false_does_not_modify_workflow(self):
        """reconfigure_workflow with auto_fix='false' must not apply substitutions."""
        import json as _json
        from unittest.mock import patch
        from agent.tools import comfy_provision

        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "missing.safetensors"}}
        }

        # _get_state is a local import from workflow_patch — patch at source module
        with patch("agent.tools.workflow_patch._get_state",
                   return_value={"current_workflow": workflow}), \
             patch("agent.tools.comfy_provision._scan_model_references",
                   return_value=self._make_refs(exists=False)):
            result = _json.loads(comfy_provision._handle_reconfigure_workflow(
                {"auto_fix": "false"}
            ))

        assert result.get("fixes_applied", 0) == 0, \
            "auto_fix='false' (string) was truthy — workflow was modified"

    def test_auto_fix_true_bool_applies_fix(self):
        """reconfigure_workflow with auto_fix=True (bool) must apply substitutions."""
        import json as _json
        from unittest.mock import patch
        from agent.tools import comfy_provision

        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "missing.safetensors"}}
        }

        with patch("agent.tools.workflow_patch._get_state",
                   return_value={"current_workflow": workflow}), \
             patch("agent.tools.comfy_provision._scan_model_references",
                   return_value=self._make_refs(exists=False)):
            result = _json.loads(comfy_provision._handle_reconfigure_workflow(
                {"auto_fix": True}
            ))

        assert result.get("fixes_applied", 0) == 1, \
            "auto_fix=True (bool) must apply the substitution"


# ---------------------------------------------------------------------------
# Cycle 68: repair_workflow error propagation from find_missing_nodes
# ---------------------------------------------------------------------------

class TestRepairWorkflowErrorPropagationCycle68:
    """Cycle 68: repair_workflow must surface find_missing_nodes errors, not say 'clean'."""

    def test_discover_error_surfaces_as_error_not_clean(self):
        """When find_missing_nodes returns an error dict, repair_workflow must surface it."""
        import json as _json
        from unittest.mock import patch
        from agent.tools import comfy_provision

        # find_missing_nodes fails (e.g., ComfyUI not running)
        error_result = _json.dumps({"error": "ComfyUI not reachable at localhost:8188."})

        with patch("agent.tools.comfy_discover.handle", return_value=error_result):
            result = _json.loads(comfy_provision._handle_repair_workflow({}))

        # Must NOT say "clean" — that would be a silent false positive
        assert result.get("status") != "clean", \
            "repair_workflow returned 'clean' when discovery actually failed"
        assert "error" in result, \
            "repair_workflow must propagate the discovery error"

    def test_discover_error_message_includes_original_error(self):
        """The error message from repair_workflow must include the callee's error text."""
        import json as _json
        from unittest.mock import patch
        from agent.tools import comfy_provision

        error_result = _json.dumps({"error": "ComfyUI not reachable at localhost:8188."})

        with patch("agent.tools.comfy_discover.handle", return_value=error_result):
            result = _json.loads(comfy_provision._handle_repair_workflow({}))

        assert "comfyui" in result.get("error", "").lower() or \
               "not reachable" in result.get("error", "").lower(), \
            "Error message should help user understand what failed"

    def test_successful_discover_still_works(self):
        """repair_workflow must still work when find_missing_nodes succeeds with no missing."""
        import json as _json
        from unittest.mock import patch
        from agent.tools import comfy_provision

        clean_result = _json.dumps({"missing": [], "installed": ["KSampler"]})

        with patch("agent.tools.comfy_discover.handle", return_value=clean_result):
            result = _json.loads(comfy_provision._handle_repair_workflow({}))

        assert result.get("status") == "clean"


# ---------------------------------------------------------------------------
# download_model symlink bypass — defense-in-depth path validation
# ---------------------------------------------------------------------------

class TestDownloadModelSymlinkBypass:
    """download_model must reject targets that escape MODELS_DIR via a
    symlink anywhere in the path chain.

    The previous validation only resolved `MODELS_DIR / model_type`,
    leaving subfolder + filename components unresolved. A symlink at
    `checkpoints/X` pointing to `/etc` would let the download write to
    `/etc/<filename>` even though `MODELS_DIR / checkpoints` itself
    resolves cleanly.
    """

    def test_symlink_in_subfolder_rejected(self, tmp_path):
        """A subfolder symlink pointing outside MODELS_DIR must be rejected."""
        import json as _json
        import os
        import sys
        from unittest.mock import patch
        from agent.tools import comfy_provision

        # Set up: fake MODELS_DIR with a normal `checkpoints/` subdir,
        # plus a sibling "escape" target outside MODELS_DIR.
        fake_models = tmp_path / "models"
        fake_models.mkdir()
        (fake_models / "checkpoints").mkdir()

        escape_target = tmp_path / "escape_zone"
        escape_target.mkdir()

        # Plant the malicious symlink: models/checkpoints/evil → escape_zone
        symlink_path = fake_models / "checkpoints" / "evil"
        try:
            os.symlink(escape_target, symlink_path, target_is_directory=True)
        except (OSError, NotImplementedError) as e:
            import pytest
            pytest.skip(
                f"Symlink creation not supported on this platform "
                f"(Windows requires admin/dev mode): {e}"
            )

        # Sanity: the symlink really does resolve outside MODELS_DIR
        assert symlink_path.resolve() == escape_target.resolve()
        if sys.platform == "win32":
            # Windows path compare is case-insensitive — use lower()
            assert not str(symlink_path.resolve()).lower().startswith(
                str(fake_models.resolve()).lower()
            )

        # Attempt the download into the malicious subfolder
        with patch.object(comfy_provision, "MODELS_DIR", fake_models):
            result = comfy_provision._handle_download_model({
                "url": "https://huggingface.co/test-model.safetensors",
                "model_type": "checkpoints",
                "subfolder": "evil",
                "filename": "leaked.safetensors",
            })

        result_data = _json.loads(result)
        assert "error" in result_data, f"Expected error, got: {result_data}"
        assert "Access denied" in result_data["error"], (
            f"Expected 'Access denied' in error, got: {result_data['error']}"
        )

        # Confirm nothing was written to the escape target
        assert not (escape_target / "leaked.safetensors").exists()
        assert not (escape_target / "leaked.safetensors.download").exists()

    def test_legitimate_subfolder_still_works(self, tmp_path):
        """A normal (non-symlink) subfolder must NOT trigger the validation reject.

        Verifies the fix doesn't break legitimate subfolder downloads. We
        mock httpx.stream so the test doesn't actually hit the network —
        we only need to confirm validation passes and reaches the download
        attempt.
        """
        import json as _json
        from unittest.mock import patch, MagicMock
        from agent.tools import comfy_provision

        fake_models = tmp_path / "models"
        fake_models.mkdir()
        (fake_models / "checkpoints" / "subdir").mkdir(parents=True)

        # Mock httpx.stream to fail fast — we only need to confirm the
        # validation block does NOT reject this legitimate path.
        mock_stream = MagicMock()
        mock_stream.return_value.__enter__.side_effect = RuntimeError("network mocked")

        with patch.object(comfy_provision, "MODELS_DIR", fake_models):
            with patch("agent.tools.comfy_provision.httpx.stream", mock_stream):
                result = comfy_provision._handle_download_model({
                    "url": "https://huggingface.co/test-model.safetensors",
                    "model_type": "checkpoints",
                    "subfolder": "subdir",
                    "filename": "legit.safetensors",
                })

        result_data = _json.loads(result)
        # Whatever error we get, it must NOT be the symlink-bypass rejection
        if "error" in result_data:
            assert "Access denied" not in result_data["error"], (
                f"Validation incorrectly rejected legitimate path: {result_data['error']}"
            )
