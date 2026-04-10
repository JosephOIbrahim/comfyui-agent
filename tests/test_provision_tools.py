"""Tests for agent/stage/provision_tools.py — all mocked, no real I/O."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.stage.provision_tools import (
    TOOLS,
    _clear_provisioner_cache,
    handle,
)
from agent.stage.provisioner import DownloadHandle, ProvisionerError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(result: str) -> dict:
    """Parse JSON string returned by handle()."""
    return json.loads(result)


def _make_handle(
    prim_path: str = "/models/loras/test",
    status: str = "done",
    bytes_downloaded: int = 1024,
    total_bytes: int = 1024,
    dest_path: str = "/tmp/models/loras/test.safetensors",
    sha256_expected: str = "",
    sha256_actual: str = "",
    error: str = "",
) -> DownloadHandle:
    """Create a minimal DownloadHandle for testing."""
    h = DownloadHandle(
        prim_path=prim_path,
        filename="test.safetensors",
        dest_path=Path(dest_path),
        total_bytes=total_bytes,
        bytes_downloaded=bytes_downloaded,
        sha256_expected=sha256_expected,
        sha256_actual=sha256_actual,
        status=status,
        error=error,
    )
    return h


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_cache():
    """Ensure the provisioner cache is empty before every test."""
    _clear_provisioner_cache()
    yield
    _clear_provisioner_cache()


@pytest.fixture()
def mock_prov():
    """A MagicMock standing in for a Provisioner instance."""
    return MagicMock()


@pytest.fixture()
def patch_get_provisioner(mock_prov):
    """Patch _get_provisioner to return mock_prov."""
    with patch("agent.stage.provision_tools._get_provisioner", return_value=mock_prov) as p:
        yield p, mock_prov


# ---------------------------------------------------------------------------
# TOOLS schema sanity checks
# ---------------------------------------------------------------------------

def test_tools_list_has_three_entries():
    assert len(TOOLS) == 3


def test_tools_names():
    names = {t["name"] for t in TOOLS}
    assert names == {"provision_download", "provision_verify", "provision_status"}


def test_each_tool_has_required_prim_path():
    for tool in TOOLS:
        schema = tool["input_schema"]
        assert "prim_path" in schema["properties"]
        assert "prim_path" in schema["required"]


def test_each_tool_has_description():
    for tool in TOOLS:
        assert tool["description"]


# ---------------------------------------------------------------------------
# provision_download — success paths
# ---------------------------------------------------------------------------

def test_provision_download_success_no_hash(patch_get_provisioner):
    _, mock_prov = patch_get_provisioner
    h = _make_handle(sha256_expected="", sha256_actual="")
    mock_prov.download.return_value = h

    result = _parse(handle("provision_download", {"prim_path": "/models/loras/test"}))

    assert result["status"] == "done"
    assert result["bytes_downloaded"] == 1024
    assert result["sha256_match"] is None
    mock_prov.download.assert_called_once_with("/models/loras/test")


def test_provision_download_success_with_matching_hash(patch_get_provisioner):
    _, mock_prov = patch_get_provisioner
    sha = "abc123def456"
    h = _make_handle(sha256_expected=sha, sha256_actual=sha)
    mock_prov.download.return_value = h

    result = _parse(handle("provision_download", {"prim_path": "/models/loras/test"}))

    assert result["sha256_match"] is True


def test_provision_download_success_with_mismatched_hash(patch_get_provisioner):
    _, mock_prov = patch_get_provisioner
    h = _make_handle(sha256_expected="aaa", sha256_actual="bbb")
    mock_prov.download.return_value = h

    result = _parse(handle("provision_download", {"prim_path": "/models/loras/test"}))

    assert result["sha256_match"] is False


def test_provision_download_returns_dest_path(patch_get_provisioner):
    _, mock_prov = patch_get_provisioner
    h = _make_handle(dest_path="/tmp/models/loras/test.safetensors")
    mock_prov.download.return_value = h

    result = _parse(handle("provision_download", {"prim_path": "/models/loras/test"}))

    assert "dest_path" in result
    assert "test.safetensors" in result["dest_path"]


# ---------------------------------------------------------------------------
# provision_download — error paths
# ---------------------------------------------------------------------------

def test_provision_download_provisioner_error(patch_get_provisioner):
    _, mock_prov = patch_get_provisioner
    mock_prov.download.side_effect = ProvisionerError("No source_url")

    result = _parse(handle("provision_download", {"prim_path": "/models/loras/test"}))

    assert "error" in result
    assert "No source_url" in result["error"]


def test_provision_download_no_stage():
    with patch("agent.stage.provision_tools._get_provisioner", return_value=None):
        result = _parse(handle("provision_download", {"prim_path": "/models/loras/test"}))

    assert "error" in result
    assert "CognitiveWorkflowStage" in result["error"]


def test_provision_download_unexpected_exception(patch_get_provisioner):
    _, mock_prov = patch_get_provisioner
    mock_prov.download.side_effect = RuntimeError("disk full")

    result = _parse(handle("provision_download", {"prim_path": "/models/loras/test"}))

    assert "error" in result
    assert "disk full" in result["error"]


# ---------------------------------------------------------------------------
# provision_verify — success paths
# ---------------------------------------------------------------------------

def test_provision_verify_pass(patch_get_provisioner):
    _, mock_prov = patch_get_provisioner
    mock_prov.verify.return_value = True

    result = _parse(handle("provision_verify", {"prim_path": "/models/loras/test"}))

    assert result["sha256_verified"] is True
    assert result["result"] == "pass"
    mock_prov.verify.assert_called_once_with("/models/loras/test")


def test_provision_verify_fail(patch_get_provisioner):
    _, mock_prov = patch_get_provisioner
    mock_prov.verify.return_value = False

    result = _parse(handle("provision_verify", {"prim_path": "/models/loras/test"}))

    assert result["sha256_verified"] is False
    assert result["result"] == "fail"


# ---------------------------------------------------------------------------
# provision_verify — error paths
# ---------------------------------------------------------------------------

def test_provision_verify_provisioner_error(patch_get_provisioner):
    _, mock_prov = patch_get_provisioner
    mock_prov.verify.side_effect = ProvisionerError("File not found on disk")

    result = _parse(handle("provision_verify", {"prim_path": "/models/loras/test"}))

    assert "error" in result
    assert "File not found" in result["error"]


def test_provision_verify_no_stage():
    with patch("agent.stage.provision_tools._get_provisioner", return_value=None):
        result = _parse(handle("provision_verify", {"prim_path": "/models/loras/test"}))

    assert "error" in result
    assert "CognitiveWorkflowStage" in result["error"]


# ---------------------------------------------------------------------------
# provision_status — with live Provisioner
# ---------------------------------------------------------------------------

def test_provision_status_with_handle(patch_get_provisioner):
    _, mock_prov = patch_get_provisioner
    mock_prov.status.return_value = {
        "prim_path": "/models/loras/test",
        "registry_status": "downloading",
        "handle_status": "downloading",
        "bytes_downloaded": 512,
        "total_bytes": 1024,
        "progress_pct": 50.0,
        "dest_path": "/tmp/models/loras/test.safetensors",
        "error": "",
        "sha256_match": None,
    }

    result = _parse(handle("provision_status", {"prim_path": "/models/loras/test"}))

    assert result["registry_status"] == "downloading"
    assert result["progress_pct"] == 50.0
    assert result["bytes_downloaded"] == 512
    mock_prov.status.assert_called_once_with("/models/loras/test")


def test_provision_status_no_handle(patch_get_provisioner):
    _, mock_prov = patch_get_provisioner
    mock_prov.status.return_value = {
        "prim_path": "/models/loras/test",
        "registry_status": "available",
        "handle_status": "no_active_handle",
        "bytes_downloaded": 0,
        "total_bytes": 0,
        "progress_pct": 0.0,
        "dest_path": "",
        "error": "",
        "sha256_match": None,
    }

    result = _parse(handle("provision_status", {"prim_path": "/models/loras/test"}))

    assert result["handle_status"] == "no_active_handle"
    assert result["progress_pct"] == 0.0


def test_provision_status_complete(patch_get_provisioner):
    _, mock_prov = patch_get_provisioner
    mock_prov.status.return_value = {
        "prim_path": "/models/loras/test",
        "registry_status": "materialized",
        "handle_status": "done",
        "bytes_downloaded": 2048,
        "total_bytes": 2048,
        "progress_pct": 100.0,
        "dest_path": "/tmp/models/loras/test.safetensors",
        "error": "",
        "sha256_match": True,
    }

    result = _parse(handle("provision_status", {"prim_path": "/models/loras/test"}))

    assert result["registry_status"] == "materialized"
    assert result["progress_pct"] == 100.0
    assert result["sha256_match"] is True


# ---------------------------------------------------------------------------
# provision_status — degraded mode (no Provisioner)
# ---------------------------------------------------------------------------

def test_provision_status_no_provisioner_no_stage():
    mock_ctx = MagicMock()
    mock_ctx.stage = None

    with patch("agent.stage.provision_tools._get_provisioner", return_value=None), \
         patch("agent.stage.provision_tools.get_model") as mock_get_model, \
         patch("agent.session_context.get_session_context", return_value=mock_ctx):
        result = _parse(handle("provision_status", {"prim_path": "/models/loras/test"}))

    assert result["registry_status"] == "stage_unavailable"
    mock_get_model.assert_not_called()


def test_provision_status_no_provisioner_model_not_registered():
    mock_stage = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.stage = mock_stage

    with patch("agent.stage.provision_tools._get_provisioner", return_value=None), \
         patch("agent.stage.provision_tools.get_model", return_value=None), \
         patch("agent.session_context.get_session_context", return_value=mock_ctx):
        result = _parse(handle("provision_status", {"prim_path": "/models/loras/test"}))

    assert result["registry_status"] == "not_registered"


def test_provision_status_no_provisioner_with_registry_entry():
    mock_stage = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.stage = mock_stage

    with patch("agent.stage.provision_tools._get_provisioner", return_value=None), \
         patch("agent.stage.provision_tools.get_model", return_value={"status": "materialized"}), \
         patch("agent.session_context.get_session_context", return_value=mock_ctx):
        result = _parse(handle("provision_status", {"prim_path": "/models/loras/test"}))

    assert result["registry_status"] == "materialized"


# ---------------------------------------------------------------------------
# handle() dispatch
# ---------------------------------------------------------------------------

def test_handle_unknown_tool():
    result = _parse(handle("provision_unknown", {"prim_path": "/models/loras/test"}))
    assert "error" in result
    assert "Unknown tool" in result["error"]


def test_handle_dispatches_provision_download(patch_get_provisioner):
    _, mock_prov = patch_get_provisioner
    mock_prov.download.return_value = _make_handle()

    result = _parse(handle("provision_download", {"prim_path": "/models/loras/test"}))
    assert "status" in result


def test_handle_dispatches_provision_verify(patch_get_provisioner):
    _, mock_prov = patch_get_provisioner
    mock_prov.verify.return_value = True

    result = _parse(handle("provision_verify", {"prim_path": "/models/loras/test"}))
    assert result["result"] == "pass"


def test_handle_dispatches_provision_status(patch_get_provisioner):
    _, mock_prov = patch_get_provisioner
    mock_prov.status.return_value = {
        "prim_path": "/models/loras/test",
        "registry_status": "available",
        "handle_status": "no_active_handle",
        "bytes_downloaded": 0,
        "total_bytes": 0,
        "progress_pct": 0.0,
        "dest_path": "",
        "error": "",
        "sha256_match": None,
    }

    result = _parse(handle("provision_status", {"prim_path": "/models/loras/test"}))
    assert "registry_status" in result


# ---------------------------------------------------------------------------
# _get_provisioner — caching behaviour
# ---------------------------------------------------------------------------

def test_get_provisioner_caches_instance():
    """Second call with same session_id returns the same object."""
    mock_stage = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.ensure_stage.return_value = mock_stage

    with patch("agent.session_context.get_session_context", return_value=mock_ctx), \
         patch("agent.stage.provision_tools.Provisioner") as MockProv:
        MockProv.return_value = MagicMock()

        from agent.stage.provision_tools import _get_provisioner
        p1 = _get_provisioner("session-xyz")
        p2 = _get_provisioner("session-xyz")

    assert p1 is p2
    assert MockProv.call_count == 1


def test_get_provisioner_returns_none_when_no_stage():
    mock_ctx = MagicMock()
    mock_ctx.ensure_stage.return_value = None

    with patch("agent.session_context.get_session_context", return_value=mock_ctx):
        from agent.stage.provision_tools import _get_provisioner
        result = _get_provisioner("session-no-stage-2")

    assert result is None


# ---------------------------------------------------------------------------
# JSON output sanity — all handlers return valid JSON
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool_name,extra_kwargs", [
    ("provision_download", {}),
    ("provision_verify", {}),
    ("provision_status", {}),
])
def test_handle_always_returns_valid_json(tool_name, extra_kwargs, patch_get_provisioner):
    _, mock_prov = patch_get_provisioner
    mock_prov.download.return_value = _make_handle()
    mock_prov.verify.return_value = True
    mock_prov.status.return_value = {
        "prim_path": "/models/loras/test",
        "registry_status": "available",
        "handle_status": "no_active_handle",
        "bytes_downloaded": 0,
        "total_bytes": 0,
        "progress_pct": 0.0,
        "dest_path": "",
        "error": "",
        "sha256_match": None,
    }

    raw = handle(tool_name, {"prim_path": "/models/loras/test", **extra_kwargs})
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Cycle 55 — required field guards for provision_tools handlers
# ---------------------------------------------------------------------------

class TestProvisionDownloadRequiredField:
    def test_missing_prim_path_returns_error(self):
        result = json.loads(handle("provision_download", {}))
        assert "error" in result
        assert "prim_path" in result["error"].lower()

    def test_empty_prim_path_returns_error(self):
        result = json.loads(handle("provision_download", {"prim_path": ""}))
        assert "error" in result

    def test_none_prim_path_returns_error(self):
        result = json.loads(handle("provision_download", {"prim_path": None}))
        assert "error" in result


class TestProvisionVerifyRequiredField:
    def test_missing_prim_path_returns_error(self):
        result = json.loads(handle("provision_verify", {}))
        assert "error" in result
        assert "prim_path" in result["error"].lower()

    def test_integer_prim_path_returns_error(self):
        result = json.loads(handle("provision_verify", {"prim_path": 123}))
        assert "error" in result


class TestProvisionStatusRequiredField:
    def test_missing_prim_path_returns_error(self):
        result = json.loads(handle("provision_status", {}))
        assert "error" in result
        assert "prim_path" in result["error"].lower()

    def test_none_prim_path_returns_error(self):
        result = json.loads(handle("provision_status", {"prim_path": None}))
        assert "error" in result
