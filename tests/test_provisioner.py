"""Tests for agent/stage/provisioner.py — all mocked, no real network calls."""

from __future__ import annotations

import hashlib
import io
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from agent.stage.cognitive_stage import CognitiveWorkflowStage
from agent.stage.model_registry import register_model, get_model
from agent.stage.provisioner import (
    DEFAULT_CHUNK_BYTES,
    DownloadHandle,
    Provisioner,
    ProvisionerError,
    _hash_file,
    _hash_file_partial,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cws(tmp_path):
    """Fresh CognitiveWorkflowStage backed by a tmp USD file."""
    stage_file = tmp_path / "test_stage.usda"
    return CognitiveWorkflowStage(str(stage_file))


@pytest.fixture()
def models_dir(tmp_path):
    """Empty models directory."""
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture()
def provisioner(cws, models_dir):
    """Provisioner wired to the test stage and tmp models dir."""
    return Provisioner(cws, models_dir=models_dir)


def _make_model(cws, model_type="loras", filename="test_lora.safetensors",
                source_url="https://example.com/test_lora.safetensors",
                sha256=""):
    """Register a minimal model and return prim_path."""
    return register_model(
        cws, model_type, filename,
        source_url=source_url,
        sha256=sha256,
    )


# ---------------------------------------------------------------------------
# DownloadHandle unit tests
# ---------------------------------------------------------------------------

class TestDownloadHandle:
    def test_progress_pct_zero_when_total_unknown(self):
        h = DownloadHandle(prim_path="/models/loras/x", filename="x.safetensors",
                           dest_path=Path("/tmp/x.safetensors"), total_bytes=0,
                           bytes_downloaded=500)
        assert h.progress_pct() == 0.0

    def test_progress_pct_correct(self):
        h = DownloadHandle(prim_path="/models/loras/x", filename="x.safetensors",
                           dest_path=Path("/tmp/x.safetensors"), total_bytes=1000,
                           bytes_downloaded=250)
        assert h.progress_pct() == 25.0

    def test_progress_pct_capped_at_100(self):
        h = DownloadHandle(prim_path="/models/loras/x", filename="x.safetensors",
                           dest_path=Path("/tmp/x.safetensors"), total_bytes=100,
                           bytes_downloaded=200)
        assert h.progress_pct() == 100.0


# ---------------------------------------------------------------------------
# Provisioner construction
# ---------------------------------------------------------------------------

class TestProvisionerConstruction:
    def test_chunk_size_floor(self, cws, models_dir):
        p = Provisioner(cws, models_dir, chunk_bytes=10)
        assert p._chunk_bytes == 4096

    def test_chunk_size_custom(self, cws, models_dir):
        p = Provisioner(cws, models_dir, chunk_bytes=512_000)
        assert p._chunk_bytes == 512_000

    def test_requires_httpx(self, cws, models_dir):
        with patch("agent.stage.provisioner.HAS_HTTPX", False):
            with pytest.raises(ProvisionerError, match="httpx"):
                Provisioner(cws, models_dir)


# ---------------------------------------------------------------------------
# Helpers: _hash_file, _hash_file_partial
# ---------------------------------------------------------------------------

class TestHashHelpers:
    def test_hash_file(self, tmp_path):
        f = tmp_path / "blob.bin"
        data = b"hello world"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert _hash_file(f) == expected

    def test_hash_file_partial_seeds_hasher(self, tmp_path):
        data = b"abcdefghij"  # 10 bytes
        f = tmp_path / "data.bin"
        f.write_bytes(data)
        # Hash first 5 bytes
        hasher = _hash_file_partial(f, 5)
        hasher.update(data[5:])  # feed the rest
        full_hasher = hashlib.sha256(data)
        assert hasher.hexdigest() == full_hasher.hexdigest()

    def test_hash_file_partial_empty_size(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"content")
        # 0 bytes → empty hasher
        hasher = _hash_file_partial(f, 0)
        assert hasher.hexdigest() == hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# download() — happy path
# ---------------------------------------------------------------------------

def _make_mock_response(content: bytes, status_code: int = 200, content_length: int | None = None):
    """Build a mock httpx streaming response."""
    if content_length is None:
        content_length = len(content)

    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.headers = {"content-length": str(content_length)}
    mock_resp.raise_for_status = MagicMock()

    chunk_size = 4
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    if not chunks:
        chunks = [b""]
    mock_resp.iter_bytes = MagicMock(return_value=iter(chunks))
    return mock_resp


def _patch_httpx_stream(content: bytes, status_code: int = 200):
    """Context manager that patches httpx.Client to return fake streamed content."""
    mock_resp = _make_mock_response(content, status_code)
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_resp)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_ctx)

    mock_client_ctx = MagicMock()
    mock_client_ctx.__enter__ = MagicMock(return_value=mock_client)
    mock_client_ctx.__exit__ = MagicMock(return_value=False)

    return patch("agent.stage.provisioner.httpx.Client", return_value=mock_client_ctx)


class TestDownload:
    def test_download_creates_file(self, provisioner, cws, models_dir):
        content = b"fake safetensors data"
        prim = _make_model(cws)

        with _patch_httpx_stream(content):
            handle = provisioner.download(prim)

        dest = models_dir / "loras" / "test_lora.safetensors"
        assert dest.exists()
        assert dest.read_bytes() == content
        assert handle.status == "done"
        assert handle.bytes_downloaded == len(content)

    def test_download_updates_registry_status_materialized(self, provisioner, cws, models_dir):
        content = b"model bytes"
        prim = _make_model(cws)

        with _patch_httpx_stream(content):
            provisioner.download(prim)

        model = get_model(cws, prim)
        assert model["status"] == "materialized"

    def test_download_sets_file_path_in_registry(self, provisioner, cws, models_dir):
        content = b"data"
        prim = _make_model(cws)

        with _patch_httpx_stream(content):
            provisioner.download(prim)

        model = get_model(cws, prim)
        assert "test_lora.safetensors" in model.get("file_path", "")

    def test_download_verifies_sha256_on_success(self, provisioner, cws, models_dir):
        content = b"verified content"
        sha256 = hashlib.sha256(content).hexdigest()
        prim = _make_model(cws, sha256=sha256)

        with _patch_httpx_stream(content):
            handle = provisioner.download(prim)

        assert handle.sha256_actual == sha256
        model = get_model(cws, prim)
        assert model["status"] == "materialized"

    def test_download_raises_on_sha256_mismatch(self, provisioner, cws, models_dir):
        content = b"real content"
        prim = _make_model(cws, sha256="deadbeef" * 8)  # wrong hash

        with pytest.raises(ProvisionerError, match="SHA256 mismatch"):
            with _patch_httpx_stream(content):
                provisioner.download(prim)

    def test_download_marks_failed_on_sha256_mismatch(self, provisioner, cws, models_dir):
        content = b"real content"
        prim = _make_model(cws, sha256="deadbeef" * 8)

        with pytest.raises(ProvisionerError):
            with _patch_httpx_stream(content):
                provisioner.download(prim)

        model = get_model(cws, prim)
        assert model["status"] == "failed"

    def test_download_no_sha256_skips_verification(self, provisioner, cws, models_dir):
        content = b"unverified but fine"
        prim = _make_model(cws, sha256="")

        with _patch_httpx_stream(content):
            handle = provisioner.download(prim)

        assert handle.status == "done"
        assert handle.sha256_actual == ""

    def test_download_raises_if_not_registered(self, provisioner, cws):
        with pytest.raises(ProvisionerError, match="not registered"):
            provisioner.download("/models/loras/nonexistent")

    def test_download_raises_if_no_url(self, provisioner, cws):
        prim = register_model(cws, "loras", "no_url.safetensors")
        with pytest.raises(ProvisionerError, match="source_url"):
            provisioner.download(prim)

    def test_download_calls_progress_callback(self, cws, models_dir):
        content = b"A" * 100
        prim = _make_model(cws)
        calls = []

        def cb(pp, downloaded, total):
            calls.append((pp, downloaded, total))

        p = Provisioner(cws, models_dir, progress_cb=cb)
        with _patch_httpx_stream(content):
            p.download(prim)

        assert len(calls) > 0
        assert calls[-1][0] == prim

    def test_progress_callback_error_does_not_abort(self, cws, models_dir):
        content = b"data bytes"
        prim = _make_model(cws)

        def bad_cb(pp, dl, total):
            raise RuntimeError("callback exploded")

        p = Provisioner(cws, models_dir, progress_cb=bad_cb)
        with _patch_httpx_stream(content):
            handle = p.download(prim)  # should not raise

        assert handle.status == "done"

    def test_download_http_error_marks_failed(self, provisioner, cws, models_dir):
        prim = _make_model(cws)

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.headers = {"content-length": "0"}
        mock_resp.raise_for_status = MagicMock(
            side_effect=Exception("404 Not Found")
        )
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_resp)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_ctx)
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client)
        mock_client_ctx.__exit__ = MagicMock(return_value=False)

        with patch("agent.stage.provisioner.httpx.Client", return_value=mock_client_ctx):
            with pytest.raises(ProvisionerError):
                provisioner.download(prim)

        model = get_model(cws, prim)
        assert model["status"] == "failed"


# ---------------------------------------------------------------------------
# Resume-on-failure (Range header)
# ---------------------------------------------------------------------------

class TestResumeDownload:
    def test_resume_sends_range_header(self, provisioner, cws, models_dir):
        # Write a partial file first
        partial_content = b"partial"
        remaining_content = b" rest"
        dest = models_dir / "loras" / "test_lora.safetensors"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(partial_content)

        prim = _make_model(cws)

        # Mock: server returns 206 Partial Content for the remainder
        mock_resp = _make_mock_response(remaining_content, status_code=206)
        mock_resp.headers["content-length"] = str(len(remaining_content))
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_resp)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_ctx)
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client)
        mock_client_ctx.__exit__ = MagicMock(return_value=False)

        with patch("agent.stage.provisioner.httpx.Client", return_value=mock_client_ctx):
            handle = provisioner.download(prim)

        # Verify Range header was sent
        call_kwargs = mock_client.stream.call_args
        headers_sent = call_kwargs[1].get("headers", {}) or call_kwargs[0][2] if len(call_kwargs[0]) > 2 else {}
        # Check via keyword args
        if "headers" in call_kwargs.kwargs:
            headers_sent = call_kwargs.kwargs["headers"]
        assert "Range" in headers_sent
        assert f"bytes={len(partial_content)}-" == headers_sent["Range"]

        # File should have both parts
        assert dest.read_bytes() == partial_content + remaining_content
        assert handle.status == "done"


# ---------------------------------------------------------------------------
# verify()
# ---------------------------------------------------------------------------

class TestVerify:
    def test_verify_passes_matching_hash(self, provisioner, cws, models_dir):
        content = b"model content"
        sha256 = hashlib.sha256(content).hexdigest()
        dest = models_dir / "loras" / "test_lora.safetensors"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)

        prim = _make_model(cws, sha256=sha256)
        # Update file_path in registry
        from agent.stage.model_registry import update_status
        update_status(cws, prim, "materialized", file_path=str(dest))

        assert provisioner.verify(prim) is True

    def test_verify_fails_mismatched_hash(self, provisioner, cws, models_dir):
        content = b"real content"
        dest = models_dir / "loras" / "test_lora.safetensors"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)

        prim = _make_model(cws, sha256="aabbccdd" * 8)
        from agent.stage.model_registry import update_status
        update_status(cws, prim, "materialized", file_path=str(dest))

        result = provisioner.verify(prim)
        assert result is False

        # Registry should be marked failed
        model = get_model(cws, prim)
        assert model["status"] == "failed"

    def test_verify_passes_when_no_expected_hash(self, provisioner, cws, models_dir):
        content = b"anything"
        dest = models_dir / "loras" / "test_lora.safetensors"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)

        prim = _make_model(cws, sha256="")
        from agent.stage.model_registry import update_status
        update_status(cws, prim, "materialized", file_path=str(dest))

        assert provisioner.verify(prim) is True

    def test_verify_raises_if_file_missing(self, provisioner, cws, models_dir):
        prim = _make_model(cws, sha256="abc")
        from agent.stage.model_registry import update_status
        update_status(cws, prim, "materialized",
                      file_path=str(models_dir / "loras" / "nonexistent.safetensors"))

        with pytest.raises(ProvisionerError, match="not found on disk"):
            provisioner.verify(prim)

    def test_verify_raises_if_not_registered(self, provisioner, cws):
        with pytest.raises(ProvisionerError, match="not registered"):
            provisioner.verify("/models/loras/ghost")

    def test_verify_derives_path_from_type_and_filename(self, provisioner, cws, models_dir):
        content = b"derived path test"
        sha256 = hashlib.sha256(content).hexdigest()

        # Register without file_path
        prim = _make_model(cws, sha256=sha256)
        dest = models_dir / "loras" / "test_lora.safetensors"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)

        # No file_path in registry — provisioner derives from type+filename
        assert provisioner.verify(prim) is True


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_no_handle(self, provisioner, cws):
        prim = _make_model(cws)
        s = provisioner.status(prim)
        assert s["registry_status"] == "available"
        assert s["handle_status"] == "no_active_handle"
        assert s["progress_pct"] == 0.0

    def test_status_not_registered(self, provisioner, cws):
        s = provisioner.status("/models/loras/ghost")
        assert s["registry_status"] == "not_registered"

    def test_status_after_download(self, provisioner, cws, models_dir):
        content = b"status test content"
        prim = _make_model(cws)

        with _patch_httpx_stream(content):
            provisioner.download(prim)

        s = provisioner.status(prim)
        assert s["registry_status"] == "materialized"
        assert s["handle_status"] == "done"
        assert s["bytes_downloaded"] == len(content)
        assert s["progress_pct"] == 100.0

    def test_status_sha256_match_flag(self, cws, models_dir):
        content = b"sha verified"
        sha256 = hashlib.sha256(content).hexdigest()
        prim = _make_model(cws, sha256=sha256)

        p = Provisioner(cws, models_dir)
        with _patch_httpx_stream(content):
            p.download(prim)

        s = p.status(prim)
        assert s["sha256_match"] is True

    def test_status_sha256_mismatch_flag(self, cws, models_dir):
        content = b"wrong content"
        prim = _make_model(cws, sha256="deadbeef" * 8)

        p = Provisioner(cws, models_dir)
        with pytest.raises(ProvisionerError):
            with _patch_httpx_stream(content):
                p.download(prim)

        s = p.status(prim)
        assert s["sha256_match"] is False
        assert s["error"] != ""
