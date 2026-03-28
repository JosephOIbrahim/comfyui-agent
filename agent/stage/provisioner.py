"""Provisioner — streaming model download with SHA256 verification.

Download pipeline for model files referenced in the model_registry.

Status lifecycle (in registry):
  available   → start_download()   → downloading
  downloading → download complete  → materialized  (on success)
  downloading → error/abort        → failed

Features:
- httpx streaming with configurable chunk size and progress callback
- SHA256 verification against model_registry entry
- Resume-on-failure via HTTP Range header for partial downloads
- Download destination respects COMFYUI_DATABASE / models / {type}
- Thread-safe: one DownloadHandle per active transfer, tracked by prim_path

Usage::

    from agent.stage.cognitive_stage import CognitiveWorkflowStage
    from agent.stage.model_registry import register_model
    from agent.stage.provisioner import Provisioner

    cws = CognitiveWorkflowStage()
    prim = register_model(cws, "loras", "my_lora.safetensors",
                          source_url="https://...", sha256="abc123...")
    p = Provisioner(cws, models_dir=Path("G:/COMFYUI_Database/models"))
    p.download(prim)                # blocking
    info = p.status(prim)
    print(info["bytes_downloaded"], info["progress_pct"])
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .cognitive_stage import CognitiveWorkflowStage
from .model_registry import get_model, update_status

# Default chunk size: 1 MB
DEFAULT_CHUNK_BYTES = 1_048_576


class ProvisionerError(Exception):
    """Raised for provisioner-level failures (not registry failures)."""


@dataclass
class DownloadHandle:
    """In-memory state for an active or completed download."""

    prim_path: str
    filename: str
    dest_path: Path
    total_bytes: int = 0
    bytes_downloaded: int = 0
    sha256_expected: str = ""
    sha256_actual: str = ""
    status: str = "pending"          # pending | downloading | done | failed | verified
    error: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def progress_pct(self) -> float:
        """Return download progress as 0–100.0, or 0 if total unknown."""
        if self.total_bytes <= 0:
            return 0.0
        return min(100.0, self.bytes_downloaded * 100.0 / self.total_bytes)


class Provisioner:
    """Download engine wired to model_registry status lifecycle.

    Args:
        cws: CognitiveWorkflowStage — registry lives here.
        models_dir: Root of the ComfyUI models directory (e.g., G:/COMFYUI_Database/models).
        chunk_bytes: Streaming chunk size in bytes (default 1 MB).
        progress_cb: Optional callback(prim_path, bytes_downloaded, total_bytes).
    """

    def __init__(
        self,
        cws: CognitiveWorkflowStage,
        models_dir: Path,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        progress_cb: Callable[[str, int, int], None] | None = None,
    ) -> None:
        if not HAS_HTTPX:
            raise ProvisionerError(
                "httpx is required for downloads. Install it: pip install httpx"
            )
        self._cws = cws
        self._models_dir = Path(models_dir)
        self._chunk_bytes = max(4096, chunk_bytes)
        self._progress_cb = progress_cb
        self._handles: dict[str, DownloadHandle] = {}
        self._global_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, prim_path: str) -> DownloadHandle:
        """Download a registered model by its USD prim path (blocking).

        Resumes a partial download if the destination file already exists.

        Args:
            prim_path: USD prim path like /models/loras/my_lora

        Returns:
            DownloadHandle with final state.

        Raises:
            ProvisionerError: If model not registered, no URL, or download fails.
        """
        model = get_model(self._cws, prim_path)
        if model is None:
            raise ProvisionerError(f"Model not registered: {prim_path}")

        source_url = model.get("source_url", "")
        if not source_url:
            raise ProvisionerError(f"No source_url for {prim_path}")

        filename = model.get("filename", "")
        model_type = model.get("model_type", "")
        sha256_expected = model.get("sha256", "")

        dest_dir = self._models_dir / model_type
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename

        handle = DownloadHandle(
            prim_path=prim_path,
            filename=filename,
            dest_path=dest_path,
            sha256_expected=sha256_expected,
        )

        with self._global_lock:
            self._handles[prim_path] = handle

        # Mark registry: downloading
        update_status(self._cws, prim_path, "downloading")
        handle.status = "downloading"

        try:
            self._stream_download(handle, source_url)
        except Exception as exc:
            handle.status = "failed"
            handle.error = str(exc)
            update_status(self._cws, prim_path, "failed", error_message=str(exc)[:256])
            raise ProvisionerError(f"Download failed for {prim_path}: {exc}") from exc

        handle.status = "done"
        update_status(
            self._cws,
            prim_path,
            "materialized",
            file_path=str(dest_path),
            size_bytes=handle.bytes_downloaded,
        )
        return handle

    def verify(self, prim_path: str) -> bool:
        """SHA256-verify an already-downloaded file against the registry.

        Updates registry status to 'failed' if hash mismatches.

        Args:
            prim_path: USD prim path of the model.

        Returns:
            True if hash matches (or no expected hash registered), False otherwise.

        Raises:
            ProvisionerError: If model not registered or file not found on disk.
        """
        model = get_model(self._cws, prim_path)
        if model is None:
            raise ProvisionerError(f"Model not registered: {prim_path}")

        file_path_str = model.get("file_path", "")
        if not file_path_str:
            # Fall back to deriving path from type + filename
            filename = model.get("filename", "")
            model_type = model.get("model_type", "")
            if not filename or not model_type:
                raise ProvisionerError(f"Cannot determine file path for {prim_path}")
            file_path_str = str(self._models_dir / model_type / filename)

        file_path = Path(file_path_str)
        if not file_path.exists():
            raise ProvisionerError(f"File not found on disk: {file_path}")

        sha256_expected = model.get("sha256", "")
        if not sha256_expected:
            # No hash to verify against — treat as pass
            return True

        sha256_actual = _hash_file(file_path)

        if sha256_actual.lower() == sha256_expected.lower():
            return True
        else:
            update_status(
                self._cws,
                prim_path,
                "failed",
                error_message=f"SHA256 mismatch: expected {sha256_expected[:12]}…, got {sha256_actual[:12]}…",
            )
            return False

    def status(self, prim_path: str) -> dict:
        """Return combined handle + registry state for a model.

        Args:
            prim_path: USD prim path of the model.

        Returns:
            Dict with keys: prim_path, registry_status, bytes_downloaded,
            total_bytes, progress_pct, dest_path, error, sha256_match.
        """
        model = get_model(self._cws, prim_path)
        registry_status = model.get("status", "unknown") if model else "not_registered"

        with self._global_lock:
            handle = self._handles.get(prim_path)

        if handle is None:
            return {
                "prim_path": prim_path,
                "registry_status": registry_status,
                "handle_status": "no_active_handle",
                "bytes_downloaded": 0,
                "total_bytes": 0,
                "progress_pct": 0.0,
                "dest_path": "",
                "error": "",
                "sha256_match": None,
            }

        sha256_match: bool | None = None
        if handle.sha256_actual and handle.sha256_expected:
            sha256_match = handle.sha256_actual.lower() == handle.sha256_expected.lower()

        return {
            "prim_path": prim_path,
            "registry_status": registry_status,
            "handle_status": handle.status,
            "bytes_downloaded": handle.bytes_downloaded,
            "total_bytes": handle.total_bytes,
            "progress_pct": handle.progress_pct(),
            "dest_path": str(handle.dest_path),
            "error": handle.error,
            "sha256_match": sha256_match,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _stream_download(self, handle: DownloadHandle, url: str) -> None:
        """Stream-download URL to handle.dest_path with resume support.

        Uses Range header if a partial file already exists.
        Verifies SHA256 on completion if expected hash is set.
        """
        dest = handle.dest_path
        resume_offset = 0

        if dest.exists():
            resume_offset = dest.stat().st_size
            handle.bytes_downloaded = resume_offset

        headers = {}
        if resume_offset > 0:
            headers["Range"] = f"bytes={resume_offset}-"

        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()

                content_length = int(response.headers.get("content-length", 0))
                if response.status_code == 206:  # Partial Content
                    handle.total_bytes = resume_offset + content_length
                elif response.status_code == 200:
                    handle.total_bytes = content_length
                    resume_offset = 0  # Server ignored Range header

                hasher = hashlib.sha256()

                # If resuming, hash the already-downloaded portion first
                if resume_offset > 0 and handle.sha256_expected:
                    hasher = _hash_file_partial(dest, resume_offset)

                mode = "ab" if resume_offset > 0 else "wb"
                with open(dest, mode) as fh:
                    for chunk in response.iter_bytes(chunk_size=self._chunk_bytes):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        if handle.sha256_expected:
                            hasher.update(chunk)
                        with handle._lock:
                            handle.bytes_downloaded += len(chunk)

                        if self._progress_cb:
                            try:
                                self._progress_cb(
                                    handle.prim_path,
                                    handle.bytes_downloaded,
                                    handle.total_bytes,
                                )
                            except Exception:
                                pass  # Progress callback errors must not abort download

        if handle.sha256_expected:
            handle.sha256_actual = hasher.hexdigest()
            if handle.sha256_actual.lower() != handle.sha256_expected.lower():
                raise ProvisionerError(
                    f"SHA256 mismatch: expected {handle.sha256_expected[:12]}…, "
                    f"got {handle.sha256_actual[:12]}…"
                )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _hash_file(path: Path, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    """Compute SHA256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_file_partial(path: Path, size: int, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> "hashlib._Hash":
    """Return a partially-fed SHA256 hasher seeded with the first `size` bytes of path."""
    hasher = hashlib.sha256()
    remaining = size
    with open(path, "rb") as fh:
        while remaining > 0:
            read_size = min(chunk_bytes, remaining)
            chunk = fh.read(read_size)
            if not chunk:
                break
            hasher.update(chunk)
            remaining -= len(chunk)
    return hasher
