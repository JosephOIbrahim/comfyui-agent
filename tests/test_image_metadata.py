"""Tests for the image_metadata module (write/read/reconstruct PNG metadata)."""

import json
import time

import pytest

from agent.tools.image_metadata import (
    TOOLS,
    _validate_metadata,
    handle,
)

# We need PIL for these tests
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402
from PIL.PngImagePlugin import PngInfo  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def png_image(tmp_path, monkeypatch):
    """Create a minimal PNG image in a safe temp dir."""
    monkeypatch.setattr(
        "agent.tools._util._SAFE_DIRS",
        [tmp_path.resolve()],
    )
    img_path = tmp_path / "test_output.png"
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(str(img_path))
    return str(img_path)


@pytest.fixture
def png_with_comfyui_chunks(tmp_path, monkeypatch):
    """Create a PNG that already has ComfyUI native chunks."""
    monkeypatch.setattr(
        "agent.tools._util._SAFE_DIRS",
        [tmp_path.resolve()],
    )
    img_path = tmp_path / "comfyui_output.png"
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    info = PngInfo()
    info.add_text("prompt", json.dumps({"1": {"class_type": "KSampler"}}))
    info.add_text("workflow", json.dumps({"nodes": [], "links": []}))
    img.save(str(img_path), pnginfo=info)
    return str(img_path)


@pytest.fixture
def sample_metadata():
    """Valid v1 metadata."""
    return {
        "schema_version": 1,
        "timestamp": time.time(),
        "intent": {
            "user_request": "Make it dreamier",
            "interpretation": "Lower CFG to 5, switch sampler to DPM++ 2M Karras",
            "style_references": ["Studio Ghibli", "soft pastel"],
            "session_context": "Iteration 2 of anime portrait series",
        },
        "iterations": [
            {
                "iteration": 1,
                "type": "initial",
                "trigger": "first generation",
                "patches": [],
                "params": {"cfg": 7.0, "steps": 20},
                "feedback": "",
                "observation": "Good composition, too sharp",
            },
            {
                "iteration": 2,
                "type": "refinement",
                "trigger": "user asked for dreamier look",
                "patches": [{"op": "replace", "path": "/2/inputs/cfg", "value": 5.0}],
                "params": {"cfg": 5.0, "steps": 20},
                "feedback": "Much better!",
                "observation": "Softer output, better mood",
            },
        ],
        "accepted_iteration": 2,
        "session": {
            "session_name": "anime_portraits",
            "workflow_path": "G:/workflows/anime.json",
            "workflow_hash": "abc123def456",
            "key_params": {"model": "dreamshaper_8.safetensors", "cfg": 5.0},
            "model_combo": ["dreamshaper_8.safetensors"],
        },
    }


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_valid_metadata(self, sample_metadata):
        assert _validate_metadata(sample_metadata) is None

    def test_missing_schema_version(self):
        err = _validate_metadata({"timestamp": 1.0})
        assert err is not None
        assert "schema_version" in err

    def test_wrong_schema_version(self):
        err = _validate_metadata({"schema_version": 2, "timestamp": 1.0})
        assert err is not None
        assert "schema_version" in err

    def test_missing_timestamp(self):
        err = _validate_metadata({"schema_version": 1})
        assert err is not None
        assert "timestamp" in err

    def test_not_a_dict(self):
        err = _validate_metadata("not a dict")
        assert err is not None
        assert "dict" in err

    def test_minimal_valid(self):
        """Just schema_version + timestamp should pass."""
        assert _validate_metadata({"schema_version": 1, "timestamp": 1.0}) is None


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

class TestToolSchemas:
    def test_tool_count(self):
        assert len(TOOLS) == 3

    def test_tool_names(self):
        names = {t["name"] for t in TOOLS}
        assert names == {"write_image_metadata", "read_image_metadata", "reconstruct_context"}


# ---------------------------------------------------------------------------
# Write + Read Round-Trip
# ---------------------------------------------------------------------------

class TestWriteRead:
    def test_round_trip(self, png_image, sample_metadata):
        """Write metadata, read it back — should be identical."""
        # Write
        result = json.loads(handle("write_image_metadata", {
            "image_path": png_image,
            "metadata": sample_metadata,
        }))
        assert result["status"] == "ok"

        # Read
        result = json.loads(handle("read_image_metadata", {
            "image_path": png_image,
        }))
        assert result["has_creative_metadata"] is True
        assert result["metadata"]["schema_version"] == 1
        assert result["metadata"]["intent"]["user_request"] == "Make it dreamier"
        assert len(result["metadata"]["iterations"]) == 2

    def test_minimal_metadata(self, png_image):
        """Minimal metadata (just version + timestamp) should work."""
        meta = {"schema_version": 1, "timestamp": time.time()}
        result = json.loads(handle("write_image_metadata", {
            "image_path": png_image,
            "metadata": meta,
        }))
        assert result["status"] == "ok"

        result = json.loads(handle("read_image_metadata", {
            "image_path": png_image,
        }))
        assert result["has_creative_metadata"] is True
        assert result["metadata"]["schema_version"] == 1

    def test_read_no_metadata(self, png_image):
        """Reading a PNG with no creative metadata should return empty."""
        result = json.loads(handle("read_image_metadata", {
            "image_path": png_image,
        }))
        assert result["has_creative_metadata"] is False
        assert result["metadata"] is None

    def test_overwrite_metadata(self, png_image):
        """Writing metadata twice should overwrite, not duplicate."""
        meta1 = {"schema_version": 1, "timestamp": 1.0, "intent": {"user_request": "first"}}
        meta2 = {"schema_version": 1, "timestamp": 2.0, "intent": {"user_request": "second"}}

        handle("write_image_metadata", {"image_path": png_image, "metadata": meta1})
        handle("write_image_metadata", {"image_path": png_image, "metadata": meta2})

        result = json.loads(handle("read_image_metadata", {"image_path": png_image}))
        assert result["metadata"]["intent"]["user_request"] == "second"
        assert result["metadata"]["timestamp"] == 2.0


# ---------------------------------------------------------------------------
# Native Chunk Preservation
# ---------------------------------------------------------------------------

class TestNativeChunkPreservation:
    def test_preserves_comfyui_prompt(self, png_with_comfyui_chunks, sample_metadata):
        """Writing our metadata must NOT overwrite ComfyUI's prompt chunk."""
        handle("write_image_metadata", {
            "image_path": png_with_comfyui_chunks,
            "metadata": sample_metadata,
        })

        # Read back and verify both exist
        result = json.loads(handle("read_image_metadata", {
            "image_path": png_with_comfyui_chunks,
        }))
        assert result["has_creative_metadata"] is True
        assert result["has_comfyui_native"] is True
        assert "prompt" in result["native_chunk_keys"]
        assert "workflow" in result["native_chunk_keys"]

    def test_native_chunks_intact(self, png_with_comfyui_chunks, sample_metadata):
        """Verify the actual content of native chunks is preserved."""
        handle("write_image_metadata", {
            "image_path": png_with_comfyui_chunks,
            "metadata": sample_metadata,
        })

        # Open directly with PIL to verify
        img = Image.open(png_with_comfyui_chunks)
        prompt_data = json.loads(img.text["prompt"])
        assert "1" in prompt_data
        assert prompt_data["1"]["class_type"] == "KSampler"
        workflow_data = json.loads(img.text["workflow"])
        assert "nodes" in workflow_data
        img.close()


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_invalid_schema_version(self, png_image):
        result = json.loads(handle("write_image_metadata", {
            "image_path": png_image,
            "metadata": {"schema_version": 99, "timestamp": 1.0},
        }))
        assert "error" in result

    def test_missing_timestamp(self, png_image):
        result = json.loads(handle("write_image_metadata", {
            "image_path": png_image,
            "metadata": {"schema_version": 1},
        }))
        assert "error" in result

    def test_nonexistent_file_read(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.tools._util._SAFE_DIRS", [tmp_path.resolve()])
        result = json.loads(handle("read_image_metadata", {
            "image_path": str(tmp_path / "does_not_exist.png"),
        }))
        assert "error" in result

    def test_non_png_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.tools._util._SAFE_DIRS", [tmp_path.resolve()])
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a png", encoding="utf-8")
        result = json.loads(handle("write_image_metadata", {
            "image_path": str(txt_file),
            "metadata": {"schema_version": 1, "timestamp": 1.0},
        }))
        assert "error" in result
        assert "PNG" in result["error"]

    def test_unknown_tool(self):
        result = json.loads(handle("nonexistent_tool", {}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Reconstruct Context
# ---------------------------------------------------------------------------

class TestReconstructContext:
    def test_full_reconstruct(self, png_image, sample_metadata):
        """Write full metadata, reconstruct — should get structured context."""
        handle("write_image_metadata", {
            "image_path": png_image,
            "metadata": sample_metadata,
        })

        result = json.loads(handle("reconstruct_context", {
            "image_path": png_image,
        }))
        assert result["has_context"] is True
        assert "dreamier" in result["summary"].lower() or "dreami" in result["summary"].lower()

        ctx = result["context"]
        assert ctx["schema_version"] == 1
        assert ctx["intent"]["what_artist_wanted"] == "Make it dreamier"
        assert ctx["iteration_history"]["total_iterations"] == 2
        assert ctx["iteration_history"]["accepted_iteration"] == 2
        assert ctx["session"]["models_used"] == ["dreamshaper_8.safetensors"]

    def test_reconstruct_no_metadata(self, png_image):
        """Reconstruct on an image with no metadata should gracefully degrade."""
        result = json.loads(handle("reconstruct_context", {
            "image_path": png_image,
        }))
        assert result["has_context"] is False
        assert result["context"] is None
        assert "No creative metadata" in result["summary"]

    def test_reconstruct_minimal_metadata(self, png_image):
        """Minimal metadata should still reconstruct without errors."""
        meta = {"schema_version": 1, "timestamp": time.time()}
        handle("write_image_metadata", {
            "image_path": png_image,
            "metadata": meta,
        })
        result = json.loads(handle("reconstruct_context", {
            "image_path": png_image,
        }))
        assert result["has_context"] is True
        assert result["context"]["schema_version"] == 1

    def test_reconstruct_intent_only(self, png_image):
        """Metadata with only intent (no iterations/session) should work."""
        meta = {
            "schema_version": 1,
            "timestamp": time.time(),
            "intent": {
                "user_request": "dramatic lighting",
                "interpretation": "Add rim light, increase contrast",
            },
        }
        handle("write_image_metadata", {
            "image_path": png_image,
            "metadata": meta,
        })
        result = json.loads(handle("reconstruct_context", {
            "image_path": png_image,
        }))
        ctx = result["context"]
        assert ctx["intent"]["what_artist_wanted"] == "dramatic lighting"
        assert "iteration_history" not in ctx
        assert "session" not in ctx


# ---------------------------------------------------------------------------
# Cycle 33: atomic write verification
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    """PNG metadata write must be atomic (temp file + rename)."""

    def test_write_leaves_no_temp_files_on_success(self, tmp_path):
        """A successful write must clean up temp files."""
        from PIL import Image
        from agent.tools.image_metadata import handle

        # Create a real PNG
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        img.save(str(img_path))
        img.close()

        handle("write_image_metadata", {
            "image_path": str(img_path),
            "metadata": {"schema_version": 1, "timestamp": 1.0, "session": {}},
        })

        # After write, no .png temp files should remain in the directory
        leftover = [f for f in tmp_path.iterdir() if f != img_path]
        assert leftover == [], f"Temp files left behind: {leftover}"

    def test_write_preserves_original_on_success(self, tmp_path):
        """Original file path must still exist and be readable after atomic write."""
        from PIL import Image
        from agent.tools.image_metadata import handle

        img_path = tmp_path / "original.png"
        img = Image.new("RGB", (8, 8), color=(0, 255, 0))
        img.save(str(img_path))
        img.close()

        handle("write_image_metadata", {
            "image_path": str(img_path),
            "metadata": {"schema_version": 1, "timestamp": 9999.0, "session": {"test": True}},
        })

        # File must still exist at the same path
        assert img_path.exists()
        # Must be a valid PNG (PIL can open it)
        reopened = Image.open(str(img_path))
        reopened.close()


# ---------------------------------------------------------------------------
# Cycle 38: fsync before atomic rename — durability guard
# ---------------------------------------------------------------------------

class TestImageMetadataFsync:
    """_write_png_metadata must fsync the temp file before os.replace.

    We can't test actual durability against power failure, but we CAN verify:
    1. os.fsync is called on the temp file's fd during a successful write.
    2. An OSError from fsync is swallowed (non-fatal) and the write still completes.
    3. The output file is a valid PNG regardless.
    """

    def _make_png(self, path):
        from PIL import Image as _Image
        img = _Image.new("RGB", (4, 4), color=(10, 20, 30))
        img.save(str(path))
        img.close()

    def test_fsync_called_on_temp_file(self, tmp_path):
        """os.fsync must be invoked once during _write_png_metadata."""
        pytest.importorskip("PIL")
        import os as _os
        from unittest.mock import patch
        from agent.tools.image_metadata import _write_png_metadata

        img_path = tmp_path / "test_fsync.png"
        self._make_png(img_path)

        fsync_calls = []
        original_fsync = _os.fsync

        def _recording_fsync(fd):
            fsync_calls.append(fd)
            return original_fsync(fd)

        with patch("os.fsync", side_effect=_recording_fsync):
            _write_png_metadata(str(img_path), {"schema_version": 1, "timestamp": 1.0})

        assert len(fsync_calls) >= 1, "os.fsync was not called during metadata write"

    def test_fsync_oserror_does_not_abort_write(self, tmp_path):
        """If os.fsync raises OSError, the write must still complete successfully."""
        pytest.importorskip("PIL")
        from unittest.mock import patch
        from agent.tools.image_metadata import _write_png_metadata

        img_path = tmp_path / "test_fsync_fail.png"
        self._make_png(img_path)

        with patch("os.fsync", side_effect=OSError("fsync not supported")):
            # Must NOT raise — fsync failure is non-fatal
            _write_png_metadata(str(img_path), {"schema_version": 1, "timestamp": 2.0})

        # Output must still be a valid PNG
        from PIL import Image as _Image
        reopened = _Image.open(str(img_path))
        reopened.close()

    def test_output_is_valid_png_after_write(self, tmp_path):
        """After _write_png_metadata the file at the original path is a valid PNG."""
        pytest.importorskip("PIL")
        from agent.tools.image_metadata import _write_png_metadata

        img_path = tmp_path / "test_valid.png"
        self._make_png(img_path)
        _write_png_metadata(str(img_path), {"schema_version": 1, "timestamp": 3.0})

        from PIL import Image as _Image
        with _Image.open(str(img_path)) as img:
            assert img.format == "PNG"


# ---------------------------------------------------------------------------
# Cycle 47 — image_metadata handler required field guards
# ---------------------------------------------------------------------------

class TestWriteMetadataRequiredFields:
    """write_image_metadata must return structured error when required fields are missing."""

    def test_missing_image_path_returns_error(self):
        from agent.tools import image_metadata
        result = json.loads(image_metadata.handle("write_image_metadata", {
            "metadata": {"schema_version": 1},
        }))
        assert "error" in result
        assert "image_path" in result["error"].lower()

    def test_missing_metadata_returns_error(self):
        from agent.tools import image_metadata
        result = json.loads(image_metadata.handle("write_image_metadata", {
            "image_path": "/some/image.png",
        }))
        assert "error" in result
        assert "metadata" in result["error"].lower()

    def test_empty_image_path_returns_error(self):
        from agent.tools import image_metadata
        result = json.loads(image_metadata.handle("write_image_metadata", {
            "image_path": "", "metadata": {"schema_version": 1},
        }))
        assert "error" in result

    def test_none_image_path_returns_error(self):
        from agent.tools import image_metadata
        result = json.loads(image_metadata.handle("write_image_metadata", {
            "image_path": None, "metadata": {"schema_version": 1},
        }))
        assert "error" in result


class TestReadMetadataRequiredField:
    """read_image_metadata must return structured error when image_path is missing."""

    def test_missing_image_path_returns_error(self):
        from agent.tools import image_metadata
        result = json.loads(image_metadata.handle("read_image_metadata", {}))
        assert "error" in result
        assert "image_path" in result["error"].lower()

    def test_empty_image_path_returns_error(self):
        from agent.tools import image_metadata
        result = json.loads(image_metadata.handle("read_image_metadata", {"image_path": ""}))
        assert "error" in result


class TestReconstructContextRequiredField:
    """reconstruct_context must return structured error when image_path is missing."""

    def test_missing_image_path_returns_error(self):
        from agent.tools import image_metadata
        result = json.loads(image_metadata.handle("reconstruct_context", {}))
        assert "error" in result
        assert "image_path" in result["error"].lower()

    def test_none_image_path_returns_error(self):
        from agent.tools import image_metadata
        result = json.loads(image_metadata.handle("reconstruct_context", {"image_path": None}))
        assert "error" in result
