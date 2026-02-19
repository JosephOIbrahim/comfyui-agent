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
