"""Tests for brain/vision.py — image analysis via LLM Vision."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.brain import handle
from agent.llm import LLMResponse, TextBlock


def _mock_vision_response(text: str) -> LLMResponse:
    """Create a mock LLMResponse with a single text block."""
    return LLMResponse(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        model="test-model",
    )


def _patch_provider(response_text: str):
    """Patch get_provider to return a mock provider with the given response."""
    mock_provider = MagicMock()
    mock_provider.create.return_value = _mock_vision_response(response_text)
    return patch("agent.brain.vision.get_provider", return_value=mock_provider)


class TestAnalyzeImage:
    def test_file_not_found(self):
        result = json.loads(handle("analyze_image", {
            "image_path": "/nonexistent/image.png",
        }))
        assert "error" in result

    def test_analyze_calls_vision_api(self, fake_image):
        response_json = json.dumps({
            "quality_score": 0.85,
            "artifacts": [],
            "composition": "Good composition",
            "prompt_adherence": 0.9,
            "strengths": ["sharp details"],
            "suggestions": ["try higher CFG"],
        })

        with _patch_provider(response_json) as mock_get:
            result = json.loads(handle("analyze_image", {
                "image_path": fake_image,
                "prompt_used": "a red pixel",
            }))

        assert result["quality_score"] == 0.85
        assert result["image_path"] == fake_image
        # Verify the provider was called
        mock_get.return_value.create.assert_called_once()

    def test_handles_non_json_response(self, fake_image):
        with _patch_provider("This is not JSON"):
            result = json.loads(handle("analyze_image", {
                "image_path": fake_image,
            }))

        assert "raw_analysis" in result
        assert "parse_error" in result


class TestCompareOutputs:
    def test_both_files_missing(self):
        result = json.loads(handle("compare_outputs", {
            "image_a": "/fake/a.png",
            "image_b": "/fake/b.png",
        }))
        assert "error" in result

    def test_compare_calls_api(self, fake_image):
        response_json = json.dumps({
            "improved": True,
            "differences": ["sharper details"],
            "quality_delta": 0.15,
            "recommendation": "The change improved quality.",
        })

        with _patch_provider(response_json):
            result = json.loads(handle("compare_outputs", {
                "image_a": fake_image,
                "image_b": fake_image,
                "change_description": "Increased steps from 20 to 30",
            }))

        assert result["improved"] is True
        assert result["quality_delta"] == 0.15


class TestSuggestImprovements:
    def test_suggest_calls_api(self, fake_image):
        response_json = json.dumps({
            "suggestions": [
                {
                    "parameter": "steps",
                    "current_value": "20",
                    "suggested_value": "30",
                    "reason": "More detail",
                    "expected_impact": "Sharper output",
                    "confidence": 0.8,
                },
            ],
            "priority_order": ["steps"],
        })

        with _patch_provider(response_json):
            result = json.loads(handle("suggest_improvements", {
                "image_path": fake_image,
                "workflow_summary": "SDXL, 20 steps, Euler, CFG 7.0",
                "goal": "more detail",
            }))

        assert len(result["suggestions"]) == 1
        assert result["suggestions"][0]["parameter"] == "steps"


class TestHashCompare:
    def test_no_pillow(self, fake_image):
        """Test graceful fallback when Pillow not available."""
        from agent.brain import vision
        with patch.object(vision, "_HAS_PIL", False):
            result = json.loads(handle("hash_compare_images", {
                "image_a": fake_image,
                "image_b": fake_image,
            }))
            assert "error" in result
            assert "Pillow" in result["error"]

    def test_file_not_found(self):
        result = json.loads(handle("hash_compare_images", {
            "image_a": "/fake/a.png",
            "image_b": "/fake/b.png",
        }))
        assert "error" in result

    @pytest.mark.skipif(not __import__("agent.brain.vision", fromlist=["_HAS_PIL"])._HAS_PIL, reason="Pillow not installed")
    def test_identical_images(self, tmp_path):
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (64, 64), (128, 128, 128))
        path = tmp_path / "identical.png"
        img.save(path)
        result = json.loads(handle("hash_compare_images", {
            "image_a": str(path),
            "image_b": str(path),
        }))
        assert result["verdict"] == "identical"
        assert result["hash_similarity"] == 1.0
        assert result["pixel_diff_pct"] == 0.0

    @pytest.mark.skipif(not __import__("agent.brain.vision", fromlist=["_HAS_PIL"])._HAS_PIL, reason="Pillow not installed")
    def test_different_images(self, tmp_path):
        """Create two visually different images and compare."""
        from PIL import Image as PILImage

        img_a = PILImage.new("RGB", (64, 64), (255, 0, 0))  # Red
        img_b = PILImage.new("RGB", (64, 64), (0, 0, 255))  # Blue
        path_a = tmp_path / "red.png"
        path_b = tmp_path / "blue.png"
        img_a.save(path_a)
        img_b.save(path_b)

        result = json.loads(handle("hash_compare_images", {
            "image_a": str(path_a),
            "image_b": str(path_b),
        }))
        assert result["verdict"] in ("different", "very_different")
        assert result["pixel_diff_pct"] > 50.0

    @pytest.mark.skipif(not __import__("agent.brain.vision", fromlist=["_HAS_PIL"])._HAS_PIL, reason="Pillow not installed")
    def test_similar_images(self, tmp_path):
        """Create two slightly different images."""
        from PIL import Image as PILImage

        img_a = PILImage.new("RGB", (64, 64), (100, 100, 100))
        img_b = PILImage.new("RGB", (64, 64), (105, 100, 100))  # Slight red shift
        path_a = tmp_path / "a.png"
        path_b = tmp_path / "b.png"
        img_a.save(path_a)
        img_b.save(path_b)

        result = json.loads(handle("hash_compare_images", {
            "image_a": str(path_a),
            "image_b": str(path_b),
        }))
        assert result["hash_similarity"] >= 0.8
        assert result["resolution_a"] == "64x64"

    def test_hamming_distance(self):
        from agent.brain.vision import _hamming_distance
        assert _hamming_distance(0, 0) == 0
        assert _hamming_distance(0b1111, 0b0000) == 4
        assert _hamming_distance(0b1010, 0b0101) == 4


class TestBrainMessageActivation:
    """Verify that vision module emits BrainMessages for inter-module communication."""

    def test_analyze_image_emits_brain_message(self, fake_image):
        """analyze_image should create a brain_message with analysis results."""
        response_json = json.dumps({
            "quality_score": 0.85,
            "artifacts": [],
            "composition": "Good",
            "prompt_adherence": 0.9,
            "strengths": [],
            "suggestions": [],
        })

        with _patch_provider(response_json), \
             patch("agent.brain.vision.brain_message") as mock_brain_msg:
            handle("analyze_image", {
                "image_path": fake_image,
                "prompt_used": "test prompt",
            })

        # brain_message should have been called
        mock_brain_msg.assert_called_once()
        call_kwargs = mock_brain_msg.call_args
        assert call_kwargs[1]["source"] == "vision"
        assert call_kwargs[1]["target"] == "memory"
        assert call_kwargs[1]["msg_type"] == "result"
        assert call_kwargs[1]["payload"]["action"] == "image_analyzed"

    def test_compare_outputs_emits_brain_message(self, fake_image):
        """compare_outputs should create a brain_message with comparison results."""
        response_json = json.dumps({
            "improved": True,
            "differences": ["better details"],
            "quality_delta": 0.1,
            "recommendation": "Keep the change.",
        })

        with _patch_provider(response_json), \
             patch("agent.brain.vision.brain_message") as mock_brain_msg:
            handle("compare_outputs", {
                "image_a": fake_image,
                "image_b": fake_image,
            })

        mock_brain_msg.assert_called_once()
        call_kwargs = mock_brain_msg.call_args
        assert call_kwargs[1]["payload"]["action"] == "images_compared"


class TestDispatchBrainMessage:
    """Verify that dispatch_brain_message routes to memory and never raises."""

    def test_dispatch_calls_record_outcome(self, fake_image):
        """dispatch_brain_message routes vision->memory to record_outcome."""
        from agent.brain._protocol import brain_message, dispatch_brain_message

        msg = brain_message(
            source="vision",
            target="memory",
            msg_type="result",
            payload={
                "action": "image_analyzed",
                "quality_score": 0.85,
            },
        )

        with patch("agent.tools.handle") as mock_dispatch:
            mock_dispatch.return_value = "{}"
            dispatch_brain_message(msg)

        mock_dispatch.assert_called_once()
        call_args = mock_dispatch.call_args
        assert call_args[0][0] == "record_outcome"
        assert call_args[0][1]["action"] == "image_analyzed"

    def test_dispatch_never_raises(self):
        """dispatch_brain_message catches all exceptions."""
        from agent.brain._protocol import brain_message, dispatch_brain_message

        msg = brain_message(
            source="vision",
            target="memory",
            msg_type="result",
            payload={"action": "test"},
        )

        with patch("agent.tools.handle", side_effect=RuntimeError("boom")):
            # Should not raise
            dispatch_brain_message(msg)


# ---------------------------------------------------------------------------
# Cycle 33: image size limit before base64 encoding
# ---------------------------------------------------------------------------

class TestImageSizeLimit:
    """_read_image_as_base64 must refuse files larger than 50 MB."""

    def test_large_image_raises_value_error(self, tmp_path):
        """Image > 50 MB must raise ValueError before base64 encoding."""
        from agent.brain.vision import VisionAgent
        from agent.brain._sdk import BrainConfig

        big_img = tmp_path / "huge.png"
        # Write a file larger than 50 MB (just header bytes, not a real PNG — size check only)
        big_img.write_bytes(b"\x89PNG" + b"\x00" * (51 * 1024 * 1024))

        cfg = BrainConfig(validate_path=lambda *a, **kw: None)
        agent = VisionAgent(cfg)

        with pytest.raises(ValueError, match="too large"):
            agent._read_image_as_base64(str(big_img))

    def test_small_image_not_rejected(self, tmp_path):
        """Image under the size limit must not be rejected (path validation aside)."""
        from agent.brain.vision import VisionAgent
        from agent.brain._sdk import BrainConfig

        # Write a tiny "image" under 1 KB
        small_img = tmp_path / "tiny.png"
        small_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        cfg = BrainConfig(validate_path=lambda *a, **kw: None)
        agent = VisionAgent(cfg)

        # Should not raise ValueError for size — may raise other errors (not PIL-related)
        try:
            agent._read_image_as_base64(str(small_img))
        except ValueError as e:
            assert "too large" not in str(e), "Small image should not fail size check"
        except Exception:
            pass  # Other exceptions (PIL decode, etc.) are acceptable here


# ---------------------------------------------------------------------------
# Cycle 34: TOCTOU — file disappears between stat() and read_bytes()
# ---------------------------------------------------------------------------

class TestTOCTOUFileDisappears:
    """_read_image_as_base64 must raise FileNotFoundError (not OSError or crash)
    if the file is removed between the size-check stat() and the read_bytes()."""

    def test_file_removed_after_stat_raises_file_not_found(self, tmp_path):
        """Simulate TOCTOU: inject OSError in read_bytes() call."""
        from agent.brain.vision import VisionAgent
        from agent.brain._sdk import BrainConfig
        from pathlib import Path
        from unittest.mock import patch

        img = tmp_path / "victim.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        cfg = BrainConfig(validate_path=lambda *a, **kw: None)
        agent = VisionAgent(cfg)

        # Patch Path.read_bytes to raise FileNotFoundError after stat() succeeds
        original_stat = Path.stat

        def stat_then_die(self_path, *args, **kwargs):
            return original_stat(self_path, *args, **kwargs)

        with patch.object(Path, "read_bytes", side_effect=FileNotFoundError("gone")):
            with pytest.raises(FileNotFoundError, match="unavailable"):
                agent._read_image_as_base64(str(img))

    def test_oserror_on_read_raises_file_not_found(self, tmp_path):
        """OSError during read_bytes() is translated to FileNotFoundError."""
        from agent.brain.vision import VisionAgent
        from agent.brain._sdk import BrainConfig
        from pathlib import Path
        from unittest.mock import patch

        img = tmp_path / "victim2.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        cfg = BrainConfig(validate_path=lambda *a, **kw: None)
        agent = VisionAgent(cfg)

        with patch.object(Path, "read_bytes", side_effect=OSError("disk error")):
            with pytest.raises(FileNotFoundError, match="unavailable"):
                agent._read_image_as_base64(str(img))


# ---------------------------------------------------------------------------
# Cycle 41: hash compute exception → JSON error not crash
# ---------------------------------------------------------------------------

class TestHashComputeException:
    """Cycle 41: exceptions in hash/pixel computation must return JSON, not crash."""

    @pytest.mark.skipif(
        not __import__("agent.brain.vision", fromlist=["_HAS_PIL"])._HAS_PIL,
        reason="Pillow not installed"
    )
    def test_compute_hash_exception_returns_json_error(self, tmp_path):
        """If _compute_average_hash raises, handle() must return error JSON."""
        from PIL import Image as PILImage
        from unittest.mock import patch
        from agent.brain import vision as _vis_mod

        img = PILImage.new("RGB", (64, 64), (100, 150, 200))
        path = tmp_path / "test.png"
        img.save(path)

        with patch.object(_vis_mod, "_compute_average_hash", side_effect=RuntimeError("hash failed")):
            result = json.loads(handle("hash_compare_images", {
                "image_a": str(path),
                "image_b": str(path),
            }))
        assert "error" in result
        assert "hash" in result["error"].lower() or "failed" in result["error"].lower()

    @pytest.mark.skipif(
        not __import__("agent.brain.vision", fromlist=["_HAS_PIL"])._HAS_PIL,
        reason="Pillow not installed"
    )
    def test_pixel_diff_exception_returns_json_error(self, tmp_path):
        """If _pixel_diff raises, handle() must return error JSON, not crash."""
        from PIL import Image as PILImage
        from unittest.mock import patch
        from agent.brain import vision as _vis_mod

        img = PILImage.new("RGB", (64, 64), (100, 150, 200))
        path = tmp_path / "test2.png"
        img.save(path)

        with patch.object(_vis_mod, "_pixel_diff", side_effect=ValueError("diff failed")):
            result = json.loads(handle("hash_compare_images", {
                "image_a": str(path),
                "image_b": str(path),
            }))
        assert "error" in result


# ---------------------------------------------------------------------------
# Cycle 46 — Vision handler required field guards
# ---------------------------------------------------------------------------

class TestAnalyzeImageRequiredField:
    """analyze_image must return structured error when image_path is missing or invalid."""

    def test_missing_image_path_returns_error(self):
        result = json.loads(handle("analyze_image", {}))
        assert "error" in result
        assert "image_path" in result["error"].lower()

    def test_empty_image_path_returns_error(self):
        result = json.loads(handle("analyze_image", {"image_path": ""}))
        assert "error" in result

    def test_none_image_path_returns_error(self):
        result = json.loads(handle("analyze_image", {"image_path": None}))
        assert "error" in result

    def test_integer_image_path_returns_error(self):
        result = json.loads(handle("analyze_image", {"image_path": 42}))
        assert "error" in result


class TestCompareOutputsRequiredFields:
    """compare_outputs must return structured error when image_a or image_b is missing."""

    def test_missing_image_a_returns_error(self):
        result = json.loads(handle("compare_outputs", {"image_b": "/some/b.png"}))
        assert "error" in result
        assert "image_a" in result["error"].lower()

    def test_missing_image_b_returns_error(self):
        result = json.loads(handle("compare_outputs", {"image_a": "/some/a.png"}))
        assert "error" in result
        assert "image_b" in result["error"].lower()

    def test_none_image_a_returns_error(self):
        result = json.loads(handle("compare_outputs", {"image_a": None, "image_b": "/b.png"}))
        assert "error" in result

    def test_none_image_b_returns_error(self):
        result = json.loads(handle("compare_outputs", {"image_a": "/a.png", "image_b": None}))
        assert "error" in result


class TestSuggestImprovementsRequiredFields:
    """suggest_improvements must return structured error when required fields are missing."""

    def test_missing_image_path_returns_error(self):
        result = json.loads(handle("suggest_improvements", {
            "workflow_summary": "txt2img pipeline",
        }))
        assert "error" in result
        assert "image_path" in result["error"].lower()

    def test_missing_workflow_summary_returns_error(self):
        result = json.loads(handle("suggest_improvements", {
            "image_path": "/some/image.png",
        }))
        assert "error" in result
        assert "workflow_summary" in result["error"].lower()

    def test_none_workflow_summary_returns_error(self):
        result = json.loads(handle("suggest_improvements", {
            "image_path": "/some/image.png",
            "workflow_summary": None,
        }))
        assert "error" in result


class TestHashCompareRequiredFields:
    """hash_compare_images must return structured error when image_a or image_b is missing."""

    def test_missing_image_a_returns_error(self):
        result = json.loads(handle("hash_compare_images", {"image_b": "/b.png"}))
        assert "error" in result
        assert "image_a" in result["error"].lower()

    def test_missing_image_b_returns_error(self):
        result = json.loads(handle("hash_compare_images", {"image_a": "/a.png"}))
        assert "error" in result
        assert "image_b" in result["error"].lower()


# ---------------------------------------------------------------------------
# Cycle 49 — FileNotFoundError user-friendly context
# ---------------------------------------------------------------------------

class TestVisionFileNotFoundContext:
    """analyze_image and suggest_improvements must wrap FileNotFoundError with context."""

    def test_analyze_image_file_not_found_has_context(self):
        import json
        from agent.brain import vision
        v = vision.VisionAgent()
        # Pass a path that definitely doesn't exist
        result = json.loads(v.handle("analyze_image", {
            "image_path": "/nonexistent/path/image_xyz_99999.png",
        }))
        assert "error" in result
        assert "image_path" in result
        assert "hint" in result

    def test_suggest_improvements_file_not_found_has_context(self):
        import json
        from agent.brain import vision
        v = vision.VisionAgent()
        result = json.loads(v.handle("suggest_improvements", {
            "image_path": "/nonexistent/path/image_xyz_99999.png",
            "workflow_summary": "SDXL checkpoint with KSampler",
        }))
        assert "error" in result
        assert "image_path" in result
        assert "hint" in result


# ---------------------------------------------------------------------------
# Cycle 53 — empty file (0-byte) guard in _read_image_as_base64
# ---------------------------------------------------------------------------

class TestVisionEmptyFileGuard:
    """_read_image_as_base64 must reject 0-byte files before calling Vision API."""

    def test_empty_file_returns_error(self, tmp_path):
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")
        from agent.brain import vision
        v = vision.VisionAgent()
        result = json.loads(v.handle("analyze_image", {"image_path": str(empty)}))
        assert "error" in result

    def test_empty_file_error_mentions_empty(self, tmp_path):
        empty = tmp_path / "zero.png"
        empty.write_bytes(b"")
        from agent.brain import vision
        v = vision.VisionAgent()
        result = json.loads(v.handle("analyze_image", {"image_path": str(empty)}))
        assert "empty" in result["error"].lower() or "0" in result["error"]

    def test_nonempty_file_not_blocked(self, fake_image):
        """A valid image file must not be rejected by the 0-byte guard."""
        from agent.brain import vision
        import unittest.mock as mock
        v = vision.VisionAgent()
        with mock.patch.object(v, "_call_vision", return_value='{"quality_score": 0.9}'):
            result = json.loads(v.handle("analyze_image", {"image_path": fake_image}))
        assert "error" not in result or "empty" not in result.get("error", "")
