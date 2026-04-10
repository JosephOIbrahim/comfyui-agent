"""Tests for agent/stage/compositor_tools.py — all mocked, no real I/O."""

from __future__ import annotations

import json

import pytest

from agent.stage.compositor_tools import TOOLS, handle, _set_scene, _get_scene


@pytest.fixture
def usd():
    return pytest.importorskip("pxr", reason="usd-core not installed")


@pytest.fixture(autouse=True)
def _reset_scene():
    """Reset module-level scene between tests."""
    _set_scene(None)
    yield
    _set_scene(None)


class TestToolSchemas:
    def test_four_tools(self):
        assert len(TOOLS) == 4

    def test_expected_names(self):
        names = {t["name"] for t in TOOLS}
        assert names == {
            "compose_scene", "validate_scene",
            "extract_conditioning", "export_scene",
        }

    def test_all_have_description(self):
        for t in TOOLS:
            assert len(t["description"]) > 10


class TestComposeScene:
    def test_minimal(self, usd):
        result = json.loads(handle("compose_scene", {}))
        assert result["composed"] is True
        assert _get_scene() is not None

    def test_with_image(self, usd, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"fake")
        result = json.loads(handle("compose_scene", {
            "image_path": str(img),
        }))
        assert result["has_image"] is True

    def test_custom_camera(self, usd):
        result = json.loads(handle("compose_scene", {
            "focal_length": 85.0, "fstop": 1.4,
        }))
        assert result["camera"]["focal_length"] == 85.0

    def test_custom_resolution(self, usd):
        result = json.loads(handle("compose_scene", {
            "resolution_width": 1024, "resolution_height": 768,
        }))
        assert result["resolution"] == [1024, 768]


class TestValidateScene:
    def test_no_scene(self):
        result = json.loads(handle("validate_scene", {}))
        assert "error" in result

    def test_with_scene(self, usd, tmp_path):
        for n in ("img.png", "depth.png", "normals.png", "seg.png"):
            (tmp_path / n).write_bytes(b"fake")
        handle("compose_scene", {
            "image_path": str(tmp_path / "img.png"),
            "depth_path": str(tmp_path / "depth.png"),
            "normals_path": str(tmp_path / "normals.png"),
            "segmentation_path": str(tmp_path / "seg.png"),
        })
        result = json.loads(handle("validate_scene", {}))
        assert "overall" in result
        assert result["overall"] > 0.5


class TestExtractConditioning:
    def test_no_scene(self):
        result = json.loads(handle("extract_conditioning", {}))
        assert "error" in result

    def test_with_scene(self, usd, tmp_path):
        (tmp_path / "depth.png").write_bytes(b"fake")
        handle("compose_scene", {"depth_path": str(tmp_path / "depth.png")})
        result = json.loads(handle("extract_conditioning", {}))
        assert "fov" in result
        assert result["controlnet_depth"] is not None


class TestExportScene:
    def test_no_scene(self):
        result = json.loads(handle("export_scene", {"output_path": "/tmp/x.usda"}))
        assert "error" in result

    def test_export_usda(self, usd, tmp_path):
        handle("compose_scene", {})
        out = tmp_path / "scene.usda"
        result = json.loads(handle("export_scene", {
            "output_path": str(out), "format": "usda",
        }))
        assert result["format"] == "usda"
        assert out.exists()


class TestDispatch:
    def test_unknown(self):
        result = json.loads(handle("nonexistent", {}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Cycle 55 — required field guard for export_scene handler
# ---------------------------------------------------------------------------

class TestExportSceneRequiredField:
    """export_scene must return error when output_path is missing or invalid."""

    def test_missing_output_path_returns_error(self):
        result = json.loads(handle("export_scene", {}))
        assert "error" in result
        assert "output_path" in result["error"].lower()

    def test_empty_output_path_returns_error(self):
        result = json.loads(handle("export_scene", {"output_path": ""}))
        assert "error" in result

    def test_none_output_path_returns_error(self):
        result = json.loads(handle("export_scene", {"output_path": None}))
        assert "error" in result

    def test_integer_output_path_returns_error(self):
        result = json.loads(handle("export_scene", {"output_path": 42}))
        assert "error" in result
