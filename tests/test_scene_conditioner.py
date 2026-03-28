"""Tests for agent/stage/scene_conditioner.py — no real I/O."""

from __future__ import annotations

import pytest

from agent.stage.scene_conditioner import SceneConditioning, extract_conditioning


@pytest.fixture
def usd():
    return pytest.importorskip("pxr", reason="usd-core not installed")


@pytest.fixture
def full_scene(usd, tmp_path):
    from agent.stage.compositor import CameraParams, compose_scene_from_outputs
    for name in ("img.png", "depth.png", "normals.png", "seg.png"):
        (tmp_path / name).write_bytes(b"fake")
    return compose_scene_from_outputs(
        image_path=str(tmp_path / "img.png"),
        depth_path=str(tmp_path / "depth.png"),
        normals_path=str(tmp_path / "normals.png"),
        segmentation_path=str(tmp_path / "seg.png"),
        camera_params=CameraParams(focal_length=50.0, fstop=2.8),
        resolution=(16, 16),
    )


class TestSceneConditioning:
    def test_to_dict(self):
        c = SceneConditioning(fov=39.6, controlnet_depth="/tmp/d.png")
        d = c.to_dict()
        assert d["fov"] == 39.6
        assert d["controlnet_depth"] == "/tmp/d.png"

    def test_to_prompt_suffix_empty(self):
        assert SceneConditioning().to_prompt_suffix() == ""

    def test_to_prompt_suffix(self):
        c = SceneConditioning(prompt_additions=["bokeh", "wide angle"])
        assert "bokeh" in c.to_prompt_suffix()


class TestExtractConditioning:
    def test_extracts_depth(self, full_scene):
        cond = extract_conditioning(full_scene)
        assert cond.controlnet_depth is not None
        assert "depth" in cond.controlnet_depth

    def test_extracts_fov(self, full_scene):
        cond = extract_conditioning(full_scene)
        assert cond.fov is not None
        assert 38.0 < cond.fov < 42.0  # 50mm ~ 39.6 degrees

    def test_extracts_dof(self, full_scene):
        cond = extract_conditioning(full_scene)
        assert "fstop" in cond.dof_params
        assert abs(cond.dof_params["fstop"] - 2.8) < 0.01

    def test_extracts_camera_params(self, full_scene):
        cond = extract_conditioning(full_scene)
        assert "focal_length" in cond.camera_params

    def test_extracts_resolution(self, full_scene):
        cond = extract_conditioning(full_scene)
        assert cond.resolution == (16, 16)

    def test_prompt_additions_from_materials(self, full_scene):
        cond = extract_conditioning(full_scene)
        assert any("texture" in p for p in cond.prompt_additions)

    def test_wide_angle_hint(self, usd, tmp_path):
        from agent.stage.compositor import CameraParams, compose_scene_from_outputs
        (tmp_path / "img.png").write_bytes(b"fake")
        stage = compose_scene_from_outputs(
            image_path=str(tmp_path / "img.png"),
            camera_params=CameraParams(focal_length=20.0),
        )
        cond = extract_conditioning(stage)
        assert any("ultra-wide" in p for p in cond.prompt_additions)

    def test_telephoto_hint(self, usd, tmp_path):
        from agent.stage.compositor import CameraParams, compose_scene_from_outputs
        (tmp_path / "img.png").write_bytes(b"fake")
        stage = compose_scene_from_outputs(
            image_path=str(tmp_path / "img.png"),
            camera_params=CameraParams(focal_length=200.0),
        )
        cond = extract_conditioning(stage)
        assert any("telephoto" in p for p in cond.prompt_additions)

    def test_bokeh_hint_low_fstop(self, usd, tmp_path):
        from agent.stage.compositor import CameraParams, compose_scene_from_outputs
        (tmp_path / "img.png").write_bytes(b"fake")
        stage = compose_scene_from_outputs(
            image_path=str(tmp_path / "img.png"),
            camera_params=CameraParams(fstop=1.4),
        )
        cond = extract_conditioning(stage)
        assert any("bokeh" in p for p in cond.prompt_additions)

    def test_empty_stage(self, usd):
        from pxr import Usd
        stage = Usd.Stage.CreateInMemory()
        cond = extract_conditioning(stage)
        assert cond.controlnet_depth is None
        assert cond.fov is None
