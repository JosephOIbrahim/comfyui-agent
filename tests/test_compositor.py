"""Tests for agent/stage/compositor.py — USD scene composition, no real I/O."""

from __future__ import annotations

import pytest

from agent.stage.compositor import (
    CameraParams,
    CompositorError,
    compose_scene_from_outputs,
    export_scene,
)


@pytest.fixture
def usd():
    return pytest.importorskip("pxr", reason="usd-core not installed")


class TestCameraParams:
    def test_defaults(self):
        cam = CameraParams()
        assert cam.focal_length == 50.0
        assert cam.fstop == 2.8

    def test_fov_horizontal(self):
        cam = CameraParams(focal_length=50.0, sensor_width=36.0)
        assert 38.0 < cam.fov_horizontal < 42.0  # ~39.6 degrees

    def test_fov_vertical(self):
        cam = CameraParams(focal_length=50.0, sensor_height=24.0)
        assert 26.0 < cam.fov_vertical < 29.0  # ~27.0 degrees

    def test_wide_lens_larger_fov(self):
        wide = CameraParams(focal_length=24.0)
        tele = CameraParams(focal_length=200.0)
        assert wide.fov_horizontal > tele.fov_horizontal

    def test_to_dict(self):
        d = CameraParams().to_dict()
        assert "focal_length" in d
        assert "fov_horizontal" in d
        assert len(d) == 11

    def test_custom_position(self):
        cam = CameraParams(position=(1.0, 2.0, 3.0))
        assert cam.position == (1.0, 2.0, 3.0)


class TestComposeScene:
    def test_minimal_scene(self, usd):
        stage = compose_scene_from_outputs()
        assert stage is not None
        assert stage.GetPrimAtPath("/camera").IsValid()
        assert stage.GetPrimAtPath("/subject").IsValid()

    def test_camera_created(self, usd):
        cam = CameraParams(focal_length=35.0, fstop=1.4)
        stage = compose_scene_from_outputs(camera_params=cam)
        camera_prim = stage.GetPrimAtPath("/camera")
        assert camera_prim.IsValid()
        fl = camera_prim.GetAttribute("focalLength").Get()
        assert abs(fl - 35.0) < 1e-6

    def test_mesh_created(self, usd):
        stage = compose_scene_from_outputs(resolution=(4, 4))
        mesh = stage.GetPrimAtPath("/subject/mesh")
        assert mesh.IsValid()
        points = mesh.GetAttribute("points").Get()
        assert len(points) == 16  # 4x4 grid

    def test_mesh_topology(self, usd):
        stage = compose_scene_from_outputs(resolution=(3, 3))
        mesh = stage.GetPrimAtPath("/subject/mesh")
        counts = mesh.GetAttribute("faceVertexCounts").Get()
        assert len(counts) == 4  # (3-1)*(3-1) = 4 quads
        assert all(c == 4 for c in counts)

    def test_with_image_path(self, usd, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"fake_png")
        stage = compose_scene_from_outputs(image_path=str(img))
        mat = stage.GetPrimAtPath("/subject/material")
        assert mat.IsValid()
        tex = stage.GetPrimAtPath("/subject/material/DiffuseTexture")
        assert tex.IsValid()

    def test_with_depth_path(self, usd, tmp_path):
        depth = tmp_path / "depth.png"
        depth.write_bytes(b"fake")
        stage = compose_scene_from_outputs(depth_path=str(depth))
        mesh = stage.GetPrimAtPath("/subject/mesh")
        attr = mesh.GetAttribute("comfyui:depth_map")
        assert attr.IsValid()
        assert str(depth) in attr.Get()

    def test_with_normals_path(self, usd, tmp_path):
        normals = tmp_path / "normals.png"
        normals.write_bytes(b"fake")
        stage = compose_scene_from_outputs(normals_path=str(normals))
        norm_tex = stage.GetPrimAtPath("/subject/material/NormalTexture")
        assert norm_tex.IsValid()

    def test_with_segmentation_path(self, usd, tmp_path):
        seg = tmp_path / "seg.png"
        seg.write_bytes(b"fake")
        stage = compose_scene_from_outputs(segmentation_path=str(seg))
        seg_prim = stage.GetPrimAtPath("/subject/segmentation")
        assert seg_prim.IsValid()

    def test_material_bound_to_mesh(self, usd, tmp_path):
        img = tmp_path / "img.png"
        img.write_bytes(b"fake")
        stage = compose_scene_from_outputs(image_path=str(img))
        from pxr import UsdShade
        mesh = stage.GetPrimAtPath("/subject/mesh")
        binding = UsdShade.MaterialBindingAPI(mesh)
        mat, _ = binding.ComputeBoundMaterial()
        assert mat.GetPath() == "/subject/material"

    def test_stage_up_axis(self, usd):
        stage = compose_scene_from_outputs()
        from pxr import UsdGeom
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y

    def test_full_scene(self, usd, tmp_path):
        for name in ("img.png", "depth.png", "normals.png", "seg.png"):
            (tmp_path / name).write_bytes(b"fake")
        stage = compose_scene_from_outputs(
            image_path=str(tmp_path / "img.png"),
            depth_path=str(tmp_path / "depth.png"),
            normals_path=str(tmp_path / "normals.png"),
            segmentation_path=str(tmp_path / "seg.png"),
            camera_params=CameraParams(focal_length=85.0),
            resolution=(8, 8),
        )
        assert stage.GetPrimAtPath("/camera").IsValid()
        assert stage.GetPrimAtPath("/subject/mesh").IsValid()
        assert stage.GetPrimAtPath("/subject/material").IsValid()
        assert stage.GetPrimAtPath("/subject/segmentation").IsValid()


class TestExportScene:
    def test_export_usda(self, usd, tmp_path):
        stage = compose_scene_from_outputs()
        path = export_scene(stage, tmp_path / "scene.usda", fmt="usda")
        assert (tmp_path / "scene.usda").exists()

    def test_export_usdc(self, usd, tmp_path):
        stage = compose_scene_from_outputs()
        path = export_scene(stage, tmp_path / "scene.usdc", fmt="usdc")
        assert (tmp_path / "scene.usdc").exists()
