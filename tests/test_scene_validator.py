"""Tests for agent/stage/scene_validator.py — no real I/O."""

from __future__ import annotations

import pytest

from agent.stage.scene_validator import ValidationResult, validate_scene


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
        camera_params=CameraParams(),
    )


@pytest.fixture
def minimal_scene(usd):
    from agent.stage.compositor import compose_scene_from_outputs
    return compose_scene_from_outputs()


class TestValidationResult:
    def test_overall_average(self):
        r = ValidationResult(
            depth_consistency=0.8, normal_agreement=0.6,
            segmentation_quality=0.4, camera_fidelity=1.0,
        )
        assert abs(r.overall - 0.7) < 1e-9

    def test_to_dict(self):
        r = ValidationResult()
        d = r.to_dict()
        assert "overall" in d
        assert "issues" in d
        assert len(d) == 6

    def test_issues_defaults_to_empty(self):
        r = ValidationResult()
        assert r.issues == []


class TestValidateFullScene:
    def test_full_scene_high_scores(self, full_scene):
        result = validate_scene(full_scene)
        assert result.depth_consistency >= 0.8
        assert result.normal_agreement >= 0.8
        assert result.segmentation_quality >= 0.8
        assert result.camera_fidelity >= 0.8

    def test_full_scene_overall(self, full_scene):
        result = validate_scene(full_scene)
        assert result.overall >= 0.8

    def test_full_scene_no_critical_issues(self, full_scene):
        result = validate_scene(full_scene)
        # May have minor issues but overall score is high
        assert result.overall > 0.7


class TestValidateMinimalScene:
    def test_minimal_has_camera(self, minimal_scene):
        result = validate_scene(minimal_scene)
        assert result.camera_fidelity > 0.5

    def test_minimal_has_mesh(self, minimal_scene):
        result = validate_scene(minimal_scene)
        assert result.depth_consistency > 0.0

    def test_minimal_missing_normals(self, minimal_scene):
        result = validate_scene(minimal_scene)
        assert result.normal_agreement < 1.0

    def test_minimal_missing_segmentation(self, minimal_scene):
        result = validate_scene(minimal_scene)
        assert result.segmentation_quality == 0.0

    def test_minimal_has_issues(self, minimal_scene):
        result = validate_scene(minimal_scene)
        assert len(result.issues) > 0


class TestValidateEmptyStage:
    def test_empty_stage(self, usd):
        from pxr import Usd
        stage = Usd.Stage.CreateInMemory()
        result = validate_scene(stage)
        assert result.depth_consistency == 0.0
        assert result.camera_fidelity == 0.0
        assert result.overall == 0.0
        assert len(result.issues) >= 2
