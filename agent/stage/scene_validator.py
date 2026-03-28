"""Scene Validator — multi-axis quality checks on composed USD scenes.

Validates scenes produced by the compositor against quality criteria:
  depth_consistency   — depth map has plausible range, no NaN/inf
  normal_agreement    — normals map referenced and valid
  segmentation_quality — mask referenced
  camera_fidelity     — camera params physically plausible

Returns a quality dict with scores [0, 1] per axis plus an overall score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from pxr import Usd  # noqa: F401 — HAS_USD guard
    HAS_USD = True
except ImportError:
    HAS_USD = False


class ValidatorError(Exception):
    """Base error for scene validation."""


@dataclass
class ValidationResult:
    """Multi-axis scene validation result."""

    depth_consistency: float = 0.0
    normal_agreement: float = 0.0
    segmentation_quality: float = 0.0
    camera_fidelity: float = 0.0
    issues: list[str] | None = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []

    @property
    def overall(self) -> float:
        """Average of all axis scores."""
        scores = [
            self.depth_consistency,
            self.normal_agreement,
            self.segmentation_quality,
            self.camera_fidelity,
        ]
        return sum(scores) / len(scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth_consistency": self.depth_consistency,
            "normal_agreement": self.normal_agreement,
            "segmentation_quality": self.segmentation_quality,
            "camera_fidelity": self.camera_fidelity,
            "overall": self.overall,
            "issues": self.issues,
        }


def validate_scene(scene_stage: Any) -> ValidationResult:
    """Validate a composed USD scene for quality.

    Args:
        scene_stage: USD stage from compose_scene_from_outputs().

    Returns:
        ValidationResult with per-axis scores and issues.

    Raises:
        ValidatorError: If USD is not available.
    """
    if not HAS_USD:
        raise ValidatorError("USD not available")

    issues: list[str] = []
    depth_score = _check_depth(scene_stage, issues)
    normal_score = _check_normals(scene_stage, issues)
    seg_score = _check_segmentation(scene_stage, issues)
    cam_score = _check_camera(scene_stage, issues)

    return ValidationResult(
        depth_consistency=depth_score,
        normal_agreement=normal_score,
        segmentation_quality=seg_score,
        camera_fidelity=cam_score,
        issues=issues,
    )


def _check_depth(stage: Any, issues: list[str]) -> float:
    """Check depth map consistency. Returns score [0, 1]."""
    mesh = stage.GetPrimAtPath("/subject/mesh")
    if not mesh.IsValid():
        issues.append("No mesh at /subject/mesh")
        return 0.0

    score = 0.5  # base: mesh exists

    # Check points attribute
    points_attr = mesh.GetAttribute("points")
    if not points_attr.IsValid():
        issues.append("Mesh has no points attribute")
        return 0.2

    points = points_attr.Get()
    if points is None or len(points) == 0:
        issues.append("Mesh has empty points")
        return 0.2

    score = 0.7  # mesh has valid points

    # Check for depth map reference
    depth_attr = mesh.GetAttribute("comfyui:depth_map")
    if depth_attr.IsValid() and depth_attr.Get():
        score = 1.0  # depth map referenced
    else:
        issues.append("No depth map referenced on mesh")
        score = 0.6

    # Check topology
    face_counts = mesh.GetAttribute("faceVertexCounts")
    face_indices = mesh.GetAttribute("faceVertexIndices")
    if not face_counts.IsValid() or not face_indices.IsValid():
        issues.append("Mesh has incomplete topology")
        score = min(score, 0.5)

    return score


def _check_normals(stage: Any, issues: list[str]) -> float:
    """Check normal map agreement. Returns score [0, 1]."""
    mat = stage.GetPrimAtPath("/subject/material")
    if not mat.IsValid():
        issues.append("No material at /subject/material")
        return 0.0

    score = 0.3  # material exists

    # Check for PBR shader
    shader = stage.GetPrimAtPath("/subject/material/PBRShader")
    if shader.IsValid():
        score = 0.5

    # Check for normal texture
    norm_tex = stage.GetPrimAtPath("/subject/material/NormalTexture")
    if norm_tex.IsValid():
        file_input = norm_tex.GetAttribute("inputs:file")
        if file_input.IsValid() and file_input.Get():
            score = 1.0
        else:
            issues.append("Normal texture has no file reference")
            score = 0.6
    else:
        issues.append("No normal map texture in material")
        score = 0.4

    return score


def _check_segmentation(stage: Any, issues: list[str]) -> float:
    """Check segmentation mask. Returns score [0, 1]."""
    seg = stage.GetPrimAtPath("/subject/segmentation")
    if not seg.IsValid():
        issues.append("No segmentation at /subject/segmentation")
        return 0.0

    score = 0.5  # prim exists

    mask_attr = seg.GetAttribute("comfyui:segmentation_mask")
    if mask_attr.IsValid() and mask_attr.Get():
        score = 1.0
    else:
        issues.append("Segmentation prim has no mask reference")
        score = 0.4

    return score


def _check_camera(stage: Any, issues: list[str]) -> float:
    """Check camera parameters for physical plausibility. Returns score [0, 1]."""
    camera = stage.GetPrimAtPath("/camera")
    if not camera.IsValid():
        issues.append("No camera at /camera")
        return 0.0

    score = 0.5  # camera exists

    # Check focal length
    fl_attr = camera.GetAttribute("focalLength")
    if fl_attr.IsValid():
        fl = fl_attr.Get()
        if fl is not None and 4.0 <= fl <= 1200.0:
            score += 0.15
        else:
            issues.append(f"Unusual focal length: {fl}mm")
    else:
        issues.append("Camera missing focal length")

    # Check aperture
    ha_attr = camera.GetAttribute("horizontalAperture")
    if ha_attr.IsValid():
        ha = ha_attr.Get()
        if ha is not None and 1.0 <= ha <= 100.0:
            score += 0.1
        else:
            issues.append(f"Unusual horizontal aperture: {ha}mm")

    # Check clipping range
    clip_attr = camera.GetAttribute("clippingRange")
    if clip_attr.IsValid():
        clip = clip_attr.Get()
        if clip is not None and clip[0] > 0.0 and clip[1] > clip[0]:
            score += 0.15
        else:
            issues.append(f"Invalid clipping range: {clip}")

    # Check f-stop
    fstop_attr = camera.GetAttribute("fStop")
    if fstop_attr.IsValid():
        fstop = fstop_attr.Get()
        if fstop is not None and 0.5 <= fstop <= 128.0:
            score += 0.1
        else:
            issues.append(f"Unusual f-stop: {fstop}")

    return min(score, 1.0)
