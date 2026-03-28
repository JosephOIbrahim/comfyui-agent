"""Scene Conditioning Extractor — derives ComfyUI workflow inputs from USD scenes.

Extracts conditioning data from a composed USD scene for use as ComfyUI
workflow inputs (Path 3: scene-driven generation):

  controlnet_depth   — depth map path for ControlNet conditioning
  fov                — horizontal FOV from camera focal length
  dof_params         — depth of field from f-stop + focus distance
  prompt_additions   — text hints from light/material analysis
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

try:
    from pxr import Usd  # noqa: F401 — HAS_USD guard
    HAS_USD = True
except ImportError:
    HAS_USD = False


class ConditionerError(Exception):
    """Base error for conditioning extraction."""


@dataclass
class SceneConditioning:
    """Extracted conditioning data from a USD scene."""

    controlnet_depth: str | None = None
    fov: float | None = None
    dof_params: dict[str, float] = field(default_factory=dict)
    prompt_additions: list[str] = field(default_factory=list)
    resolution: tuple[int, int] | None = None
    camera_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "controlnet_depth": self.controlnet_depth,
            "fov": self.fov,
            "dof_params": self.dof_params,
            "prompt_additions": self.prompt_additions,
            "resolution": list(self.resolution) if self.resolution else None,
            "camera_params": self.camera_params,
        }

    def to_prompt_suffix(self) -> str:
        """Generate a prompt suffix from extracted conditioning."""
        if not self.prompt_additions:
            return ""
        return ", ".join(self.prompt_additions)


def extract_conditioning(scene_stage: Any) -> SceneConditioning:
    """Extract conditioning data from a composed USD scene.

    Args:
        scene_stage: USD stage from compose_scene_from_outputs().

    Returns:
        SceneConditioning with extracted data.

    Raises:
        ConditionerError: If USD is not available.
    """
    if not HAS_USD:
        raise ConditionerError("USD not available")

    cond = SceneConditioning()

    _extract_depth(scene_stage, cond)
    _extract_camera(scene_stage, cond)
    _extract_material_hints(scene_stage, cond)
    _extract_resolution(scene_stage, cond)

    return cond


def _extract_depth(stage: Any, cond: SceneConditioning) -> None:
    """Extract depth map path for ControlNet."""
    mesh = stage.GetPrimAtPath("/subject/mesh")
    if not mesh.IsValid():
        return

    depth_attr = mesh.GetAttribute("comfyui:depth_map")
    if depth_attr.IsValid():
        val = depth_attr.Get()
        if val:
            cond.controlnet_depth = str(val)


def _extract_camera(stage: Any, cond: SceneConditioning) -> None:
    """Extract camera parameters: FOV, DOF, position."""
    camera = stage.GetPrimAtPath("/camera")
    if not camera.IsValid():
        return

    params: dict[str, Any] = {}

    # Focal length → FOV
    fl_attr = camera.GetAttribute("focalLength")
    ha_attr = camera.GetAttribute("horizontalAperture")
    if fl_attr.IsValid() and ha_attr.IsValid():
        fl = fl_attr.Get()
        ha = ha_attr.Get()
        if fl and ha and fl > 0:
            fov = 2.0 * math.degrees(math.atan(ha / (2.0 * fl)))
            cond.fov = round(fov, 2)
            params["focal_length"] = fl
            params["sensor_width"] = ha

            # Add lens-type prompt hints
            if fl < 28:
                cond.prompt_additions.append("ultra-wide angle perspective")
            elif fl < 50:
                cond.prompt_additions.append("wide angle shot")
            elif fl > 135:
                cond.prompt_additions.append("telephoto compression")
                cond.prompt_additions.append("shallow depth of field")

    # Vertical aperture
    va_attr = camera.GetAttribute("verticalAperture")
    if va_attr.IsValid():
        va = va_attr.Get()
        if va:
            params["sensor_height"] = va

    # DOF parameters
    fstop_attr = camera.GetAttribute("fStop")
    focus_attr = camera.GetAttribute("focusDistance")
    dof: dict[str, float] = {}

    if fstop_attr.IsValid():
        fstop = fstop_attr.Get()
        if fstop:
            dof["fstop"] = fstop
            params["fstop"] = fstop
            if fstop <= 2.0:
                cond.prompt_additions.append("bokeh background blur")
            elif fstop >= 11.0:
                cond.prompt_additions.append("deep focus, everything sharp")

    if focus_attr.IsValid():
        fd = focus_attr.Get()
        if fd:
            dof["focus_distance"] = fd
            params["focus_distance"] = fd

    if dof:
        cond.dof_params = dof

    # Clipping range
    clip_attr = camera.GetAttribute("clippingRange")
    if clip_attr.IsValid():
        clip = clip_attr.Get()
        if clip:
            params["near_clip"] = float(clip[0])
            params["far_clip"] = float(clip[1])

    cond.camera_params = params


def _extract_material_hints(stage: Any, cond: SceneConditioning) -> None:
    """Extract prompt hints from material/texture analysis."""
    mat = stage.GetPrimAtPath("/subject/material")
    if not mat.IsValid():
        return

    # Check for diffuse texture → photographic content
    diff_tex = stage.GetPrimAtPath("/subject/material/DiffuseTexture")
    if diff_tex.IsValid():
        file_attr = diff_tex.GetAttribute("inputs:file")
        if file_attr.IsValid() and file_attr.Get():
            cond.prompt_additions.append("photorealistic textures")

    # Check for normal map → detailed surface
    norm_tex = stage.GetPrimAtPath("/subject/material/NormalTexture")
    if norm_tex.IsValid():
        file_attr = norm_tex.GetAttribute("inputs:file")
        if file_attr.IsValid() and file_attr.Get():
            cond.prompt_additions.append("detailed surface normals")


def _extract_resolution(stage: Any, cond: SceneConditioning) -> None:
    """Extract resolution from mesh point count."""
    mesh = stage.GetPrimAtPath("/subject/mesh")
    if not mesh.IsValid():
        return

    points = mesh.GetAttribute("points")
    if not points.IsValid():
        return

    pts = points.Get()
    if pts is None:
        return

    # Estimate resolution from point count (sqrt for square grids)
    n = len(pts)
    side = int(math.sqrt(n))
    if side * side == n:
        cond.resolution = (side, side)
