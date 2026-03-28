"""USD Scene Compositor — builds USD scenes from ComfyUI generation outputs.

Composes a USD stage from image outputs, depth maps, normals, segmentation
masks, and camera parameters. The resulting stage contains:

  /camera          UsdGeomCamera with projection params
  /subject/mesh    UsdGeomMesh from depth map (height field)
  /subject/material  UsdShadeMaterial with image+normals as textures
  /subject/segmentation  primvar mask data

Requires usd-core: pip install usd-core
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    HAS_USD = True
except ImportError:
    HAS_USD = False


class CompositorError(Exception):
    """Base error for compositor operations."""


@dataclass
class CameraParams:
    """Camera parameters for scene composition."""

    focal_length: float = 50.0      # mm
    sensor_width: float = 36.0      # mm (full frame)
    sensor_height: float = 24.0     # mm
    near_clip: float = 0.1
    far_clip: float = 1000.0
    fstop: float = 2.8
    focus_distance: float = 5.0     # meters
    position: tuple[float, float, float] = (0.0, 0.0, 5.0)
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def fov_horizontal(self) -> float:
        """Horizontal field of view in degrees."""
        return 2.0 * math.degrees(
            math.atan(self.sensor_width / (2.0 * self.focal_length))
        )

    @property
    def fov_vertical(self) -> float:
        """Vertical field of view in degrees."""
        return 2.0 * math.degrees(
            math.atan(self.sensor_height / (2.0 * self.focal_length))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "focal_length": self.focal_length,
            "sensor_width": self.sensor_width,
            "sensor_height": self.sensor_height,
            "near_clip": self.near_clip,
            "far_clip": self.far_clip,
            "fstop": self.fstop,
            "focus_distance": self.focus_distance,
            "position": list(self.position),
            "look_at": list(self.look_at),
            "fov_horizontal": self.fov_horizontal,
            "fov_vertical": self.fov_vertical,
        }


def compose_scene_from_outputs(
    image_path: str | Path | None = None,
    depth_path: str | Path | None = None,
    normals_path: str | Path | None = None,
    segmentation_path: str | Path | None = None,
    camera_params: CameraParams | None = None,
    *,
    resolution: tuple[int, int] = (512, 512),
) -> Any:  # Returns Usd.Stage
    """Compose a USD scene from ComfyUI generation outputs.

    All paths are optional — compose what you have. The resulting stage
    always has /camera and /subject, but only populates components for
    which data was provided.

    Args:
        image_path: Path to the generated image (diffuse texture).
        depth_path: Path to depth map image.
        normals_path: Path to normals map image.
        segmentation_path: Path to segmentation mask image.
        camera_params: Camera parameters. Defaults to standard 50mm.
        resolution: Image resolution (width, height) for mesh generation.

    Returns:
        pxr.Usd.Stage with composed scene.

    Raises:
        CompositorError: If USD is not available.
    """
    if not HAS_USD:
        raise CompositorError(
            "USD not available. Install with: pip install usd-core"
        )

    cam = camera_params or CameraParams()
    stage = Usd.Stage.CreateInMemory("composed_scene.usda")

    # Set stage metadata
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetStartTimeCode(1.0)
    stage.SetEndTimeCode(1.0)

    # Create camera
    _create_camera(stage, cam)

    # Create subject scope
    UsdGeom.Scope.Define(stage, "/subject")

    # Mesh from depth (height field grid)
    width, height = resolution
    _create_mesh(stage, "/subject/mesh", width, height, depth_path)

    # Material with textures
    _create_material(stage, "/subject/material", image_path, normals_path)

    # Bind material to mesh
    mesh_prim = stage.GetPrimAtPath("/subject/mesh")
    mat_prim = stage.GetPrimAtPath("/subject/material")
    if mesh_prim.IsValid() and mat_prim.IsValid():
        UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
            UsdShade.Material(mat_prim)
        )

    # Segmentation as primvar
    if segmentation_path is not None:
        _create_segmentation_primvar(
            stage, "/subject/segmentation", str(segmentation_path),
        )

    return stage


def _create_camera(stage: Any, cam: CameraParams) -> None:
    """Create a UsdGeomCamera at /camera."""
    camera = UsdGeom.Camera.Define(stage, "/camera")
    camera.GetFocalLengthAttr().Set(cam.focal_length)
    camera.GetHorizontalApertureAttr().Set(cam.sensor_width)
    camera.GetVerticalApertureAttr().Set(cam.sensor_height)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(cam.near_clip, cam.far_clip))
    camera.GetFStopAttr().Set(cam.fstop)
    camera.GetFocusDistanceAttr().Set(cam.focus_distance)

    # Position camera
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*cam.position))


def _create_mesh(
    stage: Any,
    prim_path: str,
    width: int,
    height: int,
    depth_path: str | Path | None,
) -> None:
    """Create a height-field mesh at prim_path.

    Generates a grid of vertices. If depth_path is provided, stores
    the path as a custom attribute for later displacement.
    """
    mesh = UsdGeom.Mesh.Define(stage, prim_path)

    # Generate grid vertices (flat plane, depth applied later)
    points = []
    for y in range(height):
        for x in range(width):
            # Normalize to [-0.5, 0.5] range
            px = (x / max(width - 1, 1)) - 0.5
            py = (y / max(height - 1, 1)) - 0.5
            points.append(Gf.Vec3f(px, py, 0.0))

    mesh.GetPointsAttr().Set(points)

    # Generate quad face topology
    face_counts = []
    face_indices = []
    for y in range(height - 1):
        for x in range(width - 1):
            i = y * width + x
            face_counts.append(4)
            face_indices.extend([i, i + 1, i + width + 1, i + width])

    mesh.GetFaceVertexCountsAttr().Set(face_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_indices)

    # Store depth path as custom attribute
    if depth_path is not None:
        prim = stage.GetPrimAtPath(prim_path)
        prim.CreateAttribute(
            "comfyui:depth_map", Sdf.ValueTypeNames.String
        ).Set(str(depth_path))


def _create_material(
    stage: Any,
    prim_path: str,
    image_path: str | Path | None,
    normals_path: str | Path | None,
) -> None:
    """Create a UsdShadeMaterial with texture references."""
    material = UsdShade.Material.Define(stage, prim_path)
    shader = UsdShade.Shader.Define(stage, f"{prim_path}/PBRShader")
    shader.CreateIdAttr("UsdPreviewSurface")

    # Connect shader to material surface output
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )

    if image_path is not None:
        # Diffuse texture reader
        tex_reader = UsdShade.Shader.Define(
            stage, f"{prim_path}/DiffuseTexture"
        )
        tex_reader.CreateIdAttr("UsdUVTexture")
        tex_reader.CreateInput(
            "file", Sdf.ValueTypeNames.Asset
        ).Set(str(image_path))
        tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        shader.CreateInput(
            "diffuseColor", Sdf.ValueTypeNames.Color3f
        ).ConnectToSource(tex_reader.ConnectableAPI(), "rgb")

    if normals_path is not None:
        # Normal map reader
        norm_reader = UsdShade.Shader.Define(
            stage, f"{prim_path}/NormalTexture"
        )
        norm_reader.CreateIdAttr("UsdUVTexture")
        norm_reader.CreateInput(
            "file", Sdf.ValueTypeNames.Asset
        ).Set(str(normals_path))
        norm_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        shader.CreateInput(
            "normal", Sdf.ValueTypeNames.Normal3f
        ).ConnectToSource(norm_reader.ConnectableAPI(), "rgb")


def _create_segmentation_primvar(
    stage: Any,
    prim_path: str,
    mask_path: str,
) -> None:
    """Store segmentation mask path as a primvar on a scope prim."""
    scope = UsdGeom.Scope.Define(stage, prim_path)
    prim = scope.GetPrim()
    prim.CreateAttribute(
        "comfyui:segmentation_mask", Sdf.ValueTypeNames.String
    ).Set(mask_path)


def export_scene(
    stage: Any,
    output_path: str | Path,
    *,
    fmt: str = "usda",
) -> str:
    """Export a composed scene to disk.

    Args:
        stage: USD stage to export.
        output_path: Output file path.
        fmt: Format — "usda" (text), "usdc" (binary), "usdz" (package).

    Returns:
        Path the scene was exported to.
    """
    if not HAS_USD:
        raise CompositorError("USD not available")

    path = str(output_path)

    if fmt == "usdz":
        # USDZ requires flattening
        from pxr import UsdUtils
        flat = stage.Flatten()
        flat_path = path.replace(".usdz", ".usdc")
        flat.Export(flat_path)
        UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(flat_path), path)
        return path

    stage.GetRootLayer().Export(path)
    return path
