"""Compositor tools — expose scene composition/validation as MCP tools.

Four tools for the USD scene pipeline:
  compose_scene         — build USD scene from ComfyUI outputs
  validate_scene        — multi-axis quality check
  extract_conditioning  — derive ComfyUI inputs from scene
  export_scene          — save to usda/usdc/usdz

Tool pattern: TOOLS list[dict] + handle(name, tool_input) -> str.
"""

from __future__ import annotations

import threading

from ..tools._util import to_json

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "compose_scene",
        "description": (
            "Build a USD scene from ComfyUI generation outputs (image, depth, "
            "normals, segmentation). Creates camera, mesh, material, and "
            "segmentation prims. All paths are optional — compose what you have."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the generated image (diffuse texture).",
                },
                "depth_path": {
                    "type": "string",
                    "description": "Path to the depth map image.",
                },
                "normals_path": {
                    "type": "string",
                    "description": "Path to the normals map image.",
                },
                "segmentation_path": {
                    "type": "string",
                    "description": "Path to the segmentation mask image.",
                },
                "focal_length": {
                    "type": "number",
                    "description": "Camera focal length in mm. Default 50.",
                },
                "fstop": {
                    "type": "number",
                    "description": "Camera f-stop. Default 2.8.",
                },
                "resolution_width": {
                    "type": "integer",
                    "description": "Image width for mesh grid. Default 512.",
                },
                "resolution_height": {
                    "type": "integer",
                    "description": "Image height for mesh grid. Default 512.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "validate_scene",
        "description": (
            "Validate a composed USD scene for quality. Returns multi-axis "
            "scores (depth_consistency, normal_agreement, segmentation_quality, "
            "camera_fidelity) plus an overall score and list of issues."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "extract_conditioning",
        "description": (
            "Extract ComfyUI workflow conditioning from a composed USD scene. "
            "Returns ControlNet depth path, FOV, DOF parameters, and prompt "
            "additions derived from camera and material analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "export_scene",
        "description": (
            "Export the composed USD scene to disk. Supports usda (text), "
            "usdc (binary), and usdz (packaged) formats."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "output_path": {
                    "type": "string",
                    "description": "Path to write the scene file.",
                },
                "format": {
                    "type": "string",
                    "enum": ["usda", "usdc", "usdz"],
                    "description": "Output format. Default usda.",
                },
            },
            "required": ["output_path"],
        },
    },
]


# ---------------------------------------------------------------------------
# Session-scoped scene storage
# ---------------------------------------------------------------------------

_scenes: dict[str, object] = {}  # Per-session scene storage, keyed by session_id
_scenes_lock = threading.Lock()  # Guards _scenes mutations (Cycle 41; session-scoped)


def _get_scene():
    """Return the composed scene for the current connection's session, or None."""
    from .._conn_ctx import current_conn_session
    sid = current_conn_session()
    with _scenes_lock:
        return _scenes.get(sid)


def _set_scene(scene):
    """Store the composed scene for the current connection's session."""
    from .._conn_ctx import current_conn_session
    sid = current_conn_session()
    with _scenes_lock:
        _scenes[sid] = scene


def _clear_scene():
    """Remove the composed scene for the current connection's session, if any."""
    from .._conn_ctx import current_conn_session
    sid = current_conn_session()
    with _scenes_lock:
        _scenes.pop(sid, None)


_NO_USD = to_json({
    "error": "USD not available. Install with: pip install usd-core",
})

_NO_SCENE = to_json({
    "error": "No scene composed yet. Use compose_scene first.",
})


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_compose_scene(tool_input: dict) -> str:
    try:
        from .compositor import CameraParams, compose_scene_from_outputs

        cam = CameraParams(
            focal_length=tool_input.get("focal_length", 50.0),
            fstop=tool_input.get("fstop", 2.8),
        )
        width = tool_input.get("resolution_width", 512)
        height = tool_input.get("resolution_height", 512)

        scene = compose_scene_from_outputs(
            image_path=tool_input.get("image_path"),
            depth_path=tool_input.get("depth_path"),
            normals_path=tool_input.get("normals_path"),
            segmentation_path=tool_input.get("segmentation_path"),
            camera_params=cam,
            resolution=(width, height),
        )
        _set_scene(scene)

        return to_json({
            "composed": True,
            "resolution": [width, height],
            "has_image": tool_input.get("image_path") is not None,
            "has_depth": tool_input.get("depth_path") is not None,
            "has_normals": tool_input.get("normals_path") is not None,
            "has_segmentation": tool_input.get("segmentation_path") is not None,
            "camera": cam.to_dict(),
        })
    except Exception as exc:  # noqa: BLE001
        return to_json({"error": str(exc)})


def _handle_validate_scene(tool_input: dict) -> str:  # noqa: ARG001
    scene = _get_scene()
    if scene is None:
        return _NO_SCENE

    try:
        from .scene_validator import validate_scene
        result = validate_scene(scene)
        return to_json(result.to_dict())
    except Exception as exc:  # noqa: BLE001
        return to_json({"error": str(exc)})


def _handle_extract_conditioning(tool_input: dict) -> str:  # noqa: ARG001
    scene = _get_scene()
    if scene is None:
        return _NO_SCENE

    try:
        from .scene_conditioner import extract_conditioning
        cond = extract_conditioning(scene)
        result = cond.to_dict()
        result["prompt_suffix"] = cond.to_prompt_suffix()
        return to_json(result)
    except Exception as exc:  # noqa: BLE001
        return to_json({"error": str(exc)})


def _handle_export_scene(tool_input: dict) -> str:
    output_path = tool_input.get("output_path")  # Cycle 55: guard before scene check
    if not output_path or not isinstance(output_path, str):
        return to_json({"error": "output_path is required and must be a non-empty string."})

    scene = _get_scene()
    if scene is None:
        return _NO_SCENE

    try:
        from .compositor import export_scene
        fmt = tool_input.get("format", "usda")
        path = export_scene(scene, output_path, fmt=fmt)
        return to_json({"exported": path, "format": fmt})
    except Exception as exc:  # noqa: BLE001
        return to_json({"error": str(exc)})


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_DISPATCH = {
    "compose_scene": _handle_compose_scene,
    "validate_scene": _handle_validate_scene,
    "extract_conditioning": _handle_extract_conditioning,
    "export_scene": _handle_export_scene,
}


def handle(name: str, tool_input: dict) -> str:
    """Execute a compositor tool call. Returns JSON string."""
    try:
        handler = _DISPATCH.get(name)
        if handler is None:
            return to_json({"error": f"Unknown tool: {name}"})
        return handler(tool_input)
    except Exception as exc:  # noqa: BLE001
        return to_json({"error": str(exc)})
