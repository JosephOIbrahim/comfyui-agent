# 3D Camera Pipeline in ComfyUI

## LOAD3D_CAMERA Type

The `LOAD3D_CAMERA` connection type carries structured camera data between 3D
viewport/loader nodes and downstream consumers like ControlNet or render nodes.

### Schema

A LOAD3D_CAMERA value is a dict with these fields:

| Field | Type | Description |
|-------|------|-------------|
| position | [x, y, z] | Camera world position |
| target | [x, y, z] | Look-at point |
| up | [x, y, z] | Camera up vector (usually [0, 1, 0]) |
| fov | float | Vertical field of view in degrees |
| focal_length | float | Focal length in mm (derived from fov + sensor) |
| near | float | Near clip plane |
| far | float | Far clip plane |

### Extension Fields (carwash_ prefix)

The comfyui_3D_viewport adds cinematographic camera data via `carwash_` prefixed
fields. These are optional and ignored by nodes that don't understand them.

| Field | Type | Description |
|-------|------|-------------|
| carwash_sensor_width | float | Physical sensor width in mm |
| carwash_sensor_height | float | Physical sensor height in mm |
| carwash_camera_model | string | Camera body name (e.g. "ARRI ALEXA 35") |
| carwash_lens_model | string | Lens name (e.g. "Cooke Anamorphic/i 40mm") |
| carwash_aspect_ratio | float | Sensor aspect ratio |
| carwash_squeeze | float | Anamorphic squeeze factor (1.0 for spherical) |

## Source Nodes (Producers)

### Load3D / Load3DAnimation
- ComfyUI built-in nodes for loading 3D models
- Output: IMAGE, MASK, LOAD3D_CAMERA
- LOAD3D_CAMERA carries the viewport camera when the user interacts with the 3D preview

### comfyui_3D_viewport (External)
- Standalone GL viewport with physical camera models
- Exports LOAD3D_CAMERA JSON via bridge or file
- Adds `carwash_` extension fields for cinematic camera data

## Consumer Nodes

### AdvancedCameraControlNode
- Uses LOAD3D_CAMERA to set up camera-controlled ControlNet passes
- Reads position, target, fov to compute depth/normal maps

### ControlNet Depth/Normal
- Downstream of camera data: renders depth or normal passes from the 3D camera
- LOAD3D_CAMERA -> depth renderer -> ControlNet -> KSampler

## When to Recommend Camera Pipeline

Trigger this knowledge when the user asks about:
- Setting up camera control for 3D-to-2D workflows
- Using a 3D viewport camera with ControlNet
- Matching a real camera lens to ComfyUI's 3D preview
- Exporting camera data from a 3D viewer
- Cinematographic camera settings (focal length, sensor size, anamorphic)

## Typical Camera Pipeline

```
Load3D (or external viewport)
  |-- LOAD3D_CAMERA --> AdvancedCameraControlNode
  |                         |-- depth pass --> ControlNet --> KSampler
  |-- IMAGE ------------> img2img or reference
```

## Tips

- LOAD3D_CAMERA fov is vertical FOV in degrees. Convert from horizontal:
  `vfov = 2 * atan(tan(hfov/2) * height/width)`
- When using physical cameras, focal_length is more reliable than fov for
  matching real-world lenses
- The `carwash_` fields are passthrough -- they won't break nodes that don't
  read them, but nodes that do can provide cinema-accurate framing
