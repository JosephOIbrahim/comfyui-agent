# ControlNet Workflow Patterns

## Required Nodes

- **ControlNetLoader** — loads the ControlNet model from `models/controlnet/`
- **ControlNetApply** or **ControlNetApplyAdvanced** — applies control signal to CONDITIONING
- A preprocessor node for the control type (depth, canny, openpose, etc.)

## Connection Pattern

```
Image -> Preprocessor -> ControlNetApply.image
ControlNetLoader -> ControlNetApply.control_net
CLIPTextEncode (positive) -> ControlNetApply.conditioning
ControlNetApply.CONDITIONING -> KSampler.positive
```

## Control Types and Preprocessors

| Control Type | Preprocessor Node | Model Pattern |
|-------------|-------------------|---------------|
| Depth | DepthAnythingPreprocessor, MiDaS-DepthMapPreprocessor | `*depth*` |
| Canny | CannyEdgePreprocessor | `*canny*` |
| OpenPose | OpenposePreprocessor | `*openpose*` |
| Lineart | LineartPreprocessor | `*lineart*` |
| Scribble | ScribblePreprocessor | `*scribble*` |
| Softedge | SoftEdgePreprocessor (HED/PiDiNet) | `*softedge*` |
| IP-Adapter | IPAdapterUnifiedLoader | `*ip-adapter*` (not ControlNet) |

## Key Constraints

- ControlNet image resolution must match the latent resolution (width/height divisible by 8)
- Multiple ControlNets can be chained: connect output CONDITIONING of one to input of next
- strength parameter (0.0-1.0) controls influence — start at 0.7-0.8
- start_percent/end_percent control when ControlNet activates during sampling
- SDXL ControlNets only work with SDXL models; SD1.5 ControlNets with SD1.5 models
