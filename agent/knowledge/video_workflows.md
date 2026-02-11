# Video Workflow Patterns

## AnimateDiff

The most common video generation method in ComfyUI.

### Required Nodes
- **ADE_AnimateDiffLoaderWithContext** or **AnimateDiffLoaderV2** — loads motion model
- **ADE_AnimateDiffUniformContextOptions** — sets frame count and context length
- Standard txt2img pipeline (checkpoint, CLIP, VAE, KSampler)
- **VHS_VideoCombine** — combines frames into video output

### Key Parameters
- `motion_model`: motion model file from `models/animatediff_models/`
- `context_length`: frames processed at once (usually 16)
- `context_overlap`: overlap between chunks (usually 4)
- `video_frames`: total frames (16-128 typical)
- `fps`: frames per second in output video (8-16 typical)

### Connection Pattern
```
CheckpointLoader -> model -> AnimateDiffLoader -> model -> KSampler
AnimateDiffLoader applies motion module to the diffusion model
KSampler output -> VAEDecode -> VHS_VideoCombine
```

## Stable Video Diffusion (SVD)

Image-to-video using Stability's SVD models.

### Required Nodes
- **ImageOnlyCheckpointLoader** — loads SVD model
- **SVD_img2vid_Conditioning** — creates conditioning from input image
- **VideoLinearCFGGuidance** — applies temporal guidance
- **KSampler** with specific settings for video

### Key Settings
- `width/height`: 1024x576 (SVD default) or 576x1024
- `motion_bucket_id`: 127 (default), higher = more motion
- `augmentation_level`: 0.0 (default), controls noise added to conditioning
- `fps`: 6 for SVD, 25 for SVD-XT

## Output Nodes
- **VHS_VideoCombine**: GIF, MP4, WebM output with fps control
- **SaveAnimatedWEBP**: WebP animation output
- Image sequence: use standard SaveImage with frame numbering
