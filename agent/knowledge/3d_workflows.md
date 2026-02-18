# 3D Generation Workflows in ComfyUI

## Hunyuan3D Pipeline

Hunyuan3D (Tencent) generates 3D meshes from text or images.

### Typical Node Chain
1. **LoadImage** or **CLIPTextEncode** -- input conditioning
2. **Hunyuan3DLoader** -- loads the Hunyuan3D model weights
3. **Hunyuan3DConditioner** -- prepares conditioning for 3D generation
4. **Hunyuan3DSampler** -- generates triplane features
5. **Hunyuan3DMeshDecoder** -- decodes triplane to 3D mesh
6. **SaveGLB** or **Preview3D** -- exports the result

### Key Parameters
- **steps**: 50-100 for quality, 20-30 for speed
- **guidance_scale**: 7.0-10.0 typical
- **resolution**: triplane resolution affects detail (256 = fast, 512 = detailed)

### VRAM Requirements
- 24GB+ VRAM recommended (RTX 4090 works)
- Reduce triplane resolution to 256 for lower VRAM cards

## 3D Gaussian Splatting (3DGS)

Gaussian Splatting creates photorealistic 3D scenes from multi-view images.

### Common Node Packs
- **ComfyUI-3D-Pack**: comprehensive 3D toolkit
- **ComfyUI-Hunyuan3D**: official Hunyuan3D nodes

### Output Formats
- `.glb` -- standard 3D interchange, viewable in browsers
- `.ply` -- point cloud format, used by Gaussian Splatting
- `.obj` -- classic mesh format with material support

## Video-to-3D with Wan2.1

Wan2.1 (Alibaba) supports video generation that can feed into 3D pipelines.

### Wan2.1 Variants
- **Wan-T2V**: Text-to-video
- **Wan-I2V**: Image-to-video
- **Wan-Fun**: Creative video effects

### Integration Pattern
Image -> Wan-I2V (multi-view) -> 3D reconstruction -> Mesh export

## Model Directory Conventions
- 3D models: `models/3d/` or `models/checkpoints/` (varies by node pack)
- Wan models: `models/diffusion_models/` or `models/checkpoints/`
- Some packs use HuggingFace-style directory downloads (multiple files per model)

## Tips
- Always check VRAM before loading 3D models -- they're typically larger than 2D
- 3D generation is slower than image generation -- set timeouts to 600s+
- Preview3D nodes may not work in headless mode -- use SaveGLB for exports
- For multi-stage pipelines (image -> 3D), use the pipeline tool to chain stages
