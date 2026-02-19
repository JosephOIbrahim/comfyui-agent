# 3D Generation Workflows in ComfyUI

## 3D Data Types

ComfyUI 3D workflows use these connection types between nodes:
- **MESH** -- 3D mesh geometry (vertices, faces, UVs). Exportable as GLB/OBJ.
- **VOXEL** -- 3D volume data. Used for volumetric effects and splat-to-mesh conversion.
- **POINT_CLOUD** -- 3D point cloud from Gaussian Splatting or NeRF reconstruction.
- **TRIPLANE** -- Triplane feature tensor. Intermediate representation (e.g. Hunyuan3D sampler output).
- **CAMERA** -- Camera position and rotation. Used by viewport and 3D control nodes.
- **POSE** -- Body or hand pose skeleton. Used by VNCCS, OpenPose, ControlNet.

These follow the same strict type matching as 2D types. MESH cannot connect to IMAGE inputs.

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

## Splat-to-Mesh Conversion

Common friction point: converting 3DGS/NeRF output to exportable mesh.

### Path A: ComfyUI-3D-Pack Marching Cubes (most established)
1. Load splat (Load3DGaussian or equivalent)
2. Marching cubes conversion -- resolution 256 (fast) to 512 (detailed)
3. Optional mesh cleanup
4. GLB export (SaveGLB)

Pitfalls: low resolution = blocky mesh; GLB may lose UVs; large splats need downsampling.

### Path B: Trellis2 Native Mesh
Some models output MESH directly -- no conversion needed.

### Path C: External Bridge
Export splat -> process in Blender/Houdini -> reimport. Best quality when time permits.

## Viewport & Control Tools

These are not agent-operated. DISCOVER should surface them when relevant.

### VNCCS (3D Pose Studio) -- @wildmindai
- Full character posing, lighting, multi-pose, camera control
- Use for: "pose a character", "set up ControlNet lighting"
- Most popular 3D tool in ComfyUI ecosystem (3,126+ likes)

### Action Director -- @wildmindai
- Interactive 3D viewport, loads FBX/GLB, batch-renders ControlNet passes
- Use for: "camera angle", "render depth map from 3D", "batch ControlNet"
- Complements VNCCS: posing vs. scene/camera control

### 3DView (Orion NodeFlow) -- @kakachiex
- 3D model viewer inside the ComfyUI node graph
- Use for: "preview 3D model", "view mesh in ComfyUI"
- Prototype stage

### vewd -- @spiritform
- 3D model loading and display
- Use for: "load 3D model", "import GLB"

## Tips
- Always check VRAM before loading 3D models -- they're typically larger than 2D
- 3D generation is slower than image generation -- set timeouts to 600s+
- Preview3D nodes may not work in headless mode -- use SaveGLB for exports
- For multi-stage pipelines (image -> 3D), use the pipeline tool to chain stages
