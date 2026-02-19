# 3D Generation — Partner Nodes & Comparative Guide

## Partner Nodes (Officially Supported)

These are maintained by both Comfy-Org and the provider. They follow ComfyUI
coding standards and are less likely to break on updates. Prefer these over
equivalent community alternatives.

### Hunyuan 3D 3.0 (Tencent)
- Input: text, image, or sketch
- Output: production-quality 3D mesh (GLB)
- Best for: final production assets, detailed structures
- VRAM: 24GB+ recommended
- Strengths: highest quality, multi-input support, texture editing coming

### Meshy 6 (Meshy)
- Input: text or image
- Output: stylized 3D mesh
- Best for: game-ready stylized assets, fast generation
- Strengths: speed + style consistency, good for game art pipelines

### Tripo v3.0 (VAST AI Research)
- Input: text or image
- Output: 3D mesh
- Best for: rapid prototyping, quick iterations
- Strengths: fastest generation time, good for exploration

### Rodin 3D Gen-2 (Deemos)
- Input: text or image
- Output: high-detail 3D mesh
- Best for: realistic/detailed models, architectural assets
- Strengths: highest geometric fidelity

## Community Alternatives

### Trellis2 (Open Source)
- Open-source mesh generation
- Best for: experimentation, custom pipelines, research
- Note: may output mesh directly without conversion step

### ComfyUI-3D-Pack (3,641+ stars)
- Comprehensive 3DGS/NeRF/mesh toolkit
- Best for: splat-to-mesh conversion, NeRF workflows, Gaussian Splatting
- Includes marching cubes, point cloud processing, GLB/PLY/OBJ export

## Quick Comparison

When an artist asks "what should I use for 3D generation?":

| Need | Recommend |
|------|-----------|
| Production-quality assets | Hunyuan 3D (Partner) |
| Game-ready stylized | Meshy (Partner) |
| Quick prototyping | Tripo (Partner) |
| Maximum geometric detail | Rodin (Partner) |
| Open-source / research | Trellis2 (Community) |
| Splat/NeRF conversion | ComfyUI-3D-Pack (Community) |
| Architectural structures | Hunyuan 3D or Rodin |
| Character models | Hunyuan 3D or Meshy |

## Splat-to-Mesh Conversion Paths

Common friction point: artists have a 3D Gaussian Splat and need a clean mesh.

### Path A: ComfyUI-3D-Pack Marching Cubes (most established)
1. Load splat data (Load3DGaussian or equivalent)
2. Marching cubes conversion (adjust resolution for quality vs speed)
3. Optional mesh cleanup
4. GLB/OBJ export (SaveGLB)

Pitfalls:
- Low marching cubes resolution = blocky mesh (try 256-512)
- GLB export may lose UV coordinates without proper unwrapping
- Large splats may need downsampling before conversion

### Path B: Trellis2 Native Mesh Output
- Some newer models output mesh directly, no conversion needed
- Check if the generation model already outputs MESH type

### Path C: External Tool Bridge
- Export splat from ComfyUI
- Process in Blender or Houdini (remesh, UV unwrap, cleanup)
- Reimport cleaned mesh
- Best for highest quality when time permits
