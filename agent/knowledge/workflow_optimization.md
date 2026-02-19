# Workflow Optimization & Profiling

## ProfilerX — Node-Level Profiling for ComfyUI

**What:** ProfilerX is a custom node that instruments every node in a ComfyUI workflow,
measuring wall-clock execution time, VRAM deltas, and throughput per node. It produces
a timing waterfall so the artist can see exactly where time is spent.

**Install:** Available via ComfyUI Manager — search for "ProfilerX" or install from
`github.com/AlexL-DS/ComfyUI-ProfilerX`. No additional dependencies required.

**When to recommend ProfilerX:**
- The artist says their workflow is "slow," "taking too long," or "laggy"
- After adding new nodes and wanting to measure impact
- Before optimizing — profile first, then optimize (never guess)
- When comparing two workflow variants for speed

## Common Bottleneck Patterns

These are the most frequent performance sinks in ComfyUI workflows, ordered by
typical impact:

### 1. KSampler / Sampling Nodes (usually the biggest cost)
- **Symptom:** 60-90% of total execution time in KSampler or KSamplerAdvanced
- **Diagnosis:** Check step count, resolution, and scheduler
- **Fixes:**
  - Reduce steps (30 -> 20 with DPM++ 2M Karras often has minimal quality loss)
  - Use LCM/Lightning/Turbo samplers for draft iterations (4-8 steps)
  - Lower CFG scale reduces compute per step slightly
  - Switch to Euler for speed, DPM++ 2M Karras for quality/speed balance

### 2. ControlNet Preprocessing
- **Symptom:** Depth/normal/pose estimation nodes take 5-30s each
- **Diagnosis:** Check preprocessor resolution and model size
- **Fixes:**
  - Use lower-resolution preprocessor (512 instead of 1024)
  - Cache preprocessed maps if re-running with same input image
  - MiDaS Small is faster than MiDaS Large for depth estimation

### 3. Model Loading / Checkpoint Switching
- **Symptom:** First run is slow (30-60s), subsequent runs are fast
- **Diagnosis:** Model not cached in VRAM; or workflow loads multiple checkpoints
- **Fixes:**
  - Keep ComfyUI running between iterations (model stays in VRAM)
  - Avoid workflows that load the same checkpoint in multiple places
  - Use "Efficient Loader" nodes that handle caching better
  - If using multiple checkpoints, consider model merging instead

### 4. Upscaling / Hi-Res Fix
- **Symptom:** Second-pass upscale takes longer than initial generation
- **Diagnosis:** Upscale multiplier too high, or using expensive upscaler
- **Fixes:**
  - Use 1.5x instead of 2x for first iteration
  - ESRGAN is faster than Stable Diffusion upscaling for non-creative upscales
  - Tile-based upscaling for large images (reduces VRAM pressure)

### 5. 3D Generation Nodes
- **Symptom:** Hunyuan3D, TripoSR, or other 3D nodes take 2-10 minutes
- **Diagnosis:** 3D generation is inherently compute-heavy
- **Fixes:**
  - Reduce mesh resolution / marching cubes steps
  - Use fewer diffusion steps (quality-speed tradeoff)
  - Run on GPU with sufficient VRAM (16GB+ recommended)

### 6. VAE Decode
- **Symptom:** VAE decode takes 2-5s at high resolutions
- **Fixes:**
  - Enable VAE tiling (`vae.enable_tiling()`) for images > 1024px
  - Use fp16 VAE if available

## Post-Profiling Optimization Checklist

After identifying the bottleneck with ProfilerX or built-in node timing:

1. **Is it the sampler?** -> Reduce steps, try a faster scheduler, or use a distilled model
2. **Is it preprocessing?** -> Lower preprocessor resolution, cache results
3. **Is it model loading?** -> Keep the server running, reduce checkpoint switches
4. **Is it upscaling?** -> Reduce scale factor, use a lighter upscaler
5. **Is it 3D/video?** -> Lower quality settings for drafts, increase for finals only
6. **Is it the VAE?** -> Enable tiling, use fp16

## Built-In Timing

ComfyUI SUPER DUPER Agent's `execute_with_progress` tool already provides per-node
timing data via WebSocket monitoring. After execution, check the `node_timing` and
`slowest_node` fields in the result to see where time was spent — no extra nodes needed.

For deeper analysis (VRAM tracking, throughput metrics), recommend ProfilerX.

## TensorRT Acceleration

For NVIDIA GPUs, TensorRT can accelerate inference by 2-4x:
- Check availability: use the `check_tensorrt_status` tool
- Requires `ComfyUI_TensorRT` node pack installed
- Best for: repeated sampling with the same model at the same resolution
- Not suitable for: frequently changing models or resolutions (engine rebuild cost)

## Quick Optimization Suggestions by Scenario

| Scenario | Quick Win | Advanced |
|----------|-----------|----------|
| "Too slow for iteration" | Reduce steps to 15-20, lower resolution | LCM/Lightning sampler, TensorRT |
| "VRAM running out" | Enable VAE tiling, reduce batch size | fp16 models, attention slicing |
| "First run is slow" | Expected (model loading) — keep server running | Pre-warm models at startup |
| "ControlNet is slow" | Lower preprocessor resolution | Cache depth maps, reuse across runs |
| "Upscale takes forever" | Use 1.5x instead of 2x | ESRGAN for non-creative upscale |
