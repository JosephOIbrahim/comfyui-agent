# ComfyUI Core Reference

## Workflow JSON Schema

A ComfyUI API-format workflow is a JSON object where:
- Top-level keys are string node IDs (usually numeric strings like "3", "12")
- Each node has `class_type` (str) and `inputs` (dict)
- Input values are either literals (int, float, str, bool) or connections
- Connections are 2-element arrays: [source_node_id_str, output_index_int]

## Type System

Strict types for node connections:
- MODEL — diffusion model (UNet or DiT)
- CLIP — text encoder
- VAE — variational autoencoder
- CONDITIONING — encoded prompt (NOT raw text)
- LATENT — latent space tensor
- IMAGE — pixel tensor [B,H,W,C] float32 range 0-1
- MASK — single channel [B,H,W]
- STRING, INT, FLOAT, BOOLEAN — primitives
- COMBO — dropdown selection (defined in INPUT_TYPES)

Types must match on connections. IMAGE cannot connect to LATENT input.

## Node Registration

Custom nodes export from __init__.py:
- NODE_CLASS_MAPPINGS: dict[str, class] — internal name → class
- NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] — internal name → UI name

Each node class needs:
- INPUT_TYPES (classmethod) → {"required": {...}, "optional": {...}}
- RETURN_TYPES: tuple of type strings
- RETURN_NAMES: tuple of output names
- FUNCTION: str — method name to call
- CATEGORY: str — menu path in UI

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | /object_info | All registered nodes with full interface |
| GET | /object_info/{node} | Single node info |
| POST | /prompt | Queue workflow execution |
| GET | /queue | Running + pending items |
| GET | /history | Past executions with outputs |
| GET | /history/{prompt_id} | Specific execution result |
| GET | /system_stats | GPU, VRAM, loaded models |
| GET | /embeddings | Available text embeddings |
| WS | /ws?clientId={id} | Real-time execution progress |

## Execution Model

ComfyUI executes workflows by:
1. Topological sort of the node graph
2. Execute each node in dependency order
3. Cache outputs — unchanged nodes skip execution
4. Report progress via WebSocket

## Common Model Directories

models/checkpoints/ — full diffusion models (SD, SDXL, Flux, etc.)
models/loras/ — LoRA adapters
models/vae/ — standalone VAE models
models/controlnet/ — ControlNet models
models/clip/ — standalone CLIP models
models/upscale_models/ — upscaler models
models/embeddings/ — textual inversions

## Key Patterns

- Seed is per-sampler-node, not global
- CONDITIONING must go through CLIPTextEncode (never raw string to sampler)
- Image tensors are [B,H,W,C] NOT [B,C,H,W] (ComfyUI convention)
- ControlNet requires resolution matching with the latent
- Empty latent size must be divisible by 8 (usually 64)
- VAE decode and encode are explicit separate nodes
