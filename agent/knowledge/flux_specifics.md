# Flux Model Specifics

## Architecture Differences

Flux uses a DiT (Diffusion Transformer) architecture, different from UNet-based SD/SDXL.

### Loading
- **UNETLoader** or **CheckpointLoaderSimple** — loads the Flux model
- Flux uses dual CLIP encoders: **CLIPLoader** with `clip_type="flux"`
- VAE is separate — use standard **VAELoader**
- Flux dev/schnell models are in `models/diffusion_models/` or `models/unet/`

### Key Differences from SD/SDXL
- **No negative prompt** — Flux ignores negative conditioning. Set negative to empty or don't connect
- **CFG scale**: Use 1.0 for Flux (guidance is built into the model via FluxGuidance)
- **FluxGuidance** node: Controls generation strength (3.5 typical for dev, 0.0 for schnell)
- **Sampler**: `euler` works well; `uni_pc` and `dpmpp_2m` also supported
- **Scheduler**: `simple` or `normal` (not karras)
- **Steps**: 20-28 for dev, 1-4 for schnell

### Resolution
- Native: 1024x1024 (1 megapixel)
- Supports non-square: any resolution that's ~1MP total
- Must be divisible by 16 (not just 8 like SD)

### LoRA
- Flux LoRAs use a different format — not compatible with SD/SDXL LoRAs
- Load with standard **LoraLoader**, connects to both MODEL and CLIP

### Common Pipeline
```
UNETLoader -> FluxGuidance -> KSampler.model
CLIPLoader (flux) -> CLIPTextEncode -> KSampler.positive
EmptyLatentImage (1024x1024) -> KSampler.latent_image
KSampler -> VAEDecode -> SaveImage
```
