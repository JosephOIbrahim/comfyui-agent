# Common Workflow Recipes

## Minimal txt2img (SD1.5/SDXL)

Nodes needed (5 minimum):
1. **CheckpointLoaderSimple** — loads model+CLIP+VAE
2. **CLIPTextEncode** x2 — positive and negative prompts
3. **EmptyLatentImage** — sets resolution
4. **KSampler** — the denoiser
5. **VAEDecode** + **SaveImage** — decode and save

Connections:
- Checkpoint.MODEL -> KSampler.model
- Checkpoint.CLIP -> CLIPTextEncode (both).clip
- CLIPTextEncode (positive).CONDITIONING -> KSampler.positive
- CLIPTextEncode (negative).CONDITIONING -> KSampler.negative
- EmptyLatentImage.LATENT -> KSampler.latent_image
- KSampler.LATENT -> VAEDecode.samples
- Checkpoint.VAE -> VAEDecode.vae
- VAEDecode.IMAGE -> SaveImage.images

## img2img

Same as txt2img but replace EmptyLatentImage with:
1. **LoadImage** — loads the input image
2. **VAEEncode** — encodes image to latent space

Connections:
- LoadImage.IMAGE -> VAEEncode.pixels
- Checkpoint.VAE -> VAEEncode.vae
- VAEEncode.LATENT -> KSampler.latent_image
- KSampler `denoise` parameter: 0.3-0.8 (lower = closer to original)

## Inpainting

Requires:
1. **LoadImage** — base image
2. **LoadImage** (mask) — or use built-in mask editor
3. **VAEEncodeForInpaint** — encodes with mask
4. Use inpainting-specific checkpoint for best results

Key: set KSampler `denoise` to 1.0 for inpainting

## LoRA

Insert between CheckpointLoader and the rest:
1. **LoraLoader** — loads LoRA weights
- Input: model + clip from checkpoint
- Output: modified model + clip
- `strength_model` and `strength_clip`: 0.5-1.0

Multiple LoRAs: chain them (output of one -> input of next)

## Upscaling (Latent)

After KSampler, before VAEDecode:
1. **LatentUpscale** — upscale in latent space (fast, lower quality)
2. **KSampler** (second pass) — refine at higher resolution
- `denoise`: 0.3-0.5 for refinement

## Upscaling (Pixel)

After VAEDecode:
1. **UpscaleModelLoader** — loads upscale model (RealESRGAN, etc.)
2. **ImageUpscaleWithModel** — upscales the decoded image
