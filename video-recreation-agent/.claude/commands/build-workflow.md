# /project:build-workflow — ARCHITECT Expert

You are the **ARCHITECT** expert in the Video Recreation Agent Team.

Your job: Read the storyboard, query ComfyUI for available nodes, and construct valid workflow JSON files for each segment.

## Prerequisites
- `workspace/storyboard.json` must exist (run `/project:analyze-video` first)
- ComfyUI must be running at the configured host:port

## Steps

### 1. Load the storyboard
```bash
cat workspace/storyboard.json
```

### 2. Fetch available node schemas from ComfyUI
```bash
# This is the key move — gets EVERY node type with full input/output specs
curl -s http://${COMFYUI_HOST:-127.0.0.1}:${COMFYUI_PORT:-8188}/object_info | python3 -c "
import json, sys
data = json.load(sys.stdin)
# Save full schema for reference
with open('workspace/node_schemas.json', 'w') as f:
    json.dump(data, f, indent=2)
# Print available generation nodes
for name, info in data.items():
    cat = info.get('category', '')
    if any(k in cat.lower() for k in ['kling', 'video', 'image', 'load', 'save']):
        print(f'{name}: {cat}')
"
```

### 3. Check installed models
```bash
curl -s http://${COMFYUI_HOST:-127.0.0.1}:${COMFYUI_PORT:-8188}/object_info/CheckpointLoaderSimple | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('Available checkpoints:')
for m in data.get('CheckpointLoaderSimple', {}).get('input', {}).get('required', {}).get('ckpt_name', [[]])[0]:
    print(f'  • {m}')
"
```

### 4. For each segment, determine the workflow strategy

**Decision tree:**
```
Is it a CCTV/stylized shot?
  → Generate clean frame first (FLUX/Kling Image)
  → Apply style overlay in post (ffmpeg filter)

Is it a standard cinematic shot?
  → Use Kling 3.0 Omni with first+last frame anchoring

Does it need character consistency?
  → Use Kling Motion Control 3.0 with Element Binding

Is it a simple static shot?
  → Generate single image, Ken Burns effect in ffmpeg
```

### 5. Construct workflow JSON for each segment

Use the ComfyUI **API format** (not UI format). Rules:
- Node IDs are string numbers: "1", "2", "3"
- Connections reference: `["source_node_id", output_index]`
- Every workflow needs a terminal node (SaveImage/SaveVideo)
- Input images must be uploaded first via `/upload/image`

**Template: Kling 3.0 Image-to-Video with first/last frame:**
```json
{
  "1": {
    "class_type": "LoadImage",
    "inputs": {"image": "seg_{N}_first.png"}
  },
  "2": {
    "class_type": "LoadImage", 
    "inputs": {"image": "seg_{N}_last.png"}
  },
  "3": {
    "class_type": "KlingTextEncode",
    "inputs": {
      "prompt": "{segment_prompt}",
      "negative_prompt": "blurry, distorted, low quality"
    }
  },
  "4": {
    "class_type": "KlingVideo3",
    "inputs": {
      "first_frame": ["1", 0],
      "last_frame": ["2", 0],
      "prompt": ["3", 0],
      "duration": "{segment_duration}",
      "aspect_ratio": "9:16",
      "mode": "professional",
      "cfg_scale": 0.5
    }
  },
  "5": {
    "class_type": "SaveVideo",
    "inputs": {
      "video": ["4", 0],
      "filename_prefix": "seg_{N}"
    }
  }
}
```

**Template: FLUX image generation for keyframes:**
```json
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {"ckpt_name": "flux1-dev-fp8.safetensors"}
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "inputs": {"text": "{prompt}", "clip": ["1", 1]}
  },
  "3": {
    "class_type": "CLIPTextEncode",
    "inputs": {"text": "blurry, distorted", "clip": ["1", 1]}
  },
  "4": {
    "class_type": "EmptyLatentImage",
    "inputs": {"width": 1080, "height": 1920, "batch_size": 1}
  },
  "5": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["1", 0],
      "positive": ["2", 0],
      "negative": ["3", 0],
      "latent_image": ["4", 0],
      "seed": -1,
      "steps": 20,
      "cfg": 7.0,
      "sampler_name": "euler",
      "scheduler": "normal"
    }
  },
  "6": {
    "class_type": "VAEDecode",
    "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
  },
  "7": {
    "class_type": "SaveImage",
    "inputs": {"images": ["6", 0], "filename_prefix": "keyframe_{N}"}
  }
}
```

### 6. Validate each workflow
```bash
# Use the comfyui-agent validate tool if available, or dry-run
python3 -c "
import json
wf = json.load(open('workspace/workflows/seg_01.json'))
schemas = json.load(open('workspace/node_schemas.json'))
for nid, node in wf.items():
    ct = node['class_type']
    if ct not in schemas:
        print(f'ERROR: Node {nid} uses unknown class {ct}')
    else:
        print(f'OK: {nid} → {ct}')
"
```

### 7. Save all workflows
Save each to `workspace/workflows/seg_{N}_workflow.json`

Then show the build plan:
```
Built 6 workflows:
  seg_01: LoadImage → KlingVideo3 → SaveVideo (API, ~$0.40)
  seg_02: FLUX txt2img → KlingVideo3 → SaveVideo (Local+API, ~$0.40)
  ...
  Total estimated cost: $2.40
  Total estimated time: 8-12 minutes
```

Ask: **"Workflows ready. Proceed to generation? This will cost approximately $X."**
