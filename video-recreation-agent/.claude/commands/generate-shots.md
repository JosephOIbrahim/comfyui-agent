# /project:generate-shots — GENERATOR Expert

You are the **GENERATOR** expert in the Video Recreation Agent Team.

Your job: Execute the built workflows against ComfyUI, monitor progress, collect outputs.

## Prerequisites
- Workflow JSONs in `workspace/workflows/`
- ComfyUI running and accessible
- Any required input images uploaded

## Steps

### 1. Upload input images to ComfyUI
```bash
# For each keyframe that workflows reference
for img in workspace/keyframes/seg_*_first.png workspace/keyframes/seg_*_last.png; do
  filename=$(basename "$img")
  curl -s -X POST "http://${COMFYUI_HOST:-127.0.0.1}:${COMFYUI_PORT:-8188}/upload/image" \
    -F "image=@${img}" \
    -F "subfolder=agent_inputs" \
    -F "type=input"
  echo "Uploaded: $filename"
done
```

### 2. Queue workflows to ComfyUI

**For API-based models (Kling, LTX — cloud GPU):**
Queue ALL segments in parallel. They don't compete for local VRAM.

**For local models (FLUX, Wan — local GPU):**
Queue one at a time to avoid OOM.

```bash
# Queue a single workflow
python3 -c "
import json, urllib.request

workflow = json.load(open('workspace/workflows/seg_01_workflow.json'))
payload = json.dumps({
    'prompt': workflow,
    'client_id': 'video-recreation-agent'
}).encode()

req = urllib.request.Request(
    'http://${COMFYUI_HOST:-127.0.0.1}:${COMFYUI_PORT:-8188}/prompt',
    data=payload,
    headers={'Content-Type': 'application/json'}
)
resp = urllib.request.urlopen(req)
result = json.loads(resp.read())
print(f'Queued: prompt_id={result[\"prompt_id\"]}')
"
```

### 3. Monitor progress via WebSocket
```python
# monitor_progress.py — run as background task
import websocket, json

ws = websocket.WebSocket()
ws.connect(f"ws://{COMFYUI_HOST}:{COMFYUI_PORT}/ws?clientId=video-recreation-agent")

while True:
    msg = json.loads(ws.recv())
    if msg['type'] == 'progress':
        d = msg['data']
        pct = int(d['value'] / d['max'] * 100)
        print(f"Step {d['value']}/{d['max']} ({pct}%)")
    elif msg['type'] == 'executed':
        print(f"Node complete: {msg['data']['node']}")
    elif msg['type'] == 'execution_complete':
        print("DONE")
        break
```

### 4. Collect outputs
```bash
# Check what was generated
curl -s "http://${COMFYUI_HOST:-127.0.0.1}:${COMFYUI_PORT:-8188}/history/${PROMPT_ID}" | python3 -c "
import json, sys
history = json.load(sys.stdin)
for prompt_id, data in history.items():
    outputs = data.get('outputs', {})
    for node_id, output in outputs.items():
        for img in output.get('images', []):
            print(f'Image: {img[\"filename\"]} ({img[\"subfolder\"]})')
        for vid in output.get('videos', []):
            print(f'Video: {vid[\"filename\"]} ({vid[\"subfolder\"]})')
"
```

### 5. Download outputs to workspace
```bash
# Download each output file
mkdir -p workspace/outputs
curl -o workspace/outputs/seg_01.mp4 \
  "http://${COMFYUI_HOST:-127.0.0.1}:${COMFYUI_PORT:-8188}/view?filename=seg_01_00001.mp4&subfolder=&type=output"
```

### 6. Show progress dashboard
```
╔══════════════════════════════════════════════╗
║  GENERATION PROGRESS                         ║
╠══════════════════════════════════════════════╣
║  Seg 1: ██████████ 100%  ✓  (2.1s clip)    ║
║  Seg 2: ████████░░  80%  ⟳  KSampler 16/20 ║
║  Seg 3: ██████████ 100%  ✓  (2.0s clip)    ║
║  Seg 4: ░░░░░░░░░░   0%  ⏳ Queued          ║
║  Seg 5: ░░░░░░░░░░   0%  ⏳ Queued          ║
║  Seg 6: ░░░░░░░░░░   0%  ⏳ Queued          ║
║                                              ║
║  Elapsed: 4m 12s  |  Est. remaining: 6m     ║
║  API cost so far: $1.20                      ║
╚══════════════════════════════════════════════╝
```

When all complete: **"All 6 segments generated. Ready to assemble montage?"**
