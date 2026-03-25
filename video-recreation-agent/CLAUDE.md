# Video Recreation Agent Team — CLAUDE.md
# MOE (Mixture of Experts) Architecture for ComfyUI Agent
# v1.0.0 — Built for comfyui-agent integration

---

## Mission

Drop a video URL → get a fully recreated version using AI generation.
The agent team analyzes the reference, builds ComfyUI workflows from scratch,
generates all assets via API, assembles the final cut, and QA checks the result.

---

## Architecture: 5-Expert MOE Pipeline

```
INPUT: Video URL or local file
  │
  ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  ANALYST     │───▶│  ARCHITECT   │───▶│  GENERATOR   │
│  (Sees)      │    │  (Builds)    │    │  (Creates)   │
│              │    │              │    │              │
│ • Extract    │    │ • Read node  │    │ • Queue to   │
│   keyframes  │    │   schemas    │    │   ComfyUI    │
│ • Detect     │    │ • Construct  │    │ • Monitor    │
│   scenes     │    │   workflow   │    │   progress   │
│ • Build      │    │   JSON       │    │ • Collect    │
│   storyboard │    │ • Wire nodes │    │   outputs    │
└─────────────┘    └──────────────┘    └──────────────┘
                                              │
                                              ▼
                   ┌──────────────┐    ┌──────────────┐
                   │  QA          │◀───│  EDITOR      │
                   │  (Validates) │    │  (Assembles) │
                   │              │    │              │
                   │ • Compare    │    │ • ffmpeg     │
                   │   frames     │    │   concat     │
                   │ • Track      │    │ • Audio mix  │
                   │   poses      │    │ • Transitions│
                   │ • Score      │    │ • Export     │
                   └──────────────┘    └──────────────┘
                          │
                          ▼
                   OUTPUT: Recreated video + QA report
```

---

## Expert Routing (First Match Wins)

| User Intent | Expert | Slash Command |
|---|---|---|
| "Analyze this video" / "Break down this clip" | **ANALYST** | `/project:analyze-video` |
| "Build a workflow for..." / "Create the nodes" | **ARCHITECT** | `/project:build-workflow` |
| "Generate the shots" / "Run the workflow" | **GENERATOR** | `/project:generate-shots` |
| "Assemble the montage" / "Edit it together" | **EDITOR** | `/project:assemble-montage` |
| "Compare against reference" / "Check quality" | **QA** | `/project:qa-compare` |
| "Recreate this video" (full pipeline) | **DIRECTOR** | Runs all 5 in sequence |

---

## Environment Requirements

```bash
# This file tells Claude Code where everything lives
# Set in config/bridge.env or export before launching

COMFYUI_PATH=/path/to/ComfyUI              # Where ComfyUI is installed
COMFYUI_HOST=127.0.0.1                      # ComfyUI server address
COMFYUI_PORT=8188                           # ComfyUI server port
AGENT_PATH=/path/to/comfyui-agent           # Where THIS repo lives
WORKSPACE=/path/to/workspace                # Working directory for outputs
```

---

## Expert 1: ANALYST

**Role:** Deconstructs reference video into a structured storyboard.

**Tools used:**
- `ffmpeg` — keyframe extraction, scene detection
- `ffprobe` — video metadata (duration, fps, resolution)
- Claude Vision — shot description, camera analysis, style identification

**Process:**
1. Download video (if URL) or load from local path
2. Run `ffprobe` to get metadata (duration, fps, codec, resolution)
3. Run scene detection: `ffmpeg -i input.mp4 -vf "select='gt(scene,0.3)',showinfo" -vsync vfr frame_%04d.png`
4. For each detected scene boundary, extract the keyframe
5. Send keyframes to Claude Vision with analysis prompt
6. Produce structured storyboard JSON:

```json
{
  "source": {"url": "...", "duration": 8.3, "fps": 24, "resolution": "1080x1920"},
  "segments": [
    {
      "id": 1,
      "start": 0.0,
      "end": 2.0,
      "duration": "2s",
      "shot": "Woman enters elevator",
      "camera": "CCTV fisheye overhead",
      "style": "Scanlines, timestamp overlay, green tint",
      "mood": "Surveillance, voyeuristic",
      "keyframe_path": "workspace/keyframes/seg_01.png",
      "first_frame": null,
      "last_frame": null
    }
  ],
  "overall_style": "Elevator romance short film, alternating CCTV and cinematic",
  "color_palette": ["#1a3a2a", "#c4d4c4", "#f0f0f0"],
  "audio_notes": "Ambient hum, no dialogue"
}
```

**Key file:** `agent/tools/video_analyzer.py`

---

## Expert 2: ARCHITECT

**Role:** Constructs ComfyUI workflow JSON programmatically from node schemas.

**Tools used:**
- ComfyUI `/object_info` API — fetches all available node types + their input/output schemas
- ComfyUI `/history` API — reference successful past executions
- Python JSON construction — builds workflow from scratch

**Process:**
1. Fetch node schemas from ComfyUI: `GET http://{host}:{port}/object_info`
2. For each storyboard segment, determine the generation strategy:
   - **Image keyframes:** FLUX/Kling Image 3.0 for first-frame/last-frame pairs
   - **Video segments:** Kling 3.0 Omni with first/last frame anchors
   - **Style transfer:** ControlNet or reference image conditioning
3. Construct workflow JSON with proper node wiring:
   - Each node gets a unique string ID
   - Inputs reference other nodes as `["node_id", output_index]`
   - API format (not UI format)
4. Validate against installed nodes before submission
5. Save workflow to `workflows/` directory

**Node wiring pattern (API format):**
```json
{
  "1": {
    "class_type": "LoadImage",
    "inputs": {"image": "keyframe_01.png"}
  },
  "2": {
    "class_type": "KlingVideo3",
    "inputs": {
      "first_frame": ["1", 0],
      "prompt": "Cinematic elevator scene...",
      "duration": 2,
      "aspect_ratio": "9:16"
    }
  },
  "3": {
    "class_type": "SaveVideo",
    "inputs": {"video": ["2", 0], "filename_prefix": "seg_01"}
  }
}
```

**Key file:** `agent/tools/workflow_builder.py`

---

## Expert 3: GENERATOR

**Role:** Manages workflow execution — queuing, monitoring, parallel runs, output collection.

**Tools used:**
- ComfyUI `/prompt` API — queue workflows
- ComfyUI WebSocket — real-time progress monitoring
- Parallel execution manager for API-based models (Kling, LTX, etc.)

**Process:**
1. Load built workflow JSON
2. Upload any required input images to ComfyUI: `POST /upload/image`
3. Queue workflow: `POST /prompt` with `{"prompt": workflow_json}`
4. Monitor via WebSocket for progress events
5. On completion, collect output files from ComfyUI output directory
6. Copy outputs to workspace with structured naming
7. For parallel API models: fire all segments simultaneously, collect as they complete

**Progress tracking pattern:**
```
Segment 1/6: ████████░░ 80% — KSampler step 16/20
Segment 2/6: ██████████ 100% — Complete ✓
Segment 3/6: ░░░░░░░░░░ Queued
...
```

**Key file:** Existing `agent/tools/` execution tools (enhanced with parallel support)

---

## Expert 4: EDITOR

**Role:** Assembles generated clips into final montage using ffmpeg.

**Tools used:**
- `ffmpeg` — concatenation, transitions, audio mixing, encoding
- `ffprobe` — verify clip properties before assembly

**Process:**
1. Verify all segment clips exist and have matching properties
2. Build ffmpeg filter graph:
   - Concatenate clips in storyboard order
   - Apply transitions (crossfade, hard cut) per segment specification
   - Mix reference audio (if available) or generate silence
   - Apply style overlays (scanlines, timestamps for CCTV segments)
3. Encode final output:
   - `libx264` with `crf 18` for quality
   - Match reference fps and resolution
   - AAC audio at 192k
4. Verify output duration matches reference (±0.5s tolerance)

**Assembly command pattern:**
```bash
ffmpeg -i seg_01.mp4 -i seg_02.mp4 -i seg_03.mp4 \
  -filter_complex "[0:v][1:v][2:v]concat=n=3:v=1:a=0[outv]" \
  -map "[outv]" -c:v libx264 -preset medium -crf 18 \
  output_montage.mp4
```

**Key file:** `agent/tools/video_assembler.py`

---

## Expert 5: QA

**Role:** Compares recreation against reference, produces quality report.

**Tools used:**
- Claude Vision — side-by-side frame comparison
- `ffmpeg` — extract comparison frames at matching timestamps
- Python — SSIM/perceptual hash scoring

**Process:**
1. Extract frames from both reference and recreation at matching timestamps
2. Create side-by-side comparison images (Original | Recreation | Overlay)
3. For each comparison pair, analyze:
   - **Composition match** — subject placement, framing
   - **Style match** — color grade, lighting, camera angle
   - **Motion flow** — body pose tracking, movement direction
   - **Temporal match** — scene beats land at same timestamps
4. Generate QA report:

```
╔══════════════════════════════════════════════╗
║  QA REPORT: elevator_montage_v1.mp4         ║
╠══════════════════════════════════════════════╣
║  Overall Score:  7.2/10                      ║
║  Duration Match: 8.3s vs 8.3s ✓             ║
║  Scene Count:    6/6 ✓                       ║
║                                              ║
║  Per-Segment:                                ║
║  1. CCTV Entry    — 8/10 (good fisheye)     ║
║  2. Couple Stand  — 6/10 (pose differs)     ║
║  3. Kiss Overlay  — 7/10 (bbox accurate)    ║
║  4. Kiss Close    — 8/10 (strong match)     ║
║  5. Slit View     — 6/10 (door timing off)  ║
║  6. Man Alone     — 8/10 (good CCTV style)  ║
║                                              ║
║  Recommendations:                            ║
║  • Re-run seg 2 with stronger pose ref      ║
║  • Adjust seg 5 door animation timing       ║
╚══════════════════════════════════════════════╝
```

**Key file:** `agent/tools/visual_qa.py`

---

## Full Pipeline: DIRECTOR Mode

When user says "recreate this video" or runs the full pipeline:

```
Step 1: ANALYST   → Produces storyboard.json
Step 2: ARCHITECT → Produces workflow JSONs per segment
Step 3: GENERATOR → Produces video clips per segment
Step 4: EDITOR    → Produces assembled montage
Step 5: QA        → Produces comparison report
```

**Between each step, the Director:**
- Shows progress with mile markers
- Saves intermediate state (can resume if interrupted)
- Asks for approval before expensive generation steps
- Provides estimated time + API cost

**State file:** `workspace/pipeline_state.json`
```json
{
  "project": "elevator_recreation",
  "stage": "GENERATOR",
  "completed": ["ANALYST", "ARCHITECT"],
  "pending": ["GENERATOR", "EDITOR", "QA"],
  "storyboard": "workspace/storyboard.json",
  "workflows": ["workspace/workflows/seg_01.json", "..."],
  "outputs": {},
  "started": "2026-03-24T19:00:00Z",
  "api_cost_estimate": "$2.40"
}
```

---

## Memory & State

This agent team writes state files that persist between sessions:

| File | Purpose | Updated By |
|---|---|---|
| `workspace/storyboard.json` | Shot breakdown | ANALYST |
| `workspace/workflows/*.json` | ComfyUI workflow per segment | ARCHITECT |
| `workspace/outputs/*.mp4` | Generated clips | GENERATOR |
| `workspace/montage_v*.mp4` | Assembled versions | EDITOR |
| `workspace/qa_report.json` | Quality analysis | QA |
| `workspace/pipeline_state.json` | Pipeline progress | DIRECTOR |

---

## ComfyUI Bridge

The agent team needs access to ComfyUI even though it lives in a separate directory.

**Connection method:** HTTP API + filesystem bridge

```python
# The bridge reads from config/bridge.env and provides:
# 1. API access to ComfyUI (queue, monitor, fetch results)
# 2. Node schema access via /object_info
# 3. File transfer (upload inputs, download outputs)
# 4. Model inventory (what's installed)
```

See `agent/tools/comfyui_bridge.py` for the implementation.

---

## Error Recovery

| Failure | Recovery |
|---|---|
| ComfyUI not running | Print connection instructions, wait for retry |
| Generation fails (OOM) | Reduce resolution, retry with lower settings |
| API timeout (Kling/cloud) | Retry with exponential backoff (3 attempts) |
| Scene detection fails | Fall back to uniform time splits |
| Assembly mismatch | Re-probe clips, adjust filter graph |
| QA score < 5/10 | Suggest re-generation of lowest-scoring segments |

---

## Usage Examples

```bash
# Full pipeline — drop a URL, get a recreation
claude "Recreate this video: https://example.com/elevator.mp4"

# Step by step with slash commands
/project:analyze-video https://example.com/elevator.mp4
/project:build-workflow                    # uses storyboard from analysis
/project:generate-shots                    # queues to ComfyUI
/project:assemble-montage                  # ffmpeg assembly
/project:qa-compare                        # compare against reference

# Resume interrupted pipeline
claude "Resume the elevator recreation — we left off at generation"
```
