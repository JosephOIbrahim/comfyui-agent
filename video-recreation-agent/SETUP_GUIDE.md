# Video Recreation Agent — Setup Guide
## From Zero to "Drop a URL, Get a Video" in 7 Steps

---

## The Big Picture

You have **two things** that need to talk to each other:

```
┌─────────────────────────┐         ┌──────────────────────────┐
│  YOUR COMFYUI-AGENT     │  HTTP   │  YOUR COMFYUI INSTALL    │
│  (the brain)            │ ◄─────► │  (the muscles)           │
│                         │  API    │                          │
│  Lives in its own repo  │         │  Lives in its own folder │
│  Runs via Claude Code   │         │  Runs its own server     │
│                         │         │                          │
│  Knows WHAT to do       │         │  Knows HOW to render     │
└─────────────────────────┘         └──────────────────────────┘
```

**The bridge:** Your agent talks to ComfyUI over HTTP (port 8188).  
**They don't need to be in the same folder.** They just need to be able to reach each other over the network.

This is the same way Jo Zhang's demo works — the agent is in a terminal, ComfyUI is running separately, and they communicate via API.

---

## Step 1: Make Sure You Have the Prerequisites

**Check these off:**

- [ ] **ComfyUI** installed and able to start (you already have this)
- [ ] **Claude Code** installed (`npm install -g @anthropic-ai/claude-code`)
- [ ] **ffmpeg** installed and on your PATH
  - Windows: `winget install ffmpeg` or download from ffmpeg.org
  - Mac: `brew install ffmpeg`
  - Test: `ffmpeg -version` should show version info
- [ ] **Python 3.10+** with pip
- [ ] **Your comfyui-agent repo** cloned and working

---

## Step 2: Copy the Agent Team Files Into Your Repo

Take the files from this package and drop them into your comfyui-agent repo:

```
your-comfyui-agent/
├── CLAUDE.md                          ← REPLACE with the new one (or merge)
├── .claude/
│   └── commands/
│       ├── analyze-video.md           ← NEW (copy these 5 files)
│       ├── build-workflow.md          ← NEW
│       ├── generate-shots.md          ← NEW
│       ├── assemble-montage.md        ← NEW
│       └── qa-compare.md             ← NEW
├── agent/
│   └── tools/
│       ├── comfyui_bridge.py          ← NEW (the bridge module)
│       ├── ... (your existing tools)
├── config/
│   └── bridge.env                     ← NEW (your connection config)
├── workspace/                         ← NEW (create this empty folder)
│   ├── reference/
│   ├── keyframes/
│   ├── workflows/
│   ├── outputs/
│   └── qa/
└── ... (your existing files)
```

**Important about the CLAUDE.md:**
Your existing CLAUDE.md has your agent's personality, tool definitions, and system prompt.
The new CLAUDE.md is the orchestrator for the video recreation team.

**Best approach:** Merge them. Add the video recreation sections to your existing CLAUDE.md
so Claude Code sees both your original agent capabilities AND the new expert team.

---

## Step 3: Configure the Bridge

Open `config/bridge.env` and set YOUR paths:

```bash
# Where ComfyUI lives on YOUR machine
# This is the folder that has main.py, models/, custom_nodes/
COMFYUI_PATH=G:/COMFYUI_Database

# ComfyUI server (usually localhost)
COMFYUI_HOST=127.0.0.1
COMFYUI_PORT=8188

# Where your comfyui-agent repo lives
AGENT_PATH=C:/Users/Joe/code/comfyui-agent

# Where all generated stuff goes
WORKSPACE=C:/Users/Joe/code/comfyui-agent/workspace
```

Then load it in your shell before running Claude Code:

**Windows PowerShell:**
```powershell
# Load the bridge config
Get-Content config/bridge.env | ForEach-Object {
    if ($_ -match '^([^#]\w+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}
```

**Mac/Linux bash:**
```bash
set -a; source config/bridge.env; set +a
```

---

## Step 4: Test the Bridge Connection

Start ComfyUI first (however you normally do it).

Then test the bridge:

```bash
cd /path/to/your/comfyui-agent
python agent/tools/comfyui_bridge.py
```

You should see:
```
✓ ComfyUI is running
  GPU: NVIDIA GeForce RTX 4090 (24GB VRAM)
  checkpoints: 12 installed
  loras: 45 installed
  vae: 3 installed
  Node categories: image, conditioning, sampling, ...
  Total node types: 847
```

**If you see "✗ ComfyUI is not running":**
- Make sure ComfyUI is actually running
- Check that the port matches (8188 is default)
- If you changed ComfyUI's port, update bridge.env

**If you see connection refused:**
- Make sure ComfyUI was started with `--listen` flag if you're accessing remotely
- For local access, default settings should work

---

## Step 5: Install Kling 3.0 Nodes in ComfyUI (for Video Gen)

The video recreation pipeline uses **Kling 3.0** for AI video generation.
These are "Partner Nodes" that run on Kling's cloud API (so your local GPU
handles image gen, Kling's cloud handles video gen).

**In ComfyUI:**
1. Open ComfyUI in your browser
2. Go to the Manager (or Template Library)
3. Search for "Kling 3.0"
4. Install the Kling partner nodes
5. You'll need a Kling API account — sign up at their site

**Alternative video models (if you don't want Kling):**
- **Wan 2.2** — open source, runs locally (needs lots of VRAM)
- **LTX-2** — partner node, runs on cloud
- **FramePack** — local, good for short clips

The workflow templates in the slash commands use Kling as default.
You can swap the node types to use whatever video model you prefer.

---

## Step 6: Launch Claude Code With the Agent Team

```bash
# Navigate to your comfyui-agent repo
cd /path/to/your/comfyui-agent

# Load bridge config
set -a; source config/bridge.env; set +a    # Mac/Linux
# OR use the PowerShell version above        # Windows

# Launch Claude Code
claude
```

Claude Code will automatically read:
- Your **CLAUDE.md** (the orchestrator + expert definitions)
- Your **.claude/commands/** (the slash commands for each expert)

You should now have these slash commands available:
```
/project:analyze-video
/project:build-workflow
/project:generate-shots
/project:assemble-montage
/project:qa-compare
```

---

## Step 7: Run Your First Video Recreation

**Option A: Full pipeline (one command)**
```
Hey Claude, recreate this video: [paste URL or local path]
```

Claude will run all 5 experts in sequence, checking in with you between each stage.

**Option B: Step by step (more control)**
```
/project:analyze-video https://example.com/cool-video.mp4
```
→ Review the storyboard
```
/project:build-workflow
```
→ Review the workflow plan and cost estimate
```
/project:generate-shots
```
→ Watch progress, wait for generation
```
/project:assemble-montage
```
→ Open the montage, check it
```
/project:qa-compare
```
→ See how it compares to the original

---

## How It All Connects — The Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR TERMINAL                             │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ CLAUDE CODE                                           │   │
│  │                                                       │   │
│  │  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐  │   │
│  │  │ ANALYST │→│ ARCHITECT│→│GENERATOR │→│ EDITOR  │  │   │
│  │  │         │ │          │ │          │ │         │  │   │
│  │  │ ffmpeg  │ │ JSON     │ │ HTTP API │ │ ffmpeg  │  │   │
│  │  │ vision  │ │ builder  │ │ websocket│ │ concat  │  │   │
│  │  └─────────┘ └──────────┘ └────┬─────┘ └─────────┘  │   │
│  │                                 │                     │   │
│  │       comfyui_bridge.py ────────┤                     │   │
│  │       (HTTP on port 8188)       │                     │   │
│  └─────────────────────────────────┼─────────────────────┘   │
│                                    │                          │
└────────────────────────────────────┼──────────────────────────┘
                                     │
                              ┌──────▼──────────────────────┐
                              │  COMFYUI SERVER              │
                              │  (separate process)          │
                              │                              │
                              │  /object_info → node schemas │
                              │  /prompt      → queue jobs   │
                              │  /history     → get results  │
                              │  /upload      → send images  │
                              │  /view        → get outputs  │
                              │  WebSocket    → live progress│
                              │                              │
                              │  ┌─── models/ ────────────┐  │
                              │  │ FLUX, Wan, LoRAs, etc. │  │
                              │  └────────────────────────┘  │
                              │  ┌─── custom_nodes/ ──────┐  │
                              │  │ Kling, LTX, etc.       │  │
                              │  └────────────────────────┘  │
                              └──────────────────────────────┘
```

**Key insight:** The agent and ComfyUI are **two separate processes**.
The agent lives in your comfyui-agent repo. ComfyUI lives in its own directory.
They talk over HTTP. The bridge module handles all the communication.

This is exactly how Jo Zhang's demo works. The agent doesn't need to be
inside the ComfyUI folder. It just needs network access to the API.

---

## Troubleshooting

**"ffmpeg not found"**
→ Install ffmpeg and make sure it's on your PATH.
→ Test: `ffmpeg -version`

**"ComfyUI not connected"**
→ Start ComfyUI first, then the agent.
→ Check port: default is 8188.
→ Test: open `http://127.0.0.1:8188` in your browser.

**"Node type KlingVideo3 not found"**
→ Install Kling partner nodes in ComfyUI.
→ Or: modify the workflow templates to use a different video model.

**"OOM / Out of memory during generation"**
→ The agent will auto-retry at lower resolution.
→ For local models: try `bridge.free_vram()` to clear GPU cache.
→ For API models (Kling): this shouldn't happen (runs on cloud).

**"Workflow validation errors"**
→ Run `python agent/tools/comfyui_bridge.py` to check what's installed.
→ The ARCHITECT expert validates before queuing. It will tell you what's missing.

**"Can't find my workspace files"**
→ Check WORKSPACE in bridge.env points to the right place.
→ All intermediate files go in workspace/ subfolders.

---

## What's Next

Once you have the basic pipeline working:

1. **Custom style profiles** — Save storyboard templates for styles you reuse
2. **Batch processing** — Drop multiple videos, process overnight
3. **LoRA integration** — Train character LoRAs for consistent subjects
4. **Audio generation** — Add ElevenLabs nodes for AI voiceover
5. **Your MCP bridge** — Connect this to your Cognitive Twin for session tracking

---

## File Inventory

| File | What It Does | Where It Goes |
|---|---|---|
| `CLAUDE.md` | MOE orchestrator + expert definitions | repo root |
| `.claude/commands/analyze-video.md` | Analyst expert | .claude/commands/ |
| `.claude/commands/build-workflow.md` | Architect expert | .claude/commands/ |
| `.claude/commands/generate-shots.md` | Generator expert | .claude/commands/ |
| `.claude/commands/assemble-montage.md` | Editor expert | .claude/commands/ |
| `.claude/commands/qa-compare.md` | QA expert | .claude/commands/ |
| `agent/tools/comfyui_bridge.py` | API bridge to ComfyUI | agent/tools/ |
| `config/bridge.env` | Connection config | config/ |
| `SETUP_GUIDE.md` | This file | repo root |
