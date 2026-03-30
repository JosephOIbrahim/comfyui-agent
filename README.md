# ComfyUI Agent

AI co-pilot for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) -- 108 tools that understand your workflows and help you work faster through natural conversation.

Instead of manually editing JSON, hunting for node packs, or debugging broken workflows yourself, just describe what you want. The agent inspects, repairs, reconfigures, and executes -- including installing missing nodes, downloading models, and fixing broken model references.

## What It Does

- **"What models do I have?"** -- scans your installation instantly
- **"Load this workflow and change the seed to 42"** -- reads, modifies, and saves with undo support
- **"Repair this workflow"** -- one-shot: detects missing nodes, finds the packs, installs them all
- **"Reconfigure for my local models"** -- scans model references, finds missing files, substitutes closest local match
- **"Download the LTX-2 FP8 checkpoint"** -- downloads models directly to the correct directory
- **"Run this with 30 steps instead of 20"** -- patches the workflow and queues it to ComfyUI
- **"Find me a good anime LoRA"** -- searches local catalog, ComfyUI Manager registry, HuggingFace, and CivitAI
- **"Is this model compatible with my workflow?"** -- checks SD1.5/SDXL/Flux/SD3/LTX-2/WAN family compatibility
- **"Analyze this output -- why does it look wrong?"** -- uses Claude Vision to diagnose image issues
- **"Remember that I prefer SDXL for landscapes"** -- saves notes and learns from your outcomes over time

The agent talks to ComfyUI's API directly. It reads your actual installation, sees what's really installed, and works with your real workflows -- including component/subgraph workflows with nested node graphs.

## Installation

You'll need:

- **Python 3.10 or newer** (check with `python --version`)
- **ComfyUI** running on your machine (the agent talks to it over HTTP)
- **An Anthropic API key** (sign up at [console.anthropic.com](https://console.anthropic.com/))

### Step 1: Download

```bash
git clone https://github.com/JosephOIbrahim/comfyui-agent.git
cd comfyui-agent
```

### Step 2: Install

```bash
pip install -e .
```

For development (includes test tools):

```bash
pip install -e ".[dev]"
```

### Step 3: Configure

```bash
cp .env.example .env
```

Open `.env` in any text editor and add your API key:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

If your ComfyUI database is somewhere other than `G:/COMFYUI_Database`, also set:

```
COMFYUI_DATABASE=/path/to/your/comfyui/database
```

If your ComfyUI installation is in a separate directory (e.g., `G:/COMFY/ComfyUI`), the agent auto-detects it on Windows. Override with:

```
COMFYUI_INSTALL_DIR=/path/to/ComfyUI
```

### Step 4: Run

Make sure ComfyUI is running first, then:

```bash
agent run
```

That's it. Type what you want in plain English. Type `quit` to exit.

## Commands

### Interactive session

```bash
agent run                             # Start chatting with the agent
agent run --session my-project        # Auto-save your progress to resume later
agent run --verbose                   # Show what's happening under the hood
```

### Offline tools (no API key needed)

```bash
agent inspect                         # See what models and nodes you have installed
agent parse workflow.json             # Analyze a workflow file
agent sessions                        # See your saved sessions
```

### Search

```bash
agent search "controlnet" --nodes           # Find node packs
agent search "KSampler" --node-type         # Which pack provides this node?
agent search "sdxl" --models                # Search model registry
agent search "flux lora" --hf              # Search HuggingFace
agent search "anime" --models --type lora   # Filter by model type
```

### Orchestration

```bash
agent orchestrate workflow.json             # Load > validate > execute > verify pipeline
agent autoresearch "flux lora"              # Multi-source model/node discovery
agent autoresearch --program program.md     # FORESIGHT autoresearch pipeline
```

## How It Works

The agent uses Claude (Anthropic's AI) with 108 specialized tools across three tiers:

**Intelligence Layer (58 tools)**

| Layer | Tools | What they do |
|-------|-------|-------------|
| **UNDERSTAND** | 13 | Parse workflows (including component/subgraph format), scan models/nodes, query ComfyUI API, detect format |
| **DISCOVER** | 15 | Search local catalog + ComfyUI Manager (31k+ nodes) + HuggingFace + CivitAI, model compatibility, install instructions, GitHub releases |
| **PILOT** | 16 | RFC6902 patch engine with undo, semantic node ops (AUTOGROW_V3 support), session persistence, pipeline execution |
| **PROVISION** | 5 | Install node packs (git clone), download models (httpx), disable node packs, one-shot workflow repair, model reference reconfiguration |
| **VERIFY** | 9 | Validate, execute, WebSocket progress monitoring, post-execution verification, creative metadata embedding |

**Brain Layer (27 tools)**

| Module | Tools | What they do |
|--------|-------|-------------|
| **Vision** | 4 | Analyze generated images, A/B comparison, perceptual hashing |
| **Planner** | 4 | Goal decomposition, step tracking, replanning |
| **Memory** | 4 | Outcome learning with temporal decay, cross-session patterns |
| **Orchestrator** | 2 | Parallel sub-tasks with filtered tool access |
| **Optimizer** | 4 | GPU profiling, TensorRT detection, auto-apply optimizations |
| **Demo** | 2 | Guided walkthroughs for streams and podcasts |
| **Intent** | 4 | Artistic intent capture, MoE pipeline with iterative refinement |
| **Iteration** | 3 | Refinement journey tracking across generation cycles |

**Stage Layer (23 tools)**

| Module | Tools | What they do |
|--------|-------|-------------|
| **Provisioner** | 3 | USD-native model provisioning with download/verify lifecycle |
| **Stage** | 6 | Cognitive state read/write with delta composition and rollback |
| **FORESIGHT** | 5 | Predictive experiment planning, experience recording, counterfactuals |
| **Compositor** | 4 | USD scene composition, validation, conditioning extraction |
| **Hyperagent** | 5 | Meta-layer self-improvement proposals, calibration tracking |

When you ask a question, Claude decides which tools to use, calls them, reads the results, and responds. It streams text as it thinks, so you're never staring at a blank screen.

## Model Profiles

The agent ships with model-specific profiles that encode real behavioral knowledge:

| Profile | Architecture | Key Insight |
|---------|-------------|-------------|
| **Flux.1 Dev** | DiT | CFG 2.5-4.5, T5-XXL encoder, negative prompts near-useless |
| **SDXL** | UNet | CFG 5-9, CLIP encoder, tag-based prompts, LoRA ecosystem |
| **SD 1.5** | UNet | CFG 7-12, 512x512 native, massive LoRA support |
| **LTX-2** | DiT (video) | CFG ~25, 121 steps, Gemma-3 encoder, frame count must be (N*8)+1 |
| **WAN 2.x** | UNet (video) | CFG 1-3.5, 4-20 steps, dual-noise architecture, CLIP encoder |
| **Video (fallback)** | UNet | Conservative defaults for unknown video models |

Each profile has three sections consumed by different agents:
- **prompt_engineering** (Intent Agent) -- how to write prompts for this model
- **parameter_space** (Execution Agent) -- correct CFG, steps, sampler, resolution ranges
- **quality_signatures** (Verify Agent) -- how to judge output quality and suggest fixes

## Workflow Sources

The agent can load workflows from three locations:

| Source | Path | Description |
|--------|------|-------------|
| **Built-in templates** | `agent/templates/` | Starter workflows (txt2img, img2img, LoRA) |
| **User workflows** | `COMFYUI_Database/Workflows/` | Your saved workflow library |
| **ComfyUI blueprints** | `ComfyUI/blueprints/` | Built-in ComfyUI workflow blueprints |

Use `list_workflow_templates` to see all available workflows across sources.

## Component Workflow Support

ComfyUI 0.16+ introduced component nodes -- workflows-within-workflows where a single node on the canvas contains an entire subgraph internally. The agent handles these natively:

- **Detects** component instance nodes (UUID-style class types)
- **Parses** `definitions.subgraphs` to inspect internal node graphs
- **Validates** nodes inside components (catches missing nodes in subgraphs)
- **Supports** `COMFY_AUTOGROW_V3` dynamic inputs (dotted names like `values.a`)

## SuperDuper Panel (ComfyUI Sidebar)

When loaded as a ComfyUI extension, the agent provides an in-app AI sidebar:

- **Chat interface** with real-time streaming responses
- **Workflow-aware** -- automatically reads the current canvas
- **Missing nodes detection** with one-click install panels
- **Direct execution** -- Repair, Validate, Install, Download bypass Claude API for instant results
- **Agent dispatch cards** showing which intelligence layer is active (ROUTER / INTENT / EXECUTION / VERIFY)
- **Quick action chips** -- Run, Validate, Repair, Optimize with one click
- **Node interaction pills** -- click to highlight nodes on the canvas
- **Environment awareness** -- receives full installation snapshot (resolved paths, model counts, node packs) on connect

### Repair Flow

When you load a workflow with missing nodes, SuperDuper:
1. Detects missing nodes (including inside component subgraphs)
2. Shows a repair panel listing each missing type and its source pack
3. One click installs all required packs via `git clone`
4. Reports which packs succeeded and if a restart is needed

### Reconfigure Flow

When a workflow references models you don't have:
1. Scans all model references (checkpoints, LoRAs, VAEs, ControlNets)
2. Checks which files exist locally
3. Fuzzy-matches the closest local alternative (stem/word overlap scoring)
4. Applies substitutions automatically or shows a report

## MCP Server (Primary Interface)

All 108 tools are available via [Model Context Protocol](https://modelcontextprotocol.io/) for integration with Claude Code, Claude Desktop, or other MCP clients:

```bash
agent mcp
```

Configure in Claude Code / Claude Desktop:

```json
{
  "mcpServers": {
    "comfyui-agent": {
      "command": "agent",
      "args": ["mcp"]
    }
  }
}
```

## Configuration

All settings go in your `.env` file:

| Setting | Default | What it does |
|---------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Your API key for Claude |
| `COMFYUI_HOST` | `127.0.0.1` | Where ComfyUI is running |
| `COMFYUI_PORT` | `8188` | ComfyUI port |
| `COMFYUI_DATABASE` | `G:/COMFYUI_Database` | Your ComfyUI database folder (models, nodes, workflows) |
| `COMFYUI_INSTALL_DIR` | auto-detected | ComfyUI installation directory (if separate from database) |
| `COMFYUI_OUTPUT_DIR` | auto-detected | Where ComfyUI saves generated images |
| `AGENT_MODEL` | `claude-sonnet-4-20250514` | Which Claude model to use (CLI mode only -- MCP inherits from Claude Code) |

## Session Memory

Sessions let the agent remember your workflow state and notes between conversations:

```bash
# Start a session
agent run --session portrait-project

# The agent remembers your loaded workflow, patches, and notes
# Next time, just load the same session:
agent run --session portrait-project
```

Sessions are saved as JSON files in the `sessions/` folder. You can list them with `agent sessions`.

## Workflow Formats

ComfyUI exports workflows in different formats. The agent handles all of them:

- **API format** -- the JSON you get from "Save (API Format)" in ComfyUI. Full node data with inputs and connections. Best for the agent.
- **UI format with API data** -- the default "Save" export. Contains visual layout plus embedded API data. Agent extracts what it needs.
- **UI-only format** -- older exports with only visual layout. The agent can read the structure but can't modify or execute these.
- **Component format** -- workflows containing subgraph definitions (`definitions.subgraphs`). The agent parses both the top-level instances and the internal node graphs.

## Troubleshooting

**"ANTHROPIC_API_KEY not set"** -- Make sure your `.env` file exists and has the key. Run from the `comfyui-agent` directory.

**"Could not connect to ComfyUI"** -- Start ComfyUI first. The agent needs it running to inspect nodes, execute workflows, and validate changes.

**"Node type not found"** -- The workflow uses a custom node that isn't installed. Ask the agent: "repair this workflow" and it will detect, locate, and install the required packs in one shot.

**Unicode/encoding crash on Windows** -- If you see `UnicodeEncodeError` with Rich, ensure you're running in a terminal that supports UTF-8 (Windows Terminal recommended).

**"Port 8188 already in use"** -- Another ComfyUI instance is running. Kill it first or use a different port via `COMFYUI_PORT`.

**Disabled node packs still showing errors** -- ComfyUI scans all directories in `custom_nodes/`, including those with `.disabled_` prefixes. Move disabled packs to a folder outside `custom_nodes/` (e.g., `Custom_Nodes_Disabled/`) to eliminate startup tracebacks.

## Testing

Tests run without ComfyUI -- everything is mocked:

```bash
python -m pytest tests/ -v
# 2000+ tests, all mocked, under 60 seconds
```

## License

[MIT](LICENSE) -- use it however you want.
