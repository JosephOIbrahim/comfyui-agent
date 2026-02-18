# ComfyUI Agent

An AI assistant for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that understands your workflows and helps you work faster through natural conversation.

Instead of manually editing JSON, hunting for node packs, or debugging broken workflows yourself, just describe what you want and the agent handles it.

## What It Does

- **"What models do I have?"** — scans your installation instantly
- **"Load this workflow and change the seed to 42"** — reads, modifies, and saves with undo support
- **"What node pack do I need for IPAdapter?"** — searches 31,000+ node types to find the right pack
- **"Run this with 30 steps instead of 20"** — patches the workflow and queues it to ComfyUI
- **"Find me a good anime LoRA"** — searches ComfyUI Manager's registry, HuggingFace, and CivitAI
- **"Is this model compatible with my workflow?"** — checks SD1.5/SDXL/Flux/SD3 family compatibility
- **"Analyze this output — why does it look wrong?"** — uses Claude Vision to diagnose image issues
- **"Remember that I prefer SDXL for landscapes"** — saves notes and learns from your outcomes over time

The agent talks to ComfyUI's API directly. It reads your actual installation, sees what's really installed, and works with your real workflows.

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

If your ComfyUI is installed somewhere other than `G:/COMFYUI_Database`, also set:

```
COMFYUI_DATABASE=/path/to/your/ComfyUI
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
agent search "flux lora" --hf               # Search HuggingFace
agent search "anime" --models --type lora   # Filter by model type
```

## How It Works

The agent uses Claude (Anthropic's AI) with 65 specialized tools across two tiers:

**Intelligence Layer (44 tools)**

| Layer | Tools | What they do |
|-------|-------|-------------|
| **UNDERSTAND** | 13 | Parse workflows, scan models/nodes, query ComfyUI API, detect format |
| **DISCOVER** | 12 | Search ComfyUI Manager (31k+ nodes), HuggingFace, CivitAI, model compatibility, install instructions, GitHub releases |
| **PILOT** | 13 | RFC6902 patch engine with undo, semantic node ops, session persistence |
| **VERIFY** | 6 | Validate, execute, WebSocket progress monitoring, post-execution verification |

**Brain Layer (21 tools)**

| Module | Tools | What they do |
|--------|-------|-------------|
| **Vision** | 4 | Analyze generated images, A/B comparison, perceptual hashing |
| **Planner** | 4 | Goal decomposition, step tracking, replanning |
| **Memory** | 4 | Outcome learning with temporal decay, cross-session patterns |
| **Orchestrator** | 2 | Parallel sub-tasks with filtered tool access |
| **Optimizer** | 4 | GPU profiling, TensorRT detection, auto-apply optimizations |
| **Demo** | 2 | Guided walkthroughs for streams and podcasts |

When you ask a question, Claude decides which tools to use, calls them, reads the results, and responds. It streams text as it thinks, so you're never staring at a blank screen.

## Configuration

All settings go in your `.env` file:

| Setting | Default | What it does |
|---------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Your API key for Claude |
| `COMFYUI_HOST` | `127.0.0.1` | Where ComfyUI is running |
| `COMFYUI_PORT` | `8188` | ComfyUI port |
| `COMFYUI_DATABASE` | `G:/COMFYUI_Database` | Your ComfyUI installation folder |
| `AGENT_MODEL` | `claude-opus-4-6-20250929` | Which Claude model to use |

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

- **API format** — the JSON you get from "Save (API Format)" in ComfyUI. Full node data with inputs and connections. Best for the agent.
- **UI format with API data** — the default "Save" export. Contains visual layout plus embedded API data. Agent extracts what it needs.
- **UI-only format** — older exports with only visual layout. The agent can read the structure but can't modify or execute these.

## MCP Server (Primary Interface)

All 65 tools are available via [Model Context Protocol](https://modelcontextprotocol.io/) for integration with Claude Code, Claude Desktop, or other MCP clients:

```bash
pip install -e "."
agent mcp
```

## Troubleshooting

**"ANTHROPIC_API_KEY not set"** — Make sure your `.env` file exists and has the key. Run from the `comfyui-agent` directory.

**"Could not connect to ComfyUI"** — Start ComfyUI first. The agent needs it running to inspect nodes, execute workflows, and validate changes.

**"Node type not found"** — The workflow uses a custom node that isn't installed. Ask the agent: "find missing nodes in this workflow" and it will tell you which packs to install.

## Testing

Tests run without ComfyUI — everything is mocked:

```bash
python -m pytest tests/ -v
# 459 tests, all mocked, under 35 seconds
```

## License

[MIT](LICENSE) — use it however you want.
