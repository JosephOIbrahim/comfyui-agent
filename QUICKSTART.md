# Quick Start — ComfyUI Agent for VFX Artists

> **What this is:** An AI co-pilot that talks to your ComfyUI installation.
> Ask it questions in plain English, and it inspects your workflows, finds models,
> makes changes, and runs generations for you.

---

## Setup (5 minutes)

### 1. Prerequisites

You need:
- **Python 3.10+** — check with `python --version`
- **ComfyUI** running on your machine (the agent talks to it over HTTP)
- **An Anthropic API key** — get one at [console.anthropic.com](https://console.anthropic.com/)

### 2. Install

```bash
git clone https://github.com/JosephOIbrahim/comfyui-agent.git
cd comfyui-agent
pip install -e .
```

### 3. Configure

```bash
copy .env.example .env
```

Open `.env` in a text editor and set:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
COMFYUI_DATABASE=G:/path/to/your/ComfyUI
```

> **Where's my ComfyUI database?** It's wherever your `models/`, `Custom_Nodes/`,
> and `output/` folders live. If you're not sure, check your ComfyUI startup script.

### 4. Start everything

**Option A — Startup script (recommended):**

Edit the paths in `scripts/comfyui_with_agent.bat` to match your setup, then double-click it.
It starts ComfyUI, waits for it to be ready, then tells you how to connect.

**Option B — Manual:**

1. Start ComfyUI however you normally do
2. In a terminal, from the `comfyui-agent` folder:
   ```
   agent run
   ```
3. Type what you want. Type `quit` to exit.

---

## Using with Claude Code (Best Experience)

The agent works best as an MCP server inside Claude Code. This means Claude
can use all 77 ComfyUI tools alongside its normal coding abilities.

### Setup

1. Install [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
2. Open a terminal in the `comfyui-agent` folder
3. The MCP server is already configured in `.claude/settings.json`
4. Run `claude` — the agent tools are automatically available

---

## Using with Claude Desktop

If you prefer Claude Desktop over Claude Code:

### Setup

Add this to your Claude Desktop MCP config (`%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "comfyui-agent": {
      "command": "agent",
      "args": ["mcp"],
      "cwd": "C:\\Users\\YourName\\comfyui-agent"
    }
  }
}
```

Change the `cwd` path to wherever you cloned comfyui-agent. Restart Claude Desktop.
The 77 tools will appear automatically.

### What you can say

Just talk normally. Examples:

- "What models do I have installed?"
- "Load this workflow and show me the editable fields"
- "Change the sampler to DPM++ 2M Karras and run it"
- "Make it dreamier" (lowers CFG, adjusts sampler)
- "Find me a good LoRA for anime style"
- "What node pack do I need for IPAdapter?"
- "Why does this output look oversaturated?"

---

## Offline Tools (no API key needed)

Some commands work without an Anthropic key or running ComfyUI:

```bash
agent inspect              # See what models and nodes you have
agent parse workflow.json  # Analyze a workflow file
agent sessions             # See saved sessions
agent search "controlnet" --nodes   # Search node registry
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "ANTHROPIC_API_KEY not set" | Check your `.env` file exists and has the key |
| "Could not connect to ComfyUI" | Start ComfyUI first, then the agent |
| "Node type not found" | Ask: "find missing nodes in this workflow" |
| Agent is slow | Default model is Sonnet (fast + cheap). Set `AGENT_MODEL=claude-opus-4-6-20250929` in `.env` for higher quality |
| Wrong ComfyUI path | Set `COMFYUI_DATABASE` in `.env` to your actual ComfyUI folder |

---

## What It Won't Do

- **No full workflow generation.** It modifies existing workflows, not creates from scratch.
- **No replacing ComfyUI's GUI.** It augments your existing workflow.
- **No model training.** It helps you find and use models, not create them.

---

## Folder Structure

```
comfyui-agent/
├── agent/          # The AI co-pilot code
├── sessions/       # Your saved workflow sessions (auto-created)
├── workflows/      # Starter workflow templates
├── scripts/        # Utility scripts (startup, validation)
├── .env            # Your config (API key, paths)
└── CLAUDE.md       # Full project documentation
```
