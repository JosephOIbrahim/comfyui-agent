# ComfyUI Agent

AI co-pilot for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — inspect, discover, modify, and execute workflows through natural conversation.

Built on the [Anthropic API](https://docs.anthropic.com/) with a tool-use agent loop. The agent decides which tools to call; you just describe what you want.

## Features

- **Inspect** your ComfyUI installation — models, custom nodes, GPU stats, queue status
- **Understand** workflows — auto-detect format (API/UI/hybrid), trace connections, find editable fields
- **Modify** workflows — RFC6902 JSON patches with full undo history, preview, and diff
- **Execute** workflows — queue prompts, poll for completion, inspect outputs
- **Discover** node packs and models — search ComfyUI Manager registry (31,000+ node types) and HuggingFace Hub
- **Remember** across sessions — save/load workflow state, record notes and preferences

## Quick Start

```bash
# Install
git clone https://github.com/JosephOIbrahim/comfyui-agent.git
cd comfyui-agent
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

# Run
agent run                          # Interactive agent session
agent run --session my-project     # With session persistence
```

## Requirements

- Python >= 3.10
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running locally (default: `http://127.0.0.1:8188`)
- Anthropic API key ([get one here](https://console.anthropic.com/))

## CLI Commands

### `agent run`

Start an interactive agent session. The agent has access to all 28 tools and will use them based on your requests.

```bash
agent run                          # Basic session
agent run --session flux-portrait  # Named session (auto-save on exit)
```

### `agent inspect`

Quick summary of your local ComfyUI installation — installed models by type and custom node packs.

```bash
agent inspect
```

### `agent parse`

Analyze a workflow file offline (no ComfyUI required).

```bash
agent parse workflow.json
```

### `agent search`

Search for custom node packs or models.

```bash
agent search "controlnet" --nodes       # Search node packs by name
agent search "KSampler" --node-type     # Find which pack provides a node type
agent search "sdxl" --models            # Search model registry
agent search "flux" --hf                # Search HuggingFace Hub
agent search "anime" --models --type lora  # Filter by model type
```

### `agent sessions`

List all saved sessions.

```bash
agent sessions
```

## Tools Reference

28 tools organized across 5 functional areas:

### Inspection (6 tools)

| Tool | Description |
|------|-------------|
| `is_comfyui_running` | Health check — GPU, Python version, connectivity |
| `get_all_nodes` | All registered node types with optional filtering |
| `get_node_info` | Full schema for a single node (inputs, outputs, types) |
| `get_system_stats` | GPU memory, loaded models, system info |
| `get_queue_status` | Running and pending queue items |
| `get_history` | Past execution results by prompt ID |

### Filesystem (4 tools)

| Tool | Description |
|------|-------------|
| `list_custom_nodes` | Scan Custom_Nodes directory |
| `list_models` | List model files by type with sizes |
| `get_models_summary` | Aggregate model counts across all types |
| `read_node_source` | Read a node pack's source code |

### Workflow Analysis (3 tools)

| Tool | Description |
|------|-------------|
| `load_workflow` | Full analysis — format detection, connection tracing, editable fields |
| `validate_workflow` | Check against running ComfyUI (node types, required inputs, type compatibility) |
| `get_editable_fields` | List tunable parameters grouped by node class |

### Workflow Editing (8 tools)

| Tool | Description |
|------|-------------|
| `apply_workflow_patch` | Apply RFC6902 JSON patches with undo support |
| `preview_workflow_patch` | Dry-run — see what would change without committing |
| `undo_workflow_patch` | Step back through patch history |
| `get_workflow_diff` | Show all changes from base workflow |
| `save_workflow` | Persist current state to file |
| `reset_workflow` | Discard all patches, return to base |
| `execute_workflow` | Queue prompt and poll for completion |
| `get_execution_status` | Check execution status by prompt ID |

### Discovery (3 tools)

| Tool | Description |
|------|-------------|
| `search_custom_nodes` | Search ComfyUI Manager registry by name or node type |
| `search_models` | Search local registry or HuggingFace Hub |
| `find_missing_nodes` | Analyze workflow dependencies, suggest packs to install |

### Session Memory (4 tools)

| Tool | Description |
|------|-------------|
| `save_session` | Persist workflow state and notes |
| `load_session` | Restore a previous session |
| `list_sessions` | List all saved sessions with metadata |
| `add_note` | Record observations and preferences for future sessions |

## Architecture

```
User --> CLI (typer/rich) --> Agent Loop (anthropic tool-use) --> Tool Registry
                                  |                                    |
                                  v                                    v
                            System Prompt                   7 Tool Modules (28 tools)
                            + Knowledge                            |
                                                        +----------+----------+
                                                        v          v          v
                                                  ComfyUI API   Filesystem  Session Memory
                                                  (httpx)       (pathlib)   (JSON files)
```

**Single-agent, synchronous tool-use loop.** Claude picks which tools to call based on your request. No multi-agent orchestration, no async streaming — deliberately simple.

The agent loop (`agent/main.py`) sends messages to the Anthropic API with all 28 tool definitions. When Claude responds with `tool_use` blocks, we execute them via the tool registry and feed results back. This continues until Claude produces a final text response.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single agent | ComfyUI workflows are sequential; multi-agent adds complexity without benefit |
| HTTP polling | Simpler tool context than WebSocket streaming for execution monitoring |
| RFC6902 patches | Standard, composable, reversible — enables undo/preview/diff for free |
| Module-level state | Single user, single session — avoids database complexity |
| ComfyUI Manager JSONs | 31,312 node types + 527 models available offline, no API key needed |
| Deterministic JSON | `sort_keys=True` everywhere for reproducible outputs ([He2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)) |
| Format auto-detection | Users shouldn't need to know API vs UI format; agent handles both |

## Configuration

All configuration via environment variables (`.env` file or shell):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Your Anthropic API key |
| `COMFYUI_HOST` | `127.0.0.1` | ComfyUI server host |
| `COMFYUI_PORT` | `8188` | ComfyUI server port |
| `COMFYUI_DATABASE` | `G:/COMFYUI_Database` | ComfyUI installation path |
| `AGENT_MODEL` | `claude-opus-4-6-20250929` | Claude model to use |

## Testing

```bash
# Run all tests (no ComfyUI required — everything is mocked)
python -m pytest tests/ -v

# 121 tests, ~0.5s
```

Tests cover all 28 tools with mocked HTTP responses, temporary filesystems, and isolated session state. No live ComfyUI instance needed.

## Project Structure

```
comfyui-agent/
  agent/
    cli.py              # 5 CLI commands (typer)
    main.py             # Agent loop (tool-use)
    config.py           # Environment + paths
    system_prompt.py    # Prompt builder + rules
    knowledge/          # ComfyUI reference docs
    memory/             # Session persistence
    tools/              # 28 tools across 7 modules
      _util.py          # Deterministic JSON helper
      comfy_api.py      # Live API tools
      comfy_inspect.py  # Filesystem tools
      workflow_parse.py # Workflow analysis
      workflow_patch.py # RFC6902 editing + undo
      comfy_execute.py  # Execution + polling
      comfy_discover.py # Registry + HuggingFace search
      session_tools.py  # Session memory bridge
  tests/                # 121 tests (8 modules)
  sessions/             # Saved session JSON files
  workflows/            # User workflow storage
```

## License

[MIT](LICENSE)
