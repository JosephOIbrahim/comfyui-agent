# CLAUDE.md — ComfyUI SUPER DUPER Agent

> **Core Thesis:** We are not building plumbing. We are building the first AI assistant
> that *understands* a VFX artist's ComfyUI workflow — and keeps up with the ecosystem they can't.
>
> **The Pace Problem:** New models drop daily. Custom nodes ship weekly. No artist can track it all.
> This agent does.

---

## IDENTITY

```
YOU:        The VFX artist's co-pilot. Not a developer tool.
AUDIENCE:   Lighting TDs, compositors, texture artists — NOT engineers.
VOICE:      Knowledgeable colleague, not a terminal. Explain what you're doing and why.
PRINCIPLE:  Driver, not generator. Small validated changes, never full workflow rewrites.
```

---

## Project Summary

ComfyUI SUPER DUPER Agent is an AI co-pilot for ComfyUI workflows. It uses Claude with
60 specialized tools organized into two tiers: four intelligence layers (UNDERSTAND,
DISCOVER, PILOT, VERIFY) and a brain layer (VISION, PLANNER, MEMORY, ORCHESTRATOR,
OPTIMIZER, DEMO). Natural conversation drives workflow inspection, discovery, modification,
execution, optimization, and learning. Built with the Anthropic SDK, httpx, and jsonpatch.

The primary interface is MCP (Model Context Protocol) via `agent mcp`, making all 65 tools
available to Claude Code, Claude Desktop, or any MCP client. The CLI agent (`agent run`)
serves as a standalone fallback. Our value lives in the intelligence and brain layers above
the transport.

---

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run (requires .env with ANTHROPIC_API_KEY)
agent run
agent run --session my-project --verbose

# Tests (497 tests, all mocked, <35s)
python -m pytest tests/ -v
python -m pytest tests/test_workflow_patch.py -v                              # single file
python -m pytest tests/test_session.py::TestSaveSession -v                    # single class
python -m pytest tests/test_context.py::TestTokenEstimation::test_simple_string -v  # single test

# Lint
ruff check agent/ tests/
ruff format agent/ tests/
```

---

## Claude Code MCP Setup

This project is MCP-first: Claude Code **is** the agent runtime; these tools are its domain expertise for ComfyUI.

**Install:**
```bash
pip install -e "."
```

**Configure** `.claude/settings.json` (or project-level):
```json
{
  "mcpServers": {
    "comfyui-agent": {
      "command": "agent",
      "args": ["mcp"],
      "cwd": "C:/Users/User/comfyui-agent"
    }
  }
}
```

---

## Tool Usage Rules

When using the ComfyUI agent tools via MCP, follow these rules:

1. **NEVER claim to know about specific models from memory. ALWAYS use tools.** Model ecosystems change daily.
2. When asked "what model should I use for X?" -- search first (`discover`), recommend after.
3. When modifying workflows, ALWAYS show the proposed patch and get confirmation before applying.
4. When something fails, read the error, check node compatibility, suggest fixes.
5. If ComfyUI is not running, say so immediately. Most tools require it.
6. Prefer `get_node_info` over memory for node interfaces. It's always current.
7. When suggesting new nodes/models, check if they're already installed first (`list_models`, `list_custom_nodes`).
8. Log key decisions and preferences to session notes (`add_note`) for continuity.
9. At conversation end, save the session so the user can resume later.
10. Use `format='names_only'` or `format='summary'` for large tool queries; drill down with specific tools.
11. For workflow creation, prefer loading a template (`list_workflow_templates`) and patching it.
12. Before executing, use `validate_before_execute` to catch errors early.
13. Use `add_node`/`connect_nodes`/`set_input` for building workflows instead of raw JSON patches when possible.
14. Never generate entire workflows from scratch. Make surgical, validated modifications.
15. Every patch is validated before application. No exceptions.

### Tool Overview

| Category | Tools |
|----------|-------|
| **Live API** | `is_comfyui_running`, `get_all_nodes`, `get_node_info`, `get_system_stats`, `get_queue_status`, `get_history` |
| **Filesystem** | `list_custom_nodes`, `list_models`, `get_models_summary`, `read_node_source` |
| **Workflow Understanding** | `load_workflow`, `validate_workflow`, `get_editable_fields` |
| **Workflow Editing** | `apply_workflow_patch`, `preview_workflow_patch`, `undo_workflow_patch`, `get_workflow_diff`, `save_workflow`, `reset_workflow` |
| **Semantic Building** | `add_node`, `connect_nodes`, `set_input` |
| **Execution** | `validate_before_execute`, `execute_workflow`, `get_execution_status`, `execute_with_progress` |
| **Discovery** | `discover`, `find_missing_nodes`, `check_registry_freshness`, `get_install_instructions` |
| **CivitAI** | `get_civitai_model`, `get_trending_models` |
| **Model Compat** | `identify_model_family`, `check_model_compatibility` |
| **Templates** | `list_workflow_templates`, `get_workflow_template` |
| **Session** | `save_session`, `load_session`, `list_sessions`, `add_note` |
| **Brain: Vision** | `analyze_image`, `compare_outputs`, `suggest_improvements`, `hash_compare_images` |
| **Brain: Planner** | `plan_goal`, `get_plan`, `complete_step`, `replan` |
| **Brain: Memory** | `record_outcome`, `get_learned_patterns`, `get_recommendations`, `detect_implicit_feedback` |
| **Brain: Orchestrator** | `spawn_subtask`, `check_subtasks` |
| **Brain: Optimizer** | `profile_workflow`, `suggest_optimizations`, `check_tensorrt_status`, `apply_optimization` |
| **Brain: Demo** | `start_demo`, `demo_checkpoint` |

---

## Artistic Intent to Parameter Translation

When the artist describes what they want in natural language, translate to ComfyUI parameters:

| Artist Says | Parameter Direction |
|------------|-------------------|
| "dreamier" / "softer" | Lower CFG (5-7), increase steps, use DPM++ 2M Karras |
| "sharper" / "crisper" | Higher CFG (8-12), use Euler or DPM++ SDE |
| "more photorealistic" | Higher CFG (7-10), use realistic checkpoint, negative: "cartoon, anime, illustration" |
| "more stylized" / "painterly" | Lower CFG (4-6), use artistic checkpoint or LoRA |
| "faster" | Fewer steps (15-20), use LCM/Lightning/Turbo sampler, smaller resolution |
| "higher quality" | More steps (30-50), enable hires fix, use upscaler |
| "consistent lighting" | Add ControlNet depth/normal, reference image via IP-Adapter |
| "similar to this image" | IP-Adapter with reference, or img2img with moderate denoise (0.4-0.6) |
| "more variation" | Higher denoise, different seed, lower CFG |
| "less variation" | Lower denoise, same seed, higher CFG |

---

## Model Family Quick Reference

| Family | Resolution | CFG Range | Negative Prompt | Key Notes |
|--------|-----------|-----------|----------------|-----------|
| **SD 1.5** | 512x512 | 7-12 | Yes, important | Massive LoRA ecosystem, fast |
| **SDXL** | 1024x1024 | 5-9 | Yes, but less critical | Base + refiner pipeline, better hands |
| **Flux** | 512-1024 | 1.0 (guidance_scale) | No negative prompt | Uses FluxGuidance node, T5 text encoder |
| **SD3** | 1024x1024 | 5-7 | Optional | Triple text encoder (CLIP-G, CLIP-L, T5) |

**Compatibility rules:** Models within the same family are generally compatible. Never mix SD1.5 LoRAs with SDXL checkpoints. Flux has its own LoRA format. ControlNets must match the base model family.

---

## Workflow JSON Schema (API Format)

ComfyUI API format is a flat dict of nodes keyed by string IDs:
```json
{
  "node_id": {
    "class_type": "NodeClassName",
    "inputs": {
      "literal_field": "value",
      "connection_field": ["source_node_id", output_index]
    }
  }
}
```
- Literal inputs: strings, numbers, booleans
- Connection inputs: `[node_id_str, output_index_int]` tuples
- The patch engine operates on this format exclusively

---

## Tool Redundancy Notes

Some tools overlap with Claude Code's native capabilities. All are kept for domain-specific value:

- **Filesystem tools** (`list_models`, `list_custom_nodes`, `read_node_source`): Overlap with Glob/Read, but provide ComfyUI-aware parsing (model type detection, node registration scanning). Keep.
- **Discovery tools** (`discover`): Unified cross-source search (ComfyUI Manager 31k+ nodes, CivitAI, HuggingFace). Deduplication, ranking, installed-status. No native equivalent. Core value.
- **Workflow tools** (entire PILOT layer): RFC6902 patch engine with undo. No native equivalent. Core value.
- **Session tools**: Could potentially be replaced by Claude Code's own memory, but workflow state serialization is domain-specific. Keep for now.
- **Brain tools**: Higher-order capabilities (vision analysis, planning, optimization). Core value, no native equivalent.

---

## Architecture

### Four Intelligence Layers

The agent's 44 intelligence + 21 brain tools are organized into four layers, each solving
a distinct problem for the artist. The transport underneath is commodity plumbing.

```
┌──────────────────── SUPER DUPER AGENT v0.3.0 ─────────────────────┐
│                                                                    │
│  BRAIN LAYER (21 tools)                                            │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌───────┐ ┌───────┐ ┌───────┐ │
│  │PLANNER │ │ VISION │ │ MEMORY │ │ ORCH  │ │OPTIM  │ │ DEMO  │ │
│  │4 tools │ │4 tools │ │4 tools │ │2 tools│ │4 tools│ │2 tools│ │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬───┘ └───┬───┘ └───┬───┘ │
│      └──────────┴──────┬───┴──────────┴─────────┴─────────┘     │
│                   _protocol.py (BrainMessage)                     │
│      ┌──────────────┬──┴───────┬──────────────┐                   │
│                                                                    │
│  INTELLIGENCE LAYERS (44 tools)                                    │
│  ┌───────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ UNDERSTAND│  │ DISCOVER  │  │  PILOT   │  │   VERIFY     │    │
│  │ 13 tools  │  │  6 tools  │  │ 13 tools │  │   7 tools    │    │
│  └─────┬─────┘  └─────┬─────┘  └────┬─────┘  └──────┬───────┘    │
│        └──────────────┴──────┬───────┴───────────────┘            │
│                    ┌─────────▼─────────┐                          │
│                    │    TRANSPORT      │  <- Thin, swappable      │
│                    │  (HTTP/WS + MCP)  │                          │
│                    └─────────┬─────────┘                          │
└──────────────────────────────┼────────────────────────────────────┘
                    ┌──────────▼──────────┐
                    │   ComfyUI Instance  │
                    │   (localhost:8188)   │
                    └─────────────────────┘
```

### Layer -> Module Mapping

| Layer | Module | Tools | Status |
|-------|--------|-------|--------|
| **UNDERSTAND** | `tools/workflow_parse.py` | 3 | loads, detects format, traces connections, extracts editable fields |
| **UNDERSTAND** | `tools/comfy_inspect.py` | 4 | filesystem scanning, `list_models` with progressive disclosure |
| **UNDERSTAND** | `tools/comfy_api.py` | 6 | live HTTP queries, `format` param for progressive disclosure |
| **DISCOVER** | `tools/comfy_discover.py` | 4 | unified `discover` + ComfyUI Manager registries + HuggingFace + freshness tracking + install instructions |
| **DISCOVER** | `tools/workflow_templates.py` | 2 | starter workflows in `agent/templates/` |
| **DISCOVER** | `tools/civitai_api.py` | 2 | CivitAI model details, trending models + local cross-ref |
| **DISCOVER** | `tools/model_compat.py` | 2 | model family identification (SD1.5/SDXL/Flux/SD3), compatibility checking |
| **PILOT** | `tools/workflow_patch.py` | 9 | RFC6902 patching (6) + semantic: `add_node`, `connect_nodes`, `set_input` (3) |
| **PILOT** | `tools/session_tools.py` | 4 | save/load/list sessions via `memory/session.py` |
| **VERIFY** | `tools/comfy_execute.py` | 4 | `validate_before_execute`, `execute_workflow`, `get_execution_status`, `execute_with_progress` (WebSocket) |
| **VERIFY** | `tools/verify_execution.py` | 2 | `get_output_path`, `verify_execution` — post-execution verification loop |
| **DISCOVER** | `tools/github_releases.py` | 2 | `check_node_updates`, `get_repo_releases` — GitHub release tracking for custom nodes |
| **BRAIN:VISION** | `brain/vision.py` | 4 | `analyze_image`, `compare_outputs`, `suggest_improvements`, `hash_compare_images` |
| **BRAIN:PLANNER** | `brain/planner.py` | 4 | `plan_goal`, `get_plan`, `complete_step`, `replan` — goal decomposition |
| **BRAIN:MEMORY** | `brain/memory.py` | 4 | `record_outcome`, `get_learned_patterns`, `get_recommendations`, `detect_implicit_feedback` |
| **BRAIN:ORCH** | `brain/orchestrator.py` | 2 | `spawn_subtask`, `check_subtasks` — parallel work with filtered tool access |
| **BRAIN:OPTIM** | `brain/optimizer.py` | 4 | `profile_workflow`, `suggest_optimizations`, `check_tensorrt_status`, `apply_optimization` |
| **BRAIN:DEMO** | `brain/demo.py` | 2 | `start_demo`, `demo_checkpoint` — guided walkthroughs for streams/podcasts |
| **TRANSPORT** | `mcp_server.py` | — | MCP server exposing all 65 tools via Model Context Protocol (primary interface) |

### What's Built vs What's Next

```
BUILT (v0.3.1 — working today):
  ✅ 65 tools: 44 intelligence layer + 21 brain layer
  ✅ MCP as primary interface (core dependency, not optional)
  ✅ Session isolation (WorkflowSession with per-session locking)
  ✅ CLAUDE.md knowledge layer (tool rules, artistic intent, model families)
  ✅ Agent loop with streaming, tool dispatch, context management
  ✅ RFC6902 patch engine with undo history
  ✅ ComfyUI Manager registry search (31k+ nodes)
  ✅ HuggingFace model search
  ✅ CivitAI integration (search, trending, model details, local cross-ref)
  ✅ Install instructions tool (discovery -> installation bridge)
  ✅ Session persistence and resume
  ✅ Knowledge system (ControlNet, Flux, video, recipes)
  ✅ Brain: Vision (Claude Vision + perceptual hash A/B comparison)
  ✅ Brain: Planner (goal decomposition, progress tracking, replanning)
  ✅ Brain: Memory (JSONL outcomes, pattern learning, contextual recs, implicit feedback)
  ✅ Brain: Orchestrator (parallel sub-tasks, tool access profiles, TTL eviction)
  ✅ Brain: Optimizer (GPU profiles, TensorRT/CUTLASS, auto-apply)
  ✅ Brain: Demo (4 guided scenarios for streams/podcasts)
  ✅ WebSocket execution monitoring (real-time progress)
  ✅ Model compatibility tracking (SD1.5/SDXL/Flux/SD3 family detection)
  ✅ Freshness tracking (registry staleness, cache management)
  ✅ Path sanitization and thread safety
  ✅ Circuit breaker (CLOSED/OPEN/HALF_OPEN) for ComfyUI HTTP resilience
  ✅ Rate limiter (token bucket: CivitAI, HuggingFace, Vision)
  ✅ Structured logging (JSON + human formatters, correlation IDs, rotation)
  ✅ Temporal decay in memory aggregations (7-day half-life)
  ✅ Goal tracking (planner goal_id -> memory outcome linking)
  ✅ Cross-session learning (scope=global aggregates all sessions)
  ✅ BrainMessage protocol activated (vision -> memory)
  ✅ He2025 determinism (3-pass audit, 21 violations fixed, full compliance)
  ✅ 497 tests, all mocked, <35s, 0 lint warnings

NEXT:
  ✅ Unified discovery tool (merge search_custom_nodes + search_models + CivitAI)
  ✅ Agent SDK extraction (BrainConfig + BrainAgent base class, 6 standalone agent classes)
  🔲 Rich CLI formatting (panels, tables, syntax highlighting)
  🔲 GitHub API release tracking for key custom node repos
  🔲 Proactive surfacing: recommend when relevant, not firehose
```

---

## Agent Loop

`cli.py` (Typer CLI) → `main.py:run_interactive()` → streaming Claude API calls → tool dispatch → repeat.

The agent loop in `main.py` handles: streaming responses via `client.messages.stream()`, tool call detection and dispatch, context window management (observation masking + token estimation + structured compaction at 120k tokens), parallel tool execution via ThreadPoolExecutor, and exponential backoff retry (3 retries, 1s/2s/4s delays).

### Context Engineering Pipeline

Messages flow through three stages before each API call:
1. **Observation masking** (`_mask_processed_results`): replaces large tool results from prior turns with compact references (>1500 chars), preserving the most recent results intact
2. **Compaction pass 1**: truncates tool results >2000 chars
3. **Compaction pass 2**: if still over 120k tokens, drops old messages with a structured summary (`_summarize_dropped`) that preserves user requests, tools used, and workflow context

### Session-Aware System Prompt

`build_system_prompt(session_context=...)` injects session notes and workflow state into the system prompt when resuming a named session. This places user preferences (e.g., "prefers SDXL") in privileged position before knowledge files. The `_detect_relevant_knowledge()` function loads domain-specific knowledge files (ControlNet, Flux, video, recipes) based on detected workflow node types and session notes.

---

## Layer Details

### UNDERSTAND Layer (Parse + Inspect + Explain)

**Purpose:** Know what the artist has. Explain it back to them in human terms.

**Current implementation:**
- `workflow_parse.py` — loads workflows, detects format (API / UI+API / UI-only), traces connections, extracts editable fields
- `comfy_inspect.py` — filesystem scanning, `list_models` with `format` param (`names_only`/`summary`/`full`)
- `comfy_api.py` — live HTTP queries to ComfyUI with progressive disclosure

**Workflow Format Handling** (three formats, handled transparently):
- **API format**: `{node_id: {class_type, inputs}}` — full support
- **UI with API**: ComfyUI default export with `extra.prompt` embedded — agent extracts API data
- **UI-only**: Layout only, no API data — read-only, cannot patch or execute

Format detection happens in `workflow_parse.py:_extract_api_format()`.

**Next steps for this layer:**
- Workflow pattern classification ("This is an img2img pipeline with ControlNet guidance")
- Plain-English 3-sentence summaries of any workflow
- Issue detection: missing models, deprecated nodes, suboptimal connections

### DISCOVER Layer (Model Hub + Node Index + Recommend) ⭐ PRIMARY DIFFERENTIATOR

**Purpose:** Solve the pace problem. Track what's new, what's good, what's relevant.

**This is what nobody else builds.** CLI bridges parse JSON. MCP servers call APIs.
We are the only system that says: "A new SDXL lightning model dropped yesterday
that would cut your render time in half for this workflow."

**Current implementation:**
- `comfy_discover.py` — searches ComfyUI Manager registries (31k+ node types) + HuggingFace
- `workflow_templates.py` — starter workflows in `agent/templates/`

**Data Sources (Real-Time ACCESS, Not Learned):**
- HuggingFace API — model search, metadata, download counts
- ComfyUI Manager node registry — available nodes, versions, compatibility
- Local filesystem scan — what's already installed (via `comfy_inspect.py`)
- CivitAI API — community models, ratings, usage stats, trending
- Freshness tracking — registry staleness, cache management, model directory stats
- Model compatibility — SD1.5/SDXL/Flux/SD3 family detection via regex patterns

**Recommendation Format (target):**
```
💡 DISCOVERY: {model_or_node_name}
   What: {one-line description}
   Why: {relevance to current workflow}
   Impact: {what changes — speed, quality, capability}
   Risk: {compatibility notes, if any}
   Action: "Want me to swap it in?" / "Want me to install it?"
```

### PILOT Layer (Natural Language → JSON Patches)

**Purpose:** The artist says what they want. We make validated, reversible changes.

**Core Principle: DRIVER, NOT GENERATOR.**
We never generate entire workflows from scratch. We make surgical, validated
modifications to existing workflows using RFC6902 JSON patches.

**Current implementation:**
- `workflow_patch.py` — 6 RFC6902 patch tools + 3 semantic composition tools (`add_node`, `connect_nodes`, `set_input`)
- `session_tools.py` — save/load/list sessions, delegates to `memory/session.py`

**Stateful Workflow Editing:**
`workflow_patch.py` maintains module-level state (`_state` dict) with: original workflow (immutable), working copy, undo history stack, loaded file path, and detected format. This enables multi-step editing sessions without reloading. Templates loaded via `get_workflow_template` also populate this state. State is reset between test runs via the `reset_workflow_state` fixture.

**Patch operates on the extracted API format** from `workflow_parse.py`.

**Forbidden Operations:**
- Never delete all nodes
- Never replace the entire workflow JSON
- Never modify node types (only inputs/connections)
- Never apply unvalidated patches
- If a change would break the DAG (disconnect outputs), refuse and explain

### VERIFY Layer (Pre-flight + Execute + Status)

**Purpose:** Trust but verify. Prove the change did what we said it would.

**Current implementation:**
- `comfy_execute.py` — `validate_before_execute`, `execute_workflow`, `get_execution_status`, `execute_with_progress` (WebSocket real-time monitoring with progress events)
- `brain/vision.py` — `hash_compare_images` for perceptual hash A/B comparison (no API call)
- `brain/memory.py` — `detect_implicit_feedback` for behavioral signal detection

**Verification Report (target):**
```
✓ VERIFIED: {change_description}
  Render time: {before} → {after} ({delta})
  Output: {same_seed_comparison}
  Confidence: {high/medium/low}
```

---

## Brain Layer (Hybrid B+C Architecture)

The brain layer sits above the intelligence layers and provides higher-order capabilities.
Each module lives in `agent/brain/` and registers tools through the same pattern as
intelligence layers (`TOOLS` list + `handle()` function). Modules communicate via
`_protocol.py:brain_message()`. Brain tools are lazily loaded to avoid circular imports
with `tools/_util.py`.

**Design:** SDK-ready agents with dependency injection. Each module defines a
`BrainAgent` subclass with `BrainConfig` for DI. `_sdk.py` provides the foundation:
`BrainConfig` (dataclass with `to_json`, `validate_path`, `sessions_dir`, etc.) and
`BrainAgent` (base class with `TOOLS`, `handle()`, `self.cfg`). Modules can be
instantiated standalone with custom config or integrated via `get_integrated_config()`.
Module-level `TOOLS` and `handle()` are preserved via lazy singleton for backward compat.

### Brain: Vision (`brain/vision.py`)
Uses separate Claude Vision API calls with 120s timeout (keeps images out of main context window).
Analyzes generated images, compares A/B outputs, suggests parameter improvements.
Returns structured JSON (quality_score, artifacts, composition, suggestions).
Also provides instant perceptual hash comparison (`hash_compare_images`) via Pillow aHash + pixel diff.

### Brain: Planner (`brain/planner.py`)
Template-based goal decomposition — 6 patterns (build_workflow, optimize_workflow,
debug_workflow, swap_model, add_controlnet, explore_ecosystem) + generic fallback.
State persists to `sessions/{name}_goals.json`. Supports step completion, replanning.

### Brain: Memory (`brain/memory.py`)
Append-only JSONL outcomes in `sessions/{name}_outcomes.jsonl`. Aggregation-based
pattern detection: best model combos, optimal params, speed analysis, quality trends.
Contextual recommendations (workflow-aware), negative pattern avoidance, goal-specific recs.
Implicit feedback detection: reuse (positive), abandonment (negative), refinement bursts
(positive), parameter regression (negative) — with inferred satisfaction scoring.

### Brain: Orchestrator (`brain/orchestrator.py`)
Parallel sub-tasks via ThreadPoolExecutor with thread safety (locks on `_active_tasks`).
Three tool access profiles: researcher (read-only), builder (can modify workflows),
validator (can execute + analyze). Max 3 concurrent, 60s timeout, TTL eviction of
completed tasks after 10 minutes, results in original order.

### Brain: Optimizer (`brain/optimizer.py`)
GPU profiles for RTX 4090/4080/3090/3080. TensorRT integration via ComfyUI_TensorRT
node pack detection. Optimization catalog ranked by impact/effort. Auto-apply for:
vae_tiling, batch_size, step_optimization, sampler_efficiency.

### Brain: Demo (`brain/demo.py`)
4 scripted scenarios: model_swap, speed_run, controlnet_add, full_pipeline. Each
has narration text, suggested tools, and pacing checkpoints. Module-level state
tracks active demo progress.

---

## Knowledge System

`agent/knowledge/` contains markdown reference files:
- `comfyui_core.md` — always loaded (type system, API endpoints, execution model)
- `controlnet_patterns.md` — loaded when ControlNet nodes detected
- `video_workflows.md` — loaded when AnimateDiff/SVD nodes detected
- `flux_specifics.md` — loaded when Flux models/nodes detected
- `common_recipes.md` — loaded when "create workflow" intent detected

Detection is keyword-based via `_KNOWLEDGE_TRIGGERS` in `system_prompt.py`.

---

## Session Persistence

`memory/session.py` saves/loads JSON files in `sessions/`. Captures: workflow state (path, base, current, history depth), user notes, and metadata. On resume, session context is injected into the system prompt (not just returned as a tool result).

---

## Tool System

Every tool module in `agent/tools/` exports the same interface:
- `TOOLS: list[dict]` — Anthropic tool schemas (name, description, input_schema)
- `handle(name: str, tool_input: dict) -> str` — executes and returns JSON

`tools/__init__.py` auto-registers all modules and dispatches by tool name. To add a new tool: create a module with `TOOLS` and `handle()`, then add it to `_MODULES` in `__init__.py`.

---

## Transport Layer (Deliberately Thin)

**Current:** Direct HTTP/WS to ComfyUI's native API at `:8188` via `comfy_api.py`.

**Future:** MCP adapter as an alternative transport. Same tool interface, different backend.
Artist doesn't know or care which transport is active.

**Decision: MCP is the primary interface. HTTP/WS is the transport underneath.**
Claude Code is the agent runtime; we provide the tool belt via MCP.

### MCP Server (`agent/mcp_server.py`)

All 65 tools are exposed via Model Context Protocol using `mcp.server.Server`. MCP is a
core dependency (`pip install -e "."`). Run `agent mcp` to start the stdio transport.
Schema conversion bridges Anthropic tool schemas to MCP JSON Schema format. Sync tool
handlers are wrapped with `run_in_executor` for the async MCP runtime. Session isolation
via `WorkflowSession` enables concurrent tool calls within a single Claude Code session.

### Supported Backends (Priority Order)
1. **MCP stdio** — `agent mcp` command, primary interface for Claude Code / Claude Desktop
2. **Direct HTTP/WS** — ComfyUI's native API (transport layer, always works)
3. **CLI agent** — `agent run` standalone fallback with built-in agent loop

---

## Implementation Roadmap

### Phase 1: Foundation -- COMPLETE
34 tools, agent loop, patch engine, session persistence, knowledge system, 169 tests.

### Phase 1.5: Brain Layer -- COMPLETE
18 brain tools: vision, planner, memory, orchestrator, optimizer, demo. 236 total tests.

### Phase 2: DISCOVER Enhancement -- COMPLETE
CivitAI integration, contextual recommendations, freshness tracking, model compatibility,
implicit feedback detection, perceptual hash comparison, WebSocket monitoring, MCP adapter.
347 tests. 61 tools.

### Phase 3: Hardening -- COMPLETE
Error handling (graceful JSON errors), path sanitization, thread safety (locks on mutable
state), orchestrator TTL eviction, vision API timeout, agent loop tests (27 tests for
main.py). Rate limiting, structured logging, graceful shutdown, schema versioning,
atomic writes, Docker, CI. 397 tests.

### Phase 3.5: Intelligence Upgrade -- COMPLETE
Circuit breaker for HTTP resilience, temporal decay in memory aggregations,
goal tracking (planner->memory linking), cross-session learning, BrainMessage
protocol activation, He2025 determinism audit and fixes. 429 tests.

### Phase 3.75: MCP-First Transformation -- COMPLETE
MCP as core dependency (not optional), CLAUDE.md knowledge layer (tool rules,
artistic intent translation, model family reference), WorkflowSession for
per-session state isolation, `get_install_instructions` tool, He2025 deep
audit (3 passes, 21 violations fixed). 497 tests. 60 tools.

### Phase 4: Next
**Goal:** Unified discovery and agent SDK extraction.

**Tasks:**
1. ✅ Unified discovery tool (merge search_custom_nodes + search_models + CivitAI)
2. ✅ Agent SDK extraction (BrainConfig + BrainAgent, 6 standalone agent classes)
3. 🔲 Rich CLI formatting (panels, tables, syntax highlighting)
3. 🔲 GitHub API release tracking for key custom node repos
4. 🔲 Proactive surfacing (recommend when relevant, not firehose)

**Success Criteria:**
- Non-technical artist can modify a workflow using natural language
- Agent explains every action in terms the artist understands
- Demo mode runs start-to-finish without errors
- Looks good enough to show on a podcast/stream

---

## Key Conventions

- **Deterministic JSON**: All serialization uses `sort_keys=True` (He2025 pattern). The `_util.py:to_json()` helper enforces this.
- **Line length**: 99 chars (ruff config in pyproject.toml)
- **All tests are mocked**: No ComfyUI server or API key needed. HTTP calls mocked via `unittest.mock.patch`.
- **Config via .env**: `agent/config.py` loads environment variables with `python-dotenv`. Key settings: `ANTHROPIC_API_KEY` (required), `COMFYUI_DATABASE` (default `G:/COMFYUI_Database`), `COMFYUI_HOST`/`COMFYUI_PORT`.
- **Custom_Nodes capitalization**: ComfyUI uses `Custom_Nodes` (capital C, capital N) for its custom nodes directory.
- **asyncio_mode = "auto"**: Configured in pyproject.toml for pytest-asyncio.
- **Parallel tool calls**: When the model returns multiple tool_use blocks, they execute concurrently via ThreadPoolExecutor (max 4 workers). Single tool calls skip the thread pool.

### Python Standards
- Python 3.11+ (match ComfyUI's requirement)
- Type hints everywhere
- `httpx` for async HTTP (not requests)
- `pytest` + `pytest-asyncio` for tests
- All workflow modifications go through the patch engine. No direct JSON mutation.
- Every patch is validated before application. No exceptions.

### Error Handling
- Transport errors -> retry with backoff, then surface to artist in plain language
- Patch validation errors -> explain what went wrong and suggest alternatives
- Missing model errors -> trigger DISCOVER layer to find alternatives
- Tool exceptions caught at dispatch level (both `tools/__init__.py` and `brain/__init__.py`), returned as JSON error strings to prevent agent loop crashes
- File path sanitization in `_util.validate_path()` blocks access outside allowed directories
- Thread safety: workflow_patch, orchestrator, and demo modules use `threading.Lock` on mutable state
- Never show raw tracebacks to the artist. Translate to human language.

### Commit Messages
```
[UNDERSTAND] Add workflow pattern recognition for ControlNet pipelines
[PILOT] Fix patch validation for multi-output nodes
[DISCOVER] Integrate CivitAI trending models endpoint
[VERIFY] Add perceptual hash comparison for output images
[TRANSPORT] Handle WebSocket reconnection on ComfyUI restart
[CLI] Add demo mode walkthrough for model swapping
[TEST] Add fixture for SDXL + ControlNet + IP-Adapter workflow
```

---

## Competitive Position

```
                        Developer-Focused ◄──────────► Artist-Focused
                              │                              │
  CLI Bridges                 │                              │
  (VibeComfy, Comfy Agent)   ●                              │
                              │                              │
  MCP Servers                 │                              │
  (IO-Ateliertech, Pilot)    ●                              │
                              │                              │
  ComfyUI Nodes              │                              │
  (claude-code-comfyui)       ●                              │
                              │                              │
  SUPER DUPER                 │                              ●  ← US
                              │                              │
                              │                              │
  Nobody ─────────────────────┼──────────────────────────────│
                              │                              │
                 No Discovery ◄──────────────────► Discovery │
```

**We win by being the only project in the artist-focused + discovery quadrant.**

---

## Demo Scenarios (Phase 5 Targets)

### Scenario 1: "What Do I Have?"
```
Artist: "What's in this workflow?"
Agent:  Parses → Explains in plain English
        "This is a txt2img pipeline using SD 1.5 with a LoRA for
        anime style. It's using 30 Euler steps at 0.7 CFG.
        You have 3 models installed that could work here."
```

### Scenario 2: "Make It Better"
```
Artist: "Can we make this faster without losing quality?"
Agent:  Analyzes → Recommends → Patches
        "Your sampler is doing 30 steps with Euler. Switching to
        DPM++ 2M Karras at 20 steps will be ~40% faster with
        nearly identical output. Want me to swap it?"
```

### Scenario 3: "What's New?"
```
Artist: "Anything new for SDXL ControlNet this week?"
Agent:  Searches → Filters → Recommends
        "Two things: a new depth preprocessor node (MiDaS v3.1)
        that's more accurate on architecture shots, and a
        community LoRA for consistent lighting. The depth
        preprocessor would plug right into your current workflow."
```

### Scenario 4: "I Broke It"
```
Artist: "The output looks weird after that change"
Agent:  Compares → Diagnoses → Reverts
        "Same seed comparison shows the color shift started with
        the CFG change. Your old CFG was 7.0, new is 12.0 —
        that's pushing it into oversaturation. Want me to revert,
        or try 8.5 as a middle ground?"
```

---

## Non-Goals (Explicitly Out of Scope)

- **We do not generate workflows from scratch.** We modify existing ones.
- **We do not replace the ComfyUI GUI.** We augment it.
- **We do not compete with CLI bridges.** Different audience entirely.
- **We do not train or fine-tune models.** We help artists find and use them.
- **We do not build a full MCP server.** We use a thin adapter over existing ones.
- **We do not optimize for developers.** Every interaction assumes a VFX artist.

---

## Guiding Quotes

> "The pace problem isn't about speed. It's about relevance. Artists don't need to
> know about every model — they need to know about the ones that matter for their work."

> "Driver, not generator. A good co-pilot doesn't grab the wheel. They read the map,
> check the mirrors, and say 'turn coming up' at the right moment."

> "If you have to read the docs to use it, we failed."
