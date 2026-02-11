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
52 specialized tools organized into two tiers: four intelligence layers (UNDERSTAND,
DISCOVER, PILOT, VERIFY) and a brain layer (VISION, PLANNER, MEMORY, ORCHESTRATOR,
OPTIMIZER, DEMO). Natural conversation drives workflow inspection, discovery, modification,
execution, optimization, and learning. Built with the Anthropic SDK, httpx, and jsonpatch.

The transport layer (HTTP/WS to ComfyUI) is deliberately thin and swappable. Our value
lives in the intelligence and brain layers above it.

---

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run (requires .env with ANTHROPIC_API_KEY)
agent run
agent run --session my-project --verbose

# Tests (236 tests, all mocked, <10s)
python -m pytest tests/ -v
python -m pytest tests/test_workflow_patch.py -v                              # single file
python -m pytest tests/test_session.py::TestSaveSession -v                    # single class
python -m pytest tests/test_context.py::TestTokenEstimation::test_simple_string -v  # single test

# Lint
ruff check agent/ tests/
ruff format agent/ tests/
```

---

## Architecture

### Four Intelligence Layers

The agent's 34 tools are organized into four layers, each solving a distinct problem
for the artist. The transport underneath is commodity plumbing — our value lives here.

```
┌──────────────────── SUPER DUPER AGENT v0.2.0 ─────────────────────┐
│                                                                    │
│  BRAIN LAYER (18 tools)                                            │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌───────┐ ┌───────┐ ┌───────┐ │
│  │PLANNER │ │ VISION │ │ MEMORY │ │ ORCH  │ │OPTIM  │ │ DEMO  │ │
│  │4 tools │ │3 tools │ │3 tools │ │2 tools│ │4 tools│ │2 tools│ │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬───┘ └───┬───┘ └───┬───┘ │
│      └──────────┴──────┬───┴──────────┴─────────┴─────────┘     │
│                   _protocol.py (BrainMessage)                     │
│      ┌──────────────┬──┴───────┬──────────────┐                   │
│                                                                    │
│  INTELLIGENCE LAYERS (34 tools)                                    │
│  ┌───────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ UNDERSTAND│  │ DISCOVER  │  │  PILOT   │  │   VERIFY     │    │
│  │ 13 tools  │  │  5 tools  │  │ 13 tools │  │   3 tools    │    │
│  └─────┬─────┘  └─────┬─────┘  └────┬─────┘  └──────┬───────┘    │
│        └──────────────┴──────┬───────┴───────────────┘            │
│                    ┌─────────▼─────────┐                          │
│                    │    TRANSPORT      │  <- Thin, swappable      │
│                    │  (HTTP/WS today)  │                          │
│                    │  (MCP tomorrow)   │                          │
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
| **UNDERSTAND** | `tools/workflow_parse.py` | 3 | ✅ loads, detects format, traces connections, extracts editable fields |
| **UNDERSTAND** | `tools/comfy_inspect.py` | 4 | ✅ filesystem scanning, `list_models` with progressive disclosure |
| **UNDERSTAND** | `tools/comfy_api.py` | 6 | ✅ live HTTP queries, `format` param for progressive disclosure |
| **DISCOVER** | `tools/comfy_discover.py` | 3 | ✅ ComfyUI Manager registries (31k+ node types) + HuggingFace |
| **DISCOVER** | `tools/workflow_templates.py` | 2 | ✅ starter workflows in `agent/templates/` |
| **PILOT** | `tools/workflow_patch.py` | 9 | ✅ RFC6902 patching (6) + semantic: `add_node`, `connect_nodes`, `set_input` (3) |
| **PILOT** | `tools/session_tools.py` | 4 | ✅ save/load/list sessions via `memory/session.py` |
| **VERIFY** | `tools/comfy_execute.py` | 3 | ✅ `validate_before_execute`, `execute_workflow`, `get_execution_status` |
| **BRAIN:VISION** | `brain/vision.py` | 3 | ✅ `analyze_image`, `compare_outputs`, `suggest_improvements` via Claude Vision |
| **BRAIN:PLANNER** | `brain/planner.py` | 4 | ✅ `plan_goal`, `get_plan`, `complete_step`, `replan` — goal decomposition |
| **BRAIN:MEMORY** | `brain/memory.py` | 3 | ✅ `record_outcome`, `get_learned_patterns`, `get_recommendations` — JSONL outcomes |
| **BRAIN:ORCH** | `brain/orchestrator.py` | 2 | ✅ `spawn_subtask`, `check_subtasks` — parallel work with filtered tool access |
| **BRAIN:OPTIM** | `brain/optimizer.py` | 4 | ✅ `profile_workflow`, `suggest_optimizations`, `check_tensorrt_status`, `apply_optimization` |
| **BRAIN:DEMO** | `brain/demo.py` | 2 | ✅ `start_demo`, `demo_checkpoint` — guided walkthroughs for streams/podcasts |

### What's Built vs What's Next

```
BUILT (v0.2.0 — working today):
  ✅ 52 tools: 34 intelligence layer + 18 brain layer
  ✅ Agent loop with streaming, tool dispatch, context management
  ✅ RFC6902 patch engine with undo history
  ✅ ComfyUI Manager registry search (31k+ nodes)
  ✅ HuggingFace model search
  ✅ Session persistence and resume
  ✅ Knowledge system (ControlNet, Flux, video, recipes)
  ✅ Brain: Vision (Claude Vision image analysis + A/B comparison)
  ✅ Brain: Planner (goal decomposition, progress tracking, replanning)
  ✅ Brain: Memory (outcome JSONL, pattern learning, recommendations)
  ✅ Brain: Orchestrator (parallel sub-tasks, tool access profiles)
  ✅ Brain: Optimizer (GPU profiles, TensorRT/CUTLASS, auto-apply)
  ✅ Brain: Demo (4 guided scenarios for streams/podcasts)
  ✅ 236 tests, all mocked, <10s

NEXT (the moat — where nobody else is building):
  🔲 CivitAI integration (community models, ratings, trending)
  🔲 MCP transport adapter (swap in alongside HTTP/WS)
  🔲 Perceptual hash comparison for output images
  🔲 Agent SDK extraction (brain modules -> standalone agents)
  🔲 WebSocket execution monitoring (real-time progress)
  🔲 Model compatibility tracking (which checkpoints <-> which nodes)
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
- ✅ HuggingFace API — model search, metadata, download counts
- ✅ ComfyUI Manager node registry — available nodes, versions, compatibility
- ✅ Local filesystem scan — what's already installed (via `comfy_inspect.py`)
- 🔲 CivitAI API — community models, ratings, usage stats
- 🔲 GitHub API — release tracking for key custom node repos

**Next steps (the moat):**
- Contextual recommendations: compare workflow needs against available options
- CivitAI integration for community model discovery
- Freshness tracking: "new this week" vs "been around for months"
- Model compatibility mapping (which checkpoints ↔ which nodes/samplers)
- Proactive surfacing: recommend only when relevant, not as a firehose

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
- `comfy_execute.py` — `validate_before_execute` for pre-flight checks, `execute_workflow`, `get_execution_status`

**Next steps:**
- Output capture (hash outputs per workflow+seed)
- Same-seed A/B comparison (before/after modification)
- Render time delta tracking
- Perceptual hash comparison for image outputs
- Regression detection (unexpected output changes)

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

**Design:** Hybrid B+C — modules today, agent-ready seams for Agent SDK extraction
tomorrow. Each module is stateless per-call but state-aware via persistence.

### Brain: Vision (`brain/vision.py`)
Uses separate Claude Vision API calls (keeps images out of main context window).
Analyzes generated images, compares A/B outputs, suggests parameter improvements.
Returns structured JSON (quality_score, artifacts, composition, suggestions).

### Brain: Planner (`brain/planner.py`)
Template-based goal decomposition — 6 patterns (build_workflow, optimize_workflow,
debug_workflow, swap_model, add_controlnet, explore_ecosystem) + generic fallback.
State persists to `sessions/{name}_goals.json`. Supports step completion, replanning.

### Brain: Memory (`brain/memory.py`)
Append-only JSONL outcomes in `sessions/{name}_outcomes.jsonl`. Aggregation-based
pattern detection: best model combos, optimal params, speed analysis, quality trends.
Implicit feedback from conversation ("that looks great" -> positive).

### Brain: Orchestrator (`brain/orchestrator.py`)
Parallel sub-tasks via ThreadPoolExecutor. Three tool access profiles: researcher
(read-only), builder (can modify workflows), validator (can execute + analyze).
Max 3 concurrent, 60s timeout, results in original order.

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

**Decision: HTTP/WS is the primary transport. MCP is additive, not a replacement.**
The intelligence layers don't care which transport is underneath — that's the point.

### MCP Integration Strategy

We do NOT build a full MCP server from scratch. When MCP is added:
1. Create `transport/mcp_adapter.py` implementing the same interface as HTTP
2. Auto-detect: if MCP server is available, use it; otherwise fall back to HTTP
3. No tool changes needed — transport is invisible to the intelligence layers

### Supported Backends (Priority Order)
1. **Direct HTTP/WS** — ComfyUI's native API (current, always works)
2. **IO-AtelierTech MCP** — if artist already has it installed
3. **Comfy Pilot MCP** — alternative MCP implementation
4. **Custom thin MCP** — only if we need features others don't expose

---

## Implementation Roadmap

### ✅ Phase 1: Foundation (COMPLETE)
34 tools, agent loop, patch engine, session persistence, knowledge system, 169 tests.

### ✅ Phase 1.5: Brain Layer (COMPLETE)
18 brain tools: vision, planner, memory, orchestrator, optimizer, demo. 236 total tests.

### Phase 2: DISCOVER Enhancement ⭐ THE MOAT
**Goal:** Contextual, real-time ecosystem awareness matched to artist context.

**Tasks:**
1. 🔲 Build contextual recommendation engine (workflow context → relevant discoveries)
2. 🔲 Integrate CivitAI API (community models, ratings, trending)
3. 🔲 Build freshness tracker (when did we last scan? what's new since?)
4. 🔲 Model compatibility mapping (checkpoints ↔ nodes/samplers)
5. 🔲 Proactive surfacing logic (recommend when relevant, not firehose)

**Success Criteria:**
- "What models do I have for SDXL?" → instant, accurate answer from local index
- "Anything new for ControlNet this week?" → real-time search, filtered to relevant
- "Can I speed up this workflow?" → analyze pipeline, suggest swaps with reasoning

### Phase 3: VERIFY Enhancement
**Goal:** Prove changes work. Catch regressions. Build trust.

**Tasks:**
1. 🔲 Output capture system (hash outputs per workflow+seed)
2. 🔲 Same-seed A/B comparison runner
3. 🔲 Render time delta tracking
4. 🔲 Perceptual hash comparison for images
5. 🔲 Regression detection and flagging

### Phase 4: MCP Transport Adapter
**Goal:** Swappable transport without touching intelligence layers.

**Tasks:**
1. 🔲 Define abstract adapter interface in `transport/adapter.py`
2. 🔲 Refactor `comfy_api.py` to implement adapter interface
3. 🔲 Build MCP adapter implementing same interface
4. 🔲 Auto-detection: use MCP if available, fall back to HTTP

### Phase 5: Demo Shell (THE PRODUCT)
**Goal:** Interactive experience suitable for live demonstration to VFX artists.

**Tasks:**
1. 🔲 Rich CLI formatting (panels, tables, syntax highlighting)
2. 🔲 Demo mode: guided walkthrough of capabilities
3. 🔲 "Explain as you go" narration
4. 🔲 Scripted demo scenarios (lighting setup, model swap, ControlNet addition)

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
- Transport errors → retry with backoff, then surface to artist in plain language
- Patch validation errors → explain what went wrong and suggest alternatives
- Missing model errors → trigger DISCOVER layer to find alternatives
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
