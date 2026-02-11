# Brain Layer Design — ComfyUI SUPER DUPER Agent v0.2.0

> **Date:** 2026-02-11
> **Status:** Design — pending implementation
> **Author:** Brainstorm session (human + Claude)

---

## Executive Summary

Add a **Brain Layer** on top of the existing four intelligence layers (UNDERSTAND,
DISCOVER, PILOT, VERIFY) to transform the agent from a tool-using assistant (3.9/10
autonomy) into a goal-driven, self-improving creative co-pilot with eyes.

**Architecture:** Hybrid B+C — modules today, agent-ready seams for Agent SDK
extraction tomorrow. Six new modules, ~18 new tools, bringing the total to ~52.

**The six capabilities:**
1. **Vision** — see and critique generated images via Claude Vision
2. **Planner** — decompose goals into tracked sub-tasks
3. **Memory** — learn from outcomes, recommend what works
4. **Orchestrator** — coordinate parallel sub-tasks
5. **Optimizer** — TensorRT/CUTLASS-aware performance engineering
6. **Demo** — guided walkthroughs for streams/podcasts

---

## Architecture

### Hybrid B+C Pattern

Every brain capability lives in `agent/brain/` as a module registering tools through
the existing pattern (`TOOLS` list + `handle()` function). Each module also defines an
**Agent Contract** — a clean interface boundary for future Agent SDK extraction.

```
agent/brain/
  __init__.py          # registers all brain tools
  _protocol.py         # shared BrainMessage format
  vision.py            # image analysis, quality feedback
  planner.py           # goal decomposition, progress tracking
  memory.py            # outcome recording, pattern learning
  orchestrator.py      # parallel sub-task coordination
  optimizer.py         # GPU-aware performance optimization
  demo.py              # guided demo scenarios
```

**Design constraint:** Every brain module is stateless per-call but state-aware via
persistence. Planner reads/writes goal state to disk. Memory reads/writes outcomes
to disk. Vision and Optimizer are purely functional. This enables future extraction
into separate Agent SDK agents without rewriting.

### Protocol (`_protocol.py`)

Shared message format for brain-to-brain communication:

```python
@dataclass
class BrainMessage:
    source: str          # "planner", "vision", "memory", "orchestrator"
    target: str          # who it's for
    msg_type: str        # "request", "result", "status", "error"
    payload: dict        # the actual data
    correlation_id: str  # links request -> response
    timestamp: float
```

Today: dicts passed between functions (same process).
Tomorrow: serialized JSON between Agent SDK agents (same structure, different transport).

### Full Architecture Diagram

```
+-------------------- SUPER DUPER AGENT v0.2.0 ----------------------+
|                                                                      |
|  BRAIN LAYER (new)                                                   |
|  +--------+ +--------+ +--------+ +-------+ +-------+ +--------+    |
|  |PLANNER | | VISION | | MEMORY | | ORCH  | |OPTIM  | | DEMO   |    |
|  |4 tools | |3 tools | |3 tools | |2 tools| |4 tools| |2 tools |    |
|  +---+----+ +---+----+ +---+----+ +---+---+ +---+---+ +---+----+    |
|      +----------+------+---+----------+-----+---+---------+         |
|                   _protocol.py (BrainMessage)                        |
|      +--------------+--+-------+---------------+                     |
|  INTELLIGENCE LAYERS (existing, 34 tools)      |                     |
|  +---------+ +----------+ +---------+ +--------+----+                |
|  |UNDERSTAND| | DISCOVER | |  PILOT  | |  VERIFY    |                |
|  | 13 tools | |  5 tools | | 13 tools| |  3 tools   |                |
|  +---------+ +----------+ +---------+ +------------+                |
|                            |                                         |
|                  +---------v---------+                                |
|                  |    TRANSPORT      |                                |
|                  +---------+---------+                                |
+--------------------+-------+-----------------------------------------+
                  +--+-------v------+
                  | ComfyUI Instance|
                  +-----------------+

TOTAL: ~52 tools (34 existing + 18 brain)
```

---

## Module Designs

### 1. Vision (`brain/vision.py`) — 3 tools

**Purpose:** Give the agent eyes. See generated images, critique quality,
compare before/after.

| Tool | Description |
|------|-------------|
| `analyze_image` | Feed image to Claude Vision. Returns: quality score, artifact detection (banding, color shift, anatomy), composition notes, prompt adherence |
| `compare_outputs` | Same-seed A/B comparison. Two image paths -> what changed, improved or not, specific differences |
| `suggest_improvements` | Given image + workflow, suggest parameter tweaks (CFG, steps, sampler, prompt) |

**Implementation:**
- Vision calls use a *separate* Claude API call with the image (not the main
  conversation context — images are large and would blow up the context window)
- Uses `anthropic.Anthropic().messages.create()` with image content blocks
- Returns structured JSON, not free-form text
- Image paths come from `get_history` (execution outputs) or user-provided paths

**The feedback loop:**
```
execute_workflow -> get output path -> analyze_image -> suggest_improvements
-> apply changes -> execute again -> compare_outputs -> record_outcome
```

**Agent Contract:** `VisionAgent` receives image paths + workflow context, returns
structured analysis. No PILOT or DISCOVER access — pure evaluation.

### 2. Planner (`brain/planner.py`) — 4 tools

**Purpose:** Turn high-level goals into tracked sub-task sequences.

| Tool | Description |
|------|-------------|
| `plan_goal` | Decompose goal string into ordered sub-tasks with dependencies. Persists to `sessions/{name}_goals.json` |
| `get_plan` | Return current plan with step statuses (pending/active/done/failed) |
| `complete_step` | Mark step done, record what was accomplished, trigger next |
| `replan` | Revise remaining steps when context changes, without losing progress |

**Decomposition strategy:** Template-based pattern matching (deterministic, fast):

```python
GOAL_PATTERNS = {
    "build_workflow": [
        "identify_base_model",
        "select_template",
        "configure_base_params",
        "add_specializations",
        "validate_pipeline",
        "test_execute",
        "evaluate_output",
        "iterate_or_complete",
    ],
    "optimize_workflow": [
        "profile_current",
        "identify_bottlenecks",
        "rank_optimizations",
        "apply_top_optimization",
        "benchmark_comparison",
        "iterate_or_complete",
    ],
    "debug_workflow": [
        "reproduce_issue",
        "isolate_failing_node",
        "check_inputs_outputs",
        "identify_fix",
        "apply_fix",
        "verify_fix",
    ],
    "swap_model": [
        "identify_current_model",
        "find_alternatives",
        "check_compatibility",
        "apply_swap",
        "same_seed_comparison",
    ],
}
```

Falls back to generic 4-step plan if no pattern matches:
understand -> modify -> validate -> execute.

**State:** Goals persist as JSON alongside sessions. Resume a session and the
agent knows "we were on step 5 of building the portrait pipeline."

**Agent Contract:** `PlannerAgent` — read access to UNDERSTAND tools only.
No write access to workflows.

### 3. Memory (`brain/memory.py`) — 3 tools

**Purpose:** Record every execution outcome. Learn patterns. Recommend
what works for this artist on this machine.

| Tool | Description |
|------|-------------|
| `record_outcome` | Store execution result: workflow fingerprint, params, model combo, quality score, render time, vision notes, user feedback |
| `get_learned_patterns` | Query outcome history: "What worked for SDXL portraits?" Returns aggregated insights |
| `get_recommendations` | Given current workflow, cross-reference outcomes to suggest improvements |

**Outcome record schema:**

```json
{
  "timestamp": "2026-02-11T...",
  "session": "portrait-project",
  "workflow_hash": "sha256:...",
  "key_params": {
    "model": "sdxl_base_1.0.safetensors",
    "steps": 20,
    "cfg": 7.0,
    "sampler": "euler",
    "scheduler": "normal",
    "resolution": [1024, 1024]
  },
  "model_combo": ["sdxl_base", "depth_controlnet_v1.1", "detail_lora_v2"],
  "render_time_s": 12.4,
  "quality_score": 0.82,
  "vision_notes": ["slight banding in gradient", "good composition", "accurate prompt adherence"],
  "user_feedback": "positive",
  "output_hash": "sha256:..."
}
```

**Storage:** Append-only JSONL in `sessions/{name}_outcomes.jsonl`.
One line per execution. Simple, greppable, no database needed.

**Pattern detection** (aggregation, not ML):
- Group by model combo -> average quality scores -> surface top combos
- Group by sampler+steps -> find efficient frontier (quality vs speed)
- Track render times -> flag regressions ("this used to be 8s, now 15s")
- Cross-reference user feedback -> weight quality scores

**Implicit feedback:** Agent infers from natural language.
"That looks great" -> positive. "Undo that" -> negative. "Try something different" -> neutral.

**Agent Contract:** `ChroniclerAgent` — write-only to outcomes, read-only from
history. No tool execution. Pure observation and recall.

### 4. Orchestrator (`brain/orchestrator.py`) — 2 tools

**Purpose:** Coordinate parallel sub-tasks when the planner identifies
independent work.

| Tool | Description |
|------|-------------|
| `spawn_subtask` | Launch focused sub-task in background thread with filtered tool access. Returns task ID. |
| `check_subtasks` | Poll status of running subtasks. Returns results or progress. |

**When it fires:** Only when planner identifies parallelizable steps. Not every turn.

Example:
```
Goal: "Build a ControlNet depth pipeline for portraits"
Plan step 3: "Add specializations"
  -> Sub-tasks (parallel):
    a. Research best depth preprocessor (DISCOVER tools only)
    b. Find compatible ControlNet model (DISCOVER tools only)
    c. Check installed upscale nodes (UNDERSTAND tools only)
  -> ThreadPoolExecutor, max 3 workers
  -> Results aggregated, planner advances
```

**Today:** Sub-tasks are function calls in threads with filtered tool access.
**Tomorrow:** Agent SDK sub-agents with isolated context windows.

**Safety:**
- Max 3 concurrent sub-tasks
- Sub-tasks are READ-ONLY by default (no PILOT tools unless explicit)
- 60-second timeout per sub-task
- One retry on failure, then report to planner

**Agent Contract:** `OrchestratorAgent` — spawns and monitors sub-agents.
Never executes tools directly. Pure coordination.

### 5. Optimizer (`brain/optimizer.py`) — 4 tools

**Purpose:** GPU-aware performance engineering. Knows TensorRT, CUTLASS,
VRAM management, and ComfyUI execution characteristics.

| Tool | Description |
|------|-------------|
| `profile_workflow` | Analyze execution characteristics: estimated VRAM peak, node bottlenecks, GPU vs CPU bound analysis |
| `suggest_optimizations` | Ranked optimization opportunities: TensorRT, CUDA graphs, batch size, precision, tiling, sampler efficiency |
| `check_tensorrt_status` | Scan for TRT node packs, check engine cache, report compatibility |
| `apply_optimization` | Swap standard nodes for optimized equivalents, adjust precision, add tiling |

**GPU profiles (built-in knowledge):**

```python
GPU_PROFILES = {
    "RTX 4090": {
        "vram_gb": 24,
        "cuda_cores": 16384,
        "tensor_cores": 512,       # Ada 4th gen
        "trt_supported": True,
        "cutlass_tier": "sm_89",   # Ada arch
        "sweet_spots": {
            "sd15_batch": 4,
            "sdxl_batch": 2,
            "sdxl_trt_batch": 4,
            "flux_batch": 1,
            "max_resolution_no_tiling": [1536, 1536],
        },
    },
    "RTX 4080": { ... },
    "RTX 3090": { ... },
    "RTX 3080": { ... },
}
```

GPU auto-detected via `get_system_stats` (ComfyUI API reports GPU name + VRAM).

**TensorRT integration:**

ComfyUI has `ComfyUI_TensorRT` node pack. Optimizer orchestrates existing nodes:

1. Detect if TRT pack installed (via `list_custom_nodes`)
2. Check for cached engines matching checkpoint hash
3. Recommend: "SDXL base can be TRT-compiled for ~2.5x faster inference"
4. Apply: Swap `CheckpointLoaderSimple` -> `TensorRTLoader`, preserve wiring
5. Validate: Same-seed comparison via Vision to confirm quality preserved

**CUTLASS awareness:**
- Track xformers vs torch SDPA (via system stats)
- Flash Attention availability for current CUDA arch
- Custom CUTLASS kernels in installed node packs

**Optimization ranking (impact/effort):**

```
HIGH IMPACT / LOW EFFORT (do first):
  fp16 precision                    ~30% speed, free
  Optimal batch size for VRAM       ~20% speed, free
  VAE tiling for large images       prevents OOM, free

HIGH IMPACT / MEDIUM EFFORT:
  TensorRT engine compilation       ~2-3x speed, 5min build
  CUDA graphs (if nodes support)    ~15% speed

MEDIUM IMPACT:
  Sampler selection (DPM++ 2M)      quality tradeoff
  Step count optimization           quality tradeoff

SITUATIONAL:
  Model offloading                  prevents OOM in multi-model
  ControlNet resolution tuning      speed vs precision
```

**Agent Contract:** `OptimizerAgent` — read access to UNDERSTAND + VERIFY,
write access to PILOT. Always runs same-seed comparison before/after.

### 6. Demo (`brain/demo.py`) — 2 tools

**Purpose:** Guided walkthroughs for live demos, streams, podcasts.

| Tool | Description |
|------|-------------|
| `start_demo` | Activate demo mode with a named scenario. Agent narrates actions, explains tool calls in artist terms, paces for audience. |
| `demo_checkpoint` | Mark demo milestone. Summarize what happened, preview what's next. |

**Demo scenarios:**

```python
DEMO_SCENARIOS = {
    "model_swap": {
        "title": "Upgrading Your Pipeline",
        "narrative": "Let's take your SD 1.5 workflow and upgrade it to SDXL...",
        "steps": ["analyze_current", "find_upgrade", "apply_swap", "compare_results"],
    },
    "speed_run": {
        "title": "Making It Fast",
        "narrative": "Your workflow takes 15 seconds. Let's get it under 5...",
        "steps": ["profile", "identify_bottleneck", "apply_trt", "benchmark"],
    },
    "controlnet_add": {
        "title": "Adding ControlNet Guidance",
        "steps": ["explain_controlnet", "find_nodes", "wire_up", "test_run"],
    },
    "full_pipeline": {
        "title": "From Zero to Rendered",
        "steps": ["choose_template", "customize", "optimize", "execute", "evaluate"],
    },
}
```

**Demo mode modifies agent behavior:**
- Longer, explanatory responses (artist audience)
- Narration of tool calls ("I'm checking what models you have installed...")
- Pause points between major steps
- Rich formatting (panels, tables, before/after)
- Milestone summaries

**Agent Contract:** `DemoAgent` — has full tool access but operates in
narration mode. Wraps normal agent behavior with pacing and explanation.

---

## Implementation Order

These can be built in parallel (6 independent modules), but the dependency
graph for integration testing suggests this order:

```
Phase 1 (Foundation):
  1. _protocol.py        # Shared message format (everything depends on this)
  2. brain/__init__.py    # Registration (same pattern as tools/__init__.py)

Phase 2 (Core capabilities, parallel):
  3. vision.py            # Eyes (enables feedback loop)
  4. planner.py           # Brain (enables goal tracking)
  5. memory.py            # Long-term memory (enables learning)

Phase 3 (Coordination):
  6. orchestrator.py      # Parallel execution (uses planner)
  7. optimizer.py         # Performance (uses vision for validation)

Phase 4 (Experience):
  8. demo.py              # Showmanship (uses everything)

Phase 5 (Integration):
  9. Update main.py       # Wire brain tools into agent loop
  10. Update cli.py       # Add demo command, optimizer flags
  11. Update system_prompt.py  # Brain-aware prompt engineering
  12. Update CLAUDE.md     # Document new architecture
```

## Testing Strategy

Same approach as existing tests: 100% mocked, no API keys or ComfyUI needed.

- Vision: mock Claude API call, verify structured output parsing
- Planner: test decomposition patterns, state persistence, replanning
- Memory: test outcome recording, pattern aggregation, recommendations
- Orchestrator: test parallel dispatch, timeout handling, result assembly
- Optimizer: test GPU profile matching, optimization ranking, TRT detection
- Demo: test scenario loading, checkpoint pacing, narration mode

Target: ~50-60 new tests (bringing total to ~220-230).

---

## Agent Contract Summary (Future SDK Extraction)

| Module | Reads | Writes | Spawns |
|--------|-------|--------|--------|
| Vision | Images, workflow context | Analysis results | None |
| Planner | UNDERSTAND tools, goal state | Goal state files | None |
| Memory | Outcome history | Outcome records | None |
| Orchestrator | Planner state | None | Sub-tasks |
| Optimizer | UNDERSTAND, VERIFY, GPU stats | PILOT (optimizations) | Vision (for validation) |
| Demo | Everything | Demo state | None |

Each contract defines: inputs, outputs, allowed tool access, and state access.
When extracted to Agent SDK, these become the agent's tool allowlist.

---

## Success Criteria

**Vision:** Agent can look at a generated image and say "The composition is good
but there's banding in the gradient — bumping steps from 20 to 30 should fix it."

**Planner:** User says "Build me a Flux portrait pipeline with ControlNet depth"
and the agent creates an 8-step plan, tracks progress, handles failures.

**Memory:** After 10 sessions, the agent says "For SDXL portraits, DPM++ 2M Karras
at 25 steps with CFG 7.5 has consistently given you the best results."

**Orchestrator:** Research tasks (find models, check nodes) run in parallel,
cutting multi-step discovery from 30s to 10s.

**Optimizer:** Agent profiles a 15-second workflow and says "TensorRT will get this
under 5 seconds. Want me to set it up?" Then does it and proves it with a benchmark.

**Demo:** A non-technical artist can watch the agent modify a workflow, optimize it,
and explain every step — suitable for a live stream or podcast demo.

---

## Non-Goals

- We do NOT build a training/fine-tuning system
- We do NOT replace ComfyUI's built-in queue management
- We do NOT build a GUI (CLI with rich formatting is the interface)
- We do NOT require a database (JSONL + JSON files are sufficient)
- We do NOT build the Agent SDK extraction in this phase (just the seams)
