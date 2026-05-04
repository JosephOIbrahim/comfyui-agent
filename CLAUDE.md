# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> AI co-pilot for VFX artists using ComfyUI. Driver, not generator.
> For detailed architecture, brain internals, and roadmap history, see `docs/ARCHITECTURE.md`.

## Identity

- **Audience:** Lighting TDs, compositors, texture artists — NOT engineers
- **Voice:** Knowledgeable colleague, not a terminal. Explain what and why.
- **Principle:** Small validated changes, never full workflow rewrites.

## Commands

```bash
pip install -e ".[dev]"                    # Install
agent run                                  # CLI agent (standalone fallback)
agent mcp                                  # MCP server (primary interface)
python -m pytest tests/ -v                 # All tests (~3600, all mocked, <60s)
python -m pytest tests/test_workflow_patch.py -v          # Single file
python -m pytest tests/test_workflow_patch.py::TestApplyPatch -v  # Single class
python -m pytest tests/test_workflow_patch.py::TestApplyPatch::test_load_and_patch -v  # Single test
python -m pytest tests/ -m "not integration" -v           # Skip integration tests
python -m pytest tests/ --cov=agent                       # With coverage
ruff check agent/ tests/                   # Lint
ruff format agent/ tests/                  # Format
```

## MCP Setup (Claude Code / Claude Desktop)

```json
{
  "mcpServers": {
    "comfyui-agent": {
      "command": "agent",
      "args": ["mcp"],
      "cwd": "G:/Comfy-Cozy"
    }
  }
}
```

## Tool Usage Rules

1. **NEVER claim to know about specific models from memory. ALWAYS use tools.** Model ecosystems change daily.
2. When asked "what model should I use for X?" -- search first (`discover`), recommend after.
3. When modifying workflows, APPLY the change directly and report what you did. Do NOT ask for permission -- act, then show the result. Use preview only when the user explicitly asks. Every change is reversible (`undo_workflow_patch`), so bias toward action.
4. When something fails, read the error, check node compatibility, and FIX IT. Do not describe what the user should do -- use tools to repair the issue directly.
5. When `validate_before_execute` reports missing nodes, call `repair_workflow(auto_install=true)` to install required packs automatically. Then re-validate. Then execute. One continuous flow, no stopping to ask.
6. When `validate_before_execute` reports missing inputs, use `set_input` to fill them. Wrong model names → `discover` the right model, then `set_input` to fix.
7. If ComfyUI is not running, say so immediately. Most tools require it.
8. Prefer `get_node_info` over memory for node interfaces. It's always current.
9. Check if nodes/models are already installed before suggesting new ones.
10. Log key decisions to session notes (`add_note`) for continuity.
11. Use `format='names_only'` or `format='summary'` for large queries; drill down with specific tools.
12. Before executing, use `validate_before_execute` to catch errors early. If errors found, FIX them, re-validate, and execute. Do not stop at validation.
13. Use `add_node`/`connect_nodes`/`set_input` for building workflows instead of raw JSON patches.
14. Never generate entire workflows from scratch. Make surgical, validated modifications.
15. Every patch is validated before application. No exceptions.

## Tool Overview (~85 dispatched tools: ~64 intelligence + ~7 brain TOOLS-list + ~22 stage; brain also exposes 9 BrainAgent SDK specialists)

| Category | Tools |
|----------|-------|
| **Live API** | `is_comfyui_running`, `get_all_nodes`, `get_node_info`, `get_system_stats`, `get_queue_status`, `get_history` |
| **Filesystem** | `list_custom_nodes`, `list_models`, `get_models_summary`, `read_node_source` |
| **Workflow** | `load_workflow`, `validate_workflow`, `get_editable_fields` |
| **Editing** | `apply_workflow_patch`, `preview_workflow_patch`, `undo_workflow_patch`, `get_workflow_diff`, `save_workflow`, `reset_workflow` |
| **Semantic Build** | `add_node`, `connect_nodes`, `set_input` |
| **Execution** | `validate_before_execute`, `execute_workflow`, `get_execution_status`, `execute_with_progress` |
| **Discovery** | `discover`, `find_missing_nodes`, `check_registry_freshness`, `get_install_instructions` |
| **Provision** | `install_node_pack`, `download_model`, `uninstall_node_pack` |
| **CivitAI** | `get_civitai_model`, `get_trending_models` |
| **Model Compat** | `identify_model_family`, `check_model_compatibility` |
| **Node Replace** | `get_node_replacements`, `check_workflow_deprecations`, `migrate_deprecated_nodes` |
| **Templates** | `list_workflow_templates`, `get_workflow_template` |
| **Session** | `save_session`, `load_session`, `list_sessions`, `add_note` |
| **Vision** | `analyze_image`, `compare_outputs`, `suggest_improvements`, `hash_compare_images` |
| **Planner** | `plan_goal`, `get_plan`, `complete_step`, `replan` |
| **Memory** | `record_outcome`, `get_learned_patterns`, `get_recommendations`, `detect_implicit_feedback` |
| **Orchestrator** | `spawn_subtask`, `check_subtasks` |
| **Optimizer** | `profile_workflow`, `suggest_optimizations`, `check_tensorrt_status`, `apply_optimization` |
| **Demo** | `start_demo`, `demo_checkpoint` |
| **Intent** | `capture_intent`, `get_current_intent` |
| **Iteration** | `start_iteration_tracking`, `record_iteration_step`, `finalize_iterations` |
| **Metadata** | `write_image_metadata`, `read_image_metadata`, `reconstruct_context` |

## Artistic Intent Translation

| Artist Says | Parameter Direction |
|------------|-------------------|
| "dreamier" / "softer" | Lower CFG (5-7), increase steps, DPM++ 2M Karras |
| "sharper" / "crisper" | Higher CFG (8-12), Euler or DPM++ SDE |
| "more photorealistic" | CFG 7-10, realistic checkpoint, negative: "cartoon, anime" |
| "more stylized" | Lower CFG (4-6), artistic checkpoint or LoRA |
| "faster" | Fewer steps (15-20), LCM/Lightning/Turbo, smaller resolution |
| "higher quality" | More steps (30-50), hires fix, upscaler |
| "more variation" | Higher denoise, different seed, lower CFG |
| "less variation" | Lower denoise, same seed, higher CFG |

## Model Family Quick Reference

| Family | Resolution | CFG Range | Negative Prompt | Key Notes |
|--------|-----------|-----------|----------------|-----------|
| **SD 1.5** | 512x512 | 7-12 | Yes, important | Massive LoRA ecosystem, fast |
| **SDXL** | 1024x1024 | 5-9 | Yes, less critical | Base + refiner, better hands |
| **Flux** | 512-1024 | 1.0 (guidance) | No | FluxGuidance node, T5 encoder |
| **SD3** | 1024x1024 | 5-7 | Optional | Triple text encoder (CLIP-G, CLIP-L, T5) |

Never mix model families (e.g., SD1.5 LoRAs with SDXL checkpoints). ControlNets must match base family.

## Workflow JSON Schema (API Format)

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

Patch engine operates on this format exclusively. Three input formats handled transparently (API / UI+API / UI-only); detection in `workflow_parse.py:_extract_api_format()`.

## Project Structure

```
agent/
  main.py          # Agent loop (streaming, context management, retry)
  cli.py           # Typer CLI (run, mcp, inspect, parse commands)
  mcp_server.py    # MCP server exposing all tools
  config.py        # .env loading (ANTHROPIC_API_KEY, COMFYUI_DATABASE, etc.)
  system_prompt.py # Session-aware prompt builder + knowledge detection
  tools/           # Intelligence layer (~53 tools, TOOLS+handle() pattern)
    __init__.py    # Central dispatch: _HANDLERS map, lazy brain loading, gate integration
    workflow_patch.py  # Session-scoped workflow state, undo history, semantic build (add_node, connect_nodes)
    workflow_parse.py  # Load/analyze workflows (API/UI format detection)
    comfy_api.py   # REST calls to ComfyUI
    comfy_execute.py   # Queue prompts, stream progress via websocket
    comfy_discover.py  # Search CivitAI, HuggingFace, ComfyUI Manager
    _util.py       # to_json() (deterministic), validate_path() (sandbox)
  brain/           # Brain layer (~27 tools, BrainAgent SDK pattern)
    _sdk.py        # BrainAgent base class, BrainConfig DI container, auto-registration
  stage/           # Stage layer (~23 tools, USD/LIVRPS composition)
    stage_tools.py     # stage_read, stage_write, stage_add_delta, stage_list_deltas
    provision_tools.py # Provision operations
    foresight_tools.py # Predictive analysis
  profiles/        # YAML model profiles (Flux, SDXL, LTX-2, WAN 2.x + architecture fallbacks)
  schemas/         # Schema system (loader, validator, generator)
  agents/          # MoE specialists (intent_agent, verify_agent, router)
  knowledge/       # Markdown reference files (loaded by keyword triggers)
  memory/          # Session persistence (JSONL outcomes, JSON state)
  templates/       # Starter workflow JSON files
cognitive/         # Standalone library — does NOT import agent.* (clean dependency boundary)
  core/            # CognitiveGraphEngine, DeltaLayer, LIVRPS composition
  experience/      # ExperienceChunk, JSONL accumulator, WorkflowSignature hashing
  pipeline/        # Autonomous generation (create_default_pipeline(), PipelineConfig)
  prediction/      # CWM, arbiter, counterfactuals
  tools/           # Standalone async functions (NOT in MCP registry, consumed by pipeline only)
  transport/       # Events, interrupts
tests/             # ~3600 tests, all mocked, pytest + pytest-asyncio
  conftest.py      # autouse fixtures: _reset_conn_session, reset_workflow_state
  fixtures/        # Shared test data (sample workflows, fake images)
```

**Tool module pattern:** Every module in `tools/` and `brain/` exports `TOOLS: list[dict]` + `handle(name, tool_input) -> str`. Registration in `tools/__init__.py` and `brain/__init__.py`.

## Architecture

### Three-Layer Tool Dispatch

```
agent/tools/__init__.py:handle()  ←  Central dispatcher
  ├── Intelligence layer (agent/tools/*.py)  — TOOLS+handle() pattern, ~53 tools
  ├── Brain layer (agent/brain/*.py)         — BrainAgent subclasses, ~27 tools, lazy-loaded
  └── Stage layer (agent/stage/*.py)         — TOOLS+handle() pattern, ~23 tools
```

All tool modules export `TOOLS: list[dict]` (Anthropic schema) + `handle(name, tool_input) -> str`. Brain modules inherit from `BrainAgent` (in `brain/_sdk.py`) which auto-registers subclasses via `__init_subclass__`. Modules that fail to import are logged and skipped (graceful degradation).

### Session Isolation

Workflow state is per-connection via `_conn_session` ContextVar in `workflow_patch.py`. Each MCP connection, sidebar, and CLI session gets its own `WorkflowSession` containing: `current_workflow` (API-format dict), `history` (undo stack, 50-item limit), and `_engine` (CognitiveGraphEngine if available). Tests use autouse fixtures to snapshot/restore ContextVar and workflow state between tests.

### Cognitive Layer Boundary

`cognitive/` is a standalone library that does NOT import `agent.*`. It provides:
- **LIVRPS delta composition** (`core/delta.py`): DeltaLayer with Opinion tiers (P < R < V < I < L < S) and SHA-256 integrity
- **CognitiveGraphEngine** (`core/graph.py`): syncs with workflow_patch state
- **Experience persistence** (`experience/`): JSONL accumulator, WorkflowSignature hashing
- **Autonomous pipeline** (`pipeline/`): `create_default_pipeline()`, PipelineConfig, rule-based QualityScore

Cognitive tools (`cognitive/tools/`) use standalone async functions — they are NOT in the MCP registry.

### BrainAgent SDK

Brain agents (`brain/*.py`) inherit from `BrainAgent` (`brain/_sdk.py`). Dependency injection via `BrainConfig` dataclass provides: `to_json`, `validate_path`, `comfyui_url`, `custom_nodes_dir`, `models_dir`, `tool_dispatcher`, `get_workflow_state`, etc. Config auto-populated from `agent.config` + `agent.tools._util` via `get_integrated_config()` singleton.

## Key Conventions

- **Deterministic JSON**: `sort_keys=True` everywhere (He2025 pattern). Use `_util.py:to_json()`.
- **Line length**: 99 chars (ruff config in pyproject.toml)
- **All tests mocked**: No ComfyUI server or API key needed. HTTP via `unittest.mock.patch`.
- **Config via .env**: `ANTHROPIC_API_KEY` (required), `COMFYUI_DATABASE` (default `G:/COMFYUI_Database`), `COMFYUI_HOST`/`COMFYUI_PORT`, `LLM_PROVIDER` (anthropic|openai|gemini|ollama), `AGENT_MODEL`, `BRAIN_ENABLED`, `GATE_ENABLED`.
- **Custom_Nodes**: Capital C, capital N (ComfyUI convention).
- **asyncio_mode = "auto"**: In pyproject.toml for pytest-asyncio.
- **Python 3.10+**: Matches `pyproject.toml` `requires-python`. Type hints everywhere. `httpx` for HTTP.
- **Thread safety**: workflow_patch, orchestrator, demo, intent_collector, iteration_accumulator use `threading.Lock`.
- **Path sanitization**: `_util.validate_path()` blocks access outside allowed directories.
- **Error messages**: Never show raw tracebacks. Translate to human language.

## Test Patterns

- **All tests are mocked** — no ComfyUI server or API key needed. HTTP mocked via `unittest.mock.patch`.
- **autouse fixtures** in `conftest.py`: `_reset_conn_session` (snapshot/restore ContextVar) and `reset_workflow_state` (deep-copy/restore workflow_patch state). These ensure test isolation.
- **Common fixtures**: `sample_workflow` (minimal SD1.5 API-format dict), `sample_workflow_file` (JSON on disk), `fake_image` (tiny valid PNG).
- **Pattern**: Load workflow → call `handle()` → `json.loads()` result → assert fields. Tools return JSON strings.
- **Integration tests**: marked with `@pytest.mark.integration`, excluded by default with `-m "not integration"`.

## Forbidden Operations

- Never delete all nodes
- Never replace the entire workflow JSON
- Never modify node types (only inputs/connections)
- Never apply unvalidated patches
- If a change would break the DAG, refuse and explain

## Git Authority Map

This repo uses an agent-managed git workflow. Claude Code agents operate under these rules every session.

### Autonomous (no per-call approval needed)

- `git status`
- `git diff` (any form)
- `git log` (read-only)
- `git branch --list`
- `git show`
- `git grep`
- Any pure read/inspection operation

### Authorized per-session under autobuild prompts

When a prompt explicitly grants session-level git authorization, the agent may run these in sequence without per-step approval:

- `git add` (staging specific files — never `git add -A`)
- `git commit` (with the exact message provided in the prompt)
- `git tag` (lightweight tags at prompt-specified milestones)

### Requires per-call approval from Joe, no exceptions

- `git push` (any form, any remote)
- `git reset` (any form)
- `git rebase`
- `git branch -D` (branch deletion)
- `git tag -d` (tag deletion)
- `git stash drop`
- Any `--force` flag
- Any operation touching origin or a remote

### Permanently forbidden

- `git push --force` (including `--force-with-lease`)
- `git reflog expire`
- `git filter-branch` / `git filter-repo`
- `rm -rf .git`
- Any history-rewriting operation

### Session workflow

- Autobuild prompts explicitly grant session-level authorization for the middle tier. Without that grant, only the autonomous tier runs.
- Every mutation step produces a verification output before moving on.
- Hard halts are non-negotiable: unexpected staging, unexpected diffs, test regressions below the last verified baseline, or non-zero exit codes all trigger an immediate STOP.
- Per C3: 3 retries max per step, then `BLOCKER.md`.
- Per C8: push to remote is always a separate, deliberate decision.

## Commit Messages

```
[UNDERSTAND] Add workflow pattern recognition for ControlNet pipelines
[PILOT] Fix patch validation for multi-output nodes
[DISCOVER] Integrate CivitAI trending models endpoint
[VERIFY] Add perceptual hash comparison for output images
[TEST] Add fixture for SDXL + ControlNet + IP-Adapter workflow
```

## Current TODO (Phase 7 / Cozy)

Phase 6 complete. Cozy persistence + harness shipped (4150+ tests passing).

**Completed (Phase 6 — archived):**
- ~~test_health.py mock leak~~ — fixed (6/6 passing)
- ~~Windows grep portability~~ — fixed (pathlib.rglob)
- ~~Default executor wire~~ — EXECUTE calls real `execute_workflow` when `config.executor` is None
- ~~Template loading~~ — COMPOSE loads from `agent/templates/` with SD1.5 fallback
- ~~Default evaluator~~ — rule-based QualityScore (0.7 success / 0.1 failure)
- ~~ExperienceChunk parameter shape~~ — flat `parameters=params`
- ~~Post-COMPOSE diagnostic~~ — `analyze_workflow` warns on zero-node workflows
- ~~`create_default_pipeline()`~~ — bootstrap factory in `cognitive/pipeline/__init__.py`

**Completed (Cozy — see `.claude/COZY_CONSTITUTION.md`):**
- ~~Stage persistence~~ — `STAGE_DEFAULT_PATH` cold-load + autosave timer + MCP atexit
- ~~Lazy experience-loop wiring~~ — `STAGE_AUTOLOAD_EXPERIENCE` invokes `create_default_pipeline()` on `ensure_stage()`
- ~~MCP resource support~~ — `stage://workflows`, `stage://experience`, `stage://agents`, `stage://scenes` exposed in `agent/mcp_server.py`
- ~~Stage event surface~~ — `CognitiveWorkflowStage.subscribe(callback)` registry; daemon-thread fan-out; failures isolated from writers
- ~~SCRIBE specialist~~ — chain terminator per Article II of the Cozy Constitution
- ~~Two new commandments~~ — `persistence_durability` (post-check), `self_healing_ladder` (classifier)
- ~~Long-running harness~~ — `agent/harness/cozy_loop.py` with checkpointing, self-healing ladder, optional `repair_fn` and MetaAgent Tier-1 dial integration
- ~~Moneta reference adapter~~ — `agent/integrations/moneta.py` bidirectional file-watch transport (placeholder for Moneta API)
- ~~Cozy MoE subagents~~ — `.claude/agents/cozy-{scout,architect,provisioner,forge,crucible,vision,scribe}.md`

**Phase 7 — Next:**
1. Vision-based evaluator — replace rule-based 0.7/0.1 with `analyze_image` scoring
2. Auto-retry loop — re-COMPOSE when `quality.overall < threshold` (stub exists in pipeline)
3. Integration test harness — `@pytest.mark.integration` for live ComfyUI tests
4. Real Moneta wire format — replace file-watch transport in `agent/integrations/moneta.py` with HTTP/RPC once API contract lands

## Cozy Environment Variables

```bash
STAGE_DEFAULT_PATH=/path/to/stage.usda     # cold-load + flush target; "" = in-memory
STAGE_AUTOSAVE_SECONDS=300                 # daemon Timer interval; 0 disables
STAGE_AUTOLOAD_EXPERIENCE=true             # wires the cognitive ExperienceAccumulator
MONETA_OUTBOX_DIR=/path/to/moneta/outbox   # enables the Moneta reference adapter
MONETA_INBOX_DIR=/path/to/moneta/inbox     # optional; enables bidirectional ingest
MONETA_POLL_SECONDS=2.0                    # inbox poll interval
```

## Cozy Autonomous Harness

```bash
# Smoke test the harness loop without ComfyUI (synthetic scores)
agent autonomous --execute-mode dry-run --hours 0.001 --max-experiments 5

# Real run against ComfyUI (mutates a workflow's steps/cfg/seed each iter)
agent autonomous --execute-mode real --workflow path/to/wf.json \
                 --hours 24 --max-experiments 1000 --session cozy_run

# Default mode (mock) — harness errors at first iteration unless callbacks
# are injected programmatically. Used by Python tests, not for live runs.
agent autonomous --hours 24
```

Execute modes:
- `mock` (default) — no callbacks; harness raises at first iteration.
- `dry-run` — real proposal cycle (steps/cfg/seed); synthetic axis scores;
  no ComfyUI contact.
- `real` — requires `--workflow PATH`; loads the workflow once, applies
  RFC6902 patches per iteration, executes via `execute_with_progress`,
  derives axis scores from `{status, total_time_s, outputs}`. Failures
  return zero scores so the ratchet rejects the experiment without
  halting; the circuit breaker's TRANSIENT classification keeps the
  harness retrying.

Per-iteration checkpoint to `STAGE_DEFAULT_PATH`. Halts only on TERMINAL
(constitution-violation / disk-full / repeated-RECOVERABLE>3) or budget
exhaustion. Writes `BLOCKER.md` on TERMINAL halt. See
`.claude/COZY_CONSTITUTION.md` Article III for the bounded-failure ladder.

## Integration tests (opt-in)

```bash
# 1k-iteration soak validating no thread/FD leaks under sustained load
python -m pytest tests/integration/test_cozy_soak.py -v
# Bidirectional Moneta round-trip via subprocess fake consumer
python -m pytest tests/integration/test_moneta_e2e.py -v
```

Both are marked `@pytest.mark.integration`, deselected by default. Run
manually before shipping changes that touch persistence, dispatchers,
or the Moneta adapter.

## Non-Goals

- We do not generate workflows from scratch. We modify existing ones.
- We do not replace the ComfyUI GUI. We augment it.
- We do not train or fine-tune models. We help artists find and use them.
- We do not optimize for developers. Every interaction assumes a VFX artist.
