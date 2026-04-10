# CLAUDE.md — ComfyUI Comfy Cozy Agent

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
python -m pytest tests/ -v                 # 2000+ tests, all mocked, <60s
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
4. When something fails, read the error, check node compatibility, suggest fixes.
5. If ComfyUI is not running, say so immediately. Most tools require it.
6. Prefer `get_node_info` over memory for node interfaces. It's always current.
7. Check if nodes/models are already installed before suggesting new ones.
8. Log key decisions to session notes (`add_note`) for continuity.
9. Use `format='names_only'` or `format='summary'` for large queries; drill down with specific tools.
10. Before executing, use `validate_before_execute` to catch errors early.
11. Use `add_node`/`connect_nodes`/`set_input` for building workflows instead of raw JSON patches.
12. Never generate entire workflows from scratch. Make surgical, validated modifications.
13. Every patch is validated before application. No exceptions.

## Tool Overview (85 tools)

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
  cli.py           # Typer CLI (run, mcp commands)
  mcp_server.py    # MCP server exposing all 85 tools
  config.py        # .env loading (ANTHROPIC_API_KEY, COMFYUI_DATABASE, etc.)
  system_prompt.py # Session-aware prompt builder + knowledge detection
  tools/           # Intelligence layer (56 tools, TOOLS+handle() pattern)
  brain/           # Brain layer (27 tools, BrainAgent SDK pattern)
  profiles/        # YAML model profiles (Flux, SDXL, LTX-2, WAN 2.x + architecture fallbacks)
  schemas/         # Schema system (loader, validator, generator)
  agents/          # MoE specialists (intent_agent, verify_agent, router)
  knowledge/       # Markdown reference files (loaded by keyword triggers)
  memory/          # Session persistence (JSONL outcomes, JSON state)
  templates/       # Starter workflow JSON files
tests/             # 2000+ tests, all mocked, pytest + pytest-asyncio
```

**Tool module pattern:** Every module in `tools/` and `brain/` exports `TOOLS: list[dict]` + `handle(name, tool_input) -> str`. Registration in `tools/__init__.py` and `brain/__init__.py`.

**Exception — `cognitive/tools/`:** This layer intentionally uses standalone async functions (e.g. `analyze_workflow()`, `execute_workflow()`) rather than the TOOLS+handle() pattern. Cognitive tools are consumed directly by the cognitive pipeline (`cognitive/pipeline/`) — they are not registered in the MCP/agent tool registry and are never called through `handle()`. This is by design: the cognitive layer is forbidden from importing `agent.*` to keep the dependency boundary clean.

## Key Conventions

- **Deterministic JSON**: `sort_keys=True` everywhere (He2025 pattern). Use `_util.py:to_json()`.
- **Line length**: 99 chars (ruff config in pyproject.toml)
- **All tests mocked**: No ComfyUI server or API key needed. HTTP via `unittest.mock.patch`.
- **Config via .env**: `ANTHROPIC_API_KEY` (required), `COMFYUI_DATABASE` (default `G:/COMFYUI_Database`), `COMFYUI_HOST`/`COMFYUI_PORT`.
- **Custom_Nodes**: Capital C, capital N (ComfyUI convention).
- **asyncio_mode = "auto"**: In pyproject.toml for pytest-asyncio.
- **Python 3.11+**: Match ComfyUI's requirement. Type hints everywhere. `httpx` for HTTP.
- **Thread safety**: workflow_patch, orchestrator, demo, intent_collector, iteration_accumulator use `threading.Lock`.
- **Path sanitization**: `_util.validate_path()` blocks access outside allowed directories.
- **Error messages**: Never show raw tracebacks. Translate to human language.

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

## Current TODO (Phase 7)

Phase 6 complete. All 8 items verified passing (3579 tests, 30 pipeline tests).

**Completed (Phase 6 — archived):**
- ~~test_health.py mock leak~~ — fixed (6/6 passing)
- ~~Windows grep portability~~ — fixed (pathlib.rglob)
- ~~Default executor wire~~ — EXECUTE calls real `execute_workflow` when `config.executor` is None
- ~~Template loading~~ — COMPOSE loads from `agent/templates/` with SD1.5 fallback
- ~~Default evaluator~~ — rule-based QualityScore (0.7 success / 0.1 failure)
- ~~ExperienceChunk parameter shape~~ — flat `parameters=params`
- ~~Post-COMPOSE diagnostic~~ — `analyze_workflow` warns on zero-node workflows
- ~~`create_default_pipeline()`~~ — bootstrap factory in `cognitive/pipeline/__init__.py`

**Phase 7 — Next:**
1. Vision-based evaluator — replace rule-based 0.7/0.1 with `analyze_image` scoring
2. Auto-retry loop — re-COMPOSE when `quality.overall < threshold` (stub exists in pipeline)
3. MCP resource support — expose workflow state as MCP resources
4. Integration test harness — `@pytest.mark.integration` for live ComfyUI tests

## Non-Goals

- We do not generate workflows from scratch. We modify existing ones.
- We do not replace the ComfyUI GUI. We augment it.
- We do not train or fine-tune models. We help artists find and use them.
- We do not optimize for developers. Every interaction assumes a VFX artist.
