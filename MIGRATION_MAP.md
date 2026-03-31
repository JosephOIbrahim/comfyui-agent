# MIGRATION MAP — Scaffolded Brain Phase 0

**Agent:** `[SCAFFOLD x SCOUT]`
**Date:** 2026-03-31
**Baseline:** 2140 passed, 156 skipped, 27 errors (all `test_provisioner.py` — `usd-core` not installed)
**Total Tools:** 108 (81 intelligence + 27 brain, includes 22 stage tools counted in intelligence)

---

## Baseline Test Summary

```
2140 passed | 156 skipped | 27 errors | ~55s runtime
```

The 27 errors are all in `tests/test_provisioner.py` — they require `usd-core` which is not installed in the current venv. These are **pre-existing** and not caused by any change. The effective passing baseline is **2140 tests**.

---

## Module Classifications

### Intelligence Layer (`agent/tools/`) — 16 modules, 81 tools

| Module | Tools | Classification | Rationale |
|--------|-------|---------------|-----------|
| `comfy_api.py` | `is_comfyui_running`, `get_all_nodes`, `get_node_info`, `get_queue_status`, `get_system_stats`, `get_history` | **PRESERVE** | Pure ComfyUI REST API wrappers. No mutation. |
| `comfy_inspect.py` | `list_custom_nodes`, `list_models`, `get_models_summary`, `read_node_source` | **PRESERVE** | Filesystem read-only. No mutation. |
| `workflow_parse.py` | `load_workflow`, `validate_workflow`, `get_editable_fields`, `classify_workflow`, `reconfigure_workflow` | **PRESERVE** | Read/analyze workflows. `load_workflow` also loads into patch state but does not mutate. |
| **`workflow_patch.py`** | `apply_workflow_patch`, `preview_workflow_patch`, `undo_workflow_patch`, `get_workflow_diff`, `save_workflow`, `reset_workflow`, `add_node`, `connect_nodes`, `set_input` | **TRANSFORM** | **Primary mutation target.** 9 tools. 6 perform destructive in-place mutation via `jsonpatch` or direct dict writes. These must be wrapped to route through `CognitiveGraphEngine`. |
| `comfy_execute.py` | `validate_before_execute`, `execute_workflow`, `execute_with_progress`, `get_execution_status` | **PRESERVE** | Submits workflow to ComfyUI. Reads `get_current_workflow()` from `workflow_patch`. No direct mutation. |
| `comfy_discover.py` | `discover`, `find_missing_nodes`, `check_node_updates`, `check_registry_freshness`, `get_install_instructions` | **PRESERVE** | Discovery/search. No mutation. |
| `comfy_provision.py` | `install_node_pack`, `download_model`, `uninstall_node_pack` | **PRESERVE** | Package management. No workflow mutation. |
| `civitai_api.py` | `get_civitai_model`, `get_trending_models` | **PRESERVE** | External API. No mutation. |
| `model_compat.py` | `identify_model_family`, `check_model_compatibility` | **PRESERVE** | Analysis only. |
| `node_replacement.py` | `get_node_replacements`, `check_workflow_deprecations`, `migrate_deprecated_nodes` | **PRESERVE** | `migrate_deprecated_nodes` patches workflows but uses `workflow_patch` internally — wrapping `workflow_patch` covers this. |
| `session_tools.py` | `save_session`, `load_session`, `list_sessions`, `add_note` | **PRESERVE** | Session I/O. Calls into `memory.session`. |
| `workflow_templates.py` | `list_workflow_templates`, `get_workflow_template` | **PRESERVE** | Read-only template access. |
| `verify_execution.py` | `verify_execution` | **PRESERVE** | Post-execution verification. |
| `github_releases.py` | `get_repo_releases` | **PRESERVE** | External API. |
| `pipeline.py` | `create_pipeline`, `run_pipeline`, `get_pipeline_status` | **PRESERVE** | Pipeline orchestration. Uses execute tools. |
| `image_metadata.py` | `write_image_metadata`, `read_image_metadata`, `reconstruct_context` | **PRESERVE** | Metadata I/O. No workflow mutation. |

### Brain Layer (`agent/brain/`) — 9 modules, 27 tools

| Module | Tools | Classification | Rationale |
|--------|-------|---------------|-----------|
| `vision.py` | `analyze_image`, `compare_outputs`, `suggest_improvements`, `hash_compare_images` | **PRESERVE** | Quality signal source. Wire as input to Experience in Phase 4. |
| `planner.py` | `plan_goal`, `get_plan`, `complete_step`, `replan` | **REPLACE** | Will be replaced by CWM (Phase 5). Keep interface, rebuild internals. |
| `memory.py` | `record_outcome`, `get_learned_patterns`, `get_recommendations`, `detect_implicit_feedback` | **EXTEND** | Storage backend for Experience Accumulator (Phase 4). Add structured schema on top. |
| `orchestrator.py` | `spawn_subtask`, `check_subtasks` | **EXTEND** | Will gain autonomous pipeline orchestration in Phase 6. |
| `optimizer.py` | `profile_workflow`, `suggest_optimizations`, `check_tensorrt_status`, `apply_optimization` | **PRESERVE** | GPU profiling. No changes needed. |
| `demo.py` | `start_demo`, `demo_checkpoint` | **PRESERVE** | Guided walkthroughs. |
| `iterative_refine.py` | `iterative_refine` | **PRESERVE** | Autonomous refinement loop. |
| `intent_collector.py` | `capture_intent`, `get_current_intent`, `classify_intent` | **PRESERVE** | Intent capture for metadata. |
| `iteration_accumulator.py` | `start_iteration_tracking`, `record_iteration_step`, `finalize_iterations` | **PRESERVE** | Refinement journey tracking. |

### Stage Layer (`agent/stage/`) — 5 tool modules, 22 tools + 15 internal modules

| Module | Tools | Classification | Rationale |
|--------|-------|---------------|-----------|
| `stage_tools.py` | `stage_read`, `stage_write`, `stage_add_delta`, `stage_rollback`, `stage_reconstruct_clean`, `stage_list_deltas` | **PRESERVE** | USD-native stage tools. These are the *existing* delta-layer implementation. CognitiveGraphEngine may eventually supersede but must not break these. |
| `foresight_tools.py` | `predict_experiment`, `record_experience`, `get_experience_stats`, `list_counterfactuals`, `get_prediction_accuracy` | **PRESERVE** | Existing CWM/experience tools. Phase 4-5 builds on top. |
| `compositor_tools.py` | `compose_scene`, `validate_scene`, `extract_conditioning`, `export_scene` | **PRESERVE** | Scene composition. |
| `provision_tools.py` | `provision_download`, `provision_verify`, `provision_status` | **PRESERVE** | Model provisioning via USD stage. |
| `hyperagent_tools.py` | `propose_improvement`, `check_evolution_tier`, `get_meta_history`, `get_calibration_stats` | **PRESERVE** | Meta-learning / evolution tracking. |

**Internal stage modules (no direct tool exposure):**

| Module | Classification | Rationale |
|--------|---------------|-----------|
| `cognitive_stage.py` | **PRESERVE** | USD-native CognitiveWorkflowStage. The Graph Engine is a *parallel* system, not a replacement. |
| `cwm.py` | **EXTEND** | Cognitive World Model. Phase 5 extends prediction logic. |
| `experience.py` | **EXTEND** | Experience storage. Phase 4 adds structured ExperienceChunks. |
| `arbiter.py` | **PRESERVE** | Simulation Arbiter. Phase 5 may extend delivery modes. |
| `ratchet.py` | **PRESERVE** | Autoresearch ratchet. Phase 6 integrates. |
| `counterfactuals.py` | **PRESERVE** | Counterfactual generation. |
| `model_registry.py` | **PRESERVE** | Model registration for provisioning. |
| `workflow_mapper.py` | **PRESERVE** | Workflow-to-USD mapping. |
| `workflow_signature.py` | **PRESERVE** | Workflow fingerprinting. |
| `compositor.py` | **PRESERVE** | Scene compositor internals. |
| `constitution.py` | **PRESERVE** | Safety constraints. |
| `anchors.py` | **PRESERVE** | Anchor prims for USD stage. |
| `scene_conditioner.py`, `scene_validator.py` | **PRESERVE** | Scene validation. |
| `creative_profiles.py`, `injection.py` | **PRESERVE** | Creative profile management. |
| `moe_dispatcher.py`, `moe_profiles.py` | **PRESERVE** | MoE routing (existing). |
| `hyperagent.py` | **PRESERVE** | HyperAgent evolution engine. |
| `morning_report.py` | **PRESERVE** | Session start report. |
| `program_parser.py` | **PRESERVE** | Program parsing. |
| `autoresearch_runner.py` | **PRESERVE** | Autoresearch pipeline. |

### Infrastructure (`agent/` root)

| Module | Classification | Rationale |
|--------|---------------|-----------|
| `mcp_server.py` | **PRESERVE** | MCP tool exposure. Registration unchanged. |
| `cli.py` | **PRESERVE** | CLI entry points. |
| `config.py` | **PRESERVE** | `.env` config loading. |
| `main.py` | **PRESERVE** | Agent loop. |
| `system_prompt.py` | **PRESERVE** | Prompt builder. |
| `workflow_session.py` | **PRESERVE** | Dict-like session state container. Engine instance will be stored here or in `SessionContext`. |
| `session_context.py` | **EXTEND** | Add `cognitive_graph_engine` field to `SessionContext` for engine persistence across tool calls. |
| `startup.py` | **PRESERVE** | Auto-initialization. |
| `errors.py` | **PRESERVE** | Error formatting. |
| `context.py` | **PRESERVE** | Context management. |
| `tool_scope.py` | **PRESERVE** | Tool access filtering. |
| `rate_limiter.py` | **PRESERVE** | Rate limiting. |
| `progress.py` | **PRESERVE** | Progress reporting. |
| `streaming.py` | **PRESERVE** | Streaming support. |
| `circuit_breaker.py` | **PRESERVE** | Circuit breaker. |
| `discovery_cache.py` | **PRESERVE** | Discovery result caching. |
| `logging_config.py` | **PRESERVE** | Logging setup. |
| `artifacts.py` | **PRESERVE** | Artifact management. |

### Other Directories

| Directory | Classification | Rationale |
|-----------|---------------|-----------|
| `agent/agents/` | **PRESERVE** | Intent/router/verify agents. |
| `agent/memory/` | **EXTEND** | Session persistence. Phase 4 adds structured experience storage. |
| `agent/profiles/` | **PRESERVE** | YAML model profiles. |
| `agent/schemas/` | **PRESERVE** | Schema system. |
| `agent/templates/` | **PRESERVE** | Starter workflow JSONs. |
| `agent/testing/` | **PRESERVE** | Test utilities. |
| `tests/` | **SACRED** | 2140 passing tests. Never break. Only add. |

---

## PILOT Module Deep Dive — Destructive Mutation Inventory

### Module: `agent/tools/workflow_patch.py`

**State Management:**
- Uses module-level `_state = get_session("default")` — a `WorkflowSession` instance
- `_state_lock = _state._lock` — `threading.RLock` for thread safety
- All handlers wrapped in `with _state_lock:` via the `handle()` dispatcher
- State keys: `loaded_path`, `base_workflow`, `current_workflow`, `history`, `format`

**Destructive Mutation Tools (6 of 9):**

| Tool | Function | Mutation Type | Undo Support |
|------|----------|--------------|--------------|
| `apply_workflow_patch` | `_handle_apply_patch()` | `jsonpatch.JsonPatch.apply()` on `current_workflow` | Yes — pushes `deepcopy` to `history` before applying |
| `add_node` | `_handle_add_node()` | Direct dict insertion: `_state["current_workflow"][node_id] = {...}` | Yes — pushes `deepcopy` to `history` |
| `connect_nodes` | `_handle_connect_nodes()` | Direct dict mutation: `workflow[to_node]["inputs"][to_input] = [from_node, from_output]` | Yes — pushes `deepcopy` to `history` |
| `set_input` | `_handle_set_input()` | Direct dict mutation: `workflow[node_id]["inputs"][input_name] = value` | Yes — pushes `deepcopy` to `history` |
| `undo_workflow_patch` | `_handle_undo()` | Pops from `history`, replaces `current_workflow` | N/A (is the undo) |
| `reset_workflow` | `_handle_reset()` | Replaces `current_workflow` with `deepcopy(base_workflow)`, clears `history` | No (destructive reset) |

**Non-Destructive Tools (3 of 9):**

| Tool | Function | Type |
|------|----------|------|
| `preview_workflow_patch` | `_handle_preview_patch()` | Read-only preview on `deepcopy` |
| `get_workflow_diff` | `_handle_get_diff()` | Read-only diff computation |
| `save_workflow` | `_handle_save()` | Filesystem write only (no state mutation) |

**Key Signatures (for wrapper contracts):**

```python
# Dispatcher — the entry point for ALL workflow_patch tools
def handle(name: str, tool_input: dict) -> str:

# Internal handlers (private, but show mutation pattern)
def _handle_apply_patch(tool_input: dict) -> str:    # patches: list[{op, path, value}]
def _handle_add_node(tool_input: dict) -> str:        # class_type: str, inputs: dict
def _handle_connect_nodes(tool_input: dict) -> str:    # from_node, from_output, to_node, to_input
def _handle_set_input(tool_input: dict) -> str:        # node_id, input_name, value
def _handle_undo() -> str:
def _handle_reset() -> str:
def _handle_preview_patch(tool_input: dict) -> str:
def _handle_get_diff() -> str:
def _handle_save(tool_input: dict) -> str:

# Public helpers called from other modules
def load_workflow_from_data(data: dict, source: str = "<sidebar>") -> str | None:
def get_current_workflow() -> dict | None:
```

**Test Coverage:** `tests/test_workflow_patch.py` — comprehensive coverage of all 9 tools plus edge cases.

---

## Convention Documentation

### Tool Module Pattern
Every module in `agent/tools/` and `agent/brain/` exports:
```python
TOOLS: list[dict]                              # Anthropic tool schemas
def handle(name: str, tool_input: dict) -> str:  # Execute + return JSON string
```

### Tool Schema Format
```python
{
    "name": "tool_name",
    "description": "Human-readable description.",
    "input_schema": {
        "type": "object",
        "properties": { ... },
        "required": [ ... ],
    },
}
```

### Brain Module Pattern (SDK-based)
Brain modules use a class-based SDK pattern:
```python
class MyAgent(BrainAgent):
    NAME = "my_agent"
    TOOLS = [...]       # Tool schemas
    def handle_tool(self, name, tool_input) -> str: ...
```
Registration is automatic via `BrainAgent` metaclass. `BrainAgent.dispatch(name, tool_input)` routes to the correct agent.

### JSON Serialization
Always `sort_keys=True` (He2025 pattern). Use `_util.to_json()`.

### Path Validation
All filesystem operations go through `_util.validate_path()`. Blocks access outside allowed directories.

### Test File Naming
`tests/test_{module_name}.py` — mirrors the source module. All tests are mocked (no live ComfyUI).

### Import Convention
- Relative imports within `agent/` package
- `from ._util import to_json` for tool modules
- `from ..config import X` for configuration
- `from ..workflow_session import get_session` for session state

---

## Phase 1 Wrapper Strategy

### What Gets Wrapped
The 6 destructive tools in `workflow_patch.py` will be wrapped to route mutations through `CognitiveGraphEngine` instead of raw `jsonpatch` / direct dict writes.

### Wrapper Approach
1. `CognitiveGraphEngine` instance stored on `SessionContext` (new field: `graph_engine`)
2. `workflow_patch.handle()` stays as the single dispatcher — no MCP registration changes
3. Internal handlers are modified to:
   - Call `engine.mutate_workflow()` instead of `jsonpatch.apply()` or direct dict mutation
   - Call `engine.temporal_query()` instead of manual history pop for undo
   - Call `engine.to_api_json()` to get resolved workflow for `get_current_workflow()`
4. `get_current_workflow()` returns `engine.to_api_json()` (resolved graph)
5. `base_workflow` is still stored for diff computation
6. History is managed by the engine's delta stack, not a separate list

### What Does NOT Change
- Tool names, schemas, descriptions
- MCP registration in `mcp_server.py`
- Return value JSON format
- `handle()` function signature
- Thread safety pattern (`with _state_lock:`)
- `load_workflow_from_data()` public interface

### New Directory
```
src/
  cognitive/
    __init__.py
    core/
      __init__.py
      models.py     # ComfyNode, WorkflowGraph
      delta.py       # DeltaLayer with SHA-256 integrity
      graph.py       # CognitiveGraphEngine with LIVRPS resolver
```

This is a new `src/` directory — does not conflict with existing `agent/` package.

---

## Verification

```
No code changes made. Only this file (MIGRATION_MAP.md) is new.
git status should show:
  new file: MIGRATION_MAP.md
```

---

## GATE: Awaiting human approval before Phase 1.
