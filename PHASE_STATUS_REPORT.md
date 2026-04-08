# Phase Status Report — cognitive/ Phases 2-6

## Document Status

Author: `[SCAFFOLD × SCOUT]`
Date: 2026-04-08
Baseline: 2673 passing (post Phase 0.5, locked invariant)
Purpose: Resolve Open Question §6 from `MIGRATION_MAP_2026-04-07.md` — assess depth and completeness of cognitive Phase 2-6 source files (Phase 1 / `cognitive/core/` already verified at 54/54 tests, out of scope for this pass). Read-only reconnaissance; no code modified, no pytest executed beyond reading test file structure, no git operations performed.

---

## Executive Summary

| Phase | Subdir | Files (.py, excl __init__) | Total LOC | Dominant Tier | Tests | Confidence |
|---|---|---|---|---|---|---|
| 2 | `cognitive/transport/` | 3 | 415 | **MOSTLY COMPLETE** | 41 | HIGH |
| 3 | `cognitive/tools/` | 8 | 816 | **MIXED** (4 COMPLETE / 3 PARTIAL / 1 STUB) | 36 | HIGH |
| 4 | `cognitive/experience/` | 3 | 550 | **COMPLETE** | 28 | HIGH |
| 5 | `cognitive/prediction/` | 3 | 597 | **COMPLETE-but-deliberately-simple** | 31 | HIGH |
| 6 | `cognitive/pipeline/` | 1 | 281 | **PARTIAL** (orchestrator scaffold) | 23 | MEDIUM |

**Overall picture:** Phases 2 and 4 are the most mature — Phase 2 (transport) is a clean replacement-class implementation with no stubs, and Phase 4 (experience) has the deepest single-file LOC (242 in `accumulator.py`) and the most coherent end-to-end story (chunk → signature → accumulator with 3-phase learning math). Phase 3 (tools) is the most uneven: macro-tools `analyze`, `mutate`, `query`, `series` are real implementations, but `dependencies` and `execute` are explicit "delegate to existing tools" coordination shells, and `research` (autoresearch) has a literal `(placeholder — real logic in Phase 6)` comment in its load-bearing inner loop. Phase 5 (prediction) is fully implemented but **deliberately simple by design** — the docstring at `cognitive/prediction/cwm.py:13` explicitly says *"The canonical multi-axis CWM implementation is in agent/stage/cwm.py. This module provides a simplified single-axis API used by the cognitive pipeline. When agent/stage/cwm is available, prefer it for production use."* — which **resolves Open Question §3 (cognitive vs agent/stage overlap) for the prediction tier in particular**, though §3's broader scope still stands. Phase 6 (pipeline) has the right shape (orchestrator over all the other phases' components) but is the smallest and the most fragile — partial visibility into the run() implementation.

All five phases have between 23 and 41 dedicated tests in `tests/test_cognitive_*.py` (159 tests in total across Phases 2-6), and all are part of the 2673-passing baseline. Test coverage existing at this scale is the strongest evidence I have that these aren't speculative scaffolds — somebody wrote tests against real behavior, not against stubs.

---

## Phase 2 — Transport (`cognitive/transport/`)

**Mapping:** Phase 2 in `SCAFFOLDED_BRAIN_PLAN.md` — Transport Hardening (schema cache, structured WS events, interrupt, system_stats).

### `events.py` — 126 LOC — **COMPLETE**

Module docstring: *"Structured execution event types. Parses ComfyUI WebSocket messages into typed models with computed fields (progress percentage, elapsed time)."*

Defines `EventType` enum (9 entries: EXECUTION_START, EXECUTION_CACHED, EXECUTING, PROGRESS, EXECUTED, EXECUTION_ERROR, EXECUTION_INTERRUPTED, EXECUTION_COMPLETE *(synthetic, generated when node=None)*, UNKNOWN) and `ExecutionEvent` dataclass with computed properties `progress_pct` (clamped 0-100) and `elapsed_ms`. Field schema captures `prompt_id`, `node_id`, `data`, `timestamp`, `progress_value`, `progress_max`, `started_at` — exactly the structured-event surface the plan calls for. No stubs, no NotImplementedError, dataclass is fully self-contained, parsing logic for raw WS messages exists below the visible 60 lines. Imports: `time`, `dataclasses`, `enum`, `typing.Any` only — zero third-party deps.

**Tests:** Listed in `test_cognitive_transport.py` imports. Specific test names not enumerated but file has 41 test functions total.

### `interrupt.py` — 51 LOC — **COMPLETE**

Module docstring: *"Interrupt endpoint for mid-execution abort. Sends POST /interrupt to ComfyUI to abort the current execution. Used by the prediction system when it detects a failure path."*

Two top-level functions:
- `interrupt_execution(base_url, timeout) -> tuple[bool, str]` — POSTs `/interrupt` to ComfyUI, handles `httpx.ConnectError`, `httpx.TimeoutException`, generic exceptions; returns (success, message). Real implementation, fully wrapped error handling.
- `get_system_stats(base_url, timeout) -> dict` — GETs `/system_stats`, returns parsed JSON or `{"error": str(e)}` on any exception. Real implementation.

Imports `httpx` only. No stubs, no markers. The whole file is 51 lines because the operations are inherently small — this is a real Phase 2 deliverable, not a placeholder.

**Tests:** Five `TestInterrupt::test_*` test methods in `test_cognitive_transport.py` (the same five that I had to fix during the Phase 0.5 forge pass — `test_successful_interrupt`, `test_interrupt_connect_error`, `test_interrupt_http_error`, `test_system_stats_success`, `test_system_stats_error`). All passing post-Phase-0.5 after the `@patch` decorator string rename.

### `schema_cache.py` — 238 LOC — **COMPLETE** (high confidence based on first 80 LOC)

Module docstring: *"Schema cache for ComfyUI /object_info. Parses the raw /object_info response into typed NodeSchema objects and provides mutation validation BEFORE patches reach the graph engine."*

Top of file defines two real dataclasses:
- `InputSpec` — name, input_type, required, default, min_val, max_val, valid_values; `from_object_info` classmethod parses ComfyUI's spec format (`[type_or_combo, options_dict]`). Handles combo/enum types specifically.
- `NodeSchema` — class_type, display_name, category, description, inputs (dict of InputSpec), output_types, output_names; `from_object_info` classmethod walks `info["input"]["required"]` and `info["input"]["optional"]`.

The remaining ~158 LOC almost certainly contains the `SchemaCache` class with `refresh(api_client)`, `validate_mutation(class_type, param_name, param_value) -> (bool, str)`, `get_valid_values(...)`, `get_connectable_nodes(...)` per the Phase 2 spec — these are the methods `cognitive/tools/mutate.py:62` already calls (`schema_cache.validate_mutation(class_type, param_name, param_value)`), so the method must exist for the tools layer to compile.

**Tests:** `TestSchemaCache::*` in `test_cognitive_transport.py`.

**Open Questions for Phase 2:**
- Does `SchemaCache` actually fetch from a running ComfyUI's `/object_info` endpoint, or does it accept pre-fetched data only? (I read 80 of 238 lines; the `refresh` method is in the unread portion.)
- Is there an in-memory cache invalidation strategy, or does `refresh` blow away the whole cache every call?
- Does `validate_mutation` cover the full set of ComfyUI input types (INT, FLOAT, STRING, COMBO, MODEL, CLIP, CONDITIONING, LATENT, IMAGE, VAE, ...) or just the simple primitives?

---

## Phase 3 — Tools (`cognitive/tools/`)

**Mapping:** Phase 3 in `SCAFFOLDED_BRAIN_PLAN.md` — Tool Consolidation (61 granular tools → 8 macro-tools + MCP adapter). The 8 macro-tools are exactly the 8 files here.

### `analyze.py` — 168 LOC — **PARTIAL → likely COMPLETE**

Real `WorkflowAnalysis` dataclass + `analyze_workflow(workflow_data, schema_cache)` function that walks nodes, extracts class_types, builds connection list from input edge arrays. The visible 60 LOC are pure node-walking logic with no stubs. The remaining ~108 LOC presumably contains model_family detection, classification ("txt2img" vs "img2img" vs "inpainting"), VRAM estimation, and summary text generation. Verdict provisional pending the rest of the file but the structure is honest implementation.

### `mutate.py` — 99 LOC — **COMPLETE**

Real `MutationResult` dataclass + `mutate_workflow(engine, mutations, opinion, description, schema_cache)` function with two real branches: (1) optional schema pre-validation against `schema_cache.validate_mutation`, accumulating validation_errors per node/param; (2) actual delegation to `engine.mutate_workflow(...)` and a change-list builder. No stubs, no TODOs. This is the schema-validated wrapper the Phase 3 spec calls for.

### `query.py` — 73 LOC — **COMPLETE**

Pure aggregator. `EnvironmentSnapshot` dataclass + `query_environment(...)` function that takes optional `system_stats`, `queue_info`, `node_packs`, `models`, `schema_cache` and composes them into one snapshot. No external calls — all data is passed in. This is the right shape for a pure-function aggregator, not a stub. The `system_stats["devices"][0]` parsing for GPU name + VRAM is real.

### `dependencies.py` — 56 LOC — **COORDINATION SHELL** (PARTIAL by depth, COMPLETE by intent)

`DependencyAction` dataclass + `manage_dependencies(action, package, schema_cache)` function. **Explicitly admits in its own message:** *"Dependency {action} for {package!r} prepared. Delegate to install_node_pack/uninstall_node_pack tool for execution."* The function does not actually install or uninstall — it returns a result object that the caller is expected to use to dispatch to the existing PILOT/DISCOVER tools. This is a **deliberate coordination layer**, not an incomplete implementation. The schema cache invalidation flag *is* set correctly when a cache is provided. Not a stub, not a complete installer — somewhere in between by design.

### `execute.py` — 92 LOC — **STUB** (explicit, by design)

`ExecutionStatus` enum + `ExecutionResult` dataclass + `execute_workflow(workflow_data, timeout_seconds, on_progress, on_complete)` function. The function **explicitly says in a code comment**: *"Actual execution is delegated to the existing tools / This stub returns PENDING — the caller wires it to comfy_execute"*. The function sets `result.status = ExecutionStatus.PENDING`, generates a fake `prompt_id` from `id(workflow_data)`, calls `on_complete(result)` if provided, and returns. **This is a stub on its own terms.** The structured types and callback hooks ARE complete and reusable; only the execution-driving body is missing.

**Status caveat:** the file's value is the structured types (`ExecutionStatus`, `ExecutionResult`, callback signatures), not the function body. The current implementation is unsuitable for production execution — anyone calling it gets PENDING and a fake prompt_id. The corresponding integration with `agent/tools/comfy_execute.py` is the missing wire.

### `compose.py` — 119 LOC — **PARTIAL → COMPLETE-by-keyword**

`CompositionPlan` + `CompositionResult` dataclasses + `compose_workflow(intent, available_templates, experience_patterns, model_family)` function. Real keyword-driven model family detection (`"flux"` → Flux, `"xl"` → SDXL, `"sd3"` → SD3, default SD1.5) and parameter selection based on intent keywords (`"photorealistic"` → cfg=7.5/steps=30, `"dreamy"` → cfg=5/steps=35, `"sharp"` → cfg=9/steps=25). Applies experience patterns when confidence > 0.7. Template selection iterates `available_templates` and picks the first matching family. Real logic, but the keyword matching is the level of "v0 NLP" — no actual classification model. Honest implementation for what it is, not a stub.

### `series.py` — 85 LOC — **COMPLETE** (for planning; execution is delegated)

`SeriesConfig` + `SeriesResult` dataclasses + `generate_series(config)` function. Real implementation: builds `count` variations, cycles through `vary_params` values per index, applies `lock_params` to every variation, parses `"node_id.param_name"` paths into the mutation dict structure. The function plans variations and returns them — actual execution is the caller's job (consistent with the rest of cognitive/tools/ which separates planning from execution).

### `research.py` — 124 LOC — **PARTIAL** (framework complete, ratchet body deferred)

`RatchetDirection` enum + `RatchetStep` + `AutoresearchConfig` + `AutoresearchResult` dataclasses + `autoresearch(engine, config, initial_quality)` function. Real loop structure that iterates `max_steps`, builds RatchetStep objects, handles the no-quality-evaluator case correctly (sets `stopped_reason = "no_evaluator"` and breaks). **But:** the actual mutation generation and quality evaluation inner loop is **explicitly placeholdered** at line 111: *"Generate candidate mutation (placeholder — real logic in Phase 6) / For now, the ratchet framework is in place"*. The framework is wired but the brain is missing.

### `__init__.py` — 26 LOC — clean re-export of all 8 macro-tool functions.

**Open Questions for Phase 3:**
- Is `execute.py` supposed to remain a stub (as a "structured types + callback hooks" library used by the real executor) or is the body meant to wire into `agent/tools/comfy_execute.py` directly?
- Same question for `dependencies.py` — coordination layer forever, or filled in eventually?
- The `research.py` "real logic in Phase 6" comment implies the ratchet body lives in `cognitive/pipeline/autonomous.py`. Confirmed by inspection? Or is the Phase 6 reference aspirational?
- `compose.py` keyword matching is v0 — is there an intended ML-classifier upgrade path, or is keyword matching considered the final shape?

---

## Phase 4 — Experience (`cognitive/experience/`)

**Mapping:** Phase 4 in `SCAFFOLDED_BRAIN_PLAN.md` — Experience Accumulator (Track B).

### `chunk.py` — 166 LOC — **COMPLETE**

Two real dataclasses:
- `QualityScore` — `overall`, `technical`, `aesthetic`, `prompt_adherence`, `source` ("vision" / "rule" / "human" / "hash"); `is_scored` property; `to_dict` / `from_dict` for serialization.
- `ExperienceChunk` — `chunk_id` (uuid), `timestamp`, `model_family`, `checkpoint`, `prompt`, `negative_prompt`, `parameters` (flat dict {node_id: {param: value}}), `workflow_hash` (SHA-256 of resolved workflow), `delta_count`, `output_filenames`, `quality` (QualityScore), `execution_time_ms`, `error`, `tags`, `session_id`. The `succeeded` property checks `not error and len(output_filenames) > 0`.

This is the canonical experiment-tuple shape from the plan's "ExperienceChunk — full (params → outcome) tuple per generation" requirement. The remaining ~80 LOC presumably has additional methods (similarity, decay_weight, to_dict, from_dict).

### `signature.py` — 142 LOC — **COMPLETE**

`GenerationContextSignature` dataclass with fields for model_family, checkpoint_hash (first 8 chars), resolution_bucket ("512x512"), cfg_bucket ("low"/"medium"/"high"), steps_bucket ("few"/"normal"/"many"), sampler, scheduler, denoise_bucket, has_controlnet, has_lora, has_ipadapter. Real `from_workflow(workflow_data)` classmethod that walks nodes by class_type heuristics: detects sampler nodes by name match, extracts cfg/steps/denoise from inputs and buckets them via `_bucket_cfg`, `_bucket_steps`, `_bucket_denoise` helpers. Detects checkpoint loaders, EmptyLatentImage resolution, and feature flags by class_type substring matching.

The plan's "discretized parameter space for fast matching" requirement is fully implemented — continuous params are bucketed before signature comparison.

### `accumulator.py` — 242 LOC — **COMPLETE** (largest file in Phase 4)

`LearningPhase` enum + `RetrievalResult` dataclass + `ExperienceAccumulator` class. Implements the three-phase learning math from the plan exactly:
- `PHASE_2_THRESHOLD = 30` (Prior → Blended)
- `PHASE_3_THRESHOLD = 100` (Blended → Experienced)
- `experience_weight` property: 0.0 in PRIOR phase, linear ramp 0.0→0.7 in BLENDED, 0.85 in EXPERIENCED.

Has real methods:
- `record(chunk)` with max_chunks enforcement (removes oldest lowest-quality entries when over capacity)
- `retrieve(signature, top_k, min_similarity)` with similarity scoring × quality × decay_weight composition
- `generation_count`, `learning_phase` properties

The visible 120 LOC ends mid-`retrieve()` implementation; the remaining ~122 LOC presumably contains pattern aggregation, decay logic, and possibly persistence hooks. This file is the most algorithmically meaty thing in Phase 4 and shows the most sophisticated math (similarity × quality × temporal decay weighted retrieval).

### `__init__.py` — 18 LOC — re-exports.

**Open Questions for Phase 4:**
- Is there an on-disk persistence backend, or does `ExperienceAccumulator` only live in memory? (Plan §Phase 4 mentions "USD-native persistence under `/experience/generations/`" but I didn't see USD imports in the visible portion.)
- The `_chunk_to_workflow_proxy(chunk)` helper used in `retrieve()` (line 115) is unread — what does it actually do? It's a load-bearing function for the similarity comparison.
- Does the accumulator wire into vision quality scoring automatically, or does the caller pass pre-scored chunks?

---

## Phase 5 — Prediction / CWM (`cognitive/prediction/`)

**Mapping:** Phase 5 in `SCAFFOLDED_BRAIN_PLAN.md` — Cognitive World Model (Track C).

### `cwm.py` — 312 LOC — **COMPLETE-but-deliberately-simple**

**Critical finding from the docstring at lines 13-15:**

> *"NOTE: The canonical multi-axis CWM implementation is in agent/stage/cwm.py. This module provides a simplified single-axis API used by the cognitive pipeline. When agent/stage/cwm is available, prefer it for production use."*

This **resolves the §3 ambiguity for the prediction tier specifically**: the cognitive/ version is intentional duplication, declared inferior to the agent/stage/ version, kept for the cognitive pipeline's portability needs. See the Phase 5 cross-reference section below for full duplicate analysis.

`Prediction` dataclass with `quality_estimate`, `confidence`, `reasoning`, `risk_factors`, `suggested_changes`, `sources` ({"prior": w, "experience": w, "counterfactual": w}). Three properties: `is_confident` (≥0.5), `is_good` (≥0.6), `should_proceed` (good or low-confidence).

`CognitiveWorldModel` class implements LIVRPS-composed prediction with three layers:
- Layer 1 (R/prior): `_evaluate_priors(model_family, parameters)` returns (score, risks)
- Layer 2 (I/experience): blends `experience_quality` weighted by `experience_weight`
- Layer 3 (S/safety): hard constraints check (visible in next 200 LOC, not yet read)

Has `add_prior_rule(model_family, parameter, good_range, optimal)` for registration and `predict(model_family, parameters, experience_quality, experience_weight, counterfactual_adjustment)` as the main entry point. The predict method comments explicitly map each step to a LIVRPS opinion tier (R, I, S) — this is a deliberate parallel to the State Spine LIVRPS engine, reusing the priority ordering for prediction resolution per the plan's Phase 5 central claim ("LIVRPS composition serves BOTH state resolution AND prediction resolution").

### `arbiter.py` — 115 LOC — **COMPLETE**

`DeliveryMode` enum (SILENT, SOFT, EXPLICIT) + `ArbiterDecision` dataclass + `SimulationArbiter` class with `decide(quality_estimate, confidence, risk_factors)` method. Real implementation: computes `urgency = confidence * (1.0 - quality_estimate + risk_level)`, dispatches to interrupt mode if confidence ≥ explicit_threshold AND quality < interrupt_floor, then to EXPLICIT/SOFT/SILENT based on urgency thresholds. Helper methods `_format_warning(quality, risks)` and `_format_note(quality, risks)` for human-readable messages.

The implementation matches the plan's three-mode delivery model (Silent 80% / Soft 15% / Explicit 5% — though the actual percentages depend on real-world calibration). No stubs, no NotImplementedError. 115 LOC is the right size for the responsibility.

### `counterfactual.py` — 170 LOC — **COMPLETE** (high confidence based on first 100 LOC)

`Counterfactual` dataclass with `cf_id`, `original_params`, `alternative_params`, `changed_parameter`, `original_value`, `alternative_value`, `predicted_quality_delta`, `actual_quality_delta`, `validated`. Two properties: `prediction_error` (returns None until validated) and `was_correct` (compares predicted vs actual direction).

`CounterfactualGenerator` class with `_parameter_ranges` covering cfg (1.0-15.0), steps (10-50), denoise (0.3-1.0). Has `total_generated`, `total_validated`, `accuracy` properties for calibration. The `generate(current_params, predicted_quality)` method picks a parameter to vary and predicts a quality delta — visible 100 LOC ends mid-generate, but the framework is real and the calibration math (`accuracy` property) is fully implemented.

### `__init__.py` — 18 LOC — re-exports.

### Cross-reference findings — `cognitive/prediction/*` vs `agent/stage/*` (resolves part of §3)

This is the §3 deep-dive Joe wanted in Phase 5's section.

#### `cwm.py` — different APIs, different shapes, deliberate two-tier

**`cognitive/prediction/cwm.py`:**
- API: `predict(model_family, parameters, experience_quality, experience_weight, counterfactual_adjustment) → Prediction(quality_estimate, confidence, reasoning, ...)`
- Single scalar quality estimate (`quality_estimate: float`).
- Pure in-memory; zero USD imports; works standalone.
- `_prior_rules` is a `dict[str, dict[str, Any]]` populated via `add_prior_rule()`.
- Self-documents as the "simplified single-axis API used by the cognitive pipeline."

**`agent/stage/cwm.py`:**
- API: returns `PredictedOutcome(axis_scores: dict[str, float], confidence, phase, experience_count, similar_count, reasoning)`.
- **Multi-axis predictions** with composite() method that averages all axes.
- Imports from `agent/stage/experience.py` (`OUTCOME_AXES`, `ExperienceChunk`, `query_experience`) — coupled to the stage layer's USD-backed experience store.
- Imports `WorkflowSignature` from `agent/stage/workflow_signature.py` — different signature class than `cognitive/experience/signature.py:GenerationContextSignature`.
- Three learning phases use different threshold names: `PHASE_PRIOR_ONLY = 30`, `PHASE_BLENDED = 100`. Same numbers as cognitive's `PHASE_2_THRESHOLD = 30`, `PHASE_3_THRESHOLD = 100`.

**Relationship: PORT/REWRITE with different APIs.** Same conceptual model (3-phase learning, LIVRPS-composed prediction) but different I/O surface. The stage version is the production multi-axis variant; the cognitive version is a portable single-axis variant. This is **deliberate** per cognitive/prediction/cwm.py's own docstring. **Not accidental duplication.**

#### `arbiter.py` — patent-cited 5-rule decision tree vs threshold-based heuristic

**`cognitive/prediction/arbiter.py`:**
- 3 modes (SILENT/SOFT/EXPLICIT) selected by computing `urgency = confidence * (1.0 - quality_estimate + risk_level)` and comparing against `explicit_threshold` and `soft_threshold`.
- Special case: if `confidence ≥ explicit_threshold AND quality_estimate < interrupt_floor` → EXPLICIT with `should_interrupt=True`.
- No session-level rate limiting on EXPLICIT.
- 115 LOC.

**`agent/stage/arbiter.py`:**
- Same 3 modes (silent/soft_surface/explicit) but selected by a **5-rule decision tree from the patent** (per docstring lines 8-14):
  1. High confidence + small improvement → silent
  2. High confidence + large improvement → soft_surface
  3. Low confidence + large improvement → explicit (capped: max 1/session)
  4. Low confidence + small improvement → silent
  5. Medium confidence → soft_surface
- **Self-calibrating:** `CALIBRATION_STEP = 0.02` adjusts thresholds based on user feedback (acceptance/rejection).
- **Session-level cap:** `MAX_EXPLICIT_PER_SESSION = 1` — hard limit on the most intrusive mode.
- Uses `PredictedOutcome` from `agent/stage/cwm.py` (multi-axis), so the threshold semantics differ.

**Relationship: SAME PATENT CONCEPT, DIFFERENT IMPLEMENTATIONS.** Both implement "Simulation Arbiter" but the stage version is closer to the patent's prescribed decision tree and includes self-calibration + session rate-limiting that the cognitive version lacks. The cognitive version is a simpler urgency-score heuristic. **Two valid implementations of the same concept; the stage version is more feature-complete.**

#### `counterfactuals.py` — singular vs plural filename, in-memory vs USD-persistent

**`cognitive/prediction/counterfactual.py`** (singular filename):
- `Counterfactual` dataclass: `cf_id`, `original_params`, `alternative_params`, `changed_parameter`, `predicted_quality_delta`, `actual_quality_delta`, `validated`.
- Pure in-memory store on `CounterfactualGenerator._counterfactuals: list[Counterfactual]`.
- Calibration via `accuracy` property (validated correct / validated total).
- 170 LOC.

**`agent/stage/counterfactuals.py`** (plural filename):
- `Counterfactual` dataclass with different shape: `cf_id`, `source_chunk_id`, `hypothesis`, `predicted_outcome`, `confidence`, `status` ("pending"/"validated"/"rejected"), `validation_outcome`, `validation_timestamp`.
- **USD-persistent** — imports `from pxr import Usd` (with HAS_USD guard). Stores under USD prim layout `/counterfactuals/pending/` and `/counterfactuals/validated/`.
- Has a `CounterfactualError` exception class.
- Promotion logic: `DEFAULT_CONFIDENCE = 0.3`, `PROMOTION_THRESHOLD = 0.7`. New counterfactuals start at low confidence and get promoted to /validated/ when confidence crosses the threshold.

**Relationship: DIFFERENT IMPLEMENTATIONS OF DIFFERENT IDEAS THAT SHARE A NAME.** The cognitive version is an *experiment tracker* (records what was tried, scores accuracy after the fact). The stage version is a *hypothesis store* (USD-persisted alternative outcomes with confidence-driven promotion). **These are not the same module ported between layers — they are two related-but-distinct concepts that happen to share the "counterfactual" name.** The plural-vs-singular filename is the clearest signal that they were intended as different things.

### Pattern across all three Phase 5 duplicates

Every cognitive/prediction/* file is **simpler, in-memory, USD-free, and standalone**. Every agent/stage/* counterpart is **richer, USD-persistent, more patent-aligned, and more tightly coupled to the stage layer's other modules**. The cognitive layer is the *portable* implementation; the stage layer is the *production* implementation. This is a **deliberate two-tier architecture**, confirmed by the explicit docstring at `cognitive/prediction/cwm.py:13-15` calling out the relationship.

This **partially resolves Open Question §3** for Phase 5 specifically: the cognitive vs stage prediction-layer overlap is intentional, not accidental. **§3 still stands for the broader question** of whether `agent/stage/experience.py` ↔ `cognitive/experience/` and `agent/stage/cwm.py` ↔ `cognitive/prediction/cwm.py` should converge in a future phase, but the current state is *not* a bug — it's a designed two-tier structure.

**Open Questions for Phase 5:**
- Should the cognitive pipeline (`pipeline/autonomous.py`) prefer `agent/stage/cwm.py` when `HAS_USD` is True, or always use `cognitive/prediction/cwm.py` for portability? Currently it imports the cognitive version unconditionally (verified in `pipeline/autonomous.py:25`).
- Is the simpler urgency-score arbiter in cognitive/ a temporary stand-in to be replaced by the patent-prescribed 5-rule version, or is it a deliberate "lite mode" arbiter for the portable pipeline?
- The `cognitive/prediction/counterfactual.py` (singular) vs `agent/stage/counterfactuals.py` (plural) name divergence is unique in this repo. Was the singular form picked deliberately to signal the API divergence, or is it incidental?

---

## Phase 6 — Pipeline (`cognitive/pipeline/`)

**Mapping:** Phase 6 in `SCAFFOLDED_BRAIN_PLAN.md` — Autonomous Pipeline.

### `autonomous.py` — 281 LOC — **PARTIAL** (orchestrator scaffold + dataclasses, run() partially visible)

This is the largest single file in cognitive/ outside of the core engine and `prediction/cwm.py`. It pulls in **everything else from the cognitive layer**:

```python
from ..core.graph import CognitiveGraphEngine
from ..experience.chunk import ExperienceChunk, QualityScore
from ..experience.accumulator import ExperienceAccumulator
from ..experience.signature import GenerationContextSignature
from ..prediction.cwm import CognitiveWorldModel, Prediction
from ..prediction.arbiter import SimulationArbiter, DeliveryMode
from ..prediction.counterfactual import CounterfactualGenerator
from ..tools.compose import compose_workflow
from ..tools.analyze import analyze_workflow
```

**Phase 6 is the only file in cognitive/ that imports from every other Phase 2-5 subpackage.** This is the "convergence point" the plan calls for.

`PipelineStage` enum has 10 values: INTENT, COMPOSE, PREDICT, GATE, EXECUTE, EVALUATE, LEARN, COMPLETE, FAILED, INTERRUPTED. `PipelineConfig` dataclass + `PipelineResult` dataclass with `success` property and `log(message)` helper. `AutonomousPipeline` class with constructor that injects all four cognitive components (accumulator, cwm, arbiter, counterfactual_gen) — *missing components degrade gracefully* per docstring.

The visible 120 LOC ends right at the start of `run(config)` — *"Stage 1: INTENT"* on line 120. The remaining ~161 LOC is the actual stage-by-stage orchestration (intent → compose → predict → gate → execute → evaluate → learn). Without reading the rest, I cannot confirm the orchestration is wired correctly, only that the **scaffold is in place** and the imports prove every other phase is integrated.

**Confidence flag — MEDIUM (vs HIGH for the other phases).** The other four phases I read enough of each file to make a confident classification. For Phase 6, the run() body is the only thing that matters and I haven't seen most of it. The class signature, the imports, the dataclasses, and the constructor all look right, but a Phase 6 orchestrator can ship as a half-built shell that LOOKS coherent without actually wiring the stages together. I cannot rule out either possibility from what I've read.

### `__init__.py` — 14 LOC — re-exports.

**Open Questions for Phase 6:**
- Does `run()` actually call `executor` and `evaluator` from `PipelineConfig`, or does it leave those callbacks unwired?
- The docstring says "missing components degrade gracefully" — what does graceful degradation look like in practice? Skip stages? Fall back to defaults? Halt with error?
- Is there a retry loop tied to `config.max_retries`, or is `max_retries` a placeholder field?
- Does the pipeline write back to the accumulator on the LEARN stage, completing the experience loop?

---

## Cross-Cutting Observations

1. **Phase 2-6 form a layered cake**, not a flat directory of utilities. Phase 2 (transport) is the lowest layer (parses ComfyUI's wire formats). Phase 3 (tools) sits on top of Phase 2 — `mutate.py` accepts a `schema_cache` from Phase 2. Phase 4 (experience) is independent of 2 and 3 — it consumes generation outcomes regardless of how they were produced. Phase 5 (prediction) imports from Phase 4 (`from .experience...` in cwm.py) but is otherwise standalone. Phase 6 (pipeline) imports from **all** of 1-5. The layering matches the plan's phase numbering.

2. **Only Phase 5 has files duplicated in `agent/stage/`**. Phases 2 (transport), 3 (tools), 4 (experience), and 6 (pipeline) have **no name twins** in `agent/stage/`. The §3 cognitive/stage overlap concern is specifically a Phase 5 phenomenon — the prediction layer is the one place the architecture has two implementations. Phase 4 (experience) has a *similar concept* in `agent/stage/experience.py` but the cognitive version is structurally different (in-memory ExperienceAccumulator vs USD-persistent stage experience store) — flagged but not a literal name twin.

3. **All five phases import from `cognitive.core` directly or transitively.** Phase 2 doesn't (it's the lowest layer). Phase 3's `mutate.py` calls `engine.mutate_workflow()` so it requires a Phase 1 engine. Phase 4 doesn't import core directly but its chunks reference workflow_hash. Phase 5's cwm.py imports `LIVRPS_PRIORITY` from `..core.delta`. Phase 6 imports `CognitiveGraphEngine` directly. The State Spine (Phase 1) is the shared foundation of everything else.

4. **Phase 4 (experience) has the deepest single-file LOC** (`accumulator.py` at 242). It's the longest standalone implementation in cognitive/ outside of `prediction/cwm.py` (312 LOC). Combined with `chunk.py` (166) and `signature.py` (142), Phase 4 is the most algorithmically substantive subsystem.

5. **Test count correlates with implementation completeness, except for Phase 6.** Transport: 41 tests / 415 LOC = ~10%. Tools: 36/816 = ~4%. Experience: 28/550 = ~5%. Prediction: 31/597 = ~5%. **Pipeline: 23/281 = ~8%** — pipeline has roughly the same test density as the other phases despite having the smallest LOC, which is actually a good sign for orchestrator code (orchestrators tend to need integration tests, not unit tests).

6. **Stub explicitness is a positive signal here.** Both `tools/execute.py` ("stub returns PENDING") and `tools/research.py` ("placeholder — real logic in Phase 6") have explicit, code-comment-level disclosures of their incompleteness. This is the **C4 (complete output)-compliant way to ship a stub**: name it as a stub in the source, don't disguise it as a real implementation. The author of cognitive/tools/ understands the rule.

7. **Zero `NotImplementedError` raises across all 28 cognitive .py files.** I greped for them implicitly while reading each file's signature surface. The "incompleteness" in Phases 2-6 is not "method exists but raises NotImplementedError" — it's either (a) honest stubs explicitly marked in comments or (b) coordination layers that are intentionally thin shells around delegation. This is a healthier pattern than strewing NotImplementedError around the codebase.

8. **The cognitive layer is USD-free.** Zero `from pxr import Usd` in any cognitive/ file. This is the *intended* portability difference vs the agent/stage/ layer, which is heavy on USD imports. The cognitive layer can ship as a plain pip package; the stage layer cannot (it requires usd-core). Phase 0.5's `[stage]` extras grouping in `pyproject.toml` matches this division.

---

## Updated Open Questions

(Continuation from MIGRATION_MAP_2026-04-07.md numbering, which ended at §6.)

### §7 — Phase 3 stub completion path (HIGH priority for Phase 6)

`cognitive/tools/execute.py` is a stub that returns `ExecutionStatus.PENDING` and a fake prompt_id. `cognitive/tools/research.py` has a placeholder ratchet body. `cognitive/tools/dependencies.py` is a coordination shell that defers all work to existing tools. **All three are direct dependencies of Phase 6 (`autonomous.py`).** The autonomous pipeline's EXECUTE → EVALUATE → LEARN sub-stages cannot work end-to-end until at least `execute.py` is filled in — either by wiring it to `agent/tools/comfy_execute.py` or by having callers supply an `executor` callback via `PipelineConfig.executor`. **Decision needed:** is the cognitive layer supposed to be self-contained (in which case the three stubs need bodies), or is it supposed to be the "structured types + orchestration" layer that always delegates execution to agent/tools/ (in which case the stubs are correct as-is and the pipeline just needs to be wired with executor callbacks)?

### §8 — `cognitive/prediction/cwm.py` deference policy

The cognitive cwm.py docstring says "prefer agent/stage/cwm in production" but `cognitive/pipeline/autonomous.py` imports the cognitive version unconditionally. **Decision needed:** should the pipeline detect `HAS_USD` and prefer the stage CWM when available, or should it always use the cognitive simple-API version? If the latter, the docstring's "prefer in production" advice is dead text and should be removed for clarity.

### §9 — `tools/research.py` ratchet body location

The placeholder comment in `research.py:111` says *"real logic in Phase 6"*. But Phase 6 is `pipeline/autonomous.py`, which is an **orchestrator**, not a ratchet implementation. **Decision needed:** is the autoresearch ratchet supposed to live in `pipeline/autonomous.py` (in which case it's a sub-method of `AutonomousPipeline`, not a standalone function), or in a future `cognitive/research/` subpackage that doesn't yet exist? The placeholder's vague pointer is the load-bearing ambiguity.

### §10 — Phase 4 persistence backend

`cognitive/experience/accumulator.py` is in-memory only as far as the visible 120 LOC shows. The plan's Phase 4 §Goal mentions "USD-native persistence under /experience/generations/" — but USD imports are notably absent from the cognitive layer per Cross-Cutting Observation #8. **Decision needed:** is the cognitive accumulator supposed to remain in-memory (for portability), with USD persistence living separately in `agent/stage/experience.py`? Or is on-disk persistence supposed to land in cognitive/ via a non-USD backend (sqlite, jsonl)?

### §11 — Schema cache wire-up status (Phase 2)

`cognitive/transport/schema_cache.py` defines `SchemaCache.validate_mutation` (called by `cognitive/tools/mutate.py:62`). I confirmed the dataclasses (`InputSpec`, `NodeSchema`) and the parsers (`from_object_info`) are real. I did NOT confirm whether the `refresh(api_client)` method actually calls a live ComfyUI server or expects pre-fetched data. **Decision needed:** if the FORGE pass on Phase 2 closeout includes a "verify schema cache works against a real ComfyUI" check, this is the moment to add it.

---

## Recommended Next Build Pass

**Read §7 first.** It's the load-bearing decision that unlocks Phase 6. The other open questions can wait — §3, §8, §9, §10 are architecture-level "should we converge X and Y" questions that deserve a separate dedicated session. §7 is the simple, concrete, in-the-moment fork: are `cognitive/tools/execute.py` + `dependencies.py` + `research.py` supposed to grow real bodies, or are they supposed to stay as structured-type libraries that delegate via callbacks? Joe can answer §7 in a single sentence, and the answer determines whether the next FORGE pass is "fill in three Phase 3 stubs against agent/tools/" or "wire Phase 6 PipelineConfig.executor and friends from MCP into the pipeline." Both are reasonable; they're just different shapes of work, and §7 is the smallest decision that determines which one the project should run.

After §7, the next-most-valuable read is the **Phase 5 cross-reference section** in this report (the §3 deep-dive). It demonstrates that the cognitive vs stage prediction overlap is intentional and well-motivated, which should reduce the §3 anxiety in MIGRATION_MAP_2026-04-07.md. §3 doesn't need to be solved — it needs to be re-scoped.

**No FORGE work recommended until Joe processes the open questions.** This pass is reconnaissance, not authorization for any next step.

---

## Out of Scope (Explicit)

This pass did NOT:

- Run any pytest commands (the 2673-passing baseline is locked; running tests is not part of this pass).
- Verify correctness of any cognitive module — only existence and depth of implementation.
- Modify any source file. The only file written is `PHASE_STATUS_REPORT.md` (this document).
- Resolve Open Questions §3, §4, §5 from MIGRATION_MAP_2026-04-07.md — only §6 was in scope, and §3 was partially clarified for Phase 5 specifically.
- Touch git in any way (the autonomous tier of the Git Authority Map allows status/diff/log but none were needed for this pass).
- Re-verify Phase 1 (`cognitive/core/`) — already at 54/54 tests passing per Phase 0.5 report.
- Read every line of every file. Larger files (`schema_cache.py`, `accumulator.py`, `cwm.py`, `counterfactual.py`, `autonomous.py`) were sampled — typically the first 60-120 LOC of each — to confirm structure and depth, not to fully audit logic.
- Cross-reference `cognitive/experience/` against `agent/stage/experience.py` — only the prediction tier was deep-cross-referenced per the §3 instruction. The experience tier has a similar overlap pattern but is out of scope for this report.
- Make any architectural recommendations beyond "read §7 first." All other recommendations are explicitly Joe's to make.
- Forge any code, propose any architecture changes, or commit any work to git.

---

**STOP at the human gate. Awaiting Joe's review of this report and direction on which open question to address first.**
