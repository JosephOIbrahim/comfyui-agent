# SYSTEM DESIGN REVIEW — Phase 7B

**Agent:** All domain experts × `[SCOUT]` + `[DESIGN × ARCHITECT]`
**Date:** 2026-03-31
**Baseline:** 2350 passed (2140 original + 210 cognitive), 156 skipped, 27 pre-existing errors

---

## Executive Summary

The Scaffolded Brain architecture is **structurally sound but operationally incomplete**. The core engine (Phase 1) is solid. Everything built on top — transport, tools, experience, prediction, pipeline — exists as well-tested specification ahead of integration. The gap is not design; it's wiring. Components are islands. The bridges between them are missing or stubbed.

**Counts: 7 SOLID, 16 ROUGH, 12 BROKEN.**

---

## 1. Composition Integrity — [GRAPH × SCOUT]

| Finding | Status | Detail |
|---------|--------|--------|
| LIVRPS resolution correctness | **SOLID** | Priority ordering, stable sort, strongest-wins — all verified across 54 adversarial tests. Complex multi-opinion scenarios pass. |
| SHA-256 tamper detection | **SOLID** | Deterministic hash (sort_keys=True), creation vs recomputed comparison, engine-level verify_stack_integrity — all working. |
| Link preservation | **SOLID** | ["node_id", output_index] arrays survive parsing, mutation, multi-delta, round-trip. 4 dedicated test categories. |
| AUTOGROW dotted names | **SOLID** | "values.a" notation handled in set_input, connect_nodes, and engine mutations. Tests pass. |
| Delta stack unbounded growth | **ROUGH** | No MAX_DELTAS limit. 50+ delta sessions accumulate without compaction. _MAX_HISTORY (50) only governs the jsonpatch fallback, not the engine stack. **Suggestion:** Add configurable limit with oldest-same-opinion compaction. |
| Engine/state sync on fallback | **ROUGH** | When _patches_to_mutations returns None (unsupported RFC6902 ops), jsonpatch runs and engine is rebuilt from scratch — **losing the entire delta stack**. Silent data loss. **Suggestion:** Fail the patch instead of silent rebuild; or reject unsupported ops early. |
| Nested JSON path handling | **ROUGH** | Paths like /1/inputs/config/color/r produce "config/color/r" as a flat key instead of nested dict traversal. ComfyUI would reject at execution. **Suggestion:** Reject patches deeper than /node_id/inputs/param_name; route through set_input instead. |
| Undo after fallback | **ROUGH** | History list and engine delta stack can desync after a jsonpatch fallback. Undo may pop from both, but engine was rebuilt — base already includes the fallback patch. **Suggestion:** Same fix as sync issue — don't rebuild, fail loudly. |
| Module-level _engine global | **BROKEN** | _engine is not protected by _state_lock. Race condition: two threads can read/write _engine concurrently. _sync_state_from_engine() reads _engine outside any lock. **Fix:** Protect all _engine access with _state_lock, or move engine into _state dict. |
| Multi-session isolation | **BROKEN** | _engine is a single global. Two MCP sessions loading different workflows overwrite each other's engine. SessionContext.ensure_graph_engine exists but is never called. **Fix:** Move _engine to SessionContext.graph_engine and wire through handler dispatch. |
| Dead code: get_engine(), ensure_graph_engine | **BROKEN** | Both exist, neither is called anywhere. Creates confusion about the intended integration path. **Fix:** Wire or remove. |

---

## 2. Learning Quality — [EXPERIENCE × SCOUT]

| Finding | Status | Detail |
|---------|--------|--------|
| ExperienceChunk structure | **SOLID** | Captures model_family, checkpoint, prompt, parameters, quality (4-axis), execution_time, error, tags. Temporal decay with 7-day half-life is mathematically sound. |
| _extract_patterns voting | **SOLID** | Most-common-value-wins from successful chunks. Simple, effective. |
| Signature discretization: CFG/steps | **ROUGH** | Too coarse. CFG 7 and 8 are the same bucket ("medium") but behave differently on SDXL. Steps 15-30 all "normal" but Flux needs 16-24 (narrow range). **Suggestion:** Model-family-aware bucket boundaries. |
| Signature: resolution bucket | **ROUGH** | Too fine. "1024x1024" vs "1024x1008" are different buckets despite 0.8% difference. **Suggestion:** Tier buckets (small/medium/large). |
| Signature: feature detection | **ROUGH** | Binary only (has_controlnet: true/false). Loses count and weight information. 1 LoRA at 0.1 = 5 LoRAs at 0.9 in the signature. |
| Signature: missing checkpoint_hash | **ROUGH** | checkpoint_hash field exists but is NOT included in the similarity() comparison. Bug — should be added to the fields list. |
| Phase 2→3 discontinuity | **ROUGH** | experience_weight jumps from linear ramp (0→0.7 at gen 100) to constant 0.85 at gen 101. +0.15 discontinuity. **Suggestion:** Smooth the transition or make Phase 3 ramp from 0.7→0.85. |
| No quality gating on phase transitions | **ROUGH** | Phase transitions at 30/100 count regardless of success rate. 30 failed runs still trigger Phase 2. **Suggestion:** Require minimum success rate (e.g., 50%) to advance. |
| _chunk_to_workflow_proxy | **ROUGH** | Hardcodes "CheckpointLoaderSimple" and "KSampler" class_types. Loses resolution (no EmptyLatentImage reconstruction). Fragile assumptions. |
| Scale performance (10k+ chunks) | **ROUGH** | Linear scan on every retrieve(). Eviction is O(n log n) per addition at capacity. No indexing, no caching. Acceptable <1k; needs work at scale. |
| Session persistence | **BROKEN** | ExperienceAccumulator is in-memory only. No save/load methods. All chunks lost on process exit. Cross-session learning is impossible. The entire experience loop is ephemeral. **Fix:** Add JSONL persistence, or bridge to existing MemoryAgent backend. |
| Integration with brain.memory | **BROKEN** | Two parallel systems with incompatible schemas. ExperienceChunk uses 4-axis quality + nested parameters. MemoryAgent uses 1-axis quality + flat key_params. No data bridge. No shared storage. **Fix:** Unify schema or build adapter between the two. |

---

## 3. Prediction Accuracy — [PREDICTION × SCOUT]

| Finding | Status | Detail |
|---------|--------|--------|
| Graceful degradation (thin experience) | **SOLID** | Both implementations fall back to priors (0.5 neutral) when experience is absent. Smooth ramp from prior-only to experience-dominant. |
| CWM prediction signal | **ROUGH** | src/cognitive version uses prior rules + experience quality + safety check. Prior rules exist for SD1.5 only (3 params). No SDXL, Flux, SD3 rules. Sparse. |
| Confidence calculation | **ROUGH** | Heuristic math with hardcoded constants (0.3 base, +0.5 for experience, +0.1 for priors, -0.1 per risk). Calibration data is tracked via record_accuracy() but never fed back into the formula. |
| Arbiter surfacing frequency | **ROUGH** | Decision tree based on confidence × improvement threshold, not actual 80/15/5 distribution. Hard cap of 1 explicit per session. Actual frequencies depend entirely on session data. |
| Safety checks: model-family-specific | **ROUGH** | Checks cfg≤0, cfg>30, steps<1, denoise≤0. No per-model ranges (SD1.5 cfg 7-12 vs SDXL cfg 5-9 vs Flux guidance ~1.0). No interaction checks (cfg × steps). |
| Prior rules completeness | **BROKEN** | Only SD1.5 has 3 rules (cfg, steps, denoise). No SDXL, Flux, SD3. No sampler/scheduler/resolution rules. No LoRA/ControlNet rules. Skeletal. **Fix:** Load from agent/profiles/ YAML files which already contain model family knowledge. |
| LIVRPS dual-function claim | **BROKEN** | State mutations use true LIVRPS composition (sort by priority, apply lower→higher). Predictions use weighted averaging with manual safety floor. Same priority constants, completely different composition models. The "one engine, two functions" claim is false. **Fix:** Either implement true LIVRPS prediction composition or drop the claim. |
| Counterfactual validation pipeline | **BROKEN** | generate() creates "what if" alternatives. validate() exists but is never called automatically. No code links a counterfactual ID to a future generation's outcome. get_adjustment() returns corrections but is only consumed if manually passed as counterfactual_adjustment. **Fix:** Wire validation into the experience capture loop. |
| Two parallel CWM implementations | **BROKEN** | src/cognitive/prediction/cwm.py (simple, single-axis) and agent/stage/cwm.py (rich, 6-axis, integrated). Different APIs, different schemas, no migration path. **Fix:** Deprecate src/cognitive version; keep agent/stage as canonical. Or merge the best of both. |

---

## 4. Transport Reliability — [TRANSPORT × SCOUT]

| Finding | Status | Detail |
|---------|--------|--------|
| SchemaCache design | **SOLID** | Well-typed NodeSchema, InputSpec. Combo validation, numeric range checks, connectable node discovery. 41 tests pass. |
| ExecutionEvent design | **SOLID** | 8 typed event types with computed progress_pct, elapsed_ms, is_terminal, is_error. Clean from_ws_message parser. |
| SchemaCache: no global instance | **BROKEN** | Never instantiated outside tests. No singleton. Not wired to comfy_api._get("/object_info"). The cache is dead on arrival — well-designed but never fed. **Fix:** Create singleton in API layer, populate on first /object_info call. |
| SchemaCache: refresh never called | **BROKEN** | Zero call sites for refresh() or async_refresh() in production code. **Fix:** Call refresh() from _handle_get_all_nodes() or on session init. |
| Cache invalidation after install | **BROKEN** | manage_dependencies() sets schema_invalidated=True but never actually calls cache.refresh(). comfy_provision doesn't wire to manage_dependencies(). **Fix:** Hook refresh() after install_node_pack succeeds. |
| ExecutionEvent: no consumers | **BROKEN** | Zero call sites outside tests. Not consumed by pipeline, not wired to progress reporting. Orphaned specification. **Fix:** Create EventAggregator that feeds into progress system. |
| Interrupt endpoint: not exposed | **BROKEN** | interrupt_execution() exists but is not an MCP tool, not called from prediction pipeline, no call sites. **Fix:** Expose as tool, wire to arbiter interrupt decisions. |
| async_refresh vs sync tools | **BROKEN** | async_refresh() expects httpx.AsyncClient but all tool handlers are synchronous. Sync/async impedance mismatch. **Fix:** Remove async_refresh() or provide sync-only alternative. |

---

## 5. Pipeline Completeness — [AUTONOMY × SCOUT]

| Finding | Status | Detail |
|---------|--------|--------|
| Pipeline architecture | **SOLID** | 7-stage loop (intent → compose → predict → gate → execute → evaluate → learn) is well-structured with clear stage transitions and logging. |
| generate_series() | **SOLID** | Variation planning with vary_params cycling and lock_params is complete and correct. |
| Mock mode | **SOLID** | All components work together in simulation mode. 23 tests pass. |
| compose_workflow() | **ROUGH** | Returns parameter dict, not full workflow JSON. Only populates workflow_data if templates are provided. No LoRA/ControlNet selection. It's a parameter tuner, not a workflow composer. |
| Style-locked consistency | **ROUGH** | Consistency comes from experience re-use, not explicit locking. Not tested that parameters actually stay the same across runs. |
| execute_workflow() | **BROKEN** | Stub. Validates workflow exists, returns PENDING, never submits to ComfyUI. No /api/prompt call, no websocket, no output polling. **Fix:** Wire to existing comfy_execute tools or provide default executor. |
| autoresearch ratchet | **BROKEN** | Framework only. No mutations generated. Quality evaluator accepted but never called in the loop. Every step returns UNCHANGED. Immediate "no_evaluator" exit. **Fix:** Implement mutation proposal logic using CWM suggestions. |
| Retry logic | **BROKEN** | Detects quality below threshold, increments counter, logs a message, and exits COMPLETE. No re-execution with adjusted parameters. **Fix:** Actually re-run the pipeline with CWM-suggested adjustments. |
| CognitiveGraphEngine usage | **BROKEN** | Imported but never instantiated in the pipeline. autoresearch accepts engine parameter but ignores it. Compose doesn't use it. **Fix:** Wire engine into compose and autoresearch. |

---

## 6. Scaffold Integrity — [SCAFFOLD × SCOUT]

| Finding | Status | Detail |
|---------|--------|--------|
| Original test regression | **SOLID** | 2140 original tests still pass. 210 new cognitive tests added. Total: 2350 passed. Zero regressions. |
| MCP dispatch transparency | **SOLID** | mcp_server.py dispatch layer doesn't interfere with return values. JSON format unchanged. |
| src.cognitive import safety | **SOLID** | Import wrapped in try/except with graceful fallback. If src/ is missing, existing code path runs unchanged. |
| Sidebar path (load_workflow_from_data) | **SOLID** | Creates engine inside _state_lock. Correct implementation. |
| Wrapper return value format | **SOLID** | All 22 workflow_patch tests pass with identical return JSON format. |
| _patches_to_mutations coverage | **ROUGH** | Only handles replace/add. remove/move/copy fall back to jsonpatch (safe but loses delta tracking). **Acceptable** — these ops are rare in ComfyUI workflows. |
| Test fixture: _engine not reset | **ROUGH** | Test fixture resets _state but not _engine. Engine persists across tests. Low impact but violates isolation. |
| _engine thread safety | **BROKEN** | (Same as GRAPH finding.) Global not protected by lock. Race conditions possible. |
| Session-scoped engine | **BROKEN** | (Same as GRAPH finding.) ensure_graph_engine on SessionContext is dead code. _engine is module-level, not session-scoped. |

---

## Summary Matrix

| Domain | SOLID | ROUGH | BROKEN |
|--------|-------|-------|--------|
| **GRAPH** (Composition) | 4 | 4 | 3 |
| **EXPERIENCE** (Learning) | 2 | 8 | 2 |
| **PREDICTION** (CWM) | 1 | 4 | 4 |
| **TRANSPORT** (Communication) | 2 | 0 | 6 |
| **AUTONOMY** (Pipeline) | 3 | 2 | 4 |
| **SCAFFOLD** (Regression) | 5 | 2 | 2 |
| **TOTAL** | **17** | **20** | **21** |

---

## Priority Classification

### BROKEN → Immediate Fixes (Before Phase 7C)

| # | Finding | Domain | Impact |
|---|---------|--------|--------|
| B1 | _engine thread safety — global not protected by lock | GRAPH + SCAFFOLD | **Data corruption under concurrent access** |
| B2 | _engine module-level — not session-scoped | GRAPH + SCAFFOLD | **Multi-session workflows collide** |
| B3 | Experience persistence — in-memory only | EXPERIENCE | **All learning lost on restart** |
| B4 | Two parallel CWM implementations | PREDICTION | **Architectural confusion, divergent behavior** |
| B5 | LIVRPS dual-function claim is false | PREDICTION | **Architecture doc is misleading** |

### BROKEN → Phase 8 Tasks (Wiring Work)

| # | Finding | Domain | Impact |
|---|---------|--------|--------|
| B6 | SchemaCache never instantiated or fed | TRANSPORT | Transport layer is dead code |
| B7 | ExecutionEvent has no consumers | TRANSPORT | Event system is orphaned |
| B8 | Interrupt not exposed as tool | TRANSPORT | Can't abort failed executions |
| B9 | execute_workflow() is a stub | AUTONOMY | Pipeline never actually generates |
| B10 | autoresearch generates no mutations | AUTONOMY | Ratchet is non-functional |
| B11 | Retry doesn't re-execute | AUTONOMY | Poor quality never corrected |
| B12 | Counterfactual validation not wired | PREDICTION | Learning from "what ifs" impossible |
| B13 | Experience ↔ brain.memory not integrated | EXPERIENCE | Two parallel learning systems |

### ROUGH → Refinement (Phase 8+)

| # | Finding | Domain |
|---|---------|--------|
| R1 | Delta stack unbounded growth | GRAPH |
| R2 | Engine/state sync on jsonpatch fallback | GRAPH |
| R3 | Signature discretization too coarse/fine | EXPERIENCE |
| R4 | Phase transition discontinuity | EXPERIENCE |
| R5 | Prior rules only cover SD1.5 (3 params) | PREDICTION |
| R6 | Confidence not calibrated from data | PREDICTION |
| R7 | compose_workflow returns params not workflows | AUTONOMY |
| R8 | Dead code cleanup (ensure_graph_engine, get_engine, async_refresh) | SCAFFOLD |

---

## Architectural Truth

The Scaffolded Brain has a **solid nervous system** (graph engine) and **good bones** (typed schemas, LIVRPS priority, delta layers, experience chunks). What it lacks is **connective tissue** — the wiring between components that makes the whole system breathe.

The Phase 1-6 build was correct to prioritize foundations over integration. You can't wire what doesn't exist. Now it all exists. The next work is bridges, not buildings.

---

**GATE: Review complete. Awaiting Creative Director classification of ROUGH → Phase 8 tasks and BROKEN → immediate fixes before 7C.**
