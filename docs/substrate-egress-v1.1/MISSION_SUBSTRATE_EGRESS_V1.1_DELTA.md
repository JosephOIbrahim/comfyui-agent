# MISSION SUBSTRATE EGRESS V1.1 — RFC DELTA

**Scope.** Architectural physics. The seven axioms governing Substrate Egress v1.1 — `[RFC]` sections only. Companion to `CLAUDE_CODE_EXECUTION_QUEUE.md`, which holds the parked syntax (`[EXEC SPEC]` sections) and cites these axioms.

**Source.** Split from `SUBSTRATE_EGRESS_V1.1_BUNDLE.md` (Mile 9 deliverable, post-R6 scrubbing). Axiom content preserved verbatim.

---

## Spine quick-reference *(both files inherit)*

| # | Axiom | Frame |
|---|---|---|
| 1 | WAL + materialized view *(LIVRPS-tick stamping; Intra-Session Blindness)* | Distributed Systems |
| 2 | Resource partitioning *(BGE on CPU; hardware mutex avoidance)* | Distributed Systems |
| 3 | Statistical density floor *(empirically-tunable thresholds; council reads replay artifact)* | Information Theory |
| 4 | Open-loop preemption *(current state lexically overrides advisory priors)* | Control Theory |
| 5 | Closed-loop validation *(Shadow + Poison Pill + Held-Out Replay + Environment Pinning)* | Control Theory |
| 6 | Crash-boundary atomicity *(per-session isolation; atomic rename; Last-Valid-Tick Truncation)* | Distributed Systems |
| 7 | Semantic compression boundary *(intent-topology deduplication with thrash_count preservation)* | Information Theory |

---

# AXIOM 1 — WAL + materialized view

## [RFC]

Brain JSONL, cognitive JSONL, and USD prims function as a synchronous Write-Ahead Log. Three writers stamp every cross-store write with a **causal sequence ID** — the canonical clock for the session.

The clock satisfies five invariants:

1. **Read-only.** Mutation by observers would poison causality. Enforced at the type-system level of the host language.
2. **Strictly monotonic** across consecutive reads. The bridge raises `CausalInversionError` and triggers Last-Valid-Tick Truncation *(Axiom 6)* on detection of non-monotonic sequence at ingest.
3. **Idempotent within a state-transition boundary.** Multiple reads at the same agent cognitive step must return identical values. This guarantees that async writers in a single step share one logical tick — the property that makes fsync barrier batching mathematically clean.
4. **Decoupled from bounded-queue eviction.** The clock is an absolute cumulative odometer of irreversible state transitions, never a recycling array index. Bounded-queue evictions in the State Spine MUST NOT mutate the clock.
5. **Resets only on session boundary.** Per-session UUID partitioning *(Axiom 6)* makes the session UUID the canonical epoch.

Async tools must receive the clock value **by-value at invocation**, not query it at write-completion. Querying at completion would stamp a future tick onto a write spawned by a past intent — causal inversion.

Out-of-band writes *(initialization telemetry, pre-loop state)* anchor to a reserved sentinel value distinguishable from any in-loop tick.

The clock primitive is named `causal_sequence_id`.

**Intra-Session Blindness.** Moneta operates as an asynchronous materialized view over the WAL — vector-indexed projection optimized for cross-session associative retrieval. The agent **cannot** query its own current-session deposits. Consolidation is session-end only. The user-facing semantic is *"the agent remembers across sessions"*; intra-session recall is served by the LIVRPS State Spine working memory, not by Moneta.

Consolidation occurs at session-end via the bridge boundary. The consolidated artifact is a low-entropy summary of the WAL — see Axiom 7 for the compression contract.

---

# AXIOM 2 — Resource partitioning under shared physical substrate

## [RFC]

The agent loop is destabilized by VRAM contention with the diffusion pipeline. The embedding workload must execute on a hardware partition strictly isolated from the GPU VRAM pool consumed by diffusion.

The selected partition's encoding cost must remain negligible relative to the agent turn budget. The cost ceiling is a behavioral invariant, not a target — embedding inference that approaches the budget signals partition mis-selection.

This is hardware mutex avoidance, not loop stabilization. The control-theoretic stabilization downstream of the partition is a consequence, not the cause.

---

# AXIOM 3 — Statistical density floor for evidence-based vote

## [RFC]

The Shadow → Live cutover gate is volumetric, not temporal. The MOE council vote on ARCH-1 schedules when telemetry meets BOTH:

- A **volume threshold** — minimum query count recorded
- A **diversity threshold** — queries distributed across a minimum number of distinct intent topologies, with intent-class clustering methodology locked at start of shadow

Both thresholds are **empirically tunable heuristics, not derived from BGE vector space mathematics.** Recalibration is required post-Phase 2 telemetry once corpus statistics are observed.

Wall-clock time is irrelevant. If usage is dense, the gate clears in days; if usage is bursty, it may take months. Either is fine — the goal is statistical density sufficient to support an evidence-based vote.

The artifact the council reads at vote time is the **Held-Out Replay results** *(see Axiom 5)*. Latency histograms and DB-read charts are appendices, not the headline.

---

# AXIOM 4 — Open-loop preemption rule

## [RFC]

Current session working state strictly overrides advisory priors at the prompt boundary. The override is enforced via explicit lexical instruction in the prompt template — not via attention-weighting heuristics, not via cognitive-architecture analogy.

Without an explicit override, attention dilution causes probabilistic historical priors to compete with deterministic current state, producing agent paralysis or override of fresh user constraints.

**Acceptance** *(behavioral guarantee, language-agnostic)*: under contradictory working-memory rejection vs. episodic-memory affirmation, the agent must honor the working-memory rejection in 100% of test cases. The Held-Out Replay harness *(Axiom 5)* provides the test surface.

This is open-loop preemption, not working-memory dominance. The rule operates at the prompt boundary; it does not invoke any cognitive-architecture mechanism.

---

# AXIOM 5 — Closed-loop validation of an open-loop observation pattern

## [RFC]

Shadow Mode is open-loop observability — it proves the database read works but cannot validate that injecting retrieved context makes the agent smarter. Closing the loop requires three mechanisms running concurrently during the shadow window:

**Poison Pill canary.** Open-loop observability cannot detect read-path payload saturation. If retrieved priors would exceed the LLM's context window or break prompt parsing, shadow telemetry reports 100% health while live cutover would crash. The canary closes this gap by simulating live-cutover prompt rendering during shadow without injection. Any *"would crash live"* event is a hard signal that the prompt template + retrieval combination needs adjustment before going live.

**Held-Out Replay harness.** The artifact the MOE council reads at vote time. Telemetry alone doesn't prove the agent gets smarter for using retrieved memory — replay does. The replay corpus is curated once and **locked at shadow start**; re-running replay across shadow-period checkpoints produces a time series the council can read.

**Environment Pinning.** Every shadow telemetry write must record a content-addressed identifier of:
1. The active embedding model file
2. The active system prompt
3. The hardware execution context *(specifically, the partition required by Axiom 2)*

The MOE council vote auto-aborts on identifier drift across the validation sample set. Without environment pinning, mid-window drift in any of these poisons the replay artifact silently.

**Bounded injection.** Episodic memory injection is capped by a hard token budget. If retrieved priors exceed the budget, oldest records are truncated first before prompt rendering.

---

# AXIOM 6 — Crash-boundary atomicity

## [RFC]

Bridge consolidation requires explicit session commit. Three invariants govern the boundary:

**Per-session isolation.** Each session writes to a session-isolated WAL file. Concurrent sessions *(multi-process today; fleet deployment future)* cannot interleave bytes into a shared log. Isolation is enforced at the filename level, not at the application level.

**Atomic commit-via-rename.** On graceful close, the bridge performs an atomic filesystem rename from an *active* state to a *committed* state. The rename is the commit; no separate marker write is required. Atomicity is guaranteed by the underlying filesystem, not by application logic.

**Last-Valid-Tick Truncation on torn write.** On bridge boot, an orphaned active file *(indicating a hard crash — OOM-kill, kernel panic, power loss)* does NOT result in tombstoning. The bridge performs truncation: the final torn record is stripped; the remainder is ingested. This salvages partial-session telemetry rather than discarding entire sessions.

**Durability barrier before commit.** The shutdown sequence mandates a blocking durability barrier *(fsync-equivalent on POSIX systems; corresponding primitive on other platforms)* across all writer file descriptors before executing the atomic rename. Without the barrier, async writes can leave valid records orphaned across the commit boundary.

The Last-Valid-Tick Truncation invariant is load-bearing for Axiom 3's volumetric gate — without it, the most complex sessions *(which crash most frequently)* would systematically delete themselves, starving the council of the exact telemetry needed to vote.

---

# AXIOM 7 — Semantic compression boundary

## [RFC]

The bridge performs lossy compression prior to vector embedding. Without compression, raw WAL records *(failed tools, syntax errors, retry loops, wait-states)* dominate the consolidated artifact and overwhelm Moneta's retrieval channel capacity with noise.

**Compression rule.** Contiguous WAL records sharing identical intent topologies are collapsed into a single record. The `causal_sequence_id` of the **first occurrence** is retained. A `thrash_count` integer is appended to the surviving record, preserving *volume* of repeated transitions even when *sequence* is collapsed.

This preserves the unique topological sequence of cognitive transitions and their failure states; it discards zero-delta micro-thrashing.

**Performance budget.** The compression operation must complete within a wall-clock budget at session-end consolidation. Exceeding the budget logs a `SlowConsolidationWarning` but does NOT fail the consolidation — slow is acceptable, dropped is not.

**Error handling.** Malformed records *(e.g., torn writes from a hard crash that survived Axiom 6's truncation pass)* are dropped with a logged warning; valid records continue through. The compression operation **must never** tombstone the session over a parsing error.

**Phase 2.5 compatibility.** This compression is a structuring algorithm — it prunes the topological graph. The deferred Phase 2.5 work *(`signal_attention` + vision evaluator)* is a weighting algorithm — it maps retroactive confidence scores onto the deduplicated nodes this compression produces. The two are orthogonal; Phase 2.5 enhances v1.1 compression rather than replacing it.

---

# DECISIONS LOGGED *(carried forward from R1–R6)*

These decisions were made during the adversarial review process and are not relitigated:

**Adopted from Gemini Round 2:**
- Tiered-memory framing rejected → WAL + materialized view *(distributed-systems-native)*
- Q2 freeze-decay rejected *(violates Moneta-frozen-as-law)*
- HF_HOME hybrid approach for BGE-small download

**Adopted from Gemini Round 3:**
- All 6 R3 absorbs *(Crash-Boundary Atomicity new axiom; Edit 4 heuristic tagging; Environment Pinning sub-claim; Axiom 1 narrowing with LIVRPS-tick reuse + Intra-Session Blindness; Axiom 2 reframe to DS; Axiom 4 reframe to CT — cognitive science evicted)*

**Adopted from Gemini Round 4:**
- All 6 R4 absorbs *(LIVRPS-tick repairs: get_absolute_tick + by-value + PRE_LOOP; Axiom 6 rewrite with atomic renames + Last-Valid-Tick Truncation; Environment Pinning + `_CPU_FP32` hardware context; fsync barrier; Intra-Session Blindness × demo arc audit; Axiom 7 NEW — Semantic Compression Boundary)*

**Adopted from Gemini Round 5:**
- API name `causal_sequence_id` over `get_absolute_tick()`
- Idempotency-within-state-transition-boundary as the load-bearing CT property
- Intent-topology deduplication as Axiom 7 default *(terminal-state-extraction and LLM-summarization both fatal for Phase 2.5 compatibility)*
- `thrash_count` field for volume preservation

**Adopted from Gemini Round 6:**
- Causal-Invariant Code Trace as verification protocol *(returns `PRIMARY_OK` / `FALLBACK_REQUIRED` / `PARTIAL_REFINE_CONTRACT`)*
- Rust Translation Test as RFC/Exec-Spec classification rule
- Canonical Ownership Rule for dual-membership artifacts

**Pushbacks rejected with reasoning:**
- Semantic checksumming at boot *(R3 A3 CRUCIBLE)* — single-workstation deployment, deferred to fleet hardening
- Semantic deduplication at >0.98 cosine *(R4 B4)* — bursty iteration produces near-duplicates by design; this is the data signal_attention will need

---

# OPEN ITEMS

**Verification protocol — must run before Edit 5's Exec Spec entry collapses to one branch:**

Apply R6 Q1 Causal-Invariant Code Trace to `Moneta/` repo, LIVRPS State Spine class. Verdict determines which Exec Spec branch under Edit 5 is canonical.

**MOE council scope — parked for council:**
- Continuous in-session consolidation *(v2 question)*
- Cross-hardware vector stability *(future hardening when bridge ships beyond single workstation)*

**Demo arc audit — conditional:**
Fires only if verification protocol returns `FALLBACK_REQUIRED`. Without native LIVRPS integer, 1:1 scalar mapping of tick-to-UI-state is lost; demo language must update to remove any *"frame-perfect UI synchronization during replay"* claim.
