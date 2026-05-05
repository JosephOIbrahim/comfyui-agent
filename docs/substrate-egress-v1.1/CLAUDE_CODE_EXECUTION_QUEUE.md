# CLAUDE CODE EXECUTION QUEUE — SUBSTRATE EGRESS V1.1

**Scope.** Parked syntax. The seven `[EXEC SPEC]` sections — symbol names, type signatures, threshold values, file paths, acceptance tests — that operationalize the RFC delta. Each section cites the axiom it implements *(per Canonical Ownership Rule)*.

**Companion.** `MISSION_SUBSTRATE_EGRESS_V1.1_DELTA.md` holds the architectural physics *(the `[RFC]` axioms)*. Do not modify spec content here without first updating the corresponding RFC.

**Source.** Split from `SUBSTRATE_EGRESS_V1.1_BUNDLE.md` (Mile 9 deliverable, post-R6 scrubbing). Spec content preserved verbatim.

**Path D.** Markdown only. No code execution until Jun 16.

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

## [EXEC SPEC]

*Implements RFC §Axiom 1.*

**Primary path** *(if R6 Q1 protocol returns `PRIMARY_OK` or `PARTIAL_REFINE_CONTRACT`)*:

```python
@property
def causal_sequence_id(self) -> int:
    """
    Strictly monotonic, read-only uint64. Cumulative state transitions.
    
    Behavioral guarantees (per RFC §Axiom 1):
    - Read-only: setter raises AttributeError.
    - Idempotency: identical return within state-transition boundary.
    - Decoupling: never resets/decrements during deque eviction.
    - Reset: only on new session UUID.
    
    Failure modes:
    - Pre-init access: RuntimeError.
    - Non-monotonic detected at ingest: CausalInversionError + truncate.
    """
```

Wraps existing LIVRPS internal counter (per protocol verdict). Type: `int` (Python `uint64`-equivalent — 18 quintillion IDs; wraps in 584M years at 1000 steps/sec).

**Fallback path** *(if R6 Q1 protocol returns `FALLBACK_REQUIRED`)*:

```python
# Sampled exactly once at top of LIVRPS while-loop iteration.
# Passed by-value through async writer payloads.
loop_tick = time.monotonic_ns()
```

Loses 1:1 mapping to LIVRPS UI render states. Demo arc audit fires if this branch is selected *(R6 synthesis-killer)*.

**Other parked specifics:**

- Sentinel value for pre-loop writes: `PRE_LOOP = -1`
- Exception classes: `CausalInversionError(Exception)`, raised by bridge ingest layer
- By-value passing: tool payloads include `"causal_sequence_id": int` field; tools read from payload, not from a live State Spine reference

---

# AXIOM 2 — Resource partitioning under shared physical substrate

## [EXEC SPEC]

*Implements RFC §Axiom 2.*

```python
@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

def _embed(text: str, dim: int = DIMENSION) -> list[float]:
    vec = _model().encode(text, normalize_embeddings=True).tolist()
    assert len(vec) == dim
    return vec
```

**Parked specifics:**

- Model: BGE-small-en-v1.5 *(dual-membership name; RFC-level abstraction is "embedding model")*
- Device: `cpu`
- Expected encoding latency: 30–50ms *(observation, not contract)*

**Acceptance test:**

- BGE-small loads with `device='cpu'` — verified by inspecting the loaded model's `device` attribute
- No CUDA memory allocated by `sentence-transformers` during `_embed` calls

---

# AXIOM 3 — Statistical density floor for evidence-based vote

## [EXEC SPEC]

*Implements RFC §Axiom 3.*

**Threshold values** *(empirical heuristics, locked for v1.1, recalibrate post-Phase 2)*:

- Volume: `N_MIN = 500`
- Diversity: `TOPOLOGIES_MIN = 4`

**Acceptance:**

- Telemetry gate clears volumetric thresholds: `count(_safe_query) >= N_MIN` AND `len(distinct_topologies) >= TOPOLOGIES_MIN`
- Topology clustering methodology committed to repo at shadow start *(file path TBD at implementation time)*

---

# AXIOM 4 — Open-loop preemption rule

## [EXEC SPEC]

*Implements RFC §Axiom 4.*

```python
def _summarize_memory(memory) -> dict:
    """Translation layer: Moneta Memory → LLM-readable advisory context.
    
    Output is wrapped in <ADVISORY_PRIORS> tags downstream and explicitly
    subordinated to current working state per ADVISORY_PRIOR_OVERRIDE_RULE.
    """
    payload = json.loads(memory.payload)
    return {
        "summary": payload.get("workflow_summary", ""),
        "key_params": payload.get("key_params", {}),
        "quality": payload.get("quality_score"),
        "feedback": payload.get("user_feedback"),
        "age_days": _age_days(payload.get("timestamp")),
    }

ADVISORY_PRIOR_OVERRIDE_RULE = (
    "CRITICAL: Current session working state STRICTLY OVERRIDES "
    "advisory priors. Ignore priors if they contradict the immediate context."
)
```

**Prompt template fragment** *(downstream of `_summarize_memory`)*:

```
<ADVISORY_PRIORS>
[priors here, formatted by _summarize_memory]
</ADVISORY_PRIORS>

CRITICAL: Current session working state STRICTLY OVERRIDES advisory
priors. Ignore priors if they contradict the immediate context.
```

**Parked specifics:**

- XML tag choice: `<ADVISORY_PRIORS>` *(stable identifier; renames go through RFC)*
- Override rule constant name: `ADVISORY_PRIOR_OVERRIDE_RULE`

---

# AXIOM 5 — Closed-loop validation of an open-loop observation pattern

## [EXEC SPEC]

*Implements RFC §Axiom 5.*

```python
def _poison_pill_canary(retrieved_priors: list[dict],
                        prompt_template: str,
                        token_budget: int = 400) -> dict:
    """Compile hypothetical live prompt during shadow. Drop after validation.
    
    Returns canary metrics: token_count, within_budget, parse_ok, would_crash_live.
    """
    hypothetical_prompt = _render_with_priors(prompt_template, retrieved_priors)
    return {
        "token_count": _count_tokens(hypothetical_prompt),
        "within_budget": _count_tokens(hypothetical_prompt) <= token_budget,
        "parse_ok": _validate_prompt_parses(hypothetical_prompt),
        "would_crash_live": (
            _count_tokens(hypothetical_prompt) > token_budget
            or not _validate_prompt_parses(hypothetical_prompt)
        ),
    }

# tests/integration/test_held_out_replay.py
def test_held_out_replay_efficacy():
    """50 historical intents through offline LLM, with vs. without retrieved context.
    Measure tool-call divergence. Output: divergence histogram + qualitative diff log.
    """
    intents = _load_replay_corpus()  # locked at shadow start
    baseline = [_offline_llm_no_priors(i) for i in intents]
    augmented = [_offline_llm_with_priors(i, _safe_query(i.text)) for i in intents]
    _emit_council_artifact(
        _measure_divergence(baseline, augmented),
        _human_reviewable_diff(baseline, augmented),
    )
```

**Environment Pinning hash invariant:**

```python
ENV_FINGERPRINT = sha256(
    open(MODEL_FILE_PATH, 'rb').read()
).hexdigest() + "_CPU_FP32"
# Append SHA256 of system prompt string at every shadow telemetry write.
```

**Parked specifics:**

- Token budget: 400 tokens *(empirical; locked for v1.1)*
- Replay corpus size: 50 intents *(locked at shadow start)*
- Canary log path: `logs/shadow_canary.jsonl`
- Hash algorithm: SHA-256
- Hardware suffix: `"_CPU_FP32"` *(must update if Axiom 2 partition ever changes)*

**Acceptance:**

- `would_crash_live: true` count is zero across the telemetry window
- Replay corpus locked, replay test runnable
- Hard token budget enforced; oldest-first truncation verified
- Hash drift detection raises auto-abort signal to council

---

# AXIOM 6 — Crash-boundary atomicity

## [EXEC SPEC]

*Implements RFC §Axiom 6.*

**Filename pattern:**

- Active session: `wal_{session_uuid}.jsonl.active`
- Committed session: `wal_{session_uuid}.jsonl`

**Commit sequence on graceful close:**

```python
def _commit_session(session_uuid: str) -> None:
    fds = _get_open_writer_fds(session_uuid)
    for fd in fds:
        os.fsync(fd)  # blocking durability barrier
    active_path = f"wal_{session_uuid}.jsonl.active"
    final_path = f"wal_{session_uuid}.jsonl"
    os.rename(active_path, final_path)  # atomic on POSIX
```

**Boot-time recovery:**

```python
def _recover_orphaned_active_files(wal_dir: Path) -> None:
    for active_file in wal_dir.glob("wal_*.jsonl.active"):
        truncated = _last_valid_tick_truncate(active_file)
        truncated.rename(active_file.with_suffix(""))  # remove .active
```

**Truncation logic:**

- Read records from end of file backwards until a complete-and-valid JSON record is reached
- Strip everything after that record
- Rename to committed name

**Parked specifics:**

- Suffix strings: `.active`, `.jsonl`
- Filename template: `wal_{uuid}.jsonl[.active]`
- Recovery on boot is mandatory before any new session begins ingest

---

# AXIOM 7 — Semantic compression boundary

## [EXEC SPEC]

*Implements RFC §Axiom 7.*

```python
def deduplicate_intent_topologies(
    wal_stream: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Input: parsed JSON dicts from session WAL, chronologically sorted.
    Output: list matching v1.1 Moneta outcomes.jsonl shape.
    
    Per RFC §Axiom 7:
    - Collapse contiguous identical-topology records.
    - Retain causal_sequence_id of first occurrence.
    - Append thrash_count to surviving node.
    - On malformed record: log warning, drop, continue. NEVER tombstone.
    - Wall-clock budget: BUDGET_MS; exceeding logs SlowConsolidationWarning.
    """
```

**Implementation hint:** `itertools.groupby` with topology-extraction key function.

**Performance budget:** `BUDGET_MS = 500` for a 10,000-line WAL.

**Parked specifics:**

- Type signature: `Iterable[Dict[str, Any]] -> List[Dict[str, Any]]`
- Output schema: matches existing v1.1 `outcomes.jsonl` shape
- New field on surviving record: `thrash_count: int`
- Warning class: `SlowConsolidationWarning(UserWarning)`
- Topology-extraction key: TBD at implementation time *(stable hash of intent + key params)*

**Acceptance:**

- Compression ratio observed: 5:1 to 15:1 for typical WAL profiles
- 100% deterministic given same input
- <500ms wall-clock for 10K-line WAL on Joe's workstation profile
- Phase 2.5 forward-compatibility: output schema accommodates `signal_attention_weight: float` field added later without re-deduplication

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
