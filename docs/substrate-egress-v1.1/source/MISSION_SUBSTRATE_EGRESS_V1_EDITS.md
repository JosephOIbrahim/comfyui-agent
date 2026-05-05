# MISSION DOC EDITS — Substrate Egress v1

**Date:** May 1, 2026
**Source mission:** `MISSION — Substrate Egress v1`
**Driver:** Gemini Round 1 + Round 2 review · synthesis adopted
**Type:** Targeted delta · Five edits · Not a redraft
**Apply order:** Sequential, top-to-bottom

---

## Marathon markers

```
Edit 1 ─── Phase 1: BGE inference forced to CPU
Edit 2 ─── Phase 2 Point 1: ADVISORY_PRIORS XML + lexical override rule
Edit 3 ─── Phase 2 Point 1: shadow scaffolding (Poison Pill canary + Held-Out Replay)
Edit 4 ─── Telemetry gate: calendar → volumetric (N≥500, ≥4 topologies)
Edit 5 ─── Opening framing: WAL + materialized view, not isolated peers
─── EDITS COMPLETE ───
```

Each edit below has three parts: **WHERE** *(section locator)*, **WHY** *(one-sentence rationale)*, **REPLACEMENT** *(the new text)*.

---

# EDIT 1 — Force BGE inference onto CPU

### WHERE

**Phase 1 → "The model — BGE-small-en-v1.5"** section · the `_embed` implementation block.

### WHY

The 4090 runs the diffusion pipeline. BGE-small competing for VRAM during intent capture risks CUDA OOM and 1.5s VRAM-swap stalls in the synchronous agent loop. CPU inference adds ~30-50ms — negligible against agent turn budget — and completely isolates the egress memory pool. *(Adopted from Gemini Round 2 B5.)*

### REPLACEMENT

Replace the `_embed` block with:

```python
# vector.py — _embed implementation
from functools import lru_cache

@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    # CPU-only by deliberate choice. BGE-small is ~130MB and CPU
    # encoding adds ~30-50ms per query — negligible against agent
    # turn budget. This isolates the egress memory pool from
    # ComfyUI's VRAM, preventing CUDA OOM and VRAM-swap stalls
    # in the synchronous agent loop.
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

def _embed(text: str, dim: int = DIMENSION) -> list[float]:
    vec = _model().encode(text, normalize_embeddings=True).tolist()
    assert len(vec) == dim, f"BGE-small returned dim={len(vec)}, expected {dim}"
    return vec
```

Add a Phase 1 acceptance criterion under **"Phase 1 acceptance criteria"**:

- [ ] BGE-small runs strictly on `device='cpu'` — verified by inspecting the loaded model's device attribute and confirming no CUDA memory is allocated by `sentence-transformers` during `_embed` calls

---

# EDIT 2 — ADVISORY_PRIORS XML structure + lexical override rule

### WHERE

**Phase 2 → Point 1 — `intent_collector.capture_intent`** · the `_summarize_memory` translation layer + the prompt structure that injects retrieved priors.

### WHY

When working memory holds a fresh user rejection *("user just rejected DPM++ 2M Karras")* and episodic memory retrieves *("DPM++ 2M Karras worked three weeks ago")*, the agent risks paralysis or override of the current constraint. **This is the highest-priority failure mode found in Round 2.** Without an explicit lexical priority rule, attention dilution causes probabilistic historical priors to compete with deterministic current state. *(Adopted from Gemini Round 2 B3 — verbatim mitigation.)*

### REPLACEMENT

In **`_summarize_memory`**, modify the return shape to include explicit advisory framing:

```python
def _summarize_memory(memory) -> dict:
    """Turn a Moneta Memory into LLM-readable advisory context.

    Output is wrapped in <ADVISORY_PRIORS> tags downstream and
    explicitly subordinated to current working state in the
    system prompt. See ADVISORY_PRIOR_OVERRIDE_RULE below.
    """
    payload = json.loads(memory.payload)
    return {
        "summary": payload.get("workflow_summary", ""),
        "key_params": payload.get("key_params", {}),
        "quality": payload.get("quality_score"),
        "feedback": payload.get("user_feedback"),
        "age_days": _age_days(payload.get("timestamp")),
    }
```

Add a new constant in `intent_collector.py`:

```python
# This rule is non-negotiable. Episodic priors must never override
# current session state. Without this, agent paralysis is observable
# under conflicting working/episodic memory (Gemini Round 2 B3).
ADVISORY_PRIOR_OVERRIDE_RULE = (
    "CRITICAL: Current session working state STRICTLY OVERRIDES "
    "advisory priors. Ignore priors if they contradict the immediate context."
)
```

In the prompt construction *(wherever `prior_context` flows back to the LLM)*, wrap the retrieved memories explicitly:

```python
if prior:
    response["prior_context"] = {
        "format": "advisory_priors",
        "override_rule": ADVISORY_PRIOR_OVERRIDE_RULE,
        "priors": [_summarize_memory(m) for m in prior],
    }
```

The downstream prompt template must render this as:

```
<ADVISORY_PRIORS>
[priors here, formatted by _summarize_memory]
</ADVISORY_PRIORS>

CRITICAL: Current session working state STRICTLY OVERRIDES advisory
priors. Ignore priors if they contradict the immediate context.
```

Add a Phase 2 acceptance criterion:

- [ ] Held-Out Replay test confirms that when working memory contains a parameter rejection AND Moneta retrieves a contradictory historical success, the agent honors the working memory rejection in 100% of cases *(see Edit 3 for replay harness)*

---

# EDIT 3 — Shadow scaffolding: Poison Pill canary + Held-Out Replay harness

### WHERE

**Phase 2 → Point 1 — `intent_collector.capture_intent`** · new subsection added directly after Point 1's pattern block, before Point 2.

### WHY

Shadow Mode by itself is open-loop observability — it proves the database read works but cannot validate that injecting retrieved context makes the agent smarter. Two mechanisms close the gap: a Poison Pill canary that catches shadow-looks-healthy / live-cutover-explodes failures during the shadow window, and a Held-Out Replay harness that produces the council-readable artifact proving control efficacy. *(Adopted from Gemini Round 2 A2 — both mitigations.)*

### REPLACEMENT

Add this new subsection after Point 1's pattern block:

---

#### Point 1 — Shadow Mode scaffolding

Phase 2 Point 1 ships in **Shadow Mode** before going live. `intent_collector` performs `_safe_query`, computes vector distances, and logs results to disk — but does NOT inject the retrieved context into the LLM prompt. This honors the MOE council process *(ARCH-1 vote on outside-in vs inside-out)* while accumulating empirical telemetry that informs the vote.

Two mechanisms run during Shadow Mode:

**Poison Pill canary.** Open-loop observability of database reads cannot detect read-path payload saturation. If retrieved priors would exceed the LLM's context window or break prompt parsing, shadow telemetry reports 100% health while live cutover would crash instantly. The canary closes this gap:

```python
def _poison_pill_canary(retrieved_priors: list[dict],
                        prompt_template: str,
                        token_budget: int = 400) -> dict:
    """Compile the hypothetical live prompt during shadow mode.

    Logs token length and parse validation. Drops the prompt
    after validation — never injected into the LLM during shadow.
    Returns canary metrics for telemetry.
    """
    hypothetical_prompt = _render_with_priors(prompt_template, retrieved_priors)
    token_count = _count_tokens(hypothetical_prompt)
    parse_ok = _validate_prompt_parses(hypothetical_prompt)
    return {
        "token_count": token_count,
        "within_budget": token_count <= token_budget,
        "parse_ok": parse_ok,
        "would_crash_live": (token_count > token_budget) or (not parse_ok),
    }
```

Canary metrics are written to `logs/shadow_canary.jsonl`. Any `would_crash_live: true` event is a hard signal that the prompt template + retrieval combination needs adjustment before going live.

**Held-Out Replay harness.** This is the artifact the MOE council reads at vote time. Telemetry alone doesn't prove the agent gets smarter for using retrieved memory — replay does:

```python
# tests/integration/test_held_out_replay.py
def test_held_out_replay_efficacy():
    """Route 50 historical intents through an offline LLM with
    shadow-retrieved context injected. Measure tool-call divergence
    against baseline (no retrieval).

    Output: divergence histogram + qualitative diff log.
    Council reads this, not raw telemetry.
    """
    intents = _load_replay_corpus()  # 50 historical intents, fixed
    baseline_calls = [_offline_llm_no_priors(i) for i in intents]
    augmented_calls = [_offline_llm_with_priors(i, _safe_query(i.text)) for i in intents]

    divergence = _measure_divergence(baseline_calls, augmented_calls)
    quality_delta = _human_reviewable_diff(baseline_calls, augmented_calls)

    _emit_council_artifact(divergence, quality_delta)
```

The replay corpus *(50 intents, fixed)* is curated once and locked. Re-running replay across shadow-period checkpoints produces a time series the council can read.

**Hard token budget.** Episodic memory injection is capped at **400 tokens**. If `_safe_query` returns priors exceeding the budget, oldest records are truncated first *before* prompt rendering. *(Adopted from Gemini Round 2 B1 mitigation.)*

Add Phase 2 acceptance criteria:

- [ ] Shadow Mode runs `_poison_pill_canary` on every `_safe_query` result · `would_crash_live: true` count is zero across the telemetry window
- [ ] Held-Out Replay corpus *(50 intents)* curated and locked · `tests/integration/test_held_out_replay.py` runnable
- [ ] Hard 400-token budget enforced on episodic memory injection · oldest-first truncation verified

---

# EDIT 4 — Telemetry gate: calendar → volumetric

### WHERE

**Phase 2 acceptance criteria** + the synthesis sequencing in any "after shadow runs..." language.

### WHY

Bursty user iteration means calendar time is uncorrelated with statistical density of the telemetry corpus. A 30-day window with 45 deposits in a tight cosine cluster produces a council-ready artifact that's actually statistical noise. Volumetric thresholds tie the council vote to corpus diversity, not wall-clock time. *(Adopted from Gemini Round 2 B2.)*

### REPLACEMENT

Replace any sequencing language that says *"after ~30 days of shadow telemetry"* or similar with:

> **The Shadow → Live cutover gate is volumetric, not temporal.** The MOE council vote on ARCH-1 is scheduled when telemetry meets BOTH thresholds:
>
> - **Volume:** N ≥ 500 `_safe_query` invocations recorded
> - **Diversity:** Queries distributed across ≥ 4 distinct intent topologies *(measured by intent-class clustering of the user-request strings — methodology locked at the start of shadow)*
>
> Wall-clock time is irrelevant. If usage is dense, the gate clears in days; if usage is bursty, it may take months. Either is fine — the goal is statistical density sufficient to support an evidence-based vote, not a fixed timeline.
>
> The artifact the council reads at vote time is the **Held-Out Replay results** *(see Phase 2 Point 1 shadow scaffolding)*, not raw telemetry. Latency histograms and DB-read charts are appendices, not the headline.

Add a Phase 2 acceptance criterion:

- [ ] Telemetry gate clears volumetric thresholds: N ≥ 500 queries · ≥ 4 distinct intent topologies · methodology for topology clustering committed to repo at shadow start

---

# EDIT 5 — Opening framing: WAL + materialized view

### WHERE

**Top of mission · "Why This Mission, In One Paragraph"** section.

### WHY

The original framing said "powered by Moneta is brand-true and bridge-true but not code-true inside Comfy-Cozy." That's accurate but incomplete. After Round 2, the architectural relationship between the existing experience stores *(brain JSONL, cognitive JSONL, USD prims)* and Moneta is best framed as **WAL + materialized view**, not "isolated peers" and not "tiered memory hierarchy." This framing survives first-principles attack from distributed systems theory, doesn't depend on a cognitive-science metaphor that wobbles under pressure, and is the framing the MOE council should vote against. *(Adopted from Gemini Round 2 A1 — distributed-systems framing, with the consolidation critique restated honestly.)*

### REPLACEMENT

Replace the existing "Why This Mission, In One Paragraph" with:

> ## Why This Mission, In One Paragraph
>
> The bridge writes outcomes to Moneta with a **deterministic synthetic vector keyed off the session string** — a deliberate v0 trade-off, not a defect. It supported the same-session cold-vs-warm replay demo with stdlib-only dependencies and microsecond ingest latency. But the demo arc has expanded: cosine retrieval today returns deposits *from the same session*, not deposits *with similar meaning*. Comfy-Cozy itself never queries Moneta; the substrate is write-only from the consumer's perspective. **"Powered by Moneta" is brand-true and bridge-true but not code-true inside Comfy-Cozy.**
>
> This mission graduates v0 → v1 by replacing synthetic embeddings with real BGE-small embeddings *and* wiring the brain to query the substrate.
>
> **The architectural framing that survives first-principles review:** brain JSONL, cognitive JSONL, and USD prims function as a **synchronous Write-Ahead Log** *(deterministic, intra-session, transactional)*. Moneta functions as an **asynchronous materialized view** over the WAL — a vector-indexed projection optimized for cross-session associative retrieval. The bridge performs consolidation at session-end timing: `outcomes.jsonl` is the low-entropy summary that gets written to Moneta. *(In-session continuous consolidation is a v2 question, parked for the council.)*
>
> This is not "isolated peers." It is not "tiered cognitive memory." It is a WAL and its materialized view, with an explicit consolidation step at the bridge boundary. That framing carries to the MOE council vote on ARCH-1 / ARCH-2.

---

## Cross-references unchanged

The following sections of the original mission doc remain untouched:

- Phase 1 model selection *(BGE-small-en-v1.5)*
- Phase 1 storage migration strategy *(new `v2/` URI)*
- Phase 1 capsule.py adjustment
- Phase 1 test suite *(semantic similarity, deterministic embedding, etc.)*
- Phase 2 graceful degradation contract *(`_safe_query` pattern)*
- Phase 2 brain config / storage URI alignment
- Phase 2.5 deferral of optimizer prior + Point 2 signal_attention
- Out-of-scope list *(FailureModePrim, cozy_doctor, agentskills.io, query_by_metadata)*
- Risk table *(no edits — all five mitigations land in this delta covered failure modes new to Round 2)*

---

## Pushbacks NOT incorporated *(decisions logged for council)*

Two Gemini Round 2 recommendations were rejected with reasoning. Logged here so they don't resurface as accidental gaps:

**Semantic checksumming at boot** *(Gemini A3 CRUCIBLE)* — rejected for v1. Hash-equality is the wrong invariant in theory; in practice, Joe runs one workstation with one GPU, so cross-hardware drift cannot fire. Filed as future hardening for fleet deployment.

**Semantic deduplication at >0.98 cosine** *(Gemini B4 mitigation)* — rejected. Bursty iteration produces near-duplicates by design, and that iteration history is exactly what `signal_attention` needs to weight retroactively once vision evaluator ships. The right move is to monitor cosine distribution and prioritize the evaluator if narrowing is observed; not to pre-filter the data.

---

## Open questions parked for MOE council

Not for Gemini, not for Claude Code. Council scope:

- **Continuous in-session consolidation.** Bridge does session-end consolidation today. Is that sufficient for v2, or does the substrate need an in-session compressor between Comfy-Cozy and Moneta?
- **Cross-hardware vector stability.** When the bridge ships beyond Joe's workstation, semantic checksumming or deterministic-inference pinning becomes load-bearing. Out of scope for v1; on the roadmap for fleet deployment.

---

## Apply order

```
1.  Edit 5 first  — opening framing (sets the mental model for the rest)
2.  Edit 1        — Phase 1 CPU enforcement (smallest, lowest risk)
3.  Edit 4        — telemetry gate language (no code, just sequencing)
4.  Edit 2        — Point 1 ADVISORY_PRIORS (highest-priority code change)
5.  Edit 3        — Point 1 shadow scaffolding (largest code addition)
```

Edit 5 first because it changes how the rest of the doc reads. Edit 3 last because it's the largest delta and benefits from the rest being settled first.

---

*Five edits. No redraft. Mission doc shape preserved. Phase 1 unblocked once Edit 5 lands.*
