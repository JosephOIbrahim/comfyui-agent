# HANDOFF — Substrate Egress v1.1: Round 5 Gap Closure

**To:** Gemini Deep Think
**From:** Joe Ibrahim (Creative Director)
**Date:** May 5, 2026
**Round:** 5 of N *(R4 absorbed in full; this round closes two specific gaps before drafting)*
**Type:** Gap closure · Decision-only · No forge · No mission redraft
**Companion artifacts:**
- All prior handoffs *(R1, R2, R3, R4)* + your responses
- `MISSION_SUBSTRATE_EGRESS_V1_EDITS.md` *(original 5 edits — now joined by Edits 6 and 7)*
- Mike Gold strategic review *(Path D verdict — markdown RFC scope until Jun 16; Claude Code execution thereafter)*

---

## 0 · Why R5 exists

R4 absorbed in full. Architecture verified across operational frames. Spine refined to seven axioms across three populated frames *(DS-3, IT-2, CT-2, CogSci-0)*.

**Two gaps remain before drafting can proceed faithfully.** Your R4 response surfaced them but did not close them — they were drafting-stage details rather than architectural questions:

1. The exact API contract for cross-store tick exposure *(R4 Q1)*
2. The default compression mechanism for Axiom 7 *(R4 Q4)*

R4 cleared us to proceed. R5 tightens these last two specifics so the v1.1 RFC is **precise enough to feed directly into Claude Code execution after Jun 16**.

**This is precision-closure, not adversarial pressure-testing.** R3 verified direction. R4 verified system-level integrity. R5 forces specific decisions on the two remaining open calls — derived from first principles, not preference.

The deliverable shape changes here. R3 and R4 produced architectural verdicts. R5 produces **implementation contracts** — the kind of specifications that survive translation from markdown RFC to executable code without ambiguity.

This is decision-only work. No code changes. No mission redraft.

---

## 1 · Hard scope rules *(unchanged)*

- **Moneta is frozen.** No surgery proposals.
- **LIVRPS State Spine is immutable AND established.** Tick semantics belong to the State Spine. R5 may surface concerns about whether the existing tick infrastructure satisfies the contract — but the surface is *"use it, expose a wrapper, or fall back to an external primitive."* Never *"modify the State Spine."*
- **MOE routing already phase-gates tools.** No Capability Registry refactor.
- **Patent-claimed subsystems out of scope.** State Spine, FORESIGHT, Lossless Signal — surface, don't propose.
- **Path D constraint active.** Markdown RFC only until Jun 16; Claude Code execution thereafter.
- **Four-frame discipline** *(IT · CT · CogSci · DS)*. Cognitive science currently unpopulated; available if a derivation genuinely needs it.

---

## 2 · Open calls under derivation

### Open call 1 — LIVRPS tick API contract

**Context.** R4 Q1 specified the cross-store tick must be a strictly monotonic, non-resetting `uint64` counter, decoupled from deque eviction logic. The repair vector named `get_absolute_tick()` as the API surface. **The contract — name, return type, call-site signature, and minimum behavior guarantees — has not been derived from first principles.** Joe maps the contract to existing or new LIVRPS surface only after the contract is precise.

**Subordinate question.** What if LIVRPS does not naturally expose a non-resetting counter — i.e., the State Spine's monotonic indexing is structurally bound to a queue that recycles? Then exposing it requires modification *(forbidden)*. Need a fallback derived from first principles that does not violate immutability.

### Open call 2 — Compression mechanism for Axiom 7

**Context.** R4 Q4 added Axiom 7 *(Semantic Compression Boundary)* in the Information Theory frame: bridge performs lossy compression prior to vector embedding. R4 named three candidate mechanisms — terminal state extraction, intent-topology deduplication, LLM summarization — but did not select one.

**Subordinate question.** What's the upgrade path from v1.1 default to higher-fidelity compression? Specifically — is the Phase 2.5 boundary *(deferred optimizer prior + Point 2 signal_attention)* the right boundary for the compression upgrade, or is compression upgrade path orthogonal?

---

## 3 · The questions

Two questions. Each demands a derivation, a verdict, a counter-argument, and an implementation contract precise enough to feed both the markdown RFC and the Claude Code execution spec.

---

### ▸ Q1 — LIVRPS tick API contract (from first principles)

**Required derivation.** Six properties to derive from the four-frame discipline *(distributed systems primarily; control theory where stability matters)*:

- **Read-only?** *(by the immutability rule — and how is this enforced at the type-system level, not just by convention?)*
- **Return type?** Why `uint64` specifically — what's the wrap horizon under realistic session-count assumptions over the v1.1 lifetime? Justify against `uint128`, against a structured `(epoch, tick)` tuple, against simpler primitives.
- **Strict monotonicity?** What guarantees enforce *"strictly increasing across consecutive reads"* — and what's the failure semantics if monotonicity is violated *(e.g., concurrent State Spine writes; replays after crash recovery)*?
- **Idempotency?** Should two reads at the same agent step return the same value, or different values? Derive both options' consequences for Q3b's fsync barrier and for Held-Out Replay reproducibility.
- **Bound-state coupling invariant.** The tick must be decoupled from deque eviction. Specify the invariant that enforces decoupling *(e.g., "tick is incremented on every state mutation, never on queue-position-recycle")* — and specify how the invariant is enforced structurally, not by convention.
- **API name.** Propose. Justify against the four-frame discipline. *(`get_absolute_tick()` is gestural; rename if first principles dictate a different name.)*

**Fallback derivation.** If LIVRPS does not naturally expose a non-resetting counter, propose the minimum-violation alternative that satisfies cross-store causal ordering without modifying the State Spine. Three candidates worth deriving:

- An external **Hybrid Logical Clock** synchronized at every State Spine write boundary *(but read independently of the State Spine)*
- A **monotonic wall-clock primitive** with logical tiebreaker *(e.g., `(time.monotonic_ns(), process_seq)`)*, ordered consistently with State Spine writes via fsync barriers
- A **tick proxy** maintained by the bridge itself, observing State Spine writes via filesystem mtime or equivalent

For each: what does it preserve? What does it lose vs. native LIVRPS tick reuse? Which one survives Q3b's fsync barrier interaction cleanly?

**Verdict.** Pick one — native LIVRPS reuse OR a specific fallback — as the v1.1 default. State explicitly which assumption about LIVRPS internals your verdict requires Joe to confirm.

**Counter-argument.** Strongest case for *no logical clock at all* — wall-clock ordering only. What does that lose, and is the loss acceptable at v1.1 scale?

**Implementation contract.** Specify:

- Function or property declaration *(name, parameters, return type)*
- Type-system enforcement of read-only *(Python `@property` without setter, TypeScript `readonly`, etc.)*
- Behavioral guarantees as documentation language *(precise enough to copy-paste into the v1.1 RFC and into the Claude Code spec without further translation)*
- Failure modes that should raise vs. failure modes that should silently degrade

---

### ▸ Q2 — Compression mechanism for Axiom 7 (from first principles)

**Required derivation per candidate.** For each of the three R4-named candidates *(terminal state extraction · intent-topology deduplication · LLM summarization)*, derive:

- **Compression ratio** *(WAL bytes → consolidated bytes, order of magnitude)*
- **Information preserved** *(what survives — final outcomes? intent labels? full reasoning chain?)*
- **Information lost** *(what's gone — and is the loss load-bearing for cross-session retrieval?)*
- **Determinism** *(reproducible across runs given the same input?)*
- **Cost per session-end** *(latency, model calls, dollars at v1.1 scale)*
- **Failure mode under unavailability** *(LLM summarization only — what's the fallback when the model call fails or times out?)*
- **Phase 2.5 compatibility** — critical: does the compression remove iteration history that `signal_attention` will need to weight retroactively once vision evaluator ships? If yes, the candidate is incompatible with the deferred Phase 2.5 work and must be rejected regardless of other strengths.

**Verdict.** Pick one as the v1.1 default. Justify against the IT frame *(channel capacity / signal-to-noise)* and the DS frame *(consolidation pipeline failure semantics)*.

**Counter-argument.** Strongest case for picking a *different* candidate, or for picking *none* and deferring all compression to Phase 2.5 entirely *(letting Moneta ingest raw consolidated `outcomes.jsonl` for v1.1 with no semantic compression)*.

**Upgrade path.** Cleanest progression from v1.1 default to higher-fidelity compression. Is the Phase 2.5 boundary *(optimizer prior + Point 2 signal_attention)* the right place for the compression upgrade, or is compression upgrade path orthogonal — i.e., independent of the vision evaluator and signal_attention work?

**Implementation contract.** Specify:

- Bridge-side function signature *(name, parameters, return type)*
- Input schema — what does it consume from the WAL *(raw JSONL? structured outcome records? something else?)*
- Output schema — what does it produce for Moneta ingest *(the existing `outcomes.jsonl` shape? a new compressed form?)*
- Error handling — what happens when compression fails partially? Does the bridge fall back to a less-aggressive mechanism, or does it tombstone the session?
- Performance budget — maximum allowed wall-clock time at session-end before the bridge logs a slow-consolidation warning

Precise enough to feed directly into Claude Code execution.

---

## 4 · Synthesis-killer call

End your response with a single paragraph addressing two distinct questions:

1. **Which derivation has higher implementation risk if drafted wrong?** *(I.e., which one, if your verdict turns out to need revision after drafting begins, costs more to repair?)*
2. **Do the markdown RFC and the Claude Code execution spec converge or diverge in shape across these two derivations?** If converge, they should be co-drafted to avoid drift. If diverge, the RFC ships first *(Path D)* and the execution spec follows *(post-Jun 16)*. Which is it, and why?

This question is workflow-relevant. Path D allows the markdown RFC now and forbids code-touching execution work until Jun 16. If the two artifacts converge in shape, sequential drafting risks drift between them; if they diverge, sequential drafting is the right shape.

---

## 5 · What this handoff does NOT do

- Does not relitigate R1–R4 absorbs *(those are decided)*
- Does not propose code changes *(Path D forbids until Jun 16; this round produces specifications, not implementations)*
- Does not modify the LIVRPS State Spine, Moneta, or any patent-claimed subsystem
- Does not pre-empt the MOE council vote on ARCH-1 / ARCH-2 / ARCH-5
- Does not redraft the mission

---

## 6 · Marathon markers

```
Mile 1 ─  You read R3, R4, R4 response, this handoff
Mile 2 ─  You return two derivations + synthesis-killer paragraph
Mile 3 ─  Joe synthesizes — gaps closed, drafting unblocked
Mile 4 ─  V1.1 markdown RFC drafted from R5-closed spine
Mile 5 ─  Claude Code execution spec drafted (sequenced or co-drafted per Q4 outcome)
Mile 6 ─  Joe review + commit
─────  R5 COMPLETE  ─────
```

---

## 7 · Cross-references

- R1, R2, R3, R4 handoffs + responses
- `MISSION_SUBSTRATE_EGRESS_V1_EDITS.md` *(original 5 edits — joined by Edits 6 and 7)*
- Mike Gold strategic review · Path D — markdown RFC only until Jun 16
- Test baseline: Comfy-Cozy 2717 passing · `comfy-moneta-bridge` v0.1.0 · 49 tests green
- Repos:
  - `github.com/JosephOIbrahim/Comfy-Cozy`
  - `github.com/JosephOIbrahim/Moneta`
  - `github.com/JosephOIbrahim/comfy-moneta-bridge`

---

*This is precision-closure. The architecture is verified. Two specific calls remain. Derive both from first principles; return implementation contracts precise enough to feed Claude Code execution without translation.*
