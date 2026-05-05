# HANDOFF — Substrate Egress v1.1: Round 6 Closure-of-Closure

**To:** Gemini Deep Think
**From:** Joe Ibrahim (Creative Director)
**Date:** May 5, 2026
**Round:** 6 of N *(R5 absorbed in full; this round closes the two action items R5's verdict surfaced)*
**Type:** Verification-protocol + drafting-discipline derivation · Decision-only · No forge · No mission redraft
**Companion artifacts:**
- All prior handoffs *(R1, R2, R3, R4, R5)* + your responses
- `MISSION_SUBSTRATE_EGRESS_V1_EDITS.md` *(original 5 edits — joined by Edits 6 and 7 from R3+R4 absorbs)*
- Mike Gold strategic review *(Path D verdict — markdown RFC scope until Jun 16; Claude Code execution thereafter)*

---

## 0 · Why R6 exists

R5 closed the two open calls and produced clean implementation contracts:
- **Q1:** `causal_sequence_id` property *(primary)* with Loop-Scoped Monotonic Snapshot fallback
- **Q2:** `deduplicate_intent_topologies()` mechanism with `thrash_count` preservation

R5's verdict surfaced two action items that drafting can't proceed without. Both are first-principles-answerable, and both produce executable specifications:

1. **Verification protocol** — the primary Q1 verdict assumes LIVRPS maintains a `total_transitions` integer independent of ring-buffer eviction. The assumption needs verification against actual LIVRPS source. **What's the protocol that determines whether the primary verdict or the fallback applies?** This isn't asking you to verify Joe's code — it's asking you to derive the protocol Joe *(or Claude Code post-Jun 16)* executes.

2. **Drafting discipline** — your R5 synthesis-killer said RFC and Claude Code execution spec *"strictly diverge."* That's a verdict. **What's the discipline that distinguishes RFC content from execution-spec content during drafting** — so the verdict survives contact with actual writing instead of degrading into vibes?

Both are precision-of-precision questions. Same shape as R5: derivation, verdict, counter-argument, implementation contract precise enough to act on.

This is decision-only work. No code changes. No mission redraft.

---

## 1 · Hard scope rules *(unchanged)*

- **Moneta is frozen.** No surgery proposals.
- **LIVRPS State Spine is immutable AND established.** R6 derives a verification protocol; it does not propose State Spine changes.
- **MOE routing already phase-gates tools.** No Capability Registry refactor.
- **Patent-claimed subsystems out of scope.** State Spine, FORESIGHT, Lossless Signal — surface, don't propose.
- **Path D constraint active.** Markdown RFC only until Jun 16; Claude Code execution thereafter. **Verification protocol must be runnable as a Claude Code task post-Jun 16, OR as a manual code-read by Joe before — the protocol shape must support both.**
- **Four-frame discipline** *(IT · CT · CogSci · DS)*. Cognitive science currently unpopulated; available if a derivation needs it.

---

## 2 · Open calls under derivation

### Open call 1 — Verification protocol for the `total_transitions` assumption

**Context.** R5 Q1 made the primary verdict *(`causal_sequence_id` property reusing native LIVRPS state)* conditional on an assumption: *"the LIVRPS state machine inherently maintains a `total_transitions` integer that operates independently of its ring-buffer `maxlen` eviction."* If false, fallback to Loop-Scoped Monotonic Snapshot.

**Gap.** Joe needs to confirm or refute the assumption against the actual LIVRPS source. **The verification protocol — what to look for, what tests definitively distinguish primary from fallback, what edge cases matter — has not been derived from first principles.**

**Subordinate question.** What if LIVRPS partial-cases the assumption — e.g., has a counter that's independent of eviction *but* resets on session boundaries, or has the counter *but* it's incremented only on specific event types rather than all state transitions? Does the protocol produce a third verdict *(partial — refine the contract)*, or does any partial-case automatically fall back?

### Open call 2 — Drafting discipline distinguishing RFC content from execution-spec content

**Context.** R5's synthesis-killer said the markdown RFC *(architectural physics — invariants, boundary guarantees, failure semantics)* and the Claude Code execution spec *(tactical implementation — Python AST, regex, type imports)* strictly diverge in shape. Sequential drafting is correct; co-drafting risks polluting the RFC with implementation trivia.

**Gap.** *"Strict divergence"* is a verdict. **The discipline that enforces it during drafting — the test or rule a piece of content must pass to belong in the RFC vs the execution spec — has not been derived.** Without the discipline, *"don't pollute the RFC"* is aspirational, not enforceable.

**Subordinate question.** Are there spec-shaped artifacts that legitimately belong in *both* documents *(e.g., the `causal_sequence_id` API name)*, and if so, what's the rule for distinguishing duplication-as-cohesion from duplication-as-leakage?

---

## 3 · The questions

Two questions. Same response shape as R5:

- **Required derivation** *(per-property first-principles analysis)*
- **Verdict** *(specific protocol or specific discipline)*
- **Counter-argument** *(strongest case against your verdict — including failure modes of the protocol/discipline itself)*
- **Implementation contract** *(precise enough to act on)*

---

### ▸ Q1 — Verification protocol for the `total_transitions` assumption (from first principles)

**Required derivation.** Five properties to derive from the DS frame:

- **What evidence in LIVRPS source proves the primary verdict correct?** Specifically: what code structures would constitute proof that a `total_transitions`-equivalent integer exists, increments on every state mutation, and is decoupled from ring-buffer modulo arithmetic? Name the structures *(class attribute? module-level counter? generator function? something else?)*.
- **What evidence proves the fallback required?** Specifically: what code structures would constitute proof that no such counter exists, OR that the counter exists but is structurally tied to bounded-queue position?
- **What test sequence definitively distinguishes primary from fallback?** Concrete steps — e.g., *"step 1: locate the State Spine class; step 2: enumerate integer-typed instance attributes; step 3: for each, run trace X to verify behavior under deque eviction."* The protocol must be deterministic — running it twice on the same code must yield the same verdict.
- **Edge cases: partial-case verdicts.** If LIVRPS has a counter that *partially* satisfies the assumption — independent of eviction but resets on session boundaries; or independent and non-resetting but incremented only on a subset of state transitions — what verdict does the protocol return? *(Primary with caveat? Fallback? Refined contract?)* Show the decision tree.
- **Failure modes of the protocol itself.** Can the protocol return a false positive *(approves the primary verdict when fallback is required)*? Can it return a false negative *(forces fallback when primary would have been fine)*? What checks catch these?

**Verdict.** The verification protocol — a specific, deterministic sequence of code-reading steps + tests that returns one of {`PRIMARY_OK`, `FALLBACK_REQUIRED`, `PARTIAL_REFINE_CONTRACT`}.

**Counter-argument.** Strongest case for *not* deriving a protocol — i.e., for treating the LIVRPS assumption as *"verify by inspection, no formal protocol needed."* What does that lose at v1.1 scale, and is the loss acceptable?

**Implementation contract.** The protocol must be runnable in two contexts:
- **Manual code-read by Joe** *(Path D pre-Jun 16)* — protocol shape must be a checklist a human can execute by reading source
- **Claude Code execution task** *(Path D post-Jun 16)* — protocol shape must be expressible as a deterministic test that returns a verdict

Specify the protocol structure precisely enough to feed into both modes without translation.

---

### ▸ Q2 — Drafting discipline distinguishing RFC content from execution-spec content (from first principles)

**Required derivation.** Five properties to derive from the IT frame *(information content / abstraction level)* and the DS frame *(versioning + drift semantics)*:

- **The RFC content type signature.** What specific properties does a piece of content have if it belongs in the markdown RFC? *(Invariant statements? Boundary guarantees? Failure-mode classifications? Explicit anti-properties — what RFC content does NOT have?)* Be specific enough that a borderline case can be classified by checking the signature.
- **The execution-spec content type signature.** Same, for the execution spec. What properties define it?
- **The discipline test.** For an arbitrary candidate piece of content, what's the test that classifies it RFC / execution-spec / both / neither? The test must be applicable during drafting *(real-time)*, not only as a post-hoc audit.
- **The dual-membership rule.** Are there artifacts that legitimately belong in *both* documents *(e.g., the `causal_sequence_id` API name appears in the RFC as architectural commitment AND in the execution spec as concrete declaration)*? If yes, what's the rule for distinguishing duplication-as-cohesion *(intentional, both documents reference the same canonical form)* from duplication-as-leakage *(accidental, the documents drift)*?
- **Failure modes of the discipline.** Can the discipline produce false positives *(rejects content that should be in the RFC because it superficially looks tactical)*? False negatives *(accepts implementation trivia because it superficially looks architectural)*? What guards catch these?

**Verdict.** The discipline — a specific classification rule + dual-membership rule + drift-detection rule.

**Counter-argument.** Strongest case for *not* enforcing a discipline — i.e., for treating *"RFC vs execution spec"* as a stylistic preference rather than a structural rule. What does that lose under Path D, and what's the failure mode at scale *(when the RFC is being read by the MOE council 3 months from now)*?

**Implementation contract.** Specify:
- The classification rule as a procedure *(input: a piece of content; output: RFC / execution-spec / both / neither)*
- The dual-membership rule as a procedure *(input: a candidate dual-membership artifact; output: ALLOWED with canonical-form-source / FORBIDDEN with reason)*
- The drift-detection rule as a procedure *(input: the RFC and execution spec at a given point in time; output: a list of dual-membership artifacts where the documents have drifted)*
- Examples spanning at least three borderline cases — RFC-only, execution-spec-only, dual-membership

Precise enough that Joe can apply the discipline during RFC drafting *(now)* and during execution-spec drafting *(post-Jun 16)* without ambiguity.

---

## 4 · Synthesis-killer call

End your response with a single paragraph addressing two questions:

1. **Which derivation has higher leverage on v1.1 RFC quality?** The verification protocol *(determines whether Edit 5 can be cleanly drafted at all)*, or the drafting discipline *(determines whether the entire RFC stays clean across all seven edits)*?
2. **Are there second-order action items either derivation surfaces?** Specifically: does the verification protocol's verdict shape *(`PRIMARY_OK` / `FALLBACK_REQUIRED` / `PARTIAL_REFINE_CONTRACT`)* require any additional spine-level handling we haven't captured? Does the drafting discipline reveal RFC-quality-relevant concerns *(e.g., the existing 7 edits' content already drifts across the RFC/execution-spec boundary, or specific edits are higher-risk than others under the discipline)*?

If the answer to #2 is *"no, both derivations are clean closure"* — say so explicitly, and we're done with handoff rounds. **R6 is the planned final round.**

---

## 5 · What this handoff does NOT do

- Does not relitigate R1–R5 absorbs *(those are decided)*
- Does not propose code changes *(Path D forbids until Jun 16; this round produces specifications, not implementations)*
- Does not modify the LIVRPS State Spine, Moneta, or any patent-claimed subsystem
- Does not attempt to verify Joe's actual LIVRPS source *(only Joe or Claude Code can do that — R6 produces the protocol, not the verdict)*
- Does not pre-empt the MOE council vote on ARCH-1 / ARCH-2 / ARCH-5
- Does not redraft the mission

---

## 6 · Marathon markers

```
Mile 1 ─  You read R5, R5 response, this handoff
Mile 2 ─  You return two derivations + synthesis-killer paragraph
Mile 3 ─  Joe synthesizes — protocol locked, discipline locked, drafting unblocked
Mile 4 ─  Joe runs verification protocol against LIVRPS source (or queues for Claude Code post-Jun 16)
Mile 5 ─  V1.1 markdown RFC drafted under the discipline
Mile 6 ─  Claude Code execution spec drafted post-Jun 16
Mile 7 ─  Joe review + commit
─────  R6 COMPLETE — planned final round  ─────
```

---

## 7 · Cross-references

- R1, R2, R3, R4, R5 handoffs + responses
- `MISSION_SUBSTRATE_EGRESS_V1_EDITS.md` *(7 edits total post-R4 absorb)*
- Mike Gold strategic review · Path D — markdown RFC only until Jun 16
- Test baseline: Comfy-Cozy 2717 passing · `comfy-moneta-bridge` v0.1.0 · 49 tests green
- Repos:
  - `github.com/JosephOIbrahim/Comfy-Cozy`
  - `github.com/JosephOIbrahim/Moneta`
  - `github.com/JosephOIbrahim/comfy-moneta-bridge`

---

*This is closure-of-closure. R5 closed the architectural gaps. R6 closes the gaps R5's verdict opened — verification protocol so the conditional verdict resolves cleanly, drafting discipline so the strict-divergence call survives contact with actual writing. Last planned round before drafting begins.*
