# Strategic Review Brief — Mike Gold Re-Engage

**For:** Joe Ibrahim (Creative Director / Architect) & Claude (Implementation Partner)
**From:** Gemini (Deep Think Mode)
**Date:** May 02, 2026

This brief establishes the strategy to clear the Mike Gold re-engagement gate within the ~28 calendar days before the Jun 02 window opens. It derives priority from hard architectural dependencies rather than narrative tiering, explicitly resolving the entanglement Claude identified between the 01 Demo and the 5.1 exploration.

The constraints defined in §2 and §3—including LIVRPS reversibility, patent boundaries, and Comfy-Cozy's current un-unified adjacent state—are fully respected and incorporated into the pathing logic.

---

## 6.1 First-Principles Priority Order

Mike Gold's roadmap (Tier 1/2/3) reflects the sequence in which an investor consumes proof, not the engineering dependency graph required to build it. The 9 active items (treating 10–13 as locked constraints and 14 as ambient) are prioritized here based on physical blockers and structural necessity.

1. **05 · Cozy operates against codeless-typed cognitive prims** *(Mike's Probe 02)*
   *Reasoning:* This is the absolute physical prerequisite for the visual proof. Harlo/Moneta schemas must be migrated upstream, and Cozy must consume them downstream. If this is skipped, the demo (01) records anonymous attribute soup, explicitly failing Mike's credential check and undermining the "substrate-over-product" thesis.
2. **01 · Cozy × Moneta working demo — the "it remembered" moment** *(Mike's Tier 1)*
   *Reasoning:* The visceral anchor of the gate. Positioned immediately after 05 because it relies on the typed prims for visual integrity. Integration friction is high-risk; secure the existential proof of cross-session memory before perfecting the measurement layer.
3. **02 · Token economics telemetry instrumented inside Cozy** *(Mike's Tier 1)*
   *Reasoning:* Telemetry is low-level instrumentation. It must be built *before* the benchmark framework is finalized so the scaffolding can natively ingest token/cost logs without requiring a refactor.
4. **03 · Benchmark scaffolding published on Cozy repo** *(Mike's Tier 1)*
   *Reasoning:* The structural container required for Gate 01. Needs heavy calendar time to bake and standardize, but strictly depends on the telemetry hooks existing first.
5. **04 · Eval set — VFX-flavored, real workloads** *(Mike's Tier 2)*
   *Reasoning:* Populates the scaffolding. Elevated above its Tier 2 status because a benchmark framework running synthetic prompts reads as a toy. Real VFX workloads are required to prove domain applicability.
6. **09 · Cozy README and release-note narrative** *(Mike's Tier 3)*
   *Reasoning:* Synthesizes the telemetry and demo into the substrate-over-product positioning. Dictates how Mike reads the metrics.
7. **08 · Demo recording packaged for the re-engage email** *(Mike's Tier 3)*
   *Reasoning:* The terminal assembly step inside the Jun 02 window.
8. **06 · Thin cloud surface** *(Mike's Tier 2)*
   *Reasoning:* A distribution test. High-value if time permits, but structurally expendable.
9. **07 · EmbeddingGemma spike** *(Mike's Tier 2)*
   *Reasoning:* Pure intel. A context-switching distraction from the gate.

---

## 6.2 Critical Path

With ~28 calendar days available—yielding roughly **~20 focused working days**—the project fractures into two parallel chains.

**Chain Alpha: The Visual Proof (Joe's Primary Track)**
*   Upstream Harlo/Moneta Codeless Migration *(est. 4–6 days)* → **Blocks 05**
*   **05** Cozy schema consumption *(est. 1–2 days)* → **Blocks 01**
*   **01** Cozy × Moneta minimal demo integration *(est. 4–6 days)*
*   *Total Focus:* ~9–14 days. Slack exists, but the cross-repo boundary is volatile.

**Chain Beta: The Measurement Engine (Claude's Primary Track)**
*   **02** Telemetry instrumentation *(est. 2–3 days)* → **Blocks 03**
*   **03** Benchmark scaffolding *(est. 7–10 days)* → **Blocks 04**
*   **04** VFX Eval sets definitions/runs *(est. 5–8 days)*
*   *Total Focus:* ~14–21 days. **Zero Slack.** Chain Beta consumes the entire calendar and is the true critical path.

---

## 6.3 Parallelization Map

Parallel execution is mandatory, but you must distinguish between true parallelization (distinct code surfaces) and context thrashing (overlapping surfaces).

**True Parallel (Safe):**
*   **Chain Alpha vs. Chain Beta:** Claude can instrument token telemetry (02) and build scaffolding (03) in the Cozy repo while Joe handles the upstream Pydantic/USD schema migrations in Harlo/Moneta. They operate on entirely different execution layers.
*   **Workload Definition vs. Scaffolding:** Joe can define the prompt/state pairs for the VFX workloads (04) in plain text or JSON while Claude builds the harness (03) that will eventually execute them.

**Theatrical Parallel (The Thrash Traps):**
*   **Item 01 (Demo Integration) vs. 5.1 ARCH-1/2 Exploration:** Claude's uncertainty was 100% correct. You cannot build a stable cross-session demo across the Cozy ↔ Moneta boundary if 5.1 is actively tearing down and redesigning that exact boundary. *(See §6.6 for the resolution).*
*   **Item 04 (Eval Scripting) vs. Item 03 (Scaffolding):** Claude cannot write the executable evals while the benchmark format is still thrashing. The interfaces will constantly break.

---

## 6.4 Cut List

If the ~20 available working days compress, execute these cuts defensively.

**Scenario 1: 5 Weeks Available (Max-Sufficient)**
*   All 9 items ship. The package includes the cloud link (06), providing an immediate frictionless credential check for Mike.

**Scenario 2: 4 Weeks Available / 1 Slip (The Base Case)**
*   **Cut:** 06 (Thin Cloud) and 07 (EmbeddingGemma spike).
*   **Impact:** You lose the distribution probe and minor local-retrieval intel.
*   **Mike's Read:** Unaffected. The core gate (measurement and cross-session persistence) remains satisfied.

**Scenario 3: 3 Weeks Available / 2 Slips (Min-Sufficient Emergency)**
*   **Cut:** 06 and 07.
*   **Scale down 04 (Evals):** Truncate the comprehensive suite into 1–2 "Hero" VFX tracer workloads.
*   **Scale down 03 (Scaffolding):** Strip cross-repo execution requirements. Make it standardize *only* Cozy's local output, dropping the multi-repo harmonization until July.
*   **Mike's Read:** You lose the appearance of a vast, harmonized ecosystem, but you preserve the *infrastructure* of measurement and the core demo. This fulfills the literal requirements of Gate 01 and holds the window open.

---

## 6.5 Risk Surface (Top 5 Items)

| Item | Slips 1 Week | Fails for Jun 02 | Impact on Mike's Read |
| :--- | :--- | :--- | :--- |
| **05 (Schemas)** | Compresses Item 01 into a dangerous sprint. | Demo records anonymous USD attribute soup. | **Fails credential check.** Visually invalidates the "substrate" narrative; Cozy looks like a fragile, untyped plugin. |
| **01 (Demo)** | Pushes the package to the Jun 16 boundary limit. | Cross-session memory visual proof is missing. | **Fatal.** "It remembered" was the explicit price of re-engagement. The send is aborted. |
| **02 (Telemetry)** | Forces later rewrite of scaffolding to ingest metrics. | Generation and token costs remain opaque. | Collapses the "software dissolves due to zero inference costs" thesis. Looks like a naive AI wrapper. |
| **03 (Scaffold)** | Forces immediate cut to Scenario 3 (Micro-evals). | Evaluators have no reproducible, harmonized standard. | **Violates Gate 01.** Fails the literal "measurement over invention" prerequisite. |
| **04 (Evals)** | Scaffolding ships empty or runs synthetic prompts. | Substrate lacks proof of domain complexity. | Severe credibility hit. Mike (SVA faculty) will immediately spot that the system cannot handle real visual workloads. |

---

## 6.6 Path Recommendation: Path D — "State Freeze & Air-Gapped Parallel"

**I explicitly reject Path A (sprawl) and Path B (sequential delay). I recommend Path D: a redefined, protected version of Claude's Path C.**

Claude correctly identified the fatal entanglement: *If ARCH-1/ARCH-2 decisions materially change how Cozy talks to Moneta, building item 01 before they land is wasted work.*

This fear assumes the demo must be built on the *future* architecture. It does not. First principles dictate that Mike is gating you on instrumenting *what exists today* (§1.2, §3.5). Comfy-Cozy already functions as an agent with an un-unified, adjacent cognitive substrate.

**The Strategy: Sever the Entanglement**
1. **Lock the Gate to Phase 6A:** Execute the 5.2 track strictly against the current `agent/stage/` codebase (`origin/master`).
2. **Shim the Demo:** Build the Item 01 demo using the *currently isolated* USD stage prim store. Do not attempt to wire the un-invoked `AutonomousPipeline`, and do not wait to unify the Experience Cerebellum. Wire the shortest possible path to demonstrate cross-session memory today.
3. **Air-Gap 5.1:** Run 5.1 (architecture exploration) concurrently, but purely as a design track—markdown RFCs, whiteboards, or an isolated read-only branch. It is forbidden from altering the `agent/stage/` code surface until after Jun 16.
4. **Accept Integration Debt:** If 5.1 eventually dictates a better boundary that obsoletes the Item 01 demo wiring, *let it*. Throwing away 4 days of integration code in July is a cheap, highly asymmetric price to pay for successfully passing a VC gate in June.

---

## 6.7 Unknowns and Open Questions

Joe, to unblock Claude and launch both chains safely tomorrow morning, you must resolve these decisions:

1. **Upstream Schema Bandwidth:** Chain Alpha is entirely bottlenecked by the 4–6 day Harlo/Moneta Codeless migrations. Are you personally executing this upstream, or is there external capacity? If it is you, your 5.1 architecture work must pause completely until those schemas land.
2. **The Minimal Demo Vector:** What is the precise, cheapest code path in Phase 6A that writes state to the isolated USD store and reads it back without triggering an Experience Cerebellum refactor? Define this interaction today so Claude doesn't accidentally over-engineer a Phase 7 integration.
3. **Scaffolding Scope:** Does Mike's Gate 01 expect the scaffolding to physically execute evals across *all five repos* by Jun 02, or just be standardized on Cozy with placeholders for the others? If he expects multi-repo execution, Chain Beta's 7–10 day estimate is dangerously low and must be triaged immediately.
