# ARCH-2 — Moneta v1.3 Evolution: State-Machine Layer Absorption

**Type:** MOE Council Brief — DECISION ONLY, NO CODE
**Status:** PENDING council session
**Owner:** Joe Ibrahim (decision authority)
**Roles in council:** ARCHITECT (design), CRUCIBLE (adversarial), FORGE (cost), SCOUT (current Moneta surface)

---

## 0. The Question

**Once Comfy-Cozy needs real memory (post-Tier-1-2 hardening), should Moneta v1.3 absorb Harlo's state-machine pattern (thrashing, velocity, token budget, allostasis) — or should Moneta stay focused on outcome memory only?**

Three possible council outcomes:

- 🟢 **GREEN** — Approve absorption. Moneta v1.3 includes a state-machine layer scoped for autonomous-agent semantics.
- 🟡 **YELLOW** — Approve a thin telemetry component but not full state-machine absorption.
- 🔴 **RED** — Reject. Moneta stays outcome-memory-only. State-machine concerns get a separate substrate or never get built.

---

## 1. Background

### 1.1 What Moneta currently does

From userMemories + 4.27 closeout:

- USD-native cognitive substrate
- LIVRPS composition with Safety (S) as strongest opinion (patent-relevant)
- AttentionLog with biomimetic decay (the v1.2.0-rc2 work)
- "She who warns" — memory as advisory, not just storage
- Cross-session persistence via OpenUSD + WAL SQLite
- v1.2.0-rc2 shipped 4.28 (free-threading guard added)

**What Moneta does NOT currently do:**

- AI-system-health telemetry
- Real-time state machines
- Decision gating on velocity/thrashing/budget signals
- Working-condition assessment for autonomous loops

### 1.2 What Harlo's state-machine pattern does

From the SYNAPSE_HARLO_BRIDGE_INTEGRATION v1.1 spec §3.2:

> *Dynamic state machines (momentum, burnout, energy, burst, allostasis) — operating on AI-system-health telemetry, not human health (Semantic Inversion)*

Harlo in `production_dcc` profile reinterprets the state machines as:

| State machine | Originally | DCC profile reinterpretation |
|---|---|---|
| Burnout | Human exhaustion | AI Thrashing |
| Momentum | Human flow state | Execution Velocity |
| Energy | Human capacity | Token Budget |
| Allostasis | Human adaptation | Domain Volatility |

The pattern is: **continuous telemetry → state-machine inference → decision gates.**

That pattern is **portable to autonomous agents** with one more semantic inversion.

### 1.3 Why this matters for Comfy-Cozy

Once Tier 1-2 hardens and Comfy-Cozy starts running real autoresearch loops (Path A or Path B), the agent will need to answer questions like:

- *Am I thrashing on this iteration? (same workflow attempted N times with no quality improvement)*
- *Is my token budget dropping below the threshold for committing to another iteration?*
- *Is the domain shifting under me? (model selection, prompt distribution drifting)*
- *What is my execution velocity right now — should I commit a longer-running plan or short ones?*

These are not outcome-memory questions. **AttentionLog cannot answer them.** They need state-machine inference over real-time telemetry.

The choice is: where does that state-machine layer live?

---

## 2. Three options

### 2.1 Option A — Full Moneta v1.3 absorption (🟢)

Moneta gains a `Moneta.Telemetry` module with autonomous-agent state machines:

| State machine | Autonomous-agent semantics |
|---|---|
| Thrashing | Same workflow attempted N times without quality score improvement |
| Velocity | Successful iterations per unit wall-clock time |
| Token Budget | Tokens consumed vs budget cap for current run |
| Domain Volatility | Distribution shift in prompt categories or model selection |

The state machines write their state into Moneta's USD substrate (under a new `Telemetry` prim hierarchy). Decisions consult both AttentionLog (outcome memory) and Telemetry (working-state) before committing to next actions.

**Pros:**
- Single substrate, single source of truth
- LIVRPS composition extends naturally to telemetry layer
- Patent thesis stays unified

**Cons:**
- Moneta becomes more than a memory substrate (scope creep)
- v1.3 surface area grows significantly
- Requires absorbing Harlo's state-machine implementation patterns (legal/IP question if Harlo is being separately patent-protected)

### 2.2 Option B — Thin telemetry component only (🟡)

Moneta gains a small `Moneta.HealthGauge` surface — not full state machines, but four scalar signals (thrashing_score, velocity, token_remaining, volatility_index) computed from rolling windows. Decision gates remain in Comfy-Cozy's runtime, not in Moneta.

**Pros:**
- Smaller surface area
- Doesn't absorb Harlo's full pattern (cleaner IP separation)
- Faster to implement
- Easy to remove if unused

**Cons:**
- Moneta still doesn't make decisions; consumers do
- Two-place logic (Moneta computes signals, Comfy-Cozy interprets them) — coordination complexity
- May need to grow into Option A anyway later

### 2.3 Option C — Reject; outcome memory only (🔴)

Moneta stays as v1.2.0 + future memory-shaped extensions. State-machine concerns live in Comfy-Cozy's runtime directly, or get a separate substrate.

**Pros:**
- Moneta stays focused
- Clean IP boundary
- v1.3 is a different conversation entirely

**Cons:**
- Three-isolated-experience-stores problem (already flagged in userMemories) gets worse — now there's a fourth source of state
- State-machine logic ends up in Comfy-Cozy's runtime, where it's not USD-resolvable and not patent-claimable
- Telemetry stays a runtime concern, not a substrate concern

---

## 3. The IP question

This is the part of the decision that gets glossed over. **Both Moneta and Harlo have patent posture files.** Moneta's central novel claim, per userMemories, is *"one LIVRPS composition engine serves both state resolution and prediction resolution simultaneously."*

Adding a state-machine layer to Moneta is consistent with that claim — state resolution naturally includes telemetry-derived state. **But it must be filed under Moneta's claim structure, not Harlo's.**

If Option A is chosen:
- Patent CIP filing on Moneta to extend claims to telemetry-derived state
- Explicit non-overlap with Harlo's state-machine claims (Harlo's claims are scoped to AI-system-health-as-coworker-input; Moneta's would be scoped to autonomous-agent-self-regulation)
- Both products stay defensible

If Option B is chosen:
- No CIP filing needed; telemetry component is a thin wrapper, not a claim
- Harlo's state-machine claims stay intact and Moneta-side stays unburdened

**This question requires patent counsel review before Option A is committed.** The brief surfaces it; the council does not resolve it.

---

## 4. CRUCIBLE — Adversarial Pressure

### 4.1 "You're absorbing Harlo through the back door"

The strongest adversarial framing. The previous chat ruled out absorbing Harlo into Moneta. Adding Moneta v1.3 with Harlo's state-machine pattern is doing exactly that, just renamed.

**Counter:** Absorbing the *pattern* is not absorbing the *substrate*. Harlo's full surface area (Hydra Delegate registry, Rust Hippocampus, Elenchus verification, 33 inviolable rules) is not in scope for Option A. Only the state-machine pattern is, and it's being re-implemented for autonomous-agent semantics, not copy-pasted.

**Counter-counter:** The state-machine pattern is the most patent-relevant part of Harlo. If Moneta absorbs it, the IP partition between the two products gets blurry.

**Resolution:** Patent counsel review is the gate. Option A is conditional on a clean IP partition; if counsel says the partition is not defensible, fall back to Option B.

### 4.2 "v1.3 scope is bigger than the substrate launch you just shipped"

Real concern. Moneta v1.2.0 was the AttentionLog; v1.3 with full telemetry is a major expansion.

**Counter:** Yes, but it's gated on Tier-1-2 closeout. Moneta v1.3 is not next sprint; it's after the hardening track lands.

**Counter-counter:** Scope creep on a freshly-shipped substrate is itself a risk. v1.2.0-rc2 just stabilized; v1.3 ambitions can drag the codebase into churn.

**Resolution:** Council can approve Option A as a *direction*, not a commit. The actual v1.3 implementation gets its own ARCH brief at the time of forge work, with concrete scope boundaries.

### 4.3 "Comfy-Cozy doesn't need this yet"

Maybe the most useful adversarial framing. Comfy-Cozy is not running real autoresearch loops yet. The vision evaluator gates Path B; Path A lite hasn't run. There's no concrete telemetry need *today*.

**Counter:** True. But the userMemories already flagged the three-isolated-experience-stores problem. Adding telemetry without a substrate plan creates a fourth isolated store.

**Counter-counter:** Solving the existing three-store problem might dissolve the need for telemetry-as-fourth-store entirely.

**Resolution:** ARCH-2 may be premature without resolving the three-store problem first. **Council should consider whether to defer ARCH-2 until after the three-store problem is addressed in a separate brief.**

This is a real and useful adversarial point. The brief stands, but the council may legitimately defer.

---

## 5. Trade-off Matrix

| Dimension | 🔴 Reject | 🟡 Thin component | 🟢 Full absorption |
|---|---|---|---|
| Implementation effort (post-Tier-2) | $0 | $$ | $$$ |
| Substrate coherence | Low (4 stores) | Medium (3 stores) | High (single state surface) |
| Patent surface | No change | No CIP needed | CIP path opens |
| Patent IP risk | None | Low | Medium (partition review needed) |
| Scope creep on Moneta | None | Low | High |
| Decision-gate latency | Low (runtime) | Medium | Low (substrate query) |
| Moneta v1.3 readiness for Path B | No | Partial | Yes |
| Three-isolated-stores problem | Worse | Better | Best (if implemented well) |

---

## 6. Recommendation

**🟡 YELLOW with re-decision conditions.**

Conditions:

1. Approve **Option B (thin telemetry component)** as the v1.3 starting scope
2. Defer Option A (full absorption) until two preconditions are met:
   - Three-isolated-experience-stores problem is addressed in a separate brief (this is upstream of telemetry questions)
   - Patent counsel reviews the IP partition between Moneta's potential CIP and Harlo's existing claims
3. Re-evaluate Option A at the v1.3 forge-pass entry point — by then, the three-store problem will be resolved or escalated, and patent counsel will have weighed in

This recommendation is more conservative than my ARCH-1 recommendation because:
- ARCH-1 is on the critical path for ARCH-5 patent extension
- ARCH-2 has no critical-path dependency
- Premature absorption risks IP partition trouble that's avoidable

**If 🟢 (full absorption) is preferred,** make patent counsel review the explicit gate before any forge work, not after.

**If 🔴 (reject) is preferred,** the three-isolated-stores problem becomes more urgent — telemetry-as-runtime-concern in Comfy-Cozy means the substrate layer continues to fragment.

---

## 7. What approval unlocks

If 🟢:

- Patent CIP filing scoped for Moneta telemetry layer
- Moneta v1.3 implementation roadmap draft
- Three-isolated-stores resolution becomes a precondition for v1.3 forge

If 🟡:

- Thin telemetry component scoped in v1.3 (smaller deliverable)
- Three-isolated-stores resolution can proceed independently
- Re-decision deferred to forge-pass entry point

If 🔴:

- Three-isolated-stores resolution becomes urgent (separate brief required)
- Comfy-Cozy runtime carries telemetry logic
- No patent CIP path on telemetry

---

## 8. Council vote sheet

```
ARCH-2 Decision: ____________________

[ ] 🟢 GREEN  — Full absorption. Patent counsel review gates forge work.
[ ] 🟡 YELLOW — Thin component (Option B). Re-decide at v1.3 forge-pass.
[ ] 🔴 RED    — Reject. Telemetry stays in runtime. Three-store problem
                escalated.

Notes: ___________________________________________________________

___________________________________________________________________

Signed: Joe Ibrahim                              Date: ____________
```

---

**End of ARCH-2 brief.** This brief explicitly acknowledges that the three-isolated-experience-stores problem is upstream and may need its own brief before ARCH-2 can proceed cleanly.
