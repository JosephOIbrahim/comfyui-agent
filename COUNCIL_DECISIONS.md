# Council Decisions — Inside-Out Architecture Session

**Date:** 2026-05-01
**Session type:** MOE council, decision-only (no code)
**Briefs reviewed:**
- `02_ARCH-1_INSIDE_OUT_BRIEF.md`
- `03_ARCH-2_MONETA_EVOLUTION_BRIEF.md`
- `04_ARCH-5_WORKFLOW_LIVRPS_BRIEF.md`

**Roles in council:** ARCHITECT (brief recommendations), CRUCIBLE (adversarial pressure), Joe (decision authority)

**Process:** ARCH-1 first (gates ARCH-5), ARCH-5 second, ARCH-2 last. Each brief surfaced its recommendation, CRUCIBLE pressure tested it, Joe voted.

---

## ARCH-1 — Transport: Outside-in vs Inside-out

**Question:** Should Comfy-Cozy migrate from outside-in (REST/WebSocket to ComfyUI on `localhost:8188`) to inside-out (ComfyUI custom node package, Moneta in-process, workflow graph as composition substrate)?

**Brief recommendation:** 🟢 GREEN with timing constraint.

**CRUCIBLE objections raised:**
1. Dependency conflict surface deferred to scout that doesn't run until after this vote — gate inverted
2. Patent claim may not require literal in-process execution — hybrid could preserve isomorphism at engine level
3. Test baseline (2717 passing in Phase 6A) likely needs reconstitution inside ComfyUI runtime — cost not in current estimate

**Decision: 🟢 GREEN**

**What this binds:**
- Inside-out is the architectural commitment
- Comfy-Cozy migrates to ComfyUI custom node package (sequencing per Phase 2 plan)
- Moneta consumed in-process
- Workflow graph treated as LIVRPS-composable surface
- Test infrastructure rebuild cost accepted
- Dependency conflict resolution falls to Phase 2 plan

---

## ARCH-5 — Patent Extension: Workflow Graph as LIVRPS Substrate

**Question:** Should ComfyUI's workflow graph be formally treated as a LIVRPS composition substrate, with a continuation-in-part (CIP) patent filing covering the substrate isomorphism claim?

**Brief recommendation:** 🟢 GREEN, conditional on ARCH-1 = 🟢 and patent counsel review. ARCH-1 condition met.

**CRUCIBLE objections raised:**
1. Today's vote = "pursue counsel review for filing," not "file the CIP" — reversibility window stays open
2. Prior art in ComfyUI workflow composition is unsurveyed — scout pass needs prior art sweep step added
3. Public claim filing increases architecture lock-in — outside-in had more optionality
4. Patent portfolio overhead — managing 6 active filings at one-person-with-ADHD scale

**Decision: 🟢 GREEN**

**What this binds:**
- CIP filing pursued in IP track
- Prior art sweep added to scout pass scope (Step 9 in `RUN_INSIDE_OUT_PASS.md`)
- Patent counsel review queued for next 90 days
- Architecture must preserve literal LIVRPS isomorphism in code
- If prior art sweep finds blocking adjacency: CIP claim radius narrows or filing reconsiders

---

## ARCH-2 — Substrate Evolution: Moneta v1.3 Absorbing Harlo State-Machine Pattern

**Question:** Should Moneta v1.3 absorb Harlo's state-machine pattern, bringing convergent state observability and pattern-sharing back into a substrate that was deliberately diverged from Harlo at v1.0?

**Brief recommendation:** 🟡 YELLOW — start with thin telemetry component, defer full absorption pending three-isolated-stores resolution and patent counsel review.

**CRUCIBLE objections raised:**

1. **Structural — gating conditions not met.** Three-isolated-stores brief doesn't exist yet. Patent counsel review not scheduled. Voting today commits a direction whose two prerequisites are unaddressed.
2. Divergence rationale potentially inverted by absorption — was the divergence wrong, or is the absorption wrong?
3. State machines presume a single source of truth; three isolated stores violate that precondition. Telemetry over desynchronized stores describes inconsistency rather than enforcing it.
4. Patent partition unresolved — Moneta and Harlo are separate properties; counsel hasn't reviewed boundary implications.
5. Bandwidth realism — next 90 days already commit to Tier 1-2, P0-J, scout pass, ARCH-5 CIP, vision evaluator, Path A lite.

**CRUCIBLE recommendation was 🔴 RED-DEFER** — return to council after three-isolated-stores brief lands and counsel weighs in.

**Decision: 🟢 GREEN (override of CRUCIBLE recommendation)**

**What this binds:**
- Moneta v1.3 absorbs Harlo state-machine pattern (full absorption, not thin telemetry)
- Three-isolated-stores resolution treated as solvable through ARCH-2 work, not gating prerequisite
- Three-isolated-stores brief written in parallel doc track (input: scout's `three_stores_inventory.md`, owner: Joe)
- Patent counsel review for partition runs in parallel, not gating
- Three-stores inventory added to scout pass scope (Step 10 in `RUN_INSIDE_OUT_PASS.md`)
- Risk accepted: state-machine layered over isolated stores until convergence work lands

---

## Critical Path

With all three GREEN, the path forward:

```
Now              →  Tier 1-2 hardening continues on outside-in
                    P0-J SSRF, path traversal, workflow JSON minification, async
                    (parallel track, not gated by inside-out work)

Inside-out run   →  RUN_INSIDE_OUT_PASS.md executed in Claude Code
                    Scout (10 steps incl. prior art + three-stores inventory)
                    GATE A
                    Architect (Phase 2 plan)
                    GATE B
                    Forge (Phase 2 first milestones)
                    GATE C → push

Parallel tracks  →  Three-isolated-stores brief drafted (Joe)
                    Patent counsel meeting scheduled (ARCH-5 CIP, ARCH-2 partition)
                    ARCH-2 thin component scoping begins after three-stores brief

Phase 2 ship     →  CIP filing (if counsel approved)
                    SuperDuper Panel + GRAPH mode lands inside-out
                    Cozy Shadow Graph operational
                    Moneta v1.3 absorption first milestones land
```

Wall-clock estimate from council: ~3–5 months to inside-out shipping, longer if ARCH-2 work surfaces three-stores complications. Estimate is directional, not committed.

---

## Open Questions Carried Forward

1. **Three-isolated-experience-stores resolution mechanism** — addressed within ARCH-2 work per this council; needs design before forge work on absorption begins.
2. **Patent counsel availability** — gates ARCH-5 CIP filing and ARCH-2 partition review.
3. **ComfyUI dependency conflict surface** — quantified by Phase 1 scout pass; if HARD risk surfaces, may force ARCH-1 re-decision via gate A.
4. **Phase 7 GRAPH mode actual usage** — empirical question affecting ARCH-5 enablement strength; resolved by usage data, not council.
5. **CRUCIBLE override on ARCH-2** — Joe's vote went against CRUCIBLE's RED-DEFER recommendation. Worth a retrospective after Phase 1 to see whether the override was load-bearing or whether the gating conditions should have held.

---

## Sign-off

**Decisions logged:**

```
ARCH-1: 🟢 GREEN
ARCH-5: 🟢 GREEN
ARCH-2: 🟢 GREEN  (overrides CRUCIBLE 🔴 RED-DEFER recommendation)
```

**Next action:** Pre-flight checklist per `INSIDE_OUT_RUN_PLAYBOOK.md`, then drop `RUN_INSIDE_OUT_PASS.md` into Claude Code at `G:\Comfy-Cozy`.

**Signed:** Joe Ibrahim
**Date:** 2026-05-01

---

*This document is the binding record of the council session. Phase 1 scout pass executes against these decisions, not against fresher reasoning. Re-decisions happen at human gates, not in the middle of work.*
