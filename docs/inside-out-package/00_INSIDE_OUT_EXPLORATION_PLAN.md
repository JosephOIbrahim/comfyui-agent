# Inside-Out Architecture Exploration Plan

**Package:** Comfy-Cozy × Moneta — Inside-Out Substrate Migration
**Status:** EXPLORATION — pre-council, decision-only documentation
**Created:** 2026-04-30
**Author:** Joe Ibrahim (Architect), Claude (translation)
**Supersedes:** None — first formal exploration of the inside-out question

---

## 0. What This Package Is

Six documents that prepare a MOE council decision session on whether Comfy-Cozy should migrate from its current outside-in architecture (REST/WebSocket talking to ComfyUI on `localhost:8188`) to an inside-out architecture (Comfy-Cozy as a ComfyUI custom node package, Moneta in-process, workflow graph as a LIVRPS composition substrate).

**This package contains zero implementation code.** It is decision work only, structured for the established Joe pattern: *architecture decisions in chat (this package's purpose), building tasks in Claude Code (Phase 2 if council approves).*

---

## 1. Why Now

Three signals converged this week:

1. **Comfy-Cozy v1.0 + comfy-moneta-bridge v0.1.0 shipped 4.29.** The substrate is wired. Now the substrate question is real, not theoretical.
2. **Hardening Tier 1-2 is on the board.** Once memory becomes a real product surface (post-Tier-2), the transport architecture decision compounds. Every month of outside-in adds refactor surface area.
3. **The Synapse v1.1 spec landed.** It crystallizes the inside-out pattern with a fully-articulated substrate isomorphism principle. The pattern is portable. The mapping is direct.

**The inflection point is now, before Tier 1-2 hardens the outside-in transport decisions into the codebase.**

---

## 2. The Strategic Frame

### 2.1 The patent thesis is broader than one product

The Synapse v1.1 spec calls the architectural philosophy **Substrate Isomorphism**:

> *The AI does not "use" the DCC. The AI writes to a USD stage. The artist writes to a USD stage. The DCC is simply the impartial viewport that renders their shared reality.*

That principle was developed for Houdini. **It generalizes.** Any creative tool with a composable graph substrate can host the same architecture:

- Houdini: USD stage
- ComfyUI: workflow graph
- Future tools: any DAG-based creative substrate

If substrate isomorphism is the patent thesis, then **Comfy-Cozy operating outside-in is leaving the patent claim on the table** for ComfyUI as a target environment.

### 2.2 The two products converge upstream

Comfy-Cozy and Synapse are different consumer surfaces. Their substrates (Moneta and Harlo) are architecturally diverged for good reasons. **But the inside-out architectural pattern is shared upstream.** Both products can implement it; the implementations will differ in detail, but the structural claims are the same.

This means the inside-out decision for Comfy-Cozy:
- Strengthens the cross-product patent moat
- Validates the substrate isomorphism principle outside of Houdini
- Creates a second consumer surface for the same architectural thesis

### 2.3 What inside-out unlocks for Comfy-Cozy specifically

Five concrete capabilities that outside-in cannot deliver:

1. **Zero-latency Moneta access** — the architectural reason `usd-core` was chosen for Moneta in the first place
2. **Cozy Shadow Graph** — proposed workflow nodes ghosted directly in ComfyUI's editor, user commits with a gesture (Frame F equivalent)
3. **Prepared-decision nodes** — Moneta-aware ComfyUI nodes that read pre-computed assertions at cook time (Frame B equivalent)
4. **Bidirectional substrate authoring** — the user authors workflow nodes; Moneta authors workflow nodes; ComfyUI's execution engine resolves the merged graph via opinion-style overrides
5. **Patent CIP filing surface** — workflow graph as LIVRPS composition substrate is a defensible extension of the existing claims

---

## 3. The Three Architectural Questions

The council session decides three orthogonal questions. Each has its own brief in this package.

### ARCH-1 — Outside-In vs Inside-Out

**The transport question.** Does Comfy-Cozy stay a standalone process talking to ComfyUI over REST/WebSocket, or does it migrate (in whole or in part) into ComfyUI's process as a custom node package?

See: `02_ARCH-1_INSIDE_OUT_BRIEF.md`

### ARCH-2 — Moneta v1.3 Evolution

**The substrate question.** Once Comfy-Cozy needs real memory (post-Tier-1-2), Moneta needs more than the AttentionLog. Does Moneta absorb Harlo's state-machine pattern (thrashing, velocity, token budget) for the autonomous-agent case, or does it stay focused on outcome memory only?

See: `03_ARCH-2_MONETA_EVOLUTION_BRIEF.md`

### ARCH-5 — Workflow Graph as LIVRPS Substrate

**The patent question.** ComfyUI's workflow graph is structurally a composition substrate. Should it be formally treated as a LIVRPS-ordered substrate (with Moneta authoring workflow opinions and the user authoring workflow opinions, ComfyUI's engine resolving via override hierarchy), and should that be filed as a patent extension?

See: `04_ARCH-5_WORKFLOW_LIVRPS_BRIEF.md`

---

## 4. Decision Sequence

Each question is independent in principle, but the order matters for momentum:

```
ARCH-1 (transport)        →   if outside-in stays, ARCH-5 dies
       ↓
ARCH-5 (patent extension) →   if approved, shapes how ARCH-2 implements
       ↓
ARCH-2 (Moneta evolution) →   the implementation roadmap follows
```

**Recommended council session order:** ARCH-1 first, ARCH-5 second, ARCH-2 last.

---

## 5. Read Order for This Package

```
00 — This document (strategic frame)                  ← you are here
01 — Scout pass for Claude Code (current state inventory)
02 — ARCH-1 Brief (transport)
03 — ARCH-2 Brief (Moneta evolution)
04 — ARCH-5 Brief (workflow LIVRPS)
05 — README index
```

The scout pass (`01_SCOUT_INSIDE_OUT_v0_1.md`) is **not required reading before the council session** — it's a Claude Code prompt to be run *after* council decisions, to ground the implementation phase in real code state.

---

## 6. What This Package Does Not Do

Explicit non-goals, to keep the council focused:

- **Does not commit to a timeline.** Phase 2 work depends on council outcomes.
- **Does not specify implementation details.** The briefs surface trade-offs, not file diffs.
- **Does not re-litigate Moneta vs Harlo.** That decision is settled (4.29 ship). These briefs assume Moneta as Comfy-Cozy's substrate.
- **Does not decide patent filings.** ARCH-5 surfaces the option; the filing decision is downstream of council approval.
- **Does not touch the Synapse-Harlo-CB integration spec.** That is its own product with its own roadmap.

---

## 7. Mile Markers

```
Mile 1:  Council reads exploration plan + three briefs       (this package)
Mile 2:  Council session — 3 decisions logged                (decision-only)
Mile 3:  Scout pass executed via Claude Code                 (if any green)
Mile 4:  Phase 2 implementation plan drafted from scout      (if any green)
Mile 5:  Forge pass begins                                   (gated on plan)
```

This package gets you through Mile 1. Mile 2 is the council session — owned by Joe, no Claude Code involvement. Miles 3-5 are downstream of council outcomes.

---

## 8. The Honest Frame

The recommendation in each brief is mine, but I want to be clear about the asymmetry:

- **Outside-in is the safer path.** Less work, less risk, ships faster, current architecture continues.
- **Inside-out is the higher-ceiling path.** More work, more risk, ships slower, but unlocks the substrate isomorphism patent claim for ComfyUI and creates a second product validating the same architectural thesis.

This is not a "Claude says inside-out" conclusion. The briefs lay out the case for both directions and identify which conditions favor which. The council decides. I'm an instrument for surfacing the trade-offs cleanly.

---

**End of exploration plan. Continue to `01_SCOUT_INSIDE_OUT_v0_1.md` for the Claude Code reconnaissance prompt, or skip to the briefs (`02`, `03`, `04`) for the council reading order.**
