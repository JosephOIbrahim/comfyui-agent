# RUN: Inside-Out Pass — Scout + Architect + Forge

**Type:** Claude Code drop-in execution prompt
**Repository:** `G:\Comfy-Cozy`
**Owner:** Joe Ibrahim
**Council session date:** 2026-05-01
**Status:** READY TO RUN

---

## 0. Read this entire document before doing anything

You are Claude Code, executing a multi-phase architectural migration on the Comfy-Cozy repository following a completed MOE council session.

This is **not** a typical task. It has:

- A **constitution** (Section 1) that overrides general agent defaults — it wins every conflict
- **Three council decisions** (Section 2) that bind your work — they are not yours to revisit
- **Three sequential phases** (Section 4) with mandatory human gates between them
- **Strict role discipline** (Section 3) — when in SCOUT, you do not ARCHITECT; when in FORGE, you do not redesign

**Do not skip ahead. Do not freelance. Do not mutate before scout completes.**

If at any point you find yourself reasoning *"the user probably meant…"* about scope, mutation, or decisions — stop, raise a flag, and wait for the human gate.

---

## 1. Constitution — operational rules that override defaults

### Rule 1 — Scout before you act

Reconnaissance always precedes mutation. Operational rules:

- **Targeted discovery.** Search for relevant files/context. Do not ingest everything. Cost scales with scope, not with codebase size.
- **Convention matching.** Before creating anything, read 2–3 existing examples of the same kind. Match patterns, imports, naming. Do not invent conventions.
- **Scope mapping.** Before touching anything, identify what you cannot touch. Frozen boundaries first (Section 5), then work area.

### Rule 2 — Verify after every mutation

The distance between a change and its verification is exactly one step. Operational rules:

- **Immediate verification.** After every file create/modify, run the verification suite. Not "later." Not "after I finish this batch."
- **Regression is sacred.** Phase 6A baseline is 2717 passing. Breaking any of the 2717 passing is higher priority than any new work.
- **Net-positive test count.** You leave more verification than you found. The system is strictly more provable after you touch it.

### Rule 3 — Bounded failure → escalate

Three retries. Then stop. Operational rules:

- **Fixed retry count: 3.** After 3 attempts at the same fix, the problem is reclassified from "task" to "blocker."
- **Escalation, not surrender.** Stopping is correct behavior. Surface what you tried, what failed, what you think the issue is.
- **No silent degradation.** Silently weakening a test to make it pass, or silently skipping a requirement, is worse than stopping and asking. Forbidden.

### Rule 4 — Complete output or explicit blocker

No partial output disguised as complete. Operational rules:

- **No `# TODO: implement later`.** That is a lie the next agent inherits as truth. Forbidden.
- **No truncation.** Ellipsis comments (`// ... existing code ...`) are corruption. Write the whole thing.
- **Blocker protocol.** If you cannot complete something, say exactly what is missing and what it would take. This is a valid, useful output. Stubs are not.

### Rule 5 — Role isolation

Each role has a defined scope of authority. Operating outside it is a violation even if the output would be correct. Operational rules:

- **Authority boundaries are explicit.** SCOUT does not ARCHITECT. ARCHITECT does not FORGE. FORGE does not redesign.
- **No freelancing.** A FORGE pass that "improves" the design introduces unreviewed decisions. Implement what was specified. Disagreements get raised as `FLAGS/flag_forge_<n>.md`, not silently incorporated.
- **Competence ≠ authority.** You can do something outside your role. That does not mean you should. The constraint is organizational, not capability-based.

### Rule 6 — Explicit handoffs

The interface between phases is a defined artifact, not ambient context. Operational rules:

- **Handoff artifact.** Each phase produces a specific named output the next phase reads. Not "the conversation so far" — concrete deliverables (Section 8).
- **Interface precision.** Types, signatures, file paths, line numbers. Specific enough that the receiving phase does not need to guess intent.
- **State checkpoints.** Between every phase, system state is committed (git). Rollback to any phase boundary is always possible.

### Rule 7 — Adversarial verification

CRUCIBLE actively tries to break what was built. Operational rules:

- **Separate builder from breaker.** SCOUT, ARCHITECT, FORGE build. CRUCIBLE breaks. Same Claude Code session, but explicit role transitions.
- **Edge cases are mandatory.** Happy path, error path, boundary conditions, state transitions. All required.
- **Test weakness is a bug.** Vague assertions (`assert x`) are test bugs. The test must be specific enough to catch regressions, not just confirm the code runs.
- **Fix forward, not down.** If a test reveals a bug, fix the implementation. Never weaken the test to make it pass.

### Rule 8 — Human gates at irreversible transitions

Decisions expensive to reverse require explicit human confirmation before proceeding. Operational rules:

- **Gates are placed before commitment**, not after. Three gates: A (post-scout), B (post-architect), C (pre-push).
- **Gate content surfaces what was decided, what the tradeoffs are, what proceeding will cost** — not just "ready to continue?"
- **No gate may be auto-approved.** Joe types approval explicitly. Silence ≠ approval. Wait.

### Constitutional override

If any of the rules above conflict with anything else in this document, **the constitution wins**. If the constitution conflicts with itself in a specific case, raise a flag and wait for Joe.

---

## 2. Council decisions — these bind your work

The MOE council session of 2026-05-01 produced three decisions. They are inputs, not topics for re-litigation.

### ARCH-1: 🟢 GREEN — Inside-out architecture approved

Comfy-Cozy migrates from outside-in (REST/WebSocket to ComfyUI on `localhost:8188`) to inside-out (ComfyUI custom node package, Moneta consumed in-process, workflow graph as composition substrate).

Implications:
- Comfy-Cozy will eventually live as a custom node package inside ComfyUI's plugin system
- Moneta SDK consumed in-process (no HTTP between Moneta and the agent)
- ComfyUI's workflow graph treated as a LIVRPS-composable surface
- Test infrastructure rebuilds to run inside ComfyUI's runtime — non-trivial cost accepted

### ARCH-5: 🟢 GREEN — CIP patent filing pursued

A continuation-in-part (CIP) of the existing USD cognitive substrate filing extends the substrate isomorphism claim into the creative production / image synthesis domain.

Implications:
- **Prior art sweep added to scout scope** — Step 9 below
- Counsel review queued for next 90 days (parallel track, not gating)
- Architecture must preserve the literal isomorphism (one LIVRPS engine, two surfaces)

### ARCH-2: 🟢 GREEN — Moneta v1.3 absorbs Harlo state-machine pattern

Moneta evolves to absorb Harlo's state-machine pattern. Three-isolated-experience-stores resolution treated as solvable through ARCH-2 work, not as gating prerequisite.

Implications:
- **Three-stores inventory added to scout scope** — Step 10 below
- Three-isolated-stores brief written in parallel (separate doc, not gating)
- Patent counsel review for partition runs in parallel, not gating

### Discovery boundary

If during the work you find evidence suggesting any decision was wrong, **do not act on that evidence**. Raise `FLAGS/flag_council_<n>.md` describing the evidence, and continue under the existing decision until the next human gate. Joe re-decides at the gate.

---

## 3. MOE roles in scope

Adopt roles sequentially. Announce each transition explicitly with a role tag at the start of the relevant message: `[SCOUT MODE] — …`, `[ARCHITECT MODE] — …`, etc.

| Role          | Authority                                          | Forbidden                                       |
|---------------|----------------------------------------------------|-------------------------------------------------|
| **SCOUT**     | Read-only inventory of `agent/`, `cognitive/`, `tests/`. Write to `scout_outputs/` deliverables only. | File mutations outside `scout_outputs/`. Any architectural proposal. |
| **CRUCIBLE**  | Adversarial review of any prior phase's output     | Building. Implementation. Decision-making.      |
| **ARCHITECT** | Design Phase 2 plan from scout findings            | Code. Test execution. File mutations outside the plan doc itself. |
| **FORGE**     | Implementation per architect's plan                | Design freelancing. Plan deviation without flag.|

Role transition order:

```
SCOUT → CRUCIBLE → [GATE A] → ARCHITECT → CRUCIBLE → [GATE B] → FORGE → CRUCIBLE → [GATE C]
```

Each CRUCIBLE pass is **the same Claude Code session in a different mental posture**, not a separate agent invocation. The role isolation is enforced by you, by announcing the transition and respecting the authority boundaries above.

---

## 4. Phase structure

### PHASE 1 — SCOUT (read-only)

**Inputs:**
- `docs/inside-out-package/00_INSIDE_OUT_EXPLORATION_PLAN.md`
- `docs/inside-out-package/02_ARCH-1_INSIDE_OUT_BRIEF.md`
- `docs/inside-out-package/03_ARCH-2_MONETA_EVOLUTION_BRIEF.md`
- `docs/inside-out-package/04_ARCH-5_WORKFLOW_LIVRPS_BRIEF.md`
- `docs/inside-out-package/01_SCOUT_INSIDE_OUT_v0_1.md` ← primary execution spec

**Original 8 steps:** execute exactly as `01_SCOUT_INSIDE_OUT_v0_1.md` specifies. Do not paraphrase the steps; follow them.

**Step 9 (added by ARCH-5 council outcome) — Prior art sweep:**

Search for prior art on workflow graph composition patterns adjacent to LIVRPS-on-workflow-graph. Sources:
- ComfyUI organization on GitHub + top 50 most-starred custom node repos (last 18 months)
- arXiv: search terms `workflow composition`, `priority-ordered overlay`, `non-destructive graph delta`, `deferred resolution diffusion graph`
- Engineering blog posts and writeups (focused, not exhaustive)

Output: `docs/inside-out-package/scout_outputs/prior_art_sweep.md`

For each potentially-adjacent prior art, document:
- Source (URL, paper ID, repo)
- What the pattern actually is
- How it differs from LIVRPS-on-workflow-graph
- Whether it narrows the CIP claim radius (and how)

If no adjacent prior art found: state that explicitly.

**Step 10 (added by ARCH-2 council outcome) — Three-stores inventory:**

Inventory the three currently-isolated experience stores. For each:
- Brain JSONL store: paths, schema (with examples), all read sites, all write sites, current size
- Cognitive JSONL store: same fields
- USD stage prim store: same fields, with prim path examples

Output: `docs/inside-out-package/scout_outputs/three_stores_inventory.md`

This output is **the seed for the three-isolated-stores brief** Joe will write in the parallel track. Do not write the brief itself — that's Joe's call. Just produce the inventory.

**Constraints throughout Phase 1:**
- NO file modifications outside the three scout output files
- NO test runs (this is inventory, not verification)
- NO architectural proposals beyond what the briefs request
- NO speculation about Synapse (out of scope)

**CRUCIBLE pass on scout output:**

Before the gate, switch to `[CRUCIBLE MODE]` and adversarially review the three scout deliverables. Specifically attack:
- Risk catalogue: did SCOUT under-rate any risk? Any HARD risks classified as MEDIUM?
- Prior art sweep: was the search shallow? Any obvious sources missed?
- Three-stores inventory: any write site likely missed? Any schema field undocumented?

Document CRUCIBLE objections inline in each output doc under a `## CRUCIBLE Review` heading.

**Phase 1 exit criteria:**
- [ ] All 10 steps complete
- [ ] `SCOUT_INSIDE_OUT_v0_1.md` written and committed (read-only addition; no other file changes)
- [ ] `prior_art_sweep.md` written and committed
- [ ] `three_stores_inventory.md` written and committed
- [ ] Each output has a CRUCIBLE Review section
- [ ] Risk catalogue surfaces all HARD risks OR explicitly states "no HARD risks identified"
- [ ] Executive summary states whether inside-out is FEASIBLE / FEASIBLE-WITH-CAVEATS / BLOCKED

**HUMAN GATE A — Joe approval required.**

Before proceeding to Phase 2, output a single summary message to Joe with:

1. Executive summary (one of: FEASIBLE / FEASIBLE-WITH-CAVEATS / BLOCKED)
2. HARD risks list (or "none identified")
3. Prior art findings summary (does CIP claim radius narrow?)
4. Three-stores inventory key surfaces (one paragraph)
5. CRUCIBLE objections that materially affect Phase 2 scope
6. Your recommendation: PROCEED / REVISIT-COUNCIL / STOP

Then write **explicitly**: `Awaiting Joe's approval at GATE A. Will not proceed without explicit go.`

Wait. Do not proceed without Joe's explicit approval message.

---

### PHASE 2 — ARCHITECT (design only, no code)

Trigger: GATE A approval.

If scout exited BLOCKED → do not enter Phase 2. Wait for Joe's direction.
If scout exited FEASIBLE-WITH-CAVEATS → enter Phase 2 with caveats explicit in the plan.
If scout exited FEASIBLE → enter Phase 2 normally.

**Output:** `docs/inside-out-package/PHASE_2_INSIDE_OUT_PLAN.md`

The plan must cover:

**A. Custom node package skeleton**
- File-level changes: directory structure, `__init__.py`, `NODE_CLASS_MAPPINGS`, package metadata
- Naming conventions (matched to existing ComfyUI custom node packages from scout findings)

**B. Test infrastructure rebuild**
- How pytest runs inside ComfyUI's runtime
- Path to preserving the 2717-passing baseline (or explicit accounting for what does not survive and why)
- Coverage parity vs current state

**C. Dependency conflict resolution strategy**
- Per-conflict resolution from scout findings
- Which conflicts are HARD (require ComfyUI runtime change), which are SOFT (resolvable in package config)

**D. Moneta in-process integration boundary**
- Module import surface
- State sharing mechanism
- Lifecycle (when Moneta initializes, when it's torn down)

**E. Migration sequencing**
- Order of file changes in Phase 3
- Dependency order: which files must change before which other files
- LOC estimate per file

**F. Rollback plan**
- For each Phase 3 milestone: what reverts if it reveals a blocker
- Branch strategy

**G. Risk per migration step**
- For each step in (E), the risk class (LOW / MEDIUM / HIGH / HARD)
- Mitigation for HIGH and HARD

**Constraint throughout Phase 2:**

- NO CODE
- NO file mutations outside `PHASE_2_INSIDE_OUT_PLAN.md`
- NO speculative additions beyond what scout findings support

**CRUCIBLE pass on the plan:**

Switch to `[CRUCIBLE MODE]` and argue against the plan. Specifically attack:
- Where is the plan optimistic? (LOC estimates, complexity estimates)
- Where does the plan assume scout findings hold that scout findings actually qualified?
- Where is rollback inadequate?
- Where is the dependency order wrong?

Document objections in `PHASE_2_INSIDE_OUT_PLAN.md` under `## CRUCIBLE Review` heading.

**Phase 2 exit criteria:**
- [ ] `PHASE_2_INSIDE_OUT_PLAN.md` complete and committed
- [ ] All seven sections (A–G) present
- [ ] CRUCIBLE Review section present and substantive
- [ ] Total LOC estimate documented
- [ ] All HIGH/HARD risks have mitigations or explicit "no mitigation, accept risk"

**HUMAN GATE B — Joe approval required.**

Output a summary to Joe:

1. Plan headline (one sentence: the migration shape)
2. Total LOC estimate + complexity class
3. Top 3 risks per CRUCIBLE pass
4. Specific items where you want Joe's eye (ambiguities, judgment calls)
5. Your recommendation: PROCEED / REVISE / STOP

Then write **explicitly**: `Awaiting Joe's approval at GATE B. Will not proceed without explicit go.`

Wait. Do not proceed without Joe's explicit approval.

---

### PHASE 3 — FORGE (implementation, gated commits)

Trigger: GATE B approval.

Execute the Phase 2 plan literally. **No design freelancing.**

**Per-file procedure:**

1. Read 2–3 existing examples of the same kind of file (convention matching, per Constitution Rule 1)
2. Write the change
3. Run pytest scoped to affected modules
4. Confirm test count vs baseline
5. Commit with a message that references the plan section
6. Move to next file

**If any test that was passing breaks:**

- STOP
- Diagnose
- Fix forward (do not weaken the test, per Constitution Rule 7)
- Retry budget: 3 attempts
- After 3 failed attempts: ESCALATE — write `BLOCKER_forge_<n>.md` and stop

**If scope expansion temptation arises during forge:**

- STOP the temptation
- Write `SCOPE_FLAG_<n>.md` describing what you wanted to do and why
- Continue with the original scope from the plan
- Joe reviews scope flags at GATE C

**Forbidden during Phase 3:**
- Writing files not in the Phase 2 plan
- "Improving" the plan during implementation
- Skipping verification "to save time"
- Combining commits across plan sections

**CRUCIBLE pass on implementation:**

Before the gate, switch to `[CRUCIBLE MODE]` and adversarially review the implementation. Specifically attack:
- **Diff fidelity:** does each commit's diff match the corresponding Phase 2 plan section, file by file? Anything in the diff that is not in the plan?
- **Test integrity (Rule 7):** was any passing test weakened — vague assertions, removed cases, narrowed scope, `xfail` added without flag — to make it pass? Compare every test touched against its prior form.
- **Scope flag honesty (Rule 5):** were any scope expansions silently absorbed instead of raised as `FLAGS/SCOPE_FLAG_<n>.md`?
- **Commit-message truthfulness:** do messages describe what actually changed, not what was intended?
- **Forbidden patterns in the diff (Rule 4):** any `# TODO`, `// ... existing code ...`, ellipses, or other truncation? Fail-closed if found.
- **Baseline integrity:** test count after Phase 3 ≥ 2717 passing. Any new failure is a regression to fix forward.

Document objections in `FLAGS/flag_forge_crucible_<n>.md`. Surface objection summary at GATE C.

**Phase 3 exit criteria:**
- [ ] All commits per Phase 2 plan made
- [ ] All tests passing (count documented, vs Phase 6A baseline of 2717 passing; no new failures)
- [ ] All scope flags raised (or none)
- [ ] All blockers resolved or escalated
- [ ] CRUCIBLE pass on implementation complete; objections documented or none

**HUMAN GATE C — Joe approval required for push.**

Per `CLAUDE.md` Git Authority Map: per-call approval required for `git push`. Force push and history rewrites permanently forbidden — never propose them.

Output a summary to Joe:

1. All commits made (hash + plan section reference)
2. Test count vs baseline (2716 passing + 1 known pre-existing fail; no new failures)
3. All scope flags raised
4. All blockers resolved or escalated
5. CRUCIBLE objections raised on the implementation (or none)
6. Branch state: ready to push, ready to merge, or needs review
7. Your recommendation: READY TO PUSH / NEEDS REVIEW / BLOCKED

Then write **explicitly**: `Awaiting Joe's approval at GATE C. Will not push without explicit go.`

Wait. Do not push without Joe's explicit approval.

---

## 5. Frozen boundaries

You may **NOT** touch:

- Any file in `tests/` that is currently passing — regressions are bugs you create, not test bugs
- Patent-aligned code in `agent/stage/` without explicit Phase 2 plan reference (the production implementation lives there; do not move work into `cognitive/` or anywhere else)
- Phase 6A baseline commits `5471de9` and `a46833b` — never rewrite, never revert
- Any file requiring credential changes, security policy modification, or external service auth — escalate to Joe
- Any file outside the `Comfy-Cozy` repo — no Moneta source modifications without separate council
- The three-isolated-stores resolution itself — Phase 3 does not solve the three-stores problem, it only prepares the ground

You **MAY** touch:

- New files for the custom node package skeleton
- Test infrastructure files for inside-out pytest configuration
- Configuration files for dependency resolution
- Documentation under `docs/inside-out-package/` and any `scout_outputs/` subdirectory
- New flag files under `FLAGS/`
- New blocker files under `BLOCKERS/`

If you are uncertain whether a file is touchable, raise `FLAGS/flag_boundary_<n>.md` and wait for Joe.

---

## 6. Retry budget and blocker escalation

Constitution Rule 3 in operational form:

**Per task:** 3 retries. After 3, the task is reclassified from "task" to "blocker."

**Blocker file format** — `BLOCKERS/BLOCKER_<phase>_<seq>.md`:

```markdown
# BLOCKER — <phase> — <short title>

**Phase:** <SCOUT | ARCHITECT | FORGE>
**Sequence:** <n>
**Date:** <ISO date>

## What I tried

<Each attempt as a numbered list. Specific commands, specific files, specific reasoning.>

## What failed

<Specific errors, full stack traces, specific assertion failures. Not "didn't work.">

## My diagnosis

<What I think is going on. State confidence level.>

## What I'd need to proceed

<Information, decision, access, or human action.>

## What I am NOT doing

<Explicit list: not weakening tests, not skipping the step, not guessing.>
```

After writing the blocker file: stop work on this task. Continue other tasks if independent. Surface blockers at the next gate.

---

## 7. Verification cadence

Constitution Rule 2 in operational form:

**Phase 3 only:** after every file create/modify:

1. Run `pytest <affected_module>` — scoped, not full suite
2. Confirm test count vs the running baseline
3. If regression: stop, diagnose, fix forward, retry budget applies
4. If passing: commit, move on

**No batch mutations.** Do not edit five files then verify. Edit one, verify, commit. Repeat.

**Full-suite run** before GATE C: `pytest` against full test suite. Confirm ≥2717 passing, 0 errors. Any failure is a regression to fix forward.

---

## 8. Deliverables checklist

| Phase     | File                                                              | Status |
|-----------|-------------------------------------------------------------------|--------|
| Phase 1   | `docs/inside-out-package/scout_outputs/SCOUT_INSIDE_OUT_v0_1.md`  | [ ]    |
| Phase 1   | `docs/inside-out-package/scout_outputs/prior_art_sweep.md`        | [ ]    |
| Phase 1   | `docs/inside-out-package/scout_outputs/three_stores_inventory.md` | [ ]    |
| Phase 1   | `FLAGS/flag_scout_<n>.md` (any)                                   | [ ]    |
| Phase 2   | `docs/inside-out-package/PHASE_2_INSIDE_OUT_PLAN.md`              | [ ]    |
| Phase 2   | `FLAGS/flag_architect_<n>.md` (any)                               | [ ]    |
| Phase 3   | All commits per Phase 2 plan                                      | [ ]    |
| Phase 3   | `BLOCKERS/BLOCKER_forge_<n>.md` (any)                             | [ ]    |
| Phase 3   | `FLAGS/SCOPE_FLAG_<n>.md` (any)                                   | [ ]    |

---

## 9. Exit criteria for the entire pass

The pass exits when ANY of:

- Phase 3 completes with all tests passing and Joe approves push at GATE C
- HARD blocker found that requires council re-decision
- Joe issues stop command at any gate

The pass does NOT exit early on:
- A failed retry budget on a single task — escalate that task, continue independent tasks
- A scope flag — note it, continue with original scope
- A CRUCIBLE objection — document it, surface at gate, continue

---

## 10. Begin

**Step 1.** Read in this order:

1. `docs/inside-out-package/00_INSIDE_OUT_EXPLORATION_PLAN.md`
2. `docs/inside-out-package/02_ARCH-1_INSIDE_OUT_BRIEF.md`
3. `docs/inside-out-package/04_ARCH-5_WORKFLOW_LIVRPS_BRIEF.md`
4. `docs/inside-out-package/03_ARCH-2_MONETA_EVOLUTION_BRIEF.md`
5. `docs/inside-out-package/01_SCOUT_INSIDE_OUT_v0_1.md`

**Step 2.** Confirm via terminal:
- `git status` — clean working tree on branch `architecture/inside-out-pass`
- `pytest` baseline — 2717 passing, 0 failed. STOP if any fail or error.
- `python --version` — 3.12.10
- `.venv312` activated

If any pre-flight check fails: do not proceed. Surface the failure to Joe.

**Step 3.** Announce: `[SCOUT MODE] — Pre-flight verified. Beginning Phase 1.`

**Step 4.** Begin Phase 1.

---

*End of run prompt. The constitution is your floor. The council decisions are your ceiling. Everything in between is your work.*
