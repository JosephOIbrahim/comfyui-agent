# Playbook — Running the Inside-Out Pass

**For:** Joe
**Companion to:** `RUN_INSIDE_OUT_PASS.md`
**Format:** ADHD-native. Mile-marker style. Scan first, read second.

---

## The shape of this run

```
Pre-flight  →  Drop the prompt  →  GATE A  →  GATE B  →  GATE C  →  Done
   (you)         (Claude Code)    (you)     (you)     (you)
```

Three gates. You approve each one explicitly. Claude Code waits between phases. You can step away during work, you must show up at gates.

**Realistic time budget:** Pre-flight ~10min. Scout phase ~2–4 hours wall clock. Gate A ~15min. Architect ~1–2 hours. Gate B ~15min. Forge variable (LOC-dependent, plausibly 4–12 hours over multiple sessions). Gate C ~20min.

This is **not a one-sitting run.** It's a one-prompt run that spans multiple work blocks across days.

---

## Pre-flight checklist

Before pasting the prompt:

- [ ] **Branch created** — `git checkout -b architecture/inside-out-pass`
- [ ] **Working tree clean** — `git status` shows nothing to commit
- [ ] **Tests baseline confirmed** — `pytest` returns 2717 passing, 0 failures
- [ ] **`.venv312` activated** — Python 3.12.10
- [ ] **ComfyUI process killed** — port 8188 free (scout is read-only on the repo, but ComfyUI running while you scout is just noise)
- [ ] **Six docs in place** — `docs/inside-out-package/` contains:
   - [ ] `00_INSIDE_OUT_EXPLORATION_PLAN.md`
   - [ ] `01_SCOUT_INSIDE_OUT_v0_1.md`
   - [ ] `02_ARCH-1_INSIDE_OUT_BRIEF.md`
   - [ ] `03_ARCH-2_MONETA_EVOLUTION_BRIEF.md`
   - [ ] `04_ARCH-5_WORKFLOW_LIVRPS_BRIEF.md`
   - [ ] `README.md`
- [ ] **`COUNCIL_DECISIONS.md` written** — at repo root, documenting the three GREEN votes from 2026-05-01
- [ ] **`RUN_INSIDE_OUT_PASS.md` saved at repo root** — not inside `docs/`
- [ ] **Coffee. Snack within reach. Quiet block scheduled.**

If pre-flight fails on any item: fix the failed item before pasting. Don't paste against a dirty tree.

---

## How to launch

1. Open Claude Code in `G:\Comfy-Cozy`
2. Paste the entire contents of `RUN_INSIDE_OUT_PASS.md` as your first message
3. Wait for Claude Code to announce: `[SCOUT MODE] — Pre-flight verified. Beginning Phase 1.`
4. If pre-flight fails inside Claude Code: read what it says, fix, re-paste

You can step away after the announcement. Scout is mostly read-only inventory. Come back periodically.

---

## Mile markers — the run in shape

```
Mile 1  →  Pre-flight verified, [SCOUT MODE] announced
Mile 2  →  SCOUT working through 10 steps (read-only)
Mile 3  →  CRUCIBLE pass on scout outputs (in same session)
Mile 4  →  GATE A — your review needed

         ⚠️  PROCEED  /  REVISIT-COUNCIL  /  STOP

Mile 5  →  ARCHITECT drafts Phase 2 plan
Mile 6  →  CRUCIBLE pass on plan (in same session)
Mile 7  →  GATE B — your review needed

         ⚠️  PROCEED  /  REVISE  /  STOP

Mile 8  →  FORGE begins. File-by-file. Verify after each.
Mile 9  →  Forge complete. CRUCIBLE pass on changes.
Mile 10 →  GATE C — push approval needed

         ⚠️  PUSH  /  NEEDS REVIEW  /  BLOCKED

Mile 11 →  Pass complete. Branch ready.
```

---

## What "good" looks like at each gate

### GATE A — Post-scout

Claude Code surfaces:

- Executive summary: **FEASIBLE** / FEASIBLE-WITH-CAVEATS / BLOCKED
- HARD risks (or "none identified")
- Prior art findings
- Three-stores inventory key surfaces
- CRUCIBLE objections
- Recommendation: PROCEED / REVISIT-COUNCIL / STOP

**Approve PROCEED if:**
- Summary is FEASIBLE or FEASIBLE-WITH-CAVEATS
- No HARD risks unmitigated
- Prior art findings don't fundamentally narrow the CIP claim
- Three-stores inventory looks complete
- CRUCIBLE objections are acknowledgeable, not deal-breakers

**Trigger REVISIT-COUNCIL if:**
- Any HARD risk found that wasn't anticipated by ARCH-1 brief
- Prior art genuinely narrows or invalidates ARCH-5's CIP play
- Scout discovered something architectural that ARCH-1/-2/-5 didn't account for

**Trigger STOP if:**
- Pre-conditions for the inside-out architecture turn out to be unsatisfiable
- You've changed your mind on a council decision (it happens — that's what gates are for)

### GATE B — Post-architect

Claude Code surfaces:

- Plan headline (one sentence)
- Total LOC estimate + complexity class
- Top 3 risks per CRUCIBLE
- Items wanting your eye
- Recommendation

**Approve PROCEED if:**
- LOC estimate is realistic for your bandwidth
- The migration sequencing makes sense
- Test rebuild plan is concrete, not hand-wavy
- Rollback plan exists for each milestone
- HIGH/HARD risks have mitigations or you accept them

**Trigger REVISE if:**
- Plan is optimistic in a way that worries you
- A specific section feels under-thought
- You see an architectural call you want to re-make

**Trigger STOP if:**
- The plan reveals the work is bigger than the council understood
- You want to change the council vote

### GATE C — Pre-push

Claude Code surfaces:

- All commits made (hashes + plan section refs)
- Test count vs 2717 baseline
- Scope flags raised
- Blockers resolved/escalated
- Recommendation: READY TO PUSH / NEEDS REVIEW / BLOCKED

**Approve PUSH if:**
- All tests passing (≥2717, ideally more)
- All blockers resolved (not deferred)
- Scope flags reviewed and you're okay with what got deferred
- The branch state matches the plan you approved at GATE B

**Trigger NEEDS REVIEW if:**
- You want to inspect specific commits before push
- A scope flag concerns you
- A test count looks weird

**Trigger BLOCKED if:**
- Forge surfaced a blocker that requires council attention
- Tests are not at baseline

---

## What "stop" looks like — red flags

Things to watch for during the run:

🚩 **Claude Code starts implementing during scout.** Constitution violation. Stop, point at Rule 5 (Role Isolation), reset.

🚩 **Test count drops without an explanation.** Stop the work. Have Claude Code diagnose before any further commits.

🚩 **A commit happens before a gate approval.** Constitution violation. Stop, identify what was committed, decide whether to revert.

🚩 **Claude Code tries to push without GATE C approval.** Per Git Authority Map: forbidden. Stop hard.

🚩 **A `# TODO` or `// ... existing code ...` shows up in any committed file.** Constitution Rule 4 violation. Stop, fix, commit-amend.

🚩 **A test gets weakened to make something pass.** Constitution Rule 7 violation. Stop, restore the test, fix the implementation instead.

🚩 **A scope flag describes work Claude Code is about to do anyway.** That's not a flag — that's freelancing. Stop, push back, reset to original scope.

If any of these happen: name the rule, name the violation, ask Claude Code to revert and resume from the last clean state.

---

## Recovery — if it stalls or surfaces a blocker

**Stall (no progress, no error):** ask "what's the current task and what's the retry count?"

**Stall during scout:** likely an unreachable resource (rate-limited API, blocked domain). Have Claude Code document the gap in the relevant scout output and continue.

**Stall during architect:** likely a missing piece of scout data. Identify the gap, decide whether to re-run that scout step or proceed with the gap acknowledged.

**Stall during forge:** likely a test failure beyond retry budget. Read the blocker file. Decide: human fix, council re-decision, or stop.

**Hard blocker:** Claude Code wrote `BLOCKERS/BLOCKER_<phase>_<n>.md`. That's correct behavior. Read it. Decide. Tell Claude Code to either resume (with the blocker resolved) or stop (with the blocker carrying forward to Joe's separate handling).

---

## After the run

When GATE C lands and you push, the inside-out pass closes. **Open work after that:**

- **Three-isolated-stores brief** — your parallel-track doc, derived from Phase 1's `three_stores_inventory.md`. Write when you have bandwidth.
- **Patent counsel review** — schedule for ARCH-5 CIP. The `prior_art_sweep.md` from Phase 1 is the input.
- **Phase 3 work continuation** — the forge pass implements the *first* migration milestones, not the entire inside-out architecture. Subsequent migration milestones are separate runs (each with their own scout+architect+forge if the scope warrants).
- **Council retrospective** — note for next time: which CRUCIBLE objections turned out to matter, which were noise. Sharpen the next council session.

---

## ADHD-native execution notes

- **Don't push during a burst.** Save GATE C for a clean-headed moment. Push is irreversible-ish.
- **Run gates with the doc open.** Don't try to remember what GATE B's checklist was — open this file, read it, decide.
- **If frustration rises during a gate review:** that's signal. Take a walk. Come back. Gates are not timed.
- **If it feels too big mid-forge:** scope flags are your friend. Tell Claude Code to flag the part that feels too big, ship the smaller scope, return to the flagged piece in a new run.
- **Wife or kids interrupt:** clean exit. Claude Code holds state. Resume when you're back.

---

## One-line summary

**Three gates. You approve each. Claude Code does the work between. Constitution wins every conflict.**

That's the run.
