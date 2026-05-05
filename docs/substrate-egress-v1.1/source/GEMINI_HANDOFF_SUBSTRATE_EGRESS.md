# HANDOFF — Substrate Egress v1: Strategic Review Request

**To:** Gemini Deep Think
**From:** Joe Ibrahim (Creative Director)
**Date:** May 1, 2026
**Companion artifact:** `MISSION — Substrate Egress v1` *(attached separately — read before answering)*
**Type:** Strategic review — **decision-only, no forge, no mission redraft**
**Parallel track:** `hardening/gemini-t1-t4` *(your existing branch — out of scope here, but Question 4 touches it)*

---

## 1. Why this handoff exists

Substrate Egress v1 is a two-phase mission to graduate `comfy-moneta-bridge` from v0 (synthetic embeddings, write-only substrate) to v1 (real BGE-small embeddings + brain query egress). The mission closes the gap between **"powered by Moneta" as a brand claim and as a runtime mechanism inside Comfy-Cozy**.

The mission as drafted is internally coherent. Phase split is correct. Acceptance gates are testable.

**Four integration questions remain that span beyond the mission's scope.** They touch the MOE council process, the vision evaluator milestone, the three-isolated-experience-stores problem, and your own T3 hardening work. I want adversarial review on each before Phase 1 starts.

This is decision-only work. **No code changes. No mission rewrite. No 9-item hitlist.** Four structured answers, ranked, with the strongest counter-arguments you can produce.

---

## 2. Context primer

Three things you should know going in. The first two are corrections from prior reviews.

**Synthetic embeddings were correct for v0, not a defect.** Your 4.29 architecture pushed back on my over-engineered BGE-from-day-one draft. You were right then. Substrate Egress graduates v0→v1 because the demo arc has expanded beyond same-session cold-vs-warm replay. Phase 1's BGE adoption *honors* your v0 reasoning — it doesn't reverse it.

**LIVRPS State Spine and patent-claimed subsystems remain out of scope.** Same boundary as your T1-T4 track. Substrate Egress doesn't touch State Spine, FORESIGHT, MOE routing core, or Lossless Signal. Your answers should not propose changes there either.

**The three isolated experience stores is a known upstream problem.** Brain JSONL · cognitive JSONL · USD stage prims have zero data sharing. This is sitting in front of the MOE council as a question that needs its own brief. Relevant to Question 3.

---

## 3. The four questions

### ▸ Question 1 — Sequencing against the MOE council vote on ARCH-1

**Background.** The 4.30 council package included ARCH-1: outside-in vs inside-out transport. My provisional recommendation in that brief was 🟢 GREEN with timing constraint — *"after Tier 1-2 hardening closes."* The council has not yet voted.

**Tension.** Substrate Egress Phase 2 *is* inside-out wiring. `intent_collector._safe_query` calls Moneta directly inside the agent turn loop. Running Phase 2 before the council votes on ARCH-1 is a soft override of the council process I deliberately set up.

**My provisional read.** Phase 1 before the council. Phase 2 paused until ARCH-1 votes. Phase 1 is internally additive — new embedding path, isolated `v2/` storage URI, old `v1/` data preserved. It *informs* the council's ARCH-1 reasoning by giving them a real semantic index to argue about, not a stub. Phase 2 commits the brain to inside-out and should not precede the vote.

**What I want from you:**

- Does Phase 1 *actually* inform the ARCH-1 vote, or is that justification for running it early?
- Is there a coherent argument for running Phase 2 *before* the council that I'm not seeing?
- **CRUCIBLE:** What's the strongest case for ignoring the council process here entirely?

---

### ▸ Question 2 — Phase 2 Point 2's quality_score dependency on a vision evaluator that doesn't exist

**Background.** Phase 2 Point 2 wires `signal_attention` to fire on high-quality iterations:

```python
quality = output_analysis.get("quality_score", 0.0)
if quality >= 0.7 and self._memories_used:
    _safe_signal_attention(weights)
```

The vision evaluator is a pre-existing milestone that gates Path B autoresearch. It has not been built.

**Failure mode.** Without vision evaluator, `quality_score` defaults to `0.0`. The gate never fires. `signal_attention` never gets called. **The decay math becomes meaningless** — exactly the failure mode the mission warns about *("Session 100 knows your style only works with attention signal")*. Memories decay at the same rate they're learned. The substrate forgets as fast as it learns.

The mission doc treats Point 2 as part of Phase 2's done-definition. **It silently no-ops.** Demo-day surprise risk.

**Three options on the table:**

| Option | Move |
|---|---|
| **A** | Defer Point 2 to Phase 2.5, alongside the optimizer prior that already waits on corpus volume |
| **B** | Wire Point 2 to a placeholder quality signal (user_feedback rating, iteration-completed boolean, last-pixel SSIM against reference) and label it honestly as a stub |
| **C** | Sequence vision evaluator before Phase 2 (changes the mission timeline) |

**What I want from you:**

- Which of A / B / C is right? Show the reasoning.
- If B, what's the most honest placeholder that doesn't pretend to be quality scoring?
- **CRUCIBLE:** Is there a fourth option I'm missing? Is there a version of Point 2 that does useful work *without* a quality score at all?

---

### ▸ Question 3 — Does Moneta become the canonical experience store, or sit alongside the other three?

**Background.** Currently three isolated experience stores in Comfy-Cozy: Brain JSONL · cognitive JSONL · USD stage prims. Zero data sharing. This isolation is a known upstream architectural problem flagged for the MOE council.

**Substrate Egress introduces Moneta as something the brain queries.** Two divergent paths:

| Path | Architectural effect |
|---|---|
| **Canonical** | Moneta becomes *the* experience store. Brain reads from it. Other three get deprecation paths. "Powered by Moneta" becomes architecturally true at every layer of Comfy-Cozy, not just the bridge. |
| **Additive** | Moneta sits as a fourth store. Brain queries it for cross-session memory only. Other three remain unchanged. The isolation problem now has one more node. |

The mission doc does not pick. **It has to.** Otherwise Phase 2 lands and Comfy-Cozy has *four* isolated stores, not three. This is partially an ARCH-2 question (Moneta v1.3 evolution) but Substrate Egress forces it earlier than ARCH-2's timeline assumed.

**What I want from you:**

- Which path is architecturally cleaner? Show the trade-offs explicitly.
- Is "Moneta becomes canonical" a reasonable v1 commitment, or is it a v2 decision wearing v1 clothes?
- **CRUCIBLE:** What does *additive* break that *canonical* doesn't? What does *canonical* break that *additive* doesn't?

---

### ▸ Question 4 — BGE-small download intersects your T3 track

**Background.** Your T3 task is resumable model downloads + SHA256 verification. Phase 1 introduces a new ~130MB BGE-small download via `sentence-transformers`. The mission doc treats BGE download as out-of-scope to T3 and gates it behind a `[real-embed]` pip extra.

**What I want from you:**

- Should BGE-small ingest through your T3 pipeline — resumable + SHA256 verified + cache-managed alongside the rest of model provisioning?
- If yes, what's the minimum coordination needed between Phase 1 and T3? Shared cache dir? Shared download helper? Joint `models.json` manifest?
- If no, what's the argument for keeping them separate beyond *"sentence-transformers handles its own download"*?

This is operational, not strategic. Give me the cleanest answer.

---

## 4. What I want back

A structured response per question. For each, exactly three sections:

- **Recommendation** *(with reasoning — not just a verdict)*
- **Counter-arguments** *(the strongest case against your own recommendation — adversarial pressure)*
- **What you'd do differently** *(if you were authoring the Substrate Egress mission directly)*

Not a redraft of the mission. Not a hitlist. Four answers, ranked by which one most changes the plan. Tell me which one matters most.

---

## 5. What this handoff does NOT do

- Does not authorize any code changes
- Does not unblock Phase 1 start
- Does not modify your T1-T4 track timeline
- Does not pre-empt the MOE council vote on ARCH-1 / ARCH-2 / ARCH-5
- Does not touch patent-claimed subsystems
- Does not relitigate v0 architecture decisions *(synthetic embeddings were right for v0)*

---

## 6. Hard rules unchanged from prior handoffs

- **LIVRPS State Spine is immutable.** No `deque(maxlen=N)` proposals on state history.
- **MOE routing already phase-gates tools.** No Capability Registry refactor proposals.
- **ComfyUI process is stable.** No supervisor wrappers, no automated restart loops.
- **If a question pushes you toward patent-adjacent territory, surface it instead of answering.**

---

## 7. Marathon markers

```
Mile 1 ─  You read this handoff and the Substrate Egress mission doc end-to-end
Mile 2 ─  You return four structured answers (one per question)
Mile 3 ─  Joe reads the answers, decides on each
Mile 4 ─  Mission doc gets a delta only if an answer changes the plan
Mile 5 ─  Phase 1 starts · council session schedules · vision evaluator queued
───────  HANDOFF COMPLETE  ───────
```

Standby. No execution work in this loop.

---

## 8. Cross-references

- **Companion artifact:** `MISSION — Substrate Egress v1` *(the doc this handoff reviews)*
- **Upstream context:** MOE council package (4.30) — `00_strategic_frame`, `02_ARCH-1_brief`, `03_ARCH-2_brief`, `04_ARCH-5_brief`
- **Test baseline:** Comfy-Cozy 2717 passing · `comfy-moneta-bridge` v0.1.0 · 49 tests green
- **Repos:**
  - `github.com/JosephOIbrahim/Comfy-Cozy`
  - `github.com/JosephOIbrahim/Moneta`
  - `github.com/JosephOIbrahim/comfy-moneta-bridge`

---

*Adversarial pressure expected, not optional. CRUCIBLE the recommendations harder than you'd CRUCIBLE a junior's.*
