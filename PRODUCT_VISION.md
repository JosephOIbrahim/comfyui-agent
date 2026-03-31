# PRODUCT VISION

**Agent:** `[DESIGN × ARCHITECT]`
**Status:** DRAFT — Awaiting Creative Director approval

---

## The Sentence

**SuperDuper is the first AI generation tool that gets better at your job by doing your job.**

Not better at generating images. Better at generating *your* images — the ones with your taste, your constraints, your definition of "done." Every generation is practice. Every session is a lesson. By session 100, it's not suggesting defaults. It's suggesting what *you* would have chosen, faster than you could have chosen it.

---

## What the First 5 Minutes Feel Like

You open ComfyUI. A small pill button sits in the bottom-right corner. You click it. A panel slides open. No onboarding wizard. No tutorial carousel.

You type: "cinematic portrait, golden hour, film grain."

The agent doesn't ask you to pick a checkpoint. It scans what you have installed, identifies which models can do photorealistic, picks one. It composes a workflow — not from a template library, from capability matching. It shows you the plan: "Flux, 28 steps, DPM++ 2M Karras, film_grain LoRA at 0.3." One button: Generate.

The progress bar moves. Node by node. When it's done, the image appears in ComfyUI's output — where it always appears. Nothing moved. Nothing rearranged. The agent didn't take over your workspace. It just did the setup work you would have done manually in 4 minutes, in 8 seconds.

The wow isn't the image. The wow is that you never left your workflow.

---

## What Session 50 Feels Like

You type: "same vibe, but cooler tones."

The agent doesn't ask what "same vibe" means. It has 49 sessions of context. It knows "same vibe" means Flux, DPM++ 2M, film_grain LoRA — but it also knows your last three sessions trended toward lower CFG and longer step counts. It predicts: "Adjusting color temperature via prompt. Predicted quality: 84% based on 12 similar generations."

You generate. It's right. Not because it's smart — because it's practiced.

You switch to GRAPH mode. The workflow inspector shows four delta layers. Two are yours (Local). One came from experience (Inherits — the agent learned that DPM++ outperforms Euler for your portrait work). One is a safety constraint (your LoRA weight was creeping past the point where artifacts appear). Every opinion is labeled. Every decision is reversible.

You didn't configure any of this. It emerged.

---

## What the Overnight Loop Feels Like

Before bed, you type: "optimize this portrait workflow. Run overnight. Ratchet only."

Morning. Coffee. You open the autoresearch monitor. 18 iterations ran. Quality climbed from 0.62 to 0.89. The agent found that reducing CFG from 8.0 to 7.2 and switching to karras scheduler eliminated a subtle banding artifact you hadn't consciously noticed but had been rejecting images over for weeks.

The quality trajectory chart shows the ratchet: steady climb, plateau, locked. It never went backward. Six variants were kept. Twelve were discarded. Every discard is logged with the reason.

One button: "Apply winning parameters." Your workflow is updated. The delta layer is tagged as experience-derived. Tomorrow's predictions will be calibrated against tonight's discoveries.

The agent didn't create art. It eliminated variables. You create art faster because the parameter space is smaller and better-mapped.

---

## What Failure Feels Like

The generation fails. CUDA out of memory. The agent doesn't show a stack trace. It says: "Generation failed — not enough VRAM for this workflow at 1024x1024. Two options: reduce to 768x768 (predicted quality: still 0.79) or switch to the pruned checkpoint (saves 2GB, predicted quality: 0.81)."

You pick one. It applies the fix as a Safety delta — strongest opinion tier. If the workflow somehow gets those parameters back through an experience layer or a local edit, the safety constraint still wins. Resolution.

The agent remembers that this checkpoint at this resolution OOMs on your hardware. It won't suggest it again. Not because someone wrote a rule. Because it has experience.

---

## What This Product Does Not Do

- **Not a prompt engineer.** It doesn't help you write better prompts. It helps you execute the vision you already have.
- **Not a social generator.** No "share to community." No upvote system. No trending page. This is a production tool, not a content platform.
- **Not a ComfyUI tutorial.** It doesn't teach you nodes. If you need to learn ComfyUI, learn ComfyUI. This assumes you know what you're doing and makes you faster at doing it.
- **Not a replacement for artistic judgment.** The agent suggests. The artist decides. The only exception is safety — structural constraints that prevent wasted compute on known-degenerate configurations.
- **Not autonomous by default.** Every autonomous behavior (overnight runs, auto-retry, experience-derived mutations) is explicitly requested. The agent never changes your workflow without you asking it to. The line between "helpful" and "presumptuous" is consent.

---

## The Body Metaphor

The Scaffolded Brain isn't a metaphor. It's the architecture.

**The graph is the nervous system.** Every workflow is a living graph with non-destructive delta layers. Mutations compose, not overwrite. History is structural, not a log file. You can query the state of your workflow at any point in time, because the graph remembers every opinion that shaped it.

**Experience is the memory.** Every generation is an experiment with a typed result. Parameters, outcome, quality score, context signature. The memory decays — recent experience matters more than old experience. But the patterns extracted from memory persist. The agent doesn't remember every image. It remembers what works.

**The CWM is the imagination.** Before executing, the agent imagines the outcome. Not hallucination — structured prediction from accumulated evidence. "Based on 12 similar generations, this configuration produces 82% quality." When it's wrong, it records the error and calibrates. The imagination gets more accurate through practice.

**LIVRPS is how competing knowledge resolves.** Your edit says CFG 9. Experience says CFG 7.5 works better for this model. A safety constraint says CFG above 30 is degenerate. Who wins? LIVRPS resolves it: Safety overrides everything, then your local edits, then experience, then inherited defaults. Every conflict has a deterministic resolution. Every resolution is transparent and reversible.

**It gets better through practice, not programming.** No one tunes the model. No one writes rules. The architecture creates the conditions for learning, and the learning happens through use. Session 1 is a capable tool. Session 100 is a capable tool that knows you.

---

**GATE: Draft complete. Awaiting Creative Director review.**
