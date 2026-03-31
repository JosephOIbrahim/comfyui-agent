# PHASE 7: CREATIVE DIRECTION

## Product Vision + System Design + Visual Design
**Append to:** SCAFFOLDED_BRAIN_PLAN.md
**Role:** Joe Ibrahim as Creative Director. Claude Code as execution partner.
**Agent Configuration:** New domain expert — `DESIGN`

---

## NEW DOMAIN EXPERT

Add to the MOE routing matrix:

```
┌──────────────┬──────────────────────────────────────────────┐
│ DESIGN       │ Product vision, UI/UX, visual language,      │
│              │ interaction design, system design review.     │
│              │ Authority: what the product FEELS like,       │
│              │ what the user SEES, what the creative         │
│              │ director APPROVES.                            │
│              │ Frozen boundary: Pentagram design language.   │
│              │ Note: This expert serves the Creative         │
│              │ Director (Joe). It proposes; Joe decides.     │
└──────────────┴──────────────────────────────────────────────┘
```

Phase 7 runs three sub-phases in sequence. Each has its own SCOUT → ARCHITECT → GATE → FORGE → CRUCIBLE pipeline.

---

## 7A: PRODUCT VISION (What Should This Feel Like?)

**Agent Configuration:** `[DESIGN × ARCHITECT]` — Joe as Creative Director

This is not engineering. This is the creative director defining what the product IS from the user's perspective. The output is a product vision document that every subsequent design and engineering decision references.

### Questions the Vision Must Answer

**Identity:**
- What is this product? Not technically — emotionally. What's the elevator pitch to a VFX artist who's never used AI generation tools?
- What separates this from every other ComfyUI wrapper, agent, or automation tool?
- The answer should be one sentence. Everything else flows from it.

**Experience:**
- What does the first 5 minutes feel like? (Cold start → first generation → first "wow" moment)
- What does session 50 feel like? (The agent knows your style. It predicts before you ask.)
- What does the overnight loop feel like? (Morning review of what the agent discovered autonomously)
- What does failure feel like? (Something broke. How does the agent communicate and recover?)

**Audience:**
- Primary: VFX professionals who use ComfyUI for production work (Joe's peers)
- Secondary: Technical artists exploring AI generation pipelines
- Tertiary: AI researchers interested in cognitive architectures applied to creative tools
- What does each audience need from the interface that the others don't?

**Boundaries:**
- What does this product explicitly NOT do? (Not a prompt engineering tool. Not a social image generator. Not a ComfyUI tutorial.)
- Where is the line between autonomous and presumptuous? (The agent suggests, but the artist decides — except when safety overrides.)

### Output
`PRODUCT_VISION.md` — one document, max 2 pages. Dense. Every sentence earns its place. This is the creative brief that the UI/UX phase builds against.

### Gate
Joe reviews and approves the vision before any visual design begins.

---

## 7B: SYSTEM DESIGN REVIEW (Architecture Validation)

**Agent Configuration:** All domain experts × `[SCOUT]` + `[DESIGN × ARCHITECT]`

Now that Phases 1-6 are built, this is the structured review pass. Every domain expert scouts their track and surfaces what's working, what's rough, and what needs refinement.

### Review Dimensions

**1. Composition Integrity (GRAPH expert)**
- Is LIVRPS resolution producing correct results in real workflows? 
- Are delta layers accumulating correctly across long sessions?
- Is SHA-256 verification catching real tamper scenarios?
- Are there edge cases in link preservation that the CRUCIBLE didn't cover?

**2. Experience Quality (EXPERIENCE expert)**
- Are ExperienceChunks capturing enough signal to enable prediction?
- Is the ContextSignature discretization too coarse or too fine?
- Is temporal decay rate tuned correctly? (Too fast = amnesia. Too slow = noise.)
- Are the three learning phases transitioning at the right thresholds?

**3. Prediction Accuracy (PREDICTION expert)**
- Is the CWM producing useful predictions after 100+ generations?
- Is the Simulation Arbiter surfacing predictions at the right frequency?
- Are counterfactuals actually getting validated against future sessions?
- Is the dual-function LIVRPS unity (state + prediction) working cleanly?

**4. Transport Reliability (TRANSPORT expert)**
- Is the schema cache staying in sync with ComfyUI after custom node installs?
- Is WebSocket monitoring reliable during long generation sessions?
- Are interrupt + retry working for mid-execution corrections?
- Is resource-aware scheduling making good VRAM decisions?

**5. Autonomous Pipeline (AUTONOMY expert)**
- Does the full loop (intent → compose → predict → execute → evaluate → learn) work end-to-end?
- Is the autoresearch ratchet producing measurable quality improvements?
- Is style-locked series generation maintaining consistency?
- Where does the pipeline still need human intervention that it shouldn't?

**6. Scaffold Integrity (SCAFFOLD expert)**
- Are all 429+ original tests still passing?
- Are the PILOT wrappers transparent to existing consumers?
- Is session persistence working with the CognitiveGraphEngine state?
- Are MCP clients getting the same behavior they got before the rebuild?

### Output
`SYSTEM_DESIGN_REVIEW.md` — findings per dimension, with three buckets:
- **SOLID** — working correctly, no changes needed
- **ROUGH** — working but needs refinement (specific suggestions)
- **BROKEN** — not working as designed (specific blockers)

### Gate
Joe reviews findings. Rough items become Phase 8 tasks. Broken items become immediate fixes.

---

## 7C: VISUAL DESIGN (The Pentagram Phase)

**Agent Configuration:** `[DESIGN × ARCHITECT]` → GATE → `[DESIGN × FORGE]` → `[DESIGN × CRUCIBLE]`

This is where the cognitive architecture gets a face. The design language is established. Joe's Pentagram reference is the north star.

### Design Language (Established — From Prior Work)

```
PALETTE
  Background:     #0D0D0D (near-black)
  Surface:        #1A1A1A (cards, panels)
  Border:         #2A2A2A (1px, subtle)
  Text Primary:   #F0F0F0 (high contrast)
  Text Secondary: #888888 (muted)
  Accent:         #0066FF (electric blue — one accent only)
  Success:        #00CC66 (green, sparingly)
  Warning:        #FF9900 (amber, sparingly)
  Danger:         #FF3344 (red, sparingly)

TYPOGRAPHY
  Font:           Inter or Helvetica Neue
  Hierarchy:      Weight and size only. No color variation for hierarchy.
  Body:           14px / 1.5 line-height
  Monospace:      JetBrains Mono for code/params

PRINCIPLES
  - No gradients. No drop shadows. No rounded corners over 4px.
  - Thin 1px borders. High contrast text.
  - Minimal chrome — content fills the space.
  - Animation: opacity and transform only, 200ms ease-out, never bouncy.
  - Information density over decoration.
  - Every pixel earns its place.
```

### Component 1: SuperDuper Panel (ComfyUI Extension)

The collapsible right sidebar inside ComfyUI. Two modes:

**APP Mode (Chat)**
```
┌─────────────────────────────────┐
│ [APP] [GRAPH]          [─] [×] │
├─────────────────────────────────┤
│                                 │
│  User: cinematic portrait,      │
│  golden hour, film grain        │
│                                 │
│  Agent: Composing workflow      │
│  from experience...             │
│                                 │
│  ┌─ 🔧 compose_workflow ─────┐ │
│  │  Model: Flux (capability   │ │
│  │  match: photorealistic)    │ │
│  │  LoRA: film_grain @ 0.3   │ │
│  │  CFG: 7.5 (experience:    │ │
│  │  best for cinematic)       │ │
│  └────────────────────────────┘ │
│                                 │
│  Prediction: 82% quality        │
│  confidence. Generating...      │
│                                 │
│  ████████████░░░ 73%  Node 5/8  │
│                                 │
├─────────────────────────────────┤
│ [                         ] [→] │
└─────────────────────────────────┘
```

- Messages render markdown
- Tool calls as collapsed cards (expand to see params)
- Streaming responses
- Generation progress bar with node-level tracking
- Prediction confidence shown inline
- Shift+Enter multiline, Enter to send

**GRAPH Mode (Workflow Inspector)**
```
┌─────────────────────────────────┐
│ [APP] [GRAPH]          [─] [×] │
├─────────────────────────────────┤
│                                 │
│ WORKFLOW STATE                  │
│ ──────────────────────────────  │
│ Layers: 4 (2L, 1I, 1S)         │
│ Integrity: ✓ verified           │
│                                 │
│ ┌ 3: KSampler ───────────────┐ │
│ │ seed     156680... [edit]   │ │
│ │ steps    20        [edit]   │ │
│ │ cfg      7.5 (I)   [edit]  │ │
│ │ sampler  euler_a   [edit]   │ │
│ │ model    ← [4] CheckpointLo│ │
│ └─────────────────────────────┘ │
│                                 │
│ ┌ 4: CheckpointLoaderSimple ─┐ │
│ │ ckpt     flux1-dev  [edit]  │ │
│ └─────────────────────────────┘ │
│                                 │
│ DELTA HISTORY                   │
│ ──────────────────────────────  │
│ ▸ layer_003 (S) VRAM cap       │
│ ▸ layer_002 (I) experience cfg │
│ ▸ layer_001 (L) user edit seed │
│ ▸ layer_000 (L) user edit steps│
│                                 │
│ [Rollback to...] [Compare...]   │
└─────────────────────────────────┘
```

- Live workflow state from CognitiveGraphEngine
- Inline param editing → pushes delta layers
- Opinion level shown per param (L/I/V/R/P/S)
- Delta history with rollback controls
- Connection visualization (which nodes feed which)
- Click a node → highlights in ComfyUI's canvas

**The Pill Button**
```
 ┌──────┐
 │ ⚡SD │  ← Bottom-right floating pill. Opens/closes panel.
 └──────┘     Subtle pulse animation when agent has a prediction to surface.
```

### Component 2: Experience Dashboard

A dedicated view (accessible from APP mode: `/experience` command or tab) showing what the agent has learned.

```
┌─────────────────────────────────────────────────────────┐
│ EXPERIENCE DASHBOARD                         [Export]   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ LEARNING PHASE: Phase 2 — Blended (67 generations)      │
│ ████████████████████████████░░░░░░ 67/100 → Phase 3     │
│                                                         │
│ TOP PATTERNS LEARNED                                     │
│ ───────────────────────────────────────────────────────  │
│ 1. Flux + film_grain @ 0.3 → cinematic (92% confidence) │
│ 2. DPM++ 2M > Euler for architecture (78% confidence)   │
│ 3. CFG > 10 + SDXL → oversaturation (85% confidence)    │
│                                                         │
│ PREDICTION ACCURACY (last 20 generations)                │
│     ·  · ·                                               │
│   ·       · ·  · · ·                                     │
│  ·              ·     · · · ·  ← trending up             │
│ ─────────────────────────────                            │
│ 0.4            0.6           0.82                        │
│                                                         │
│ COUNTERFACTUALS                                          │
│ ───────────────────────────────────────────────────────  │
│ 3 pending validation │ 7 validated │ 2 disconfirmed      │
│                                                         │
│ RECENT GENERATIONS                                       │
│ ───────────────────────────────────────────────────────  │
│ [thumb] cfg:7.5 steps:20 flux  │ accepted  │ 0.84 qual  │
│ [thumb] cfg:9.0 steps:25 sdxl  │ iterated  │ 0.71 qual  │
│ [thumb] cfg:7.5 steps:30 flux  │ accepted  │ 0.88 qual  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Component 3: Autoresearch Monitor

The overnight view. What did the agent do while you were sleeping?

```
┌─────────────────────────────────────────────────────────┐
│ AUTORESEARCH: "cinematic portrait optimization"          │
│ Status: COMPLETE │ 18 iterations │ 3h 42m │ Ratchet: ✓  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ QUALITY TRAJECTORY                                       │
│ 0.9 ┤                              ●───●───● locked     │
│ 0.8 ┤              ●───●───●──●                          │
│ 0.7 ┤     ●───●──●                                      │
│ 0.6 ┤ ●──●                                              │
│     └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬  │
│        1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 18  │
│                                                         │
│ WINNING PARAMETERS (vs starting)                         │
│ ───────────────────────────────────────────────────────  │
│ CFG:        8.0 → 7.2  (↓ reduced saturation)           │
│ Sampler:    euler → dpmpp_2m  (↑ detail)                 │
│ Steps:      20 → 28  (↑ quality ceiling)                 │
│ LoRA wt:    0.5 → 0.32  (↓ subtlety)                    │
│ Scheduler:  normal → karras  (↑ contrast)                │
│                                                         │
│ VARIANTS TRIED: 18 │ KEPT: 6 │ DISCARDED: 12            │
│                                                         │
│ [View all variants] [Apply winning params] [New run...]  │
└─────────────────────────────────────────────────────────┘
```

### Component 4: Prediction Overlay

When the Simulation Arbiter surfaces a Soft or Explicit prediction, it appears as an inline card in APP mode:

```
┌─ PREDICTION (78% confidence) ─────────────────────┐
│                                                    │
│  Based on 12 similar generations:                  │
│  Reducing CFG to 7.0 would improve skin tones.     │
│                                                    │
│  Forward paths:                                    │
│  ► Apply CFG 7.0  → predicted quality: 0.86        │
│  ► Keep CFG 9.0   → predicted quality: 0.72        │
│  ► Try CFG 6.0    → predicted quality: 0.81        │
│                                                    │
│  [Apply recommended] [Ignore] [Show evidence]      │
└────────────────────────────────────────────────────┘
```

### Technical Implementation

**ComfyUI Extension Format:**
```
G:\COMFYUI_Database\Custom_Nodes\ComfyUI-SuperDuper-Panel\
├── __init__.py            # ComfyUI extension registration
├── web/
│   └── js/
│       ├── superduperPanel.js    # Main panel logic
│       ├── appMode.js            # Chat interface
│       ├── graphMode.js          # Workflow inspector
│       ├── experienceDash.js     # Experience dashboard
│       ├── autoresearchMonitor.js # Overnight results
│       └── styles.css            # Pentagram design system
└── README.md
```

**Communication:**
- Panel ↔ Agent: HTTP to localhost (the MCP server or thin REST wrapper)
- Panel ↔ ComfyUI: LiteGraph API (`app.graph`) for reading/writing graph state
- Panel ↔ CognitiveGraphEngine: via agent REST endpoints for delta layer state

**Rules:**
- No React. Vanilla JS + CSS. Matches ComfyUI's extension pattern.
- No external dependencies. Everything self-contained.
- LocalStorage for chat history persistence.
- SSE or polling for streaming responses.

### Design Crucible

**Agent Configuration:** `[DESIGN × CRUCIBLE]`

The design crucible doesn't test code — it tests the design against the product vision:

1. **First-use test:** Can a new user understand what to do in 30 seconds?
2. **Information density test:** Is every element on screen earning its space?
3. **Pentagram test:** Would Pentagram ship this? No gradients? No decorative elements? High contrast? Minimal chrome?
4. **Mobile/compressed test:** Does GRAPH mode work when the panel is narrow?
5. **Flow state test:** During rapid generation (burst mode), does the UI stay out of the way?
6. **Prediction trust test:** Does the prediction overlay build confidence or create anxiety?
7. **Overnight test:** Is the autoresearch monitor clear enough to review in 2 minutes over morning coffee?

---

## Phase 7 Complete Criteria

```
ARTIFACTS:
  □ PRODUCT_VISION.md — one-sentence identity + experience definition
  □ SYSTEM_DESIGN_REVIEW.md — per-dimension findings (SOLID/ROUGH/BROKEN)
  □ ComfyUI-SuperDuper-Panel extension — installed, functional
  □ Design system documented in styles.css with variables

VERIFICATION:
  □ Panel loads in ComfyUI without errors
  □ APP mode: chat works, tool cards expand, streaming renders
  □ GRAPH mode: nodes display, inline edit pushes delta layers
  □ Experience dashboard renders from real ExperienceChunk data
  □ Autoresearch monitor renders from real autoresearch results
  □ Prediction overlay appears on Soft/Explicit arbiter decisions
  □ All 429+ original tests still pass
  □ Pentagram design test passes (no gradients, no shadows, minimal chrome)
```

---

## ROUTING UPDATE

Add to the Claude Code system prompt:

```
Phase 7 introduces the DESIGN domain expert.
When working on visual design, product vision, or UI/UX:
  State: [DESIGN × ROLE] at the top of your work.
  
The Creative Director (Joe) has final authority on all design decisions.
The DESIGN expert proposes; Joe approves. No design ships without
explicit Creative Director approval at the human gate.

Design language reference: Pentagram-inspired.
  - Monochrome + one accent (#0066FF on #0D0D0D)
  - Inter typography. No gradients. No shadows. 4px max radius.
  - 1px borders. High contrast. Content fills space.
  - Animation: opacity + transform, 200ms ease-out, never bouncy.
```
