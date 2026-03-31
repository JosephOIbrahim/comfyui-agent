# PANEL DESIGN — Component Architecture

**Agent:** `[DESIGN x ARCHITECT]`
**Status:** DRAFT — Awaiting Creative Director approval before FORGE

---

## Context

An existing `ComfyUI-SuperDuper` extension provides a working chat sidebar with WebSocket streaming, quick actions, readability controls, and a v2 design system using ComfyUI's native blue (#4e9bcd). This new extension (`ComfyUI-SuperDuper-Panel`) is a ground-up rebuild with:

- The Pentagram design spec (#0066FF on #0D0D0D)
- GRAPH mode (workflow inspector with delta layers)
- Experience Dashboard
- Autoresearch Monitor
- Prediction Overlay
- Session-aware communication to the cognitive layer

The existing extension continues to work. This one runs alongside it or replaces it.

---

## File Structure

```
G:\COMFYUI_Database\Custom_Nodes\ComfyUI-SuperDuper-Panel\
├── __init__.py                    # WEB_DIRECTORY = "./web"
├── web/
│   └── js/
│       ├── superduperPanel.js     # Entry: extension registration, pill button, panel shell
│       ├── appMode.js             # APP mode: chat, streaming, tool cards
│       ├── graphMode.js           # GRAPH mode: workflow inspector, delta history
│       ├── experienceDash.js      # Experience dashboard view
│       ├── autoresearchMonitor.js # Autoresearch results view
│       ├── predictionOverlay.js   # Inline prediction cards
│       ├── agentClient.js         # HTTP/SSE client for agent communication
│       └── styles.css             # Pentagram design system
```

8 files. No build step. No external dependencies. Every file loadable by ComfyUI's static serving.

---

## Design System (styles.css)

```css
:root {
  /* Pentagram Palette */
  --p-bg:           #0D0D0D;
  --p-surface:      #1A1A1A;
  --p-border:       #2A2A2A;
  --p-text:         #F0F0F0;
  --p-text-muted:   #888888;
  --p-accent:       #0066FF;
  --p-accent-dim:   rgba(0, 102, 255, 0.15);
  --p-success:      #00CC66;
  --p-warning:      #FF9900;
  --p-danger:       #FF3344;

  /* Typography */
  --p-sans:         'Inter', -apple-system, system-ui, sans-serif;
  --p-mono:         'JetBrains Mono', 'Consolas', monospace;
  --p-body:         14px;
  --p-body-sm:      12px;
  --p-heading:      14px;     /* Same size, different weight */
  --p-caption:      11px;
  --p-lh:           1.5;

  /* Spacing (4px base) */
  --p-1: 4px;  --p-2: 8px;  --p-3: 12px;  --p-4: 16px;
  --p-5: 20px; --p-6: 24px; --p-8: 32px;

  /* Radius */
  --p-radius:       2px;      /* Max 4px. Prefer 2. */

  /* Motion */
  --p-ease:         200ms ease-out;

  /* LIVRPS opinion colors */
  --p-opinion-P:    #555555;
  --p-opinion-R:    #777777;
  --p-opinion-V:    #9966CC;
  --p-opinion-I:    #0066FF;
  --p-opinion-L:    #F0F0F0;
  --p-opinion-S:    #FF3344;
}
```

**Rules enforced in CSS:**
- `border-radius` never exceeds `4px`
- `box-shadow: none` on everything
- `background-image: none` on everything (no gradients)
- `transition` uses only `opacity` and `transform`, 200ms ease-out
- `font-weight` and `font-size` are the only hierarchy tools

---

## Component 1: Panel Shell (superduperPanel.js)

### Registration
```javascript
import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "superduper.panel",
  async setup() {
    createPillButton();
    createPanelShell();
  }
});
```

### Pill Button
- Position: fixed, bottom-right, `right: 20px; bottom: 20px`
- Content: "SD" in monospace, 14px, `--p-accent` on `--p-surface`
- Border: 1px `--p-border`
- Size: 48x28px, `border-radius: 14px` (exception: the pill is a pill)
- Click: toggles panel visibility (opacity + translateX, 200ms)
- Pulse: when arbiter has a Soft/Explicit prediction pending, a subtle 2s opacity pulse (0.7 → 1.0) on the accent background. Stops when panel is opened.

### Panel Shell
- Position: fixed right sidebar, `width: 380px; height: 100vh`
- Background: `--p-bg`
- Border-left: 1px `--p-border`
- Z-index: 1000 (above ComfyUI canvas, below modals)
- Flex column layout: header → content → input bar

### Header
```
┌──────────────────────────────────────┐
│ APP  GRAPH                    ─   ×  │
└──────────────────────────────────────┘
```
- Two mode tabs: APP, GRAPH. Active tab: `--p-text` + 2px bottom border in `--p-accent`. Inactive: `--p-text-muted`.
- Minimize: hides to pill. Close: same.
- Font: `--p-sans`, 12px, weight 600, uppercase, `letter-spacing: 0.08em`
- Background: `--p-surface`
- Border-bottom: 1px `--p-border`
- Height: 40px

### Content Area
- Flex: `1 1 auto; overflow-y: auto`
- Scrollbar: thin, `--p-border` thumb on `--p-bg` track
- Switches between APP and GRAPH mode content
- Transition: opacity crossfade, 200ms

### Resize
- Left edge drag handle (4px wide, cursor: col-resize)
- Min width: 300px, max width: 600px
- Persisted to localStorage

---

## Component 2: APP Mode (appMode.js)

### Message Stream
Each message is a `div.sd-msg` with role class:

```
.sd-msg--user     background: transparent, text: --p-text
.sd-msg--agent    background: transparent, text: --p-text
.sd-msg--system   text: --p-text-muted, font-size: --p-body-sm
```

- No bubbles. No colored backgrounds. Messages differentiated by label only.
- Label: "You" / "Agent" in `--p-text-muted`, `--p-caption`, uppercase, `letter-spacing: 0.06em`, above message body
- Body: `--p-sans`, `--p-body`, `--p-lh`
- Agent body renders markdown: **bold**, *italic*, `code`, ```code blocks```, lists
- Code blocks: `--p-mono`, `--p-body-sm`, background `--p-surface`, 1px `--p-border`, `border-radius: 2px`

### Tool Cards
When agent calls a tool (e.g., `compose_workflow`), render as a collapsible card:

```
┌─ compose_workflow ──────────────────┐
│ Model: Flux                         │
│ CFG: 7.5 (experience-derived)       │
│ Steps: 28                           │
└─────────────────────────────────────┘
```

- Default: collapsed (shows tool name only as a single line)
- Click: expands to show key-value parameters
- Header: `--p-mono`, `--p-body-sm`, `--p-text-muted`
- Border: 1px `--p-border`, `border-radius: 2px`
- Background: `--p-surface`
- No icons. Tool name is the only identifier. Tool name in the header is left-aligned, chevron (▸/▾) right-aligned.

### Streaming
- Agent text arrives as SSE deltas (or polling fallback)
- Text appended character-by-character to current message body
- Typing indicator: three dots pulsing opacity (0.3 → 1.0, staggered 100ms), `--p-text-muted`
- No cursor animation. No skeleton loading.

### Progress Bar
- Appears during execution. Below the last message, above the input bar.
- Full-width bar, 2px height, `--p-accent` fill on `--p-border` track
- Text below: "Node 5/8 — KSampler" in `--p-mono`, `--p-caption`, `--p-text-muted`
- Percentage: right-aligned, `--p-text`

### Input Bar
```
┌──────────────────────────────────────┐
│ [                              ] [→] │
└──────────────────────────────────────┘
```
- Input: `--p-surface` background, 1px `--p-border`, `border-radius: 2px`
- Placeholder: "Ask about your workflow..." in `--p-text-muted`
- Send button: `→` character, `--p-accent`, no background, no border
- Enter to send, Shift+Enter for newline (textarea, auto-growing, max 4 lines)
- When busy: input disabled, send shows "..." in `--p-text-muted`

### Chat History
- Stored in localStorage key `sd-panel-history` as JSON array
- Max 100 messages retained
- Loaded on panel open, appended on new messages

---

## Component 3: GRAPH Mode (graphMode.js)

### Workflow State Header
```
WORKFLOW STATE
────────────────────────────────
Layers: 4 (2L, 1I, 1S)
Integrity: ✓ verified
```

- Section label: `--p-mono`, `--p-caption`, uppercase, `letter-spacing: 0.08em`, `--p-text-muted`
- Rule line: 1px `--p-border`, full width
- Stats: `--p-sans`, `--p-body-sm`. Layer counts by opinion. Integrity check result.
- Integrity icon: ✓ in `--p-success` when intact, ✗ in `--p-danger` when tampered

### Node Cards
Each workflow node renders as a card:

```
┌ 3: KSampler ──────────────────────┐
│ seed     156680...          [edit] │
│ steps    20                 [edit] │
│ cfg      7.5 (I)            [edit] │
│ sampler  euler_a            [edit] │
│ model    ← [1] CheckpointLoader   │
└───────────────────────────────────┘
```

- Card header: node_id + class_type, `--p-mono`, `--p-body-sm`, weight 600
- Background: `--p-surface`, border: 1px `--p-border`, `border-radius: 2px`
- Param rows: grid layout, 3 columns:
  - Name: `--p-mono`, `--p-body-sm`, `--p-text-muted`, right-aligned
  - Value: `--p-mono`, `--p-body-sm`, `--p-text`. Long values truncated with `...`
  - Action: `[edit]` link in `--p-accent`, `--p-caption` or connection indicator
- Opinion badge: `(L)`, `(I)`, `(S)` etc. inline after value, colored by `--p-opinion-*` vars
- Connection indicator: `← [node_id] ClassName` in `--p-text-muted` for link inputs

### Inline Editing
- Click `[edit]`: value becomes a text input, same size, `--p-surface` background, 1px `--p-accent` border (focus ring)
- Enter: commits edit → sends `set_input` to agent → pushes delta layer → re-renders card
- Escape: cancels edit
- During edit: all other edit buttons hidden (one edit at a time)

### Delta History
```
DELTA HISTORY
────────────────────────────────
▸ layer_003 (S) VRAM cap
▸ layer_002 (I) experience cfg
▸ layer_001 (L) user edit seed
▸ layer_000 (L) user edit steps
```

- Each row: collapsible. Click to show mutations dict.
- Opinion badge: colored by `--p-opinion-*`
- Description: `--p-sans`, `--p-body-sm`, `--p-text`
- Timestamp: right-aligned, `--p-caption`, `--p-text-muted`
- Expand: shows `{node_id: {param: value}}` as formatted JSON in `--p-mono`

### Rollback Controls
- "Rollback to..." button: opens a list of delta layers. Click one to `temporal_query(back_steps=N)`.
- "Compare..." button: shows side-by-side diff of current vs selected historical state
- Both buttons: `--p-text-muted`, `--p-body-sm`, `border: 1px --p-border`, `border-radius: 2px`, hover: `--p-accent` text

### Data Source
- Fetches from agent via HTTP: `GET /superduper-panel/graph-state`
- Response includes: `base_node_count`, `delta_stack` (array of delta metadata), `resolved_workflow` (current state), `integrity_check`
- Polls every 2s when GRAPH mode is active. Stops when switched to APP.

---

## Component 4: Experience Dashboard (experienceDash.js)

Accessible from APP mode via `/experience` command or a tab at the top.

### Layout (single scrollable column)

**Section 1: Learning Phase**
```
LEARNING PHASE
────────────────────────────────
Phase 2 — Blended (67 generations)
████████████████████████░░░░░░░░ 67/100 → Phase 3
```
- Progress bar: 4px height, `--p-accent` fill on `--p-border` track
- Phase name: `--p-sans`, `--p-body`, weight 600
- Count: `--p-mono`, `--p-body-sm`, `--p-text-muted`

**Section 2: Top Patterns**
```
TOP PATTERNS
────────────────────────────────
Flux + film_grain @ 0.3 → cinematic   92%
DPM++ 2M > Euler for architecture     78%
CFG > 10 + SDXL → oversaturation      85%
```
- Each row: pattern description left, confidence percentage right
- Description: `--p-sans`, `--p-body-sm`, `--p-text`
- Confidence: `--p-mono`, `--p-body-sm`, right-aligned. Color: `--p-success` if >= 80%, `--p-warning` if 60-79%, `--p-text-muted` if < 60%

**Section 3: Prediction Accuracy**
```
PREDICTION ACCURACY
────────────────────────────────
[ASCII sparkline or dot plot]
0.4          0.6          0.82
```
- Canvas element, 100% width, 60px height
- Dots plotted as filled circles (2px radius, `--p-accent`)
- Connecting line: 1px `--p-border`
- X-axis: generation index (last 20). Y-axis: 0.0 - 1.0
- Trend label: "trending up" / "stable" / "trending down" in `--p-text-muted`, right-aligned

**Section 4: Counterfactuals**
```
COUNTERFACTUALS
────────────────────────────────
3 pending  ·  7 validated  ·  2 disconfirmed
```
- Single line of stats. Numbers in `--p-text`, labels in `--p-text-muted`
- Dot separator: `·` in `--p-text-muted`

**Section 5: Recent Generations**
```
RECENT GENERATIONS
────────────────────────────────
cfg:7.5 steps:20 flux    accepted  0.84
cfg:9.0 steps:25 sdxl    iterated  0.71
cfg:7.5 steps:30 flux    accepted  0.88
```
- No thumbnails (this is a data panel, not a gallery)
- Each row: params left, status center, quality right
- Params: `--p-mono`, `--p-body-sm`, `--p-text`
- Status: `--p-sans`, `--p-body-sm`. "accepted" in `--p-success`, "iterated" in `--p-warning`, "rejected" in `--p-danger`
- Quality: `--p-mono`, `--p-body-sm`

### Data Source
- `GET /superduper-panel/experience`
- Response: `{learning_phase, generation_count, threshold, patterns, accuracy_history, counterfactual_stats, recent_generations}`
- Fetched on view open. No polling (manual refresh button at top-right).

---

## Component 5: Autoresearch Monitor (autoresearchMonitor.js)

Accessible from APP mode via `/research` command.

### Layout

**Header**
```
AUTORESEARCH: "cinematic portrait optimization"
Status: COMPLETE  ·  18 iterations  ·  3h 42m  ·  Ratchet: ✓
```
- Title: `--p-sans`, `--p-body`, weight 600, `--p-text`
- Stats line: `--p-mono`, `--p-body-sm`, `--p-text-muted`
- Status badge: COMPLETE in `--p-success`, RUNNING in `--p-accent`, FAILED in `--p-danger`

**Quality Trajectory**
- Canvas: 100% width, 120px height
- Line chart: connected dots
- Y-axis: 0.0 - 1.0, labeled at 0.2 intervals, `--p-caption`, `--p-text-muted`
- X-axis: iteration number, `--p-caption`
- Line: 1px `--p-accent`
- Dots: 3px radius, `--p-accent` fill
- "Locked" region (ratchet plateau): line changes to `--p-success`

**Winning Parameters**
```
WINNING PARAMETERS
────────────────────────────────
CFG        8.0 → 7.2    ↓ reduced saturation
Sampler    euler → dpmpp_2m    ↑ detail
Steps      20 → 28      ↑ quality ceiling
```
- Grid: param name, before → after, direction + note
- Arrow direction: `↑` in `--p-success`, `↓` in `--p-warning` (context-dependent — "↓ reduced saturation" is positive)
- Values: `--p-mono`, `--p-body-sm`
- Note: `--p-sans`, `--p-body-sm`, `--p-text-muted`

**Summary**
```
VARIANTS TRIED: 18  ·  KEPT: 6  ·  DISCARDED: 12
```

**Actions**
```
[Apply winning params]   [New run...]
```
- Primary button (Apply): `--p-accent` background, `--p-bg` text, `border-radius: 2px`, height 32px
- Secondary button (New run): transparent background, 1px `--p-border`, `--p-text`, hover: `--p-accent` text

### Data Source
- `GET /superduper-panel/autoresearch`
- Response: `{title, status, iterations, duration_ms, quality_trajectory, winning_params, variants_tried, kept, discarded}`

---

## Component 6: Prediction Overlay (predictionOverlay.js)

Injected inline in APP mode message stream when arbiter decision is Soft or Explicit.

### Card Design
```
┌─ PREDICTION ─────────────────────────┐
│ 78% confidence                       │
│                                      │
│ Based on 12 similar generations:     │
│ Reducing CFG to 7.0 improves skin    │
│ tones.                               │
│                                      │
│ ► Apply CFG 7.0    quality: 0.86     │
│ ► Keep CFG 9.0     quality: 0.72     │
│ ► Try CFG 6.0      quality: 0.81     │
│                                      │
│ [Apply]  [Ignore]  [Evidence]        │
└──────────────────────────────────────┘
```

- Border: 1px `--p-accent` (the only element with accent border)
- Background: `--p-surface`
- Header: "PREDICTION" in `--p-mono`, `--p-caption`, uppercase, `--p-accent`
- Confidence: `--p-mono`, `--p-body-sm`, `--p-text`
- Body text: `--p-sans`, `--p-body-sm`, `--p-text`
- Forward paths: `--p-mono`, `--p-body-sm`. ► prefix in `--p-accent`. Quality value right-aligned.
- Recommended path: weight 600 (bold) to distinguish from alternatives
- Buttons: same pattern as Autoresearch — Apply is primary (`--p-accent` bg), Ignore/Evidence are secondary

### Behavior
- "Apply": sends mutation to agent, collapses card to "Applied CFG 7.0" confirmation line
- "Ignore": collapses to "Ignored" in `--p-text-muted`, records feedback
- "Evidence": expands to show the 12 similar generations (params + quality) as a compact table

### Explicit vs Soft
- Soft: appears in flow, no special treatment. User can scroll past.
- Explicit: pinned to top of message area with a thin `--p-danger` left border (2px). Stays visible until dismissed.

---

## Communication Layer (agentClient.js)

### API Surface

```javascript
class AgentClient {
  constructor(baseUrl = `${location.origin}/superduper-panel`)

  // Chat
  async sendMessage(text)                    // POST /chat → SSE stream

  // Graph state
  async getGraphState()                      // GET /graph-state
  async setInput(nodeId, inputName, value)   // POST /set-input
  async rollback(backSteps)                  // POST /rollback

  // Experience
  async getExperience()                      // GET /experience

  // Autoresearch
  async getAutoresearch()                    // GET /autoresearch
  async applyWinningParams(runId)            // POST /autoresearch/apply

  // Prediction
  async applyPrediction(predictionId, path)  // POST /prediction/apply
  async ignorePrediction(predictionId)       // POST /prediction/ignore
}
```

### Server-Sent Events for Streaming
```javascript
// POST /chat sends message, returns SSE stream
const response = await fetch(url, { method: 'POST', body: JSON.stringify({text}) });
const reader = response.body.getReader();
// Read chunks, parse as SSE: "data: {type, text}" lines
```

### Fallback
If SSE is unavailable (proxy stripping), fall back to polling:
- POST /chat → returns `{id}`
- GET /chat/{id}/status → returns `{complete, messages}`
- Poll every 500ms until complete

---

## Backend Routes (thin REST wrapper)

Added to the existing `ComfyUI-SuperDuper-Panel/__init__.py` → `server/routes.py`:

```python
# Mounted on PromptServer via aiohttp routes
@routes.post('/superduper-panel/chat')          # Forward to agent, return SSE stream
@routes.get('/superduper-panel/graph-state')     # Read CognitiveGraphEngine state
@routes.post('/superduper-panel/set-input')      # Push delta layer via set_input tool
@routes.post('/superduper-panel/rollback')       # temporal_query on engine
@routes.get('/superduper-panel/experience')       # Read ExperienceAccumulator stats
@routes.get('/superduper-panel/autoresearch')     # Read autoresearch results
@routes.post('/superduper-panel/autoresearch/apply')  # Apply winning params
@routes.post('/superduper-panel/prediction/apply')    # Apply prediction suggestion
@routes.post('/superduper-panel/prediction/ignore')   # Record ignored prediction
```

These routes are thin wrappers that call into the existing tool handlers. No new logic — just HTTP ↔ tool_input translation.

---

## State Management

| State | Storage | Lifetime |
|-------|---------|----------|
| Chat history | localStorage `sd-panel-history` | Persists across page loads |
| Panel width | localStorage `sd-panel-width` | Persists |
| Active mode (APP/GRAPH) | localStorage `sd-panel-mode` | Persists |
| Panel open/closed | sessionStorage `sd-panel-open` | Per browser session |
| Pending prediction | In-memory | Cleared on dismiss |
| Graph state cache | In-memory | Refreshed every 2s in GRAPH mode |

---

## Accessibility

- All interactive elements have `aria-label`
- Messages container: `role="log"`, `aria-live="polite"`
- Keyboard: Tab navigates, Enter activates, Escape closes editor/panel
- Focus trap: when panel is open, Tab cycles within panel
- Reduced motion: `@media (prefers-reduced-motion: reduce)` disables all transitions

---

## Performance Constraints

- Panel JS must load in < 100ms (no heavy parsing)
- GRAPH mode poll: 2s interval, abort on mode switch
- Experience dashboard: single fetch on open, no background polling
- Chat history: cap at 100 messages, prune oldest on overflow
- Canvas charts: no animation library. Raw `CanvasRenderingContext2D`.

---

**GATE: Component architecture complete. Awaiting Creative Director review before FORGE implementation.**

### Design Decisions Needing Sign-Off

1. **No thumbnails in Experience Dashboard recent generations.** The spec shows `[thumb]` but I've omitted them — this is a data panel, not a gallery. Thumbnails would be decorative, not informational, at sidebar width. Override?

2. **Pill button exception on border-radius.** The pill shape requires `border-radius: 14px`, breaking the 4px max rule. It's a pill — it should look like a pill. Accept the exception?

3. **Prediction Overlay accent border.** This is the only element with a `--p-accent` colored border, making it visually distinct. The Explicit variant adds a `--p-danger` left stripe. This breaks the "1px `--p-border` everywhere" rule deliberately for attention hierarchy. Accept?

4. **APP mode has no user message bubbles/backgrounds.** Messages are differentiated only by the label ("You" / "Agent") and vertical position. This is maximum Pentagram (content-only, zero decoration). It may feel too sparse. Accept, or add a subtle `--p-surface` background to agent messages?

5. **Existing ComfyUI-SuperDuper relationship.** This new extension runs alongside the old one. They use different WebSocket/HTTP paths (`/superduper/ws` vs `/superduper-panel/*`). No conflict. But: should the old one be deprecated, or do both co-exist long-term?
