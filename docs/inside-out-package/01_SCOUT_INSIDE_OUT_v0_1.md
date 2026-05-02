# [SCAFFOLD × SCOUT] — Inside-Out Reconnaissance Pass

**Type:** SCOUT artifact for Claude Code execution
**Status:** READY (do not run until ARCH-1 council decision is GREEN)
**Target:** Comfy-Cozy + Moneta + comfy-moneta-bridge repos
**Output:** `SCOUT_INSIDE_OUT_v0_1.md` written to Comfy-Cozy repo root

---

## 0. Pre-flight check

**Before running this scout pass, confirm:**

- [ ] ARCH-1 council decision is GREEN (inside-out approved, in whole or in part)
- [ ] ARCH-5 council decision is logged (workflow LIVRPS approved, deferred, or rejected)
- [ ] No active forge work is in flight on Comfy-Cozy `master`

If any of these are unmet, **STOP**. This is not a pre-council pass.

---

## 1. Mission

Read-only inventory of three repos to produce a substrate map showing:

1. **Where outside-in is currently encoded** in Comfy-Cozy (transport assumptions, process boundaries, serialization layers)
2. **Where inside-out would land** (custom node package boundaries, in-process call surfaces, ComfyUI extension hooks)
3. **What dissolves** in the migration (HTTP transport in the bridge, separate process management, REST round-trips for cognitive state)
4. **What survives** the migration (Moneta as substrate repo, MOE constitution, LIVRPS composition, patent claims)
5. **What's at risk** (versioning conflicts with other ComfyUI custom nodes, install-path discipline, Python dependency resolution)

This is **pure SCOUT work. No FORGE actions.** No file modifications. Inventory only.

---

## 2. Repos in scope

| Repo | Path | Role |
|---|---|---|
| Comfy-Cozy | `G:\Comfy-Cozy` | Consumer surface — to migrate |
| Moneta | `C:\Users\User\Moneta` | Substrate — to be consumed in-process |
| comfy-moneta-bridge | `G:\Comfy-Cozy\..\comfy-moneta-bridge` (verify path) | Transport layer — to be reshaped |
| ComfyUI | `G:\COMFY\ComfyUI` | Host environment — for inside-out integration target |

---

## 3. Eight numbered steps

### Step 1 — Inventory the transport boundary in Comfy-Cozy

Search for every place Comfy-Cozy talks to ComfyUI:

```bash
cd G:\Comfy-Cozy
grep -rn "localhost:8188" --include="*.py"
grep -rn "127.0.0.1:8188" --include="*.py"
grep -rn "http://" --include="*.py" | grep -v ".git"
grep -rn "ws://" --include="*.py" | grep -v ".git"
grep -rn "websocket" --include="*.py" | grep -i -v "test"
grep -rn "/object_info" --include="*.py"
grep -rn "/prompt" --include="*.py"
grep -rn "/queue" --include="*.py"
grep -rn "/history" --include="*.py"
```

For each match, record:
- File path
- Line number
- What ComfyUI endpoint is being called
- Whether it's transport-layer code or business-logic code

**Output section in scout doc:** "Transport Surface Inventory"

### Step 2 — Map the comfy-moneta-bridge transport layer

Read `comfy-moneta-bridge` end-to-end. Record:

- HTTP/WebSocket vs in-process call boundaries
- Serialization points (where Pydantic/dataclass conversion happens)
- Async boundaries
- The 49 tests — categorize by what they verify (transport, business logic, contract)

For each transport-layer concern, mark it as:
- `DISSOLVES` — in-process eliminates this
- `RESHAPES` — survives but changes form
- `SURVIVES` — independent of transport choice

**Output section in scout doc:** "Bridge Transport Map"

### Step 3 — Inventory ComfyUI custom node package conventions

Read three reference custom node packages already installed in `G:\COMFY\ComfyUI\custom_nodes\`. Pick the most-starred or best-maintained ones. Record:

- `__init__.py` structure (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, WEB_DIRECTORY)
- Dependency declaration patterns (requirements.txt, pyproject.toml)
- Versioning conventions
- Web extension patterns (vanilla JS, no React — matches Comfy-Cozy Phase 7 design language)
- How they handle ComfyUI version compatibility

**Output section in scout doc:** "Custom Node Package Conventions"

### Step 4 — Map the workflow graph surface

In ComfyUI itself (`G:\COMFY\ComfyUI`), inventory:

- How the workflow JSON is structured (node graph schema)
- Where the in-process Python representation of the graph lives
- The `/object_info` endpoint internals — how nodes register themselves
- Frontend event hooks (graph mutations, node selection, etc.)
- Whether ComfyUI exposes a Python-side API for graph mutation, or whether all mutation goes through the frontend → backend cycle

**Output section in scout doc:** "Workflow Graph Surface"

### Step 5 — Identify Moneta's in-process readiness

Read Moneta's public API (`Moneta/moneta/__init__.py` and exports). Record:

- Which methods assume HTTP transport vs assume in-process consumption
- Whether the SDK already supports being imported and called directly
- usd-core dependency resolution within ComfyUI's Python environment (potential conflict surface)
- Threading model — is Moneta safe to run in the same process as ComfyUI's worker threads?

**Output section in scout doc:** "Moneta SDK Surface"

### Step 6 — Catalogue ComfyUI Python environment risks

Inside-out means Moneta runs in ComfyUI's Python interpreter. Inventory the conflict surface:

- ComfyUI's Python version (check `G:\COMFY\ComfyUI\python_embeded\` if portable, or system Python if installed)
- Pinned dependencies in ComfyUI's requirements.txt
- Conflicts between Moneta's deps (`usd-core`, etc.) and ComfyUI's deps
- Other custom nodes' dependency demands (PyTorch versions, transformers, etc.)

For each conflict, mark severity:
- `HARD` — same package, incompatible versions, no workaround
- `SOFT` — same package, compatible versions, but pinning needs care
- `CLEAN` — no conflict

**Output section in scout doc:** "Dependency Conflict Catalogue"

### Step 7 — Map Frame B and Frame F equivalents

For each of Frame B (prepared decisions in native nodes) and Frame F (Shadow Graph) from the Synapse v1.1 spec, identify the ComfyUI equivalent:

**Frame B for ComfyUI:**
- Hypothesis: "Moneta-aware ComfyUI nodes that read pre-computed assertions at cook time"
- Concrete: which ComfyUI node types could host this? Custom node? Subclass of an existing node?
- Cook-time hook: where in ComfyUI's execution does a node read its pre-computed input?

**Frame F for ComfyUI:**
- Hypothesis: "Cozy Shadow Graph — proposed workflow nodes ghosted in ComfyUI's editor, user commits with a gesture"
- Concrete: how does ComfyUI's frontend express "ghosted" or "uncommitted" node state?
- Commit gesture: right-click menu? keyboard shortcut? frontend API call?
- Reject gesture: Delete key? frontend API call?

**Output section in scout doc:** "Frame B / Frame F ComfyUI Mapping"

### Step 8 — Risk catalogue

List every concrete risk discovered, with severity:

| Severity | Definition |
|---|---|
| 🔴 HARD | Blocks inside-out implementation entirely |
| 🟡 SOFT | Requires care but has known mitigation |
| 🟢 CLEAN | No risk identified |

For each 🔴 HARD risk, propose:
- The smallest viable workaround
- The phase 2 mitigation
- Whether the risk is ARCH-1, ARCH-5, or ARCH-2 territory

**Output section in scout doc:** "Risk Catalogue"

---

## 4. Output document structure

The single deliverable is `SCOUT_INSIDE_OUT_v0_1.md` written to the Comfy-Cozy repo root with this structure:

```markdown
# Inside-Out Scout — Comfy-Cozy × Moneta × ComfyUI

**Date:** <run date>
**Branch:** <git branch>
**HEAD:** <commit SHA>

## Executive Summary
<3-5 sentence summary of inside-out feasibility>

## 1. Transport Surface Inventory (from Step 1)
<table or list>

## 2. Bridge Transport Map (from Step 2)
<categorized list with dissolves/reshapes/survives>

## 3. Custom Node Package Conventions (from Step 3)
<3 reference packages, conventions extracted>

## 4. Workflow Graph Surface (from Step 4)
<schema + mutation API map>

## 5. Moneta SDK Surface (from Step 5)
<in-process readiness assessment>

## 6. Dependency Conflict Catalogue (from Step 6)
<table with severity>

## 7. Frame B / Frame F ComfyUI Mapping (from Step 7)
<two subsections>

## 8. Risk Catalogue (from Step 8)
<severity-sorted list>

## Recommendations for ARCH-1 Phase 2 Plan
<concrete next moves derived from inventory>
```

---

## 5. Constraints

- **NO file modifications.** Read-only across all repos.
- **NO test runs.** This is inventory, not verification.
- **NO architectural proposals beyond the brief.** If the scout discovers something not covered by ARCH-1/ARCH-5/ARCH-2, raise a `FLAGS/flag_scout_<n>.md` and continue.
- **NO speculation about Synapse.** This scout is Comfy-Cozy + Moneta + comfy-moneta-bridge + ComfyUI only.

---

## 6. Exit criteria

- [ ] All 8 steps completed
- [ ] `SCOUT_INSIDE_OUT_v0_1.md` written and committed (read-only addition; no other file changes)
- [ ] Risk catalogue surfaces at least one 🔴 HARD risk OR explicitly states "no HARD risks identified"
- [ ] Executive summary states whether inside-out is FEASIBLE / FEASIBLE-WITH-CAVEATS / BLOCKED

If any 🔴 HARD risk is discovered, the scout pass exits with a recommendation to revisit ARCH-1 council decision before forge work begins.

---

**End of scout brief.** This document is for Claude Code execution after council approval. Do not run this against the codebase until ARCH-1 returns GREEN.
