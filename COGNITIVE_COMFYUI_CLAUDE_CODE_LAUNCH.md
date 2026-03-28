# CLAUDE.md — Cognitive ComfyUI Ecosystem
## v2.0 Launch Brief for Claude Code

> **What this is:** A handoff from an architecture session into a build session.
> The architecture is designed. Your job is to SCOUT the codebases, GROUND the
> design in what actually exists, and BUILD Phase 1. Read first. Build second.
> Verify always.

---

## The Vision (30 seconds)

We're building an agentic generative AI ecosystem where:
- **6 specialist agents** (Scout, Architect, Provisioner, Forge, Crucible, Vision) work as a team
- Each agent is a **thinking team member** — with persistent memory, domain expertise, opinions, and independent reasoning
- **USD is the skeleton** — all state (workflows, recipes, agent memory, execution history, scene outputs) lives in a native `pxr.Usd.Stage` with real LIVRPS composition
- **ComfyUI is the execution engine** — agents compose workflows, provision models, execute, analyze outputs, and self-improve
- **The system provisions its own models** (Ollama pattern) — declare what you need, it downloads, verifies, registers
- **Generation outputs become USD scenes** — image + depth + normals + segmentation → composable 3D asset
- **USD scenes can drive generation** — camera prims → ControlNet conditioning, proxy geo → depth maps
- **Three runtime modes:** Interactive (human in loop), Orchestrated (human sets goal, agents execute), Autoresearch (Karpathy Loop — ratchet runs overnight, 1,000+ experiments, best recipe by morning)

This isn't a ComfyUI plugin. It's an ecosystem whose nervous system is thinking agents, whose skeleton is USD, whose brain is composition, and whose heartbeat is the ratchet.

---

## Constitutional Commandments (Non-Negotiable)

These govern YOUR behavior as the build agent, and they govern the system we're building:

1. **Scout Before You Act** — Your first action is ALWAYS reading, never writing. Map the codebase before touching it.
2. **Verify After Every Mutation** — After every file change, run tests. Not later. Not in batch. Immediately.
3. **Bounded Failure → Escalate** — 3 retries max on any fix. Then stop and report what you tried and what failed.
4. **Complete Output or Explicit Blocker** — No stubs. No TODOs. No `# ... existing code ...`. Complete or blocked.
5. **Role Isolation** — You're building. Don't redesign the architecture. Flag disagreements as DESIGN_NOTEs.
6. **Explicit Handoffs** — Every phase produces a named artifact. The next phase reads that artifact.
7. **Adversarial Verification** — Write tests that try to BREAK what you built, not confirm it works.
8. **Human Gates at Irreversible Transitions** — Pause and confirm before: large refactors, new dependencies, structural changes.

---

## Your Mission: Scout → Ground → Build

### PHASE 0: SCOUT (Do this FIRST — before writing ANY code)

Read the following codebases in this order. For each one, produce a **RECON REPORT** summarizing what you found. Do NOT start building until all recon is complete.

#### 0A: Read the comfyui-agent codebase

```
Location: C:\Users\User\comfyui-agent
```

Map the full structure. Understand:

- **agent/** — How are the 61 tools organized? What layers exist (UNDERSTAND, DISCOVER, PILOT, VERIFY, Vision, Planner, Memory, Orchestrator, Optimizer)?
- **agent/tools/** — What does each tool actually do? What are the function signatures? What APIs do they call?
- **agent/brain/** — The Brain layer (Vision, Planner, Memory, Orchestrator, Optimizer, Demo) — how are these implemented? Are they stubs or real?
- **Orchestrator specifically** — It claims "parallel sub-tasks with filtered tool access." How does it ACTUALLY work? This is the foundation we build the MoE router on.
- **Memory module** — "Outcome learning with temporal decay, cross-session patterns." How is this stored? JSON? SQLite? What's the schema?
- **Session persistence** — How do sessions save/load? What format? What survives across sessions?
- **MCP server** — The `agent mcp` command. How does it expose tools? What's the transport?
- **Configuration** — How does `.env` work? What paths are configurable?
- **Entry points** — `agent run`, `agent inspect`, `agent parse`, `agent search`, `agent mcp` — how does the CLI dispatch?
- **Tests** — 429 tests. What do they cover? What's mocked? What's the test infrastructure?
- **pyproject.toml** — Dependencies. Entry points. Optional extras.
- **Existing CLAUDE.md** — What instructions already exist for Claude Code?
- **docs/plans/** — Any existing roadmap or design docs?
- **workflows/** — What workflow templates ship with the agent?

**RECON REPORT 0A deliverable:** A structured summary of the codebase — what exists, what's real vs stubbed, what the architecture actually is (not what the README claims), and what we can build on vs what needs replacing.

#### 0B: Verify USD Python Bindings

```
Test: Can we import pxr from Houdini's Python?
```

Find where Houdini's Python lives on this machine. Test:

```python
import sys
sys.path.append("path/to/houdini/python")  # Find the right path
from pxr import Usd, UsdGeom, Sdf, Vt
```

If this doesn't work from the system Python that comfyui-agent uses, we need to figure out how to bridge Houdini's Python environment with the agent's environment. Options:
- Use Houdini's Python directly (hython)
- Install OpenUSD separately via pip (`pip install usd-core`)
- Subprocess calls to hython for USD operations

**RECON REPORT 0B deliverable:** Can we use pxr? From which Python? What's the integration path?

#### 0C: Probe the ComfyUI API

```
ComfyUI should be running at: http://127.0.0.1:8188
```

If ComfyUI is running, probe:
- `GET /object_info` — What nodes are available? What are their input/output schemas?
- `GET /queue` — What's the queue state?
- `GET /system_stats` — GPU info, VRAM usage
- `POST /prompt` — How does workflow submission actually work?
- WebSocket at `ws://127.0.0.1:8188/ws` — What events fire during execution?

If ComfyUI is NOT running, note it as a blocker and proceed with the rest.

**RECON REPORT 0C deliverable:** ComfyUI API actual behavior — endpoints, response formats, what's documented vs undocumented.

#### 0D: Read the Cognitive Substrate Skills

```
Location: /mnt/skills/user/ (if available) or these files:
- cognitive-twin-spec
- cognitive-bridge-spec  
- cognitive-lossless-spec
- cognitive-injection-spec
- solaris-usd-composition
```

If the skill files aren't at that path (they're from claude.ai, not local), skip this and work from the architecture docs provided.

**RECON REPORT 0D deliverable:** What's implemented vs spec-only in the cognitive substrate.

#### 0E: Map the ComfyUI Database

```
Location: G:\COMFYUI_Database
```

- What models are installed? (checkpoints, loras, controlnet, vae, embeddings)
- What custom nodes are installed?
- What's the directory structure?
- How much disk space is available?

```
Location: G:\COMFY\ComfyUI
```

- What version of ComfyUI is installed?
- What's the Python environment?
- Any custom configuration?

**RECON REPORT 0E deliverable:** Full inventory of the ComfyUI installation.

---

### PHASE 1: GROUND (After ALL recon is complete)

Take the architecture vision and ground it in what you found. Produce a **GROUNDING REPORT** that answers:

1. **What already exists that we can use directly?** (agent tools, orchestrator, memory, MCP server)
2. **What needs to be modified?** (memory format → USD, orchestrator → MoE router)
3. **What needs to be built from scratch?** (CognitiveWorkflowStage, Provisioner, COMPOSITOR)
4. **What are the actual blockers?** (pxr availability, API gaps, missing dependencies)
5. **What's the realistic Phase 1 scope?** (not the architecture doc's 15-week plan — what can we actually build and verify in one session?)

### Key Grounding Questions:

- **The Memory module** currently uses some storage format. Can it be migrated to USD prims, or should we build a parallel USD-native memory alongside it?
- **The Orchestrator** already spawns sub-tasks. How far is it from MoE routing? Is it a refactor or a rewrite?
- **Session persistence** currently uses JSON files in `sessions/`. The USD stage replaces this — but do we need backward compatibility?
- **The 61 existing tools** are the system's hands. Which ones can the Provisioner reuse from DISCOVER? Which need new implementations?
- **pxr availability** — if we can't import pxr directly, is `pip install usd-core` viable in the agent's Python environment? What's the fallback?

---

### PHASE 2: BUILD (After grounding is confirmed)

Build the USD-native foundation. The minimum viable system:

#### 2A: Move Repo + Workspace Symlinks

```bash
# Move from C: to G:
robocopy "C:\Users\User\comfyui-agent" "G:\comfyui-agent" /E /MOVE

# Create workspace symlinks
cd G:\comfyui-agent
mkdir workspace
mklink /D "workspace\models" "G:\COMFYUI_Database\models"
mklink /D "workspace\output" "G:\COMFYUI_Database\output"
mklink /D "workspace\input" "G:\COMFYUI_Database\input"  
mklink /D "workspace\nodes" "G:\COMFYUI_Database\custom_nodes"
mklink /D "workspace\comfyui" "G:\COMFY\ComfyUI"
mkdir workspace\stage

# Verify
agent inspect
```

**Human gate:** Confirm the move worked and `agent inspect` runs clean before proceeding.

#### 2B: CognitiveWorkflowStage Class

The core. A Python class wrapping `pxr.Usd.Stage`:

```python
# agent/stage/cognitive_stage.py

class CognitiveWorkflowStage:
    """
    USD-native composed stage for the entire ecosystem.
    All agent state, workflow data, recipes, and scenes live here.
    """
    
    def __init__(self, root_path):
        # Open or create the stage
        # Bootstrap hierarchy: /workflows, /recipes, /executions, /agents, /models, /scenes
    
    def read(self, prim_path, attr_name):
        # Microsecond read via pxr
    
    def write(self, prim_path, attr_name, value):
        # Microsecond write to session sublayer
    
    def add_agent_delta(self, agent_name, delta_dict):
        # Agent modification as a new .usdc sublayer
    
    def select_profile(self, profile_name):
        # Variant selection for creative profiles
    
    def reconstruct_clean(self):
        # Read only the base layer — lossless reconstruction
    
    def rollback_to(self, n_deltas_ago):
        # Remove top N sublayers
    
    def flush(self):
        # Persist to .usdc
    
    def export(self, output_path, format="usdc"):
        # Export as .usdc, .usda, or .usdz
```

**Verify:** Write a test that creates a stage, writes attributes, adds sublayers, verifies LIVRPS resolution, rolls back, and confirms clean reconstruction. This is the integrity test for the entire system.

#### 2C: Workflow JSON ↔ USD Prim Mapper

Bidirectional translation between ComfyUI's workflow JSON format and USD prim hierarchy:

```python
# agent/stage/workflow_mapper.py

def workflow_json_to_prims(stage, workflow_json, workflow_name):
    """
    ComfyUI workflow JSON → USD prims under /workflows/{name}/
    Each node becomes a prim. Each input becomes an attribute.
    Connections become USD relationships.
    """

def prims_to_workflow_json(stage, workflow_name):
    """
    USD prims under /workflows/{name}/ → ComfyUI workflow JSON
    Flattens the composed stage (all LIVRPS resolved) into 
    a single workflow JSON ready to POST to ComfyUI.
    """
```

**Verify:** Round-trip test. Load a real workflow JSON from `workflows/`, convert to prims, convert back, confirm the output JSON is functionally identical (node IDs and connections match, even if key ordering differs).

#### 2D: Anchor Parameter Immunity

Define which workflow parameters are structurally immune to agent modification:

```python
# agent/stage/anchors.py

ANCHOR_PARAMS = {
    "CheckpointLoaderSimple": ["ckpt_name"],  # Model selection
    "EmptyLatentImage": ["width", "height"],    # Resolution
    # ... safety filters, node graph topology
}

def is_anchor(node_type, param_name):
    """Returns True if this parameter is constitutionally protected."""
    return param_name in ANCHOR_PARAMS.get(node_type, [])
```

The `write()` method in CognitiveWorkflowStage checks `is_anchor()` before any write. If it's an anchor, the write is rejected with a constitutional violation error. Not a soft warning — a hard stop.

**Verify:** Test that writing to an anchor param raises an error. Test that non-anchor params write normally. Test that fidelity verification catches any bypass.

---

## Architecture Reference (Condensed)

### The Ecosystem Stack

```
Human / USD Scene Brief
        ↓
Router (MoE Facilitator)
        ↓
6 Thinking Agents (each a Claude instance with memory + tools + opinions)
  Scout:        21 tools — reconnaissance, discovery, mapping (READ ONLY)
  Architect:     8 tools — design, planning, decomposition (DESIGN ONLY)
  Provisioner:  13 tools — resolve, download, verify, register models (WRITE: models/ only)
  Forge:        19 tools — build, patch, execute workflows (WRITE: workflows/)
  Crucible:      9 tools — break, test, verify (WRITE: tests/ only)
  Vision:        6 tools — analyze, score, diagnose (READ + COMPOSITOR)
        ↓
Cognitive Workflow Stage (pxr.Usd.Stage, native, in-memory)
  /workflows/   — node prims, params as attributes, connections as relationships
  /recipes/     — learned parameter combinations (strongest patterns)
  /executions/  — full history as AIMemoryChunk prims with decay_weight
  /agents/      — per-agent memory (episodic, semantic, procedural, decision)
  /models/      — inventory (materialized: on disk | available: known, not downloaded)
  /scenes/      — USD scene outputs (image+depth+normals → 3D)
  /inputs/      — USD scene inputs (camera+lights+proxy → conditioning)
        ↓
ComfyUI Server :8188
  REST API for workflow submission
  WebSocket for execution monitoring
```

### LIVRPS Composition (How conflicts resolve)

```
L (Local)      → Agent's direct edit          STRONGEST — always wins
I (Inherit)    → Agent role constraints        Role isolation rules
V (Variants)   → Creative profiles             /inject explore, creative, radical
R (References) → Learned recipes               Best-scoring param combos from memory
P (Payloads)   → Lazy-loaded components        Models, ControlNet configs (Provisioner materializes)
S (Specialize) → Base workflow templates       WEAKEST — default foundation
```

### Self-Improving Loop + Ratchet

```
ORCHESTRATED (interactive/goal-driven):
  Design → Provision → Build → Execute → Compose Scene → Validate → Learn → Iterate
  Convergence: score plateau across 3 runs, or budget exhausted

AUTORESEARCH (overnight, autonomous — the Karpathy Loop):
  Read program.md → Propose delta → Apply as sublayer → Execute (fixed time budget)
  → Score on 6 axes → Beat baseline? KEEP sublayer : DISCARD sublayer → Repeat
  Ratchet: binary keep/discard. 1,000+ experiments overnight. Recipe by morning.
```

### Three Runtime Modes

```
agent run            — Interactive. Human in loop. Conversational.
agent orchestrate    — Orchestrated. Human sets goal + approves design. Agents execute chain.
agent autoresearch   — Autoresearch. Human writes program.md. Ratchet runs overnight.
```

### USD Scene I/O

**Output (Path 2):** After execution, compose image + depth + normals + segmentation into USD scene with UsdGeomCamera, UsdGeomMesh, UsdShadeMaterial. Enables geometric validation.

**Input (Path 3):** USD scene with camera/lights/proxy geometry → extract conditioning for ComfyUI (ControlNet depth, focal length → DoF, light direction → prompt).

---

## What to Build vs What Exists

| Component | Status | Action |
|---|---|---|
| 61 ComfyUI tools | **EXISTS** in agent/ | Use directly. These are the system's hands. |
| CLI + entry points | **EXISTS** | Keep. Add `agent orchestrate` + `agent autoresearch` commands. |
| MCP server | **EXISTS** | Keep. Will expose new tools alongside existing ones. |
| Session persistence | **EXISTS** (JSON) | Migrate to USD stage. Keep JSON as fallback. |
| Memory module | **EXISTS** (Brain layer) | Extend to use AIMemoryChunk prims in USD stage. |
| Orchestrator | **EXISTS** (2 tools) | Foundation for MoE router. Evaluate: refactor or wrap. |
| Vision/Planner/Optimizer | **EXISTS** (Brain layer) | Keep. Wire into self-improving loop + ratchet scoring. |
| Tests (429) | **EXISTS** | Must all pass after every change. Sacred. |
| CognitiveWorkflowStage | **NEW** | Build in Phase 2B. Core of everything. |
| Workflow ↔ Prim mapper | **NEW** | Build in Phase 2C. Bidirectional translation. |
| Anchor immunity | **NEW** | Build in Phase 2D. Constitutional enforcement. |
| Provisioner tools | **NEW** | Build in Phase 3 (after foundation verified). |
| COMPOSITOR tools | **NEW** | Build in Phase 4 (after agent integration). |
| Per-agent memory (USD) | **NEW** | Build in Phase 3 (agent cognitive state). |
| MoE Router | **NEW** | Build in Phase 3 (extends existing Orchestrator). |
| Ratchet class | **NEW** | Build in Phase 5. Binary keep/discard on USD sublayers. |
| program.md parser | **NEW** | Build in Phase 5. Extract axes, ranges, strategies. |
| Morning report generator | **NEW** | Build in Phase 5. Experiment log + recipe extraction. |
| `agent autoresearch` CLI | **NEW** | Build in Phase 5. Overnight autonomous mode. |
| Creative profiles | **NEW** | Build in Phase 6 (variant sets). |
| Meta-agent (Hyperagent) | **NEW** | Build in Phase 7. Self-referential self-improvement. |
| Three-tier evolution boundary | **NEW** | Build in Phase 7. Auto / Ratchet-validated / Human-gated. |
| Meta-agent strategy (USD sublayer) | **NEW** | Build in Phase 7. The modification procedure is itself modifiable. |

---

## Critical First-Principles Decisions (Already Made)

1. **USD-native from day one.** No SQLite. No database simulating composition. `pxr.Usd.Stage` directly. The composition engine is free — don't reimplement it.

2. **Models are Payloads.** The Provisioner materializes them. Declare what you need, the system downloads, verifies (SHA256), and registers it as a prim.

3. **Generation outputs become USD scenes.** Image + depth + normals + segmentation → composable 3D asset. This gives the self-improving loop geometric validation, not just aesthetic scoring.

4. **Agents are thinkers, not dispatchers.** Each agent maintains persistent memory as USD prims, develops domain expertise over time, and reasons independently before acting.

5. **Constitutional commandments are code, not just prompts.** `commandments.py` enforces the 8 rules as pre/post checks. The CLAUDE.md teaches WHY. The code enforces regardless.

6. **Data auto-evolves. Architecture requires human approval.** Recipes, routing weights, memory patterns, ratchet-discovered recipes evolve freely. Agent roles, constitutional rules, tool access lists — human gate required.

7. **The Ratchet is the Karpathy Loop on USD sublayers.** Binary keep/discard. Sublayers replace git commits — composable, non-destructive, branchable. Multi-axis scoring replaces single-metric optimization. Overnight autonomous mode: write program.md, go to sleep, check morning report. 1,000+ experiments on a 4090.

8. **The Hyperagent pattern (Zhang et al.) makes the meta-layer self-modifying.** The meta-agent can tune agent prompts and its own modification strategy — but only through ratchet validation. The scoring function is a constitutional anchor: the meta-agent can change HOW it searches, never how results are JUDGED. All meta-modifications are USD sublayers — composable, reversible, inspectable. Improvements transfer across domains and accumulate across runs.

---

## Session Management

After each major phase, produce a **STATE CAPSULE:**

```
+==============================================================+
| SESSION CAPSULE: Cognitive ComfyUI Build                     |
| Updated: {timestamp}  |  Phase: {current_phase}             |
+==============================================================+
| WHERE WE ARE: {current_position}                             |
| MILE MARKER: {X} of ~{Y}                                    |
| WHAT WAS DONE: {summary of completed work}                   |
| WHAT WAS LEARNED: {discoveries, surprises, corrections}      |
| IMMEDIATE NEXT ACTION: {single next step}                    |
| BLOCKERS: {any blockers encountered}                         |
| TESTS: {test count — passed/failed/new}                      |
| FILES CHANGED: {list of modified files}                      |
+==============================================================+
```

---

## Launch Command

```bash
cd G:\comfyui-agent   # or C:\Users\User\comfyui-agent if not yet moved
claude --dangerously-skip-permissions
```

Then paste or reference this file as context. First words to Claude Code:

> "Read this CLAUDE.md. Start Phase 0: Scout all codebases. Produce recon reports before writing any code."

---

## File Locations Quick Reference

```
comfyui-agent repo:     C:\Users\User\comfyui-agent  (moves to G:\comfyui-agent)
ComfyUI software:       G:\COMFY\ComfyUI
ComfyUI database:       G:\COMFYUI_Database
Houdini (for pxr):      Find via: where hython  or  check C:\Program Files\Side Effects Software\
Architecture docs:       (attached alongside this file)
```

---

*Scout before you act. Verify after every mutation. Build what was designed. Break what was built.*
*The nervous system is thinking agents. The skeleton is USD. The brain is composition. The heartbeat is the ratchet.*
*v2.0.0 — THINKING AGENTS*
