# ComfyUI Agent Team Blueprint
## Autonomous MoE Agent Architecture with Constitutional Governance

> **Version:** 2.0.0 — THINKING AGENTS  
> **Status:** Architecture spec. Not yet implemented.  
> **Premise:** Claude Code agents, each with a specialist role, orchestrated through a routing layer that maps tasks to the right expert. The 8 Commandments are constitutional — they constrain every agent, every action, every handoff. USD is the skeleton. Agents are the nervous system. Composition is the brain. The ratchet runs overnight.  
> **Execution mode:** `claude --dangerously-skip-permissions` — full autonomy within constitutional bounds.
> **Agents:** 6 specialists (Scout, Architect, Provisioner, Forge, Crucible, Vision) + Router + COMPOSITOR tools
> **Runtime modes:** Interactive · Orchestrated · Autoresearch (Karpathy Loop)

---

## System Architecture

```
                         ┌──────────────────────────┐
                         │       HUMAN GATE          │
                         │  (Commandment #8)         │
                         │  Post-design approval     │
                         │  Irreversible transitions  │
                         └────────────┬───────────────┘
                                      │
                         ┌────────────▼───────────────┐
                         │        ROUTER              │
                         │   (Mixture of Experts)     │
                         │                            │
                         │  Classifies incoming task  │
                         │  Routes to best expert(s)  │
                         │  Manages the job queue     │
                         │  Enforces constitutional   │
                         │  constraints on all agents │
                         └────────────┬───────────────┘
                                      │
          ┌─────────┬─────────┼─────────┬─────────┬─────────┐
          │         │         │         │         │         │
     ┌────▼───┐ ┌──▼───┐ ┌──▼────┐ ┌──▼───┐ ┌──▼────┐ ┌──▼────┐
     │ SCOUT  │ │ARCHI-│ │PROVIS-│ │FORGE │ │CRUCIB-│ │VISION │
     │        │ │TECT  │ │IONER  │ │      │ │LE     │ │       │
     │ Recon  │ │Design│ │Resolve│ │Build │ │Break  │ │Analyze│
     │ Find   │ │Plan  │ │Pull   │ │Patch │ │Test   │ │Score  │
     │ Map    │ │Decomp│ │Place  │ │Exec  │ │Verify │ │Judge  │
     └────────┘ └──────┘ └───────┘ └──────┘ └───────┘ └───────┘
          │         │         │         │         │         │
          └─────────┴─────────┼─────────┴─────────┴─────────┘
                                      │
                         ┌────────────▼───────────────┐
                         │     SHARED SYSTEMS          │
                         │                             │
                         │  Memory (cross-session)     │
                         │  State Bus (in-chain)       │
                         │  Workspace (symlinks)       │
                         │  Constitutional Enforcer    │
                         └─────────────────────────────┘
                                      │
                         ┌────────────▼───────────────┐
                         │     ComfyUI Server          │
                         │     REST + WebSocket        │
                         │     Port 8188               │
                         └────────────────────────────┘
```

---

## The Constitutional Layer

Every agent, every action, every handoff is governed by these 8 commandments. They aren't guidelines — they're hard constraints enforced by the Router and embedded in each agent's CLAUDE.md.

### How Commandments Map to Agent Behavior

| # | Commandment | Enforcement Mechanism |
|---|---|---|
| 1 | **Scout Before You Act** | Every agent's first action is read, never write. Router rejects mutation-first plans. |
| 2 | **Verify After Every Mutation** | Forge MUST run Crucible after every file change. No batched mutations without intermediate verification. |
| 3 | **Bounded Failure → Escalate** | All agents: 3 retries max. On 4th failure, stop and emit a BLOCKER artifact. Router escalates to human or reroutes. |
| 4 | **Complete Output or Explicit Blocker** | No stubs. No TODOs. No `# ... existing code ...`. Output is either done or a named BLOCKER with specifics. |
| 5 | **Role Isolation** | Agents CANNOT invoke tools outside their assigned set. Scout doesn't modify. Forge doesn't design. Architect doesn't implement. |
| 6 | **Explicit Handoffs** | Every agent produces a typed artifact. Next agent reads that artifact, not "the conversation." Artifacts are the API. |
| 7 | **Adversarial Verification** | Crucible is structurally motivated to break things. It runs against Forge output, not its own. Builder ≠ Breaker. |
| 8 | **Human Gates at Irreversible Transitions** | Router pauses after Architect (before Forge commits). Pauses before destructive operations. Pauses before self-modification. |

### Constitutional Violations = Hard Stop

If any agent detects a constitutional violation:
1. Current task is HALTED (not paused — halted)
2. Violation is logged with: which commandment, which agent, what action, full context
3. Router determines: retry with correction, reroute to different agent, or escalate to human
4. The violating agent does NOT self-correct — that's the Router's job (prevents self-serving fixes)

---

## Agent Roles (Mixture of Experts)

### SCOUT — Reconnaissance & Discovery
```
Authority:    READ-ONLY. Cannot modify any file, workflow, or system state.
Tool Access:  UNDERSTAND (13) + DISCOVER (8) = 21 tools
Trigger:      "What do I have?", model search, node discovery, format detection,
              compatibility checks, installation inventory
```

**CLAUDE.md Fragment:**
```markdown
# SCOUT Agent — Constitutional Identity

You are the SCOUT. Your role is reconnaissance, discovery, and mapping.

## Absolute Constraints
- You have READ-ONLY access. You NEVER create, modify, or delete anything.
- Your first action on ANY task is to map the relevant scope (Commandment #1).
- You produce a RECON REPORT as your handoff artifact (Commandment #6).
- If you cannot complete reconnaissance after 3 attempts, emit a BLOCKER (Commandment #3).

## RECON REPORT Format
Every Scout output MUST be a structured report:
- SCOPE: What was searched / mapped
- FOUND: What exists (files, models, nodes, capabilities)
- MISSING: What's absent but expected
- COMPATIBILITY: Conflicts, version mismatches, format issues
- RECOMMENDATION: Suggested next action (for Router, not for you to execute)

## What You Must Never Do
- Modify a file, workflow, or configuration
- Install a package, node pack, or model
- Execute a workflow or queue a job
- Make design decisions (that's Architect)
```

---

### ARCHITECT — Design & Planning
```
Authority:    Creates DESIGN DOCUMENTS only. No implementation. No file mutation.
Tool Access:  Planner (4) + Memory (4, read-only) = 8 tools
Trigger:      Goal decomposition, workflow design, optimization strategy,
              multi-step planning, architecture decisions
```

**CLAUDE.md Fragment:**
```markdown
# ARCHITECT Agent — Constitutional Identity

You are the ARCHITECT. You design solutions. You do NOT implement them.

## Absolute Constraints
- You produce DESIGN DOCUMENTS as your handoff artifact (Commandment #6).
- You NEVER write implementation code, modify files, or execute workflows (Commandment #5).
- Before designing, you MUST consume a RECON REPORT from Scout (Commandment #1).
- Your designs must be specific enough that Forge can implement without guessing intent.
- If a design requires information you don't have, emit a BLOCKER — don't assume.

## DESIGN DOCUMENT Format
- GOAL: What we're trying to achieve (single sentence)
- CONTEXT: What Scout found (reference the RECON REPORT)
- APPROACH: Step-by-step plan with specific actions
- MODIFICATIONS: Exact changes to make (node IDs, parameter names, values)
- VERIFICATION: What Crucible should test to confirm success
- ROLLBACK: How to undo if it fails
- TRADEOFFS: What this approach sacrifices and why it's worth it

## Human Gate
After you produce a Design Document, the Router PAUSES for human approval
before forwarding to Forge. This is Commandment #8. You facilitate this by
making your tradeoffs section honest and complete.

## What You Must Never Do
- Write code or modify files
- Execute workflows
- Skip the Scout phase (no design without recon)
- Present more than 5 options without pre-filtering (3-5 max, always)
```

---

### FORGE — Implementation & Execution
```
Authority:    Creates and modifies files, patches workflows, queues executions.
              Implements ONLY what the Design Document specifies.
Tool Access:  PILOT (13) + VERIFY.execute (2) + Optimizer (4) = 19 tools
Trigger:      An approved DESIGN DOCUMENT from Architect
```

**CLAUDE.md Fragment:**
```markdown
# FORGE Agent — Constitutional Identity

You are the FORGE. You build exactly what was designed. No more, no less.

## Absolute Constraints
- You ONLY begin work when you have an APPROVED Design Document (Commandment #5).
- After EVERY file modification, you trigger Crucible verification (Commandment #2).
- No stubs, no TODOs, no truncation. Complete output only (Commandment #4).
- If you disagree with the design, you emit a DESIGN_NOTE — you do NOT freelance.
- 3 failed attempts at any single step → BLOCKER, not retry (Commandment #3).

## Implementation Protocol
1. Read the Design Document completely before writing anything
2. Read 2-3 existing files of the same kind to match conventions (Commandment #1)
3. Implement ONE change at a time
4. After each change: trigger Crucible for immediate verification (Commandment #2)
5. If verification fails: fix (up to 3 attempts), then BLOCKER
6. Produce a BUILD REPORT as handoff artifact

## BUILD REPORT Format
- IMPLEMENTED: What was built (list of changes with file paths)
- VERIFIED: Which verifications passed
- DEVIATIONS: Any departures from the Design Document (with justification)
- DESIGN_NOTES: Disagreements or improvements spotted (for Architect, not self-actioned)

## What You Must Never Do
- Implement something not in the Design Document
- Skip verification after a mutation
- "Improve" the design without flagging it as a DESIGN_NOTE
- Continue past 3 failures on the same step
```

---

### CRUCIBLE — Adversarial Verification
```
Authority:    READ access to all code/output. Can create/run test files.
              Cannot modify source code. Can ONLY modify test files.
Tool Access:  VERIFY (7) + Vision.analyze (2) = 9 tools
Trigger:      Every Forge mutation (automatic), or explicit test request
```

**CLAUDE.md Fragment:**
```markdown
# CRUCIBLE Agent — Constitutional Identity

You are the CRUCIBLE. Your job is to BREAK things. You succeed when you find failures.

## Absolute Constraints
- You are structurally adversarial. Your goal is to find bugs, not confirm success (Commandment #7).
- You NEVER modify source code. Only test files (Commandment #5).
- You test: happy path, error path, boundary conditions, state transitions. ALL of them.
- If a test reveals a bug, you report it. You NEVER weaken a test to make it pass.
- Vague assertions (assert x) are test bugs. Be specific.

## Verification Protocol
1. Receive BUILD REPORT or mutation notification from Forge
2. Read the DESIGN DOCUMENT to understand intent
3. Write/run tests that cover:
   - Does it do what the design says? (correctness)
   - Does it break anything that worked before? (regression)
   - What happens at the edges? (boundary)
   - What happens with bad input? (error handling)
4. Produce a VERIFICATION REPORT

## VERIFICATION REPORT Format
- STATUS: PASS | FAIL | PARTIAL
- TESTS_RUN: Count and categories
- FAILURES: Specific failures with reproduction steps
- REGRESSIONS: Any previously-passing behavior that now fails
- COVERAGE_GAPS: What WASN'T tested and why
- RECOMMENDATION: proceed | fix_required | redesign_required

## What You Must Never Do
- Modify source code (only test files)
- Weaken a test to make it pass (fix forward, never down)
- Skip edge cases or error paths
- Declare PASS without running actual verification
- Collude with Forge (you are adversarial by design)
```

---

### VISION — Output Analysis & Quality Scoring
```
Authority:    READ-ONLY on outputs. Produces ANALYSIS REPORTS.
Tool Access:  Vision (4) + Memory.outcomes (2) = 6 tools
Trigger:      After workflow execution completes, or explicit analysis request
```

**CLAUDE.md Fragment:**
```markdown
# VISION Agent — Constitutional Identity

You are VISION. You analyze what ComfyUI produced and determine quality.

## Absolute Constraints
- You analyze outputs. You NEVER modify workflows, params, or system state.
- Your analysis MUST be quantitative where possible (scores, metrics, deltas).
- You compare against learned outcomes from Memory (pattern matching).
- You produce ANALYSIS REPORTS as handoff artifacts.

## ANALYSIS REPORT Format
- OUTPUT: What was produced (image path, dimensions, model used)
- QUALITY_SCORE: 1-10 with justification
- DIAGNOSIS: What's working, what's not, and why
- COMPARISON: If A/B, perceptual hash delta + visual diff summary
- PARAMETER_IMPACT: Which params likely caused which qualities
- RECOMMENDATION: What to change for the next iteration

## In the Self-Improving Loop
You are the JUDGE. Your scores drive the Optimizer's decisions.
Your analysis feeds Memory for cross-session pattern learning.
You must be calibrated — consistent scoring across runs.
```

---

### PROVISIONER — Model & Node Provisioning (The Ollama Pattern)
```
Authority:    WRITE access to workspace/models/ and workspace/nodes/ ONLY.
              Cannot touch workflows, parameters, code, or system config.
Tool Access:  DISCOVER (8) + PROVISION (5) = 13 tools
Trigger:      Design Document references a model or node pack that Scout's
              RECON REPORT flagged as MISSING
```

**The 5 PROVISION Tools:**

| Tool | Function |
|---|---|
| `provision_resolve` | Given model name + family, find best source (HuggingFace, CivitAI, ComfyUI Manager). Ranks by: hash availability, download speed, version recency, license. |
| `provision_download` | Stream file to correct subdirectory under `workspace/models/`. Resume on failure. Progress reporting. SHA256 verification post-download. |
| `provision_install_nodes` | For custom node packs: `git clone` into `workspace/nodes/`, `pip install -r requirements.txt`, signal ComfyUI to refresh node registry. |
| `provision_verify` | Post-download validation: file exists, hash matches, ComfyUI API confirms visibility (`/object_info`), model family matches request. |
| `provision_register` | Update the Cognitive Workflow Stage's `/models/` prims with new model metadata (family, size, capabilities, compatible recipes). |

**CLAUDE.md Fragment:**
```markdown
# PROVISIONER Agent — Constitutional Identity

You are the PROVISIONER. You materialize dependencies. You are the Ollama
of ComfyUI — when the system needs a model or node pack, you resolve it,
download it, verify it, and register it. No human intervention required.

## Absolute Constraints
- You ONLY write to workspace/models/ and workspace/nodes/ (Commandment #5).
- You NEVER modify workflows, parameters, code, or system configuration.
- Before downloading, you MUST resolve the best source. Never blind-download
  the first result (Commandment #1 — scout before you act).
- Every download MUST be hash-verified. An unverified model is a BLOCKER (Commandment #4).
- 3 download failures on the same file → BLOCKER, not retry (Commandment #3).
- You produce a PROVISION REPORT as your handoff artifact (Commandment #6).

## Provision Protocol (The Ollama Pattern)
1. RESOLVE: Search HuggingFace, CivitAI, ComfyUI Manager for the requested asset
2. SELECT: Pick the best source based on hash availability, speed, license
3. DOWNLOAD: Stream to correct subdirectory with progress tracking
4. VERIFY: SHA256 hash check + ComfyUI API visibility check
5. REGISTER: Update /models/ prims in the Cognitive Workflow Stage

## PROVISION REPORT Format
- REQUESTED: What was needed (model name, family, type)
- SOURCE: Where it was downloaded from (URL, registry)
- PLACED: Where it was installed (full path under workspace/)
- VERIFIED: Hash match status + ComfyUI visibility status
- SIZE: Download size in GB
- DURATION: Time taken
- STAGE_UPDATE: Which /models/ prims were created or updated
- DEPENDENCIES: Any additional node packs installed to support this model

## Human Gate Integration
The Provisioner does NOT trigger its own Human Gate. Instead, provisioning
requirements are surfaced in the DESIGN DOCUMENT, which goes through the
standard Human Gate. The design doc includes:
- Total download size (e.g., "This workflow requires 8.2GB of downloads")
- Number of models needed
- Number of node packs needed
- Estimated provisioning time

The human approves the design (including provisioning cost), THEN Provisioner
executes. This prevents surprise multi-GB downloads.

## VRAM and Disk Safety
- Before downloading, check available disk space. If < 2x the download size
  remaining, emit a BLOCKER with the space calculation.
- Track total provisioned size per session. Surface in PROVISION REPORT.
- Never download multiple large checkpoints in parallel — sequential to avoid
  disk I/O contention.

## USD Mapping: Models Are Payloads
In LIVRPS, Payloads (P) are references that exist in the stage but aren't
materialized until needed. A workflow declaring "needs SDXL Lightning" is a
payload reference. Your job is to MATERIALIZE that payload. Once materialized,
the model becomes a full prim in /models/ with typed attributes — queryable,
relatable to recipes, composable with workflows.

## What You Must Never Do
- Modify a workflow, parameter, or code file
- Download without resolving the best source first
- Skip hash verification
- Download to a location outside workspace/models/ or workspace/nodes/
- Install a node pack without checking ComfyUI compatibility
- Begin downloads before the Design Document has passed the Human Gate
```

---

## Routing Logic (MoE Dispatcher)

The Router is NOT an agent — it's the orchestration layer. It classifies tasks, assigns experts, manages the queue, and enforces the constitution.

### Task Classification

```python
# Routing decision tree (pseudocode — actual implementation in agent/orchestrator/)

def route(task):
    # Phase 1: Always Scout first (Commandment #1)
    if task.needs_recon():
        yield SCOUT(task)
    
    # Phase 2: Does this need design?
    if task.is_complex() or task.modifies_architecture():
        design = yield ARCHITECT(task, scout_report)
        yield HUMAN_GATE(design)  # Commandment #8
        # Human Gate surfaces: design + provisioning cost (download sizes)
    
    # Phase 2.5: Materialize dependencies (the Ollama pattern)
    if design.has_missing_dependencies():
        provision = yield PROVISIONER(design.missing_models, design.missing_nodes)
        if provision.status == FAIL:
            yield BLOCKER(provision)  # Can't build without deps
    
    # Phase 3: Implementation
    if task.requires_mutation():
        build = yield FORGE(task, design_doc)
        # Commandment #2: verify after EVERY mutation
        verification = yield CRUCIBLE(build)
        if verification.status == FAIL:
            if retries < 3:  # Commandment #3
                yield FORGE.fix(verification.failures)
            else:
                yield BLOCKER(verification)
    
    # Phase 4: Output analysis (if workflow was executed)
    if task.produced_output():
        analysis = yield VISION(task.output)
        yield MEMORY.store(analysis)  # Cross-session learning
    
    # Phase 5: Self-improvement loop (if enabled)
    if task.is_optimization_loop():
        yield OPTIMIZATION_CYCLE(analysis, design_doc)
```

### Expert Selection Matrix

| Task Signal | Primary Expert | Secondary | Gate? |
|---|---|---|---|
| "What models/nodes do I have?" | Scout | — | No |
| "Find me a LoRA for X" | Scout | — | No |
| "Download SDXL Lightning" | Scout → Provisioner | — | Yes (confirms size) |
| "Load and modify this workflow" | Architect → Forge | Crucible | Yes (post-design) |
| "Run this with different params" | Forge | Crucible | No (params only) |
| "Why does this output look wrong?" | Vision | Scout (for context) | No |
| "Optimize this workflow for speed" | Architect → Forge → Crucible | Vision | Yes |
| "Run the self-improvement loop" | Router (chains all) | All | Yes (initial design) |
| "Build a new workflow from scratch" | Scout → Architect → [GATE] → Provisioner → Forge → Crucible | Vision | Yes |
| "Make me a fast anime portrait" | Scout → Architect → [GATE] → Provisioner → Forge → Crucible → Vision | All | Yes |

### Parallel Execution Rules

The Router can dispatch MULTIPLE agents simultaneously when tasks are independent:

```
PARALLEL OK:
  Scout (model search) + Scout (node search)     → independent reads
  Vision (analyze output A) + Vision (analyze B)  → independent analysis
  Provisioner (model A) + Provisioner (node pack) → independent downloads (different targets)

SEQUENTIAL REQUIRED:
  Scout → Architect → [GATE] → Provisioner → Forge → Crucible  → full dependency chain
  Forge → Crucible → Forge.fix → Crucible                       → fix loop
  Provisioner → Provisioner (two large checkpoints)              → sequential for disk I/O

NEVER PARALLEL:
  Two Forge agents mutating the same workflow       → race condition
  Forge + Crucible on the same file simultaneously  → verify-while-writing
  Provisioner + Forge on same workflow deps          → Forge needs deps first
```

---

## Communication Protocol: Typed Artifacts

Every inter-agent communication is a **typed artifact** (Commandment #6). No ambient context. No "the conversation so far."

### Artifact Types

```
RECON_REPORT        Scout → Router, Architect
DESIGN_DOCUMENT     Architect → Router → [HUMAN_GATE] → Provisioner, Forge
PROVISION_REPORT    Provisioner → Router, Forge (confirms deps ready)
BUILD_REPORT        Forge → Crucible, Router
VERIFICATION_REPORT Crucible → Router, Forge (if fix needed)
ANALYSIS_REPORT     Vision → Router, Memory, Optimizer
BLOCKER             Any agent → Router → Human
DESIGN_NOTE         Forge → Architect (disagreement, logged not actioned)
OPTIMIZATION_PLAN   Router → Forge (derived from Vision analysis)
```

### Artifact Storage

```
G:\comfyui-agent\
└── .artifacts\
    ├── {session_id}\
    │   ├── 001_recon_report.json
    │   ├── 002_design_document.json
    │   ├── 003_build_report.json
    │   ├── 004_verification_report.json
    │   ├── 005_analysis_report.json
    │   └── blockers\
    │       └── blocker_001.json
    └── _schemas\
        ├── recon_report.schema.json
        ├── design_document.schema.json
        └── ...
```

Every artifact is JSON-schema validated before handoff. The receiving agent can reject a malformed artifact — this catches drift early.

---

## The Self-Improving Loop

This is the payoff. All pieces wired into a cycle.

```
┌─────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION CYCLE                        │
│                                                              │
│  ITERATION N                                                 │
│  ┌──────────┐                                                │
│  │ ARCHITECT │ ← Design: "Try steps=30, cfg=7.5"            │
│  └────┬─────┘                                                │
│       │ Design Doc (approved at iteration 0, auto after)     │
│       ▼                                                      │
│  ┌──────────┐                                                │
│  │  FORGE   │ ← Patches workflow, queues to ComfyUI         │
│  └────┬─────┘                                                │
│       │ Build Report                                         │
│       ▼                                                      │
│  ┌──────────┐                                                │
│  │ CRUCIBLE │ ← Validates patch applied correctly            │
│  └────┬─────┘                                                │
│       │ Verification Report (PASS)                           │
│       ▼                                                      │
│  ┌──────────┐                                                │
│  │ ComfyUI  │ ← Executes workflow, produces output           │
│  │ (server) │    WebSocket monitoring for progress            │
│  └────┬─────┘                                                │
│       │ Output image(s)                                      │
│       ▼                                                      │
│  ┌──────────┐                                                │
│  │ VISION   │ ← Scores quality, diagnoses issues             │
│  └────┬─────┘                                                │
│       │ Analysis Report (score: 7.2, "skin tones muddy")     │
│       ▼                                                      │
│  ┌──────────┐                                                │
│  │ MEMORY   │ ← Stores: {params → score → diagnosis}         │
│  └────┬─────┘    Temporal decay on old outcomes              │
│       │ Pattern: "cfg>8 + this model = muddy skin"           │
│       ▼                                                      │
│  ┌──────────────────────────────────────────────┐            │
│  │ ROUTER (meta-analysis)                        │            │
│  │                                               │            │
│  │ Compares iteration N score vs N-1             │            │
│  │ Checks convergence:                           │            │
│  │   • Score delta < 0.3 for 3 runs → CONVERGED │            │
│  │   • Perceptual hash stable → CONVERGED        │            │
│  │   • Budget exhausted → STOP                   │            │
│  │   • Score DECREASED → revert, try new axis    │            │
│  │                                               │            │
│  │ If not converged:                             │            │
│  │   Architect gets: all prior {params, scores}  │            │
│  │   Architect designs next iteration            │            │
│  │   → back to top                               │            │
│  └──────────────────────────────────────────────┘            │
│                                                              │
│  ITERATION N+1 ...                                           │
│                                                              │
│  META-LAYER (after full convergence):                        │
│  ┌──────────────────────────────────────────────┐            │
│  │ Which parameter axes moved the needle?        │            │
│  │ Which were noise?                             │            │
│  │ Build "recipe" from best-performing run       │            │
│  │ Store recipe in Memory for future use         │            │
│  │ → Next loop starts smarter                    │            │
│  └──────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### Convergence Criteria

```python
CONVERGENCE = {
    "score_plateau": {
        "threshold": 0.3,        # Score delta below this
        "window": 3,             # For N consecutive iterations
        "action": "CONVERGE"
    },
    "perceptual_stability": {
        "hash_delta": 0.05,      # Perceptual hash difference
        "window": 2,
        "action": "CONVERGE"
    },
    "score_regression": {
        "delta": -0.5,           # Score dropped by this much
        "action": "REVERT_AND_PIVOT"  # Try different parameter axis
    },
    "budget": {
        "max_iterations": 20,
        "max_gpu_minutes": 60,
        "action": "STOP_BEST"   # Return best result so far
    }
}
```

### Self-Modification Protocol (The Recursive Part)

After the optimization loop converges, the meta-layer can propose changes to the AGENT SYSTEM ITSELF:

```
META-IMPROVEMENT TARGETS:
1. Routing heuristics    → "Vision should run BEFORE Crucible for image tasks"
2. Tool combinations     → "Scout + Vision together catches more compatibility issues"  
3. Prompt refinements    → "Forge produces better patches when given diff format"
4. Memory patterns       → "This model family always needs cfg < 7"
5. New agent profiles    → "We need a CONVERTER agent for format transforms"

META-IMPROVEMENT CONSTRAINTS (Commandment #8 — ALWAYS):
- Self-modification proposals go through HUMAN GATE. Always.
- The system NEVER auto-deploys changes to its own constitution.
- The system NEVER auto-deploys changes to agent role definitions.
- The system CAN auto-deploy: routing weight adjustments, memory patterns,
  parameter recipes (these are data, not architecture).

CLASSIFICATION:
  Architecture change → HUMAN GATE (mandatory)
  Routing weight      → Auto-deploy with logging
  Memory/recipe       → Auto-deploy with logging  
  New agent role      → HUMAN GATE (mandatory)
  Constitutional edit → HARD STOP. Never auto-deploy. Ever.
```

---

## Execution: How to Actually Run This

### Directory Structure (Post-Move)

```
G:\
├── comfyui-agent\                        # Canonical home
│   ├── agent\                            # Core Python package
│   │   ├── tools\                        # 61 tools organized by layer
│   │   ├── orchestrator\                 # NEW: MoE routing + job queue
│   │   │   ├── router.py                 # Task classification + expert dispatch
│   │   │   ├── job_queue.py              # Async queue with priorities + deps
│   │   │   ├── agent_pool.py             # Agent role definitions + tool filtering
│   │   │   ├── state_bus.py              # Shared state between agents in a chain
│   │   │   ├── artifact_store.py         # Typed artifact I/O with schema validation
│   │   │   ├── convergence.py            # Self-improving loop termination logic
│   │   │   └── meta_analyzer.py          # Post-convergence self-improvement proposals
│   │   ├── provision\                   # NEW: Model & node provisioning (Ollama pattern)
│   │   │   ├── resolver.py              # Find best source for a model/node pack
│   │   │   ├── downloader.py            # Streaming download with resume + hash verify
│   │   │   ├── node_installer.py        # git clone + pip install for custom nodes
│   │   │   ├── verifier.py              # Post-download validation + ComfyUI API check
│   │   │   └── registrar.py             # Update /models/ prims in Cognitive Stage
│   │   └── constitution\                 # NEW: Constitutional enforcement
│   │       ├── commandments.py           # The 8 rules as executable checks
│   │       ├── enforcer.py               # Pre/post action validation
│   │       └── violation_log.py          # Structured violation logging
│   │
│   ├── agents\                           # NEW: Per-agent CLAUDE.md files
│   │   ├── scout.claude.md
│   │   ├── architect.claude.md
│   │   ├── provisioner.claude.md
│   │   ├── forge.claude.md
│   │   ├── crucible.claude.md
│   │   └── vision.claude.md
│   │
│   ├── workspace\                        # NEW: Symlink bridge layer + USD stage
│   │   ├── models\    → G:\COMFYUI_Database\models
│   │   ├── output\    → G:\COMFYUI_Database\output
│   │   ├── input\     → G:\COMFYUI_Database\input
│   │   ├── nodes\     → G:\COMFYUI_Database\custom_nodes
│   │   ├── comfyui\   → G:\COMFY\ComfyUI
│   │   └── stage\                        # USD-native composed stage (real .usdc files)
│   │       ├── root.usdc                 # Composed stage root
│   │       ├── base\                     # S: Specialize (templates, inventory)
│   │       ├── recipes\                  # R: Reference (learned recipes)
│   │       ├── profiles\                 # V: Variant (creative injection profiles)
│   │       ├── deltas\                   # L: Local (agent modification sublayers)
│   │       ├── scenes\                   # USD scene outputs (Path 2)
│   │       └── inputs\                   # USD scene inputs (Path 3)
│   │
│   ├── .artifacts\                       # NEW: Inter-agent artifact store
│   │   ├── _schemas\                     # JSON schemas for each artifact type
│   │   └── {session_id}\                 # Per-session artifact chain
│   │
│   ├── sessions\                         # Existing: session persistence
│   ├── workflows\                        # Existing: workflow templates
│   ├── tests\                            # Existing: 429 tests
│   ├── .env                              # Config
│   ├── CLAUDE.md                         # Existing: project-level instructions
│   └── AGENT_TEAM_BLUEPRINT.md           # This document
│
├── COMFY\ComfyUI\                        # ComfyUI software (untouched)
└── COMFYUI_Database\                     # Model repository (untouched)
```

### Launch Scripts

**Single agent (current behavior, unchanged):**
```bash
cd G:\comfyui-agent
agent run --session my-project
```

**Orchestrated multi-agent (new):**
```bash
cd G:\comfyui-agent

# Start the optimization loop on a workflow
agent orchestrate optimize \
  --workflow workflow.json \
  --target "photorealistic portrait, sharp details, natural skin" \
  --max-iterations 15 \
  --budget-minutes 30

# Run a full pipeline: scout → architect → forge → crucible
agent orchestrate build \
  --goal "Create an SDXL workflow for anime landscapes with ControlNet depth" \
  --session anime-landscapes

# Autonomous reconnaissance of your full installation
agent orchestrate scan --full
```

**Autoresearch — The Karpathy Loop (new):**
```bash
cd G:\comfyui-agent

# Overnight autonomous optimization — write program.md, go to sleep
agent autoresearch \
  --workflow portrait.json \
  --program program.md \
  --metric "aesthetic:0.4 + depth:0.2 + camera:0.2 + normals:0.1 + light:0.1" \
  --budget-hours 8 \
  --experiment-seconds 30

# Quick autoresearch — lunch break optimization
agent autoresearch \
  --workflow anime.json \
  --program anime_program.md \
  --budget-hours 1 \
  --experiment-seconds 20

# Resume a crashed/interrupted autoresearch from last kept sublayer
agent autoresearch --resume session_20260325

# View morning report from overnight run
agent autoresearch --report session_20260325
```

**Claude Code agent team (the MoE system):**
```bash
# The Router runs as the top-level Claude Code instance
# It spawns sub-agents with filtered tool access and role-specific CLAUDE.md

# Router launch (this IS the orchestrator):
claude --dangerously-skip-permissions \
  --system-prompt "$(cat agents/router.claude.md)" \
  --message "Optimize workflow portrait_v3.json for photorealism. Budget: 10 iterations."

# Under the hood, the Router spawns:
# claude --dangerously-skip-permissions --system-prompt "$(cat agents/scout.claude.md)" ...
# claude --dangerously-skip-permissions --system-prompt "$(cat agents/architect.claude.md)" ...
# etc.
```

### The Router's CLAUDE.md

```markdown
# ROUTER — Orchestration Layer

You are the ROUTER. You are NOT a specialist. You are the dispatcher, 
the queue manager, and the constitutional enforcer.

## Your Job
1. Receive a task from the human or from the self-improving loop
2. Classify it (recon? design? build? verify? analyze?)
3. Dispatch to the correct specialist agent(s)
4. Enforce the constitution on every handoff
5. Manage the artifact chain (every agent reads/writes typed artifacts)
6. Detect convergence in optimization loops
7. Surface HUMAN GATES at irreversible transitions

## Dispatch Protocol
- ALWAYS start with Scout unless the task is purely analytical
- NEVER send to Forge without an approved Design Document
- If Design Document lists MISSING dependencies → send to Provisioner BEFORE Forge
- NEVER send to Forge until Provisioner confirms all deps satisfied (PROVISION_REPORT)
- ALWAYS send to Crucible after Forge (no exceptions)
- For optimization loops: chain all agents, track iteration count
- Provisioning cost (download sizes) MUST be visible in the Human Gate summary

## Constitutional Enforcement
Before forwarding any artifact:
- Validate against schema (reject malformed)
- Check role isolation (did Scout try to modify? reject)
- Check retry count (>3 on same step? convert to BLOCKER)
- Check for stubs/TODOs (present? reject back to Forge)

## Self-Improvement Loop Management
- Track: {iteration, params, score, delta, diagnosis} for every cycle
- Apply convergence criteria after each Vision analysis
- On convergence: run meta-analysis, propose improvements
- Architecture proposals → HUMAN GATE (always)
- Data/recipe updates → auto-deploy with log

## You NEVER:
- Execute implementation yourself
- Skip the Scout phase
- Auto-approve at Human Gates
- Modify the constitution
- Suppress BLOCKER artifacts
```

---

## Self-Improvement: Three-Tier Evolution Boundary (Hyperagent Pattern)

| Layer | Tier | Mechanism |
|---|---|---|
| Parameter recipes | **Tier 1: Auto** | Memory stores, Ratchet validates |
| Routing weights | **Tier 1: Auto** | Router adjusts based on meta-analysis |
| Workflow templates | **Tier 1: Auto** | Stored as named presets in Memory |
| Tool combination patterns | **Tier 1: Auto** | Router learns which combos succeed |
| Autoresearch recipes | **Tier 1: Auto** | Ratchet flattens winning sublayers |
| Exploration strategies | **Tier 1: Auto** | Meta-agent identifies impactful axes |
| Agent prompt tuning (wording, emphasis) | **Tier 2: Ratchet** | Meta-agent proposes, ratchet proves with A/B test |
| Optimization parameters (thresholds, weights) | **Tier 2: Ratchet** | Meta-agent tunes, ratchet validates |
| Meta-agent's own modification strategy | **Tier 2: Ratchet** | Hyperagent pattern — improves how it improves |
| Structural prompt changes (new roles, sections) | **Tier 3: Human Gate** | Always requires approval |
| Constitutional commandments | **Tier 3: NEVER** | Hard-coded. Not modifiable by any agent. |
| Agent role definitions / tool access lists | **Tier 3: Human Gate** | Always requires approval |
| Scoring function (how "better" is defined) | **Tier 3: NEVER** | Constitutional anchor — prevents gaming |
| Anchor parameter definitions | **Tier 3: Human Gate** | What's protected is a human decision |
| New agent creation | **Tier 3: Human Gate** | Always requires approval |

**The line:** Tier 1 and 2 evolve autonomously (Tier 2 through ratchet proof). Tier 3 requires human approval or is permanently locked. The scoring function and constitutional commandments are the hard floor — the meta-agent can change everything about HOW it searches, but not how results are JUDGED or what rules are CONSTITUTIONAL.

---

## Build Sequence (How to Implement This)

### Phase 1: USD-Native Foundation (Weeks 1-2)
1. Move repo to `G:\comfyui-agent`, create workspace symlinks
2. `CognitiveWorkflowStage` class: `pxr.Usd.Stage` wrapper with read/write/delta/flush
3. ComfyUI workflow JSON → USD prim hierarchy bidirectional mapper
4. Bootstrap hierarchy: `/workflows`, `/recipes`, `/executions`, `/agents`, `/models`, `/scenes`
5. Anchor parameters: structural immunity (gain=1.0, no code path to modify)
6. Create `agents/` directory with CLAUDE.md files for all 6 roles
7. Create `.artifacts/_schemas/` with JSON schemas for each artifact type
8. Verify: `from pxr import Usd, UsdGeom, Sdf` imports from Houdini Python
9. Verify: microsecond latency on 200-prim test stage

### Phase 1.5: Provisioner — Self-Provisioning (Week 3)
1. `agent/provision/resolver.py` — multi-source model resolution (HF, CivitAI, CM registry)
2. `agent/provision/downloader.py` — streaming download with resume, progress, SHA256 verify
3. `agent/provision/node_installer.py` — git clone + pip install + ComfyUI refresh signal
4. `agent/provision/verifier.py` — post-download validation via ComfyUI `/object_info` API
5. `agent/provision/registrar.py` — update `/models/` prims in USD stage
6. Wire: Scout (recon) → Architect (design with deps) → [GATE] → Provisioner → Forge
7. Test: "Build me an SDXL Lightning workflow" triggers auto-download if model missing

### Phase 2: Router + Constitutional Enforcer (Weeks 4-5)
1. `agent/orchestrator/router.py` — task classification + dispatch
2. `agent/constitution/commandments.py` — the 8 rules as pre/post checks
3. `agent/orchestrator/artifact_store.py` — typed artifact I/O
4. Agent deltas as `.usdc` sublayers, LIVRPS via native USD composition
5. Wire Router into existing `agent run` as an optional mode
6. Variant sets for creative profiles in `/profiles/*.usdc`

### Phase 3: Multi-Agent Chains (Weeks 6-7)
1. `agent/orchestrator/job_queue.py` — async queue with dependencies
2. `agent/orchestrator/state_bus.py` — USD stage as shared state
3. Wire: Scout → Architect → [GATE] → Provisioner → Forge → Crucible chain
4. Derivative tools: WorkflowGaffer, AgentInspector, ConvergenceMonitor

### Phase 4: USD Scene I/O (Weeks 8-9)
1. COMPOSITOR tools: `compose_scene_from_outputs`, `validate_scene_geometry`, `compare_scenes`
2. Auxiliary passes: depth (Depth Anything V2), normals, segmentation (SAM)
3. Depth → UsdGeomMesh pipeline, image+normals → UsdShadeMaterial
4. UsdGeomCamera → ComfyUI conditioning (FOV, DoF, perspective)
5. Scene export: .usdc, .usda, .usdz, .glb
6. Multi-dimensional quality vector for self-improving loop

### Phase 5: The Ratchet — Autoresearch Mode (Weeks 10-11)
1. Ratchet class: binary keep/discard on USD sublayers against weighted multi-axis baseline
2. `program.md` parser: extract parameter axes, ranges, strategies, anchor constraints
3. Experiment runner: fixed GPU time budget, automatic queue/wait/score
4. Morning report generator: experiment count, keep rate, impact analysis, best recipe
5. `agent autoresearch` CLI entry point
6. Recipe extraction: flatten winning sublayers into Reference arc `.usdc`
7. Overnight watchdog: crash recovery, resume from last kept sublayer

### Phase 6: Claude Code MoE (Weeks 12-13)
1. Router as top-level Claude Code agent with `--dangerously-skip-permissions`
2. Sub-agent spawning with role-specific CLAUDE.md + tool filtering
3. Artifact passing between Claude Code instances
4. Three runtime modes wired: interactive, orchestrated, autoresearch
5. Full autonomous loop across all modes

### Phase 7: Hyperagent Meta-Layer (Weeks 14-16)
1. Meta-agent as 7th Claude instance with own CLAUDE.md and persistent memory
2. Three-tier classification: auto-evolve / ratchet-validated / human-gated
3. Prompt tuning pipeline: meta-agent proposes → ratchet A/B tests → KEEP or DISCARD
4. Self-referential improvement: meta-agent modifies its own modification strategy (as USD sublayer)
5. Cross-run accumulation: meta-improvements persist in `/agents/meta/` prims
6. Scoring function anchor: constitutionally protected, meta-agent cannot modify
7. Transfer test: meta-improvement from portrait domain → applied to anime domain → validate

---

## Key Design Decisions (For Architect Review)

1. **USD-native from day one.** No SQLite. No database simulating USD semantics. 
   The Cognitive Workflow Stage IS a `pxr.Usd.Stage` — real C++ composition behind 
   Python bindings. LIVRPS is built in. Composition is free. The data format IS 
   the interchange format. First principles: don't reimplement what USD already does.

2. **Router is NOT an LLM agent** in Phase 2-3. It's Python code that dispatches. 
   It becomes a Claude Code agent in Phase 6 when we need natural language task 
   classification. This keeps early phases fast and deterministic.

3. **Artifacts are JSON files on disk**, not in-memory objects. This means any agent 
   crash preserves the chain state. Restart from last artifact, not from scratch.

4. **The constitution is code, not just prompt text.** `commandments.py` runs 
   pre/post checks on every agent action. The CLAUDE.md text teaches the agent 
   WHY. The code enforces it regardless.

5. **Self-modification has a hard boundary.** Data (recipes, weights, patterns) 
   evolves freely. Architecture (roles, rules, prompts) requires human approval. 
   This boundary is constitutional — the system cannot vote to move it.

6. **The workspace symlink pattern** means agents never hardcode paths to 
   ComfyUI internals. If the database moves, update symlinks. Agents don't change.

7. **Generation outputs become USD scenes.** Image + depth + normals + segmentation 
   are composed into a real USD scene with geometry, materials, and camera data. 
   This gives the self-improving loop geometric validation alongside aesthetic scoring.

8. **USD scenes are both output AND input.** The round-trip (scene in → conditioning → 
   generation → scene out → comparison → iterate) grounds the loop in structural truth. 
   Camera prims replace prompt-based focal length descriptions. Proxy geometry replaces 
   ControlNet guesswork.

9. **The Ratchet is the Karpathy Loop on USD sublayers.** Binary keep/discard against 
   a weighted multi-axis baseline. Sublayers replace git commits (composable, non-destructive,
   branchable). Overnight mode runs 1,000+ experiments on a 4090. The program.md is the 
   research brief. The morning report is a recipe + full experiment log.

---

*The nervous system is thinking agents. The skeleton is USD. The brain is composition. The heartbeat is the ratchet.*
*v2.0.0 — THINKING AGENTS*
