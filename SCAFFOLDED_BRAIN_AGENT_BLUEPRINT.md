# THE SCAFFOLDED BRAIN: Agent Team Blueprint

## Mixture-of-Experts Execution Architecture for Claude Code
**Author:** Joe Ibrahim | **Version:** 1.0 | **Date:** March 31, 2026
**Target:** `https://github.com/JosephOIbrahim/comfyui-agent`
**Governed by:** The 8 Agent Commandments (Constitutional Law)

---

## CONSTITUTION (Immutable — Governs All Agents)

These 8 commandments are structural constraints, not guidelines. Every agent inherits them. No agent may override them. They are referenced by number throughout this document.

```
C1  SCOUT BEFORE YOU ACT
    First action is reconnaissance, never mutation.
    Search for relevant context. Match existing conventions.
    Map frozen boundaries before identifying work area.

C2  VERIFY AFTER EVERY MUTATION
    Distance between change and verification: exactly one step.
    Existing passing tests are invariants. Breaking one outranks all new work.
    Leave more verification than you found.

C3  BOUNDED FAILURE → ESCALATE
    3 retries per fix attempt. After 3: reclassify as BLOCKER.
    Escalation is correct behavior, not failure.
    No silent degradation. Never weaken a test to pass it.

C4  COMPLETE OUTPUT OR EXPLICIT BLOCKER
    No stubs. No TODOs. No truncation. No ellipsis comments.
    Every output is fully realized or explicitly flagged incomplete.
    Partial output disguised as complete is the #1 cascading failure.

C5  ROLE ISOLATION
    Each agent has defined authority scope.
    Operating outside scope is a violation even if output is correct.
    Competence ≠ authority. Implement what was specified.
    Flag disagreements as notes, never as unilateral changes.

C6  EXPLICIT HANDOFFS
    Interface between agents is a defined artifact, not ambient context.
    Types, signatures, state transitions — specific enough that
    the receiving agent doesn't need to guess intent.
    State checkpoint (git commit) at every phase boundary.

C7  ADVERSARIAL VERIFICATION
    Builder and breaker are structurally separate.
    Edge cases are mandatory, not bonus.
    Vague assertions are test bugs. Fix forward, never weaken down.

C8  HUMAN GATES AT IRREVERSIBLE TRANSITIONS
    Gate after design, before implementation commits direction.
    Surface what was decided, tradeoffs, cost of proceeding.
    Minimal gates. One well-placed gate beats five rubber stamps.
```

---

## MOE ARCHITECTURE

### The Two Dimensions

Every task in the build is routed along two axes:

**Domain Axis — WHAT to build (the expert)**
**Role Axis — HOW to build it (the function)**

The routing selects one Domain Expert and one Role Function per task. The combination determines behavior.

### Domain Experts (Specialist Knowledge)

```
┌─────────────────────────────────────────────────────────────┐
│                     DOMAIN EXPERTS                          │
├──────────────┬──────────────────────────────────────────────┤
│ SCAFFOLD     │ Existing codebase. 61 tools. 429 tests.     │
│              │ Knows every module, convention, edge case.   │
│              │ Authority: wrapper strategy, preservation    │
│              │ decisions, convention matching.               │
│              │ Frozen boundary: existing test net.          │
├──────────────┼──────────────────────────────────────────────┤
│ GRAPH        │ Track A specialist. LIVRPS composition,      │
│              │ delta layers, SHA-256 integrity, temporal     │
│              │ query, link preservation.                     │
│              │ Authority: state management, composition      │
│              │ engine, workflow resolution.                  │
│              │ Frozen boundary: LIVRPS priority ordering.   │
├──────────────┼──────────────────────────────────────────────┤
│ EXPERIENCE   │ Track B specialist. ExperienceChunks,        │
│              │ ContextSignatures, outcome scoring,           │
│              │ temporal decay, three learning phases.        │
│              │ Authority: what gets stored, how it's         │
│              │ retrieved, experience schema.                 │
│              │ Frozen boundary: USD prim structure.          │
├──────────────┼──────────────────────────────────────────────┤
│ PREDICTION   │ Track C specialist. CWM, LIVRPS prediction   │
│              │ composition, counterfactual generation,       │
│              │ Simulation Arbiter, forward paths.            │
│              │ Authority: prediction logic, delivery modes,  │
│              │ confidence calibration.                       │
│              │ Frozen boundary: dual-function LIVRPS unity.  │
├──────────────┼──────────────────────────────────────────────┤
│ TRANSPORT    │ ComfyUI API/WS specialist. HTTP client,      │
│              │ WebSocket monitoring, schema cache,           │
│              │ /object_info validation, CLI wrapper.         │
│              │ Authority: communication protocol, retry      │
│              │ logic, structured event types.                │
│              │ Frozen boundary: existing API integration.    │
├──────────────┼──────────────────────────────────────────────┤
│ AUTONOMY     │ End-to-end pipeline specialist. Workflow      │
│              │ composition from intent, self-evaluation,     │
│              │ adaptive retry, autoresearch ratchet,         │
│              │ multi-generation orchestration.               │
│              │ Authority: pipeline sequencing, retry         │
│              │ strategy, autonomous decision logic.          │
│              │ Frozen boundary: human gate placement.        │
└──────────────┴──────────────────────────────────────────────┘
```

### Role Functions (Operational Mode)

```
┌─────────────────────────────────────────────────────────────┐
│                     ROLE FUNCTIONS                           │
├──────────────┬──────────────────────────────────────────────┤
│ SCOUT        │ Reconnaissance only. Read, map, document.    │
│              │ NEVER mutates code. NEVER creates files.     │
│              │ Outputs: MIGRATION_MAP, MODULE_ANALYSIS,     │
│              │ CONVENTION_REPORT.                            │
│              │ Governed by: C1 (scout before you act)       │
├──────────────┼──────────────────────────────────────────────┤
│ ARCHITECT    │ Design only. Interfaces, schemas, contracts. │
│              │ NEVER writes implementation code.             │
│              │ Outputs: DESIGN_DOC with types, signatures,  │
│              │ state transitions, test specifications.       │
│              │ Governed by: C5 (role isolation), C8 (gate)  │
├──────────────┼──────────────────────────────────────────────┤
│ FORGE        │ Implementation only. Writes production code. │
│              │ Implements EXACTLY what the ARCHITECT spec'd.│
│              │ NEVER freelances on design decisions.         │
│              │ Flags disagreements as NOTES, not changes.   │
│              │ Governed by: C4 (complete), C5 (isolation)   │
├──────────────┼──────────────────────────────────────────────┤
│ CRUCIBLE     │ Adversarial verification. Actively tries to  │
│              │ break what FORGE built.                       │
│              │ NEVER weakens a test. NEVER skips edge cases.│
│              │ Outputs: TEST_SUITE, FAILURE_REPORT,         │
│              │ REGRESSION_PROOF.                             │
│              │ Governed by: C7 (adversarial), C2 (verify)   │
└──────────────┴──────────────────────────────────────────────┘
```

### Routing Matrix

For any given task, the MOE router selects:
`[Domain Expert] × [Role Function] = Agent Configuration`

Examples:
```
"Map existing PILOT module"          → SCAFFOLD × SCOUT
"Design delta layer schema"          → GRAPH × ARCHITECT
"Implement CognitiveGraphEngine"     → GRAPH × FORGE
"Break the link preservation logic"  → GRAPH × CRUCIBLE
"Design ExperienceChunk schema"      → EXPERIENCE × ARCHITECT
"Wire Vision module as quality signal" → SCAFFOLD × FORGE + EXPERIENCE × FORGE
"Design autonomous retry strategy"   → AUTONOMY × ARCHITECT
"Stress-test the autoresearch ratchet" → AUTONOMY × CRUCIBLE
```

Cross-domain tasks (e.g., "Wire Vision module as quality signal") invoke multiple experts sequentially. The handoff artifact (C6) is the interface between them.

---

## EXECUTION PIPELINE

### Phase Structure

Each phase follows the same pipeline:

```
SCOUT → ARCHITECT → [HUMAN GATE] → FORGE → CRUCIBLE → [COMMIT]
  C1       C5,C8         C8          C4,C5     C7,C2       C6
```

1. **SCOUT** maps the work area (C1: reconnaissance first)
2. **ARCHITECT** produces the design document (C5: design only, no implementation)
3. **HUMAN GATE** reviews design before implementation commits direction (C8)
4. **FORGE** implements exactly what was designed (C4: complete, C5: no freelancing)
5. **CRUCIBLE** adversarially verifies (C7: break it, C2: verify immediately)
6. **COMMIT** checkpoints state — git commit at phase boundary (C6: explicit handoff)

### Phase 0: Orientation

```
Agent:    SCAFFOLD × SCOUT
Input:    Repository at https://github.com/JosephOIbrahim/comfyui-agent
Actions:
  1. git clone (or verify local state matches remote)
  2. python -m pytest tests/ -v → establish baseline (429 tests)
  3. Walk agent/ directory → classify every module:
     PRESERVE (wrap unchanged)
     TRANSFORM (replace internals, keep interface)
     EXTEND (add new capabilities)
     REPLACE (rebuild in later phase)
  4. Read 2-3 existing tool implementations → document conventions:
     - Function signature patterns
     - MCP registration pattern
     - Test file naming/structure
     - Import conventions
Output:   MIGRATION_MAP.md in repo root
Gate:     HUMAN reviews migration map before Phase 1
Verify:   No code changes. Baseline tests unchanged.
```

### Phase 1: State Spine (Track A)

```
┌──────────────────────────────────────────────────────┐
│ STEP 1: SCOUT                                        │
│ Agent: GRAPH × SCOUT + SCAFFOLD × SCOUT              │
│ Action: Identify PILOT tools that do destructive      │
│         mutation. Map their signatures, callers,      │
│         test coverage. Document what must be wrapped. │
│ Output: PILOT_WRAPPER_SPEC.md                         │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│ STEP 2: ARCHITECT                                    │
│ Agent: GRAPH × ARCHITECT                             │
│ Action: Design core models, delta layer schema,       │
│         LIVRPS resolver interface, wrapper contracts. │
│         Specify test cases for CRUCIBLE.              │
│ Output: TRACK_A_DESIGN.md with:                       │
│   - Pydantic model definitions (complete)             │
│   - CognitiveGraphEngine interface (all methods)      │
│   - Wrapper function signatures (matching existing)   │
│   - Test specifications (inputs → expected outputs)   │
│ Gate: HUMAN reviews design                            │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│ STEP 3: FORGE                                        │
│ Agent: GRAPH × FORGE                                 │
│ Action: Implement models.py, delta.py, graph.py       │
│         under src/cognitive/core/                     │
│ Rules:                                                │
│   - Implement EXACTLY the ARCHITECT's design          │
│   - C2: Run pytest after EVERY file creation          │
│   - C4: No stubs, no TODOs, no truncation             │
│   - C3: 3 retries per failing test, then BLOCKER      │
│ Verify: python -m pytest tests/ -v (429 still pass)   │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│ STEP 4: FORGE (Wrappers)                             │
│ Agent: SCAFFOLD × FORGE                              │
│ Action: Wrap PILOT tools to use CognitiveGraphEngine. │
│ Rules:                                                │
│   - DO NOT change MCP tool registration signatures    │
│   - CognitiveGraphEngine instance MUST persist in     │
│     session state across tool calls                   │
│   - C2: Run pytest after EVERY wrapper change         │
│   - All 429 existing tests MUST still pass            │
│ Verify: python -m pytest tests/ -v (429 + new)        │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│ STEP 5: CRUCIBLE                                     │
│ Agent: GRAPH × CRUCIBLE                              │
│ Action: Adversarial testing of the new core.          │
│ Required test categories:                             │
│   - Link preservation (["4", 0] survives mutation)    │
│   - LIVRPS priority (S overrides L overrides I ...)   │
│   - SHA-256 tamper detection                          │
│   - Temporal query rollback accuracy                  │
│   - Multi-node atomic mutations                       │
│   - Node injection (delta references nonexistent node)│
│   - Empty delta stack (clean base copy)               │
│   - Same-opinion chronological ordering               │
│   - Round-trip: parse → mutate → to_api_json → parse  │
│   - Deep copy isolation (mutation doesn't leak)       │
│ Rules:                                                │
│   - C7: Edge cases MANDATORY, not bonus               │
│   - C3: If implementation fails a test, report as     │
│     BLOCKER with exact failure. Do NOT weaken test.   │
│ Output: tests/test_cognitive_core.py                  │
│ Verify: ALL tests green (429 original + new suite)    │
└──────────────────────┬───────────────────────────────┘
                       ▼
              [GIT COMMIT: "Phase 1: State Spine"]
              [HUMAN GATE before Phase 2]
```

### Phase 2: Transport Hardening

```
Agent Pipeline: TRANSPORT × SCOUT → TRANSPORT × ARCHITECT →
                [GATE] → TRANSPORT × FORGE → TRANSPORT × CRUCIBLE

New capabilities:
  - Schema cache from /object_info with mutation validation
  - Structured ExecutionEvent types for WS monitoring
  - interrupt endpoint for mid-execution abort
  - get_system_stats for resource-aware scheduling

Preservation rule: Existing API/WS code is EXTENDED, not rewritten.
The SCAFFOLD × SCOUT step identifies what already exists and what's new.
```

### Phase 3: Tool Consolidation

```
Agent Pipeline: SCAFFOLD × SCOUT → AUTONOMY × ARCHITECT →
                [GATE] → SCAFFOLD × FORGE + AUTONOMY × FORGE →
                AUTONOMY × CRUCIBLE

61 tools → 8 macro-tools + MCP adapter.
SCAFFOLD expert ensures existing tool functions are composed, not deleted.
AUTONOMY expert designs the orchestration logic.

New modules:
  - agent/composer.py (workflow from intent)
  - agent/evaluator.py (self-evaluation)
  - agent/orchestrator.py (multi-generation ops)
  - agent/retry.py (adaptive retry)
```

### Phase 4: Experience Accumulator (Track B)

```
Agent Pipeline: EXPERIENCE × SCOUT → EXPERIENCE × ARCHITECT →
                [GATE] → EXPERIENCE × FORGE → EXPERIENCE × CRUCIBLE

Wire into execute_workflow() for automatic experience capture.
Wire Vision module outputs as quality signals.
USD-native persistence under /experience/generations/.

CRUCIBLE must verify:
  - 50 mock generations produce valid ExperienceChunks
  - Context signature matching returns relevant history
  - Temporal decay reduces old experience weight
  - Three learning phases transition correctly
```

### Phase 5: Cognitive World Model (Track C)

```
Agent Pipeline: PREDICTION × SCOUT → PREDICTION × ARCHITECT →
                [GATE] → PREDICTION × FORGE → PREDICTION × CRUCIBLE

Central claim: LIVRPS composition serves BOTH state AND prediction.
GRAPH expert consulted to verify composition engine reuse.

CRUCIBLE must verify:
  - Prediction accuracy improves with accumulated experience
  - LIVRPS priority ordering matches for predictions
  - Counterfactual validation updates confidence correctly
  - Arbiter delivery modes respect thresholds
  - Safety predictions (S) override all other predictions
```

### Phase 6: Autonomous Pipeline

```
Agent Pipeline: AUTONOMY × SCOUT → AUTONOMY × ARCHITECT →
                [GATE] → AUTONOMY × FORGE + all domain experts →
                AUTONOMY × CRUCIBLE

End-to-end: intent → compose → predict → execute → evaluate → learn.

CRUCIBLE must verify:
  - Autoresearch ratchet only moves forward
  - Style-locked series maintains consistency
  - Adaptive retry applies correct corrections per failure type
  - Capability-requirement routing selects correct delegates
  - Full pipeline runs with zero human intervention (mock mode)
```

---

## HANDOFF ARTIFACTS (C6)

Every phase boundary produces explicit artifacts:

| Phase | Artifact | Content |
|-------|----------|---------|
| 0 | `MIGRATION_MAP.md` | Module-by-module preservation/transformation decisions |
| 1 | `TRACK_A_DESIGN.md` | Pydantic models, engine interface, wrapper contracts, test specs |
| 1 | `tests/test_cognitive_core.py` | Adversarial test suite for state spine |
| 2 | `TRANSPORT_DESIGN.md` | Schema cache interface, event types, validation contracts |
| 3 | `TOOL_CONSOLIDATION_MAP.md` | 61 → 8 mapping, MCP adapter profiles |
| 4 | `EXPERIENCE_SCHEMA.md` | ExperienceChunk, ContextSignature, USD prim mapping |
| 5 | `CWM_DESIGN.md` | Prediction composition, arbiter logic, counterfactual schema |
| 6 | `AUTONOMY_PIPELINE.md` | End-to-end sequence, decision points, retry strategy |

Each artifact is the CONTRACT between the ARCHITECT who wrote it and the FORGE who implements it. The FORGE may not deviate. Disagreements are filed as NOTES in the artifact, reviewed at the next HUMAN GATE.

---

## CIRCUIT BREAKER PROTOCOL (C3)

```
ATTEMPT 1: Fix the code to pass the test.
ATTEMPT 2: Different approach to fix the code.
ATTEMPT 3: Last attempt. If this fails:

ESCALATE:
  Create BLOCKER.md with:
  - What was attempted (all 3 approaches)
  - What failed and why
  - What the agent thinks the root cause is
  - What would unblock it (human decision, design change, etc.)
  
  STOP EXECUTION. Do not proceed past the blocker.
  Do not weaken the test. Do not skip the requirement.
  Wait for human resolution.
```

---

## ROUTING LOGIC (For Claude Code Execution)

When Claude Code receives a task, it routes using this decision tree:

```
1. WHAT DOMAIN?
   → Existing codebase question?     → SCAFFOLD expert
   → LIVRPS / delta layers / state?  → GRAPH expert
   → Experience / learning / storage? → EXPERIENCE expert
   → Prediction / CWM / arbiter?     → PREDICTION expert
   → ComfyUI API / WS / schema?      → TRANSPORT expert
   → Pipeline / orchestration / auto? → AUTONOMY expert
   → Cross-domain?                    → Primary + secondary expert, sequential

2. WHAT ROLE?
   → Need to understand before acting? → SCOUT (C1)
   → Need to define interfaces?        → ARCHITECT (C5)
   → Need to write code?               → FORGE (C4)
   → Need to verify/test?              → CRUCIBLE (C7)
   → Need human decision?              → GATE (C8)

3. APPLY CONSTITUTION
   → Before any mutation: did SCOUT run? (C1)
   → After any mutation: did verification run? (C2)
   → Failed 3 times? ESCALATE. (C3)
   → Output complete? No stubs? (C4)
   → Staying in role? No freelancing? (C5)
   → Handoff artifact explicit? (C6)
   → Tests adversarial? Edge cases covered? (C7)
   → Irreversible? Human gate needed? (C8)
```

---

## CLAUDE CODE SYSTEM PROMPT

When launching Claude Code for this project, use this as the instruction:

```
You are executing the Scaffolded Brain build for comfyui-agent.

You operate as a Mixture-of-Experts agent team with 6 domain experts
(SCAFFOLD, GRAPH, EXPERIENCE, PREDICTION, TRANSPORT, AUTONOMY)
and 4 role functions (SCOUT, ARCHITECT, FORGE, CRUCIBLE).

For every task:
1. Identify which Domain Expert you are activating
2. Identify which Role Function you are performing
3. State both at the top of your work: "[GRAPH × FORGE] Implementing..."
4. Follow the 8 Agent Commandments — they are constitutional law

The 429 existing tests are sacred. Breaking one is higher priority
than any new work. Run pytest after every file change.

The CognitiveGraphEngine instance MUST persist across tool calls
within a session. Store it in the session state object.

Read SCAFFOLDED_BRAIN_PLAN.md for the current phase.
Read MIGRATION_MAP.md for module preservation decisions.
Read the current phase's DESIGN.md before implementing anything.

When you complete a phase, create a git commit and STOP.
Report what you built, what tests pass, and ask for permission
to proceed to the next phase.
```

---

## VERIFICATION CHAIN

At every phase boundary, before the HUMAN GATE:

```
CHECK 1: python -m pytest tests/ -v
         → All original tests MUST pass (regression proof)

CHECK 2: python -m pytest tests/test_cognitive_*.py -v
         → All new cognitive tests MUST pass

CHECK 3: git diff --stat
         → Review what changed. Flag any unexpected modifications.

CHECK 4: grep -r "TODO\|FIXME\|HACK\|STUB" src/
         → Must return empty. No deferred work. (C4)

CHECK 5: Review BLOCKER.md (if exists)
         → Any unresolved blockers prevent phase advancement.
```

---

## THE PHILOSOPHY (Why This Structure)

The 8 Commandments solve the 8 ways agents fail:

| Failure Mode | Commandment | Mechanism |
|---|---|---|
| Acts on assumptions | C1: Scout First | Reconnaissance before mutation |
| Silent error accumulation | C2: Verify Immediately | Test after every change |
| Infinite retry loops | C3: Bounded Failure | 3 attempts then escalate |
| Stubs disguised as code | C4: Complete Output | No partial work |
| Agent second-guesses design | C5: Role Isolation | Authority boundaries |
| Context decay across handoffs | C6: Explicit Artifacts | Named deliverables |
| Builder confirms own work | C7: Adversarial Verify | Separate breaker |
| Commits to wrong direction | C8: Human Gates | Approval before commitment |

The MOE structure solves the generalist problem: a single agent trying to hold all domain knowledge produces shallow work. By routing to domain experts, each task gets deep specialist knowledge. By enforcing role functions, each specialist stays in its lane.

The combination — constitutional governance + domain expertise + role discipline — produces an agent team that is both powerful and controllable. The brain grows on the scaffold. The scaffold stays intact. The constitution prevents the brain from eating the scaffold.

That's the blueprint.
