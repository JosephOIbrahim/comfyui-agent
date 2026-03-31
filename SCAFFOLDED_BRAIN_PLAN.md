# SCAFFOLDED BRAIN: Execution Plan

## Brownfield Cognitive Architecture Upgrade
**Repo:** G:\comfyui-agent | **Python:** 3.12 | **ComfyUI:** localhost:8188

---

## CRITICAL RULES OF ENGAGEMENT

You are executing a **brownfield transformation** of a working ComfyUI agent. This repo contains 61 battle-tested tools, ~429 passing tests, and proven ComfyUI API integration. These are the scaffold. Your mission is to build a cognitive brain on top of this scaffold.

**Rule 1 — Preserve the Scaffold.** You are forbidden from deleting existing tools or rewriting the core ComfyUI API/WS transport. Wrap existing tools; do not destroy them.

**Rule 2 — The Test Net.** The ~429 existing tests MUST continue to pass after every change. Run `python -m pytest tests/ -v` after every file creation or modification. Breaking an existing test is higher priority than any new work.

**Rule 3 — LIVRPS is Law.** Workflow mutations are strictly non-destructive delta layers resolved via strongest-opinion-wins composition. Safety (S) is STRONGEST [Level 6]. This is an intentional inversion of standard USD LIVRPS where Specializes is weakest.

**Rule 4 — Link Preservation is Sacred.** ComfyUI represents connections as `["node_id", output_index]` arrays inside node inputs. These MUST survive all graph parsing, mutation, and round-trip serialization. This is the #1 failure mode in ComfyUI agents.

**Rule 5 — Engine Persistence.** The CognitiveGraphEngine instance MUST persist across tool calls within a session. Store it in the session state object alongside the workflow JSON. A fresh engine per tool call loses the delta stack.

**Rule 6 — MOE Roles.** You operate as a Mixture-of-Experts team. For every task, state your active configuration at the top: `[DOMAIN × ROLE]`. See SCAFFOLDED_BRAIN_AGENT_BLUEPRINT.md for the full MOE architecture and 8 Agent Commandments.

---

## WHAT EXISTS (The Scaffold You Must Preserve)

### Intelligence Layer (41 tools)
- **UNDERSTAND** (13 tools): Parse workflows, scan models/nodes, query ComfyUI API, detect format
- **DISCOVER** (8 tools): Search ComfyUI Manager (31k+ nodes), HuggingFace, CivitAI, model compatibility
- **PILOT** (13 tools): RFC6902 patch engine with undo, semantic node ops, session persistence
- **VERIFY** (7 tools): Validate, execute, WebSocket progress monitoring, execution status

### Brain Layer (20 tools)
- **Vision** (4 tools): Analyze generated images, A/B comparison, perceptual hashing
- **Planner** (4 tools): Goal decomposition, step tracking, replanning
- **Memory** (4 tools): Outcome learning with temporal decay, cross-session patterns
- **Orchestrator** (2 tools): Parallel sub-tasks with filtered tool access
- **Optimizer** (4 tools): GPU profiling, TensorRT detection, auto-apply optimizations
- **Demo** (2 tools): Guided walkthroughs

### Infrastructure
- Session persistence in `sessions/` (JSON)
- MCP server exposure for all tools
- CLI interface (`agent run`, `agent inspect`, `agent search`)
- Docker support
- ComfyUI API integration (REST + WebSocket on localhost:8188)

---

## PHASE 0: ORIENTATION (Read-Only)

**Agent Configuration:** `[SCAFFOLD × SCOUT]`
**Rule:** Do NOT modify any code. This phase is reconnaissance only.

### Actions
1. Run `python -m pytest tests/ -v` — record exact count of passing tests as baseline.
2. Walk the `agent/` directory. For every module, classify as:
   - **PRESERVE** — wrap unchanged (UNDERSTAND, DISCOVER, VERIFY, Vision, Optimizer)
   - **TRANSFORM** — replace internals, keep interface (PILOT)
   - **EXTEND** — add new capabilities on top (Memory, Orchestrator)
   - **REPLACE** — rebuild in later phase (Planner → CWM)
3. Read 2-3 existing tool implementations. Document the conventions:
   - Function signature patterns
   - MCP tool registration pattern
   - Test file naming and structure
   - Import conventions
4. Identify the specific PILOT module tools that perform destructive JSON mutation:
   - Which functions patch workflow JSON in place?
   - What are their exact function signatures?
   - Which tests cover them?
   - What session state do they read/write?

### Output
Create `MIGRATION_MAP.md` in the repo root with:
- Module-by-module classification (PRESERVE / TRANSFORM / EXTEND / REPLACE)
- PILOT tool inventory with signatures
- Convention documentation
- Baseline test count

### Verify
No code changes. `git status` should show only `MIGRATION_MAP.md` as new.

### Gate
**STOP.** Present MIGRATION_MAP.md to the user. Wait for approval before Phase 1.

---

## PHASE 1: STATE SPINE (Track A)

### Step 1 — Scout
**Agent Configuration:** `[GRAPH × SCOUT]` + `[SCAFFOLD × SCOUT]`

Identify all PILOT tools that do destructive mutation. Map their function signatures, callers, and test coverage. Document what must be wrapped and how the wrapper preserves the existing interface.

Output: Update MIGRATION_MAP.md with PILOT wrapper specifications.

### Step 2 — Architect
**Agent Configuration:** `[GRAPH × ARCHITECT]`

Design the core cognitive models. Do NOT write implementation code. Produce a design document with:
- Complete Pydantic model definitions
- CognitiveGraphEngine full interface (all methods, all signatures)
- Wrapper function signatures that match existing PILOT interfaces
- Test specifications (inputs → expected outputs) for the CRUCIBLE

The core architecture (implement these exactly):

#### Directory Structure
```
src/
└── cognitive/
    ├── __init__.py
    └── core/
        ├── __init__.py
        ├── models.py      # ComfyNode, WorkflowGraph
        ├── delta.py        # DeltaLayer with SHA-256 integrity
        └── graph.py        # CognitiveGraphEngine with LIVRPS resolver
```

#### LIVRPS Priority (Inverted S)
```python
# NOTE: USD's native LIVRPS makes Specializes weakest.
# This architecture INVERTS S to be strongest for safety-critical override.
# This is a deliberate architectural decision from patent P1v3.
LIVRPS_PRIORITY = {
    'P': 1,  # Payloads — deep archive, loaded on demand
    'R': 2,  # References — base templates, prior rules
    'V': 3,  # VariantSets — context-dependent alternatives
    'I': 4,  # Inherits — experience-derived patterns
    'L': 5,  # Local — current session edits (strongest creative opinion)
    'S': 6,  # Safety — structural constraints (INVERTED: always wins)
}
```

#### Required Engine Methods
```
__init__(base_workflow_data: Dict[str, Any])
mutate_workflow(mutations: Dict[str, Dict[str, Any]], opinion, layer_id, description) → DeltaLayer
get_resolved_graph(up_to_index: Optional[int]) → WorkflowGraph
verify_stack_integrity() → Tuple[bool, List[str]]
temporal_query(back_steps: int) → WorkflowGraph
to_api_json() → Dict[str, Any]
```

#### DeltaLayer Requirements
- `layer_id: str` — unique identifier
- `opinion: Literal["L", "I", "V", "R", "P", "S"]` — LIVRPS tier
- `timestamp: float` — UTC timestamp at creation
- `description: str` — human-readable description of what changed
- `mutations: Dict[str, Dict[str, Any]]` — `{node_id: {param: value}}`
- `creation_hash: str` — SHA-256 computed at creation, compared against `layer_hash` property for tamper detection
- `layer_hash` property recomputes SHA-256 from current opinion + mutations (sorted JSON, deterministic)

#### Resolution Logic
- Deep copy base workflow (never mutate base)
- Sort delta stack by LIVRPS priority (stable sort preserves chronological order for ties)
- Apply mutations weakest-to-strongest (last write wins = strongest opinion wins)
- For each mutation: update only specified keys in node inputs, preserving all other inputs and link arrays
- If mutation references a node not in base graph AND includes `class_type`: inject new node

Output: `TRACK_A_DESIGN.md`

### Gate
**STOP.** Present TRACK_A_DESIGN.md to user. Wait for approval before implementation.

### Step 3 — Forge (Core)
**Agent Configuration:** `[GRAPH × FORGE]`

Implement `src/cognitive/core/models.py`, `src/cognitive/core/delta.py`, `src/cognitive/core/graph.py` exactly as specified in TRACK_A_DESIGN.md. Do NOT deviate from the design. If you disagree with a design decision, note it as a comment but implement as specified.

After creating each file, run: `python -m pytest tests/ -v`
All existing tests must still pass.

### Step 4 — Forge (Wrappers)
**Agent Configuration:** `[SCAFFOLD × FORGE]`

Wrap the PILOT module tools to use CognitiveGraphEngine internally.

**Critical rules for wrappers:**
- DO NOT change function signatures
- DO NOT change MCP tool registration
- DO NOT change return value format
- The CognitiveGraphEngine instance MUST persist in session state
- Remove direct `jsonpatch` or raw dict mutation from wrapped tools
- Route through `engine.mutate_workflow()` instead

Mapping:
- `workflow_patch` → translate patch arguments into `engine.mutate_workflow()` calls
- `node_set_input` → `engine.mutate_workflow({node_id: {param: value}}, opinion="L")`
- `workflow_undo` → `engine.temporal_query(back_steps=1)`

After wrapping each tool, run: `python -m pytest tests/ -v`
All 429+ existing tests MUST still pass. If a test breaks, fix the wrapper — never the test.

### Step 5 — Crucible
**Agent Configuration:** `[GRAPH × CRUCIBLE]`

Write adversarial tests in `tests/test_cognitive_core.py`. These tests actively try to break the implementation.

**Required test categories (all mandatory, not optional):**

1. **Link preservation** — `["4", 0]` arrays survive parsing, mutation, and round-trip
2. **LIVRPS strongest-opinion-wins** — S overrides L, L overrides I, etc.
3. **SHA-256 tamper detection** — modify mutations after creation, verify `verify_stack_integrity()` catches it
4. **Temporal query rollback** — verify correct historical state at any point
5. **Multi-node atomic mutations** — single delta layer modifies multiple nodes
6. **Node injection** — delta references node not in base graph, with class_type provided
7. **Empty delta stack** — resolving with no deltas returns clean copy of base
8. **Same-opinion chronological ordering** — two L-opinion layers applied in insertion order
9. **Round-trip fidelity** — parse → mutate → to_api_json() → parse again → compare
10. **Deep copy isolation** — mutating resolved graph doesn't affect base or delta stack
11. **Existing test regression** — run full suite, confirm 429+ original tests still pass

**Rules:**
- Edge cases are mandatory. Do not skip any of the above.
- Vague assertions (`assert x`) are test bugs. Every assertion must be specific.
- If implementation fails a test: file it as a BLOCKER. Do NOT weaken the test.

### Phase 1 Complete
Run verification chain:
```bash
python -m pytest tests/ -v                           # ALL tests pass
python -m pytest tests/test_cognitive_core.py -v      # New tests pass
git diff --stat                                       # Review changes
grep -r "TODO\|FIXME\|HACK\|STUB" src/cognitive/     # Must be empty
```

Create git commit: `"Phase 1: State Spine — LIVRPS composition engine"`

**STOP.** Report results to user. Wait for approval before Phase 2.

---

## PHASE 2: TRANSPORT HARDENING (Preserve + Extend)

**Agent Pipeline:** `[TRANSPORT × SCOUT]` → `[TRANSPORT × ARCHITECT]` → GATE → `[TRANSPORT × FORGE]` → `[TRANSPORT × CRUCIBLE]`

### Goal
Strengthen existing ComfyUI communication for autonomous operation. EXTEND, don't rewrite.

### New Capabilities
1. **Schema Cache** — parse `/object_info` into typed `NodeSchema` objects. Cache in memory. Validate proposed mutations against schema BEFORE they reach the graph engine.
2. **Structured Execution Events** — parse WebSocket messages into typed `ExecutionEvent` models with computed fields (progress_pct, elapsed_ms).
3. **Interrupt endpoint** — `POST /interrupt` for mid-execution abort when prediction detects failure path.
4. **System stats** — `GET /system_stats` for VRAM usage, device info, queue depth. Enables resource-aware scheduling.

### Preservation Rule
Existing API and WebSocket code is EXTENDED with new methods/types. Do NOT rewrite existing working transport code. Add alongside it.

### Schema Validation (Critical for Autonomy)
```python
class SchemaCache:
    async def refresh(self, api_client)           # Fetch and parse /object_info
    def validate_mutation(self, class_type, param_name, param_value) → (bool, str)
    def get_valid_values(self, class_type, param_name) → list | None
    def get_connectable_nodes(self, target_class_type, target_input) → list[str]
```

`validate_mutation` is the gatekeeper. In Phase 3, no mutation reaches the graph engine without passing schema validation. This prevents the #1 failure mode: invalid parameter combinations that produce cryptic ComfyUI execution errors.

### Test Gate
All tests mocked — no live ComfyUI dependency. Test schema parsing, validation (valid pass, invalid reject with reason), event type parsing, progress computation.

---

## PHASE 3: TOOL CONSOLIDATION

**Agent Pipeline:** `[SCAFFOLD × SCOUT]` → `[AUTONOMY × ARCHITECT]` → GATE → `[SCAFFOLD × FORGE]` + `[AUTONOMY × FORGE]` → `[AUTONOMY × CRUCIBLE]`

### Goal
61 tools → 8 macro-tools + MCP adapter layer.

### The 8 Macro-Tools
| Tool | Absorbs | What It Does |
|------|---------|--------------|
| `analyze_workflow()` | UNDERSTAND tools | Semantic analysis + validation + resource estimate |
| `mutate_workflow()` | PILOT tools | Schema-validated non-destructive mutation |
| `query_environment()` | UNDERSTAND scan + DISCOVER | Unified environment snapshot |
| `manage_dependencies()` | DISCOVER install | Custom node management + schema cache invalidation |
| `execute_workflow()` | VERIFY tools | Submit + monitor + evaluate + retry |
| `compose_workflow()` | NEW | Build workflow from intent + experience |
| `generate_series()` | NEW | Multi-image with style consistency |
| `autoresearch()` | NEW | Karpathy ratchet for optimization |

### MCP Adapter
Same engine, configurable surface area per consumer. The adapter reshapes macro-tools into granular endpoints for LLM consumers that need finer granularity.

### New Modules
```
agent/
├── composer.py       # Workflow composition from creative intent
├── evaluator.py      # Output self-evaluation (rule-based, ML-ready later)
├── orchestrator.py   # Multi-generation: batch, series, A/B, autoresearch
└── retry.py          # Adaptive retry by failure type
```

---

## PHASE 4: EXPERIENCE ACCUMULATOR (Track B)

**Agent Pipeline:** `[EXPERIENCE × SCOUT]` → `[EXPERIENCE × ARCHITECT]` → GATE → `[EXPERIENCE × FORGE]` → `[EXPERIENCE × CRUCIBLE]`

### Goal
Every generation is an experiment. Structured learning from outcomes.

### The Experience Loop
```
CAPTURE initial state → PREDICT outcome → GENERATE → CAPTURE outcomes →
COMPARE prediction vs actual → STORE experience tuple → GENERATE counterfactual
```

### Key Schemas
- `ExperienceChunk` — full (params → outcome) tuple per generation
- `GenerationContextSignature` — discretized parameter space for fast matching
- USD-native persistence under `/experience/generations/`

### Three Learning Phases
- Phase 1 (0-30 gens): Prior rules only
- Phase 2 (30-100 gens): Blended prior + experience
- Phase 3 (100+ gens): Experience-dominant

### Integration Points
- Wire into `execute_workflow()` for automatic experience capture
- Wire Vision module outputs as quality signals
- Wire existing Memory module as storage backend

---

## PHASE 5: COGNITIVE WORLD MODEL (Track C)

**Agent Pipeline:** `[PREDICTION × SCOUT]` → `[PREDICTION × ARCHITECT]` → GATE → `[PREDICTION × FORGE]` → `[PREDICTION × CRUCIBLE]`

### Goal
Predict generation outcomes from accumulated experience. Central claim: LIVRPS composition serves BOTH state resolution AND prediction resolution. One engine, two functions.

### Components
- **CWM** — composes predictions from experience + prior + counterfactuals via LIVRPS
- **Simulation Arbiter** — Silent (80%) / Soft Surface (15%) / Explicit (5%) delivery
- **Counterfactual Generator** — one "what if" per generation, validated against future data

---

## PHASE 6: AUTONOMOUS PIPELINE

**Agent Pipeline:** `[AUTONOMY × SCOUT]` → `[AUTONOMY × ARCHITECT]` → GATE → all domain experts × FORGE → `[AUTONOMY × CRUCIBLE]`

### Goal
End-to-end: creative intent → composed workflow → predicted outcome → execution → evaluation → learning. Zero human intervention.

### The End State
```
INPUT: "cinematic portrait, golden hour, film grain aesthetic"

AGENT:
  1. Classifies intent from experience
  2. Selects model by capability, not name
  3. Composes workflow from learned parameter combinations
  4. Predicts output quality before executing
  5. Generates, evaluates, corrects if needed
  6. Stores full experiment for future learning
  7. Gets measurably better over time
```

---

## ENVIRONMENT REFERENCE

```
Repository:     G:\comfyui-agent
Python:         3.12 (venv at G:\comfyui-agent\.venv312)
ComfyUI:        G:\COMFY\ComfyUI (localhost:8188)
Model DB:       G:\COMFYUI_Database
USD:            usd-core (in-process, zero latency)
GPU:            RTX 4090
RAM:            128GB DDR5
CPU:            Threadripper PRO 7965WX
Overnight:      autobuild.ps1 + BUILD_QUEUE.md
```

## CIRCUIT BREAKER

```
ATTEMPT 1: Fix the code to pass the test.
ATTEMPT 2: Different approach to fix the code.
ATTEMPT 3: Last attempt.

If all 3 fail → Create BLOCKER.md with:
  - What was attempted (all 3 approaches)
  - What failed and why
  - What you think the root cause is
  - What would unblock it

STOP. Do not proceed past a blocker.
Do not weaken a test to make it pass.
Do not skip a requirement.
Wait for human resolution.
```
