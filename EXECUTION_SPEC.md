# COMFYUI-3D-VIEWPORT — EXECUTION SPEC
## For: Claude Code + Agent Teams
## Repo: github.com/JosephOIbrahim/comfyui-3D-viewport
## Parent: C:\Users\User\comfyui-agent\
## Date: 2026-02-19

---

# HOW TO USE THIS FILE

1. Save as `EXECUTION_SPEC.md` in the comfyui-3D-viewport repo root
2. The existing `CLAUDE.md` and `SUPERDUPER_3D_GRAFT.md` stay as-is
3. Tell Claude Code: `Read EXECUTION_SPEC.md. Execute the current phase.`
4. **All code changes happen in `C:\Users\User\comfyui-agent\` (parent)**
5. This repo holds planning docs + reference data only — no source code lives here

**Claude Code: Execute ONE phase at a time. Stop and report at gates.**

---

# SITUATION ASSESSMENT

## What's Done
- **Grafts A–D: SHIPPED.** 665 tests green. Pushed.
  - Graft A: 3D data types (MESH, VOXEL, POINT_CLOUD, CAMERA, POSE) in UNDERSTAND
  - Graft B: Partner Node `source_tier` field + comparative knowledge in DISCOVER
  - Graft C: Splat-to-mesh conversion path knowledge
  - Graft D: Viewport tool discovery (VNCCS, Action Director, 3DView)

## What's In Flight
- **Graft F: ProfilerX in VERIFY** — Separate dispatch (DISPATCH_PROFILERX.md). Runs independently.

## What's NOT Done
- **Graft E:** Demo scenarios — waiting on Phase 5 (Multi-Pipe)
- **SUPERDUPER_UI_PLAN Phases 0–3:** Likely do not exist as shipped code yet
- **Frontend surface:** The graft knowledge is injected into agent context but has NO sidebar, NO badges, NO DISCOVER panel

## The Gap
Grafts A–D added backend knowledge. Without Phases 1–3 (sidebar, backend bridge, DISCOVER panel), artists can't SEE this knowledge through a UI. The question Phase 1 answers: **can they access it through conversation?**

## Two Parallel Tracks

```
AGENT TRACK (comfyui-agent)          VIEWPORT TRACK (comfyui-3D-viewport)
─────────────────────────────        ──────────────────────────────────────
68 tools, 4 layers                   Started as graft layer for agent
Grafts A-D shipped                   Evolving toward standalone 3D viewport
SUPERDUPER_UI_PLAN phases            CarWash codebase as reference
Conversational interface             Future: real cameras, Hydra, ComfyUI node
                    \                /
                     \              /
                      CONVERGENCE
              ┌─────────────────────────┐
              │ Agent understands 3D    │
              │ Viewport renders 3D     │
              │ Agent drives viewport   │
              │ Load3D compatibility     │
              └─────────────────────────┘
```

Each track ships value independently. Convergence is future.

---

# PHASE MAP

```
PHASE 1: Verify graft knowledge is usable          [DO NOW]
    │     (codebase audit + live tests)
    │                                          ┌──────────────────────────────┐
    │                                          │ GRAFT F: ProfilerX in VERIFY │
    │                                          │ (DISPATCH_PROFILERX.md)      │
    │                                          │ Runs independently / parallel│
    ▼                                          └──────────────────────────────┘
PHASE 2: Fix knowledge routing gaps                 [IF NEEDED]
    │     (trigger keywords, response formatting)
    │     NOTE: Coordinate with Graft F if both touch _KNOWLEDGE_TRIGGERS
    ▼
PHASE 3: Graft E test preparation                   [NEXT SPRINT]
    │     (mock workflows, test fixtures)
    ▼
PHASE 4: LOAD3D_CAMERA compatibility                [CONVERGENCE PREP]
    │     (teach agent about camera pipeline)
    ▼
PHASE 5: Viewport architecture planning             [DESIGN ONLY]
          (decision doc, schema, integration contract)
          │
          ▼
    ┌─────────────────────────────────────────┐
    │  FUTURE: Viewport implementation        │
    │  (separate execution spec when ready)   │
    │                                         │
    │  Uses: CarWash as reference             │
    │  Tech: TBD (custom node / standalone)   │
    │  Camera data: camera_lens_database.json │
    │  Protocol: LOAD3D_CAMERA compatible     │
    └─────────────────────────────────────────┘
```

---

# PHASE 1: VERIFY GRAFT KNOWLEDGE IS USABLE
**Type:** [AGENT-GRIND] — Autonomous audit, no owner decisions needed
**Duration:** ~30 minutes
**Working directory:** `C:\Users\User\comfyui-agent\`

## Objective
Confirm Grafts A–D work in conversation — that knowledge is routed to responses, not just stored.

## 1.1 Codebase Audit

Answer all questions by inspecting code. Do not guess.

### Q1: SUPERDUPER_UI_PLAN Phase Status

Search the codebase for evidence of shipped phases:

```bash
# Phase 0 (Recon): Any audit/scan output?
grep -r "recon\|audit\|scan" agent/ --include="*.py" -l

# Phase 1 (Sidebar Skeleton): sidebar/panel/drawer components?
grep -r "sidebar\|panel\|drawer" agent/ --include="*.py" --include="*.js" --include="*.html" -l
find . -name "*.html" -o -name "*.js" -o -name "*.tsx" -o -name "*.jsx" | head -20

# Phase 2 (Backend Bridge): API routes, bridge, connector code?
grep -r "bridge\|/api/\|fastapi\|flask" agent/ --include="*.py" -l

# Phase 3 (DISCOVER Panel): badge/partner/community rendering?
grep -r "badge\|🤝\|Partner\|community.*badge" agent/ --include="*.py" --include="*.js" -l
```

Report format:
```
Phase 0 (Recon):     [EXISTS path/to/file | NOT FOUND]
Phase 1 (Sidebar):   [EXISTS path/to/file | NOT FOUND]
Phase 2 (Bridge):    [EXISTS path/to/file | NOT FOUND]
Phase 3 (DISCOVER):  [EXISTS path/to/file | NOT FOUND]
```

### Q2: Knowledge Routing

```bash
# What triggers load 3d_workflows.md?
grep -n "3d_workflows" agent/system_prompt.py

# What other 3D knowledge files exist?
ls agent/knowledge/ | grep -i "partner\|3d\|mesh\|viewport\|camera\|splat"

# What's in _KNOWLEDGE_TRIGGERS?
grep -A 50 "_KNOWLEDGE_TRIGGERS" agent/system_prompt.py | head -60
```

**Trace the path:** When a user asks "what's the best 3D generator?", does the agent receive comparative knowledge from Graft B in its system prompt context? Follow the code from user input → trigger matching → knowledge injection → response generation.

Report format:
```
Knowledge files:               [list all 3D-related .md files]
Trigger keywords for each:     [keyword → file mappings]
Comparative knowledge reachable via conversation: [YES with path | NO with gap description]
```

### Q3: source_tier Field

```bash
# Is source_tier in comfy_discover.py's return schema?
grep -n "source_tier\|partner\|community\|core" agent/tools/comfy_discover.py

# Is it surfaced in any response formatting?
grep -rn "source_tier" agent/ --include="*.py"
```

Report format:
```
source_tier in DISCOVER schema: [YES line N | NO]
source_tier in response formatting: [YES how | NO]
```

### Q4: Test Coverage

```bash
python -m pytest tests/ -v --co -q 2>&1 | grep -ci "3d\|mesh\|partner\|graft\|splat\|viewport\|voxel\|point_cloud\|camera.*pose"

# List the actual test names
python -m pytest tests/ -v --co -q 2>&1 | grep -i "3d\|mesh\|partner\|graft\|splat\|viewport\|voxel\|point_cloud"
```

Report format:
```
3D-related test count: [N]
Test names: [list]
```

### Q5: LOAD3D_CAMERA Schema (Future Convergence — Non-Blocking)

```bash
# Clone ComfyUI source (shallow)
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /tmp/comfyui-src 2>/dev/null

# Find LOAD3D_CAMERA type definition
grep -rn "LOAD3D_CAMERA\|load_3d\|Load3D\|camera_info" /tmp/comfyui-src/ \
  --include="*.py" | head -30

# Check AdvancedCameraControlNode
git clone --depth 1 https://github.com/jandan520/ComfyUI-AdvancedCameraPrompts.git \
  /tmp/adv-camera 2>/dev/null
grep -rn "camera_info\|focal_length\|object_scale\|LOAD3D" /tmp/adv-camera/ \
  --include="*.py" | head -30
```

Report format:
```
LOAD3D_CAMERA schema: { fields } or "could not determine"
AdvancedCameraControlNode expects: { fields } or "could not determine"
```

## 1.2 Live Conversation Tests

If the agent can run (`agent run` or via MCP), execute these tests:

**Test 1: Mesh Generation Comparison**
```
Ask: "What can I use for 3D mesh generation?"

PASS if: Partner Nodes (Hunyuan, Meshy, Tripo, Rodin) mentioned with any
         trust signal distinguishing them from community nodes
FAIL if: Generic list with no partner/community distinction
```

**Test 2: Splat-to-Mesh Conversion**
```
Ask: "I have a gaussian splat, how do I get a mesh?"

PASS if: Specific conversion path with node names (marching cubes, Trellis2)
         and pitfall warnings (resolution, UVs)
FAIL if: Generic "you can convert it" without actionable guidance
```

**Test 3: Camera/Viewport Tool Discovery**
```
Ask: "How do I control camera angle for ControlNet?"

PASS if: Names specific viewport tools (Action Director, VNCCS)
FAIL if: Only mentions generic ControlNet workflow
```

**Test 4: 3D Workflow Parsing**
```
If a 3D workflow JSON is available:
  agent parse <workflow.json>

PASS if: "Node X generates a 3D mesh" (human-readable type names)
FAIL if: "Node X outputs unknown type"

If no 3D workflow available: note this and skip.
```

## Gate: Phase 1

- [ ] All Q1–Q5 answered with file paths and evidence
- [ ] Live tests run (or documented why they couldn't)
- [ ] Clear picture: what works vs what's stored-but-not-surfaced

## Report Template

```
═══════════════════════════════════════════════
PHASE 1 REPORT: GRAFT VERIFICATION
═══════════════════════════════════════════════

UI Phases Status:
  Phase 0 (Recon):    [EXISTS/NOT FOUND]
  Phase 1 (Sidebar):  [EXISTS/NOT FOUND]
  Phase 2 (Bridge):   [EXISTS/NOT FOUND]
  Phase 3 (DISCOVER): [EXISTS/NOT FOUND]

Knowledge Routing:
  3D knowledge files:          [list]
  Trigger keywords:            [list per file]
  Comparative knowledge path:  [traced / gap]
  source_tier in schema:       [YES/NO]
  source_tier in responses:    [YES/NO]

3D-Related Tests:
  Count: [N]
  Names: [list]

Live Test Results:
  Test 1 (mesh gen comparison):    [PASS/FAIL] — {what happened}
  Test 2 (splat-to-mesh):          [PASS/FAIL] — {what happened}
  Test 3 (camera/viewport tools):  [PASS/FAIL] — {what happened}
  Test 4 (3D workflow parse):      [PASS/FAIL/SKIPPED] — {what happened}

LOAD3D_CAMERA Schema:
  {schema or "could not determine"}

Gaps Found:
  {list any knowledge that's stored but not reachable}
```

---

# PHASE 2: MAKE GRAFT KNOWLEDGE CONVERSATIONALLY ACTIVE
**Type:** [AGENT-GRIND] — Conditional on Phase 1 findings
**Duration:** ~30 minutes
**Working directory:** `C:\Users\User\comfyui-agent\`
**Skip condition:** If all 4 live tests PASS in Phase 1, skip to Phase 3.

## Objective
Fix gaps found in Phase 1 so Grafts A–D knowledge is reachable through conversation.

## Conditional Tasks

### IF comparative knowledge is NOT reachable

The agent has knowledge in files but doesn't inject it when relevant.

**Check trigger keywords in `_KNOWLEDGE_TRIGGERS`:**
```python
# These phrasings should load 3D comparison knowledge:
"3d generator"      → relevant knowledge file
"mesh generation"   → relevant knowledge file
"best 3d"           → relevant knowledge file
"compare.*3d"       → relevant knowledge file
"partner node"      → relevant knowledge file
"hunyuan"           → relevant knowledge file
"meshy"             → relevant knowledge file
"tripo"             → relevant knowledge file
"rodin"             → relevant knowledge file
```

If the comparative knowledge (from Graft B) lives in a separate file from `3d_workflows.md`, add triggers for that file. If it's inside `3d_workflows.md` but triggers don't match common phrasings, add more trigger patterns.

### IF source_tier is in schema but NOT surfaced

Option A (minimal — ship first): Add to the agent's system prompt instructions:
```
When presenting DISCOVER results that include 3D nodes, always mention
the source_tier: "Partner Node (officially supported)" vs "Community node".
Recommend Partner Nodes first for production use cases.
```

Option B (proper — ship second): Modify discover result formatting to include tier badges in output.

### IF 3D types show "unknown" in workflow parsing

Graft A may not have fully wired type descriptions. Check `workflow_parse.py` — find where type descriptions are generated. Verify MESH, VOXEL, POINT_CLOUD, CAMERA, POSE all have entries.

### IF viewport tools aren't surfaced

Graft D knowledge may not trigger on relevant phrases. Check: does asking about "camera control" or "pose character" load the knowledge file containing VNCCS and Action Director info? If not, add trigger keywords.

## Gate: Phase 2

- [ ] All 4 live tests from Phase 1 now PASS
- [ ] Changes limited to knowledge routing / trigger keywords / system prompt
- [ ] Existing 665 tests still green

## Report Template

```
═══════════════════════════════════════════════
PHASE 2 REPORT: KNOWLEDGE ACTIVATION
═══════════════════════════════════════════════

Changes made:
  [file] — [what changed]
  [file] — [what changed]

Live Test Results (re-run):
  Test 1: [PASS]
  Test 2: [PASS]
  Test 3: [PASS]
  Test 4: [PASS]

Test suite: [665 tests still passing / N failures]
```

---

# PHASE 3: GRAFT E TEST PREPARATION
**Type:** [AGENT-GRIND] — Test infrastructure, no owner decisions
**Duration:** ~45 minutes
**Working directory:** `C:\Users\User\comfyui-agent\`

## Objective
Create mock 3D workflow JSONs and test fixtures for Graft E's three demo scenarios. Graft E ships with Phase 5 (Multi-Pipe), but test infrastructure can be built now.

## 3.1 Mock Workflow: Splat-to-Mesh Export

Minimal workflow JSON representing:
```
[LoadSplat] → [MarchingCubes] → [MeshCleanup] → [ExportGLB]
```
Connection types: POINT_CLOUD → MESH → MESH → FILE

Save as: `tests/fixtures/workflow_splat_to_mesh.json`

**Structure:** Follow the ComfyUI workflow JSON format already used in existing test fixtures. Check `tests/fixtures/` for examples of the expected schema.

## 3.2 Mock Workflow: ControlNet 3D Setup

Minimal workflow JSON representing:
```
[VNCCSPose] → [ActionDirectorRender] → [ControlNetApply] → [KSampler]
```
Connection types: POSE → IMAGE (depth/normal) → CONDITIONING → LATENT

Save as: `tests/fixtures/workflow_controlnet_3d.json`

## 3.3 Mock Workflow: Partner Node Comparison

Minimal workflow JSON showing parallel paths from same input:
```
[TextPrompt] → [Hunyuan3D]  → [ExportGLB]
             → [Meshy6]     → [ExportGLB]
             → [TripoV3]    → [ExportGLB]
```

Save as: `tests/fixtures/workflow_partner_comparison.json`

## 3.4 Unit Tests for Demo Scenarios

```python
# tests/test_3d_demos.py
# Follow existing test patterns — all mocked, no live ComfyUI

def test_splat_to_mesh_discovery():
    """When user asks about splat-to-mesh, agent surfaces conversion path."""
    # Load mock workflow
    # Trigger DISCOVER with "convert gaussian splat to mesh"
    # Assert: response contains marching cubes OR Trellis2
    # Assert: response mentions resolution pitfall

def test_controlnet_3d_tool_discovery():
    """When user asks about 3D posing for ControlNet, agent surfaces tools."""
    # Trigger DISCOVER with "pose character for ControlNet"
    # Assert: response mentions VNCCS or Action Director
    # Assert: distinguishes posing vs rendering tools

def test_partner_node_comparison():
    """When user asks for 3D generation, agent compares Partner Nodes."""
    # Trigger DISCOVER with "best 3D mesh generator"
    # Assert: mentions multiple Partner Nodes
    # Assert: Partner Nodes listed before community alternatives
    # Assert: comparative recommendation based on use case

def test_splat_to_mesh_workflow_parse():
    """UNDERSTAND correctly parses splat-to-mesh workflow."""
    # Load workflow_splat_to_mesh.json
    # Parse with workflow_parse
    # Assert: POINT_CLOUD and MESH types described correctly
    # Assert: no "unknown type" in output
```

## 3.5 Update SUPERDUPER_3D_GRAFT.md

Replace the stub content in the comfyui-3D-viewport repo's `SUPERDUPER_3D_GRAFT.md` with the full graft plan (the complete document provided to the agent). This makes the repo self-documenting.

**Note:** The full content is in the uploaded `SUPERDUPER_3D_GRAFT.md` document. Copy it verbatim.

## Gate: Phase 3

- [ ] 3 mock workflow JSONs created in `tests/fixtures/`
- [ ] `tests/test_3d_demos.py` created with 4+ test functions
- [ ] All new tests pass (with mocks)
- [ ] `SUPERDUPER_3D_GRAFT.md` in the viewport repo populated with full content
- [ ] 665+ tests still green

## Report Template

```
═══════════════════════════════════════════════
PHASE 3 REPORT: GRAFT E PREPARATION
═══════════════════════════════════════════════

Mock workflows created:
  workflow_splat_to_mesh.json:     [N] nodes, [connection types]
  workflow_controlnet_3d.json:     [N] nodes, [connection types]
  workflow_partner_comparison.json: [N] nodes, [connection types]

Tests created: tests/test_3d_demos.py
  test_splat_to_mesh_discovery:      [PASS/FAIL]
  test_controlnet_3d_tool_discovery: [PASS/FAIL]
  test_partner_node_comparison:      [PASS/FAIL]
  test_splat_to_mesh_workflow_parse: [PASS/FAIL]

Total test count: [N] (was 665)
SUPERDUPER_3D_GRAFT.md: [UPDATED / NOT UPDATED]
```

---

# PHASE 4: LOAD3D_CAMERA COMPATIBILITY LAYER
**Type:** [DIRECTOR-LED] — Camera knowledge requires domain expertise validation
**Duration:** ~45 minutes
**Working directory:** `C:\Users\User\comfyui-agent\`

## Objective
Teach the agent about ComfyUI's existing 3D camera data pipeline (Load3D → LOAD3D_CAMERA → AdvancedCameraControlNode), preparing for convergence with the future standalone viewport.

## Context

ComfyUI's built-in `Load3D` node outputs a `LOAD3D_CAMERA` type containing camera position, orientation, FOV, and focal length. The community `AdvancedCameraControlNode` reads this and generates shot-type prompts ("medium close-up", "high angle", etc.) for conditioning.

This is the data contract the future standalone viewport (CarWash-2 concept) will also speak. Teaching the agent NOW means the viewport becomes a drop-in replacement for Load3D's camera — but with real ARRI/RED sensors and Cooke/Atlas lenses.

## 4.1 Document the Camera Pipeline

Using the LOAD3D_CAMERA schema discovered in Phase 1 Q5, create:

```markdown
# agent/knowledge/3d_camera_pipeline.md

## ComfyUI 3D Camera Pipeline

### Load3D → Camera Data
The built-in Load3D node outputs LOAD3D_CAMERA data containing:
- Camera position (x, y, z)
- Camera target/orientation
- Field of view
- Focal length (mm)

### AdvancedCameraControlNode
Reads LOAD3D_CAMERA and generates:
- Shot type classification (close-up, medium, wide, etc.)
- Camera angle description (high angle, low angle, bird's eye)
- Structured JSON metadata
- Prompt strings for MultiAngle LoRA / conditioning

### Workflow Pattern: Camera-Controlled Generation
Load3D → AdvancedCameraControlNode → prompt enrichment → ControlNet/LoRA → generation

### When to Recommend
- User wants specific camera angle: "shoot from low angle"
- User wants cinematic framing: "medium close-up of character"
- User has 3D scene and wants composition control
- User asks about camera control for AI generation

### Future: Cinematographic Camera Upgrade
The standard Load3D camera uses a simple pinhole model. A future upgrade
will support real camera bodies (ARRI Alexa 35, RED V-RAPTOR, Sony VENICE 2)
and real lenses (Cooke Anamorphic, Atlas Orion) with proper sensor projection.
This will output LOAD3D_CAMERA-compatible data with extended metadata.
```

**CHECKPOINT:** Owner validates camera pipeline description accuracy before proceeding. This is domain knowledge — 16 years of cinematography expertise applies here.

## 4.2 Add Trigger Keywords

In `system_prompt.py`, add triggers for the new knowledge file:

```python
"camera control":    "3d_camera_pipeline.md",
"camera angle":      "3d_camera_pipeline.md",
"shot type":         "3d_camera_pipeline.md",
"framing":           "3d_camera_pipeline.md",
"load3d camera":     "3d_camera_pipeline.md",
"LOAD3D_CAMERA":     "3d_camera_pipeline.md",
"cinematic":         "3d_camera_pipeline.md",
"focal length":      "3d_camera_pipeline.md",
```

## 4.3 Add LOAD3D_CAMERA to Type Registry

If not already present from Graft A, ensure the workflow parser recognizes `LOAD3D_CAMERA`:

```
LOAD3D_CAMERA: "3D camera position and settings (from Load3D node)"
```

## 4.4 Test

```
Test 1: "How do I control the camera angle for my 3D generation?"
  Expected: Explains Load3D → AdvancedCameraControlNode pipeline with node names.

Test 2: "I want a low angle shot of my 3D model"
  Expected: Recommends Load3D camera manipulation, then AdvancedCameraControlNode.

Test 3: Parse a workflow with LOAD3D_CAMERA connections
  Expected: "Camera settings from Load3D feed into prompt generation"
```

## Gate: Phase 4

- [ ] `3d_camera_pipeline.md` created with accurate schema (owner-validated)
- [ ] Trigger keywords added
- [ ] LOAD3D_CAMERA type recognized in parser
- [ ] Live tests pass
- [ ] All existing tests still green

---

# PHASE 5: VIEWPORT ARCHITECTURE PLANNING
**Type:** [DIRECTOR-LED] — Architecture decisions require owner
**Duration:** ~2 hours (planning session, not implementation)
**Output:** Design document, NOT code

## Objective
Write an architecture document for the future standalone 3D viewport that converges with the agent. This is the CarWash-2 concept — real cameras, Hydra rendering, ComfyUI bridge.

**This phase produces a DESIGN DOCUMENT. No implementation.**

## 5.1 Architecture Decision Record

The owner decides. The agent documents.

```
DECISION: ComfyUI custom node vs standalone app vs hybrid?

┌─────────────────────────────────────────────────────────────┐
│ CUSTOM NODE PATH (Python + Three.js)                        │
│                                                             │
│  + Installs via ComfyUI-Manager (huge distribution)         │
│  + Three.js viewport in browser (like Load3D)               │
│  + Python backend, JS frontend                              │
│  + Familiar ecosystem for ComfyUI users                     │
│  - Limited to browser rendering                             │
│  - Can't use Hydra/Storm for accurate AOVs                  │
│  - Three.js pinhole camera, not real sensor projection      │
│  - No true anamorphic squeeze in projection math            │
├─────────────────────────────────────────────────────────────┤
│ STANDALONE APP PATH (C++ / Qt / Hydra)                      │
│                                                             │
│  + Real Hydra/Storm rendering with accurate AOVs            │
│  + True ARRI/RED sensor projection matrices                 │
│  + Cooke anamorphic squeeze in actual projection math       │
│  + Professional cinematography tool                         │
│  + CarWash codebase as proven reference                     │
│  - C++ build system (USD dependency management)             │
│  - Separate install from ComfyUI                            │
│  - Smaller distribution reach                               │
│  - Higher development effort                                │
├─────────────────────────────────────────────────────────────┤
│ HYBRID PATH                                                 │
│                                                             │
│  + Custom node for distribution (Three.js viewport)         │
│  + Standalone app for production (Hydra viewport)           │
│  + Both output LOAD3D_CAMERA-compatible data                │
│  + Agent works with either seamlessly                       │
│  - Two codebases to maintain                                │
│  - Feature parity pressure                                  │
└─────────────────────────────────────────────────────────────┘
```

## 5.2 LOAD3D_CAMERA Extension Schema

Define what the viewport adds beyond standard Load3D:

```json
{
  "_comment": "Standard LOAD3D_CAMERA fields (compatibility layer)",
  "position": [0, 0, 5],
  "target": [0, 0, 0],
  "up": [0, 1, 0],
  "fov": 73.2,
  "focal_length": 40,

  "_comment": "CarWash-2 cinematographic extensions",
  "carwash_version": "2.0",
  "carwash_sensor_body": "ARRI Alexa 35",
  "carwash_sensor_gate": "Open Gate 4.6K",
  "carwash_sensor_width_mm": 27.99,
  "carwash_sensor_height_mm": 19.22,
  "carwash_lens_model": "Cooke Anamorphic/i 40mm",
  "carwash_focal_mm": 40,
  "carwash_squeeze_ratio": 2.0,
  "carwash_t_stop": 2.3,
  "carwash_lens_metadata_protocol": "Cooke /i Technology",
  "carwash_aov_passes": ["depth", "normal", "beauty"]
}
```

Prefix all extended fields with `carwash_` to avoid namespace collisions with standard Load3D fields or future ComfyUI additions.

## 5.3 Integration Contract

```
Viewport → Agent:
  - LOAD3D_CAMERA JSON (standard + carwash_ extended fields)
  - Depth AOV as PNG
  - Normal AOV as PNG
  - Camera metadata string for prompt injection

Agent → Viewport (future — when agent gains tool API):
  - Camera preset commands ("set Alexa 35 + Cooke 40mm")
  - Asset placement commands ("load character at origin")
  - Render trigger ("render depth pass")

Protocol: TBD (WebSocket / JSON-RPC / ComfyUI API extension)
```

## 5.4 Camera + Lens Database

Compile reference data used by both viewport AND agent knowledge. This is the owner's domain — cinematography expertise drives selection.

Save as: `data/camera_lens_database.json` in this repo.

```json
{
  "cameras": [
    {
      "id": "alexa35_og",
      "name": "ARRI ALEXA 35",
      "manufacturer": "ARRI",
      "gate": "Open Gate 4.6K",
      "sensor_width_mm": 27.99,
      "sensor_height_mm": 19.22,
      "max_resolution": [4608, 3164],
      "native_iso": 800
    },
    {
      "id": "red_vraptor_ff",
      "name": "RED V-RAPTOR",
      "manufacturer": "RED",
      "gate": "Full Frame 8K",
      "sensor_width_mm": 40.96,
      "sensor_height_mm": 21.60,
      "max_resolution": [8192, 4320],
      "native_iso": 800
    },
    {
      "id": "venice2_ff",
      "name": "Sony VENICE 2",
      "manufacturer": "Sony",
      "gate": "Full Frame 8K",
      "sensor_width_mm": 36.20,
      "sensor_height_mm": 24.10,
      "max_resolution": [8640, 5760],
      "native_iso": 800
    }
  ],
  "lenses": [
    {
      "id": "cooke_ana_40",
      "name": "Cooke Anamorphic/i 40mm",
      "manufacturer": "Cooke",
      "focal_mm": 40,
      "squeeze_ratio": 2.0,
      "type": "anamorphic",
      "t_stop_range": [2.3, 22],
      "metadata_protocol": "Cooke /i Technology"
    },
    {
      "id": "cooke_s7i_25",
      "name": "Cooke S7/i 25mm",
      "manufacturer": "Cooke",
      "focal_mm": 25,
      "squeeze_ratio": 1.0,
      "type": "spherical",
      "t_stop_range": [2.0, 22],
      "metadata_protocol": "Cooke /i Technology"
    },
    {
      "id": "atlas_orion_40",
      "name": "Atlas Orion 40mm",
      "manufacturer": "Atlas",
      "focal_mm": 40,
      "squeeze_ratio": 2.0,
      "type": "anamorphic",
      "t_stop_range": [2.0, 16],
      "metadata_protocol": null
    }
  ]
}
```

## Gate: Phase 5

- [ ] Architecture decision documented (owner chose: custom node / standalone / hybrid)
- [ ] LOAD3D_CAMERA extension schema defined with `carwash_` prefix convention
- [ ] Integration contract between viewport and agent documented
- [ ] Camera/lens database compiled and owner-validated
- [ ] Owner reviews and approves architecture direction

**No code ships in this phase.**

---

# DELEGATION RULES

## What Agent Teams Do Autonomously [AGENT-GRIND]
- Codebase auditing (grep, find, trace code paths)
- Knowledge file creation and trigger keyword wiring
- Mock workflow JSON creation (following existing fixture patterns)
- Unit test writing (following existing test patterns with mocks)
- Documentation and report generation

## What Requires Owner Direction [DIRECTOR-LED]
- Architecture decisions (custom node vs standalone vs hybrid)
- Camera/lens specifications and projection math (domain expertise)
- Comparative knowledge accuracy (which 3D tool is best for what)
- Graft E demo scenario sign-off
- Convergence timing (when to start viewport implementation)
- LOAD3D_CAMERA schema validation (Phase 4 checkpoint)

## Escalation Protocol
If blocked for >15 minutes:
1. Document what was tried
2. Report with numbered options
3. Do NOT spend time guessing at architecture or domain decisions
4. Do NOT modify files outside `C:\Users\User\comfyui-agent\` without explicit instruction

---

# REFERENCE: GRAFT DISPATCH PROMPTS

Grafts A–E dispatch prompts live in `SUPERDUPER_3D_GRAFT.md` in this repo.
Graft F dispatch lives in `DISPATCH_PROFILERX.md` (separate file, both repos).

If any graft needs re-execution or modification:

```
# Grafts A-E:
Read SUPERDUPER_3D_GRAFT.md. Re-run Graft [A/B/C/D/E] dispatch prompt.

# Graft F:
Read DISPATCH_PROFILERX.md. Execute the dispatch prompt.
```

---

# WHAT TO TELL CLAUDE CODE RIGHT NOW

```
# Main execution track:
Read EXECUTION_SPEC.md in the comfyui-3D-viewport repo.
Read CLAUDE.md for parent project context.
Read SUPERDUPER_3D_GRAFT.md for graft plan reference.

Execute Phase 1: Verify Graft Knowledge.
Working directory: C:\Users\User\comfyui-agent\

Answer all Q1-Q5 by inspecting the codebase.
Run live tests if possible.
Report at the gate using the report template.
Do not proceed to Phase 2 without reporting.
```

```
# Parallel dispatch (can run on a separate agent):
Read DISPATCH_PROFILERX.md. Execute the Graft F dispatch prompt.
Working directories:
  Code changes: C:\Users\User\comfyui-agent\
  Graft tracking: comfyui-3D-viewport repo
Report at step 9.
```

## FILE DISTRIBUTION

Drop all files into both repos:

```
comfyui-agent/
  EXECUTION_SPEC.md
  DISPATCH_PROFILERX.md

comfyui-3D-viewport/
  CLAUDE.md                  (already exists)
  SUPERDUPER_3D_GRAFT.md     (already exists — Graft F dispatch updates it)
  EXECUTION_SPEC.md
  DISPATCH_PROFILERX.md
```
