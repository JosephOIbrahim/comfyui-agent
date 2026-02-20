# ComfyUI 3D Viewport — Phased Strategy
## Project: CarWash-2 | Status: ACTIVE (parallel track)
## Updated: 2026-02-19

---

## Questions for Claude Code (Agent Track)

Before building further on the viewport, we need ground truth on the agent's UI plan state. Paste these to Claude Code:

```
1. Do Phases 0-3 of the SUPERDUPER_UI_PLAN exist as shipped code?
   - Phase 0 (Recon) — any audit output?
   - Phase 1 (Sidebar Skeleton) — does a sidebar UI component exist?
   - Phase 2 (Backend Bridge) — does a frontend↔backend bridge exist beyond Graft A?
   - Phase 3 (DISCOVER Panel) — does any frontend panel render partner/community/core badges?

2. Grafts A-D shipped backend knowledge. What is the FIRST UI surface that can display it?
   Is it the agent's conversational responses? Or does it need a panel?

3. Is the agent currently capable of answering "what's the best 3D generator?"
   with the comparative knowledge from Graft C/D? Or is that knowledge
   injected but not yet routed to response generation?

4. 665 tests green — are there any test categories marked skip/pending
   that relate to UI rendering or frontend integration?
```

---

## Two Parallel Tracks — How They Converge

```
AGENT TRACK                          VIEWPORT TRACK
(ComfyUI-Agent / Super Duper)        (CarWash-2 / 3D Viewport)
─────────────────────────────        ──────────────────────────
Knowledge layer (Grafts A-D) ✅       Architecture blueprint
68 tools                              Hydra + Qt + Storm
DISCOVER/PILOT/BRAIN layers           Real camera system (ARRI/Cooke)
Conversational UI                     Depth/Normal → ComfyUI pipeline
                    \                /
                     \              /
              EXISTING COMFYUI ECOSYSTEM
              ┌─────────────────────────┐
              │ Load3D → LOAD3D_CAMERA  │
              │ AdvancedCameraControl   │
              │ ControlNet pipeline     │
              │ MultiAngle LoRAs        │
              └────────────┬────────────┘
                           │
                      CONVERGENCE
                     ┌────────────┐
                     │ CarWash-2  │
                     │ as custom  │
                     │ node with  │
                     │ real cams  │
                     │ + agent    │
                     │ tool API   │
                     └────────────┘
```

**Key insight:** ComfyUI already has Load3D → AdvancedCameraControlNode → prompt generation. CarWash-2 doesn't replace this — it plugs in as a premium upstream that adds real cinematographic accuracy. The existing `LOAD3D_CAMERA` data type is the integration contract.

**But that's Phase 6+ territory.** Each track ships independently first.

---

## Viewport Track — Phased Strategy

### Phase 0: Foundation (Sprint 1-2)
**Goal:** Qt window with Storm rendering a cube. Build system working.
**Energy:** Agent-team grindable. You validate it compiles + renders.

**AND-node subtasks (all required):**
- [ ] Fork CarWash repo, strip Houdini HDK dependency
- [ ] CMake links against standalone OpenUSD/Hydra (not Houdini's bundled USD)
- [ ] Qt window hosts OpenGL context
- [ ] Storm render delegate renders a UsdGeomCube
- [ ] Depth AOV renders to buffer (verify pixel values)

**Hardest branch:** USD/Hydra build system on Windows with Qt. This is the 60% calendar-time problem. Gaffer's CMake patterns (BSD licensed) are the cheat code. CarWash's existing build infrastructure transfers partially — strip Houdini, add standalone USD.

**Existing assets that transfer:**
- CarWash's 2,360-line ComfyUI client (WebSocket, HTTP, PNG encoding, base64)
- CarWash's CMake structure (adapt, don't rewrite)
- CarWash's framebuffer + FNV-1a determinism framework

**Claude Code task:**
```
Get OpenUSD 24.11 compiling on Windows with Qt6 and Storm.
Reference Gaffer's CMake patterns (BSD, github.com/GafferHQ/gaffer).
Port CarWash's CMakeLists.txt — strip find_package(Houdini),
add find_package(pxr). Target: render a UsdGeomCube in a Qt window
via Storm. No camera work yet — just prove the build pipeline works.
```

**Ship criterion:** Screenshot of a grey cube in a Qt window. Verifiable.

---

### Phase 1: Camera System (Sprint 3-4)
**Goal:** Real cinematographic camera with ARRI sensor + Cooke lens projection.
**Energy:** YOUR work. This is the unfair advantage. Agent teams implement, you direct.

**AND-node subtasks:**
- [ ] Camera prim with real sensor dimensions (Alexa 35: 27.99 × 19.22mm Open Gate)
- [ ] Projection matrix from real lens specs (Cooke anamorphic 40mm, 2x squeeze)
- [ ] Mouse orbit/pan/zoom interaction
- [ ] Camera metadata embedded in USD (sensor, lens, T-stop, focal length)
- [ ] Viewport matches what a DP sees through the eyepiece

**Camera database (you already know most of this):**

| Camera | Sensor W (mm) | Sensor H (mm) | Gate |
|--------|---------------|---------------|------|
| ALEXA 35 | 27.99 | 19.22 | Open Gate 4.6K |
| ALEXA 35 | 27.99 | 15.74 | 4:3 |
| ALEXA 35 | 27.99 | 11.81 | 2.39:1 |
| RED V-RAPTOR | 40.96 | 21.60 | Full Frame 8K |
| Sony VENICE 2 | 36.20 | 24.10 | Full Frame |

| Lens | Focal (mm) | Squeeze | Bokeh | Notes |
|------|-----------|---------|-------|-------|
| Cooke Anamorphic/i 40mm | 40 | 2x | Oval | /i metadata protocol |
| Cooke Anamorphic/i 50mm | 50 | 2x | Oval | |
| Cooke S7/i 25mm | 25 | 1x (spherical) | Round | |
| Atlas Orion 40mm | 40 | 2x | Oval | Budget anamorphic |

**Ship criterion:** Same scene rendered with Alexa 35 + Cooke 40mm anamorphic vs RED + spherical 50mm looks *cinematographically different*. The depth maps encode different projection geometry. Verifiable by visual inspection.

**Claude Code task:**
```
Build camera prim system on top of Phase 0's Hydra viewport.
UsdGeomCamera with custom projection matrix built from:
  - Sensor dimensions (width_mm, height_mm)
  - Lens focal length, squeeze ratio
  - Real T-stop for DOF (future)
Mouse orbit/pan/zoom. Camera dropdown: ARRI Alexa 35, RED V-RAPTOR.
Lens dropdown: Cooke Anamorphic 40mm, Cooke S7/i 25mm.
Switching camera+lens changes the projection matrix and viewport.
Store sensor/lens data as USD custom attributes on the camera prim.
```

---

### Phase 2: Render Passes + ComfyUI Bridge (Sprint 5-6)
**Goal:** Depth + Normal AOVs → WebSocket → ComfyUI → generated image displayed.
**Energy:** Mostly porting. CarWash's ComfyUI client code transfers directly.

**AND-node subtasks:**
- [ ] Depth AOV renders through the real camera projection
- [ ] Normal AOV renders through the real camera projection
- [ ] PNG encoding of AOV buffers (CarWash code transfers)
- [ ] WebSocket client sends buffers to ComfyUI API (CarWash code transfers)
- [ ] ControlNet workflow builder (CarWash code transfers — LTX-2 + ControlNet)
- [ ] Receive generated image, display in companion panel
- [ ] Camera metadata sent as prompt context ("Alexa 35, Cooke 40mm anamorphic, T2.8")
- [ ] **Output LOAD3D_CAMERA-compatible data** (see integration section below)

**Hardest branch:** Making sure the depth/normal passes encode the REAL projection, not a default pinhole. If this is wrong, the whole product thesis collapses — the AI gets a generic depth map instead of a cinematographic one.

**Ship criterion:** Send depth+normal from Hydra viewport to ComfyUI. Receive a generated image that respects the camera's perspective geometry. Verifiable.

**Claude Code task:**
```
Port CarWash's ComfyClient (WebSocket, HTTP, PNG encoding, base64,
workflow builder) into the Hydra viewport app. Wire Hydra's AOV system
to render depth + normal passes through the camera from Phase 1.
Send both passes to ComfyUI via the existing API. Build a ControlNet
workflow that conditions on depth. Display the returned image in a
Qt panel next to the viewport. Include camera metadata in the prompt.

IMPORTANT: Also output camera state as LOAD3D_CAMERA-compatible JSON
so it can feed into ComfyUI's existing AdvancedCameraControlNode.
See "Integration with Existing 3D Camera Nodes" section.
```

---

### Integration with Existing 3D Camera Control Nodes

ComfyUI already has a camera data pipeline. CarWash-2 should plug into it, not replace it.

**What exists today:**
```
Load3D node (built-in core)
  → Loads .gltf/.glb/.obj/.fbx/.stl
  → Three.js viewport in the node UI
  → Outputs: IMAGE, MASK, NORMAL, DEPTH
  → Outputs: LOAD3D_CAMERA (position, orientation, focal_length, FOV)
        │
        ▼
AdvancedCameraControlNode (ComfyUI-AdvancedCameraPrompts)
  → Reads LOAD3D_CAMERA data
  → Inputs: focal_length_mm (1-1000), object_scale_meters
  → Outputs: shot type classification ("medium close-up", "wide shot")
  → Outputs: camera angle description ("low angle", "bird's eye")
  → Outputs: structured JSON camera metadata
  → Optimized for dx8152's MultiAngle LoRA
        │
        ▼
Prompt → ControlNet / LoRA conditioning
```

**What CarWash-2 adds on top:**
```
CarWash-2 Hydra Viewport (REAL cameras)
  → ARRI/RED/Sony sensor dimensions (not game camera)
  → Cooke/Atlas/Zeiss lens projection (not pinhole)
  → Anamorphic squeeze, real T-stops, breathing
  → Depth/Normal AOVs through correct projection
  → Outputs everything Load3D outputs PLUS:
     - sensor_body: "ARRI Alexa 35"
     - sensor_gate: "Open Gate 4.6K"
     - sensor_dimensions_mm: [27.99, 19.22]
     - lens_model: "Cooke Anamorphic/i 40mm"
     - squeeze_ratio: 2.0
     - t_stop: 2.8
     - lens_metadata_protocol: "Cooke /i Technology"
```

**Integration strategy (two paths, both valid):**

**Path A — CarWash-2 as upstream of existing nodes:**
CarWash-2 exports LOAD3D_CAMERA-compatible data. The existing
AdvancedCameraControlNode can read it and generate shot descriptions.
CarWash-2 *enriches* the camera data with real specs that the existing
node doesn't know about, but the basic position/orientation/focal_length
contract is preserved. Zero changes to existing nodes.

**Path B — CarWash-2 as custom ComfyUI node (future):**
Wrap the viewport as a ComfyUI custom node (like Load3D but with
real cameras). Output LOAD3D_CAMERA + extended cinematography metadata.
Build a CinematographicCameraControlNode that understands sensor+lens
combinations and generates richer prompts than the generic
AdvancedCameraControlNode. This is the premium path — but it needs
the standalone viewport working first.

**Recommendation:** Build Path A first (standalone app outputting
LOAD3D_CAMERA-compatible data). Prove the pipeline. Then Path B
(custom node integration) becomes Phase 5 convergence work.

**Questions for Claude Code:**
```
1. What is the exact schema of LOAD3D_CAMERA? We need to output
   compatible data. Check ComfyUI source: comfy/nodes/load_3d.py
   or equivalent.

2. Does AdvancedCameraControlNode accept camera data via API
   (not just from Load3D widget)? If yes, CarWash-2 can inject
   camera state directly into a workflow via the ComfyUI API.

3. Can Load3D_Adv accept external image inputs for its passes?
   If yes, CarWash-2 could send its Hydra-rendered depth/normal
   as if they came from Load3D, plugging into all downstream nodes.
```

---

### Phase 3: Scene Composition (Sprint 7-8)
**Goal:** Drag-drop assets, basic transforms, scene save/load.
**Energy:** Medium. Gaffer's scene graph patterns (BSD) are the reference.

**AND-node subtasks:**
- [ ] Load USD/glTF assets via drag-drop
- [ ] Basic transform gizmos (translate, rotate, scale)
- [ ] Scene hierarchy panel (tree view of USD stage)
- [ ] Save/load scene as USD file
- [ ] Multiple assets in scene with independent transforms

**OR-node approaches (any works):**
- Port Gaffer's SceneView patterns directly (fastest)
- Build minimal scene graph from scratch (more control, more work)
- Use USD's built-in stage composition (reference/payload arcs)

**Ship criterion:** Place 3 assets in a scene, frame with Alexa 35 + Cooke 50mm, render to ComfyUI, get a coherent generated image. Save scene. Reload. Same result.

---

### Phase 4: Polish + PoC Demo (Sprint 9-10)
**Goal:** The demo reel moment.
**Energy:** High dopamine. This is the payoff.

**The demo:** "I'm looking through a Cooke 50mm anamorphic on an Alexa 35. I place a character in the scene. I hit render. ComfyUI generates what this shot looks like."

**AND-node subtasks:**
- [ ] UI polish (camera/lens dropdowns, render button, progress indicator)
- [ ] Anamorphic visual characteristics as post-process (oval bokeh, horizontal flare)
- [ ] Side-by-side: viewport wireframe | generated image
- [ ] Performance: interactive camera manipulation at 30fps+
- [ ] README + build instructions for collaborators

**Ship criterion:** Screen recording that makes people go "holy shit."

---

### Phase 5: Convergence with Agent Track + Custom Node (Future)
**Goal:** Agent can drive the viewport AND viewport ships as a ComfyUI custom node.
**Energy:** Requires both tracks to be independently functional.

**Subtasks — Node Integration (Path B from above):**
- [ ] Wrap Hydra viewport as a ComfyUI custom node (like Load3D but real cameras)
- [ ] Output type: LOAD3D_CAMERA + CARWASH_CINEMA (extended metadata)
- [ ] Build CinematographicCameraControlNode (richer prompts than AdvancedCameraControlNode)
- [ ] v3 schema compliance (stateless class interface, async execution)
- [ ] Depth/Normal passes plug directly into existing ControlNet nodes

**Subtasks — Agent Convergence:**
- [ ] Viewport exposes a tool API (JSON-RPC or similar)
- [ ] Agent gets a `viewport_camera` tool and a `viewport_render` tool
- [ ] "Set up a medium close-up with anamorphic bokeh" → agent calls viewport
- [ ] Generated image feeds back into agent's visual reasoning

**This is the moat.** Nobody else has an AI agent that understands cinematography AND controls a real camera viewport AND conditions generative models on the result. This is where 16 years of Houdini + the agent work + the viewport converge.

---

## Sprint Cadence for Small Sprints

Each phase breaks into ~2 sprints. Each sprint is scoped for a few focused sessions:

| Sprint | Phase | Deliverable | Who Does the Work |
|--------|-------|-------------|-------------------|
| 1 | 0 | CMake + USD compiling on Windows | Claude Code (grind) |
| 2 | 0 | Qt + Storm rendering a cube | Claude Code + you validate |
| 3 | 1 | Camera prim + projection matrix | You direct, CC implements |
| 4 | 1 | Mouse interaction + camera dropdown | Claude Code |
| 5 | 2 | AOV rendering through real camera | Claude Code + you verify |
| 6 | 2 | ComfyUI bridge (port CarWash client) | Claude Code |
| 7 | 3 | Asset loading + transforms | Claude Code |
| 8 | 3 | Scene save/load + hierarchy | Claude Code |
| 9 | 4 | UI polish + anamorphic FX | You direct, CC implements |
| 10 | 4 | Demo recording + README | You |

**Marathon pacing:** One sprint per week at sustainable pace. 10 sprints ≈ 10 weeks. Some sprints compress (especially 5-6 since CarWash code transfers). Realistic ship: ~8 weeks to PoC demo.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| USD build system hell on Windows | HIGH | Blocks Phase 0 | Gaffer CMake patterns. Agent teams grind it. Budget 2 sprints. |
| Storm OpenGL context + Qt conflict | MEDIUM | Blocks Phase 0 | Gaffer solved this — study their Qt integration |
| Depth map doesn't encode real projection | LOW | Kills product thesis | Verify with known geometry: render a cube at known distance, check depth values match expected projection |
| ComfyUI API changes break client | LOW | Blocks Phase 2 | CarWash client is battle-tested. Pin ComfyUI version. |
| Anamorphic FX look fake | MEDIUM | Weakens Phase 4 | You'll know. This is your domain expertise. Iterate until it looks right. |
| Agent convergence harder than expected | MEDIUM | Delays Phase 5 | Phase 5 is future. Both tracks ship independently first. |

---

## What to Paste to Claude Code Right Now

Start with **Phase 0, Sprint 1** — the build system:

```
PROJECT: CarWash-2 (ComfyUI 3D Viewport)
REPO: https://github.com/JosephOIbrahim/CarWash

TASK: Phase 0, Sprint 1 — Get OpenUSD + Qt + Storm compiling on Windows.

CONTEXT:
- CarWash is an existing C++ project with CMake, WebSocket client, PNG encoding
- Strip the Houdini HDK dependency
- Link against standalone OpenUSD 24.11 (not Houdini-bundled USD)
- Add Qt6 for the window
- Add Hydra Storm render delegate
- Reference Gaffer's CMake patterns (BSD, github.com/GafferHQ/gaffer)
  for how they link USD + Qt

TARGET: A Qt window that creates a UsdStage with a UsdGeomCube
and renders it via Storm. No camera, no interaction — just prove
the build pipeline works.

QUESTIONS I NEED ANSWERED FIRST:
1. Do Phases 0-3 of SUPERDUPER_UI_PLAN exist as shipped code?
2. What's the current state of the agent's frontend?
3. Can the agent currently use Graft C/D knowledge in conversational responses?
4. Are there any pending/skip tests related to UI rendering?
```
