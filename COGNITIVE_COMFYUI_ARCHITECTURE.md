# Cognitive ComfyUI — USD-Native Agentic Generative Ecosystem
## The Nervous System is Agents. The Skeleton is USD. The Brain is Composition.

> **Status:** Architecture design. Pre-implementation.  
> **Version:** 2.0.0 — THINKING AGENTS  
> **Foundation:** Agent Team Blueprint (MoE coordination) + Cognitive Substrate patents (state composition) + Karpathy Loop (autoresearch ratchet)  
> **Core thesis:** This isn't a ComfyUI product. It's an ecosystem whose nervous system is agents and whose skeleton is USD — pushing agentic generative AI past what anyone is currently building. ComfyUI is the execution engine. USD is the universal interchange. Agents are the intelligence. Composition is the architecture. The ratchet runs overnight.

---

## The Triple Domain Transfer

Joe's cognitive substrate patents solve a universal problem: **multiple agents with competing opinions about shared state, composed into coherent output, with guaranteed recoverability.** This problem appears identically across three domains:

```
DOMAIN 1: VFX Production (USD/Pixar)
  Scene description. Assets, lights, materials, cameras.
  Multiple departments author opinions. LIVRPS resolves conflicts.
  Shot layers compose into final render.

DOMAIN 2: Cognitive Architecture (Substrate Patents)  
  State description. Memory, routing, injection, momentum.
  Multiple contexts author opinions. LIVRPS resolves conflicts.
  Session layers compose into current cognitive state.

DOMAIN 3: ComfyUI Agentic Workflows (This Document)
  Workflow description. Nodes, connections, parameters, models.
  Multiple agents author opinions. LIVRPS resolves conflicts.
  Agent layers compose into optimal generation pipeline.
```

**The transfer isn't analogical. It's architectural.** The same composition arcs, the same conflict resolution semantics, the same lossless guarantees, the same derivative tool patterns. The data changes. The machinery doesn't.

---

## Why This Changes Everything About ComfyUI

Current ComfyUI workflow building is **manual scene assembly** — like hand-placing every light, every material, every camera in a VFX shot without a pipeline. Users wire nodes, tweak parameters, run, check, rewire. It's a human doing the work of an entire VFX department.

With the cognitive substrate mapped onto ComfyUI, the paradigm shifts:

**Before (manual):** Human designs workflow → human tweaks parameters → human evaluates output → human modifies → repeat

**After (cognitive agentic):** Human describes intent → agents compose a workflow stage from learned components → agents execute and analyze → agents modify through composed deltas → convergence is automatic → human approves the result

The agents don't just "help build workflows." They maintain a **living, composed workflow stage** — a USD-like structure where every node is a prim, every parameter is an attribute, every agent modification is a sublayer, and the full composition history is always available.

---

## Architecture: The Cognitive Workflow Stage

### The Stage

```
cognitive_workflow_stage.usda (composed)
│
├── /workflows/                          # Workflow prims (one per workflow)
│   ├── /workflows/portrait_v3/          # A specific workflow
│   │   ├── /nodes/                      # Node prims
│   │   │   ├── KSampler_01             # Prim with typed attributes
│   │   │   │   ├── steps: int = 28      # (attribute)
│   │   │   │   ├── cfg: float = 7.0     # (attribute)
│   │   │   │   ├── sampler: token = "euler_a"
│   │   │   │   └── seed: int = 42
│   │   │   ├── CheckpointLoader_01
│   │   │   │   └── ckpt_name: token = "sdxl_base.safetensors"
│   │   │   └── CLIPTextEncode_pos_01
│   │   │       └── text: string = "photorealistic portrait..."
│   │   ├── /connections/                # Relationship prims (node graph edges)
│   │   │   └── rel KSampler_01.model -> CheckpointLoader_01.MODEL
│   │   └── /metadata/
│   │       ├── model_family: token = "sdxl"
│   │       ├── target_quality: string = "photorealistic"
│   │       └── created_by: token = "architect_agent"
│   │
│   └── /workflows/anime_landscape_v1/  # Another workflow
│
├── /recipes/                            # Learned parameter combinations
│   ├── /recipes/sdxl_portrait_sharp/
│   │   ├── steps: int = 28
│   │   ├── cfg: float = 7.0
│   │   ├── sampler: token = "euler_a"
│   │   ├── quality_score: float = 8.4
│   │   └── learned_from: rel -> /executions/run_047
│   │
│   └── /recipes/flux_anime_vivid/
│
├── /executions/                         # Execution history (AIMemoryChunk)
│   ├── /executions/run_047/
│   │   ├── memory:type = "episodic"
│   │   ├── memory:content = "{params_used}"
│   │   ├── memory:timestamp = "2026-03-25T..."
│   │   ├── memory:relevance_tags = ["portrait", "sdxl", "sharp"]
│   │   ├── memory:decay_weight = 0.95
│   │   ├── quality_score: float = 8.4
│   │   ├── diagnosis: string = "sharp details, natural skin"
│   │   └── output_path: asset = "workspace/output/run_047.png"
│   │
│   └── /executions/run_048/
│
├── /agents/                             # Agent state prims
│   ├── /agents/scout/
│   │   └── last_recon: string = "{recon_report_ref}"
│   ├── /agents/architect/
│   │   └── active_design: string = "{design_doc_ref}"
│   ├── /agents/forge/
│   │   └── pending_patches: int = 0
│   ├── /agents/crucible/
│   │   └── last_verification: token = "PASS"
│   └── /agents/vision/
│       └── calibration_baseline: float = 7.0
│
└── /models/                             # Model inventory (from Scout + Provisioner)
    ├── /models/checkpoints/
    │   ├── sdxl_base.safetensors        # MATERIALIZED (on disk, verified)
    │   │   ├── family: token = "sdxl"
    │   │   ├── size_gb: float = 6.7
    │   │   ├── status: token = "ready"
    │   │   ├── hash: string = "a1b2c3..."
    │   │   └── compatible_with: rel -> [/recipes/sdxl_*]
    │   ├── flux_dev.safetensors         # MATERIALIZED
    │   └── sdxl_lightning_4step         # UNMATERIALIZED PAYLOAD (known, not downloaded)
    │       ├── family: token = "sdxl"
    │       ├── size_gb: float = 4.1
    │       ├── status: token = "available"   # ← Provisioner can materialize this
    │       ├── source: string = "huggingface:ByteDance/SDXL-Lightning"
    │       └── compatible_with: rel -> [/recipes/sdxl_*]
    └── /models/loras/
```

This isn't a database schema wearing USD clothes. It IS a USD stage — with real composition semantics, real conflict resolution, and real recoverability.

### LIVRPS for Multi-Agent Workflow Composition

The killer feature. When multiple agents want to modify the same workflow parameter, LIVRPS provides deterministic, debuggable conflict resolution:

```
L (Local)     → Agent's immediate modification
               "Change seed to 42 right now"
               STRONGEST — this is the human or the active agent's
               direct instruction. Always wins.

I (Inherit)   → Agent role constraints  
               "Forge always preserves model paths"
               "Vision never modifies generation parameters"
               Role isolation as class inheritance. An agent INHERITS
               behavioral constraints from its role definition.

V (Variants)  → Creative profiles (THE INJECTION MAPPING)
               "Anime mode" / "Photorealistic mode" / "Abstract mode"
               Variant selection switches entire parameter sets
               without recomposing the stage. This IS injection
               applied to workflows — profile switching as variant
               selection.

R (References) → Quality benchmarks from memory
               "Best score from memory: cfg=7, steps=28"
               Referenced recipes bring in proven parameters.
               Weaker than direct agent instructions, stronger
               than defaults.

P (Payloads)  → Lazy-loaded workflow components AND models
               "ControlNet depth config" (loaded on demand)
               "SDXL Lightning checkpoint" (not yet downloaded)
               Heavy components stay unloaded until an agent
               actually needs them. Exactly like deferred asset
               loading in VFX. The PROVISIONER agent materializes
               payloads — resolving, downloading, verifying, and
               registering models and node packs on demand.
               This is the Ollama pattern: declare what you need,
               the system provisions it.

S (Specialize) → Base workflow templates
               "SDXL portrait baseline"
               The weakest opinion. Everything else overrides it.
               But it's always there as the foundation.
```

**Concrete example — conflict resolution:**

```
Agent         Parameter    Value    Arc       Winner?
─────────────────────────────────────────────────────
Specialize    cfg          8.0      S (6)     No
Recipe ref    cfg          7.0      R (4)     No
Variant       cfg          7.5      V (3)     No
Forge         cfg          6.5      L (1)     YES ← strongest
```

Forge's local opinion wins. But when Forge's session ends and its sublayer is removed, the stage falls back to the Recipe's cfg=7.0 (next strongest surviving opinion). The parameter value degrades gracefully instead of disappearing.

**This is how VFX production works.** Shot overrides beat department layers beat asset defaults. Remove the shot override, the department layer's value appears. Remove that, the asset default appears. Nobody loses data. Every opinion is preserved. The composition resolves deterministically.

---

## The Lossless Workflow Guarantee

Directly from the Lossless Signal Architecture patent. Same equation, new domain:

```
workflow_output = clean_workflow + alpha * delta

clean_workflow:  The original, unmodified workflow JSON (always preserved)
alpha:           Agent confidence / iteration weight [0.0, 1.0]
delta:           The agent's modification (zero for anchor parameters)
```

### Anchor Parameters (Structurally Immune)

Some workflow elements must NEVER be modified by autonomous agents. These are the anchors — outside the modification pathway entirely:

| Anchor Parameter | Why |
|---|---|
| Model file path | Wrong model = wrong output family. Human choice. |
| Output resolution | Affects VRAM, render time, composition requirements. |
| Node graph topology | Adding/removing nodes changes the pipeline structure. |
| Safety filters | Content safety is not an optimization parameter. |

**These aren't "protected." They're structurally unreachable.** The agent tools literally cannot modify anchor parameters. The gain function returns 1.0 (no modification) before the parameter is even evaluated. Same architecture as Constitutional/Safety/Consent/Knowledge anchors in the cognitive substrate.

### Agent Modifications Are Deltas

Every agent modification is stored as a delta — not as a replacement of the original workflow. This means:

```
Run 1:  clean + 0.8 * {cfg: -1.0, steps: +8}     → cfg=7.0, steps=28
Run 2:  clean + 0.9 * {cfg: -1.5, steps: +8}     → cfg=6.5, steps=28  
Run 3:  clean + 0.7 * {cfg: -1.0, steps: +12}    → cfg=7.0, steps=32

At any point: reconstruct_clean() returns the original workflow.
No agent modification is destructive. Every change is reversible.
The full modification history is the composition layer stack.
```

### Integrity Verification (Per-Execution)

```
clean_hash:     SHA256 of original workflow JSON (must be constant)
anchor_hash:    SHA256 of anchor parameters only (must be constant)
fidelity:       1.000000 (less = modification leaked into anchors)
delta_magnitude: Bounded deviation from clean (must be < threshold)
```

If fidelity drops below 1.0, an agent modified an anchor parameter. That's a constitutional violation. Hard stop.

---

## The USD-Native Engine (First Principles)

No SQLite. No database simulating USD semantics. The Cognitive Workflow Stage IS a `pxr.Usd.Stage` — real composition arcs, real LIVRPS, real `Sdf.Layer` operations. The `pxr` Python bindings (available through Houdini's Python) give us a C++ composition engine with a Python API.

### Why Native USD Is Faster AND Simpler

**The wrong assumption:** "USD is slow, use SQLite for speed, migrate later."

**First principles:** USD gets slow loading 50M geometry prims in a VFX shot. Our stages have ~200 prims — workflows, recipes, execution history, model inventory. That's nothing. The in-memory operations are C++ behind Python bindings. An attribute read is a dictionary lookup in C++. Nanoseconds of real work, microseconds of Python overhead.

**Building a composition engine in SQLite** means: sublayer ordering as table joins, LIVRPS as query priority weights, relationships as foreign keys, variant sets as conditional row selection. That's hundreds of lines of code reimplementing what `pxr.Usd.Stage` already does. Then when we want real USD output, we need a SECOND translation layer.

**Going native means:**
- Composition engine = `stage.GetRootLayer().subLayerPaths.append(delta_layer)` — one line
- LIVRPS = built in to the `pxr` library
- Variant selection = `prim.GetVariantSet("profile").SetVariantSelection("explore")` — one line
- Conflict resolution = free (it's what USD does)
- USD output = `stage.Save()` — the data is already in the right format

### The Stage API

```python
from pxr import Usd, UsdGeom, Sdf, Vt

class CognitiveWorkflowStage:
    """
    The entire 'database layer' — it's just USD.
    All agent operations happen on an in-memory composed stage.
    Periodic flush writes .usdc (binary crate) for persistence.
    """
    
    def __init__(self, root_path="workspace/stage/root.usdc"):
        if os.path.exists(root_path):
            self.stage = Usd.Stage.Open(root_path)       # Milliseconds
        else:
            self.stage = Usd.Stage.CreateNew(root_path)
            self._bootstrap_hierarchy()
    
    def read(self, prim_path, attr_name):
        """Microsecond read — C++ lookup behind Python binding."""
        prim = self.stage.GetPrimAtPath(prim_path)
        return prim.GetAttribute(attr_name).Get()
    
    def write(self, prim_path, attr_name, value):
        """Microsecond write to session sublayer (strongest local opinion)."""
        prim = self.stage.GetPrimAtPath(prim_path)
        prim.GetAttribute(attr_name).Set(value)
    
    def add_agent_delta(self, agent_name, delta_dict):
        """
        Agent modification = new sublayer. Millisecond operation.
        The delta is a separate .usdc file that composes over the base.
        Remove it and the previous values reappear.
        """
        delta_layer = Sdf.Layer.CreateNew(
            f"workspace/stage/deltas/{agent_name}_{timestamp()}.usdc"
        )
        # Write delta values into the new layer
        for prim_path, attrs in delta_dict.items():
            prim_spec = Sdf.CreatePrimInLayer(delta_layer, prim_path)
            for attr_name, value in attrs.items():
                attr_spec = Sdf.AttributeSpec(prim_spec, attr_name, 
                                               Sdf.ValueTypeNames.Find(type(value)))
                attr_spec.default = value
        
        delta_layer.Save()
        # Sublayer into the stage (becomes strongest opinion)
        root = self.stage.GetRootLayer()
        root.subLayerPaths.insert(0, delta_layer.identifier)
    
    def select_profile(self, profile_name):
        """Variant selection = creative profile switch. Instant."""
        workflow = self.stage.GetPrimAtPath("/workflows/active")
        vset = workflow.GetVariantSet("creative_profile")
        vset.SetVariantSelection(profile_name)
    
    def reconstruct_clean(self):
        """
        Lossless reconstruction: read only the base layer, 
        ignoring all agent deltas. O(1) — it's just reading 
        a different layer, not computing anything.
        """
        base_layer = self.stage.GetRootLayer().subLayerPaths[-1]
        clean_stage = Usd.Stage.Open(base_layer)
        return clean_stage
    
    def rollback_to(self, n_deltas_ago):
        """Remove top N sublayers. Previous values reappear instantly."""
        root = self.stage.GetRootLayer()
        for _ in range(n_deltas_ago):
            root.subLayerPaths.remove(root.subLayerPaths[0])
    
    def flush(self):
        """Periodic persistence. Milliseconds for .usdc binary."""
        self.stage.GetRootLayer().Save()
    
    def export_scene(self, output_path):
        """Flatten and export as .usdc, .usda, or .usdz."""
        self.stage.Export(output_path)
    
    def _bootstrap_hierarchy(self):
        """Create the initial prim hierarchy for a new stage."""
        for path in ["/workflows", "/recipes", "/executions", 
                     "/agents", "/models", "/scenes"]:
            self.stage.DefinePrim(path)
```

### Latency Budget

| Operation | Mechanism | Latency |
|---|---|---|
| Read attribute | C++ hash lookup + Python wrapper | ~10μs |
| Write attribute | C++ set + dirty flag | ~10μs |
| Add sublayer | Append to layer stack, lazy recompose | ~1ms |
| LIVRPS resolve | Computed lazily on next read of affected attr | ~50μs |
| Variant selection | Index switch, no data copy | ~100μs |
| Flush to disk (.usdc) | Binary crate serialization | ~5ms for 200 prims |
| Load from disk (.usdc) | Binary crate deserialization | ~5ms for 200 prims |
| Export full stage (.usda) | ASCII serialization | ~20ms |
| Flatten to single layer | Composition evaluation | ~10ms |

For context: the ComfyUI REST API round-trip to queue a workflow is ~50-100ms. The USD operations are 1000x faster than the network call they support. Latency is a non-issue.

### File Layout

```
G:\comfyui-agent\workspace\stage\
├── root.usdc                           # Composed stage root
├── base\
│   ├── workflow_templates.usdc         # S: Specialize (weakest)
│   └── model_inventory.usdc           # Scout's model registry
├── recipes\
│   ├── sdxl_portrait_sharp.usdc       # R: Reference
│   └── flux_anime_vivid.usdc
├── profiles\
│   ├── explore.usdc                   # V: Variant (creative profiles)
│   ├── creative.usdc
│   ├── radical.usdc
│   └── integration.usdc
├── deltas\
│   ├── forge_20260325_001.usdc        # L: Local (agent modifications)
│   ├── forge_20260325_002.usdc
│   └── ...
├── scenes\                            # Path 2: Composed USD scene outputs
│   ├── run_047_scene.usdc
│   └── run_048_scene.usdc
└── inputs\                            # Path 3: USD scene inputs
    └── portrait_brief.usdc
```

Every `.usdc` file is a real USD layer. They compose through the root layer's sublayer stack. Git can track `.usda` exports for version history. The `.usdc` binary format is what the running system uses for speed.

---

## USD Scene Output (Path 2: Generation → Composable 3D Scene)

ComfyUI generates flat images. This system generates **composable 3D scenes.**

When a workflow executes, the agent doesn't just save the output PNG. It runs auxiliary generation passes (depth, normals, segmentation) and composes them into a USD scene. That scene is a real 3D asset — openable in any USD-compatible tool.

### Composition Pipeline

```
ComfyUI execution produces:
  image.png          (beauty render — the generated image)
  depth.exr          (from Depth Anything V2 or Marigold)
  normals.png        (from normal estimation model)
  segmentation.png   (from SAM or GroundingDINO)

Agent COMPOSITOR composes into USD scene:

/scenes/run_047/
├── /camera                    (UsdGeomCamera)
│   ├── focalLength = 85.0     (from prompt or estimation)
│   ├── fStop = 1.4
│   ├── horizontalAperture = 36.0
│   └── clippingRange = (0.1, 1000)
│
├── /environment
│   ├── /hdri                  (UsdLuxDomeLight)
│   │   └── texture:file = "environment.exr"
│   └── /lights[]              (estimated from image analysis)
│
├── /subject
│   ├── /mesh                  (UsdGeomMesh — depth → displacement)
│   │   ├── points = [...]     (reconstructed from depth map)
│   │   ├── faceVertexCounts
│   │   └── faceVertexIndices
│   ├── /material              (UsdShadeMaterial)
│   │   ├── diffuseColor       (image.png as texture)
│   │   └── normal             (normals.png as normal map)
│   └── /segmentation          (primvar for masking)
│       └── mask = [...]       (per-vertex from segmentation)
│
└── /metadata
    ├── workflow_hash           (links back to workflow stage)
    ├── quality_score = 8.4    (from Vision analysis)
    ├── generation_params       (cfg, steps, sampler, model)
    └── timestamp
```

### What This Enables

**Structural validation in the self-improving loop.** Vision currently scores aesthetically — "looks good," "skin tones muddy." With a USD scene, the system can validate geometrically:

| Validation | How | Impact |
|---|---|---|
| Depth consistency | Are depth values smooth? Discontinuities at edges? | Catches depth estimation failures before they affect downstream |
| Normal-depth agreement | Do normals match the surface implied by depth? | Catches impossible lighting/shadow combinations |
| Segmentation quality | Clean boundaries? Holes in mask? | Catches soft edges that break compositing |
| Camera consistency | Does perspective match requested focal length? | Catches "85mm look" that's actually wide-angle distortion |
| Lighting directionality | Do shadows agree with estimated light direction? | Catches physically impossible illumination |

This turns Vision's aesthetic score into a **multi-dimensional quality vector:**

```
quality = {
    "aesthetic": 8.4,          # Perceptual (existing)
    "depth_consistency": 0.92, # Geometric (new — from USD scene)
    "normal_agreement": 0.88,  # Geometric
    "segmentation_quality": 0.95,
    "camera_fidelity": 0.85,   # Does output match camera intent?
    "lighting_plausibility": 0.90
}
```

The self-improving loop now optimizes on 6 axes instead of 1. When depth consistency drops, the agent knows to adjust the depth estimation model or sampling parameters. When camera fidelity is low, it adjusts ControlNet conditioning. **The feedback is specific and actionable**, not just "try different cfg."

### Scene Export Formats

The composed scene can be exported as:
- `.usdc` — binary, fast, for programmatic access
- `.usda` — ASCII, human-readable, git-trackable
- `.usdz` — packaged with textures, shareable as single file
- `.glb` — via USD→glTF conversion, for web/AR preview

A `.usdz` output from ComfyUI is **unprecedented.** No other image generation system produces a composable 3D scene as its output. This is a first.

---

## USD Scene Input (Path 3: 3D Scene → ComfyUI Conditioning)

The reverse direction. A USD scene becomes the creative brief that DRIVES generation.

### The Camera Control Breakthrough

This solves the camera/lens control problem from first principles. Instead of:
- Prompt engineering: "shot on 85mm f/1.4, shallow depth of field" (ambiguous, model-dependent)
- LoRA training: weeks of work per lens profile (expensive, not generalizable)
- Prompt weighting: CLIP space doesn't understand optics (fundamental limitation)

The system reads a `UsdGeomCamera` prim with actual optical parameters:

```python
# Agent reads camera prim — no ambiguity
camera = UsdGeom.Camera.Get(input_stage, "/scene/camera")
focal_length = camera.GetFocalLengthAttr().Get()      # 85.0 (mm)
f_stop = camera.GetFStopAttr().Get()                    # 1.4
sensor_width = camera.GetHorizontalApertureAttr().Get() # 36.0 (mm, full frame)

# Agent translates to ComfyUI conditioning
conditioning = {
    "depth_of_field": compute_dof(focal_length, f_stop, subject_distance),
    "perspective_fov": compute_fov(focal_length, sensor_width),
    "bokeh_character": lookup_lens_profile(focal_length, f_stop),
    "prompt_additions": f"shallow depth of field, {focal_length}mm lens character",
    "controlnet_depth": render_depth_from_camera(camera, proxy_geometry),
}
```

The camera prim IS the ground truth. No interpretation needed. And it composes with everything else in the scene — lights, proxy geometry, environment — to create a complete conditioning package.

### Full Scene → Conditioning Pipeline

```
USD scene input: /scene/
│
├── /camera              → FOV, perspective, DoF simulation
│                          → ControlNet depth (rendered from camera POV)
│
├── /lights[]            → Lighting direction, intensity, color temp
│                          → Prompt weighting for light description
│                          → HDRI estimation / generation
│
├── /proxy_geometry      → ControlNet Depth (rendered depth pass)
│                          → ControlNet Normal (rendered normal pass)
│                          → ControlNet Canny (rendered edge pass)
│                          → Composition framing guidance
│
├── /materials[]         → Color palette extraction
│                          → Material descriptor prompt tokens
│                          → Style reference for IP-Adapter
│
└── /environment
    └── /hdri            → UsdLuxDomeLight texture as lighting reference
                           → Feed directly to HDRI-based lighting nodes
```

### The USD Round-Trip

Path 2 + Path 3 compose into a loop:

```
Iteration 1:
  Human provides: USD scene brief (/camera, /proxy_geo, /lights)
  Agent extracts: conditioning from scene prims
  ComfyUI generates: image + depth + normals
  Agent composes: output into USD scene
  Vision validates: geometric + aesthetic quality

Iteration 2:
  Agent reads: PREVIOUS output scene as input
  Agent compares: output geometry vs input intent
  Agent adjusts: conditioning based on geometric error
  ComfyUI generates: improved output
  Agent composes: new output scene
  Vision validates: delta from iteration 1

Convergence:
  Output scene geometry matches input scene intent
  Aesthetic quality meets threshold
  Camera fidelity confirmed against camera prim
  Recipe saved with both workflow params AND scene conditioning
```

**The USD scene IS the feedback channel.** Input scene → conditioning → generation → output scene → comparison → adjustment → iterate. Both sides of the loop speak the same language: USD.

---

## The COMPOSITOR: USD Scene Assembly

A new capability in the agent system. Not a 7th agent — a tool set available to Vision and Forge that handles USD scene composition.

### COMPOSITOR Tools (5 tools)

| Tool | Function |
|---|---|
| `compose_scene_from_outputs` | Takes generation outputs (image, depth, normals, segmentation) and composes into a USD scene with camera, mesh, materials, and metadata |
| `extract_conditioning_from_scene` | Reads a USD scene and produces a ComfyUI conditioning dictionary (ControlNet inputs, prompt additions, camera parameters) |
| `validate_scene_geometry` | Structural analysis: depth consistency, normal agreement, segmentation quality, camera fidelity |
| `compare_scenes` | Diff two USD scenes (input intent vs output result) — produces geometric error metrics for the self-improving loop |
| `export_scene` | Export composed scene as .usdc, .usda, .usdz, or .glb |

### Who Uses Them

- **Vision** uses `compose_scene_from_outputs` + `validate_scene_geometry` after every execution — adds geometric validation to aesthetic scoring
- **Architect** uses `extract_conditioning_from_scene` when designing workflows from USD scene briefs
- **Forge** uses the conditioning dict to wire ControlNet inputs and prompt modifications
- **The self-improving loop** uses `compare_scenes` to compute geometric convergence between input intent and output result

---

## Creative Injection → Workflow Profiles

The Digital Injection Framework transfers directly. Instead of modulating cognitive phases, we modulate workflow parameter groups:

### Profile → Parameter Gain Modulation

```
                    Parameter Groups (d_n = receptor density)
                    ──────────────────────────────────────────
Profile             Sampler  CFG    Steps  Denoise  LoRA_str  
────────────────────────────────────────────────────────────
none (baseline)     1.000    1.000  1.000  1.000    1.000
explore             1.003    1.010  1.005  1.002    1.008
creative            1.008    1.020  1.010  1.005    1.015
radical             1.015    1.035  1.020  1.010    1.025
integration         1.005    1.012  1.008  1.003    1.010
```

**How this works in practice:**

```
/inject explore

The agent system shifts toward exploratory parameter choices:
- Sampler selection biases toward less common samplers (dpmpp_3m_sde)
- CFG range expands (willing to try values outside proven recipes)
- Step count increases (more compute for quality exploration)
- Denoise strength varies more (willing to deviate from standard)
- LoRA strength explores edges (0.4-0.9 instead of safe 0.6-0.7)

/inject radical

Full dissolution — the agent ignores learned recipes and tries
fundamentally different parameter combinations. Useful when stuck
in a local optimum. The self-improving loop will always recover
the clean workflow if radical exploration fails.
```

**Variant sets implement this cleanly:**

```python
# Workflow prim with creative profile variants
workflow_prim.CreateVariantSet("creative_profile")

# Each profile is a variant with pre-authored gain tables
for profile in ["explore", "creative", "radical", "integration"]:
    vset.AddVariant(profile)
    vset.SetVariantSelection(profile)
    # Set gain-modulated parameter ranges for this profile
    workflow_prim.GetAttribute("cfg_range").Set(gain_table[profile]["cfg"])
    workflow_prim.GetAttribute("sampler_pool").Set(gain_table[profile]["samplers"])
```

Profile switching is a single variant selection. Because variants are weaker than local opinions in LIVRPS, a human saying "use cfg=7" still overrides any profile's suggestion. Exactly like the cognitive substrate.

---

## The Cognitive Bridge → Agent Bridge

The Cognitive Bridge dissolves Claude's project silo problem. The Agent Bridge dissolves the agent silo problem. Same architecture:

### The Problem
Each agent (Scout, Architect, Forge, Crucible, Vision) currently communicates through typed artifacts in a chain. That's fine for coordination. But for LEARNING, each agent needs access to the full history of what every other agent has discovered.

Vision knows that "cfg>8 + SDXL = muddy skin." But Architect doesn't have access to that knowledge when designing the next workflow. The artifact chain only passes forward, not backward.

### The Solution: Unified Agent Memory Stage

```
agent_bridge_stage.usda (composed — unified across all agents)
│
├── /sessions/                    # Per-execution session layers
│   ├── session_2026_03_25_001.usda
│   └── session_2026_03_25_002.usda
│
├── /patterns/                    # Cross-session learned patterns
│   ├── sdxl_skin_tones.usda     # "cfg>8 = muddy" (from Vision)
│   ├── flux_composition.usda    # "aspect ratio affects composition quality"
│   └── sampler_speed.usda       # "dpmpp_2m fastest for similar quality"
│
├── /entities/                    # Models, node packs, techniques
│   ├── sdxl_base.usda           # Everything known about this model
│   ├── controlnet_depth.usda    # Capabilities, limitations, best configs
│   └── ipadapter.usda           # Compatibility, optimal strength ranges
│
└── /decisions/                   # Key architectural decisions with rationale
    ├── chose_euler_a_for_anime.usda
    └── switched_to_flux_for_portraits.usda
```

### MCP Tools (Mapped from Cognitive Bridge)

| Cognitive Bridge Tool | Agent Bridge Tool | Function |
|---|---|---|
| `memory_query` | `agent_memory_query` | Search unified stage by model, technique, quality score |
| `memory_ingest` | `agent_memory_ingest` | Add execution result + analysis to stage |
| `memory_map` | `agent_knowledge_map` | Visualize what's known about model/technique relationships |
| `memory_timeline` | `agent_execution_timeline` | Chronological view of all runs, scores, decisions |
| `memory_stage_health` | `agent_stage_health` | Composition health: layer count, staleness, conflicts |

### Cross-Agent Knowledge Flow

```
Vision analyzes output:
  → "cfg=8.5 produced muddy skin tones with SDXL base"
  → agent_memory_ingest: pattern stored in /patterns/sdxl_skin_tones.usda

Next session, Architect designs a portrait workflow:
  → agent_memory_query: "sdxl portrait" 
  → Returns: pattern from Vision + recipe from previous best run
  → Architect's design document ALREADY KNOWS to avoid cfg>8
  → No wasted iteration. No rediscovery.

This is cross-agent learning through composed stage traversal.
Not message passing. Not shared databases. COMPOSITION.
```

---

## The Derivative Tool Pattern → Agent Inspectors

From the Cognitive Twin's SuperLayer/Gaffer derivative pattern. Each is a filtered lens into the same composed workflow stage:

### WorkflowGaffer
**What it does:** Filtered view showing only tunable parameters with their current values, which agent set them, and which composition arc they came from.

```
┌── WorkflowGaffer: portrait_v3 ──────────────────────────┐
│                                                          │
│ KSampler_01                                              │
│   steps    = 28        [Local: Forge]      score: 8.4    │
│   cfg      = 7.0       [Reference: recipe] score: 8.4   │
│   sampler  = euler_a   [Variant: anime]    —             │
│   seed     = 42        [Local: Human]      —             │
│                                                          │
│ CLIPTextEncode_pos_01                                    │
│   text     = "photo..."[Local: Architect]  —             │
│                                                          │
│ ── ANCHOR (read-only) ──                                 │
│ CheckpointLoader_01                                      │
│   ckpt     = sdxl_base [ANCHOR]            🔒            │
│                                                          │
│ Profile: explore  │  Iteration: 3/15  │  Best: 8.4      │
└──────────────────────────────────────────────────────────┘
```

**Who uses it:** Forge (for precise parameter patching), Human (for oversight)

### AgentInspector
**What it does:** Shows which agents have acted, what they decided, and why. The Router's dashboard.

```
┌── AgentInspector: session_001 ───────────────────────────┐
│                                                          │
│ Chain: Scout → Architect → [GATE ✓] → Provision → Forge │
│        → Crucible → Vision                               │
│                                                          │
│ Scout     ✓  RECON: 47 models, 312 node types, SDXL     │
│              MISSING: sdxl_lightning_4step (checkpoint)   │
│ Architect ✓  DESIGN: portrait pipeline, 6 nodes, euler_a │
│              DEPS: 1 model (4.1GB), 0 node packs         │
│ Human     ✓  GATE: approved (incl. 4.1GB download)       │
│ Provision ✓  PULLED: sdxl_lightning_4step → models/ckpt  │
│              VERIFY: hash ✓, ComfyUI visible ✓           │
│ Forge     ✓  BUILD: patched cfg=7.0, steps=4             │
│ Crucible  ✓  VERIFY: PASS (12 tests, 0 failures)        │
│ Vision    ⟳  ANALYZING output...                         │
│                                                          │
│ Constitutional violations: 0                              │
│ Provisioned this session: 4.1GB (1 model)                │
│ Artifacts: 6 produced, 6 valid                           │
└──────────────────────────────────────────────────────────┘
```

**Who uses it:** Router (for orchestration), Human (for trust building)

### ConvergenceMonitor
**What it does:** Tracks the self-improving loop's progress toward convergence.

```
┌── ConvergenceMonitor: portrait_optimization ─────────────┐
│                                                          │
│ Score trajectory:                                        │
│   Run 1: 5.2  ████░░░░░░                                │
│   Run 2: 6.8  ██████░░░░  (+1.6)                        │
│   Run 3: 7.9  ███████░░░  (+1.1)                        │
│   Run 4: 8.4  ████████░░  (+0.5)                        │
│   Run 5: 8.5  ████████░░  (+0.1) ← plateau detected    │
│                                                          │
│ Parameter impact (highest → lowest):                     │
│   cfg:     ΔScore 2.1 (changed 8.0 → 7.0)              │
│   steps:   ΔScore 0.8 (changed 20 → 28)                │
│   sampler: ΔScore 0.4 (changed euler → euler_a)         │
│   seed:    ΔScore 0.0 (random, no correlation)          │
│                                                          │
│ Status: CONVERGED (score delta < 0.3 for 2 runs)        │
│ Best run: #4 (score 8.4)                                │
│ Recipe saved: /recipes/sdxl_portrait_sharp               │
└──────────────────────────────────────────────────────────┘
```

**Who uses it:** Router (for convergence decisions), Meta-analyzer (for self-improvement)

---

## The Self-Improving Loop as USD Composition

Here's where the architecture becomes genuinely powerful. Each iteration of the self-improving loop is a **sublayer** in the composed workflow stage. This means:

### Iteration History IS Layer History

```
Layer stack (bottom to top, weak to strong):
────────────────────────────────────────────
1. base_workflow.usda          (S: Specialize — template)
2. recipe_sdxl_portrait.usda   (R: Reference — learned recipe)
3. profile_explore.usda        (V: Variant — creative profile)
4. iteration_001_delta.usda    (L: Local — agent's first attempt)
5. iteration_002_delta.usda    (L: Local — second attempt, overrides first)
6. iteration_003_delta.usda    (L: Local — third attempt)
```

**What this gives you:**

- **Rollback to any iteration:** Remove the top sublayer. Previous iteration's values appear.
- **Compare any two iterations:** Diff the sublayers. See exactly what changed.
- **Merge successful experiments:** Flatten successful deltas into a new recipe (Reference arc). That recipe is now available to ALL future workflows, not just this one.
- **Branch and explore:** Fork the layer stack. Try two different parameter directions in parallel. Compare results. Keep the winner.

This is **version control for creative exploration** — with the same composition semantics that VFX production uses for shot versioning.

### The Meta-Analyzer as Stage Traversal

After convergence, the meta-analyzer doesn't "analyze a database." It **traverses the composed stage:**

```python
# Pseudocode — meta-analysis as stage traversal

for execution in stage.GetPrimAtPath("/executions").GetChildren():
    score = execution.GetAttribute("quality_score").Get()
    params = execution.GetAttribute("memory:content").Get()
    
    # Extract parameter → score correlations
    for param_name, param_value in params.items():
        correlations[param_name].append((param_value, score))

# Identify which parameter axes drove improvement
for param, data in correlations.items():
    impact = compute_correlation(data)
    if impact > threshold:
        # This parameter matters. Store as pattern.
        pattern_prim = stage.DefinePrim(f"/patterns/{param}_impact")
        pattern_prim.CreateAttribute("direction", Sdf.ValueTypeNames.String).Set(
            f"{param}: optimal range {best_range}"
        )
```

The patterns extracted from stage traversal become new sublayers in the Agent Bridge. Next session, every agent has access to them through `agent_memory_query`. The system learned. And the learning is composable, queryable, and lossless.

---

## The Ratchet (Karpathy Loop × USD Composition)

Karpathy's autoresearch discovered 20 optimizations in 700 experiments over 2 days. One agent, one metric (`val_bpb`), one ratchet: beat the baseline or get discarded. The pattern is universal — "any metric you care about that is reasonably efficient to evaluate can be autoresearched."

This ecosystem was designed for the same loop before Karpathy named it. But our version is structurally more powerful in three ways.

### Why USD Sublayers > Git Commits

Autoresearch uses git. Linear history, one path, revert means undoing commits. Our system uses USD composition:

```
Autoresearch (git):
  commit 1: change lr → KEEP (new baseline)
  commit 2: change depth → DISCARD (revert)
  commit 3: change heads → KEEP
  History: linear. One path. Can't compare commit 1 and 3 without commit 2.

This ecosystem (USD sublayers):
  sublayer A: cfg=7.0, steps=28 → score 7.8, KEEP
  sublayer B: cfg=6.5, steps=28 → score 8.2, KEEP (composes OVER A)
  sublayer C: sampler=dpmpp_3m → score 7.9, DISCARD (remove sublayer)
  
  Sublayer A still exists beneath B. Remove B and A's values reappear.
  Fork the stack: try two directions simultaneously.
  Flatten winners into a recipe (Reference arc) for all future workflows.
```

### Why Multi-Dimensional Metrics > Single Metric

Autoresearch optimizes one number. Our 6-axis quality vector lets the ratchet be smarter:

```
Experiment 47: aesthetic=8.4, depth=0.92, normals=0.88, seg=0.95, camera=0.85, light=0.90
Experiment 48: aesthetic=8.1, depth=0.95, normals=0.93, seg=0.96, camera=0.90, light=0.88

Single-metric ratchet: 48 loses (8.1 < 8.4). DISCARD.
Multi-axis ratchet:    48 wins on 4/6 axes. KEEP — the aesthetic dip is 
                       offset by geometric improvements. Fix aesthetic 
                       independently on the next iteration.
```

The ratchet doesn't just keep/discard. It knows WHICH axes improved and which regressed, so the next experiment can target the weak axis specifically.

### Why Multi-Agent > Single Agent

Autoresearch is one agent proposing changes. Our MoE system has 6 specialists:

```
Single agent: proposes "try lr=0.001" → trains → measures → keeps/discards

MoE team:     Scout discovers "this model family responds to cfg changes"
              Architect designs "test cfg range 5.5-8.0 in 0.5 increments"
              Provisioner ensures all models ready
              Forge patches workflow, queues to ComfyUI (30sec per experiment)
              Crucible verifies patch applied correctly
              Vision scores on 6 axes with geometric validation
              
              Quality of each experiment is higher because specialists
              handle their domain. Fewer wasted experiments.
```

### The Ratchet Mechanism

Two modes, same USD machinery:

**RATCHET MODE (autonomous overnight):** Simple. Aggressive. Binary.

```python
class Ratchet:
    """The Karpathy Loop on USD sublayers."""
    
    def __init__(self, stage, workflow, target, metric_weights):
        self.stage = stage
        self.baseline_score = self.evaluate(workflow)
        self.baseline_layer = stage.GetRootLayer().subLayerPaths[0]
        self.experiments = 0
        self.keeps = 0
    
    def run_experiment(self, delta):
        """One iteration of the ratchet."""
        # Apply delta as sublayer
        self.stage.add_agent_delta("ratchet", delta)
        
        # Execute in ComfyUI (fixed time budget)
        output = comfyui.queue_and_wait(
            self.stage.flatten_to_workflow_json()
        )
        
        # Score on all axes
        score = self.evaluate(output)
        self.experiments += 1
        
        # THE RATCHET: beat baseline or get discarded
        if self.weighted_score(score) > self.weighted_score(self.baseline_score):
            # KEEP — new baseline
            self.baseline_score = score
            self.baseline_layer = self.stage.GetRootLayer().subLayerPaths[0]
            self.keeps += 1
            self.stage.flush()
            return "KEEP", score
        else:
            # DISCARD — remove sublayer, baseline values reappear
            self.stage.rollback_to(1)
            return "DISCARD", score
    
    def extract_recipe(self):
        """After N experiments, flatten winning sublayers into a recipe."""
        recipe_layer = self.stage.flatten_kept_deltas()
        # Store as Reference arc — available to all future workflows
        self.stage.add_recipe(recipe_layer)
```

**CONVERGENCE MODE (interactive, human-watched):** Sophisticated. Multi-axis. Plateau-aware.

```python
class ConvergenceMonitor:
    """For interactive sessions. Detects plateaus, suggests pivots."""
    
    def check(self, history):
        # Score plateau: delta < 0.3 for 3 consecutive runs
        if self.score_plateaued(history, threshold=0.3, window=3):
            return "CONVERGED"
        
        # Perceptual stability: output images barely changing
        if self.perceptual_hash_stable(history, window=2):
            return "CONVERGED"
        
        # Score regression: significant drop
        if self.score_regressed(history, delta=-0.5):
            return "REVERT_AND_PIVOT"  # Try different parameter axis
        
        # Budget exhausted
        if self.budget_exceeded(history):
            return "STOP_BEST"
        
        return "CONTINUE"
```

Both modes write to the same USD stage. Ratchet mode produces a layer stack overnight. Convergence mode produces a layer stack interactively. The morning after a ratchet run, you can open the stage in convergence mode and inspect every experiment.

---

## Three Runtime Modes

The ecosystem operates in three modes. Same agents, same stage, same tools. Different orchestration patterns.

### Mode 1: Interactive (`agent run`)

Human in the loop. Conversational. The current comfyui-agent experience, enhanced with cognitive state.

```bash
agent run --session portrait-project
```

The Router responds to natural language. Agents think through their approach. The human sees every step. Good for exploration, learning, creative direction.

### Mode 2: Orchestrated (`agent orchestrate`)

Human sets the goal and approves the plan. Agents execute the full chain autonomously. Human reviews the result.

```bash
agent orchestrate build \
  --goal "SDXL workflow for anime landscapes with ControlNet depth" \
  --session anime-landscapes

agent orchestrate optimize \
  --workflow portrait.json \
  --target "photorealistic portrait, sharp details, natural skin" \
  --max-iterations 15
```

The Router runs the full chain: Scout → Architect → [GATE] → Provisioner → Forge → Crucible → Vision → iterate. Human gates at design approval. Good for focused tasks with clear goals.

### Mode 3: Autoresearch (`agent autoresearch`)

Human writes the program, goes to sleep. Agents run the ratchet all night. Human reviews the results in the morning.

```bash
agent autoresearch \
  --workflow portrait.json \
  --program program.md \
  --metric "aesthetic:0.4 + depth:0.2 + camera:0.2 + normals:0.1 + light:0.1" \
  --budget-hours 8 \
  --experiment-seconds 30 \
  --max-experiments 500
```

The `program.md` is the Karpathy equivalent — a markdown file describing what to explore:

```markdown
# program.md — Portrait Quality Autoresearch

## Objective
Maximize photorealistic portrait quality with natural skin detail.

## Parameter Axes to Explore
- cfg_scale: range 5.0-9.0 (current baseline: 7.0)
- steps: range 15-40 (current baseline: 20)
- sampler: [euler_a, dpmpp_2m_sde, dpmpp_3m_sde, uni_pc]
- denoise_strength: range 0.5-1.0 (current baseline: 0.7)
- lora_strength: range 0.3-0.9 (if anime_style lora loaded)

## Fixed (Anchor — Do Not Modify)
- checkpoint: sdxl_base_1.0.safetensors
- resolution: 1024x1024
- safety_checker: enabled

## Strategy
- Start with single-axis experiments (change one param at a time)
- After establishing best single-axis values, try combinations
- If score plateaus, try a different sampler before giving up on an axis
- Prioritize experiments that improve depth_consistency (current weakest axis)

## Success Criteria
- Aesthetic score > 8.5
- All geometric axes > 0.90
- Recipe saved for future use
```

**What runs overnight:**

```
22:00  Ratchet starts. Baseline score: 6.2
22:01  Exp 1:  cfg=6.0              → 6.5  KEEP  (new baseline)
22:02  Exp 2:  cfg=5.5              → 6.1  DISCARD
22:03  Exp 3:  steps=28             → 7.1  KEEP
22:04  Exp 4:  steps=35             → 7.0  DISCARD (marginal regression)
22:05  Exp 5:  sampler=dpmpp_2m_sde → 7.4  KEEP
...
02:00  Exp 237: combination test    → 8.6  KEEP
02:01  Exp 238: lora_str=0.7        → 8.5  DISCARD
02:02  Score plateau detected (3 consecutive discards). Pivoting axis.
02:03  Exp 239: denoise=0.85        → 8.7  KEEP
...
06:00  Budget exhausted. 412 experiments. 23 keeps. Best: 8.9

MORNING REPORT:
  Experiments: 412
  Keeps: 23 (5.6% keep rate)
  Starting score: 6.2
  Final score: 8.9
  Improvement: 43.5%
  
  Most impactful axes:
    1. sampler (dpmpp_2m_sde → +0.9)
    2. cfg (6.0 → +0.3)  
    3. steps (28 → +0.6)
    4. denoise (0.85 → +0.2)
  
  Recipe saved: /recipes/sdxl_portrait_autoresearch_20260325.usdc
  Layer stack: 23 kept sublayers composable in stage
  Full log: /executions/autoresearch_20260325/
```

**1,000+ experiments overnight. On your 4090. While you sleep.** The morning report is a recipe you can use immediately, a layer stack you can inspect experiment-by-experiment, and a full log of what the agents tried and why.

### Mode Comparison

| | Interactive | Orchestrated | Autoresearch |
|---|---|---|---|
| Human involvement | Continuous | Gate at design | Write program.md, check morning |
| Agent reasoning | Full dialogue | Full chain, silent | Minimal — ratchet is tight loop |
| Experiments/hour | 2-5 (human bottleneck) | 10-20 (agent chain) | 120-360 (30sec GPU budget) |
| Quality per experiment | Highest (human judgment) | High (full agent chain) | Lower (speed over depth) |
| Best for | Exploration, learning | Focused builds | Parameter optimization, recipe discovery |
| Duration | Session | Minutes to hours | Hours to overnight |
| USD output | Layer stack | Layer stack | Layer stack + recipe |

---

## The Full System: Agent Team + Cognitive Stage

Putting it all together. The Agent Team Blueprint provides the MoE coordination layer. The Cognitive Stage provides the shared state and composition layer. Together:

```
┌──────────────────────────────────────────────────────────────────┐
│                        HUMAN                                      │
│            (intent, gates, creative direction)                     │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                     ROUTER (MoE)                                  │
│                                                                   │
│  Task classification │ Expert dispatch │ Constitutional enforcer  │
│  Convergence logic   │ Human gates     │ Retry budgets            │
│                                                                   │
│  READS FROM: Cognitive Workflow Stage (current state)             │
│  WRITES TO:  Agent sublayers (orchestration decisions)            │
└───┬─────┬───────┬──────────┬──────────┬──────────┬──────────┘
    │     │       │          │          │          │
┌───▼──┐┌─▼───┐┌──▼─────┐┌──▼───┐┌────▼───┐┌────▼────┐
│SCOUT ││ARCHI-││PROVIS- ││FORGE ││CRUCIB- ││ VISION  │
│      ││TECT  ││IONER   ││      ││LE      ││         │
│ READ ││DESIGN││RESOLVE ││BUILD ││BREAK   ││ANALYZE  │
│ ONLY ││ ONLY ││PULL    ││+EXEC ││+TEST   ││+SCORE   │
│      ││      ││PLACE   ││      ││        ││         │
└───┬──┘└──┬───┘└──┬─────┘└──┬───┘└───┬────┘└────┬────┘
    │      │       │         │        │          │
    │      │       │         │        │          │    (Typed artifacts
    │      │       │         │        │          │     between agents)
    │      │       │         │        │          │
┌───▼──────▼───────▼─────────▼────────▼──────────▼────────────┐
│              COGNITIVE WORKFLOW STAGE                              │
│              (USD-composed shared state)                           │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ COMPOSITION LAYER STACK                                   │    │
│  │                                                           │    │
│  │  Human overrides          (L — strongest)                 │    │
│  │  Agent role constraints   (I — inherit from role def)     │    │
│  │  Creative profiles        (V — variant selection)         │    │
│  │  Learned recipes          (R — from memory stage)         │    │
│  │  Lazy components          (P — ControlNet, IP-Adapter)    │    │
│  │  Base templates           (S — weakest, always present)   │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐      │
│  │ Anchor      │  │ Lossless     │  │ Integrity          │      │
│  │ Parameters  │  │ Deltas       │  │ Verification       │      │
│  │ (immune)    │  │ (reversible) │  │ (per-execution)    │      │
│  └─────────────┘  └──────────────┘  └────────────────────┘      │
│                                                                   │
│  DERIVATIVE TOOLS:                                                │
│  WorkflowGaffer │ AgentInspector │ ConvergenceMonitor            │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    AGENT BRIDGE                                    │
│              (Cross-agent memory stage)                            │
│                                                                   │
│  /patterns/    Cross-session learned knowledge                    │
│  /entities/    Model & technique understanding                    │
│  /decisions/   Architectural choices with rationale               │
│  /executions/  Full history as AIMemoryChunk prims                │
│                                                                   │
│  MCP Tools: query │ ingest │ knowledge_map │ timeline │ health   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           │  REST API + WebSocket
┌──────────────────────────▼───────────────────────────────────────┐
│                      ComfyUI Server                               │
│                      Port 8188                                    │
│                                                                   │
│  Receives: Composed workflow JSON (flattened from stage)          │
│  Returns:  Execution progress + output images                     │
│  Monitors: WebSocket for real-time status                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## What Makes This Different From Any Other Agent System

### 1. Conflict Resolution Is Solved
Most multi-agent systems break when two agents disagree. One overwrites the other, or they deadlock. LIVRPS means there's NEVER a conflict — just opinions at different composition strengths. The stage resolves deterministically. Every time.

### 2. Nothing Is Ever Lost
Every agent modification is a sublayer. Every execution result is an AIMemoryChunk prim. Remove a sublayer — previous values appear. The system has perfect memory with graceful degradation through `decay_weight`.

### 3. Learning Is Structural, Not Statistical
Most "learning" systems build statistical models from execution data. This system learns by composing new sublayers into the stage. A learned recipe IS a USD layer. It composes with everything else using the same rules. It can be inspected, modified, overridden, or removed — because it's data, not a trained weight.

### 4. Creative Profiles Are First-Class
The injection mapping means "try something wildly different" is a variant selection, not a prayer. Switch to "radical" profile, the parameter ranges widen. Switch back to "none," the proven recipes return. The creative exploration is contained, reversible, and composable.

### 5. VFX Pipeline Wisdom Applies Directly
Shot versioning patterns, department layer organization, asset referencing, deferred payload loading, render-time overrides — 16 years of VFX pipeline knowledge transfers directly. Every workflow is a shot. Every agent is a department. Every modification is a layer.

### 6. The Derivative Tools Give Real Visibility
WorkflowGaffer, AgentInspector, ConvergenceMonitor — these aren't dashboards bolted on after the fact. They're filtered views into the same composed stage, exactly like SuperLayer and Gaffer are filtered views into a Katana scene. The data is one thing. The views are many.

### 7. Self-Provisioning Closes the Autonomy Loop
Most agent systems can design and build workflows but stop dead when a model isn't installed. "Please download SDXL Lightning and try again." The Provisioner agent — following the Ollama pattern — resolves, downloads, verifies, and registers models and node packs autonomously. Models are Payload prims in the composed stage: declared as references, materialized on demand, tracked with hash verification and compatibility metadata. The system provisions its own dependencies. No human file management. No broken workflows because a LoRA is missing. Declare what you need, the system handles it.

### 8. USD-Native From Day One — No Translation Layers
The stage isn't a database simulating USD. It IS a `pxr.Usd.Stage` running real C++ composition behind Python bindings. Microsecond reads. Millisecond writes. LIVRPS is built in, not reimplemented. This means every tool in the USD ecosystem — Houdini, Katana, usdview, custom pipeline tools — can read the stage directly. No export step. No format conversion. The workflow stage IS the interchange format.

### 9. Generation Outputs Are Composable 3D Scenes
No other image generation system produces a composable 3D scene as its output. This system composes depth maps, normal maps, and segmentation masks into USD scenes with real geometry, materials, and camera data. That scene is openable in any USD tool, renderable, and — critically — validatable against geometric ground truth. The self-improving loop doesn't just optimize aesthetics. It optimizes geometry.

### 10. USD Round-Trip Grounds the Loop in Physics
Input scene (camera + lights + proxy geometry) → conditioning → generation → output scene → geometric comparison → iterate. Both sides of the loop speak USD. The feedback is structural, not just perceptual. "Depth inconsistency at left edge" is actionable in a way "looks a bit off" isn't.

### 11. The Karpathy Loop, Industrialized
Autoresearch runs one agent in a git loop with one metric. This ecosystem runs 6 specialist agents on a USD-composed stage with 6-axis scoring. The ratchet is the same — beat the baseline or get discarded — but sublayers replace git commits (composable, non-destructive, branchable), multi-dimensional metrics replace single-number optimization, and specialist agents replace a generalist. The result: 1,000+ overnight experiments on a single 4090, producing a recipe that's immediately usable, a layer stack that's fully inspectable, and learned patterns that persist across sessions.

### 12. The System Improves How It Improves (Hyperagent Pattern)
Most self-improving systems have a fixed meta-layer. They optimize the task but the optimization process itself never changes. From the Hyperagents paper (Zhang et al.): the meta-level modification procedure is itself editable. Our meta-agent can rewrite agent prompts, routing heuristics, and its own modification strategy — but only if the ratchet proves the change actually works. The scoring function is an anchor (constitutionally protected). The constitutional commandments are anchors. Everything else can evolve. And because improvements are USD sublayers, they transfer across domains and accumulate across runs. The system doesn't just search for better workflows — it continually improves its search for how to improve.

---

## Implementation Phases

### Phase 1: USD-Native Foundation (Weeks 1-2)
- Move repo to `G:\comfyui-agent`, create workspace symlinks
- `CognitiveWorkflowStage` class: `pxr.Usd.Stage` wrapper with read/write/delta/flush API
- ComfyUI workflow JSON → USD prim hierarchy bidirectional mapper
- Bootstrap hierarchy: `/workflows`, `/recipes`, `/executions`, `/agents`, `/models`, `/scenes`
- Define anchor parameters and implement structural immunity (gain=1.0, no code path to modify)
- Agent Team Blueprint Phase 1 (CLAUDE.md files for all 6 agents, artifact schemas)
- Verify: `from pxr import Usd, UsdGeom, Sdf` imports clean from Houdini Python
- Verify: microsecond read/write latency on 200-prim test stage

### Phase 1.5: Self-Provisioning — The Ollama Pattern (Week 3)
- Multi-source model resolver (HuggingFace API, CivitAI API, ComfyUI Manager registry)
- Streaming downloader with resume, progress callbacks, SHA256 verification
- Node pack installer (git clone + pip install + ComfyUI refresh signal)
- Post-download verifier (ComfyUI `/object_info` API confirms model visible)
- Stage registrar (new model → new prim in `/models/` with typed attributes, status: available→ready)
- Wire into chain: Design Document lists deps + sizes → Human Gate → Provisioner → Forge
- Disk space safety: check before download, sequential large checkpoints, budget tracking

### Phase 2: Composition Engine (Weeks 4-5)
- LIVRPS conflict resolution via native USD sublayer ordering
- Agent deltas as `.usdc` sublayers (add/remove = instant opinion change)
- `reconstruct_clean()` as base-layer read (not computation — just read a different layer)
- Integrity verification per-execution (clean_hash, anchor_hash, fidelity)
- Agent Bridge MCP tools (query, ingest, knowledge_map, timeline, health)
- Variant sets for creative profiles (pre-authored in `/profiles/*.usdc`)

### Phase 3: Agent Integration (Weeks 6-7)
- Router connected to Cognitive Workflow Stage (reads state, writes orchestration sublayers)
- Each agent reads/writes through composition arcs, not direct mutation
- Derivative tools: WorkflowGaffer, AgentInspector, ConvergenceMonitor
- Self-improving loop wired through composition (iteration = sublayer)
- Tool count update: 61 existing + 5 PROVISION + 5 COMPOSITOR = 71 tools

### Phase 4: USD Scene I/O (Weeks 8-9)
- COMPOSITOR tools: `compose_scene_from_outputs`, `validate_scene_geometry`, `compare_scenes`
- Auxiliary generation passes: depth (Depth Anything V2), normals, segmentation (SAM)
- Depth → UsdGeomMesh reconstruction pipeline
- Image + normals → UsdShadeMaterial composition
- UsdGeomCamera extraction → ComfyUI conditioning pipeline (FOV, DoF, perspective)
- Scene export: .usdc, .usda, .usdz, .glb
- Multi-dimensional quality vector (aesthetic + geometric) for self-improving loop

### Phase 5: The Ratchet — Autoresearch Mode (Weeks 10-11)
- Ratchet class: binary keep/discard on USD sublayers against weighted multi-axis baseline
- `program.md` parser: extract parameter axes, ranges, strategies, anchor constraints
- Experiment runner: fixed GPU time budget (configurable, default 30s), automatic queue/wait/score
- Morning report generator: experiment count, keep rate, impact analysis, best recipe
- `agent autoresearch` CLI entry point with --program, --metric, --budget-hours, --experiment-seconds
- Recipe extraction: flatten winning sublayers into a Reference arc recipe `.usdc`
- Integration with existing ConvergenceMonitor (ratchet feeds the same monitoring dashboard)
- Overnight mode: watchdog process, crash recovery, resume from last kept sublayer

### Phase 6: Creative Injection (Weeks 12-13)
- Gain modulation tables for parameter groups (sampler, cfg, steps, denoise, lora)
- Profile switching via `/inject` command → variant selection on workflow prim
- Alpha interpolation for smooth profile transitions
- Injection profiles as exploration strategies in autoresearch (radical profile = wider parameter ranges)

### Phase 7: Claude Code MoE (Weeks 14-15)
- Router as top-level Claude Code agent with `--dangerously-skip-permissions`
- Sub-agent spawning with role-specific CLAUDE.md + tool filtering
- Constitutional enforcement as pre/post action validation
- Three runtime modes wired: interactive, orchestrated, autoresearch
- Full autonomous loop across all modes

### Phase 8: Hyperagent Meta-Layer (Weeks 16-18)

The meta-layer is not a fixed analyzer. It's a **meta-agent** — a 7th Claude instance that can modify the system AND modify how it modifies the system. From the Hyperagents paper (Zhang et al., 2603.19461): the meta-level modification procedure is itself editable, enabling metacognitive self-modification.

**Three-tier evolution boundary:**

```
TIER 1 — AUTO-EVOLVE FREELY (no gate, no ratchet needed)
  Recipes (parameter combinations that scored well)
  Routing weights (which agent combos produce best results)
  Memory patterns (cross-session learned knowledge)
  Exploration strategies (which parameter axes to try first)

TIER 2 — AUTO-EVOLVE WITH RATCHET VALIDATION (meta-agent proposes, ratchet proves)
  Agent prompt tuning (wording changes, example additions, emphasis shifts)
  Optimization parameters (convergence thresholds, scoring axis weights)
  Ratchet exploration strategy (single-axis vs multi-axis, pivot timing)
  Meta-agent's own modification strategy (how it decides what to change next)
  
  Mechanism: meta-agent proposes change → ratchet runs N experiments with 
  old prompt vs new prompt → statistically significant improvement? KEEP : DISCARD
  Both versions are USD sublayers. The old version is always recoverable.

TIER 3 — HUMAN GATE (always, no exceptions)
  Constitutional commandments (the 8 rules)
  Agent role definitions (tool access lists, authority boundaries)
  Structural prompt changes (new agent roles, new tool categories, new phases)
  Anchor parameter definitions (what's constitutionally protected)
  Scoring function (how "better" is defined — THE critical anchor)
  New agent creation
```

**The scoring function is an anchor.** The meta-agent can change what gets tested and how experiments are structured, but it CANNOT change how results are scored. If it could rewrite the scoring function, it could trivially game itself into "improving" while actually degrading. The 6-axis quality vector's evaluation logic is constitutionally protected — same as model paths and safety filters.

**Implementation:**

```python
class MetaAgent:
    """
    The Hyperagent. Modifies the system AND its own modification strategy.
    Lives as a Claude instance with its own CLAUDE.md and persistent memory.
    All modifications are USD sublayers — composable, reversible, inspectable.
    """
    
    def __init__(self, stage, ratchet):
        self.stage = stage
        self.ratchet = ratchet
        self.strategy_prim = stage.GetPrimAtPath("/agents/meta/strategy")
        self.history_prim = stage.GetPrimAtPath("/agents/meta/history")
    
    def propose_prompt_tuning(self, agent_name, observation):
        """
        Tier 2: Propose a prompt change for an agent.
        The change is tested via ratchet before deployment.
        """
        current_prompt = self.read_agent_prompt(agent_name)
        
        # Meta-agent reasons about what to change
        proposed_prompt = self.generate_improvement(
            current_prompt, observation, self.get_strategy()
        )
        
        # Ratchet validates: run N experiments with each prompt
        old_score = self.ratchet.evaluate_with_prompt(agent_name, current_prompt, n=10)
        new_score = self.ratchet.evaluate_with_prompt(agent_name, proposed_prompt, n=10)
        
        if statistically_significant(new_score, old_score, p=0.05):
            # KEEP — new prompt as sublayer (old is still beneath it)
            self.stage.add_agent_delta("meta", {
                f"/agents/{agent_name}/prompt": proposed_prompt
            })
            return "KEEP", new_score, old_score
        else:
            return "DISCARD", new_score, old_score
    
    def improve_own_strategy(self, run_history):
        """
        The Hyperagent pattern: modify the modification procedure itself.
        The meta-agent's strategy for proposing changes is a USD sublayer.
        It can be improved, rolled back, compared — just like everything else.
        """
        current_strategy = self.get_strategy()
        
        # Analyze: which of my proposed changes were kept vs discarded?
        keep_rate = self.compute_keep_rate(run_history)
        effective_changes = self.identify_effective_patterns(run_history)
        
        # Generate improved strategy
        new_strategy = self.generate_strategy_improvement(
            current_strategy, keep_rate, effective_changes
        )
        
        # The strategy itself goes through ratchet validation
        old_meta_score = self.evaluate_strategy(current_strategy, n=5)
        new_meta_score = self.evaluate_strategy(new_strategy, n=5)
        
        if statistically_significant(new_meta_score, old_meta_score):
            self.stage.add_agent_delta("meta_strategy", {
                "/agents/meta/strategy": new_strategy
            })
            return "KEEP"
        return "DISCARD"
    
    def classify_change(self, proposed_change):
        """Determines which tier a proposed change falls into."""
        if proposed_change.modifies_constitution():
            return "TIER_3_HUMAN_GATE"
        if proposed_change.modifies_role_definition():
            return "TIER_3_HUMAN_GATE"
        if proposed_change.modifies_scoring_function():
            return "TIER_3_HUMAN_GATE"  # THE critical anchor
        if proposed_change.modifies_prompt():
            if proposed_change.is_structural():  # new sections, role changes
                return "TIER_3_HUMAN_GATE"
            return "TIER_2_RATCHET"  # tuning, wording, emphasis
        return "TIER_1_AUTO"
```

**What transfers across runs (the Hyperagent accumulation pattern):**

The meta-agent's improvements persist as USD sublayers. When a new session starts, the meta-agent's strategy sublayer loads with it. Discoveries like "adding worked-examples to Scout's prompt improves recon report quality by 12%" don't need to be rediscovered. They compound.

```
Run 1: Meta-agent discovers "Scout benefits from worked examples" → KEEP
Run 2: Meta-agent discovers "Forge benefits from diff-format specs" → KEEP
Run 3: Meta-agent's strategy now includes "try worked examples first" 
        (learned from Run 1) → proposes worked examples for Vision → KEEP
Run 4: Meta-agent improves its own strategy: "prioritize prompt changes 
        that scored >15% improvement in previous runs" → KEEP

The system gets better at getting better. And the improvement history is a 
USD layer stack — fully inspectable, composable, and reversible.
```

---

## The Pitch

**One line:** An agentic generative ecosystem whose nervous system is thinking agents, whose skeleton is USD, whose brain is composition, and whose heartbeat is the ratchet — 1,000 experiments overnight, best recipe by morning.

**Three lines:** ComfyUI is the execution engine. USD is the universal interchange — workflows are prims, modifications are sublayers, creative exploration is variant selection, and generation outputs are composable 3D scenes. Agents are the intelligence — 6 thinking specialists with persistent memory, self-provisioning their own models, self-improving through geometric and aesthetic validation on a USD-composed stage. The ratchet runs the Karpathy Loop overnight: beat the baseline or get discarded, sublayer by sublayer, until the system converges on recipes no human would find by hand.

---

## Patent Relationship

This application directly extends all three patent filings:

| Patent | How It Extends |
|---|---|
| USD Cognitive Substrate | Workflow prims with real `pxr.Usd.Stage`, LIVRPS for parameter conflict, AIMemoryChunk for execution history, USD scene output as composable 3D assets, sublayer-based ratchet (keep/discard as composition operations) |
| Cognitive Twin | Derivative tool pattern (WorkflowGaffer, AgentInspector, ConvergenceMonitor), variant sets for creative profiles, customData state tracking, USD round-trip (scene in → generation → scene out), per-agent persistent memory as USD prims |
| Cognitive Bridge | Agent Bridge MCP server, cross-agent knowledge composition, entity extraction for model/technique understanding, ratchet results feeding cross-session patterns |
| Digital Injection | Creative profiles as gain-modulated parameter groups, variant-set switching, injection profiles as autoresearch exploration strategies |
| Lossless Signal | Anchor parameter immunity, delta sublayers (not replacements), integrity verification, `reconstruct_clean()` as base-layer read, ratchet discards as non-destructive sublayer removal |

**CIP potential — strong on four fronts:**
1. Application of cognitive substrate architecture to multi-agent creative workflow orchestration (novel use case, demonstrates breadth)
2. USD scene composition from generative AI outputs (depth + normals + segmentation → composable 3D scene) — novel prior art
3. USD sublayer composition as the ratchet mechanism for autonomous iterative optimization — structurally superior to git-based approaches
4. Hyperagent self-modification with USD-native safety guarantees — meta-agent modifications as reversible sublayers with constitutional anchors (scoring function, commandments) providing a safety floor that DGM-Hyperagents lacks

**Research lineage for the Hyperagent integration:**
- Darwin Gödel Machine (DGM) — Clune et al. — open-ended self-improvement in coding
- DGM-Hyperagents (Zhang et al., arXiv:2603.19461) — self-referential agents with editable meta-modification
- This ecosystem — Hyperagent pattern + USD composition guarantees + constitutional anchors + domain-specific (generative AI) application

---

*The nervous system is thinking agents. The skeleton is USD. The brain is composition. The heartbeat is the ratchet.*
*1,000 experiments overnight. Best recipe by morning. Self-improving. Self-provisioning. First principles.*
*v2.0.0 — THINKING AGENTS*
