<p align="center">
  <img src="assets/comfy-cozy-logo.jpg" alt="Comfy Cozy" width="600">
</p>

<p align="center">
  <strong>Patent Pending</strong> &nbsp;|&nbsp; <a href="LICENSE">MIT License</a> &nbsp;|&nbsp; <a href="https://github.com/JosephOIbrahim/Harlo/blob/main/PATENTS.md">Patent Details</a>
</p>

# Comfy Cozy

**Talk to ComfyUI like a colleague. It talks back.**

You describe what you want in plain English. The agent loads workflows, swaps models, tweaks parameters, installs missing nodes, runs generations, analyzes outputs, and learns what works for you -- all without you touching JSON or hunting through menus. It doesn't ask permission -- it makes the change, reports what it did, and every change is undoable.

```mermaid
graph LR
    You([You]) -->|"make it dreamier"| Agent[Comfy Cozy]
    Agent -->|loads, patches, runs| ComfyUI[ComfyUI]
    ComfyUI -->|image| Agent
    Agent -->|"Done. Lowered CFG to 5,<br/>switched to DPM++ 2M Karras.<br/>Here's your render."| You

    style You fill:#0066FF,color:#fff
    style Agent fill:#8b5cf6,color:#fff
    style ComfyUI fill:#ef4444,color:#fff
```

> **Session 1** is a capable tool.<br/>
> **Session 100** is a capable tool that knows your style.

---

## See It In Action

| You say | What happens |
|---------|-------------|
| *"Load my portrait workflow and make it dreamier"* | Loads the file, lowers CFG, switches sampler, saves with full undo |
| *"I want to use Flux"* | Searches CivitAI + HuggingFace, downloads the model, wires it into your workflow |
| *"Repair this workflow"* | Finds missing nodes, installs the packs, fixes connections, migrates deprecated nodes |
| *"Run this with 30 steps"* | Patches the workflow, validates it, queues it to ComfyUI, shows progress |
| *"Analyze this output"* | Uses Vision AI to diagnose issues and suggest parameter changes |
| *"What model should I use for anime?"* | Searches CivitAI + HuggingFace + your local models, recommends the best fit |
| *"Optimize this for speed"* | Profiles GPU usage, checks TensorRT eligibility, applies optimizations |
| *"Repair and run this"* | Finds missing nodes, installs them, validates, executes -- no confirmation needed |

---

## Get Running

**You need three things. That's it.**

| # | What | Where to get it |
|---|------|-----------------|
| 1 | **Python 3.11+** | [python.org/downloads](https://python.org/downloads) |
| 2 | **ComfyUI running** | [github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) |
| 3 | **One LLM backend** | API key (Anthropic / OpenAI / Google) OR [Ollama](https://ollama.com) (free, local) |

Got all three? Four steps:

### Step 1 of 4 -- Clone

```bash
git clone https://github.com/JosephOIbrahim/Comfy-Cozy.git
cd Comfy-Cozy
```

### Step 2 of 4 -- Install

```bash
pip install -e .
```

Done. That's the only install command you need.

<details>
<summary>Optional installs (click to expand)</summary>

```bash
pip install -e ".[dev]"           # + test suite (3579 passing tests)
pip install -e ".[dev,stage]"     # + USD stage subsystem (~200MB, most users skip this)
```

</details>

### Step 3 of 4 -- API key

```bash
cp .env.example .env
```

Open `.env`, paste your key:

```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

<details>
<summary>Using a different LLM? (click to expand)</summary>

```bash
# OpenAI (requires: pip install openai)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here

# Gemini (requires: pip install google-genai)
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here

# Ollama (no API key needed)
LLM_PROVIDER=ollama
AGENT_MODEL=llama3.1
```

</details>

**Non-default ComfyUI location?** Add this line too:

```
COMFYUI_DATABASE=C:/path/to/your/ComfyUI
```

### Step 4 of 4 -- Go

```bash
agent run
```

Type what you want. Type `quit` when you're done.

---

## Connect the Sidebar to ComfyUI

The agent also lives **inside ComfyUI** as a native sidebar panel. To enable it, create two symlinks from ComfyUI's `custom_nodes/` folder to Comfy-Cozy:

**Windows (run as Administrator):**

```cmd
cd C:\path\to\ComfyUI\custom_nodes
mklink /D comfy-cozy-panel C:\path\to\Comfy-Cozy\panel
mklink /D comfy-cozy-ui C:\path\to\Comfy-Cozy\ui
```

**Linux / macOS:**

```bash
cd /path/to/ComfyUI/custom_nodes
ln -s /path/to/Comfy-Cozy/panel comfy-cozy-panel
ln -s /path/to/Comfy-Cozy/ui comfy-cozy-ui
```

Restart ComfyUI. The Comfy Cozy chat panel appears in the **left sidebar**.

```mermaid
graph LR
    CN["ComfyUI/custom_nodes/"] --> P["comfy-cozy-panel/ (symlink)"]
    CN --> U["comfy-cozy-ui/ (symlink)"]
    P -->|"canvas sync (headless)"| Panel["panel/__init__.py"]
    U -->|"sidebar + chat"| UI["ui/__init__.py"]

    style CN fill:#ef4444,color:#fff
    style P fill:#8b5cf6,color:#fff
    style U fill:#0066FF,color:#fff
```

**Both symlinks are required:**
- **`comfy-cozy-panel`** -- Canvas sync bridge (runs headlessly -- keeps the agent in sync with your live graph)
- **`comfy-cozy-ui`** -- The visible sidebar: chat window, quick actions, status

---

## Pick Your LLM

Comfy Cozy is **provider-agnostic**. Same 113 tools, same streaming, same vision analysis -- swap one env var.

### Anthropic (default)

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Run
agent run
```

Ships as the default. No extra install. Supports prompt caching for lower costs on long sessions.

### OpenAI

```bash
# Install the SDK (one time)
pip install openai

# .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
AGENT_MODEL=gpt-4o           # or gpt-4o-mini for faster/cheaper

# Run
agent run
```

Full tool-use support with streaming. Works with any OpenAI-compatible endpoint.

### Google Gemini

```bash
# Install the SDK (one time)
pip install google-genai

# .env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
AGENT_MODEL=gemini-2.5-flash  # or gemini-2.5-pro

# Run
agent run
```

Function declarations mapped automatically. Supports Gemini's thinking mode.

### Ollama (fully local, no API key)

```bash
# Install Ollama: https://ollama.com
# Pull a model
ollama pull llama3.1

# .env
LLM_PROVIDER=ollama
AGENT_MODEL=llama3.1          # or any model you've pulled

# Run (no API key needed)
agent run
```

Uses Ollama's OpenAI-compatible endpoint at `localhost:11434`. Override with `OLLAMA_BASE_URL` if running remotely. **No data leaves your machine.**

### Architecture

All four providers share the same abstraction layer (`agent/llm/`):

```mermaid
graph LR
    Agent[Agent Loop<br/>113 tools] --> LLM{LLM_PROVIDER}
    LLM -->|anthropic| A["Claude<br/>Streaming + Cache"]
    LLM -->|openai| B["GPT-4o<br/>Tool Calls"]
    LLM -->|gemini| C["Gemini<br/>Function Decl."]
    LLM -->|ollama| D["Ollama<br/>Local + Private"]

    style Agent fill:#8b5cf6,color:#fff
    style A fill:#d97706,color:#fff
    style B fill:#10b981,color:#fff
    style C fill:#3b82f6,color:#fff
    style D fill:#ef4444,color:#fff
```

Common types (`TextBlock`, `ToolUseBlock`, `LLMResponse`), unified error hierarchy, provider-specific format conversion handled internally. Switch providers with one env var -- no code changes.

---

## Three Ways to Use It

### A. Inside Claude Code / Claude Desktop (recommended)

The agent runs as an MCP server -- Claude can use all 113 tools directly.

Add this to your Claude Code or Claude Desktop MCP config:

```json
{
  "mcpServers": {
    "comfyui-agent": {
      "command": "agent",
      "args": ["mcp"]
    }
  }
}
```

Now talk to Claude about your ComfyUI workflows. It has full access.

### B. Standalone CLI

```bash
agent run                        # Start a conversation
agent run --session my-project   # Auto-saves so you can pick up later
agent run --verbose              # See what's happening under the hood
```

### C. One-click launcher (ComfyUI + agent together)

If you use the **ComfyUI CLI launcher** (`ComfyUI CLI.lnk`), Comfy Cozy is the default mode:

```
[ 1 ]  STABLE          Balanced. Works with everything.
[ 2 ]  DETERMINISTIC   Same prompt = same pixels.
[ 3 ]  FAST            Sage attention + async offload.
[ 4 ]  COMFY COZY  *   Talk to your workflow. (auto-selects in 10s)
```

Select **4** (or wait 10 seconds) -- ComfyUI starts in a background window, then the agent launches ready to talk.

### Handy CLI Commands (no API key needed)

```bash
agent inspect                    # See your installed models and nodes
agent parse workflow.json        # Analyze a workflow file
agent sessions                   # List your saved sessions
```

---

## What the Agent Knows About Your Models

The agent ships with built-in knowledge about how each model family actually behaves. It won't use SD 1.5 settings on a Flux workflow.

| Model | Resolution | CFG | Notes |
|-------|-----------|-----|-------|
| **SD 1.5** | 512x512 | 7-12 | Huge LoRA ecosystem. Negative prompts matter. |
| **SDXL** | 1024x1024 | 5-9 | Better anatomy. Tag-based prompts work best. |
| **Flux** | 512-1024 | ~1.0 (guidance) | No negative prompts. Needs FluxGuidance node + T5 encoder. |
| **SD3** | 1024x1024 | 5-7 | Triple text encoder (CLIP-G, CLIP-L, T5). |
| **LTX-2** (video) | 768x512 | ~25 | 121 steps. Frame count must be (N*8)+1. |
| **WAN 2.x** (video) | 832x480 | 1-3.5 | Dual-noise architecture. 4-20 steps. |

**The agent will never mix model families** -- no SD 1.5 LoRAs on SDXL checkpoints, no Flux ControlNets on SD3.

### Artist-Speak Translation

| You say | What the agent adjusts |
|---------|----------------------|
| *"dreamier"* or *"softer"* | Lower CFG (5-7), more steps, DPM++ 2M Karras |
| *"sharper"* or *"crisper"* | Higher CFG (8-12), Euler or DPM++ SDE |
| *"more photorealistic"* | CFG 7-10, realistic checkpoint, negative: "cartoon, anime" |
| *"more stylized"* | Lower CFG (4-6), artistic checkpoint or LoRA |
| *"faster"* | Fewer steps (15-20), LCM/Lightning/Turbo, smaller resolution |
| *"higher quality"* | More steps (30-50), hires fix, upscaler |
| *"more variation"* | Higher denoise, different seed, lower CFG |
| *"less variation"* | Lower denoise, same seed, higher CFG |

---

## How It Works

```mermaid
graph TB
    subgraph Browser ["ComfyUI Browser"]
        Sidebar["Comfy Cozy Sidebar<br/>Native left panel -- Chat -- Quick Actions"]
    end
    subgraph Backend ["Agent Backend (Python)"]
        Routes["49 REST Routes<br/>+ WebSocket"]
        Tools["113 Tools<br/>workflow -- models -- vision -- session -- provision"]
        Cog["Cognitive Engine<br/>LIVRPS delta stack -- CWM -- experience"]
    end
    subgraph ComfyUI ["ComfyUI"]
        API["/prompt -- /history -- /ws"]
        Canvas["Live Canvas"]
    end
    subgraph Disk ["Persistence"]
        EXP[("experience.jsonl<br/>cross-session learning")]
        Sessions[("sessions/<br/>workflow state")]
    end

    Sidebar <-->|"WebSocket + REST"| Routes
    Sidebar <-->|"canvas sync"| Canvas
    Routes --> Tools
    Tools --> Cog
    Tools -->|httpx| API
    Cog --> EXP
    Tools --> Sessions

    style Browser fill:#1a1a2e,color:#F0F0F0,stroke:#0066FF
    style Backend fill:#1a1a2e,color:#F0F0F0,stroke:#8b5cf6
    style ComfyUI fill:#1a1a2e,color:#F0F0F0,stroke:#ef4444
    style Disk fill:#1a1a2e,color:#F0F0F0,stroke:#10b981
```

```mermaid
graph LR
    You([You]) --> Agent[113 Tools]
    Agent --> Understand[UNDERSTAND<br/>What do you have?]
    Understand --> Discover[DISCOVER<br/>What do you need?]
    Discover --> Pilot[PILOT<br/>Make the changes]
    Pilot --> Verify[VERIFY<br/>Did it work?]
    Verify -->|learn| Agent

    style You fill:#0066FF,color:#fff
    style Understand fill:#3b82f6,color:#fff
    style Discover fill:#d97706,color:#fff
    style Pilot fill:#8b5cf6,color:#fff
    style Verify fill:#10b981,color:#fff
```

**Four phases, always in order:**

1. **UNDERSTAND** -- Reads your workflow, scans your models, checks what's installed
2. **DISCOVER** -- Searches CivitAI, HuggingFace, ComfyUI Manager (31k+ nodes)
3. **PILOT** -- Makes changes through safe, reversible delta layers (never edits your original)
4. **VERIFY** -- Runs the workflow, checks the output, records what worked

Every change is undoable. Every generation teaches the agent something. The agent is a doer, not a describer -- say "wire the model" and it wires the model. Say "repair this" and it finds the missing nodes, installs them, and validates. Say "run it" and it validates then executes. No confirmation dialogs, no "would you like me to..." -- it acts, then tells you what it did.

---

## Autonomous Mode

Write a creative intent. Hit go. No workflow file needed, no parameters to tune -- the agent composes a workflow, runs it on ComfyUI, scores the result, and learns from it automatically.

```mermaid
flowchart TD
    You(["Creative Intent<br/>'cinematic portrait, golden hour'"]) --> INTENT["INTENT<br/>Parse + validate"]
    INTENT --> COMPOSE["COMPOSE<br/>Load template<br/>Blend with experience"]
    COMPOSE --> PREDICT["PREDICT<br/>CognitiveWorldModel<br/>estimates quality"]
    PREDICT --> GATE{"GATE<br/>Arbiter:<br/>proceed?"}
    GATE -->|Yes| EXECUTE["EXECUTE<br/>Post to ComfyUI<br/>Monitor WebSocket"]
    GATE -->|Interrupt| STOP(["Interrupted<br/>+ reason"])
    EXECUTE --> EVALUATE["EVALUATE<br/>Score the output"]
    EVALUATE --> LEARN["LEARN<br/>Record to accumulator<br/>Calibrate CWM"]
    LEARN --> DONE(["Complete<br/>Experience recorded"])
    EVALUATE -->|"score < threshold"| COMPOSE

    style You fill:#0066FF,color:#fff
    style GATE fill:#d97706,color:#fff
    style EXECUTE fill:#ef4444,color:#fff
    style LEARN fill:#8b5cf6,color:#fff
    style DONE fill:#10b981,color:#fff
    style STOP fill:#6b7280,color:#fff
```

**Use from Python:**

```python
from cognitive.pipeline import create_default_pipeline, PipelineConfig

pipeline = create_default_pipeline()   # fresh accumulator, CWM, arbiter
result = pipeline.run(PipelineConfig(
    intent="cinematic portrait, golden hour",
    model_family="SD1.5",              # optional -- agent detects from intent
))
print(result.success, result.quality.overall, result.stage.value)
if result.warnings:
    print("warnings:", result.warnings)  # e.g. template family fallback
```

- **No executor required.** The pipeline calls ComfyUI directly via the real `execute_workflow` implementation.
- **No evaluator required.** Rule-based scoring (success = 0.7, failure = 0.1) enables CWM calibration from day one.
- **Template library.** Workflows loaded from `agent/templates/` (SD 1.5 / SDXL / img2img / LoRA). Hardcoded 7-node SD 1.5 fallback if no template matches.
- **Experience persists across sessions -- crash-safe.** Every run saved atomically (write-to-tmp then `os.replace()`). After 30+ runs, the agent starts using your personal history to bias parameter selection.
- **Pipeline failures are graceful.** CWM exceptions return `PipelineStage.FAILED` cleanly. Template mismatches populate `result.warnings`.

```mermaid
graph LR
    subgraph Session1 ["Session 1"]
        I1["Intent"] --> C1["Compose"] --> E1["Execute"] --> S1["Score"]
    end
    subgraph Session2 ["Session 2+"]
        I2["Intent"] --> C2["Compose<br/>(+prior runs)"] --> E2["Execute"] --> S2["Score"]
    end
    S1 -->|"atomic save"| JSONL[("experience.jsonl<br/>crash-safe")]
    JSONL -->|"load on startup"| C2
    S2 -->|"atomic save -- cumulative"| JSONL

    style JSONL fill:#8b5cf6,color:#fff
    style C2 fill:#10b981,color:#fff
```

---

## Comfy Cozy Sidebar (Native ComfyUI Integration)

A typography-forward chat panel in ComfyUI's native left sidebar. No floating buttons, no separate windows. Uses ComfyUI's own CSS variables -- adapts to any theme automatically.

```mermaid
graph TB
    subgraph ComfyUI_App ["ComfyUI"]
        subgraph Sidebar ["Left Sidebar"]
            Tab["Comfy Cozy Tab<br/>registerSidebarTab()"]
            Chat["Chat Window<br/>WebSocket -- streaming -- rich text"]
            QA["Quick Actions<br/>Run -- Validate -- Repair -- Optimize -- Undo"]
        end
        Canvas["Canvas"]
    end

    subgraph Bridge ["Bidirectional Canvas Bridge"]
        C2A["Canvas --> Agent<br/>Auto-sync on change"]
        A2C["Agent --> Canvas<br/>Push mutations + highlights"]
    end

    Tab --> Chat
    Tab --> QA
    Sidebar <--> Bridge
    Bridge <--> Canvas

    style ComfyUI_App fill:#1a1a2e,color:#F0F0F0,stroke:#ef4444
    style Sidebar fill:#1a1a2e,color:#F0F0F0,stroke:#0066FF
    style Bridge fill:#1a1a2e,color:#F0F0F0,stroke:#8b5cf6
```

**What you get:**
- **Native sidebar tab** -- `app.extensionManager.registerSidebarTab()`, sits alongside ComfyUI's built-in panels
- **Design system v3** -- Inter + JetBrains Mono, ComfyUI CSS variables, Pentagram-inspired: hairline borders, generous whitespace, 2px radii, zero ornamentation
- **Chat** -- Auto-growing textarea, streaming responses, rich text (code blocks, bold, inline code), collapsible tool results
- **Node pills** -- Clickable inline node references, color-coded by slot type. Click = select + center on canvas.
- **Quick actions** -- Context-aware chips: Run, Validate, Repair, Optimize, Undo
- **Canvas bridge** -- Agent changes sync to canvas live with node highlighting; canvas re-syncs after each execution
- **Self-healing** -- Missing node warnings with one-click repair, deprecated node migration

**49 panel routes** expose the full tool surface: discovery, provisioning, repair, sessions, execution.

Every request passes through a three-layer security chain:

```mermaid
flowchart TD
    REST([REST Request]) --> Guard["_guard(request, category)"]
    WS([WebSocket /ws]) --> Guard
    Guard --> Auth{check_auth}
    Auth -->|"no token configured"| Rate{check_rate_limit}
    Auth -->|"bearer matches"| Rate
    Auth -->|"missing / wrong"| R401(["401 Unauthorized"])
    Rate -->|"tokens available"| Size{check_size}
    Rate -->|"bucket empty"| R429(["429 -- Retry-After: 1s"])
    Size -->|"Content-Length OK"| Handler(["Route handler"])
    Size -->|"> 10 MB"| R413(["413 Too Large"])
    Size -->|"chunked -- no length"| R411(["411 Length Required"])

    style R401 fill:#ef4444,color:#fff
    style R429 fill:#d97706,color:#fff
    style R413 fill:#ef4444,color:#fff
    style R411 fill:#d97706,color:#fff
    style Handler fill:#10b981,color:#fff
    style Guard fill:#8b5cf6,color:#fff
```

---

## One-Click Model Provisioning

The agent handles the entire pipeline from "I want Flux" to a wired workflow:

```mermaid
flowchart LR
    Search["Search<br/>CivitAI + HF + Registry"] --> Download["Download<br/>to correct folder"]
    Download --> Verify["Verify<br/>family + compat"]
    Verify --> Wire["Auto-Wire<br/>find loader -- set input"]
    Wire --> Ready["Ready to<br/>Queue"]

    style Search fill:#3b82f6,color:#fff
    style Download fill:#d97706,color:#fff
    style Verify fill:#ef4444,color:#fff
    style Wire fill:#8b5cf6,color:#fff
    style Ready fill:#10b981,color:#fff
```

**`provision_model`** -- one tool call that discovers, downloads, verifies compatibility, finds the right loader node in your workflow, and wires the model in.

---

<details>
<summary><b>Architecture Deep Dive</b> (click to expand)</summary>

### Seven Structural Subsystems

The agent is built on seven architectural subsystems. Each one degrades independently -- if one breaks, the rest keep working.

```mermaid
graph TB
    subgraph Foundation ["Foundation Layer"]
        DAG["Workflow Intelligence DAG<br/>6 pure computation nodes"]
        OBS["Time-Sampled State<br/>Monotonic step index"]
        CAP["Capability Registry<br/>113 tools indexed"]
    end

    subgraph Safety ["Safety Layer"]
        GATE["Pre-Dispatch Gate<br/>5 checks, default-deny"]
        BRIDGE["Mutation Bridge<br/>LIVRPS composition + audit"]
    end

    subgraph Integration ["Integration Layer"]
        ADAPT["Inter-Module Adapters<br/>Pure-function translators"]
        DEGRADE["Degradation Manager<br/>Per-subsystem fallbacks"]
    end

    Foundation --> Safety --> Integration

    style Foundation fill:#1a1a2e,color:#F0F0F0,stroke:#3b82f6
    style Safety fill:#1a1a2e,color:#F0F0F0,stroke:#ef4444
    style Integration fill:#1a1a2e,color:#F0F0F0,stroke:#10b981
```

### Workflow Intelligence DAG

Before any workflow runs, a DAG of pure functions analyzes it:

```mermaid
graph LR
    C[Complexity<br/>TRIVIAL to EXTREME] --> M[Model Requirements<br/>VRAM, family, LoRAs]
    M --> O[Optimization<br/>TensorRT, batching]
    O --> R[Risk<br/>SAFE to BLOCKED]
    R --> RD[Readiness<br/>go / no-go]

    style C fill:#3b82f6,color:#fff
    style R fill:#ef4444,color:#fff
    style RD fill:#10b981,color:#fff
```

### Pre-Dispatch Safety Gate

Every tool call passes through a default-deny gate. Read-only tools bypass it (zero overhead). Destructive tools are always locked. The gate auto-detects loaded workflows -- if a workflow is in memory (from the sidebar or CLI), execution tools are allowed without explicit session context.

```mermaid
flowchart LR
    Tool([Tool Call]) --> Session{"Workflow\nloaded?"}
    Session -->|Yes| Risk{Risk Level?}
    Session -->|No| Deny["Denied:\nno active session"]
    Risk -->|"Read-only"| Pass[Pass through]
    Risk -->|"Mutation / Execute"| Checks[5 safety checks]
    Risk -->|"Install / Download"| Escalate[Escalate to LLM]
    Risk -->|"Uninstall / Delete"| Block[Blocked]

    Checks --> OK{All pass?}
    OK -->|Yes| Go[Execute]
    OK -->|No| Stop[Denied + reason]

    style Pass fill:#10b981,color:#fff
    style Go fill:#10b981,color:#fff
    style Stop fill:#ef4444,color:#fff
    style Block fill:#ef4444,color:#fff
    style Deny fill:#ef4444,color:#fff
    style Escalate fill:#FF9900,color:#000
```

### LIVRPS -- How Conflicts Get Resolved

All workflow changes are non-destructive layers. When two opinions conflict:

| Priority | Layer | Example |
|----------|-------|---------|
| 6 (strongest) | **Safety** | "CFG above 30 is degenerate" -- always wins |
| 5 | **Local** (your edit) | "Set CFG to 9" |
| 4 | **Inherits** (experience) | "CFG 7.5 worked better last time" |
| 3 | **VariantSets** | Creative profile presets |
| 2 | **References** | Learned recipes |
| 1 (weakest) | **Payloads** | Default template values |

Your edit beats experience. Safety beats everything. Every conflict is deterministic, transparent, and reversible.

This is an intentional inversion of USD's native LIVRPS (where Specializes is weakest). Safety is promoted to strongest for safety-critical override -- the architectural decision documented in the patent application.

### Cognitive State Engine (Phase 0.5 -- live in production)

LIVRPS is no longer a table on a slide. Since Phase 0.5 the engine is a real top-level package (`cognitive/`) installed alongside `agent/`, and `agent/tools/workflow_patch.py` imports it directly at module load -- no try/except, no silent fallback. Every PILOT mutation is recorded as a delta layer with SHA-256 tamper detection, then resolved on demand.

```mermaid
graph LR
    User([Tool Call<br/>via MCP]) --> WP["agent/tools/<br/>workflow_patch.py"]
    WP -->|"from cognitive.core.graph<br/>import CognitiveGraphEngine"| CGE["CognitiveGraphEngine<br/>(session-scoped)"]
    CGE --> Stack["Delta Stack<br/>P -- R -- V -- I -- L -- S"]
    Stack -->|"sort weakest to strongest<br/>apply, preserve link arrays"| Resolved["Resolved WorkflowGraph"]
    Resolved -->|"to_api_json()"| Comfy["ComfyUI /prompt"]

    style User fill:#0066FF,color:#fff
    style WP fill:#3b82f6,color:#fff
    style CGE fill:#8b5cf6,color:#fff
    style Stack fill:#d97706,color:#fff
    style Resolved fill:#10b981,color:#fff
    style Comfy fill:#ef4444,color:#fff
```

The `cognitive/` package is layered by phase -- the core engine (Phase 1) is fully tested at 54/54 adversarial cases. Phase 6 is complete: the autonomous pipeline is fully wired with real executor, template loading, rule-based evaluator, and experience persistence.

```mermaid
graph TB
    Cognitive["cognitive/<br/>(installed top-level package)"]
    Cognitive --> Core["core/<br/>graph -- delta -- models<br/>54/54 tests passing"]
    Cognitive --> Exp["experience/<br/>chunk -- signature -- accumulator"]
    Cognitive --> Pred["prediction/<br/>cwm -- arbiter -- counterfactual"]
    Cognitive --> Trans["transport/<br/>schema_cache -- events -- interrupt"]
    Cognitive --> Pipe["pipeline/<br/>autonomous -- create_default_pipeline<br/>Phase 6 complete"]
    Cognitive --> CogTools["tools/<br/>analyze -- compose -- execute<br/>mutate -- query -- series -- dependencies"]

    style Cognitive fill:#8b5cf6,color:#fff
    style Core fill:#0066FF,color:#fff
    style Exp fill:#3b82f6,color:#fff
    style Pred fill:#d97706,color:#fff
    style Trans fill:#10b981,color:#fff
    style Pipe fill:#10b981,color:#fff
    style CogTools fill:#3b82f6,color:#fff
```

Each delta layer carries its `creation_hash` (SHA-256 of `opinion + sorted-JSON mutations`). `verify_stack_integrity()` walks the stack and flags any layer whose `layer_hash` no longer matches its `creation_hash` -- making post-hoc tampering detectable. Link arrays (`["node_id", output_index]`) are preserved through every parse/mutate/serialize round-trip, which is the #1 failure mode in ComfyUI agents.

### Graceful Degradation

Every subsystem has an independent kill switch. Set any of these to `0` in your `.env` to disable:

`BRAIN_ENABLED` `DAG_ENABLED` `GATE_ENABLED` `OBSERVATION_ENABLED`

All default to ON. The agent works fine with any combination disabled -- features gracefully disappear.

### Experience Loop

Every generation is an experiment. The agent tracks what worked:

- **Sessions 1-30**: Uses built-in knowledge only
- **Sessions 30-100**: Blends knowledge with what it's learned from your renders
- **Sessions 100+**: Primarily driven by your personal history

### Tool Inventory

**113 tools across three layers:**

| Layer | Count | Highlights |
|-------|-------|-----------|
| **Intelligence** | 86 | Workflow parsing, model search (CivitAI + HF + 31k nodes), delta patching, auto-wire, provisioning pipeline, execution |
| **Brain** | 27 | Vision analysis, goal planning, pattern memory, GPU optimization, artistic intent capture, iteration tracking |
| **Stage** | 23 | USD cognitive state, LIVRPS composition, predictive experiments, scene composition |

### Workflow Lifecycle

```mermaid
flowchart LR
    Load[Load] --> Validate[Validate]
    Validate --> Analyze[DAG<br/>Analysis]
    Analyze --> Gate[Safety<br/>Gate]
    Gate --> Patch[Patch via<br/>Delta Layer]
    Patch --> Run[Run on<br/>ComfyUI]
    Run --> Check[Check<br/>Output]
    Check --> Learn[Record<br/>Experience]

    Patch -->|Undo| Validate
    Check -->|Iterate| Patch

    style Load fill:#3b82f6,color:#fff
    style Analyze fill:#d97706,color:#fff
    style Gate fill:#ef4444,color:#fff
    style Run fill:#ef4444,color:#fff
    style Check fill:#10b981,color:#fff
    style Learn fill:#8b5cf6,color:#fff
```

### Project Structure

```
agent/
  llm/                Multi-provider abstraction (Anthropic, OpenAI, Gemini, Ollama)
  tools/              63 tools -- workflow ops, model search, provisioning, auto-wire
                      workflow_patch.py wraps the cognitive engine for non-destructive PILOT
  brain/              27 tools -- vision, planning, memory, optimization
    adapters/         Pure-function translators between brain modules
  stage/              23 tools -- USD state, prediction, composition (USD optional via [stage])
    dag/              Workflow intelligence (6 computation nodes)
  gate/               Pre-dispatch safety (5-check pipeline)
  degradation.py      Fault isolation manager
  config.py           Environment + 4 kill switches + LLM provider selection
  mcp_server.py       MCP server (primary interface)
cognitive/            LIVRPS state engine -- installed as top-level package (Phase 0.5)
  core/               CognitiveGraphEngine, DeltaLayer, WorkflowGraph (link-preserving)
  experience/         ExperienceChunk, GenerationContextSignature, Accumulator
  prediction/         CognitiveWorldModel, SimulationArbiter, CounterfactualGenerator
  transport/          SchemaCache, ExecutionEvent, interrupt, system_stats
  pipeline/           Autonomous end-to-end orchestration
  tools/              Phase 3 macro-tools (analyze, mutate, query, compose, ...)
ui/
  __init__.py         WEB_DIRECTORY + route registration
  web/js/sidebar.js   Native left sidebar -- chat, quick actions, progress
  web/css/            Design system v3 -- ComfyUI-native CSS variables
  server/routes.py    WebSocket + REST endpoints for sidebar chat
panel/
  __init__.py         WEB_DIRECTORY + route registration + sys.path injection
  server/routes.py    49 REST routes -- full tool surface
  web/js/             Canvas sync bridge (headless -- no visible UI)
tests/                3579 passing tests, all mocked, ~60s
```

### Production Hardening

| Domain | What it means |
|--------|-------------|
| **Safety** | 5-check default-deny gate. Risk levels 0-4. Destructive ops never auto-execute. |
| **Fault Isolation** | Each subsystem fails independently. Circuit breakers prevent cascading failures. `brain` (threshold=3, timeout=30s) and `comfyui_http` (threshold=5, timeout=60s) registered; `BRAIN_ENABLED=0` kill switch fully enforced in tool registry. Session isolation: each `agent mcp` process gets a unique `conn_XXXXXXXX` namespace; ContextVar set in executor thread before dispatch. Parallel tool dispatch routes through `agent.tools.handle` live module reference -- monkey-patch visible to all ThreadPoolExecutor workers. |
| **Determinism** | Pure computation DAG. Deterministic JSON. Ordinal state enums. Same input = same output. |
| **Audit Trail** | Every mutation logged: who changed what, when, and what got overridden. |
| **Security** | Bearer token auth on all routes including WebSocket. Path traversal blocked. SSRF prevented on initial URL and every redirect hop (RFC 1918 + loopback + link-local + CGNAT rejected via DNS resolution). MCP tool errors return `isError=True` per protocol. Gate exceptions deny by default (no silent allow). 10 MB + chunked-transfer size guards. Max 20 concurrent WebSocket connections. Atomic file writes (write-to-tmp-then-`os.replace()`). Thread-safe token bucket rate limiter. |
| **Bounded Resources** | Intent history (100), iteration steps (200), demo checkpoints (100). No unbounded growth. |

```mermaid
graph TB
    subgraph Sec ["Security"]
        A1["Auth -- Bearer token<br/>(REST + WebSocket)"]
        A2["Rate limit -- Token bucket<br/>per category"]
        A3["Size guard -- 10 MB limit<br/>+ chunked-transfer block"]
        A4["WS cap -- 20 connections max"]
    end
    subgraph Atom ["Persistence"]
        B1["_save_lock<br/>threading.Lock"]
        B2["Write to .jsonl.tmp"]
        B3["os.replace()<br/>atomic swap"]
        B1 --> B2 --> B3
    end
    subgraph Resil ["Resilience"]
        C1["CWM exception<br/>--> PipelineStage.FAILED"]
        C2["Template mismatch<br/>--> result.warnings"]
        C3["Save failure<br/>--> non-fatal log"]
    end

    style Sec fill:#1a1a2e,color:#F0F0F0,stroke:#ef4444
    style Atom fill:#1a1a2e,color:#F0F0F0,stroke:#8b5cf6
    style Resil fill:#1a1a2e,color:#F0F0F0,stroke:#10b981
```

</details>

---

## Configuration

All settings live in your `.env` file:

| Setting | Default | What it does |
|---------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required for Anthropic)* | Your Claude API key |
| `OPENAI_API_KEY` | | Your OpenAI API key |
| `GEMINI_API_KEY` | | Your Google Gemini API key |
| `LLM_PROVIDER` | `anthropic` | Which LLM to use: `anthropic`, `openai`, `gemini`, `ollama` |
| `AGENT_MODEL` | *(auto per provider)* | Override the model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama server URL |
| `COMFYUI_HOST` | `127.0.0.1` | Where ComfyUI runs |
| `COMFYUI_PORT` | `8188` | ComfyUI port |
| `COMFYUI_DATABASE` | `~/ComfyUI` | Your ComfyUI folder (models, nodes, workflows) |

---

## Testing

No ComfyUI needed -- everything is mocked:

```bash
python -m pytest tests/ -v        # 3579 passing tests, ~60s

# Skip tests that require a real ComfyUI server or API keys
python -m pytest tests/ -v -m "not integration"
```

The `[dev]` install runs the full test suite -- no ComfyUI server or API keys required, everything is mocked. The `test_provisioner.py` tests require `usd-core` (install with `pip install -e ".[stage]"` to resolve them).

---

## License & Patents

**Patent Pending** | [MIT](LICENSE)

Aspects of this architecture -- including deterministic state-evolution, LIVRPS non-destructive composition, predictive experiment pipelines, and the cognitive experience loop -- are the subject of pending US provisional patent applications filed by Joseph O. Ibrahim.

This project shares structural patterns with [Harlo](https://github.com/JosephOIbrahim/Harlo), a USD-native cognitive architecture for persistent AI memory. See Harlo's [PATENTS.md](https://github.com/JosephOIbrahim/Harlo/blob/main/PATENTS.md) for patent details and grant terms.

For questions about patent licensing, commercial licensing, or enterprise arrangements:
**Joseph O. Ibrahim** -- jomar.ibrahim@gmail.com
