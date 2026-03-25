# ComfyUI Agent — Phase 5B Sprint
# Sprint: Demo Polish & Metadata Wiring
# Date: 2026-03-09
# Status: READY TO EXECUTE

---

## PRE-FLIGHT: Read Before Anything

Before ANY implementation, read these files and internalize their conventions:

```bash
cat C:/Users/User/comfyui-agent/CLAUDE.md
cat C:/Users/User/comfyui-agent/agent/tools/verify_execution.py
cat C:/Users/User/comfyui-agent/agent/tools/image_metadata.py
cat C:/Users/User/comfyui-agent/agent/tools/workflow_parse.py
cat C:/Users/User/comfyui-agent/agent/brain/demo.py
cat C:/Users/User/comfyui-agent/agent/brain/intent_collector.py
cat C:/Users/User/comfyui-agent/agent/brain/iteration_accumulator.py
cat C:/Users/User/comfyui-agent/agent/system_prompt.py
cat C:/Users/User/comfyui-agent/agent/cli.py
cat C:/Users/User/comfyui-agent/agent/tools/session_tools.py
python -m pytest tests/ -q --tb=short 2>&1 | tail -5
```

Report what you find. Do NOT proceed until you understand:
- The TOOLS + handle() registration pattern used in every module
- The BrainAgent SDK auto-registration pattern in `_sdk.py`
- The `to_json()` deterministic serialization convention (sort_keys=True)
- How `_verify_prompt()` currently records outcomes but doesn't embed metadata
- How `build_system_prompt()` injects session context but doesn't read image metadata
- How `_extract_api_format()` parses workflows into node dicts
- The existing test patterns (all mocked, no live ComfyUI)

---

## STATUS REPORTING PROTOCOL

**MANDATORY:** After completing each task, print the status bar in this EXACT format.
This is the user's ONLY visibility into progress. Never skip it.

```
╔══════════════════════════════════════════════════════════════╗
║  ComfyUI Agent — Phase 5B STATUS                            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Phase 1: BUILD             [░░░░░░░░░░]  0%                ║
║    WIRE    ◆ W1 ○  W2 ○  W3 ○                               ║
║    CLASS   ⟡ C1 ○  C2 ○  C3 ○                               ║
║                                                              ║
║  Phase 2: VALIDATE + TEST   [░░░░░░░░░░]  0%                ║
║    DEMO    ◈ D1 ○  D2 ○                                     ║
║    TEST    ◇ T1 ○  T2 ○  T3 ○  T4 ○                        ║
║                                                              ║
║  Overall: [░░░░░░░░░░░░░░░░░░░░]  0%  (0/13 tasks)         ║
║  Legend: ✓ done  ▶ active  ○ pending  ✗ failed              ║
╚══════════════════════════════════════════════════════════════╝
```

**Rules:**
- Print after EVERY completed task (not just phases)
- Update symbols: ○ → ▶ (when starting) → ✓ (when done) or ✗ (on failure)
- Calculate percentages: done_tasks / total_tasks × 100
- If a task FAILS (✗), report the error message before the status bar

---

## ARCHITECTURE DECISIONS (NON-NEGOTIABLE)

### 1. Metadata flows through existing tool dispatch
The metadata embed MUST go through the existing `from . import handle as dispatch_tool` pattern in verify_execution.py. Do NOT import image_metadata directly — use the tool dispatch system so it's mockable and follows the established pattern.

### 2. Workflow classification lives in workflow_parse.py
Pattern classification is an extension of the existing `_build_summary()` function. It goes in workflow_parse.py as a new internal function `_classify_pattern()` and a new tool `classify_workflow`. Do NOT create a new module.

### 3. Session resume reads metadata via reconstruct_context tool
The metadata auto-read on session resume calls `reconstruct_context` through the existing tool dispatch — same pattern as memory recommendations in system_prompt.py.

### 4. All new functions must be deterministic
Sort keys in dicts, use sorted() for iteration, match He2025 determinism pattern visible throughout the codebase.

---

## FILE OWNERSHIP TABLE

**Every file has exactly ONE owner. Violation = merge conflicts = sprint failure.**

| Agent | Role (MOE) | Exclusive Write | Read Only |
|-------|------------|-----------------|-----------|
| WIRE  | Integration Engineer | `agent/tools/verify_execution.py`, `agent/system_prompt.py`, `agent/cli.py` | image_metadata.py, intent_collector.py, iteration_accumulator.py, session_tools.py |
| CLASS | Workflow Analysis Expert | `agent/tools/workflow_parse.py` | verify_execution.py (for `_extract_key_params` pattern) |
| DEMO  | Demo Scenario QA | `agent/brain/demo.py` | all tools/, brain/ modules |
| TEST  | Test Coverage Expert | `tests/test_verify_execution.py`, `tests/test_workflow_classify.py`, `tests/test_system_prompt_metadata.py`, `tests/test_demo_e2e.py` | all agent/ modules |

**Patch protocol:** If Agent A needs a change in Agent B's file:
1. Agent A writes the change to a `.patch` description in their task output
2. Orchestrator applies the patch after Agent B's current task completes
3. Agent B's next task incorporates the change

---

## PHASE 1: BUILD

Build the three new capabilities in parallel. No cross-dependencies between WIRE and CLASS.

Run these agents **in parallel** via Task tool.

### ═══ Agent WIRE — Integration Engineer ═══

**MOE Expertise:** Wiring existing modules together. You connect systems that already work independently but need to talk to each other. You understand data flow, tool dispatch patterns, and session lifecycle.
**You OWN:** `agent/tools/verify_execution.py`, `agent/system_prompt.py`, `agent/cli.py`
**DO NOT TOUCH:** `agent/tools/image_metadata.py`, `agent/brain/intent_collector.py`, `agent/brain/iteration_accumulator.py`, `agent/tools/workflow_parse.py`, `tests/`

**Task W1: Wire metadata auto-embed into verify_execution (post-execution)**

After `_verify_prompt()` confirms outputs exist (`all_exist is True`) and records the outcome, add metadata embedding for each PNG output.

Implementation:
1. In `_verify_prompt()`, after the outcome recording block (step 5, around line 325), add a new step 5.5:
2. Collect intent from the IntentCollectorAgent via tool dispatch:
   ```python
   intent_data = None
   try:
       from . import handle as dispatch_tool
       intent_raw = dispatch_tool("get_current_intent", {})
       intent_result = json.loads(intent_raw)
       if intent_result.get("status") == "ok":
           intent_data = intent_result["intent"]
   except Exception:
       pass
   ```
3. Collect iteration history from the IterationAccumulatorAgent via brain dispatch:
   ```python
   iteration_data = None
   try:
       from ..brain import handle as brain_dispatch
       # We don't finalize here — just get current steps for metadata
       # The finalize call happens elsewhere when the artist accepts
       iter_raw = brain_dispatch("finalize_iterations", {"accepted_iteration": 1})
       iteration_data = json.loads(iter_raw)
   except Exception:
       pass
   ```

   **CORRECTION**: Actually, don't call finalize here. Instead, just read the steps. But the iteration_accumulator doesn't have a "get_steps" tool exposed. So the cleaner approach is: only embed intent + session data + key_params. Skip iteration history for now — it requires the artist to explicitly finalize, which is handled by the MoE pipeline in `iterative_refine.py`. The metadata embed should capture what's available without forcing finalization.

4. For each PNG output that exists, dispatch `write_image_metadata`:
   ```python
   import time as _time

   metadata_embedded = False
   if all_exist and outputs:
       for out in outputs:
           if out["type"] == "image" and out["exists"] and out["absolute_path"].lower().endswith(".png"):
               metadata_payload = {
                   "schema_version": 1,
                   "timestamp": _time.time(),
                   "session": {
                       "session_name": session,
                       "workflow_hash": workflow_hash,
                       "key_params": key_params,
                       "model_combo": model_combo,
                   },
               }
               if intent_data:
                   metadata_payload["intent"] = {
                       "user_request": intent_data.get("user_request", ""),
                       "interpretation": intent_data.get("interpretation", ""),
                       "style_references": intent_data.get("style_references", []),
                       "session_context": intent_data.get("session_context", ""),
                   }
               try:
                   dispatch_tool("write_image_metadata", {
                       "image_path": out["absolute_path"],
                       "metadata": metadata_payload,
                   })
                   metadata_embedded = True
               except Exception as e:
                   log.warning("Failed to embed metadata into %s: %s", out["absolute_path"], e)
   ```
5. Add `"metadata_embedded": metadata_embedded` to the return dict (around line 342-354).
6. Add `import time as _time` at the top of the file (alongside existing imports).

Print status bar after completing.

---

**Task W2: Wire metadata auto-read into session resume (pre-conversation)**

When a session is loaded and has a workflow with previous outputs, read the last output's metadata and inject it into the system prompt.

Implementation in `agent/system_prompt.py`:
1. After the "Proactive recommendations" block (around line 184), add a new block:
   ```python
   # Auto-read creative metadata from last output (if available)
   if session_context and session_context.get("last_output_path"):
       try:
           from .tools import handle as _tools_handle
           import json as _json
           meta_raw = _tools_handle("reconstruct_context", {
               "image_path": session_context["last_output_path"],
           })
           meta = _json.loads(meta_raw)
           if meta.get("has_context"):
               parts.append("\n--- Last Output Context ---")
               parts.append(meta.get("summary", ""))
               ctx = meta.get("context", {})
               if ctx.get("intent"):
                   parts.append(f"  Artist wanted: {ctx['intent'].get('what_artist_wanted', '')}")
                   parts.append(f"  Interpretation: {ctx['intent'].get('how_agent_interpreted', '')}")
               if ctx.get("session", {}).get("key_params"):
                   kp = ctx["session"]["key_params"]
                   parts.append(f"  Last params: {', '.join(f'{k}={v}' for k, v in sorted(kp.items()))}")
               parts.append("")
       except Exception:
           pass  # Metadata unavailable -- skip silently
   ```

Implementation in `agent/cli.py`:
2. In the `run()` function, after the session context is built (around line 129), scan for the most recent output image:
   ```python
   # Detect last output image for metadata resume
   if session_context and load_result.get("workflow_restored"):
       try:
           from pathlib import Path
           from .config import COMFYUI_OUTPUT_DIR
           # Find most recent PNG in output dir
           pngs = sorted(
               COMFYUI_OUTPUT_DIR.glob("*.png"),
               key=lambda p: p.stat().st_mtime,
               reverse=True,
           )
           if pngs:
               session_context["last_output_path"] = str(pngs[0])
       except Exception:
           pass
   ```

Print status bar after completing.

---

**Task W3: Enhance workflow_summary in verify_execution**

Upgrade the flat parameter string to a richer narrative summary. The current summary (line 310-312) is just "model_name 20 steps CFG 7.0". Make it descriptive.

Implementation:
1. Add a new helper function `_build_narrative_summary()` in verify_execution.py:
   ```python
   def _build_narrative_summary(key_params: dict, workflow: dict | None = None) -> str:
       """Build a narrative workflow summary from key params.

       Produces output like: "SDXL txt2img at 1024x1024, 20 steps, CFG 7.0 with DPM++ 2M Karras"
       """
       parts = []

       model = key_params.get("model", "unknown")
       # Strip common suffixes for readability
       for suffix in (".safetensors", ".ckpt", ".pt"):
           if model.endswith(suffix):
               model = model[:-len(suffix)]
               break
       parts.append(model)

       resolution = key_params.get("resolution")
       if resolution:
           parts.append(f"at {resolution}")

       steps = key_params.get("steps")
       if steps:
           parts.append(f"{steps} steps")

       cfg = key_params.get("cfg")
       if cfg is not None:
           parts.append(f"CFG {cfg}")

       sampler = key_params.get("sampler_name")
       scheduler = key_params.get("scheduler")
       if sampler:
           sampler_str = sampler
           if scheduler:
               sampler_str = f"{sampler} {scheduler}"
           parts.append(f"with {sampler_str}")

       denoise = key_params.get("denoise")
       if denoise is not None and denoise < 1.0:
           parts.append(f"(denoise {denoise})")

       return ", ".join(parts[:2]) + (", " + ", ".join(parts[2:]) if len(parts) > 2 else "")
   ```
2. Replace the `workflow_summary` line (310-312) with:
   ```python
   "workflow_summary": _build_narrative_summary(key_params),
   ```

Print status bar after completing.

---

### ═══ Agent CLASS — Workflow Analysis Expert ═══

**MOE Expertise:** Deep understanding of ComfyUI node graphs. You classify workflow patterns, identify pipeline types, and produce human-readable summaries. You think in terms of data flow: loaders → encoders → samplers → outputs.
**You OWN:** `agent/tools/workflow_parse.py`
**DO NOT TOUCH:** `agent/tools/verify_execution.py`, `agent/brain/`, `agent/cli.py`, `agent/system_prompt.py`, `tests/`

**Task C1: Build workflow pattern classifier**

Add a `_classify_pattern()` function that analyzes a node dict and returns a structured classification.

Implementation:
1. Add after `_build_summary()` (around line 286):
   ```python
   # ---------------------------------------------------------------------------
   # Pattern classification
   # ---------------------------------------------------------------------------

   # Known pipeline patterns and their node signatures
   _PATTERN_SIGNATURES: dict[str, dict] = {
       "txt2img": {
           "required": ["EmptyLatentImage"],
           "sampler": True,
           "loader": True,
           "description": "Text-to-image generation",
       },
       "img2img": {
           "required": ["LoadImage", "VAEEncode"],
           "sampler": True,
           "loader": True,
           "description": "Image-to-image transformation",
       },
       "inpaint": {
           "required_any": [
               ["SetLatentNoiseMask"],
               ["InpaintModelConditioning"],
           ],
           "sampler": True,
           "description": "Inpainting or outpainting",
       },
       "controlnet": {
           "required_any": [
               ["ControlNetApply"],
               ["ControlNetApplyAdvanced"],
               ["Apply ControlNet"],
           ],
           "sampler": True,
           "description": "ControlNet-guided generation",
       },
       "upscale": {
           "required_any": [
               ["UpscaleModelLoader"],
               ["ImageUpscaleWithModel"],
               ["LatentUpscale"],
               ["LatentUpscaleBy"],
           ],
           "description": "Image or latent upscaling",
       },
       "lora": {
           "required_any": [
               ["LoraLoader"],
               ["LoraLoaderModelOnly"],
           ],
           "description": "LoRA model adaptation",
       },
       "ip_adapter": {
           "required_any": [
               ["IPAdapterApply"],
               ["IPAdapter"],
               ["IPAdapterAdvanced"],
           ],
           "description": "IP-Adapter image-guided generation",
       },
       "video": {
           "required_any": [
               ["VHS_VideoCombine"],
               ["AnimateDiff"],
               ["SVD_img2vid_Conditioning"],
           ],
           "description": "Video generation or animation",
       },
   }


   def _classify_pattern(nodes: dict) -> dict:
       """Classify workflow into known pipeline patterns.

       Analyzes node class_types to identify what kind of pipeline this is.
       Returns a dict with pattern name, description, modifiers, and confidence.
       """
       # Collect all class_types (He2025: sorted for determinism)
       class_types = sorted({
           n.get("class_type", "") for n in nodes.values()
           if isinstance(n, dict) and not n.get("_ui_node")
       })
       class_set = set(class_types)

       # Check for sampler and loader presence (common requirements)
       has_sampler = any(
           "sampler" in ct.lower() or "ksampler" in ct.lower()
           for ct in class_types
       )
       has_loader = any(
           "checkpoint" in ct.lower() or "unetloader" in ct.lower()
           for ct in class_types
       )

       # Score each pattern
       matched_patterns = []
       for pattern_name, sig in sorted(_PATTERN_SIGNATURES.items()):
           # Check required nodes
           if "required" in sig:
               if not all(req in class_set for req in sig["required"]):
                   continue

           # Check required_any (at least one group must match)
           if "required_any" in sig:
               found_any = False
               for group in sig["required_any"]:
                   if any(node in class_set for node in group):
                       found_any = True
                       break
               if not found_any:
                   continue

           # Check implicit requirements
           if sig.get("sampler") and not has_sampler:
               continue
           if sig.get("loader") and not has_loader:
               continue

           matched_patterns.append(pattern_name)

       # Determine base pattern (priority: most specific first)
       base_pattern = "unknown"
       if not matched_patterns:
           if has_sampler and has_loader:
               base_pattern = "custom"
           elif has_sampler:
               base_pattern = "custom_sampler"
       else:
           # Priority order for base: txt2img < img2img < inpaint
           base_priority = ["txt2img", "img2img", "inpaint"]
           for bp in base_priority:
               if bp in matched_patterns:
                   base_pattern = bp
                   break
           if base_pattern == "unknown":
               base_pattern = matched_patterns[0]

       # Collect modifiers (non-base patterns that also matched)
       modifiers = sorted([p for p in matched_patterns if p != base_pattern])

       # Build description
       desc_parts = [_PATTERN_SIGNATURES.get(base_pattern, {}).get("description", base_pattern)]
       for mod in modifiers:
           mod_desc = _PATTERN_SIGNATURES.get(mod, {}).get("description", mod)
           desc_parts.append(f"with {mod_desc.lower()}")

       description = " ".join(desc_parts)

       return {
           "base_pattern": base_pattern,
           "modifiers": modifiers,
           "all_patterns": sorted(matched_patterns),
           "description": description,
           "node_count": len(nodes),
           "class_types": class_types,
       }
   ```

Print status bar after completing.

---

**Task C2: Add classify_workflow tool**

Register a new tool that exposes pattern classification to the agent.

Implementation:
1. Add to the TOOLS list (after the `get_editable_fields` entry):
   ```python
   {
       "name": "classify_workflow",
       "description": (
           "Classify a workflow into known pipeline patterns. "
           "Returns the base pattern (txt2img, img2img, inpaint, etc.), "
           "any modifiers (controlnet, lora, upscale, etc.), and a "
           "human-readable description. Use this to understand what "
           "kind of pipeline a workflow implements."
       ),
       "input_schema": {
           "type": "object",
           "properties": {
               "path": {
                   "type": "string",
                   "description": "Absolute path to the workflow JSON file.",
               },
           },
           "required": ["path"],
       },
   },
   ```
2. Add the handler:
   ```python
   def _handle_classify_workflow(tool_input: dict) -> str:
       path_str = tool_input["path"]
       data, err = _load_json(path_str)
       if err:
           return to_json({"error": err})

       nodes, fmt = _extract_api_format(data)
       classification = _classify_pattern(nodes)
       classification["file"] = path_str
       classification["format"] = fmt
       return to_json(classification)
   ```
3. Add to the dispatch in `handle()`:
   ```python
   elif name == "classify_workflow":
       return _handle_classify_workflow(tool_input)
   ```
4. Update `summarize_workflow_data()` to include classification:
   ```python
   classification = _classify_pattern(nodes)
   ```
   Add `"classification": classification` to the return dict.

Print status bar after completing.

---

**Task C3: Enhance _build_summary with pattern classification**

Wire the pattern classifier into the existing summary builder so `load_workflow` returns richer summaries.

Implementation:
1. Modify `_build_summary()` to accept an optional classification dict:
   ```python
   def _build_summary(nodes: dict, connections: list[dict], fmt: str, classification: dict | None = None) -> str:
   ```
2. At the top of the summary, add the classification if available:
   ```python
   if classification:
       lines.append(f"Pipeline: {classification['description']}")
       if classification.get("modifiers"):
           lines.append(f"Modifiers: {', '.join(classification['modifiers'])}")
       lines.append("")
   ```
3. Update callers of `_build_summary()` to pass classification:
   - In `_handle_load_workflow`: compute `classification = _classify_pattern(nodes)` and pass it
   - In `summarize_workflow_data`: same
4. Add `"classification": classification` to the return dict of `_handle_load_workflow()`.

Print status bar after completing.

---

### ═══ PHASE 1 GATE ═══

**Run BEFORE starting Phase 2. Gate is HARD — no skip.**

```bash
# Verify new function exists in verify_execution
python -c "from agent.tools.verify_execution import _build_narrative_summary; print('W3: OK')"

# Verify classify_workflow tool exists
python -c "from agent.tools.workflow_parse import _classify_pattern, handle; print('C1-C2: OK')"

# Verify system_prompt handles last_output_path
python -c "from agent.system_prompt import build_system_prompt; print('W2: OK')"

# Zero regressions — ALL 1257 existing tests pass
python -m pytest tests/ -q --tb=short
```

**ALL checks must pass. If ANY fail, fix before proceeding.**

Print status bar after gate check.

---

## PHASE 2: VALIDATE + TEST

Validate demo scenarios and write comprehensive tests for all Phase 1 work.

Run these agents **in parallel** via Task tool.

### ═══ Agent DEMO — Demo Scenario QA ═══

**MOE Expertise:** End-to-end validation of scripted demo flows. You think from the audience's perspective — what would break the magic during a live demo? You validate that tool suggestions in each step are real tools, that step transitions make sense, and that narration is accurate.
**You OWN:** `agent/brain/demo.py`
**DO NOT TOUCH:** `agent/tools/`, `agent/system_prompt.py`, `agent/cli.py`, `tests/`
**DEPENDS ON:** Phase 1 (WIRE, CLASS)

**Task D1: Validate demo tool references**

Check that every `suggested_tools` entry in every demo scenario refers to a real, registered tool.

Implementation:
1. Read `agent/tools/__init__.py` and `agent/brain/__init__.py` to get the full list of registered tool names.
2. Cross-reference every `suggested_tools` entry in `_DEMO_SCENARIOS` against the real tool list.
3. Fix any mismatches:
   - If a tool was renamed, update the reference
   - If a tool doesn't exist, replace with the closest equivalent
   - If `classify_workflow` is relevant to any demo step, add it where appropriate
4. Add `classify_workflow` to the `model_swap` scenario's `analyze` step (it helps the agent understand what kind of pipeline it's analyzing).

Print status bar after completing.

---

**Task D2: Add pattern info to demo checkpoint narration**

Enhance the `_handle_demo_checkpoint()` method to include workflow pattern info when available.

Implementation:
1. In `_handle_demo_checkpoint()`, after the progress calculation, try to read the current workflow classification:
   ```python
   # Enrich checkpoint with workflow pattern if available
   pattern_info = None
   try:
       from ..tools.workflow_patch import get_current_workflow
       from ..tools.workflow_parse import _classify_pattern, _extract_api_format
       wf = get_current_workflow()
       if wf:
           nodes, _fmt = _extract_api_format(wf)
           pattern_info = _classify_pattern(nodes)
   except Exception:
       pass
   ```
2. Add `"workflow_pattern": pattern_info` to both the checkpoint result dict and the demo-complete result dict (where it's not None).

Print status bar after completing.

---

### ═══ Agent TEST — Test Coverage Expert ═══

**MOE Expertise:** Writing thorough, mocked pytest tests that cover happy paths, edge cases, and error conditions. You match the existing test style exactly — same fixtures, same mock patterns, same assertion style. Every test must work WITHOUT a live ComfyUI server.
**You OWN:** `tests/test_verify_execution.py` (append only — preserve existing tests), `tests/test_workflow_classify.py` (new), `tests/test_system_prompt_metadata.py` (new), `tests/test_demo_e2e.py` (new)
**DO NOT TOUCH:** Any `agent/` files
**DEPENDS ON:** Phase 1 (WIRE, CLASS) + DEMO tasks

**Task T1: Tests for metadata auto-embed in verify_execution**

Add tests to `tests/test_verify_execution.py` for the new metadata embedding behavior.

Test cases to write:
1. `test_verify_embeds_metadata_on_success` — When all outputs exist and are PNGs, metadata is embedded via `write_image_metadata` dispatch
2. `test_verify_skips_metadata_when_outputs_missing` — When `all_exist` is False, no metadata embedding attempted
3. `test_verify_metadata_includes_intent` — When intent is captured, it appears in the metadata payload
4. `test_verify_metadata_without_intent` — When no intent captured, metadata is still embedded (just without intent block)
5. `test_verify_metadata_embed_failure_doesnt_break` — If `write_image_metadata` raises, verify_execution still completes normally
6. `test_verify_result_includes_metadata_embedded_flag` — The result dict has `metadata_embedded: True/False`

Mock pattern: patch `agent.tools.verify_execution` imports — specifically the `dispatch_tool` calls. Use the existing `mock_comfyui_database` and `mock_history_success` fixtures.

Print status bar after completing.

---

**Task T2: Tests for workflow pattern classification**

Create `tests/test_workflow_classify.py` with comprehensive pattern tests.

Test cases to write:
1. `test_classify_txt2img` — Workflow with CheckpointLoaderSimple + KSampler + EmptyLatentImage → base_pattern "txt2img"
2. `test_classify_img2img` — Workflow with LoadImage + VAEEncode + KSampler → base_pattern "img2img"
3. `test_classify_controlnet_modifier` — txt2img + ControlNetApply → base_pattern "txt2img", modifiers ["controlnet"]
4. `test_classify_upscale_modifier` — txt2img + UpscaleModelLoader → modifiers include "upscale"
5. `test_classify_lora_modifier` — Workflow with LoraLoader → modifiers include "lora"
6. `test_classify_video` — Workflow with VHS_VideoCombine → matches "video"
7. `test_classify_empty_workflow` — Empty dict → base_pattern "unknown"
8. `test_classify_unknown_nodes` — Workflow with unrecognized nodes but has sampler → "custom"
9. `test_classify_tool_handler` — `handle("classify_workflow", {"path": ...})` returns valid JSON with classification
10. `test_build_summary_includes_pattern` — `_build_summary()` now includes pattern info in output
11. `test_classify_description_reads_naturally` — Compound pattern produces readable description string

Build test workflows as inline dicts (matching existing test patterns). No filesystem I/O needed except for the tool handler test (use tmp_path).

Print status bar after completing.

---

**Task T3: Tests for system prompt metadata injection**

Create `tests/test_system_prompt_metadata.py`.

Test cases:
1. `test_prompt_includes_last_output_context` — When `session_context` has `last_output_path`, the system prompt includes "Last Output Context" section
2. `test_prompt_no_metadata_when_no_path` — When `last_output_path` is absent, no metadata section
3. `test_prompt_metadata_failure_silent` — When `reconstruct_context` fails, prompt builds normally without metadata
4. `test_prompt_metadata_shows_intent` — When metadata has intent, prompt includes "Artist wanted" line

Mock pattern: patch `agent.system_prompt` imports for tool dispatch.

Print status bar after completing.

---

**Task T4: Tests for narrative summary**

Add tests to `tests/test_verify_execution.py` for `_build_narrative_summary()`.

Test cases:
1. `test_narrative_summary_full` — All params present → readable string like "modelname at 1024x1024, 30 steps, CFG 7.0, with euler normal"
2. `test_narrative_summary_minimal` — Only model → "unknown" handling
3. `test_narrative_summary_strips_extension` — Model ending in .safetensors has extension removed
4. `test_narrative_summary_denoise_shown_when_partial` — denoise=0.7 shows "(denoise 0.7)", denoise=1.0 omitted

Print status bar after completing.

---

### ═══ PHASE 2 GATE (FINAL) ═══

**Run BEFORE declaring sprint complete. Gate is HARD — no skip.**

```bash
# Full test suite — must be ALL GREEN with zero regressions
python -m pytest tests/ -q --tb=short

# Verify new test files exist and have content
python -c "
import importlib
for m in ['tests.test_workflow_classify', 'tests.test_system_prompt_metadata']:
    mod = importlib.import_module(m)
    tests = [a for a in dir(mod) if a.startswith('test_')]
    print(f'{m}: {len(tests)} tests')
    assert len(tests) >= 4, f'Too few tests in {m}'
print('All test files: OK')
"

# Verify classify_workflow tool is registered in dispatch
python -c "
from agent.tools.workflow_parse import handle
import json
# Will fail with 'File not found' but proves the tool name routes correctly
result = json.loads(handle('classify_workflow', {'path': '/nonexistent'}))
assert 'error' in result
print('classify_workflow dispatch: OK')
"

# Count total tests — should be ~1280+ (was 1257)
python -m pytest tests/ --co -q 2>&1 | tail -1
```

**ALL checks must pass. If ANY fail, fix before proceeding.**

Print status bar after gate check.

---

## FINAL STATUS BAR

Print after the last phase gate passes:

```
╔══════════════════════════════════════════════════════════════╗
║  ComfyUI Agent — Phase 5B — COMPLETE                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Phase 1: BUILD             [██████████] 100% ✓              ║
║    WIRE    ◆ W1 ✓  W2 ✓  W3 ✓                               ║
║    CLASS   ⟡ C1 ✓  C2 ✓  C3 ✓                               ║
║                                                              ║
║  Phase 2: VALIDATE + TEST   [██████████] 100% ✓              ║
║    DEMO    ◈ D1 ✓  D2 ✓                                     ║
║    TEST    ◇ T1 ✓  T2 ✓  T3 ✓  T4 ✓                        ║
║                                                              ║
║  Overall: [████████████████████] 100%  (13/13 tasks)         ║
║                                                              ║
║  New files:    2 (test_workflow_classify.py, test_system_prompt_metadata.py) ║
║  Modified:     6 (verify_execution, workflow_parse, system_prompt, cli, demo, test_verify) ║
║  Tests added:  ~25                                           ║
║  Regressions:  0                                             ║
╚══════════════════════════════════════════════════════════════╝
```

---

## SAFETY RULES (ALL AGENTS — NON-NEGOTIABLE)

1. **Read before write:** Always read existing code and match conventions
2. **File ownership:** NEVER write to another agent's files
3. **Regression zero:** Existing tests must keep passing
4. **Status reporting:** Print status bar after EVERY task completion
5. **Determinism:** `sort_keys=True` in all JSON, `sorted()` for all dict iterations
6. **No live ComfyUI:** All tests must be fully mocked — no HTTP calls
7. **Match style:** 99-char line length, type hints everywhere, `httpx` for HTTP
8. **Import pattern:** Use tool dispatch (`from . import handle as dispatch_tool`) not direct imports for cross-module calls
9. **Error messages:** Artist-friendly, never raw tracebacks
10. **Thread safety:** Any shared state must use `threading.Lock`
