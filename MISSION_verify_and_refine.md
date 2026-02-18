# MISSION: Live Integration Test → Iterative Refine Brain Tool

> **Two tasks, sequential. Do not start Task 2 until Task 1 passes.**
> Read CLAUDE.md first for full project context. This document is additive.

---

## Current State (as of 2026-02-18)

- 518 tests passing, 0 lint warnings
- 62 tools registered (43 intelligence, 19 infrastructure)
- **Just shipped:** `agent/tools/verify_execution.py` (get_output_path, verify_execution)
- **Just shipped:** auto_verify flag on `execute_with_progress` in `agent/tools/comfy_execute.py`
- **Just shipped:** `dispatch_brain_message()` in `agent/brain/_protocol.py` — fire-and-forget routing of vision→memory
- **Just shipped:** vision dispatch wiring in `agent/brain/vision.py` (analyze_image, compare_outputs → record_outcome)
- The execute→verify→vision→memory loop has all its components. It has NOT been tested end-to-end against live ComfyUI.

---

## TASK 1: Live Integration Test

### Goal
Prove the full pipeline works against a running ComfyUI instance. Real workflow, real execution, real image output, real verification, real vision analysis, real memory recording.

### Prerequisites
- ComfyUI must be running at `http://127.0.0.1:8188`
- Confirm it's up: `curl http://127.0.0.1:8188/system_stats`
- If it's not up, **stop and tell me**. Don't mock it.

### What to Test (Sequential)

**Step 1: Confirm connectivity**
```python
# Hit the ComfyUI API directly via the agent's comfy_api module
from agent.tools.comfy_api import _comfy_get
result = _comfy_get("/system_stats")
# Should return GPU info, queue status
```
Verify: GPU shows RTX 4090, queue is accessible.

**Step 2: Find a usable workflow**
Check for any `.json` workflow files in the project. If none exist, use `/object_info` to discover what models/nodes are loaded, then construct a minimal txt2img workflow:
- A checkpoint loader (whatever checkpoint is available)
- A CLIP text encoder with a simple prompt like "a red sphere on a white background"
- A KSampler (Euler, 20 steps, CFG 7)
- A VAE decode
- A SaveImage node

The point is NOT to make beautiful art. It's to prove the pipeline works. Use the simplest possible workflow.

**Step 3: Execute via `execute_with_progress`**
```python
from agent.tools.comfy_execute import execute_with_progress
result = execute_with_progress(
    workflow=workflow_dict,
    auto_verify=True,
    session="integration_test",
    goal_id="verify_pipeline"
)
```
Expected: execution completes, auto_verify triggers `_verify_prompt()`, returns result with `verification` field attached.

**Step 4: Verify output file exists**
```python
from agent.tools.verify_execution import get_output_path
path_result = get_output_path(filename=result["verification"]["output_filename"])
```
Expected: returns absolute path, file exists on disk, size > 0.

**Step 5: Vision analysis on the output**
```python
from agent.brain.vision import analyze_image
analysis = analyze_image(image_path=path_result["path"])
```
Expected: returns quality score, artifact assessment, description. The `dispatch_brain_message` should fire, routing the result toward `record_outcome` in the memory module.

**Step 6: Confirm memory recording**
Check that the outcome was recorded. Query the memory/session system for the goal_id "verify_pipeline" and confirm the vision analysis data landed.

### Success Criteria
All 6 steps complete without errors. The full loop — execute → verify → vision → memory — runs end to end against live ComfyUI.

### Failure Protocol
If any step fails:
1. **Do not skip it.** Fix it.
2. Diagnose root cause. Is it connectivity? Missing model? File path issue? API shape mismatch?
3. Fix the minimal thing that makes it work.
4. Re-run from the failing step.
5. If the fix requires code changes, run the full test suite (`pytest`) after the fix to confirm nothing broke.

### What NOT to Do
- Do NOT mock ComfyUI responses. This is a live test.
- Do NOT write a new test file for this. Run it interactively or as a script.
- Do NOT refactor anything. If it works ugly, it works. Note cleanup opportunities for later.
- Do NOT get distracted by optimization or edge cases. The goal is: does the happy path work?

---

## TASK 2: `iterative_refine` Brain Tool

> **Only start this after Task 1 passes completely.**

### Goal
Build the self-healing reinforcement loop as a brain tool. This is the feature that turns the agent from "tool that does what you say" into "co-pilot that iterates toward artistic intent."

### Architecture

The loop:
```
INTENT (user describes what they want)
  ↓
EXECUTE (run the workflow)
  ↓
PERCEIVE (vision analysis → quality score + artifact detection)
  ↓
REMEMBER (record outcome: parameters, scores, context)
  ↓
DIAGNOSE (compare against memory corpus, identify improvement vector)
  ↓
PRESCRIBE (generate a specific JSON patch to address diagnosis)
  ↓
EXECUTE (apply patch, run again)
  ↓
CONVERGE? →  threshold met: return best result
             improving: continue loop
             plateaued (2 consecutive < 0.3 improvement): stop, return best
             regressing: roll back last patch, try different vector
```

### File Location
`agent/brain/iterative_refine.py`

### Tool Registration
Register in `agent/tools/__init__.py` following existing brain tool patterns. The tool should appear in the brain tool category alongside planner, orchestrator, optimizer, etc.

### Interface

```python
def iterative_refine(
    intent: str,                    # "photorealistic portrait, no artifacts"
    mode: str = "refine",           # "quick" | "refine" | "deep" | "unlimited"
    quality_threshold: float = 7.0, # target score (1-10)
    session: str = "",              # session ID for memory
    goal_id: str = "",              # goal ID for memory tracking
) -> dict:
    """
    Autonomous quality iteration loop.
    
    Modes control iteration budget:
      quick:     1-2 iterations, fire-and-forget
      refine:    3-5 iterations, autonomous
      deep:      5-10 iterations, autonomous with progress updates
      unlimited: 1 iteration per call (caller drives the outer loop)
    
    Returns:
      {
        "iterations": [
          {
            "iteration": 1,
            "parameters": {...},      # key params for this run
            "quality_score": 6.2,
            "artifacts_detected": ["color banding"],
            "diagnosis": "CFG too high for this checkpoint",
            "patch_applied": {...},   # the RFC6902 patch
            "output_path": "/path/to/output.png"
          },
          ...
        ],
        "best_result": {
          "iteration": 3,
          "quality_score": 8.1,
          "output_path": "/path/to/best.png",
          "parameters": {...}
        },
        "converged": true,            # did we meet threshold?
        "reason": "threshold_met",    # threshold_met | plateaued | max_iterations | regression_rollback
        "recommendation": "...",      # what to try next if not converged
        "total_iterations": 4
      }
    ```
"""
```

### Mode Behavior

| Mode | Max Iterations | Loop Owner | User Involvement |
|------|---------------|-----------|-----------------|
| quick | 2 | Tool (internal) | None |
| refine | 5 | Tool (internal) | None |
| deep | 10 | Tool (internal) | None |
| unlimited | 1 | Caller (Claude Code or agent loop) | Every iteration |

For `unlimited` mode, the tool runs exactly ONE iteration and returns. The caller decides whether to call again, incorporating user feedback. This enables human-in-the-loop refinement.

### Convergence Detection

Implement three convergence signals:

1. **Threshold met:** `quality_score >= quality_threshold` → stop, return best
2. **Plateau:** Quality hasn't improved by > 0.3 for 2 consecutive iterations → stop, return best with explanation
3. **Regression:** Quality dropped vs previous iteration → roll back the last patch, try a DIFFERENT adjustment vector. If 2 consecutive regressions, stop.

Track quality scores in a list. Convergence checks run after each iteration.

### Diagnosis Engine

The diagnosis step is where memory makes the loop smarter over time.

**Early use (sparse memory):** Fall back to general heuristics.
```python
# Heuristic examples:
# - "color banding" → suggest lowering CFG
# - "blurry/soft" → suggest increasing steps or switching sampler
# - "artifacts at edges" → suggest adjusting denoise strength
# - "wrong style" → suggest different checkpoint
```

**With memory data:** Query memory for similar past outcomes.
```python
# Pseudocode:
outcomes = query_memory(
    model_combo=current_models,
    similar_params=True,
    min_quality=quality_threshold
)
if outcomes:
    recommendation = aggregate_optimal_params(outcomes)
else:
    recommendation = general_heuristic(detected_issues)
```

Build the heuristic fallback first. Memory-informed diagnosis is an enhancement — get the loop working with heuristics, then wire in memory queries.

### Patch Generation

The diagnosis produces an adjustment vector (e.g., "lower CFG by 1.5"). This must be translated into an RFC6902 JSON patch using the existing `workflow_patch.py` engine.

Use the existing patch infrastructure. Do NOT build a separate patching system.

**Adjustment vectors to support (minimum viable set):**
- CFG scale (up/down)
- Steps (up/down)
- Sampler name (switch)
- Scheduler (switch)
- Denoise strength (up/down)
- Seed (randomize for variety)

These cover the most common quality levers. More can be added later.

### Integration Points

| Component | Module | How It's Used |
|-----------|--------|---------------|
| Execute workflow | `agent/tools/comfy_execute.py` → `execute_with_progress` | Run the workflow each iteration |
| Verify execution | `agent/tools/verify_execution.py` → `verify_execution` | Confirm output exists, get path |
| Vision analysis | `agent/brain/vision.py` → `analyze_image` | Score quality, detect artifacts |
| Memory recording | Automatic via `dispatch_brain_message` | Vision results flow to memory |
| Memory query | `agent/brain/memory.py` (or equivalent) | Query past outcomes for diagnosis |
| Patch generation | `agent/tools/workflow_patch.py` | Apply parameter adjustments |
| Patch validation | `agent/tools/workflow_patch.py` | Validate patches before applying |

### Testing Strategy

**Unit tests** (`tests/test_iterative_refine.py`):
1. Mode selection: each mode respects its iteration budget
2. Convergence: threshold_met stops the loop
3. Convergence: plateau detection works (mock scores: 6.0, 6.2, 6.1 → plateau)
4. Convergence: regression detection works (mock scores: 6.0, 5.5 → rollback)
5. Patch generation: diagnosis maps to valid RFC6902 patches
6. Unlimited mode: returns after exactly 1 iteration
7. Heuristic diagnosis: known artifacts map to correct adjustments
8. Result structure: output matches the documented interface

**Mock ComfyUI for unit tests.** The live integration was Task 1. Unit tests should be fast and deterministic.

### What NOT to Do
- Do NOT build a separate execution pipeline. Use `execute_with_progress`.
- Do NOT build a separate vision system. Use `analyze_image`.
- Do NOT build a separate patching system. Use `workflow_patch.py`.
- Do NOT try to build the memory-informed diagnosis in the first pass. Heuristics first, memory later.
- Do NOT add artistic intent vocabulary mapping yet. That's a follow-on feature.
- Do NOT add `parameter_sweep` or batch execution. That's Phase 3 from the roadmap, not this task.

### Definition of Done
- `iterative_refine` tool registered and discoverable
- All unit tests pass
- Full test suite still passes (no regressions)
- Lint clean
- The tool can be called from the agent loop with a workflow and intent string

### Stretch (Only If Everything Above Is Clean)
- Live test: run `iterative_refine` in `quick` mode against the same workflow from Task 1
- Confirm it executes, analyzes, diagnoses, patches, and re-executes at least once
- This is the ultimate proof: the self-healing loop running for real

---

## General Rules

1. **Run `pytest` after every code change.** Non-negotiable.
2. **Run lint after every code change.** Match current zero-warning state.
3. **Follow existing code patterns.** Look at how other brain tools are structured before writing new ones.
4. **Read `agent/tools/__init__.py`** to understand registration patterns before adding new tools.
5. **Read `agent/brain/_protocol.py`** to understand how brain modules communicate.
6. **If something is unclear, read the code. Don't guess.**
7. **Keep the commit atomic.** Task 1 is validation only (no code changes unless bugs are found). Task 2 is new feature code + tests.
