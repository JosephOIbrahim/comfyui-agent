# DISPATCH: Graft F — ProfilerX Bottleneck Detection (VERIFY Layer)
## For: Claude Code Agent Teams
## Repos: comfyui-agent (code) + comfyui-3D-viewport (tracking)
## Parent: C:\Users\User\comfyui-agent\
## Duration: 20–25 minutes
## Type: [AGENT-GRIND] — Knowledge-only, no infrastructure

---

## DISPATCH PROMPT

```
/COMFY_LEAD

SHOW: SUPER DUPER UI — Graft F: ProfilerX in VERIFY
You are the Lead TD. Quick knowledge enhancement to the VERIFY layer.

CONTEXT: ComfyUI now has ProfilerX — a per-node execution profiler that
overlays timing data directly on the node graph. When a workflow runs slow,
this is the diagnostic tool. The agent's VERIFY layer handles execution
and error recovery but currently has no awareness of performance profiling
tools. When a user says "my workflow is slow" or "which node is the
bottleneck", the agent should know ProfilerX exists and recommend it.

This is NOT an agent-operated tool. The agent recommends it, the user
installs and reads the overlay. Same pattern as Graft D (VNCCS, Action
Director) — discovery, not operation.

YOUR DELIVERABLE: When a user reports slow workflows or asks about
performance optimization, the agent surfaces ProfilerX with actionable
guidance.

TASKS:

1. FIND WHERE PERFORMANCE KNOWLEDGE BELONGS
   Check agent/knowledge/ for existing files that cover:
   - Workflow optimization
   - Common troubleshooting
   - Performance tips
   - Error recovery

   grep -rn "slow\|performance\|bottleneck\|optimize\|speed" \
     agent/knowledge/ --include="*.md" | head -20

   Also check what triggers exist for performance-related queries:
   grep -n "slow\|performance\|bottleneck\|optimize\|speed" \
     agent/system_prompt.py

   Decision:
   - If a workflow optimization / troubleshooting file exists → append to it
   - If not → create agent/knowledge/workflow_optimization.md

2. ADD PROFILERX KNOWLEDGE

   Add this to the appropriate knowledge file:

   ```markdown
   ## Workflow Performance Profiling

   ### ProfilerX — Per-Node Bottleneck Detection
   - **What:** Overlays execution timing on every node in the graph.
     Shows exactly which node is the bottleneck.
   - **Install:** Available via ComfyUI-Manager
   - **Use case:** User reports slow workflow, wants to know which
     node to optimize or replace
   - **Not agent-operated:** User installs, runs workflow, reads the
     timing overlay. Agent's role is to recommend it and interpret results.

   ### When to Recommend ProfilerX
   - "My workflow is slow"
   - "Which node is taking the longest?"
   - "How do I speed up my pipeline?"
   - "This takes forever to run"
   - User reports a workflow taking longer than expected

   ### Common Bottleneck Patterns (for agent to explain)
   - **KSampler with high step count:** Most common. Suggest reducing
     steps or switching to a faster scheduler.
   - **ControlNet preprocessing:** Depth/normal map generation can be
     slow on large images. Suggest downscaling input.
   - **Model loading:** First run loads model to VRAM. Subsequent runs
     are faster. Not a real bottleneck.
   - **3D generation (Hunyuan3D, Meshy, etc.):** Inherently slow —
     mesh generation takes minutes, not seconds. Set expectations.
   - **Image upscaling:** Tile-based upscalers are slower but use less
     VRAM. Trade-off the user should understand.

   ### After ProfilerX Identifies the Bottleneck
   Agent should be able to suggest:
   1. Parameter adjustments (fewer steps, lower resolution, smaller batch)
   2. Alternative nodes (faster sampler, lighter model variant)
   3. Workflow restructuring (move expensive ops later, cache intermediates)
   4. Hardware reality check (some ops are just slow on the hardware available)
   ```

3. ADD TRIGGER KEYWORDS
   In agent/system_prompt.py, find _KNOWLEDGE_TRIGGERS and add entries
   so performance queries load this knowledge:

   Map these keywords to whatever file you put the knowledge in:
   - "slow"
   - "bottleneck"  
   - "optimize workflow"
   - "performance"
   - "speed up"
   - "taking too long"
   - "profiler"

   IMPORTANT: Check if any of these keywords already trigger a different
   knowledge file. If so, either:
   (a) Add ProfilerX knowledge to THAT file instead, or
   (b) Make the keyword trigger BOTH files if the system supports it

   Do NOT break existing trigger mappings.

4. VERIFY NO EXISTING CONFLICT
   The VERIFY layer in agent/tools/comfy_execute.py handles execution.
   Check if there's already any performance monitoring or timing logic:

   grep -n "time\|duration\|performance\|slow\|profile" \
     agent/tools/comfy_execute.py | head -10

   If the VERIFY layer already tracks execution time per node, note this —
   ProfilerX knowledge should reference it:
   "The agent tracks basic execution time. For detailed per-node profiling
   with visual overlay, install ProfilerX."

5. TEST
   These are conversational tests. Run if the agent is startable:

   Test 1: "My workflow is running really slow, what should I do?"
     PASS if: Agent mentions ProfilerX as a diagnostic tool
     PASS if: Agent also suggests common bottleneck patterns
     FAIL if: Agent gives only generic "try fewer steps" without
              mentioning profiling tools

   Test 2: "Which node in my workflow is the bottleneck?"
     PASS if: Agent recommends ProfilerX for per-node timing
     FAIL if: Agent says it can't determine this

   Test 3: "How do I optimize my ComfyUI workflow?"
     PASS if: Agent provides optimization strategies AND mentions
              ProfilerX for identifying what to optimize first
     FAIL if: Generic advice without profiling recommendation

   If agent is not startable, verify via:
   - Trigger keywords are wired correctly
   - Knowledge file loads when triggered (trace the code path)
   - Existing tests still pass

6. RUN FULL TEST SUITE
   python -m pytest tests/ -v
   
   All existing tests must still pass. If any fail, your changes
   broke something — revert and investigate.

7. REPORT
   ```
   DISPATCH: ProfilerX VERIFY Enhancement
   ─────────────────────────────────────────
   Knowledge added to: [filename]
   Trigger keywords added: [list]
   Existing trigger conflicts: [none / resolved how]
   VERIFY layer timing exists: [yes/no — what we found]
   
   Tests:
     Test 1 (slow workflow):     [PASS/FAIL/SKIPPED]
     Test 2 (bottleneck ID):     [PASS/FAIL/SKIPPED]
     Test 3 (optimization):      [PASS/FAIL/SKIPPED]
     Full suite:                 [N tests passed / N failed]
   
   Files modified:
     [list with one-line description of each change]
   ```

SCOPE: 20-25 minutes. This is a knowledge addition, not infrastructure.
Add the knowledge, wire the triggers, verify it works, register the graft, move on.
Do NOT redesign the VERIFY layer or add execution profiling code.
The agent RECOMMENDS ProfilerX — it doesn't BECOME ProfilerX.

8. REGISTER GRAFT F IN VIEWPORT REPO
   Open: comfyui-3D-viewport/SUPERDUPER_3D_GRAFT.md
   
   Add the following section AFTER Graft E and BEFORE "Graft Integration Map":

   ```markdown
   ---

   ## Graft F: VERIFY — Workflow Performance Profiling
   **Grafts into:** Phase 2 (Backend Bridge) or standalone
   **Department:** `/COMFY_LEAD`
   **Duration:** 20-25 minutes
   **When to run:** Any time after Grafts A-D ship

   ### DISPATCH PROMPT:
   See DISPATCH_PROFILERX.md for full dispatch.

   ### Summary:
   Adds ProfilerX awareness to the VERIFY layer. When a user reports
   slow workflows, the agent recommends ProfilerX for per-node timing
   and provides common bottleneck patterns (KSampler steps, ControlNet
   preprocessing, model loading, 3D generation expectations).

   ### Knowledge Added:
   - ProfilerX tool description and install guidance
   - Trigger keywords: slow, bottleneck, optimize, performance, speed up
   - Common bottleneck patterns with resolution strategies
   - Post-profiling optimization recommendations

   ### Source:
   February 2026 ecosystem research — ProfilerX identified as high-value
   community tool for workflow optimization. Not agent-operated; agent
   recommends and interprets.
   ```

   Then update the Graft Integration Map to include:

   ```
   Phase 2: Backend Bridge ────────→ GRAFT A: 3D data types in UNDERSTAND
                                      GRAFT F: ProfilerX in VERIFY
   ```

   And update the Execution Order:

   ```
   Phase 0 → Phase 1 → Phase 2 → GRAFTS A,F → Phase 3 + GRAFTS B,C,D → Phase 4 → Phase 5 + GRAFT E
   ```

9. REPORT
   ```
   GRAFT F: ProfilerX VERIFY Enhancement
   ─────────────────────────────────────────
   Knowledge added to: [filename in comfyui-agent]
   Trigger keywords added: [list]
   Existing trigger conflicts: [none / resolved how]
   VERIFY layer timing exists: [yes/no — what we found]
   
   Tests:
     Test 1 (slow workflow):     [PASS/FAIL/SKIPPED]
     Test 2 (bottleneck ID):     [PASS/FAIL/SKIPPED]
     Test 3 (optimization):      [PASS/FAIL/SKIPPED]
     Full suite:                 [N tests passed / N failed]
   
   Files modified (comfyui-agent):
     [list with one-line description of each change]
   
   Files modified (comfyui-3D-viewport):
     SUPERDUPER_3D_GRAFT.md — Added Graft F section + updated integration map
   ```
```

---

## EXECUTION NOTES

- This dispatch runs INDEPENDENTLY of EXECUTION_SPEC.md phases
- Can run in parallel with Phase 1 verification
- Code changes land in C:\Users\User\comfyui-agent\ (parent project)
- Graft tracking update lands in comfyui-3D-viewport repo (SUPERDUPER_3D_GRAFT.md)
- If Phase 2 of EXECUTION_SPEC also touches trigger keywords,
  coordinate — don't overwrite each other's additions
- The bottleneck pattern knowledge (KSampler steps, ControlNet
  preprocessing, model loading, etc.) is the high-value piece.
  ProfilerX recommendation alone is useful but the patterns
  make the agent genuinely helpful for performance questions.

## FILE DISTRIBUTION

This file and EXECUTION_SPEC.md go in BOTH repos:

```
comfyui-agent/
  EXECUTION_SPEC.md          # Execution phases (code changes happen here)
  DISPATCH_PROFILERX.md      # This file (Graft F dispatch)

comfyui-3D-viewport/
  CLAUDE.md                  # Project context (already exists)
  SUPERDUPER_3D_GRAFT.md     # Graft plan A-F (updated by this dispatch)
  EXECUTION_SPEC.md          # Same file — phases reference this repo too
  DISPATCH_PROFILERX.md      # Same file — graft tracking references this repo
```

## HOW TO RUN

```
Read DISPATCH_PROFILERX.md. Execute the dispatch prompt.
Working directories:
  Code changes: C:\Users\User\comfyui-agent\
  Graft tracking: comfyui-3D-viewport repo
Report at step 9. Do not proceed to other work without reporting.
```
