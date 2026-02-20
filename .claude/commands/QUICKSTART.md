# MoE AGENT TEAM — QUICK REFERENCE

## INSTALL (run from comfyui-agent/)
```bash
mkdir -p .claude/agents
# Copy all files from this delivery into .claude/agents/
# Copy PRODUCTION_HARDEN.md to project root
python bootstrap_hardening.py
```

## RUN AGENTS

### Option A: Orchestrator Script (recommended)
```bash
# Preview what will happen
python .claude/agents/orchestrate.py --phase 1 --dry-run

# Execute Phase 1 (foundation — type safety, error handling, deprecation fixes)
python .claude/agents/orchestrate.py --phase 1

# Execute Phase 2 (core — async, state, security) — can run in parallel
python .claude/agents/orchestrate.py --phase 2

# Execute single workstream
python .claude/agents/orchestrate.py --workstream WS-6

# Check status
python .claude/agents/orchestrate.py --status

# Run everything
python .claude/agents/orchestrate.py --all
```

### Option B: Direct Claude Code Invocation
```bash
# Systems Engineer — Type Safety
claude --model claude-sonnet-4-6-20250929 \
  --system-prompt "$(cat .claude/agents/SYS_ENG.md)" \
  --prompt "Execute workstream WS-1: Type Safety & Static Analysis. Branch: harden/sys-types. See PRODUCTION_HARDEN.md for full task list. Commit after each task. Run tests after each commit. GO."

# ComfyUI Lead — State Management
claude --model claude-sonnet-4-6-20250929 \
  --system-prompt "$(cat .claude/agents/COMFY_LEAD.md)" \
  --prompt "Execute workstream WS-4: State Management Hardening. Branch: harden/comfy-state. See PRODUCTION_HARDEN.md. GO."

# VFX Supervisor — Documentation
claude --model claude-sonnet-4-6-20250929 \
  --system-prompt "$(cat .claude/agents/VFX_SUPER.md)" \
  --prompt "Execute workstream WS-9: Documentation & UX. Branch: harden/vfx-docs. See PRODUCTION_HARDEN.md. GO."

# Nuke Compositor — Platform Config
claude --model claude-sonnet-4-6-20250929 \
  --system-prompt "$(cat .claude/agents/NUKE_COMP.md)" \
  --prompt "Execute workstream WS-10: Platform & Config Hardening. Branch: harden/pipeline-config. See PRODUCTION_HARDEN.md. GO."

# Producer — CI/CD
claude --model claude-sonnet-4-6-20250929 \
  --system-prompt "$(cat .claude/agents/PRODUCER.md)" \
  --prompt "Execute workstream WS-8: CI/CD & Packaging. Branch: harden/ops-cicd. See PRODUCTION_HARDEN.md. GO."
```

### Option C: Quick Win First (5 minutes)
```bash
# Fix Pillow deprecation warnings immediately
claude --model claude-sonnet-4-6-20250929 \
  --prompt "In agent/brain/vision.py, replace all calls to getdata() with get_flattened_data() (Pillow deprecation). Run tests to verify. Commit as: [HARDEN:WS-6] Fix Pillow getdata deprecation in vision.py"
```

## PHASE EXECUTION ORDER

```
Phase 1: FOUNDATION (do first, sequential)
  WS-1  SYS_ENG    Type Safety          harden/sys-types
  WS-3  SYS_ENG    Error Handling       harden/sys-errors
  WS-6  COMFY_LEAD Testing              harden/comfy-testing

Phase 2: CORE HARDENING (parallel OK)
  WS-2  SYS_ENG    Async Architecture   harden/sys-async
  WS-4  COMFY_LEAD State Management     harden/comfy-state
  WS-5  SYS_ENG    Security             harden/sys-security

Phase 3: INFRASTRUCTURE (parallel OK)
  WS-7  PRODUCER   Observability        harden/ops-observability
  WS-8  PRODUCER   CI/CD                harden/ops-cicd
  WS-10 NUKE_COMP  Platform/Config      harden/pipeline-config

Phase 4: POLISH (after Phase 2)
  WS-9  VFX_SUPER  Documentation        harden/vfx-docs
  WS-11 COMFY_LEAD MCP Protocol         harden/comfy-mcp
```

## VERIFICATION (run after each workstream)
```bash
python -m pytest tests/ -q          # 497+ pass, 0 fail
ruff check agent/ tests/            # 0 errors
ruff format --check agent/ tests/   # Clean
```

## MERGE ORDER
```bash
# After all workstreams in a phase complete:
git checkout main
git merge harden/sys-types       # WS-1
git merge harden/sys-errors      # WS-3
git merge harden/comfy-testing   # WS-6
# ... run verification ...
# Continue with next phase
```
