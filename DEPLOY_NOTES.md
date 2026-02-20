# ComfyUI Agent — Deploy Package

## Status: Ready to Share

```
Tests:    1109 passed, 2 skipped, 0 failed
Lint:     0 warnings
Tools:    77 (50 intelligence + 27 brain)
Version:  0.4.0
```

---

## What Was Fixed

### Consistency (version/count drift)
- `agent/__init__.py`: version synced to 0.4.0 (matches pyproject.toml)
- README: tool counts updated to 77 (50+27), test count to 1100+
- CLAUDE.md: brain count corrected to 27
- Model default clarified: Sonnet for CLI, MCP inherits from Claude Code/Desktop
- Added `tool_count()` function for programmatic access

### Test Resilience
- `test_routes_panels.py`: skip when aiohttp missing (UI dep)
- `test_sidebar_workflow.py`: skip UI class when aiohttp missing
- `test_3d_demos.py`: removed unused imports (lint fix)

### New Files for Team Deployment
- `QUICKSTART.md` — 5-minute onboarding for VFX artists
- `scripts/setup.py` — interactive first-time config (writes .env, updates bat paths)
- `scripts/validate_project.py` — pre-share consistency checker
- `scripts/comfyui_with_agent.bat` — launches ComfyUI + agent together
- `scripts/deploy.py` — coworker auto-setup (install, config, validate, shortcut)

---

## Coworker Instructions

Send them this:

```
1. Clone: git clone https://github.com/JosephOIbrahim/comfyui-agent.git
2. Deploy: cd comfyui-agent && python scripts/deploy.py
3. Launch: scripts\comfyui_with_agent.bat

Or read QUICKSTART.md for manual setup.
```

The deploy script:
- Checks Python version
- Installs the package
- Runs interactive setup (API key, ComfyUI path)
- Validates tool loading
- Tests ComfyUI connectivity
- Offers a desktop shortcut (Windows)

---

## Startup Integration

### How it works
`scripts/comfyui_with_agent.bat` wraps your existing `comfyui_zen.bat`:

1. Starts ComfyUI in a new window via `comfyui_zen.bat`
2. Polls `http://127.0.0.1:8188/system_stats` every 2s until ready
3. Prints connection instructions (Claude Code, CLI, or Desktop)

### Coworker customization
Each person edits two lines at the top of the bat file:
```batch
set COMFYUI_BAT=G:\COMFY\ComfyUI\comfyui_zen.bat    ← their ComfyUI launcher
set AGENT_DIR=C:\Users\TheirName\comfyui-agent       ← their agent clone
```

Or `python scripts/setup.py` does this interactively.

### Launch modes
```
comfyui_with_agent.bat              Start everything (MCP mode)
comfyui_with_agent.bat --cli        Start with CLI chat
comfyui_with_agent.bat --no-agent   ComfyUI only
```

---

## Pre-Push Checklist

```bash
python scripts/validate_project.py       # All green?
python -m pytest tests/ -q               # 1109 passed?
ruff check agent/ tests/                 # All checks passed?
git add -A && git status                 # Review changes
git commit -m "[DEPLOY] Pre-share fixes: consistency, team setup, startup integration"
git push
```

---

## Known Limitations (Tell Coworkers)

1. **MCP mode needs Claude Code or Desktop** — the `agent mcp` command starts
   the MCP server, but you need an MCP client to talk to it. Claude Code auto-connects.
   Claude Desktop needs manual MCP config (see QUICKSTART.md).

2. **CLI mode needs an API key** — `agent run` uses the Anthropic API directly.
   Each coworker needs their own key from console.anthropic.com.

3. **ComfyUI must be running** — most tools require a live ComfyUI instance.
   The bat launcher handles this, but manual users need to start ComfyUI first.

4. **End-to-end loop not yet battle-tested** — the execute→verify→vision→memory
   pipeline has all components wired but hasn't been stress-tested on diverse
   workflows. Report issues via GitHub.
