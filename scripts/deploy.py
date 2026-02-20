#!/usr/bin/env python3
"""Deploy helper — run this to set up a new machine.

Does everything a coworker needs:
  1. Checks Python version
  2. Checks pip install
  3. Runs interactive setup (API key, paths)
  4. Validates the project
  5. Creates a desktop shortcut (Windows)
  6. Tests connectivity to ComfyUI (if running)

Usage:
    python scripts/deploy.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Force UTF-8 on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PASS = "\033[92m+\033[0m"
FAIL = "\033[91mX\033[0m"
WARN = "\033[93m!\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, **kw)


def main():
    print()
    print(f"  {BOLD}╔═══════════════════════════════════════════╗{RESET}")
    print(f"  {BOLD}║  ComfyUI Agent — Deploy Setup             ║{RESET}")
    print(f"  {BOLD}╚═══════════════════════════════════════════╝{RESET}")
    print()

    errors = []

    # ── 1. Python version ───────────────────────────────────────────
    print(f"  {BOLD}[1/6] Python version{RESET}")
    v = sys.version_info
    ok = v >= (3, 10)
    status = PASS if ok else FAIL
    print(f"    {status} Python {v.major}.{v.minor}.{v.micro}")
    if not ok:
        errors.append("Python 3.10+ required")
    print()

    # ── 2. Package install ──────────────────────────────────────────
    print(f"  {BOLD}[2/6] Package installation{RESET}")
    result = run([sys.executable, "-m", "pip", "install", "-e", "."])
    if result.returncode == 0:
        print(f"    {PASS} comfyui-agent installed")
    else:
        # Try with --break-system-packages (container/managed Python)
        result = run([sys.executable, "-m", "pip", "install", "-e", ".", "--break-system-packages"])
        if result.returncode == 0:
            print(f"    {PASS} comfyui-agent installed (system packages)")
        else:
            print(f"    {FAIL} pip install failed")
            print(f"      {result.stderr[:200]}")
            errors.append("pip install failed")

    # Verify agent CLI is available
    agent_check = run(["agent", "--help"], shell=(sys.platform == "win32"))
    if agent_check.returncode == 0:
        print(f"    {PASS} 'agent' command available")
    else:
        print(f"    {WARN} 'agent' command not on PATH (may need to restart terminal)")
    print()

    # ── 3. Interactive setup (.env) ─────────────────────────────────
    print(f"  {BOLD}[3/6] Configuration{RESET}")
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        print(f"    {PASS} .env exists")
        # Quick check it has an API key
        content = env_file.read_text(encoding="utf-8")
        if "sk-ant-" in content and "PASTE" not in content:
            print(f"    {PASS} API key configured")
        else:
            print(f"    {WARN} API key looks like a placeholder — edit .env")
    else:
        print(f"    Running first-time setup...")
        print()
        result = run([sys.executable, "scripts/setup.py"])
        if result.returncode != 0:
            print(f"    {FAIL} Setup failed")
            errors.append("setup.py failed")
    print()

    # ── 4. Tool count validation ────────────────────────────────────
    print(f"  {BOLD}[4/6] Tool validation{RESET}")
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from agent.tools import ALL_TOOLS
        total = len(ALL_TOOLS)
        print(f"    {PASS} {total} tools loaded")
    except Exception as e:
        print(f"    {FAIL} Tool load failed: {e}")
        errors.append(f"Tool load: {e}")
    print()

    # ── 5. ComfyUI connectivity ─────────────────────────────────────
    print(f"  {BOLD}[5/6] ComfyUI connectivity{RESET}")
    try:
        import httpx
        r = httpx.get("http://127.0.0.1:8188/system_stats", timeout=5)
        if r.status_code == 200:
            data = r.json()
            gpus = data.get("devices", [])
            gpu_name = gpus[0].get("name", "unknown") if gpus else "no GPU info"
            print(f"    {PASS} ComfyUI running ({gpu_name})")
        else:
            print(f"    {WARN} ComfyUI responded with status {r.status_code}")
    except Exception:
        print(f"    {WARN} ComfyUI not running (that's OK — start it when ready)")
    print()

    # ── 6. Startup script ───────────────────────────────────────────
    print(f"  {BOLD}[6/6] Startup integration{RESET}")
    bat_file = PROJECT_ROOT / "scripts" / "comfyui_with_agent.bat"
    if bat_file.exists():
        print(f"    {PASS} Launcher script exists: scripts/comfyui_with_agent.bat")

        # Check if paths are configured
        bat_content = bat_file.read_text(encoding="utf-8")
        if "C:\\Users\\User" in bat_content:
            agent_dir = str(PROJECT_ROOT).replace("/", "\\")
            print(f"    {WARN} Launcher still has default paths — updating AGENT_DIR...")
            bat_content = bat_content.replace(
                "set AGENT_DIR=C:\\Users\\User\\comfyui-agent",
                f"set AGENT_DIR={agent_dir}",
            )
            bat_file.write_text(bat_content)
            print(f"    {PASS} Updated AGENT_DIR to {agent_dir}")
            print(f"    {WARN} You still need to edit COMFYUI_BAT to point to your ComfyUI .bat file")
        else:
            print(f"    {PASS} Launcher paths configured")

        # Offer to create desktop shortcut (Windows)
        if sys.platform == "win32":
            desktop = Path.home() / "Desktop"
            if desktop.exists():
                shortcut_bat = desktop / "ComfyUI+Agent.bat"
                if not shortcut_bat.exists():
                    ans = input(f"\n    Create desktop shortcut? [Y/n] ").strip().lower()
                    if ans in ("", "y", "yes"):
                        shortcut_bat.write_text(
                            f'@echo off\ncall "{bat_file}"\n'
                        )
                        print(f"    {PASS} Created: {shortcut_bat}")
                    else:
                        print(f"    Skipped.")
                else:
                    print(f"    {PASS} Desktop shortcut exists")
    else:
        print(f"    {FAIL} Launcher not found: scripts/comfyui_with_agent.bat")
        errors.append("Missing launcher script")
    print()

    # ── Summary ─────────────────────────────────────────────────────
    print(f"  {BOLD}{'='*47}{RESET}")
    if errors:
        print(f"  {FAIL} {len(errors)} issue(s):")
        for e in errors:
            print(f"     • {e}")
    else:
        print(f"  {PASS} {BOLD}Ready to go!{RESET}")
        print()
        print(f"  Next steps:")
        print(f"    1. Start ComfyUI: scripts\\comfyui_with_agent.bat")
        print(f"    2. Or manually:   start ComfyUI, then 'agent run'")
        print(f"    3. Best:          Claude Code with MCP (see QUICKSTART.md)")
    print(f"  {'='*47}")
    print()

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
