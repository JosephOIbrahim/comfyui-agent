#!/usr/bin/env python3
"""Bootstrap the MoE Agent Team system into comfyui-agent.

Run from the comfyui-agent project root:
    python bootstrap_hardening.py

This will:
1. Create .claude/agents/ directory with all agent prompts
2. Copy PRODUCTION_HARDEN.md to project root
3. Create all hardening branches
4. Run initial verification gate
"""

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
AGENTS_SRC = PROJECT_ROOT / "agents"  # Assumes agents/ is alongside this script
AGENTS_DST = PROJECT_ROOT / ".claude" / "agents"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and print it."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=check)


def main():
    print("=" * 60)
    print("MoE Agent Team — Production Hardening Bootstrap")
    print("=" * 60)

    # 1. Create .claude/agents/
    print("\n[1/5] Creating .claude/agents/ directory...")
    AGENTS_DST.mkdir(parents=True, exist_ok=True)

    agent_files = ["SYS_ENG.md", "COMFY_LEAD.md", "VFX_SUPER.md", "NUKE_COMP.md", "PRODUCER.md", "orchestrate.py"]
    for f in agent_files:
        src = AGENTS_SRC / f
        dst = AGENTS_DST / f
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✅ {f}")
        else:
            print(f"  ⚠️  {f} not found at {src}")

    # 2. Copy PRODUCTION_HARDEN.md
    print("\n[2/5] Installing PRODUCTION_HARDEN.md...")
    src = PROJECT_ROOT / "PRODUCTION_HARDEN.md"
    if src.exists():
        print(f"  ✅ Already exists")
    else:
        print(f"  ⚠️  Copy PRODUCTION_HARDEN.md to project root manually")

    # 3. Create branches
    print("\n[3/5] Creating hardening branches...")
    branches = [
        "harden/sys-types",
        "harden/sys-async",
        "harden/sys-errors",
        "harden/comfy-state",
        "harden/sys-security",
        "harden/comfy-testing",
        "harden/ops-observability",
        "harden/ops-cicd",
        "harden/vfx-docs",
        "harden/pipeline-config",
        "harden/comfy-mcp",
    ]

    # Make sure we're on main
    run(["git", "checkout", "main"], check=False)

    for branch in branches:
        result = run(["git", "branch", "--list", branch], check=False)
        if branch in result.stdout:
            print(f"  ⏭️  {branch} (already exists)")
        else:
            run(["git", "branch", branch], check=False)
            print(f"  ✅ {branch}")

    # Return to main
    run(["git", "checkout", "main"], check=False)

    # 4. Verify baseline
    print("\n[4/5] Running baseline verification...")
    result = run(["python", "-m", "pytest", "tests/", "-q", "--tb=short"], check=False)
    if result.returncode == 0:
        # Extract test count from output
        for line in result.stdout.splitlines():
            if "passed" in line:
                print(f"  ✅ Tests: {line.strip()}")
                break
    else:
        print(f"  ❌ Tests failed! Fix before starting hardening.")
        print(f"     {result.stdout[-300:]}")

    result = run(["ruff", "check", "agent/", "tests/"], check=False)
    if result.returncode == 0:
        print(f"  ✅ Lint: clean")
    else:
        print(f"  ❌ Lint errors found")

    # 5. Summary
    print("\n[5/5] Setup complete!")
    print("=" * 60)
    print("""
NEXT STEPS:

  # Quick start — run Phase 1 (foundation):
  python .claude/agents/orchestrate.py --phase 1 --dry-run   # Preview
  python .claude/agents/orchestrate.py --phase 1             # Execute

  # Or run individual workstreams:
  python .claude/agents/orchestrate.py --workstream WS-6     # Fix Pillow warnings first

  # Check status anytime:
  python .claude/agents/orchestrate.py --status

  # Or invoke agents directly via Claude Code:
  claude --model claude-sonnet-4-6-20250929 \\
    --system-prompt "$(cat .claude/agents/SYS_ENG.md)" \\
    --prompt "Execute WS-1 from PRODUCTION_HARDEN.md"

PHASE ORDER:
  Phase 1: Foundation (WS-1, WS-3, WS-6) — sequential, do first
  Phase 2: Core (WS-2, WS-4, WS-5) — parallel OK
  Phase 3: Infra (WS-7, WS-8, WS-10) — parallel OK
  Phase 4: Polish (WS-9, WS-11) — after Phase 2
""")


if __name__ == "__main__":
    main()
