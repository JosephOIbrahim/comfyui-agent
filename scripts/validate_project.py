#!/usr/bin/env python3
"""Validate project consistency — catches version drift, count drift, and test health.

Run before any release or share:
    python scripts/validate_project.py
"""

import os
import re
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
errors = []
warnings = []


def check(label: str, ok: bool, detail: str = ""):
    if ok:
        print(f"  {PASS} {label}")
    else:
        msg = f"{label}: {detail}" if detail else label
        errors.append(msg)
        print(f"  {FAIL} {msg}")


def warn(label: str, detail: str = ""):
    msg = f"{label}: {detail}" if detail else label
    warnings.append(msg)
    print(f"  {WARN} {msg}")


# ---- 1. Version consistency ----
print("\n== Version Consistency ==")

pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
pyproject_ver = re.search(r'version\s*=\s*"([^"]+)"', pyproject)
pyproject_ver = pyproject_ver.group(1) if pyproject_ver else "NOT FOUND"

# Import __version__ without loading full agent (deps might be missing)
init_text = (PROJECT_ROOT / "agent" / "__init__.py").read_text(encoding="utf-8")
init_ver = re.search(r'__version__\s*=\s*"([^"]+)"', init_text)
init_ver = init_ver.group(1) if init_ver else "NOT FOUND"

check(
    f"pyproject.toml ({pyproject_ver}) == agent/__init__.py ({init_ver})",
    pyproject_ver == init_ver,
    f"Mismatch: pyproject={pyproject_ver}, __init__={init_ver}",
)

# ---- 2. Tool counts ----
print("\n== Tool Counts ==")

try:
    sys.path.insert(0, str(PROJECT_ROOT))
    from agent.tools import _LAYER_TOOLS
    from agent.brain import ALL_BRAIN_TOOLS

    intel_count = len(_LAYER_TOOLS)
    brain_count = len(ALL_BRAIN_TOOLS)
    total = intel_count + brain_count
    print(f"  Intelligence: {intel_count}, Brain: {brain_count}, Total: {total}")

    # Check README matches
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    readme_match = re.search(r"with (\d+) specialized tools", readme)
    if readme_match:
        readme_total = int(readme_match.group(1))
        check(
            f"README tool count ({readme_total}) == actual ({total})",
            readme_total == total,
            f"README says {readme_total}, code has {total}",
        )
    else:
        warn("Could not find tool count in README.md")

    # Check CLAUDE.md matches
    claude_md = (PROJECT_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    claude_match = re.search(r"with\s*\n?(\d+) specialized tools", claude_md)
    if claude_match:
        claude_total = int(claude_match.group(1))
        check(
            f"CLAUDE.md tool count ({claude_total}) == actual ({total})",
            claude_total == total,
            f"CLAUDE.md says {claude_total}, code has {total}",
        )

except ImportError as e:
    warn(f"Could not import agent tools: {e}")

# ---- 3. Model default consistency ----
print("\n== Model Default ==")

config_text = (PROJECT_ROOT / "agent" / "config.py").read_text(encoding="utf-8")
config_model = re.search(r'AGENT_MODEL\s*=\s*os\.getenv\("AGENT_MODEL",\s*"([^"]+)"\)', config_text)
config_model = config_model.group(1) if config_model else "NOT FOUND"

env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
env_model = re.search(r"#\s*AGENT_MODEL=([^\s]+)", env_example)
env_model = env_model.group(1) if env_model else "NOT FOUND"

check(
    f"config.py default ({config_model}) == .env.example ({env_model})",
    config_model == env_model,
    f"config={config_model}, env.example={env_model}",
)

# ---- 4. Lint ----
print("\n== Lint ==")

try:
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "agent/", "tests/"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    lint_ok = result.returncode == 0
    lint_count = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
    check(f"ruff check (0 errors)", lint_ok, f"{lint_count} lint issues found")
except FileNotFoundError:
    warn("ruff not installed -- skipping lint check")

# ---- 5. Quick test run ----
print("\n== Tests ==")

try:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no", "-x"],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120,
    )
    # Parse last line for pass/fail counts
    last_lines = result.stdout.strip().split("\n")[-3:]
    summary = " ".join(last_lines)
    passed = re.search(r"(\d+) passed", summary)
    failed = re.search(r"(\d+) failed", summary)
    skipped = re.search(r"(\d+) skipped", summary)

    p = int(passed.group(1)) if passed else 0
    f = int(failed.group(1)) if failed else 0
    s = int(skipped.group(1)) if skipped else 0

    check(f"Tests: {p} passed, {f} failed, {s} skipped", f == 0, f"{f} test(s) failed")
except subprocess.TimeoutExpired:
    warn("Tests timed out after 120s")
except FileNotFoundError:
    warn("pytest not installed — skipping tests")

# ---- 6. Required files ----
print("\n== Required Files ==")

for path in [
    "CLAUDE.md", "README.md", "LICENSE", "pyproject.toml",
    ".env.example", "agent/config.py", "agent/tools/__init__.py",
    "agent/brain/__init__.py", "agent/mcp_server.py",
]:
    check(f"{path} exists", (PROJECT_ROOT / path).exists())

# ---- Summary ----
print("\n" + "=" * 50)
if errors:
    print(f"\033[91m{len(errors)} error(s) found:\033[0m")
    for e in errors:
        print(f"  - {e}")
if warnings:
    print(f"\033[93m{len(warnings)} warning(s):\033[0m")
    for w in warnings:
        print(f"  - {w}")
if not errors and not warnings:
    print(f"\033[92mAll checks passed. Ready to share.\033[0m")
elif not errors:
    print(f"\033[93mNo errors, but review warnings above.\033[0m")

sys.exit(1 if errors else 0)
