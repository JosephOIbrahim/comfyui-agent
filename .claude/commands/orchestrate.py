#!/usr/bin/env python3
"""MoE Agent Team Orchestrator for Production Hardening.

Launches specialized Claude Code sub-agents to execute workstreams
in the production hardening plan. Each agent gets a role-specific
system prompt and a focused task description.

Usage:
    # Run a single workstream
    python orchestrate.py --workstream WS-1

    # Run Phase 1 (foundation — sequential)
    python orchestrate.py --phase 1

    # Run Phase 2 (core hardening — parallel)
    python orchestrate.py --phase 2 --parallel

    # Run all phases
    python orchestrate.py --all

    # Dry run (show what would execute)
    python orchestrate.py --phase 1 --dry-run

    # Status check
    python orchestrate.py --status
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# ─── MoE Routing Table ──────────────────────────────────────────────────────

WORKSTREAMS = {
    "WS-1": {
        "title": "Type Safety & Static Analysis",
        "agent": "SYS_ENG",
        "branch": "harden/sys-types",
        "phase": 1,
        "priority": "CRITICAL",
        "depends_on": [],
        "tasks": [
            "Add py.typed marker and pyright strict config to pyproject.toml",
            "Type-annotate all public APIs in agent/tools/*.py",
            "Type-annotate brain layer in agent/brain/*.py",
            "Type-annotate core modules (config, circuit_breaker, rate_limiter, logging_config, workflow_session)",
            "Add pyright to CI (basic mode on existing, strict on new)",
            "Fix all pyright errors at basic strictness level",
        ],
    },
    "WS-2": {
        "title": "Async Architecture Hardening",
        "agent": "SYS_ENG",
        "branch": "harden/sys-async",
        "phase": 2,
        "priority": "HIGH",
        "depends_on": [],
        "review_by": "COMFY_LEAD",
        "tasks": [
            "Audit all sync-in-async bridges (run_in_executor usage in mcp_server.py)",
            "Add async context managers for httpx clients in comfy_api.py",
            "Add graceful shutdown with signal handling to MCP server",
            "Add connection pooling for ComfyUI HTTP calls",
            "Add per-tool configurable timeouts",
            "Add async health check endpoint for MCP server",
        ],
    },
    "WS-3": {
        "title": "Error Handling & Resilience",
        "agent": "SYS_ENG",
        "branch": "harden/sys-errors",
        "phase": 1,
        "priority": "CRITICAL",
        "depends_on": [],
        "review_by": "VFX_SUPER",
        "tasks": [
            "Audit all exception handlers — no silent swallows, no bare except",
            "Add structured error types: ToolError, TransportError, ValidationError in agent/errors.py",
            "Add exception chaining (raise X from Y) throughout codebase",
            "Add circuit breaker metrics (open/close counts, latency tracking)",
            "Add retry budget to prevent infinite retry storms (max 10 retries per minute)",
            "Add graceful degradation: when ComfyUI is down, discovery/session tools still work",
        ],
    },
    "WS-4": {
        "title": "State Management Hardening",
        "agent": "COMFY_LEAD",
        "branch": "harden/comfy-state",
        "phase": 2,
        "priority": "HIGH",
        "depends_on": [],
        "review_by": "SYS_ENG",
        "tasks": [
            "Audit all module-level mutable state (workflow_patch._state, 6 brain _instance vars, _cache)",
            "Replace workflow_patch._state global with session-scoped state via WorkflowSession",
            "Add session schema validation on load with version migration support",
            "Add atomic file writes for all session persistence (write to .tmp then rename)",
            "Add session garbage collection (clean up sessions older than 30 days)",
            "Add LRU eviction for comfy_discover._cache (max 1000 entries or 50MB)",
        ],
    },
    "WS-5": {
        "title": "Security Hardening",
        "agent": "SYS_ENG",
        "branch": "harden/sys-security",
        "phase": 2,
        "priority": "HIGH",
        "depends_on": [],
        "review_by": "NUKE_COMP",
        "tasks": [
            "Audit validate_path() coverage — ensure ALL file operations go through it",
            "Add strict jsonschema validation for all MCP tool inputs",
            "Add rate limiting for MCP tool calls (100 calls/min default, configurable)",
            "Audit secrets handling: no hardcoded keys, no keys in logs, .env in .gitignore",
            "Add pip-audit to dependencies and CI for vulnerability scanning",
            "Review Dockerfile: confirm non-root, no unnecessary capabilities, minimal image",
        ],
    },
    "WS-6": {
        "title": "Testing Hardening",
        "agent": "COMFY_LEAD",
        "branch": "harden/comfy-testing",
        "phase": 1,  # 6.1 is Phase 1 (fix warnings), rest is Phase 4
        "priority": "HIGH",
        "depends_on": [],
        "review_by": "SYS_ENG",
        "tasks": [
            "Fix Pillow deprecation: replace getdata() with get_flattened_data() in vision.py",
            "Add hypothesis property-based tests for workflow_patch (valid patches always apply/undo cleanly)",
            "Add integration test harness with @pytest.mark.integration (skip by default, run with --integration)",
            "Add pytest-cov with 80% coverage gate in CI",
            "Add mutation testing on workflow_patch.py critical paths (mutmut)",
            "Add fuzz testing for workflow JSON parsing (malformed inputs don't crash)",
            "Add MCP protocol conformance tests (tool listing, tool calling, error responses)",
        ],
    },
    "WS-7": {
        "title": "Observability & Monitoring",
        "agent": "PRODUCER",
        "branch": "harden/ops-observability",
        "phase": 3,
        "priority": "MEDIUM",
        "depends_on": [],
        "review_by": "SYS_ENG",
        "tasks": [
            "Add agent/metrics.py with tool call counters, latency histograms, error rates",
            "Extend correlation_id for tool chain tracing (which tools called in sequence)",
            "Add pytest-benchmark baselines for: patch apply, discover search, workflow load",
            "Add memory profiling hooks for long-running MCP sessions",
            "Add OpenTelemetry-compatible log format option",
            "Add GPU utilization monitoring wrapper (nvidia-smi) for execution reporting",
        ],
    },
    "WS-8": {
        "title": "CI/CD & Packaging",
        "agent": "PRODUCER",
        "branch": "harden/ops-cicd",
        "phase": 3,
        "priority": "MEDIUM",
        "depends_on": [],
        "tasks": [
            "Add Python 3.13 to CI matrix",
            "Add pyright check to CI (after WS-1 provides annotations)",
            "Add pip-audit dependency security scan to CI",
            "Add pytest-cov with coverage gate to CI (fail below 80%)",
            "Add docker-compose.yml for local dev (ComfyUI + agent)",
            "Add release automation workflow (version bump + PyPI publish on tag)",
            "Add .pre-commit-config.yaml (ruff + ruff-format + quick pytest)",
            "Add Windows-specific CI validation (paths, line endings)",
        ],
    },
    "WS-9": {
        "title": "Documentation & UX",
        "agent": "VFX_SUPER",
        "branch": "harden/vfx-docs",
        "phase": 4,
        "priority": "MEDIUM",
        "depends_on": ["WS-3"],  # Error messages depend on error types
        "review_by": "PRODUCER",
        "tasks": [
            "Create CONTRIBUTING.md with developer onboarding guide",
            "Create CHANGELOG.md in keepachangelog format",
            "Update README.md for production deployment (artist-first language)",
            "Create docs/troubleshooting.md with common errors and plain-English solutions",
            "Create docs/tool-reference.md auto-generated from TOOLS schemas",
            "Create docs/architecture.md with mermaid diagrams",
            "Audit ALL error messages in codebase for artist-friendly language",
        ],
    },
    "WS-10": {
        "title": "Platform & Config Hardening",
        "agent": "NUKE_COMP",
        "branch": "harden/pipeline-config",
        "phase": 3,
        "priority": "MEDIUM",
        "depends_on": [],
        "review_by": "SYS_ENG",
        "tasks": [
            "Add config validation on startup (fail fast with clear, artist-friendly errors)",
            "Add platform-aware ComfyUI path defaults (Windows/macOS/Linux common locations)",
            "Add ComfyUI auto-discovery (check running instances, common ports, comfy_cli)",
            "Add comfy_cli integration (use `comfy env` for paths, suggest `comfy launch`)",
            "Create .env.example with documented configuration options",
            "Add hardware profile auto-detection (GPU model, VRAM via nvidia-smi)",
        ],
    },
    "WS-11": {
        "title": "MCP Protocol Hardening",
        "agent": "COMFY_LEAD",
        "branch": "harden/comfy-mcp",
        "phase": 4,
        "priority": "HIGH",
        "depends_on": ["WS-2", "WS-4"],  # Needs async + state fixes first
        "tasks": [
            "Add MCP resource support (expose workflow state, session list as resources)",
            "Add MCP prompt templates (pre-built artist prompts for common tasks)",
            "Add SSE transport option for remote/web MCP clients",
            "Add MCP server version negotiation",
            "Validate all tool schemas are MCP-compliant JSON Schema",
            "Add proper JSON-RPC error codes per MCP spec",
        ],
    },
}

# ─── Phase Definitions ───────────────────────────────────────────────────────

PHASES = {
    1: {
        "name": "Foundation",
        "description": "Type safety, error handling, deprecation fixes",
        "workstreams": ["WS-1", "WS-3", "WS-6"],
        "parallel": False,  # Sequential — WS-1 before WS-3
    },
    2: {
        "name": "Core Hardening",
        "description": "Async, state management, security",
        "workstreams": ["WS-2", "WS-4", "WS-5"],
        "parallel": True,
    },
    3: {
        "name": "Infrastructure",
        "description": "CI/CD, observability, platform",
        "workstreams": ["WS-7", "WS-8", "WS-10"],
        "parallel": True,
    },
    4: {
        "name": "Polish",
        "description": "Documentation, MCP protocol, advanced testing",
        "workstreams": ["WS-9", "WS-11"],
        "parallel": True,
    },
}

# ─── Agent Execution ─────────────────────────────────────────────────────────

AGENTS_DIR = Path(__file__).parent
PROJECT_ROOT = AGENTS_DIR.parent.parent  # .claude/agents/ → project root


def build_agent_prompt(ws_id: str) -> str:
    """Build the full prompt for a workstream agent invocation."""
    ws = WORKSTREAMS[ws_id]
    tasks_formatted = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(ws["tasks"]))

    return f"""Execute workstream {ws_id}: {ws['title']}

Branch: {ws['branch']}
Priority: {ws['priority']}

## Tasks (execute sequentially):
{tasks_formatted}

## Rules:
1. Create and checkout branch '{ws['branch']}' from main
2. Execute each task as a separate commit
3. Run `python -m pytest tests/ -q` after EVERY change — zero regressions allowed
4. Run `ruff check agent/ tests/` after every change — must be clean
5. Use commit format: [HARDEN:{ws_id}] {{description}}
6. If a task requires new tests, write the test FIRST
7. If a task is blocked, skip it and note why in a commit message
8. When all tasks complete, push the branch

## Verification gate (must pass before marking complete):
```bash
python -m pytest tests/ -v     # 497+ tests, 0 failures
ruff check agent/ tests/       # 0 errors
ruff format --check agent/ tests/  # Clean
```

GO. Execute all tasks now.
"""


def run_agent(ws_id: str, dry_run: bool = False) -> dict:
    """Run a Claude Code sub-agent for a workstream."""
    ws = WORKSTREAMS[ws_id]
    agent_file = AGENTS_DIR / f"{ws['agent']}.md"
    prompt = build_agent_prompt(ws_id)

    cmd = [
        "claude",
        "--model", "claude-sonnet-4-6-20250929",
        "--system-prompt", str(agent_file),
        "--print",
        "--prompt", prompt,
    ]

    result = {
        "workstream": ws_id,
        "agent": ws["agent"],
        "branch": ws["branch"],
        "started_at": datetime.now().isoformat(),
        "status": "pending",
    }

    if dry_run:
        print(f"\n{'='*60}")
        print(f"[DRY RUN] {ws_id}: {ws['title']}")
        print(f"  Agent:  {ws['agent']}")
        print(f"  Branch: {ws['branch']}")
        print(f"  Tasks:  {len(ws['tasks'])}")
        print(f"  Cmd:    {' '.join(cmd[:6])}...")
        result["status"] = "dry_run"
        return result

    print(f"\n{'='*60}")
    print(f"[EXECUTING] {ws_id}: {ws['title']}")
    print(f"  Agent:  {ws['agent']}")
    print(f"  Branch: {ws['branch']}")
    print(f"  Tasks:  {len(ws['tasks'])}")
    print(f"{'='*60}\n")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,  # Stream output to terminal
            text=True,
            timeout=1800,  # 30 min max per workstream
        )
        result["status"] = "completed" if proc.returncode == 0 else "failed"
        result["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
    except FileNotFoundError:
        result["status"] = "error"
        result["error"] = "claude CLI not found. Install Claude Code first."

    result["finished_at"] = datetime.now().isoformat()
    return result


def run_verification() -> bool:
    """Run the verification gate."""
    print("\n[VERIFICATION GATE]")

    checks = [
        ("pytest", ["python", "-m", "pytest", "tests/", "-q", "--tb=short"]),
        ("ruff check", ["ruff", "check", "agent/", "tests/"]),
        ("ruff format", ["ruff", "format", "--check", "agent/", "tests/"]),
    ]

    all_pass = True
    for name, cmd in checks:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        status = "✅" if proc.returncode == 0 else "❌"
        print(f"  {status} {name}")
        if proc.returncode != 0:
            all_pass = False
            print(f"     {proc.stdout[-200:] if proc.stdout else proc.stderr[-200:]}")

    return all_pass


def check_status():
    """Check which branches exist and their state."""
    print("\n[WORKSTREAM STATUS]")
    print(f"{'ID':<8} {'Title':<35} {'Agent':<12} {'Branch':<25} {'Exists'}")
    print("─" * 95)

    for ws_id, ws in WORKSTREAMS.items():
        # Check if branch exists
        proc = subprocess.run(
            ["git", "branch", "--list", ws["branch"]],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True,
        )
        exists = "✅" if ws["branch"] in proc.stdout else "⬜"
        print(f"{ws_id:<8} {ws['title']:<35} {ws['agent']:<12} {ws['branch']:<25} {exists}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MoE Agent Team Orchestrator")
    parser.add_argument("--workstream", "-w", help="Run a single workstream (e.g. WS-1)")
    parser.add_argument("--phase", "-p", type=int, help="Run all workstreams in a phase (1-4)")
    parser.add_argument("--all", action="store_true", help="Run all phases sequentially")
    parser.add_argument("--parallel", action="store_true", help="Run phase workstreams in parallel")
    parser.add_argument("--dry-run", action="store_true", help="Show what would execute")
    parser.add_argument("--status", action="store_true", help="Check workstream status")
    parser.add_argument("--verify", action="store_true", help="Run verification gate")

    args = parser.parse_args()

    if args.status:
        check_status()
        return

    if args.verify:
        success = run_verification()
        sys.exit(0 if success else 1)

    if args.workstream:
        if args.workstream not in WORKSTREAMS:
            print(f"Unknown workstream: {args.workstream}")
            print(f"Available: {', '.join(WORKSTREAMS.keys())}")
            sys.exit(1)
        result = run_agent(args.workstream, dry_run=args.dry_run)
        print(f"\nResult: {json.dumps(result, indent=2)}")

    elif args.phase:
        if args.phase not in PHASES:
            print(f"Unknown phase: {args.phase}. Available: 1, 2, 3, 4")
            sys.exit(1)
        phase = PHASES[args.phase]
        print(f"\n[PHASE {args.phase}] {phase['name']}: {phase['description']}")

        results = []
        for ws_id in phase["workstreams"]:
            result = run_agent(ws_id, dry_run=args.dry_run)
            results.append(result)
            if result["status"] == "failed" and not args.parallel:
                print(f"\n❌ {ws_id} failed. Stopping phase.")
                break

        print(f"\n[PHASE {args.phase} SUMMARY]")
        for r in results:
            status = "✅" if r["status"] in ("completed", "dry_run") else "❌"
            print(f"  {status} {r['workstream']}: {r['status']}")

    elif args.all:
        for phase_num in sorted(PHASES.keys()):
            phase = PHASES[phase_num]
            print(f"\n{'='*60}")
            print(f"[PHASE {phase_num}] {phase['name']}")
            print(f"{'='*60}")

            for ws_id in phase["workstreams"]:
                result = run_agent(ws_id, dry_run=args.dry_run)
                if result["status"] == "failed":
                    print(f"\n❌ {ws_id} failed. Stopping.")
                    sys.exit(1)

            if not args.dry_run:
                print(f"\n[PHASE {phase_num} VERIFICATION]")
                if not run_verification():
                    print(f"❌ Phase {phase_num} verification failed!")
                    sys.exit(1)
                print(f"✅ Phase {phase_num} complete and verified.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
