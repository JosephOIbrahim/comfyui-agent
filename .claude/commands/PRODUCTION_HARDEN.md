# PRODUCTION HARDENING — MoE Agent Team Orchestration

> **Mission:** Autonomously production-harden ComfyUI-SUPER DUPER Agent using
> specialized agent teams routed by Mixture-of-Experts domain decomposition.
>
> **Runtime:** Claude Code with Sonnet 4.6 sub-agents
> **Target:** VFX workstation — Threadripper PRO 7965WX, RTX 4090, 128GB DDR5, Windows
> **Baseline:** 497 tests passing, 0 lint errors, 10,678 LOC agent / 6,863 LOC tests

---

## MoE TEAM ROSTER

Each agent is a Claude Code sub-agent invoked via `claude --model claude-sonnet-4-6-20250929`
with a role-specific prompt file. Agents operate on branches, submit PRs via conventional commits.

| Agent | Role | Domain | Branch Prefix |
|-------|------|--------|---------------|
| **VFX_SUPER** | VFX Supervisor | Artistic intent, UX, workflow correctness, naming | `harden/vfx-` |
| **COMFY_LEAD** | ComfyUI Team Lead | ComfyUI API contracts, node compatibility, MCP protocol | `harden/comfy-` |
| **NUKE_COMP** | Nuke Compositor | Pipeline integration, I/O patterns, cross-app interop | `harden/pipeline-` |
| **PRODUCER** | Producer | CI/CD, packaging, release, docs, metrics, scheduling | `harden/ops-` |
| **SYS_ENG** | Systems Engineer | Performance, concurrency, security, error handling, typing | `harden/sys-` |

### Routing Rules (MoE Gating)

```
Task mentions UI/UX/artist experience    → VFX_SUPER
Task mentions ComfyUI API/nodes/MCP      → COMFY_LEAD
Task mentions I/O/pipeline/integration   → NUKE_COMP
Task mentions CI/docs/packaging/release  → PRODUCER
Task mentions perf/security/types/async  → SYS_ENG
Task spans multiple domains              → Primary + Review by secondary
```

---

## AND-NODE TASK TREE

All branches must succeed. Hardest branches marked with ⚠️.

```
PRODUCTION HARDEN (AND — all required)
├── WS-1: Type Safety & Static Analysis (SYS_ENG) ⚠️ HARDEST
│   ├── 1.1 Add py.typed marker + pyright config
│   ├── 1.2 Type-annotate all public APIs (agent/tools/*.py)
│   ├── 1.3 Type-annotate brain layer (agent/brain/*.py)
│   ├── 1.4 Type-annotate core (config, circuit_breaker, rate_limiter, etc.)
│   ├── 1.5 Add pyright to CI (strict mode on new code, basic on existing)
│   └── 1.6 Fix all pyright errors at basic level
│
├── WS-2: Async Architecture Hardening (SYS_ENG + COMFY_LEAD review)
│   ├── 2.1 Audit all sync-in-async bridges (run_in_executor patterns)
│   ├── 2.2 Add proper async context managers for httpx clients
│   ├── 2.3 Fix MCP server graceful shutdown (signal handling)
│   ├── 2.4 Add connection pooling for ComfyUI HTTP calls
│   ├── 2.5 Add timeout configuration (per-tool configurable)
│   └── 2.6 Add async health check endpoint for MCP server
│
├── WS-3: Error Handling & Resilience (SYS_ENG + VFX_SUPER review)
│   ├── 3.1 Audit all exception handlers — no silent swallows
│   ├── 3.2 Add structured error types (ToolError, TransportError, ValidationError)
│   ├── 3.3 Add error context propagation (chain exceptions properly)
│   ├── 3.4 Add circuit breaker metrics (open/close counts, latency p50/p99)
│   ├── 3.5 Add retry budget (prevent infinite retry storms)
│   └── 3.6 Add graceful degradation paths (ComfyUI down → partial functionality)
│
├── WS-4: State Management Hardening (COMFY_LEAD + SYS_ENG review) ⚠️
│   ├── 4.1 Audit all module-level mutable state (6 _instance singletons, _state, _cache)
│   ├── 4.2 Add session-scoped state isolation (remove workflow_patch._state global)
│   ├── 4.3 Add session validation on load (schema versioning, migration)
│   ├── 4.4 Add atomic file writes for sessions (write-tmp + rename)
│   ├── 4.5 Add state cleanup / garbage collection for stale sessions
│   └── 4.6 Add memory limits for in-process caches (_cache in comfy_discover)
│
├── WS-5: Security Hardening (SYS_ENG + NUKE_COMP review)
│   ├── 5.1 Audit path traversal protection (validate_path coverage)
│   ├── 5.2 Add input validation for all MCP tool inputs (jsonschema strict)
│   ├── 5.3 Add rate limiting for MCP tool calls (prevent abuse)
│   ├── 5.4 Add secrets handling audit (.env, API keys, no hardcoded values)
│   ├── 5.5 Add dependency vulnerability scan (pip-audit in CI)
│   └── 5.6 Docker security review (non-root, no unnecessary capabilities)
│
├── WS-6: Testing Hardening (COMFY_LEAD + SYS_ENG)
│   ├── 6.1 Fix Pillow deprecation warnings (getdata → get_flattened_data)
│   ├── 6.2 Add property-based tests for patch engine (hypothesis)
│   ├── 6.3 Add integration test harness (requires live ComfyUI — mark as slow)
│   ├── 6.4 Add test coverage measurement + enforce minimum (pytest-cov)
│   ├── 6.5 Add mutation testing on critical paths (mutmut on workflow_patch)
│   ├── 6.6 Add fuzz testing for workflow JSON parsing
│   └── 6.7 Add MCP protocol conformance tests
│
├── WS-7: Observability & Monitoring (PRODUCER + SYS_ENG)
│   ├── 7.1 Add structured metrics (tool call counts, latencies, error rates)
│   ├── 7.2 Add tool call tracing (correlation ID → tool chain visualization)
│   ├── 7.3 Add performance baselines (benchmark suite for key operations)
│   ├── 7.4 Add memory profiling for long-running MCP sessions
│   ├── 7.5 Add log aggregation-friendly format (OpenTelemetry-compatible)
│   └── 7.6 Add GPU utilization monitoring hooks (RTX 4090 specific)
│
├── WS-8: CI/CD & Packaging (PRODUCER)
│   ├── 8.1 Add Python 3.13 to CI matrix
│   ├── 8.2 Add pyright/mypy check to CI
│   ├── 8.3 Add pip-audit (dependency security) to CI
│   ├── 8.4 Add test coverage gate to CI (fail below threshold)
│   ├── 8.5 Add docker-compose.yml for local dev (ComfyUI + agent)
│   ├── 8.6 Add release automation (version bump, changelog, PyPI publish)
│   ├── 8.7 Add pre-commit hooks config (.pre-commit-config.yaml)
│   └── 8.8 Add Windows-specific CI validation (paths, comfy_cli integration)
│
├── WS-9: Documentation & UX (VFX_SUPER + PRODUCER)
│   ├── 9.1 Add CONTRIBUTING.md (developer onboarding)
│   ├── 9.2 Add CHANGELOG.md (keepachangelog format)
│   ├── 9.3 Update README for production deployment
│   ├── 9.4 Add troubleshooting guide (common errors, ComfyUI connection issues)
│   ├── 9.5 Add MCP tool reference (auto-generated from TOOLS schemas)
│   ├── 9.6 Add architecture diagram (mermaid in docs/)
│   └── 9.7 Audit all user-facing error messages for artist-friendly language
│
├── WS-10: Platform & Config Hardening (NUKE_COMP + SYS_ENG)
│   ├── 10.1 Add config validation on startup (fail fast with clear errors)
│   ├── 10.2 Add platform-aware defaults (Windows/macOS/Linux path handling)
│   ├── 10.3 Add ComfyUI auto-discovery (find running instance, check common ports)
│   ├── 10.4 Add comfy_cli integration (leverage local ComfyUI management)
│   ├── 10.5 Add environment variable documentation (.env.example)
│   └── 10.6 Hardware profile auto-detection (GPU model, VRAM, for optimizer hints)
│
└── WS-11: MCP Protocol Hardening (COMFY_LEAD) ⚠️
    ├── 11.1 Add MCP resource support (expose workflow state as MCP resources)
    ├── 11.2 Add MCP prompt support (pre-built prompt templates)
    ├── 11.3 Add SSE transport option (for remote/web clients)
    ├── 11.4 Add MCP server versioning (protocol negotiation)
    ├── 11.5 Add tool schema validation (ensure all schemas are MCP-compliant)
    └── 11.6 Add MCP error code compliance (proper JSON-RPC error responses)
```

---

## EXECUTION PROTOCOL

### Phase 1: Foundation (WS-1, WS-3, WS-6.1) — Do First
Type safety and error handling are prerequisites for everything else.
Fix the Pillow warnings immediately (5 min task, removes noise).

### Phase 2: Core Hardening (WS-2, WS-4, WS-5) — Parallel
These three workstreams are independent AND-nodes. Run in parallel.

### Phase 3: Infrastructure (WS-7, WS-8, WS-10) — Parallel
CI/CD, observability, platform. Independent of each other.

### Phase 4: Polish (WS-9, WS-11, WS-6.2-6.7) — After Phase 2
Docs, MCP protocol, advanced testing. Depends on core being stable.

---

## EXECUTION COMMANDS

Each agent runs as a Claude Code sub-agent with a focused prompt.
Run from the `comfyui-agent/` project root.

### Orchestrator (you run this)
```bash
# Create all branches
git checkout -b harden/sys-types        # WS-1
git checkout -b harden/sys-async        # WS-2
git checkout -b harden/sys-errors       # WS-3
git checkout -b harden/comfy-state      # WS-4
git checkout -b harden/sys-security     # WS-5
git checkout -b harden/comfy-testing    # WS-6
git checkout -b harden/ops-observability # WS-7
git checkout -b harden/ops-cicd        # WS-8
git checkout -b harden/vfx-docs        # WS-9
git checkout -b harden/pipeline-config # WS-10
git checkout -b harden/comfy-mcp       # WS-11
git checkout main
```

### Agent Invocation Pattern
```bash
# Generic pattern — each agent gets its role prompt + workstream task
claude --model claude-sonnet-4-6-20250929 \
  --system-prompt "$(cat .claude/agents/SYS_ENG.md)" \
  --prompt "Execute workstream WS-1: Type Safety. See PRODUCTION_HARDEN.md for task breakdown. Branch: harden/sys-types. Run all tasks sequentially. Commit after each sub-task. Run tests after each commit. Zero test regressions allowed."

# Or use the orchestration script
python .claude/agents/orchestrate.py --workstream WS-1 --agent SYS_ENG
```

### Verification Gate (after each workstream)
```bash
# Every workstream must pass this gate before merge
python -m pytest tests/ -v                    # 497+ tests, 0 failures
ruff check agent/ tests/                       # 0 lint errors
ruff format --check agent/ tests/              # Formatting clean
# pyright agent/ (after WS-1 completes)       # Type check clean
```

---

## AGENT CONVENTIONS

### Commit Messages
```
[HARDEN:{WS}] {description}

Examples:
[HARDEN:WS-1] Add type annotations to workflow_patch.py public API
[HARDEN:WS-3] Add structured ToolError exception hierarchy
[HARDEN:WS-6] Fix Pillow getdata deprecation in vision.py
[HARDEN:WS-8] Add Python 3.13 and pyright to CI matrix
```

### PR Template
```
## Workstream: WS-{N} — {Title}
**Agent:** {ROLE}
**Branch:** harden/{prefix}-{name}

### Changes
- {bullet list of changes}

### Test Impact
- Tests before: 497
- Tests after: {N}
- New tests added: {N}
- Coverage delta: {+/-}%

### Verification
- [ ] All existing tests pass
- [ ] Lint clean
- [ ] No new warnings
- [ ] Reviewed by: {secondary agent}
```

### Rules (All Agents)
1. **NEVER break existing tests.** 497 tests must pass after every commit.
2. **NEVER change behavior.** Hardening means same behavior, better guarantees.
3. **Commit atomically.** One sub-task per commit. If a sub-task is too big, split it.
4. **Test first.** Write the test, watch it fail, then fix. Even for hardening.
5. **Document what you change.** Update CLAUDE.md if you change conventions.
6. **Respect the architecture.** The layer model is intentional. Don't reorganize.

---

## HARDWARE CONTEXT (for SYS_ENG + COMFY_LEAD)

```
Workstation Specs:
  CPU:    AMD Threadripper PRO 7965WX (32C/64T, 4.2GHz boost)
  GPU:    NVIDIA RTX 4090 (24GB VRAM, Ada Lovelace)
  RAM:    128GB DDR5 ECC
  OS:     Windows 11 Pro
  Python: 3.11+ (match ComfyUI)
  ComfyUI: Local install via comfy_cli at G:/COMFYUI_Database

Performance Context:
  - RTX 4090 is THE target GPU. Optimize TensorRT paths for Ada.
  - 128GB RAM means large model loading is not an issue.
  - 64 threads means ThreadPoolExecutor can use higher worker counts.
  - Windows paths use backslash. All path handling must be cross-platform.
  - ComfyUI runs locally — latency to API is <1ms.
```

---

## SUCCESS CRITERIA

The hardening is COMPLETE when:

1. ✅ All 11 workstreams merged to main
2. ✅ Test count ≥ 550 (net +53 minimum from new hardening tests)
3. ✅ Zero pyright errors at basic strictness
4. ✅ Zero Pillow/deprecation warnings
5. ✅ CI passes on Python 3.10, 3.11, 3.12, 3.13 × Ubuntu + Windows
6. ✅ MCP server passes protocol conformance tests
7. ✅ All user-facing errors use artist-friendly language
8. ✅ docker-compose up works for local dev
9. ✅ CHANGELOG.md and CONTRIBUTING.md exist
10. ✅ Pre-commit hooks configured and passing
