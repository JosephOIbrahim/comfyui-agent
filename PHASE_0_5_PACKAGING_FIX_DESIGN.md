# Phase 0.5 — Packaging Fix Design

**Agent:** `[GRAPH × ARCHITECT]`
**Date:** 2026-04-07
**Status:** Architect-only. No code written. Awaiting human gate before FORGE pass.
**Resolves:** Open Question §2 (src.cognitive import path) and Open Question §1 (pyproject.toml dependency gaps) from `MIGRATION_MAP_2026-04-07.md`.

---

## 1. Context and Goal

From `MIGRATION_MAP_2026-04-07.md` Open Question §2, verbatim:

> `pyproject.toml` ships only `agent/` in the wheel: `[tool.hatch.build.targets.wheel] packages = ["agent"]`. `src/cognitive/` and `panel/` are NOT in the build target. They are importable in tests only because pytest discovery puts the repo root on `sys.path`. Concrete consequences: `agent/tools/workflow_patch.py:44` does `from src.cognitive.core.graph import CognitiveGraphEngine` inside a try/except that silently swallows `ImportError`. Under `agent mcp` (the production CLI entry point) the engine import will fail at runtime because `src/` is not on the installed `sys.path`. **The PILOT engine wrapper that already exists on disk is dead code in production.**

**Goal:** Make `from cognitive.core.graph import CognitiveGraphEngine` resolve at runtime in any context where `comfyui-agent` is installed (editable or built wheel), eliminate the silent import-failure fallback in `workflow_patch.py`, and add `aiohttp` / `networkx` / `usd-core` to `pyproject.toml` in the same packaging pass — without touching any cognitive engine internals.

---

## 2. Investigation Findings

### 2a. `pyproject.toml` current state

Read in full. Relevant facts:

- **Build backend:** `hatchling` (no version pin in `[build-system] requires`). Whatever pip resolves at build time. Modern hatchling (≥1.18) supports `[tool.hatch.build.targets.wheel.sources]` mappings cleanly.
- **`requires-python = ">=3.10"`** — already set. No change needed.
- **`[project] dependencies`** (10 entries):
  ```toml
  "anthropic>=0.52.0", "jsonpatch>=1.33", "jsonschema>=4.20.0",
  "httpx>=0.27.0", "websockets>=12.0", "python-dotenv>=1.0.0",
  "typer>=0.12.0", "rich>=13.0.0", "mcp>=1.2.0", "pyyaml>=6.0",
  ```
  **Missing:** `aiohttp`, `networkx`, `usd-core`.
- **`[project.optional-dependencies] dev`** (4 entries): `pytest>=8.0`, `pytest-asyncio>=0.23`, `pytest-cov>=4.0`, `ruff>=0.3.0`. No other extras defined.
- **`[tool.hatch.build.targets.wheel]`**:
  ```toml
  packages = ["agent"]
  ```
  Single-package wheel. `panel/`, `src/cognitive/`, and `tests/` are excluded.
- **No existing `[tool.hatch.build.targets.wheel.sources]` block.** Greenfield.

### 2b. `workflow_patch.py` lines 41–48 — current try/except

```python
def _try_create_engine(workflow_data: dict):
    """Try to create a CognitiveGraphEngine. Returns None on failure."""
    try:
        from src.cognitive.core.graph import CognitiveGraphEngine
        return CognitiveGraphEngine(workflow_data)
    except (ImportError, Exception) as exc:
        log.debug("CognitiveGraphEngine not available: %s", exc)
        return None
```

Eight lines (41–48). The `except (ImportError, Exception)` clause is intentionally over-broad — catches both the missing-import case AND any constructor failure. Only `log.debug(...)` then `return None` — no operator-visible signal. This is the C3 silent-degradation violation.

### 2c. `from src.cognitive` / `import src.cognitive` — full blast radius

Greped across `agent/`, `src/`, `tests/`, and the rest of the repo. **31 import lines across 9 files** — substantially larger than the §2 inventory anticipated:

| File | Lines | Count |
|---|---|---|
| `agent/tools/workflow_patch.py` | 44 | 1 |
| `panel/server/routes.py` | 401 | **1 (NEW finding — not in §2)** |
| `tests/test_cognitive_core.py` | 13, 14, 15 | 3 |
| `tests/test_cognitive_tools.py` | 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 | 10 |
| `tests/test_cognitive_prediction.py` | 8, 9, 10, 104 | 4 |
| `tests/test_cognitive_transport.py` | 12, 13, 14 | 3 |
| `tests/test_cognitive_pipeline.py` | 9, 14, 15, 16, 17, 18 | 6 |
| `tests/test_cognitive_experience.py` | 11, 12, 13 | 3 |
| **Total** | | **31** |

`import src.cognitive` (without `from`): zero hits.

**Surprise:** `panel/server/routes.py:401` does `from src.cognitive.experience.accumulator import ExperienceAccumulator`. The §2 finding only inspected `agent/tools/workflow_patch.py`. The panel server has its own cognitive integration that suffers from the exact same dead-import problem. This is in scope for the fix because it's the same root cause.

### 2d. `from cognitive` / `import cognitive` (no `src.` prefix) in tests

```
$ grep -rn "^from cognitive\|^import cognitive" tests/
(no matches)
```

**Zero.** Joe's expectation in the prompt — that "tests/ files (probably none, since tests already use `cognitive.*`)" — is incorrect. **Every** test file uses the `src.cognitive` form. The blast radius for renaming all imports is the full 31 lines.

### 2e. `src/cognitive/__init__.py`

Exists. Real content:

```python
"""Cognitive architecture for ComfyUI Agent.

Provides non-destructive workflow mutation via LIVRPS composition.
"""

from .core.graph import CognitiveGraphEngine
from .core.delta import DeltaLayer, LIVRPS_PRIORITY, Opinion
from .core.models import ComfyNode, WorkflowGraph

__all__ = [
    "CognitiveGraphEngine", "ComfyNode", "DeltaLayer",
    "LIVRPS_PRIORITY", "Opinion", "WorkflowGraph",
]
```

After the packaging fix, `import cognitive` will work and re-export the engine at the top level. Good shape — no edits needed.

### 2f. `src/__init__.py`

**Exists. Empty (0 bytes). Created Apr 4 14:04.**

```
$ ls -la G:/Comfy-Cozy/src/__init__.py
-rw-r--r-- 1 User 197121 0 Apr  4 14:04 G:/Comfy-Cozy/src/__init__.py
```

This is the wart that makes the current `from src.cognitive...` form work under pytest's rootdir injection: `src` is itself an importable package (because of the empty `__init__.py`), so `src.cognitive` resolves as a dotted import. **For hatch's `src`-layout sources mapping to behave correctly, this file must be deleted.** A `src` directory under hatch's src-layout convention is a stripping point, not a Python package — it must NOT contain `__init__.py`. Leaving it in place would cause the wheel to ship a dead `src` package that conflicts with the stripped layout.

This deletion is a small additional file mutation in the FORGE pass, flagged here so it doesn't get missed.

### 2g. `if engine is not None` / `if engine is None` checks in workflow_patch.py

9 hits, lines: **67, 414, 438, 502, 637, 699, 709, 762, 772**. Spot-checked line 637 (in `_handle_add_node`) and line 699 (in `_handle_connect_nodes`):

```python
# line 637 — _handle_add_node
engine = _get_engine()
if engine is not None:
    mutations = {node_id: {"class_type": class_type, **inputs}}
    engine.mutate_workflow(mutations, opinion="L", description=...)
    _sync_state_from_engine()
else:
    _state["current_workflow"][node_id] = {"class_type": class_type, "inputs": inputs}
```

These are **not** simple import-availability guards. They're a dual-codepath dispatch: engine path mutates via the LIVRPS delta stack, fallback path mutates `_state["current_workflow"]` directly via dict writes. The fallback codepath was the **original** implementation; the engine path was bolted on top with the silent fallback so the engine integration could be added without regressing existing tests.

**Implication for the fix:** killing `_try_create_engine`'s ImportError swallowing means import failure becomes a hard module-load error (correct per C3). But `_get_engine()` can still return `None` if the engine is never instantiated for a given session (e.g., a workflow was loaded via a path that doesn't call `_try_create_engine`). The `else:` branches are not pure dead code — they're the path taken when no engine has been wired into `_state` yet. Reducing them is a separate refactor and **out of scope** for this surgical fix.

What IS in scope: the import itself moves to module top-of-file, the try/except disappears, `_try_create_engine` becomes `_create_engine` (no fallback to `None` on import error — the import is no longer inside the function), and any constructor failure on bad workflow data still raises so the operator sees it.

---

## 3. Proposed `pyproject.toml` Changes

### 3.1 Add cognitive to the wheel packages

Update the existing `[tool.hatch.build.targets.wheel]` block:

```toml
# Before
[tool.hatch.build.targets.wheel]
packages = ["agent"]

# After
[tool.hatch.build.targets.wheel]
packages = ["agent", "src/cognitive"]
```

**Why:** Modern hatchling (≥1.13) accepts a path like `src/cognitive` in `packages` and installs it as the top-level package `cognitive` automatically — it strips the leading directory components and uses the basename as the install name. This is the simplest one-line packaging change that ships `cognitive/` as a real top-level package without needing a separate sources mapping.

### 3.2 Explicit src-layout sources mapping (defensive — see §9)

If §3.1 alone produces an unexpected layout (e.g., older hatchling installs `cognitive` as `src.cognitive` rather than `cognitive`), add:

```toml
[tool.hatch.build.targets.wheel.sources]
"src/cognitive" = "cognitive"
```

**Why:** Explicit "install path X under name Y" mapping is the canonical way to express src-layout intent in hatch. The form in §3.1 is shorthand that *should* infer the same result, but I cannot verify this without running a test build (out of scope for the architect pass — see §9 Open Question A). The FORGE pass should run `pip install -e .` after §3.1 alone first, verify `python -c "from cognitive.core.graph import CognitiveGraphEngine"` works, and only fall back to adding §3.2 if it doesn't.

### 3.3 Add missing runtime dependencies

Update `[project] dependencies`:

```toml
# Before
dependencies = [
    "anthropic>=0.52.0",
    "jsonpatch>=1.33",
    "jsonschema>=4.20.0",
    "httpx>=0.27.0",
    "websockets>=12.0",
    "python-dotenv>=1.0.0",
    "typer>=0.12.0",
    "rich>=13.0.0",
    "mcp>=1.2.0",
    "pyyaml>=6.0",
]

# After
dependencies = [
    "anthropic>=0.52.0",
    "jsonpatch>=1.33",
    "jsonschema>=4.20.0",
    "httpx>=0.27.0",
    "websockets>=12.0",
    "python-dotenv>=1.0.0",
    "typer>=0.12.0",
    "rich>=13.0.0",
    "mcp>=1.2.0",
    "pyyaml>=6.0",
    "aiohttp>=3.9.0",       # panel/server/{middleware,routes,chat}.py — aiohttp.web
    "networkx>=3.0",        # tests/test_dag_engine.py — DAG validation
    "usd-core>=24.0",       # agent/stage/cognitive_stage.py — USD prim model
]
```

**Why each:**

- **`aiohttp>=3.9.0`** — `panel/server/middleware.py:6`, `panel/server/routes.py:10`, `panel/server/chat.py:15` all do `from aiohttp import web`. The 3.9 floor matches the version installed in this Phase 0 session (`aiohttp 3.13.5`) minus several minor versions for compatibility. aiohttp's `web` API is stable across the 3.x line — no API risk.
- **`networkx>=3.0`** — `tests/test_dag_engine.py` (15 tests) does `import networkx as nx` for graph-property assertions. The agent's own DAG engine at `agent/stage/dag/engine.py` may also use it; if not, the dep can stay as a test-only import. **Note:** I have not greped `agent/stage/dag/` to confirm whether networkx is a runtime or test-only dep. If test-only, the more correct placement is `[project.optional-dependencies] dev`. Flagged in §9 Open Question B.
- **`usd-core>=24.0`** — `agent/stage/cognitive_stage.py:91` raises `StageError("USD not available. Install with: pip install usd-core")`. usd-core is a non-trivial dep (large binary, NVIDIA OpenUSD bindings, Windows-tested but heavy on disk). Pin floor 24.0 because that's the first usd-core release that ships pre-built Windows wheels for Python 3.12. **Note:** I have not verified that 24.0 is actually the right floor — I'm reasoning from general usd-core release patterns. The FORGE pass should `pip install usd-core` first and read the resolved version, then pin that as the floor. Flagged in §9 Open Question C.

### 3.4 Optional: dependency grouping into extras

I am **not** recommending this in §3.3 as the primary path, but flagging it for Joe's consideration. An alternative shape:

```toml
[project.optional-dependencies]
dev = [...]  # unchanged
panel = ["aiohttp>=3.9.0"]
dag = ["networkx>=3.0"]
stage = ["usd-core>=24.0"]
all = ["comfyui-agent[panel,dag,stage,dev]"]
```

Then `pip install -e ".[all]"` for full dev. The advantage: usd-core in particular is heavy and might be unwanted on systems that don't run the stage layer. The disadvantage: more install-command surface area to remember, and the panel/dag/stage layers appear to be normal parts of the product, not optional plugins. **My recommendation: §3.3 (treat all three as core).** If Joe wants the extras grouping, the FORGE pass can shape it that way instead — same total work, different layout.

---

## 4. Proposed `workflow_patch.py` Changes

### 4.1 The minimal surgical change

**Lines 41–48 — replace the entire `_try_create_engine` function:**

```python
# Before (lines 41-48)
def _try_create_engine(workflow_data: dict):
    """Try to create a CognitiveGraphEngine. Returns None on failure."""
    try:
        from src.cognitive.core.graph import CognitiveGraphEngine
        return CognitiveGraphEngine(workflow_data)
    except (ImportError, Exception) as exc:
        log.debug("CognitiveGraphEngine not available: %s", exc)
        return None
```

```python
# After
def _create_engine(workflow_data: dict) -> CognitiveGraphEngine:
    """Create a CognitiveGraphEngine for the given workflow."""
    return CognitiveGraphEngine(workflow_data)
```

**Lines 17–28 — add the import to the existing top-of-file import block:**

```python
# Before — partial top-of-file (line 17 area)
import jsonpatch

from ._util import to_json
from ..workflow_session import get_session
```

```python
# After
import jsonpatch

from cognitive.core.graph import CognitiveGraphEngine

from ._util import to_json
from ..workflow_session import get_session
```

**Rename callers** — three call sites for `_try_create_engine` need updating to `_create_engine`:
- Line 121: `_set_engine(_try_create_engine(api_nodes))` (inside `_load_workflow`)
- Line 439: `_set_engine(_try_create_engine(_state["current_workflow"]))` (inside `_handle_apply_patch` fallback rebuild)
- Line 508: `_set_engine(_try_create_engine(_state["current_workflow"]))` (inside `_handle_undo` fallback rebuild)
- Line 593: `_set_engine(_try_create_engine(_state["base_workflow"]))` (inside `_handle_reset`)

(Four call sites — I missed one in my initial line scan. Verified by re-reading.)

### 4.2 What the 9 `if engine is not None` branches do after the fix

**Nothing changes for them in this surgical pass.** The branches dispatch between the engine codepath and the legacy direct-dict-mutation codepath. After the fix:

- The import itself can no longer silently fail — if `cognitive` doesn't install, `workflow_patch.py` raises `ImportError` at module load time and the entire `agent mcp` server fails to start. **This is correct C3 behavior** and exactly what Joe asked for in instruction #5.
- `_create_engine(workflow_data)` can still raise on bad input (e.g., a workflow dict that fails `WorkflowGraph.from_api_json` validation). The 4 callers all currently wrap the result in `_set_engine(...)` without handling exceptions — meaning a constructor failure already propagates to the caller of the surrounding `_handle_*` function. The dispatcher at line 797 catches generic exceptions for tool-handler wrapping. **No regression** — runtime constructor failures behave identically to today, the only thing that changes is that import failures stop being silent.
- Sessions where no workflow has been loaded yet still have `_state["_engine"] = None` (the session container's default). The `if engine is not None` branches still need to handle this case — they're not dead code. They become "engine is wired vs engine is not yet wired", not "engine is available vs engine import failed".

**Result: 9 conditional branches stay as-is.** Removing them is a separate refactor (eliminate the legacy direct-mutation codepath entirely, force-load an engine even for empty workflows) and is **out of scope** for this Phase 0.5 fix.

### 4.3 What about `panel/server/routes.py:401`?

```python
# panel/server/routes.py:401 — current
from src.cognitive.experience.accumulator import ExperienceAccumulator
```

```python
# panel/server/routes.py:401 — after
from cognitive.experience.accumulator import ExperienceAccumulator
```

One line. No try/except wraps it (verified earlier — only `workflow_patch.py:44` had the silent fallback). Same root-cause fix, same change shape, included in scope.

---

## 5. Blast Radius Analysis

**Total files touched by the FORGE pass: 11**

| Group | File | Lines changed | Change type |
|---|---|---|---|
| Packaging | `pyproject.toml` | ~5 | Add 3 deps + 1 line in wheel packages |
| Source — agent | `agent/tools/workflow_patch.py` | ~12 (function rewrite + 4 caller renames + 1 new top-of-file import) | Surgical |
| Source — panel | `panel/server/routes.py:401` | 1 | Drop `src.` prefix |
| Source — wart | `src/__init__.py` | DELETE FILE | Empty file removal (0 bytes) |
| Test imports | `tests/test_cognitive_core.py` | 3 (lines 13, 14, 15) | Drop `src.` prefix |
| Test imports | `tests/test_cognitive_tools.py` | 10 (lines 8–17) | Drop `src.` prefix |
| Test imports | `tests/test_cognitive_prediction.py` | 4 (lines 8, 9, 10, 104) | Drop `src.` prefix |
| Test imports | `tests/test_cognitive_transport.py` | 3 (lines 12, 13, 14) | Drop `src.` prefix |
| Test imports | `tests/test_cognitive_pipeline.py` | 6 (lines 9, 14, 15, 16, 17, 18) | Drop `src.` prefix |
| Test imports | `tests/test_cognitive_experience.py` | 3 (lines 11, 12, 13) | Drop `src.` prefix |

**Total line changes: ~50, mechanical (a single sed-style find-and-replace `from src.cognitive` → `from cognitive` covers 30 of them).** Plus one file deletion (`src/__init__.py`) and one packaging file edit.

**Mechanical-fix recipe for the FORGE pass** (NOT to be executed in this architect pass):

```bash
# 1. Update pyproject.toml per §3.1 and §3.3
# 2. Delete src/__init__.py
# 3. Mass rename:
grep -rln "from src\.cognitive" agent/ panel/ tests/ \
  | xargs sed -i.bak 's/from src\.cognitive/from cognitive/g'
# 4. Hand-edit agent/tools/workflow_patch.py per §4.1 (the function rewrite is not sed-able)
# 5. Reinstall: pip install -e .
# 6. Verify per §6
```

The sed-able portion is 30/31 import lines. The 1 non-sed-able change is the `_try_create_engine` → `_create_engine` rewrite plus moving the import to the top of the file.

---

## 6. Verification Plan

### 6a. Runtime import from a fresh shell (the §2 acceptance test)

```bash
cd /g/Comfy-Cozy
source .venv312/Scripts/activate
python -c "from cognitive.core.graph import CognitiveGraphEngine; print('OK:', CognitiveGraphEngine.__module__)"
```

**Expected:** `OK: cognitive.core.graph`. No `ModuleNotFoundError`.

This is the exact inverse of the failure I observed in Phase 0:
```
$ python -c "from cognitive.core import graph"
ModuleNotFoundError: No module named 'cognitive'
```

If this command prints `OK:` after the FORGE pass, §2 is resolved.

### 6b. Full pytest suite — invariant check

```bash
python -m pytest tests/ --tb=no -q
```

**Expected outcomes:**

- **Best case:** `2655 + 19 + (some of the 27 errors) passed`. Adding `networkx` resolves all 15 `test_dag_engine.py` failures. Adding `usd-core` resolves all 27 `test_provisioner.py` collection errors (assuming the only blocker was the import — the tests themselves might still fail on USD-specific issues, in which case they convert from collection errors to runnable tests, some of which pass). The 4 `test_health.py` failures are out of scope and remain.
- **Acceptance threshold:** **passing count ≥ 2655**. The invariant must not regress. Any new failures introduced by the import path rename are regressions and must be fixed before the FORGE pass completes.
- **Specific must-pass:** `tests/test_cognitive_core.py` — all 54 tests still green. This is the canary that proves the cognitive engine itself wasn't disturbed.

### 6c. MCP server runtime acceptance test for §2

This is the regression test Joe asked me to design — the adversarial check that catches runtime-vs-pytest divergence going forward.

**Chosen approach: a standalone import-and-instantiate script in `tests/test_workflow_patch_engine_live.py`.**

Rationale: the simplest verification is the one that exercises the exact import path the production MCP server uses. Adding a `/health` endpoint is a feature change with its own design surface; reading agent logs at startup is brittle (log levels, log routing). A pytest test that imports `agent.tools.workflow_patch` from a fresh interpreter and asserts the engine instantiates is the smallest possible thing that fails loudly when §2 regresses.

Test file content (specification only — FORGE writes the actual file):

```python
"""Regression test for Phase 0.5 packaging fix.

Asserts that the CognitiveGraphEngine is reachable through the
production import path used by `agent mcp` — not just the test-only
`from src.cognitive...` form that pytest's rootdir injection enables.

If this test fails, the packaging fix has regressed and the MCP
server's PILOT engine is silently dead in production.
"""

def test_workflow_patch_imports_engine_from_top_level():
    """workflow_patch must import CognitiveGraphEngine via `cognitive`,
    not via `src.cognitive`."""
    import agent.tools.workflow_patch as wp

    # The engine class should be a real class, not None.
    assert wp.CognitiveGraphEngine is not None
    assert wp.CognitiveGraphEngine.__module__ == "cognitive.core.graph"


def test_create_engine_instantiates_against_minimal_workflow():
    """_create_engine should produce a real engine for a 1-node workflow."""
    from agent.tools.workflow_patch import _create_engine

    minimal = {"1": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}}}
    engine = _create_engine(minimal)

    assert engine is not None
    assert engine.__class__.__name__ == "CognitiveGraphEngine"
    assert engine.delta_stack == []
    assert engine.to_api_json() == minimal


def test_no_legacy_src_cognitive_imports_remain():
    """No source file should import from `src.cognitive`."""
    import subprocess
    result = subprocess.run(
        ["grep", "-rln", "from src.cognitive", "agent/", "panel/", "src/"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1, (
        f"Legacy `from src.cognitive` imports still present:\n{result.stdout}"
    )
```

Three tests, one file. Test 1 catches the case where `workflow_patch.py` fails to import the engine at module load (the §2 bug). Test 2 catches a constructor regression. Test 3 is the adversarial sweep that prevents anyone from accidentally adding a new `from src.cognitive` import in the future. Test 3 explicitly excludes `tests/` because the rename of test files is part of this fix and there's no reason a new test file can't use the post-fix `from cognitive...` form.

The third test is the C7 "prevent the same bug twice" guardrail.

---

## 7. Rollback Plan

If the FORGE pass breaks the baseline and three retries don't recover it: revert `pyproject.toml`, `agent/tools/workflow_patch.py`, `panel/server/routes.py`, all 6 affected `tests/test_cognitive_*.py` files, and restore `src/__init__.py` (re-create as empty file). Run `pip install -e .` to re-establish the pre-fix editable install state. Re-run pytest to confirm the 2655-passing baseline is restored. File a `BLOCKER.md` per the circuit breaker protocol with the three attempted approaches and their failure modes.

The fix touches 11 files and one packaging config — all small, mechanical edits. There is no data loss risk because the cognitive engine internals (`graph.py`, `delta.py`, `models.py`) are explicitly out of scope and never touched. The `.venv312` itself can be rebuilt from scratch in under 60 seconds (`rm -rf .venv312 && py -3.12 -m venv .venv312 && pip install -e .`) if the editable install state becomes corrupted.

---

## 8. Out of Scope (Explicit)

This fix does NOT address:

- Open Question §3 — `agent/stage/` ↔ `src/cognitive/` overlap. The 9 conditional branches in `workflow_patch.py` that dispatch between engine and legacy paths stay in place. Eliminating the legacy codepath is a separate refactor.
- Open Question §4 — `agent/brain/memory.py` single-file shape. Not touched.
- Open Question §5 — the 4 `test_health.py` failures. Not touched. They remain in the post-fix baseline.
- Open Question §6 — Phase 1–6 completeness assessment. Not investigated. After this fix lands, Phase 1 Step 4 (PILOT wrappers) is finally effective at runtime — but Phases 2–6 status is still unknown and a separate `[X × SCOUT]` pass is needed.
- Cognitive engine internals — `src/cognitive/core/{graph,delta,models}.py` are not touched. The 54 `test_cognitive_core.py` tests prove they're correct.
- Phase 2 transport (`src/cognitive/transport/`) — files exist on disk but completeness not verified. Out of scope.
- Phase 3–6 modules — same.
- The `agent/stage/` heavy modules (`hyperagent.py`, `autoresearch_runner.py`, `compositor.py`, etc.) — not touched.
- ANY engine internal changes of any kind.

---

## 9. Open Questions for Joe

### A. Hatchling `packages = ["src/cognitive"]` shorthand vs explicit sources mapping

I cannot verify without running a test build whether modern hatchling installs `packages = ["agent", "src/cognitive"]` as `cognitive` (the post-strip basename) or as `src.cognitive` (the literal dotted form). Hatchling docs claim the basename behavior for >=1.13, and this is what I'm proposing in §3.1. **The FORGE pass must verify this with `pip install -e .` followed by `python -c "from cognitive.core.graph import CognitiveGraphEngine"`** as the first acceptance check after editing pyproject.toml. If the import fails, fall back to the explicit sources mapping in §3.2. This adds a small branch to the FORGE pass but does not change the design.

**Question for Joe:** OK to let the FORGE pass try shorthand first and fall back to explicit mapping if needed, or do you want the explicit mapping baked in from the start to remove the branch?

### B. `networkx` placement — runtime dep vs dev/test dep

`networkx` is currently imported only by test code (`tests/test_dag_engine.py`). I have not greped `agent/stage/dag/` to confirm whether the production DAG engine uses networkx at runtime. If it doesn't, the more correct placement is `[project.optional-dependencies] dev` rather than `[project] dependencies`, which keeps the runtime install slim. If it does, the placement in `dependencies` is correct.

**Question for Joe:** Should I (the FORGE pass) grep `agent/stage/dag/*.py` for `import networkx` first and decide based on what's there, or do you want it in core deps regardless for safety?

### C. `usd-core` version pin and platform availability

I proposed `usd-core>=24.0` based on general release patterns, but I have not verified this is the right floor for Python 3.12 / Windows wheels. usd-core is published by NVIDIA and historically lagged on Windows wheel releases. The FORGE pass should `pip install usd-core` first, read the resolved version, and pin to *that* version's minor floor (e.g., if pip resolves `24.8`, pin `>=24.8`). usd-core is also a large download (~200MB+) — adding it as a hard runtime dep makes a fresh install heavier.

**Question for Joe:** Three choices —
1. Hard runtime dep (every install pays the 200MB cost — recommended if `agent/stage/cognitive_stage.py` is loaded on every startup).
2. Optional `[stage]` extra (operators who want USD opt in; the 27 `test_provisioner.py` errors stay as baseline until someone runs `pip install -e ".[stage]"`).
3. Hold the line — keep `usd-core` out of pyproject for now, accept that `test_provisioner.py` stays at 27 errors, address in a later cleanup phase.

My recommendation: **option 2 (`[stage]` extra)** — usd-core is heavy enough that it shouldn't be in the default install path, and the tests that need it are isolated to one file. But I'll defer to your call.

### D. Test file rename — should the cognitive test files keep using `from src.cognitive...` or move to `from cognitive...`?

After the packaging fix, both forms will work (because `src/cognitive/` will be installed as `cognitive`, AND pytest's rootdir injection will still make `from src.cognitive...` resolve via the directory tree — though without `src/__init__.py` deleted, this second path actually breaks). To be precise: if §2f's `src/__init__.py` deletion lands, `src` is no longer a package, and `from src.cognitive...` will fail. Therefore the test file rename in §5 is **mandatory** to keep the test suite green, not optional.

**Question for Joe:** The §5 blast radius shows ~30 test-file edits as part of this fix. That's larger than the "minimal surgical" framing. I think it's still in scope because the fix can't land without it (the alternative is leaving `src/__init__.py` in place, which leaves the dead `src` package shipping in the wheel — undesirable). Confirm this is acceptable, or tell me you'd rather keep `src/__init__.py` and accept the dead package as the lesser cost.

### E. The new finding in `panel/server/routes.py`

Joe's Phase 0 §2 inventory only mentioned `agent/tools/workflow_patch.py`. I found a second `from src.cognitive...` import in `panel/server/routes.py:401` that has the same problem. It's included in the FORGE scope per §4.3 and §5. **Confirm this is acceptable scope creep** (it's the same bug, fixed once for both call sites) — or if you'd rather split it into a follow-up fix targeting the panel layer specifically.

---

## 10. Phase 0.5 Gate

This document is the Phase 0.5 architect deliverable. No code has been written. No files have been modified. No git operations have been performed. The investigation in §2 was read-only (greps and file reads).

**STOP. Awaiting Joe's review and approval before the FORGE pass.**

The FORGE pass, once approved, will execute the changes in §3, §4, §5 in the order specified by §5's mechanical recipe, then run §6's three-step verification, then commit. If any step fails, the circuit breaker protocol from `SCAFFOLDED_BRAIN_PLAN.md` applies (3 attempts then `BLOCKER.md`). Rollback per §7 if needed.

---

## 11. Open Question Resolutions (2026-04-07)

> **Numbering note:** Joe's authorization message specified this addendum as "## 10". The existing §10 in this document is "Phase 0.5 Gate". To preserve the gate as the semantic end of the architect doc and append the resolutions as the literal bottom section per Joe's intent, this is renumbered to §11. No content change.

The five Open Questions in §9 were resolved by Joe on 2026-04-07 as follows. These resolutions are part of the approved spec and override any tentative recommendation in §9.

### §9A — Hatchling sources mapping

**Resolution: Use the EXPLICIT `[tool.hatch.build.targets.wheel.sources]` mapping as the primary approach, not the shorthand. Two lines of stable config beats one line of version-dependent shorthand. If the explicit mapping fails at build time, that's a real blocker and you stop — do not retry with the shorthand.**

**Implication for FORGE:** §3.1 is no longer the primary path. Use §3.2 from the start. The `[tool.hatch.build.targets.wheel]` block becomes:

```toml
[tool.hatch.build.targets.wheel]
packages = ["agent", "cognitive"]

[tool.hatch.build.targets.wheel.sources]
"src/cognitive" = "cognitive"
```

If `pip install -e .` fails to resolve `cognitive` after this edit, the FORGE pass STOPS and files a `BLOCKER.md`. No fallback to the shorthand form.

### §9B — `networkx` placement

**Resolution: Core `[project] dependencies`. Not a test extra. Rationale: `agent/stage/` DAG code imports networkx in production, not just in tests. A user installing the wheel needs it to import `agent.stage` at all. Treating it as a test dep would ship a broken wheel.**

**Implication for FORGE:** §3.3's proposed `dependencies` list is unchanged for `networkx`. Keep `"networkx>=3.0",` in core dependencies. Do NOT grep `agent/stage/dag/` first to second-guess the placement — the answer is core deps regardless.

### §9C — `usd-core` placement

**Resolution: Create a new `[project.optional-dependencies]` group called `stage` and put usd-core there. Not core dependencies.**

**Rationale (Joe):** (a) sidesteps the 25% Windows 3.12 wheel risk flagged in §9C; (b) the 27 pre-existing `test_provisioner.py` errors remain as baseline known-issue and don't become a blocker for this session; (c) not every consumer of this package needs the provisioner subsystem; (d) matches the loosely-coupled structure of cognitive / agent / panel / stage.

**Implication for FORGE — replaces the §3.3 line for usd-core:**

The `[project] dependencies` list adds only `aiohttp` and `networkx`. Remove `usd-core` from that list and instead add a new `[project.optional-dependencies]` group:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=4.0",
    "ruff>=0.3.0",
]
stage = [
    "usd-core>=24.0",  # agent/stage/cognitive_stage.py — USD prim model
]
```

**Developer install commands:**
- Standard install (no stage subsystem): `pip install -e ".[dev]"`
- With stage subsystem: `pip install -e ".[dev,stage]"`

**Documentation note for future readers:** `usd-core` is separated into the `[stage]` extra because it is a heavy native dependency (~200MB+) with historically spotty Windows wheel support. Not all consumers of `comfyui-agent` need the stage subsystem. The 27 pre-existing `test_provisioner.py` collection errors are tracked as a known baseline issue and will remain as collection errors for any developer who installs without the `stage` extra. This is intentional and acceptable — the agent core, cognitive layer, and panel all function without USD.

**Implication for §6b verification:** the post-fix baseline comparison is now:
- Default `[dev]` install: still 27 `test_provisioner.py` collection errors (unchanged from baseline). Passing count must be ≥ 2655 + 15 (the networkx-resolved tests) = **≥ 2670**.
- `[dev,stage]` install: usd-core now installable, the 27 errors should convert to either passing tests or test-level failures (which are tracked separately, not regressions). The FORGE pass should run pytest under both install modes if usd-core resolves, but the **acceptance baseline is the `[dev]`-only install**.

### §9D — Test file rename scope

**Resolution: In scope. Do the full rename across all 6 cognitive test files. Rationale: once `src/__init__.py` is deleted, `src.cognitive.*` imports break everywhere, including in tests. There is no version of this fix that preserves the 2655-passing baseline without the test rename. It is mechanical sed work. Approved.**

**Implication for FORGE:** §5's blast-radius table stands as written. All 30 test-file import lines are renamed via the sed recipe in §5. No re-design needed.

### §9E — `panel/server/routes.py:401` rename

**Resolution: In scope. Do the one-line rename. Rationale: same as §9D. That import breaks the moment `src/__init__.py` is deleted, which would cascade into broken panel test collection. One line, cheap, required.**

**Implication for FORGE:** §4.3 stands as written. The single line at `panel/server/routes.py:401` is renamed alongside the test files. The sed recipe in §5 already covers it via the `panel/` directory in the grep target.

### Updated FORGE confidence after resolutions

With §9C resolved as the `[stage]` extra (sidestepping the usd-core Windows wheel risk) and §9A resolved as the explicit sources mapping (eliminating the shorthand-vs-explicit branching), the architect's confidence in the FORGE pass executing cleanly in one attempt rises from **MEDIUM** to **HIGH**. Estimated probability of clean first-attempt execution: **~85%**. Remaining risk concentrated in: (a) hatchling explicit sources mapping behaving as documented (~10% risk it doesn't, in which case STOP per §9A), (b) one of the 30 test-file sed renames touching a line that turns out to have additional context (~5% risk, mitigated by running pytest immediately after the rename to catch any breakage in the same step).

---

**END OF DESIGN DOC. The FORGE pass is authorized to begin upon Joe's final "begin forge" signal after reviewing this addendum.**

---

## 12. Forge Pivot — Option A (2026-04-07)

The original design proposed Path B (src/ layout with hatchling sources mapping). Forge discovered empirically that hatchling's `[tool.hatch.build.targets.wheel.sources]` block applies to built wheels but not to editable installs. The editable install's `.pth` still resolved at the repo root, where `src/cognitive` existed but `cognitive` did not. Runtime imports failed.

Per §9A's intent (no improvisation around packaging ambiguity) and C8 (stop at irreversible transitions), forge paused and escalated. Joe approved Option A: move `src/cognitive` to repo-root `cognitive`, delete `src/` entirely, use the conventional flat layout.

Result: all §2 goals achieved without violating §9A. Repo layout is now consistent across `agent/`, `cognitive/`, `panel/`, `tests/`.

Blast radius unchanged from original design (~31 imports across 8 files), now including the directory move.

### Two architect-pass errors discovered during forge

The forge pass uncovered two factual errors in the original investigation (§2c and §4.3) that did not change the shape of the fix but did change its execution:

**Error 1 — `panel/server/routes.py:401` HAS an `except ImportError` wrap.** §2c and §4.3 stated "verified earlier — only `workflow_patch.py:44` had the silent fallback." This was wrong. The function-local import at `panel/server/routes.py:401` is inside a `try:` block whose `except ImportError` at line 404 returns a friendly "Cognitive module not available" JSON response. The forge pass did the minimum spec'd change (single-line rename from `from src.cognitive...` to `from cognitive...`) per §4.3 and §11 §9E. The `except ImportError` block at line 404 is now functionally dead code because the import can no longer fail at module load time, but removing it is out of scope per §4.2 / §8 ("9 conditional branches stay as-is"). Same boundary applies here. Flagged for follow-up cleanup.

**Error 2 — The sed recipe in §5 missed `@patch("src.cognitive...")` decorator strings.** §5's recipe was `sed -i 's/from src\.cognitive/from cognitive/g'` — anchored on `from`. This caught all 30 `from src.cognitive...` import statements but missed 5 `@patch("src.cognitive.transport.interrupt.<symbol>")` decorator strings in `tests/test_cognitive_transport.py` at lines 311, 318, 325, 333, 343. The first post-rename pytest run produced 5 NEW regressions in `TestInterrupt::*` with `ModuleNotFoundError: No module named 'src'`. Forge applied a follow-up sed (`s/"src\.cognitive/"cognitive/g`) to fix the `@patch` strings, and the regressions resolved on the next run. The corrected sed recipe for any future similar fix should be a two-pass sweep: one for `from src.cognitive`, one for any `"src.cognitive` (quoted decorator argument). Lesson recorded for the sed-rename playbook.

### Final acceptance numbers

- **Pre-fix baseline (Phase 0):** 2655 passed | 19 failed | 160 skipped | 27 errors
- **Post-fix baseline (Phase 0.5 complete):** 2673 passed | 4 failed | 160 skipped | 27 errors
- **Delta:** +18 passing (15 networkx-resolved + 3 new regression tests), -15 failing (15 dag_engine networkx + 0 net new regressions)
- The 4 remaining failures are the pre-existing `test_health.py` failures (Open Question §5, out of scope).
- The 27 remaining errors are the pre-existing `test_provisioner.py` `usd-core` errors, expected under `[dev]`-only install per §9C. Installing `[dev,stage]` would resolve them.

### §2 status

**RESOLVED.** `from cognitive.core.graph import CognitiveGraphEngine` works at runtime in any context (verified by `python -c` from a fresh shell). `agent/tools/workflow_patch.py` imports the engine at module top-of-file with no try/except. The PILOT engine wrapper that was previously dead code in production is now live. The C7 adversarial regression test in `tests/test_workflow_patch_engine_live.py` (3 tests) prevents future drift.
