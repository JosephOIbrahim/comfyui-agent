"""Propose/execute closure factories for the `agent autonomous` CLI.

The CozyLoop harness is a pure runner — it doesn't know how to mutate a
workflow or call ComfyUI. Those concerns live here so the CLI can compose
them based on `--execute-mode` and so they can be unit-tested without the
CLI itself.

Modes:
  - "mock"     — no callbacks; harness errors at first iteration. Used by
                 callers that inject their own callbacks programmatically
                 (e.g., the existing test suite). Default for safety.
  - "dry-run"  — produces real proposals (steps/cfg/seed mutations) but
                 returns synthetic axis_scores without contacting ComfyUI.
                 Lets a user smoke-test the CLI plumbing offline.
  - "real"     — requires --workflow. Loads the workflow once, applies
                 mutations as RFC6902 patches per iteration, executes
                 against ComfyUI, derives axis_scores from the result.
                 Failures (timeout / error / circuit-open) return
                 {"success": 0.0, "speed": 0.0} so the ratchet rejects
                 the experiment without halting the harness.

The closures keep state via Python closure cells, NOT module globals, so
multiple harnesses in the same process don't collide.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default proposer — cycles through parameter mutations
# ---------------------------------------------------------------------------

# Order chosen so adjacent iterations probe different axes (steps then cfg
# then seed) — improves the ratchet's signal at small max_experiments.
_PROPOSAL_CYCLE: tuple[dict[str, Any], ...] = (
    {"param": "steps", "delta": +5},
    {"param": "cfg", "delta": +1.0},
    {"param": "seed", "delta": +1},
    {"param": "steps", "delta": -5},
    {"param": "cfg", "delta": -1.0},
)


def make_propose_fn() -> Callable[[], dict[str, Any]]:
    """Return a closure that yields the next proposal in the cycle.

    Each call returns one element of `_PROPOSAL_CYCLE`, indexed by an
    internal counter — so 6th call wraps to the 1st proposal again.
    """
    counter: dict[str, int] = {"n": 0}

    def _propose() -> dict[str, Any]:
        proposal = _PROPOSAL_CYCLE[counter["n"] % len(_PROPOSAL_CYCLE)]
        counter["n"] += 1
        # Return a SHALLOW COPY so the caller cannot mutate our cycle entries.
        return dict(proposal)

    return _propose


# ---------------------------------------------------------------------------
# Execute_fn factory — dry-run + real
# ---------------------------------------------------------------------------

def _synthetic_scores() -> dict[str, float]:
    """Scores returned in dry-run mode. Mid-quality — not so high that the
    ratchet starts hard-pinning, not so low that the harness halts."""
    return {"success": 1.0, "speed": 0.5, "fertility": 0.5}


def _failure_scores() -> dict[str, float]:
    """Scores returned when ComfyUI execution fails. Forces the ratchet to
    reject the experiment without halting (TRANSIENT classification handles
    retry; persistent failure is the user's signal that ComfyUI is down)."""
    return {"success": 0.0, "speed": 0.0, "fertility": 0.0}


def _derive_axis_scores(exec_result: dict) -> dict[str, float]:
    """Translate a ComfyUI execution result into ratchet axis scores.

    Args:
        exec_result: parsed dict from `comfy_execute.handle("execute_with_progress", ...)`
            with shape {status, prompt_id, outputs?, error?, total_time_s?}.

    Returns:
        {success, speed, fertility} — all in [0, 1].
    """
    status = exec_result.get("status")
    if status != "complete" or exec_result.get("error"):
        return _failure_scores()

    outputs = exec_result.get("outputs") or []
    total_time = float(exec_result.get("total_time_s") or 1.0)

    # Speed: 1.0 at 0 s, 0.0 at 60+ s, linear in between. The bands match
    # typical SD1.5 / SDXL render times — adjust later if needed.
    speed = max(0.0, 1.0 - total_time / 60.0)
    # Fertility: 1 output = 0.5, 2+ = 1.0. Higher is better but with
    # diminishing returns.
    fertility = min(1.0, 0.5 + 0.5 * max(0, len(outputs) - 1))

    return {"success": 1.0, "speed": speed, "fertility": fertility}


def _proposal_to_patches(
    proposal: dict[str, Any],
    workflow: dict,
) -> list[dict]:
    """Convert a proposal dict into RFC6902 patches against the loaded workflow.

    Walks the workflow looking for KSampler-shaped nodes (anything with
    "steps" / "cfg" / "seed" inputs) and patches the matching field. If
    multiple sampler nodes exist, all are patched the same way — fine for
    most workflows; a future enhancement could target a specific node_id.
    """
    param = proposal.get("param")
    delta = proposal.get("delta")
    if param is None or delta is None:
        return []

    patches: list[dict] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if param not in inputs:
            continue
        current = inputs[param]
        # Only patch literal values (skip wired connections, which are lists)
        if isinstance(current, (int, float)):
            new_value = current + delta
            patches.append({
                "op": "replace",
                "path": f"/{node_id}/inputs/{param}",
                "value": new_value,
            })
    return patches


def make_execute_fn(
    mode: str,
    workflow_path: str | None,
    *,
    workflow_loader: Callable[[str], dict] | None = None,
    workflow_executor: Callable[[dict], dict] | None = None,
) -> Callable[[dict[str, Any]], dict[str, float]]:
    """Build an execute_fn for the requested mode.

    Args:
        mode: one of "dry-run" | "real".
        workflow_path: required for "real" mode. Path to the base workflow JSON.
        workflow_loader: optional override for tests. Default reads + parses
            the file directly. Signature: (path) -> dict.
        workflow_executor: optional override for tests. Default routes to
            `agent.tools.comfy_execute.handle("execute_with_progress", ...)`.
            Signature: (workflow_dict) -> dict (parsed exec_result).

    Returns:
        A callable matching CozyLoop's execute_fn signature:
        `(change_context: dict) -> dict[str, float]`.

    Raises:
        ValueError: if mode == "real" and workflow_path is None.
    """
    if mode not in ("dry-run", "real"):
        raise ValueError(f"unsupported execute mode: {mode!r}")

    if mode == "dry-run":
        def _execute_dry(change_context: dict[str, Any]) -> dict[str, float]:
            # Don't actually mutate or execute anything — just return
            # synthetic scores so the harness loop can be smoke-tested.
            return _synthetic_scores()
        return _execute_dry

    # mode == "real" — requires a workflow path
    if workflow_path is None:
        raise ValueError("--execute-mode real requires --workflow PATH")

    # Lazy-resolve defaults so tests can stub them out.
    if workflow_loader is None:
        def _default_loader(path: str) -> dict:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        workflow_loader = _default_loader

    if workflow_executor is None:
        def _default_executor(workflow: dict) -> dict:
            from ..tools import comfy_execute
            raw = comfy_execute.handle(
                "execute_with_progress",
                {"workflow": workflow, "timeout": 300},
            )
            try:
                return json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                # Tool returned a plain string or non-JSON — treat as failure.
                return {"status": "error", "error": str(raw)}
        workflow_executor = _default_executor

    # Load the base workflow once. Mutate a deep-copy per iteration so we
    # don't accumulate stale parameter changes.
    base_workflow = workflow_loader(workflow_path)

    import copy

    def _execute_real(change_context: dict[str, Any]) -> dict[str, float]:
        try:
            mutated = copy.deepcopy(base_workflow)
            patches = _proposal_to_patches(change_context, mutated)
            for patch in patches:
                # Apply inline (cheap; no need to round-trip through
                # workflow_patch's session machinery for the autonomous loop).
                _, node_id, _, param = patch["path"].split("/")
                mutated[node_id]["inputs"][param] = patch["value"]
            exec_result = workflow_executor(mutated)
            return _derive_axis_scores(exec_result)
        except Exception as exc:
            log.warning("real execute_fn failed: %s — returning failure scores", exc)
            return _failure_scores()

    return _execute_real
