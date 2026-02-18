"""Iterative refinement — autonomous quality iteration loop.

Executes a workflow, analyzes output via Vision, diagnoses issues,
generates parameter patches, and re-executes until quality converges
or the iteration budget is exhausted. This is the self-healing
reinforcement loop that turns the agent from "tool that does what
you say" into "co-pilot that iterates toward artistic intent."

Integration points:
  - execute_with_progress (comfy_execute) — run workflows
  - verify_execution (verify_execution) — confirm outputs + record memory
  - analyze_image (vision) — score quality + detect artifacts
  - set_input (workflow_patch) — apply parameter adjustments
  - undo_workflow_patch (workflow_patch) — roll back bad changes
  - get_learned_patterns (memory) — query past outcomes for diagnosis
"""

import json
import logging
import random

from ._sdk import BrainAgent

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODE_BUDGETS = {
    "quick": 2,
    "refine": 5,
    "deep": 10,
    "unlimited": 1,
}

# Minimum quality improvement to count as "making progress"
IMPROVEMENT_THRESHOLD = 0.3

# Sampler options for switching
SAMPLER_OPTIONS = [
    "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde",
    "dpmpp_sde", "dpm_2", "uni_pc",
]

SCHEDULER_OPTIONS = [
    "normal", "karras", "exponential", "sgm_uniform",
]

# ---------------------------------------------------------------------------
# Heuristic diagnosis rules
# ---------------------------------------------------------------------------

# Maps detected artifact keywords -> adjustment vectors
HEURISTIC_RULES: list[dict] = [
    {
        "triggers": ["color banding", "banding", "posterization"],
        "param": "cfg",
        "direction": "decrease",
        "delta": 1.5,
        "reason": "High CFG can cause color banding — lowering should smooth gradients",
    },
    {
        "triggers": ["blurry", "soft", "lack of detail", "low detail", "fuzzy"],
        "param": "steps",
        "direction": "increase",
        "delta": 10,
        "reason": "More sampling steps add detail and sharpness",
    },
    {
        "triggers": ["oversaturated", "oversaturation", "too vivid", "harsh colors"],
        "param": "cfg",
        "direction": "decrease",
        "delta": 2.0,
        "reason": "High CFG pushes colors toward saturation — lower for natural tones",
    },
    {
        "triggers": ["noisy", "grain", "speckle", "noise"],
        "param": "sampler_name",
        "direction": "switch",
        "value": "dpmpp_2m",
        "reason": "DPM++ 2M tends to produce cleaner results with less noise",
    },
    {
        "triggers": ["artifact", "glitch", "distortion", "corruption"],
        "param": "sampler_name",
        "direction": "switch",
        "value": "dpmpp_2m_sde",
        "reason": "DPM++ 2M SDE is more robust against sampling artifacts",
    },
    {
        "triggers": ["wrong style", "style mismatch", "doesn't match"],
        "param": "cfg",
        "direction": "increase",
        "delta": 1.0,
        "reason": "Higher CFG strengthens prompt adherence for style matching",
    },
    {
        "triggers": ["edge artifact", "edge", "fringing", "halo"],
        "param": "denoise",
        "direction": "decrease",
        "delta": 0.05,
        "reason": "Slightly lower denoise can reduce edge artifacts",
    },
    {
        "triggers": ["flat", "lacking depth", "no contrast"],
        "param": "cfg",
        "direction": "increase",
        "delta": 1.0,
        "reason": "Slightly higher CFG can add contrast and depth",
    },
]

# Fallback: if no heuristic matches, try these in order
FALLBACK_ADJUSTMENTS = [
    {"param": "cfg", "direction": "decrease", "delta": 1.0,
     "reason": "General quality improvement: slightly lower CFG"},
    {"param": "steps", "direction": "increase", "delta": 5,
     "reason": "General quality improvement: more sampling steps"},
    {"param": "sampler_name", "direction": "switch", "value": "dpmpp_2m",
     "reason": "General quality improvement: switch to DPM++ 2M"},
    {"param": "scheduler", "direction": "switch", "value": "karras",
     "reason": "General quality improvement: Karras noise schedule"},
    {"param": "seed", "direction": "randomize",
     "reason": "Try a different seed for variety"},
]


# ---------------------------------------------------------------------------
# IterativeRefineAgent
# ---------------------------------------------------------------------------

class IterativeRefineAgent(BrainAgent):
    """Autonomous quality iteration loop."""

    TOOLS: list[dict] = [
        {
            "name": "iterative_refine",
            "description": (
                "Autonomous quality iteration loop. Executes the current workflow, "
                "analyzes output quality via Vision, diagnoses issues, applies "
                "parameter patches, and re-executes until quality converges or "
                "the iteration budget is exhausted. Modes: 'quick' (1-2 iterations), "
                "'refine' (3-5), 'deep' (5-10), 'unlimited' (1 per call, caller drives)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "description": (
                            "What the artist wants (e.g., 'photorealistic portrait, "
                            "no artifacts'). Used for vision analysis context."
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["quick", "refine", "deep", "unlimited"],
                        "description": (
                            "Iteration budget. quick=1-2, refine=3-5, deep=5-10, "
                            "unlimited=1 per call (caller drives outer loop). "
                            "Default: refine."
                        ),
                    },
                    "quality_threshold": {
                        "type": "number",
                        "description": (
                            "Target quality score (0-1). Stop when reached. Default: 0.7."
                        ),
                    },
                    "session": {
                        "type": "string",
                        "description": "Session name for memory recording. Default: 'default'.",
                    },
                    "goal_id": {
                        "type": "string",
                        "description": "Goal ID for planner integration.",
                    },
                },
                "required": ["intent"],
            },
        },
    ]

    def handle(self, name: str, tool_input: dict) -> str:
        if name == "iterative_refine":
            return self._handle_iterative_refine(tool_input)
        return self.to_json({"error": f"Unknown tool: {name}"})

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _handle_iterative_refine(self, tool_input: dict) -> str:
        intent = tool_input["intent"]
        mode = tool_input.get("mode", "refine")
        quality_threshold = tool_input.get("quality_threshold", 0.7)
        session = tool_input.get("session", "default")
        goal_id = tool_input.get("goal_id", "")

        max_iterations = MODE_BUDGETS.get(mode, 5)

        iterations: list[dict] = []
        scores: list[float] = []
        best_result: dict | None = None
        best_score = -1.0
        reason = "max_iterations"
        consecutive_regressions = 0
        used_adjustments: list[str] = []  # track what we've tried
        artifacts: list[str] = []  # last detected artifacts

        for i in range(max_iterations):
            iteration_num = i + 1
            log.info("iterative_refine: iteration %d/%d", iteration_num, max_iterations)

            # --- Execute ---
            exec_result = self._execute_workflow(session, goal_id)
            if "error" in exec_result:
                iterations.append({
                    "iteration": iteration_num,
                    "error": exec_result["error"],
                })
                reason = "execution_error"
                break

            output_path = exec_result.get("output_path")
            render_time = exec_result.get("render_time_s")
            key_params = exec_result.get("key_params", {})

            # --- Perceive (Vision) ---
            analysis = self._analyze_output(output_path, intent, key_params)
            quality_score = analysis.get("quality_score", 0.0)
            artifacts = analysis.get("artifacts", [])

            scores.append(quality_score)

            iteration_record: dict = {
                "iteration": iteration_num,
                "parameters": key_params,
                "quality_score": quality_score,
                "artifacts_detected": artifacts,
                "output_path": output_path,
                "render_time_s": render_time,
            }

            # Track best
            if quality_score > best_score:
                best_score = quality_score
                best_result = {
                    "iteration": iteration_num,
                    "quality_score": quality_score,
                    "output_path": output_path,
                    "parameters": key_params,
                }

            # --- Check convergence ---
            convergence = self._check_convergence(
                scores, quality_threshold, consecutive_regressions,
            )
            if convergence:
                iteration_record["diagnosis"] = "Converged"
                iteration_record["patch_applied"] = None
                iterations.append(iteration_record)
                reason = convergence
                break

            # Don't patch after the last iteration
            if iteration_num >= max_iterations:
                iteration_record["diagnosis"] = "Budget exhausted"
                iteration_record["patch_applied"] = None
                iterations.append(iteration_record)
                break

            # --- Diagnose ---
            diagnosis = self._diagnose(
                artifacts, scores, key_params, session, used_adjustments,
            )
            iteration_record["diagnosis"] = diagnosis.get("reason", "")

            # --- Check regression and handle rollback ---
            if len(scores) >= 2 and scores[-1] < scores[-2]:
                consecutive_regressions += 1
                if consecutive_regressions >= 2:
                    # Roll back last patch and stop
                    self._rollback()
                    iteration_record["diagnosis"] = (
                        "Two consecutive regressions — rolled back last patch"
                    )
                    iteration_record["patch_applied"] = None
                    iterations.append(iteration_record)
                    reason = "regression_rollback"
                    break
                else:
                    # Roll back and try a different vector
                    self._rollback()
                    diagnosis = self._diagnose(
                        artifacts, scores, key_params, session, used_adjustments,
                    )
                    iteration_record["diagnosis"] = (
                        f"Regression — rolled back, trying: {diagnosis.get('reason', '')}"
                    )
            else:
                consecutive_regressions = 0

            # --- Prescribe (apply patch) ---
            patch_result = self._apply_adjustment(diagnosis, key_params)
            iteration_record["patch_applied"] = patch_result
            if patch_result:
                used_adjustments.append(
                    f"{patch_result.get('param', '')}:{patch_result.get('direction', '')}"
                )

            iterations.append(iteration_record)

        # Build recommendation if not converged
        recommendation = ""
        if reason != "threshold_met":
            if artifacts:
                recommendation = (
                    f"Quality at {best_score:.2f} (target: {quality_threshold}). "
                    f"Still seeing: {', '.join(artifacts[:3])}. "
                    f"Consider trying a different checkpoint or adding ControlNet guidance."
                )
            else:
                recommendation = (
                    f"Quality at {best_score:.2f} (target: {quality_threshold}). "
                    f"Try a different checkpoint, adjust the prompt, or use img2img "
                    f"with a reference image."
                )

        return self.to_json({
            "iterations": iterations,
            "best_result": best_result,
            "converged": reason == "threshold_met",
            "reason": reason,
            "recommendation": recommendation if reason != "threshold_met" else "",
            "total_iterations": len(iterations),
        })

    # ------------------------------------------------------------------
    # Sub-steps
    # ------------------------------------------------------------------

    def _execute_workflow(self, session: str, goal_id: str) -> dict:
        """Execute the current workflow and verify output."""
        try:
            from ..tools import handle as dispatch

            raw = dispatch("execute_with_progress", {
                "auto_verify": True,
                "session": session,
                "goal_id": goal_id,
            })
            result = json.loads(raw)

            if result.get("status") != "complete":
                return {"error": result.get("error", "Execution did not complete")}

            verification = result.get("verification", {})
            outputs = verification.get("outputs", [])
            key_params = verification.get("key_params", {})

            # Find first existing image output
            output_path = None
            for o in outputs:
                if o.get("type") == "image" and o.get("exists"):
                    output_path = o["absolute_path"]
                    break

            if not output_path:
                # Fall back to non-verified outputs
                for o in result.get("outputs", []):
                    if o.get("type") == "image" and o.get("filename"):
                        # Try resolving via get_output_path
                        path_raw = dispatch("get_output_path", {
                            "filename": o["filename"],
                            "subfolder": o.get("subfolder", ""),
                        })
                        path_result = json.loads(path_raw)
                        if path_result.get("exists"):
                            output_path = path_result["absolute_path"]
                            break

            if not output_path:
                return {"error": "No output image found after execution"}

            return {
                "output_path": output_path,
                "render_time_s": result.get("total_time_s"),
                "key_params": key_params,
                "prompt_id": result.get("prompt_id"),
            }

        except Exception as e:
            log.error("Execute failed: %s", e)
            return {"error": str(e)}

    def _analyze_output(
        self, image_path: str | None, intent: str, key_params: dict,
    ) -> dict:
        """Run vision analysis on an output image."""
        if not image_path:
            return {"quality_score": 0.0, "artifacts": ["no output image"]}

        try:
            from ..tools import handle as dispatch

            workflow_context = (
                f"{key_params.get('model', 'unknown')} "
                f"{key_params.get('steps', '?')} steps "
                f"CFG {key_params.get('cfg', '?')} "
                f"{key_params.get('sampler_name', '?')} "
                f"{key_params.get('resolution', '?')}"
            )

            raw = dispatch("analyze_image", {
                "image_path": image_path,
                "prompt_used": intent,
                "workflow_context": workflow_context,
            })
            result = json.loads(raw)

            if "error" in result:
                log.warning("Vision analysis error: %s", result["error"])
                return {"quality_score": 0.5, "artifacts": ["vision_error"]}

            return {
                "quality_score": result.get("quality_score", 0.5),
                "artifacts": result.get("artifacts", []),
                "suggestions": result.get("suggestions", []),
                "prompt_adherence": result.get("prompt_adherence"),
            }

        except Exception as e:
            log.warning("Vision analysis failed: %s", e)
            return {"quality_score": 0.5, "artifacts": ["vision_unavailable"]}

    def _check_convergence(
        self,
        scores: list[float],
        threshold: float,
        consecutive_regressions: int,
    ) -> str | None:
        """Check if we should stop iterating. Returns reason or None."""
        if not scores:
            return None

        # Threshold met
        if scores[-1] >= threshold:
            return "threshold_met"

        # Plateau: last 2 changes both small AND not both declining
        if len(scores) >= 3:
            delta1 = scores[-1] - scores[-2]
            delta2 = scores[-2] - scores[-3]
            if (abs(delta1) < IMPROVEMENT_THRESHOLD
                    and abs(delta2) < IMPROVEMENT_THRESHOLD
                    and not (delta1 < 0 and delta2 < 0)):
                return "plateaued"

        return None

    def _diagnose(
        self,
        artifacts: list[str],
        scores: list[float],
        key_params: dict,
        session: str,
        used_adjustments: list[str],
    ) -> dict:
        """Diagnose issues and recommend an adjustment vector.

        Uses heuristic rules first, falls back to general adjustments.
        """
        # Try heuristic rules based on detected artifacts
        artifacts_lower = " ".join(str(a).lower() for a in artifacts)

        for rule in HEURISTIC_RULES:
            # Check if any trigger matches
            if any(t in artifacts_lower for t in rule["triggers"]):
                adjustment_key = f"{rule['param']}:{rule['direction']}"
                if adjustment_key in used_adjustments:
                    continue  # Already tried this

                result = {
                    "param": rule["param"],
                    "direction": rule["direction"],
                    "reason": rule["reason"],
                }
                if "delta" in rule:
                    result["delta"] = rule["delta"]
                if "value" in rule:
                    result["value"] = rule["value"]
                return result

        # No heuristic matched — use fallback sequence
        for fallback in FALLBACK_ADJUSTMENTS:
            adjustment_key = f"{fallback['param']}:{fallback['direction']}"
            if adjustment_key not in used_adjustments:
                return dict(fallback)

        # All fallbacks used — randomize seed
        return {
            "param": "seed",
            "direction": "randomize",
            "reason": "All adjustment vectors exhausted — trying new seed",
        }

    def _apply_adjustment(self, diagnosis: dict, key_params: dict) -> dict | None:
        """Apply a parameter adjustment via the workflow patch engine."""
        param = diagnosis.get("param")
        direction = diagnosis.get("direction")

        if not param:
            return None

        try:
            from ..tools import handle as dispatch

            # Find the node that has this parameter
            wf_state = None
            try:
                from ..tools.workflow_patch import get_current_workflow
                wf_state = get_current_workflow()
            except Exception:
                pass

            if not wf_state:
                log.warning("No workflow loaded — cannot apply adjustment")
                return None

            # Find the target node
            target_node_id = None
            current_value = None
            # He2025: sort for deterministic iteration
            for nid, node in sorted(wf_state.items()):
                if not isinstance(node, dict):
                    continue
                inputs = node.get("inputs", {})
                if param in inputs and not isinstance(inputs[param], list):
                    target_node_id = nid
                    current_value = inputs[param]
                    break

            if target_node_id is None:
                log.warning("Parameter '%s' not found in workflow", param)
                return None

            # Calculate new value
            new_value = self._calculate_new_value(
                param, direction, current_value, diagnosis, key_params,
            )

            if new_value is None or new_value == current_value:
                return None

            # Apply via set_input
            raw = dispatch("set_input", {
                "node_id": target_node_id,
                "input_name": param,
                "value": new_value,
            })
            result = json.loads(raw)

            if result.get("error"):
                log.warning("Patch failed: %s", result["error"])
                return None

            return {
                "param": param,
                "direction": direction,
                "old_value": current_value,
                "new_value": new_value,
                "node_id": target_node_id,
                "reason": diagnosis.get("reason", ""),
            }

        except Exception as e:
            log.warning("Apply adjustment failed: %s", e)
            return None

    def _calculate_new_value(
        self,
        param: str,
        direction: str,
        current_value,
        diagnosis: dict,
        key_params: dict,
    ):
        """Calculate the new parameter value from an adjustment vector."""
        if direction == "randomize":
            return random.randint(1, 2**31)

        if direction == "switch":
            target = diagnosis.get("value")
            if target and target != current_value:
                return target
            # Pick a different option
            if param == "sampler_name":
                options = [s for s in SAMPLER_OPTIONS if s != current_value]
                return options[0] if options else current_value
            elif param == "scheduler":
                options = [s for s in SCHEDULER_OPTIONS if s != current_value]
                return options[0] if options else current_value
            return current_value

        delta = diagnosis.get("delta", 1.0)

        if direction == "increase":
            if isinstance(current_value, (int, float)):
                new_val = current_value + delta
                # Clamp steps to reasonable range
                if param == "steps":
                    new_val = min(int(new_val), 150)
                return type(current_value)(new_val)

        elif direction == "decrease":
            if isinstance(current_value, (int, float)):
                new_val = current_value - delta
                # Clamp to minimums
                if param == "cfg":
                    new_val = max(new_val, 1.0)
                elif param == "steps":
                    new_val = max(int(new_val), 1)
                elif param == "denoise":
                    new_val = max(new_val, 0.1)
                return type(current_value)(new_val)

        return None

    def _rollback(self) -> None:
        """Undo the last workflow patch."""
        try:
            from ..tools import handle as dispatch
            dispatch("undo_workflow_patch", {})
        except Exception as e:
            log.warning("Rollback failed: %s", e)


# ---------------------------------------------------------------------------
# Backward compatibility — lazy singleton
# ---------------------------------------------------------------------------

_instance: IterativeRefineAgent | None = None


def _get_instance() -> IterativeRefineAgent:
    global _instance
    if _instance is None:
        _instance = IterativeRefineAgent()
    return _instance


TOOLS = IterativeRefineAgent.TOOLS


def handle(name: str, tool_input: dict) -> str:
    """Execute an iterative_refine tool call."""
    return _get_instance().handle(name, tool_input)
