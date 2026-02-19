"""Verify Agent -- model-aware quality judgment and iteration control.

The Verify Agent evaluates generated outputs against model-specific quality
profiles and decides whether the iterative loop continues or exits.

"Good" is model-relative. A Flux output at cfg 3.5 and an SDXL output
at cfg 7 look fundamentally different, and both can be correct.
The profile defines the quality baseline.

Consumer of the ``quality_signatures`` and ``parameter_space`` sections
of model profiles loaded via :mod:`agent.profiles`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agent.profiles import get_parameter_section, get_quality_section, is_fallback


# ---------------------------------------------------------------------------
# Issue-to-signal mapping
# ---------------------------------------------------------------------------

_ISSUE_SIGNAL_MAP: dict[str, str] = {
    "mushy": "needs_more_steps",
    "soft": "needs_more_steps",
    "blurry": "needs_more_steps",
    "noise": "needs_more_steps",
    "incomplete": "needs_more_steps",
    "undercooked": "needs_more_steps",
    "oversaturated": "needs_lower_cfg",
    "harsh": "needs_lower_cfg",
    "banding": "needs_lower_cfg",
    "color artifacts": "needs_lower_cfg",
    "neon": "needs_lower_cfg",
    "incoherent": "needs_higher_cfg",
    "random": "needs_higher_cfg",
    "prompt not reflected": "needs_higher_cfg",
    "generic": "needs_higher_cfg",
    "wrong subject": "needs_reprompt",
    "missing elements": "needs_reprompt",
    "style mismatch": "needs_reprompt",
    "wrong composition": "needs_reprompt",
    "hand": "needs_inpaint",
    "finger": "needs_inpaint",
    "face distortion": "needs_inpaint",
    "text artifacts": "needs_inpaint",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RefinementAction:
    """A single refinement instruction for the next iteration."""

    type: Literal["adjust_params", "reprompt", "inpaint", "upscale", "retry"]
    target: str  # what to change
    reason: str  # why
    priority: int = 1  # 1=highest


@dataclass
class VerificationResult:
    """Output contract of the Verify Agent."""

    overall_score: float  # 0-1 composite
    intent_alignment: float  # 0-1 did we achieve intent?
    technical_quality: float  # 0-1 model-relative quality
    decision: Literal["accept", "refine", "reprompt", "escalate"]
    refinement_actions: list[RefinementAction] = field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 3
    diagnosed_issues: list[str] = field(default_factory=list)
    model_limitations: list[str] = field(default_factory=list)
    using_fallback_profile: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON transport."""
        return {
            "decision": self.decision,
            "diagnosed_issues": self.diagnosed_issues,
            "intent_alignment": self.intent_alignment,
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
            "model_limitations": self.model_limitations,
            "overall_score": self.overall_score,
            "refinement_actions": [
                {
                    "priority": a.priority,
                    "reason": a.reason,
                    "target": a.target,
                    "type": a.type,
                }
                for a in self.refinement_actions
            ],
            "technical_quality": self.technical_quality,
            "using_fallback_profile": self.using_fallback_profile,
        }


# ---------------------------------------------------------------------------
# Signal-to-action templates
# ---------------------------------------------------------------------------

_SIGNAL_ACTION_MAP: dict[str, RefinementAction | None] = {
    "needs_more_steps": RefinementAction(
        type="adjust_params",
        target="steps",
        reason="Increase steps for more detail",
        priority=2,
    ),
    "needs_lower_cfg": RefinementAction(
        type="adjust_params",
        target="cfg",
        reason="Lower CFG to reduce artifacts",
        priority=1,
    ),
    "needs_higher_cfg": RefinementAction(
        type="adjust_params",
        target="cfg",
        reason="Increase CFG for better prompt adherence",
        priority=1,
    ),
    "needs_reprompt": RefinementAction(
        type="reprompt",
        target="prompt",
        reason="Significant prompt revision needed",
        priority=1,
    ),
    "needs_inpaint": RefinementAction(
        type="inpaint",
        target="detected_region",
        reason="Localized fix via inpainting",
        priority=3,
    ),
    "model_limitation": None,  # No action -- just flag it
}


# ---------------------------------------------------------------------------
# VerifyAgent
# ---------------------------------------------------------------------------


class VerifyAgent:
    """Model-aware quality judgment and iteration control.

    "Good" is model-relative. A Flux output at cfg 3.5 and an SDXL output
    at cfg 7 look fundamentally different, and both can be correct.
    The profile defines the quality baseline.
    """

    # Scoring weights
    INTENT_WEIGHT: float = 0.6
    TECHNICAL_WEIGHT: float = 0.4

    # Decision thresholds
    ACCEPT_THRESHOLD: float = 0.7  # overall_score >= this AND intent > 0.7
    REPROMPT_THRESHOLD: float = 0.4  # intent_alignment < this -> wrong

    def evaluate(
        self,
        output_analysis: dict[str, Any],
        original_intent: str,
        model_id: str,
        parameters_used: dict[str, Any] | None = None,
        iteration_count: int = 0,
        max_iterations: int = 3,
    ) -> VerificationResult:
        """Evaluate an output against model quality profile.

        Args:
            output_analysis: Analysis dict from vision module or manual input.
                Should contain keys like: ``quality_score``, ``artifacts``,
                ``composition``, ``issues``, ``matches_intent`` (bool or score).
            original_intent: What the user asked for.
            model_id: Active model for profile lookup.
            parameters_used: Dict of params used (e.g. ``{"cfg": 3.5, "steps": 20}``).
            iteration_count: Current iteration number.
            max_iterations: Loop safety limit.

        Returns:
            :class:`VerificationResult` with decision and any refinement actions.
        """
        quality_section = get_quality_section(model_id)
        param_section = get_parameter_section(model_id)
        fallback = is_fallback(model_id)

        technical = self._score_technical_quality(
            output_analysis, quality_section, param_section, parameters_used
        )
        intent = self._score_intent_alignment(output_analysis, original_intent)

        overall = intent * self.INTENT_WEIGHT + technical * self.TECHNICAL_WEIGHT

        diagnosed_issues, model_limitations = self._diagnose_issues(
            output_analysis, quality_section, parameters_used
        )

        # --- Decision logic ---------------------------------------------------
        if intent < self.REPROMPT_THRESHOLD:
            decision: Literal["accept", "refine", "reprompt", "escalate"] = (
                "reprompt"
            )
        elif overall >= self.ACCEPT_THRESHOLD and intent > 0.7:
            decision = "accept"
        elif iteration_count >= max_iterations:
            decision = "escalate"
        else:
            decision = "refine"

        # --- Refinement actions (only when not accepting) ---------------------
        refinement_actions: list[RefinementAction] = []
        if decision in ("refine", "reprompt"):
            iteration_signals = quality_section.get("iteration_signals", {})
            refinement_actions = self._generate_refinement_actions(
                diagnosed_issues, iteration_signals
            )

        return VerificationResult(
            overall_score=round(overall, 4),
            intent_alignment=round(intent, 4),
            technical_quality=round(technical, 4),
            decision=decision,
            refinement_actions=refinement_actions,
            iteration_count=iteration_count,
            max_iterations=max_iterations,
            diagnosed_issues=diagnosed_issues,
            model_limitations=model_limitations,
            using_fallback_profile=fallback,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_technical_quality(
        self,
        output_analysis: dict[str, Any],
        quality_section: dict[str, Any],
        param_section: dict[str, Any],
        parameters_used: dict[str, Any] | None,
    ) -> float:
        """Score technical quality relative to the model profile (0-1)."""
        quality_floor = quality_section.get("quality_floor", {})
        baseline = float(quality_floor.get("reference_score", 0.5))

        # Start from baseline
        score = baseline

        # --- Boost for expected characteristics present -------------------
        expected = quality_section.get("expected_characteristics", [])
        analysis_text = str(output_analysis).lower()

        if expected:
            matches = 0
            for char in expected:
                if isinstance(char, str) and char.lower() in analysis_text:
                    matches += 1
            if expected:
                match_ratio = matches / len(expected)
                score += match_ratio * 0.15  # Up to +0.15 for all matches

        # --- Penalize for known artifacts present -------------------------
        known_artifacts = quality_section.get("known_artifacts", [])
        issues = output_analysis.get("issues", [])
        artifacts_found = output_analysis.get("artifacts", [])
        all_issues = []
        if isinstance(issues, list):
            all_issues.extend(issues)
        if isinstance(artifacts_found, list):
            all_issues.extend(artifacts_found)

        if known_artifacts and all_issues:
            penalty = 0.0
            for artifact_def in known_artifacts:
                if not isinstance(artifact_def, dict):
                    continue
                artifact_desc = str(artifact_def.get("artifact", "")).lower()
                severity = str(artifact_def.get("severity", "medium")).lower()
                for issue in all_issues:
                    issue_lower = str(issue).lower()
                    # Check if this known artifact matches any reported issue
                    artifact_words = artifact_desc.split()
                    if any(w in issue_lower for w in artifact_words if len(w) > 3):
                        sev_map = {
                            "critical": 0.15,
                            "high": 0.10,
                            "medium": 0.05,
                            "low": 0.02,
                            "info": 0.0,
                        }
                        penalty += sev_map.get(severity, 0.05)
                        break  # one match per artifact definition
            score -= penalty

        # --- Factor in parameter sweet spots ------------------------------
        if parameters_used:
            param_bonus = self._score_parameter_fit(parameters_used, param_section)
            score += param_bonus  # can be negative

        # --- Use quality_score from analysis as additional signal ----------
        if "quality_score" in output_analysis:
            qs = float(output_analysis["quality_score"])
            # Blend: 70% profile-based, 30% reported quality_score
            score = score * 0.7 + qs * 0.3

        return max(0.0, min(1.0, score))

    def _score_parameter_fit(
        self,
        parameters_used: dict[str, Any],
        param_section: dict[str, Any],
    ) -> float:
        """Return a small bonus/penalty based on how well params fit the profile."""
        bonus = 0.0

        for param_name in ("cfg", "steps"):
            if param_name not in parameters_used:
                continue
            try:
                value = float(parameters_used[param_name])
            except (TypeError, ValueError):
                continue
            section = param_section.get(param_name, {})
            sweet_spot = section.get("sweet_spot", [])
            if isinstance(sweet_spot, list) and len(sweet_spot) == 2:
                lo, hi = float(sweet_spot[0]), float(sweet_spot[1])
                if lo <= value <= hi:
                    bonus += 0.05  # In sweet spot
                else:
                    # How far outside?
                    full_range = section.get("range", [lo, hi])
                    if isinstance(full_range, list) and len(full_range) == 2:
                        r_lo, r_hi = float(full_range[0]), float(full_range[1])
                        span = r_hi - r_lo if r_hi > r_lo else 1.0
                        if value < lo:
                            dist = (lo - value) / span
                        else:
                            dist = (value - hi) / span
                        bonus -= min(dist * 0.15, 0.10)

        return bonus

    def _score_intent_alignment(
        self,
        output_analysis: dict[str, Any],
        original_intent: str,
    ) -> float:
        """Score how well the output matches the original intent (0-1)."""
        mi = output_analysis.get("matches_intent")

        if mi is not None:
            if isinstance(mi, bool):
                return 1.0 if mi else 0.0
            try:
                return max(0.0, min(1.0, float(mi)))
            except (TypeError, ValueError):
                pass

        # Fallback: use quality_score as a weak proxy
        qs = output_analysis.get("quality_score")
        if qs is not None:
            try:
                return max(0.0, min(1.0, float(qs) * 0.5 + 0.25))
            except (TypeError, ValueError):
                pass

        # Default: uncertain
        return 0.5

    def _diagnose_issues(
        self,
        output_analysis: dict[str, Any],
        quality_section: dict[str, Any],
        parameters_used: dict[str, Any] | None,
    ) -> tuple[list[str], list[str]]:
        """Diagnose issues and model limitations from analysis + profile.

        Returns:
            Tuple of (diagnosed_issues, model_limitations).
        """
        diagnosed: list[str] = []
        limitations: list[str] = []

        issues = output_analysis.get("issues", [])
        artifacts = output_analysis.get("artifacts", [])
        all_issues: list[str] = []
        if isinstance(issues, list):
            all_issues.extend(str(i) for i in issues)
        if isinstance(artifacts, list):
            all_issues.extend(str(a) for a in artifacts)

        iteration_signals = quality_section.get("iteration_signals", {})

        for issue_text in all_issues:
            signal = self._map_issue_to_signal(issue_text, iteration_signals)
            if signal == "model_limitation":
                limitations.append(issue_text)
            elif signal is not None:
                diagnosed.append(issue_text)
            else:
                # Unknown issue -- still record it as diagnosed
                diagnosed.append(issue_text)

        # Check parameter failure modes
        if parameters_used:
            param_issues = self._check_parameter_failure_modes(
                parameters_used, get_parameter_section.__wrapped__  # type: ignore[attr-defined]
                if hasattr(get_parameter_section, "__wrapped__")
                else None
            )
            # Actually, re-fetch param section properly
            param_issues = self._check_parameter_failure_modes_from_quality(
                parameters_used, quality_section
            )
            diagnosed.extend(param_issues)

        return diagnosed, limitations

    def _map_issue_to_signal(
        self,
        issue: str,
        iteration_signals: dict[str, Any],
    ) -> str | None:
        """Map an issue description to the appropriate signal category."""
        issue_lower = issue.lower()

        # Check the static keyword map first
        for keyword, signal in sorted(_ISSUE_SIGNAL_MAP.items()):
            if keyword in issue_lower:
                return signal

        # Check against indicator lists in the profile's iteration_signals
        for signal_name, signal_data in sorted(iteration_signals.items()):
            indicators: list[str] = []
            if isinstance(signal_data, dict):
                indicators = signal_data.get("indicators", [])
            elif isinstance(signal_data, list):
                indicators = signal_data

            for indicator in indicators:
                if not isinstance(indicator, str):
                    continue
                # Check if key words from the indicator match the issue
                indicator_words = [
                    w.lower() for w in indicator.split() if len(w) > 3
                ]
                matches = sum(1 for w in indicator_words if w in issue_lower)
                if indicator_words and matches >= min(2, len(indicator_words)):
                    return signal_name

        return None

    def _check_parameter_failure_modes_from_quality(
        self,
        parameters_used: dict[str, Any],
        quality_section: dict[str, Any],
    ) -> list[str]:
        """Check if current params match known failure modes from the profile."""
        issues: list[str] = []
        known_artifacts = quality_section.get("known_artifacts", [])

        for artifact_def in known_artifacts:
            if not isinstance(artifact_def, dict):
                continue
            condition = str(artifact_def.get("condition", ""))
            if not condition:
                continue

            if self._eval_condition(condition, parameters_used):
                artifact_desc = artifact_def.get("artifact", "Unknown artifact")
                issues.append(f"Parameter issue: {artifact_desc}")

        return issues

    def _check_parameter_failure_modes(
        self,
        parameters_used: dict[str, Any],
        param_section: dict[str, Any] | None,
    ) -> list[str]:
        """Check params against parameter_space failure_modes."""
        if not param_section:
            return []

        issues: list[str] = []
        for param_name, section in sorted(param_section.items()):
            if not isinstance(section, dict):
                continue
            failure_modes = section.get("failure_modes", [])
            if isinstance(failure_modes, dict):
                # SDXL style: {too_high: {condition, ...}, too_low: ...}
                for _key, mode in sorted(failure_modes.items()):
                    if isinstance(mode, dict):
                        condition = str(mode.get("condition", ""))
                        if condition and self._eval_condition(
                            condition, parameters_used
                        ):
                            desc = mode.get(
                                "description",
                                mode.get("artifact", "Parameter issue"),
                            )
                            issues.append(f"{param_name}: {desc}")
            elif isinstance(failure_modes, list):
                # Flux style: [{condition, artifact}, ...]
                for mode in failure_modes:
                    if isinstance(mode, dict):
                        condition = str(mode.get("condition", ""))
                        if condition and self._eval_condition(
                            condition, parameters_used
                        ):
                            desc = mode.get("artifact", "Parameter issue")
                            issues.append(f"{param_name}: {desc}")

        return issues

    @staticmethod
    def _eval_condition(condition: str, parameters_used: dict[str, Any]) -> bool:
        """Evaluate a simple parameter condition like ``cfg > 7.0``.

        Supports ``>``, ``<``, ``>=``, ``<=`` operators only.
        Non-numeric or non-matching conditions return False safely.
        """
        condition = condition.strip()
        for op_str, op_fn in [
            (">=", lambda a, b: a >= b),
            ("<=", lambda a, b: a <= b),
            (">", lambda a, b: a > b),
            ("<", lambda a, b: a < b),
        ]:
            if op_str in condition:
                parts = condition.split(op_str, 1)
                if len(parts) == 2:
                    param_name = parts[0].strip()
                    try:
                        threshold = float(parts[1].strip())
                    except ValueError:
                        return False
                    if param_name in parameters_used:
                        try:
                            value = float(parameters_used[param_name])
                        except (TypeError, ValueError):
                            return False
                        return op_fn(value, threshold)
                return False
        return False

    def _generate_refinement_actions(
        self,
        diagnosed_issues: list[str],
        iteration_signals: dict[str, Any],
    ) -> list[RefinementAction]:
        """Generate refinement actions from diagnosed issues."""
        actions: list[RefinementAction] = []
        seen_signals: set[str] = set()

        for issue in diagnosed_issues:
            signal = self._map_issue_to_signal(issue, iteration_signals)
            if signal and signal not in seen_signals:
                seen_signals.add(signal)
                template = _SIGNAL_ACTION_MAP.get(signal)
                if template is not None:
                    # Create a fresh copy
                    actions.append(
                        RefinementAction(
                            type=template.type,
                            target=template.target,
                            reason=template.reason,
                            priority=template.priority,
                        )
                    )

        # Sort by priority (1 = highest priority, first in list)
        actions.sort(key=lambda a: a.priority)
        return actions
