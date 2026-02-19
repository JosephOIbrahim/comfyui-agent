"""Intent Agent — translates artistic language into structured parameter specs.

This agent is a PURE REASONING layer. It does NOT call tools. It does NOT
touch ComfyUI. It consumes a model profile and user intent, then produces
a structured IntentSpecification that the Execution Agent implements.

The translation pipeline:
  1. Parse compound intents ("dreamier and sharper") into individual words
  2. Look up each intent in the model profile's intent_translations
  3. Detect and resolve conflicts between competing directions
  4. Convert directional instructions to concrete parameter values
  5. Format prompt mutations according to the model's prompt style
  6. Return a fully specified IntentSpecification
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from agent.profiles import get_intent_section, get_parameter_section, is_fallback

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParameterMutation:
    """A single parameter change."""

    target: str  # e.g. "KSampler.cfg", "KSampler.sampler_name"
    action: Literal["set", "adjust_up", "adjust_down"]
    value: Any = None  # concrete value for "set"
    magnitude: str | None = None  # "slight", "moderate", "large" for adjust
    reason: str = ""


@dataclass
class PromptMutation:
    """A change to prompt text."""

    target: Literal["positive_prompt", "negative_prompt"]
    action: Literal["append", "prepend", "replace", "remove"]
    value: str = ""
    reason: str = ""


@dataclass
class ConflictResolution:
    """How the Intent Agent resolves competing intents."""

    intent_a: str
    intent_b: str
    conflict_dimension: str  # e.g. "cfg_direction"
    resolution_strategy: str
    explanation: str


@dataclass
class IntentSpecification:
    """Output contract of the Intent Agent."""

    model_id: str
    parameter_mutations: list[ParameterMutation] = field(default_factory=list)
    prompt_mutations: list[PromptMutation] = field(default_factory=list)
    confidence: float = 0.0  # 0-1
    conflicts_resolved: list[ConflictResolution] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    using_fallback_profile: bool = False

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON / schema validation."""
        return {
            "confidence": self.confidence,
            "conflicts_resolved": [
                {
                    "conflict_dimension": c.conflict_dimension,
                    "explanation": c.explanation,
                    "intent_a": c.intent_a,
                    "intent_b": c.intent_b,
                    "resolution_strategy": c.resolution_strategy,
                }
                for c in self.conflicts_resolved
            ],
            "model_id": self.model_id,
            "parameter_mutations": [
                {
                    "action": m.action,
                    "magnitude": m.magnitude,
                    "reason": m.reason,
                    "target": m.target,
                    "value": m.value,
                }
                for m in self.parameter_mutations
            ],
            "prompt_mutations": [
                {
                    "action": m.action,
                    "reason": m.reason,
                    "target": m.target,
                    "value": m.value,
                }
                for m in self.prompt_mutations
            ],
            "using_fallback_profile": self.using_fallback_profile,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Synonym map — common artistic words to canonical profile keys
# ---------------------------------------------------------------------------

_INTENT_SYNONYMS: dict[str, str] = {
    "abstract": "more abstract",
    "artistic": "more stylized",
    "crisp": "sharper",
    "dark": "moodier",
    "detailed": "more detailed",
    "dramatic": "more dramatic",
    "dreamy": "dreamier",
    "ethereal": "dreamier",
    "moody": "moodier",
    "painterly": "more stylized",
    "photo": "more photorealistic",
    "photorealistic": "more photorealistic",
    "realistic": "more photorealistic",
    "sharp": "sharper",
    "soft": "softer",
    "softer": "softer",
    "stylized": "more stylized",
    "surreal": "more abstract",
    "vintage": "more vintage",
}

# ---------------------------------------------------------------------------
# Direction -> magnitude classification
# ---------------------------------------------------------------------------

_DIRECTION_MAGNITUDE: dict[str, tuple[str, str]] = {
    # direction_keyword: (up_or_down, magnitude)
    "lower": ("down", "moderate"),
    "higher": ("up", "moderate"),
    "slightly_lower": ("down", "slight"),
    "slightly_higher": ("up", "slight"),
    "much_lower": ("down", "large"),
    "much_higher": ("up", "large"),
    "default": ("hold", "none"),
}

# Magnitude -> fraction of parameter range to move
_MAGNITUDE_FRACTION: dict[str, float] = {
    "slight": 0.10,
    "moderate": 0.25,
    "large": 0.50,
}

# ---------------------------------------------------------------------------
# Conflict resolution rules
# ---------------------------------------------------------------------------

_CONFLICT_RULES: dict[tuple[str, str], dict[str, str]] = {
    ("cfg_direction:lower", "cfg_direction:higher"): {
        "strategy": "hold_current_cfg_adjust_via_prompt_and_sampler",
        "explanation": (
            "Conflicting cfg demands. Holding cfg, using prompt tokens "
            "and sampler selection to achieve both intents."
        ),
    },
    ("steps_direction:lower", "steps_direction:higher"): {
        "strategy": "favor_higher_for_quality",
        "explanation": (
            "Conflicting step demands. Favoring higher steps since "
            "quality is the safer bet."
        ),
    },
    ("denoise_direction:lower", "denoise_direction:higher"): {
        "strategy": "favor_lower_for_preservation",
        "explanation": (
            "Conflicting denoise demands. Favoring lower to preserve "
            "more of the original image."
        ),
    },
}

# Dimensions we track for conflict detection
_CONFLICT_DIMENSIONS = ("cfg_direction", "steps_direction", "denoise_direction")

# Direction values that map to "lower" or "higher" buckets
_LOWER_DIRECTIONS = {"lower", "slightly_lower", "much_lower"}
_HIGHER_DIRECTIONS = {"higher", "slightly_higher", "much_higher"}


# ---------------------------------------------------------------------------
# Intent Agent
# ---------------------------------------------------------------------------


class IntentAgent:
    """Translates artistic intent into structured parameter specifications.

    This agent is tool-less. It queries model profiles at runtime and
    produces IntentSpecification objects that the Execution Agent implements.
    """

    # Expose rules for testing / introspection
    CONFLICT_RULES = _CONFLICT_RULES

    def translate(
        self,
        user_intent: str,
        model_id: str,
        workflow_state: dict[str, Any] | None = None,
        refinement_context: list[dict[str, Any]] | None = None,
    ) -> IntentSpecification:
        """Translate user intent into an IntentSpecification.

        Parameters
        ----------
        user_intent:
            Natural language from the artist (e.g. "dreamier and sharper").
        model_id:
            Model profile key (e.g. "flux1-dev", "sdxl-base").
        workflow_state:
            Optional current parameter values from the workflow.  Keys are
            param names like ``"cfg"``, ``"steps"``, ``"denoise"``.
        refinement_context:
            Optional list of refinement actions from the Verify Agent.
            Each dict should have ``"type"`` and optionally ``"target"``
            and ``"recommendation"`` keys.
        """
        workflow_state = workflow_state or {}

        # Load profile sections
        intent_section = get_intent_section(model_id)
        param_space = get_parameter_section(model_id)
        fallback = is_fallback(model_id)

        translations_map: dict[str, dict] = intent_section.get(
            "intent_translations", {}
        )
        prompt_style: str = intent_section.get("style", "hybrid")
        neg_effectiveness = _parse_effectiveness(
            intent_section.get("negative_prompt", {}).get("effectiveness", 0.5)
        )

        # 1. Parse individual intents
        intent_words = self._parse_intents(user_intent)

        # 2. Incorporate refinement context as extra intents
        if refinement_context:
            for action in refinement_context:
                extra = self._refinement_to_intent(action)
                if extra:
                    intent_words.append(extra)

        # 3. Look up each intent
        matched: list[tuple[str, dict]] = []  # (original_word, translation)
        unmatched: list[str] = []
        for word in intent_words:
            tr = self._lookup_intent(word, translations_map)
            if tr is not None:
                matched.append((word, tr))
            else:
                unmatched.append(word)

        # 4. Detect and resolve conflicts
        conflicts: list[ConflictResolution] = []
        suppressed_dims: set[str] = set()
        if len(matched) >= 2:
            raw_conflicts = self._detect_conflicts(matched)
            for ca, cb, dim in raw_conflicts:
                resolution = self._resolve_conflict(ca, cb, dim)
                conflicts.append(resolution)
                suppressed_dims.add(dim)

        # 5. Build parameter mutations
        param_mutations: list[ParameterMutation] = []
        prompt_mutations: list[PromptMutation] = []

        for word, tr in matched:
            # CFG
            cfg_dir = tr.get("cfg_direction")
            if cfg_dir and cfg_dir != "default" and "cfg_direction" not in suppressed_dims:
                value = self._direction_to_value(
                    cfg_dir, "cfg", param_space, workflow_state.get("cfg"),
                )
                if value is not None:
                    direction, mag = _DIRECTION_MAGNITUDE.get(
                        cfg_dir, ("down" if cfg_dir in _LOWER_DIRECTIONS else "up", "moderate")
                    )
                    param_mutations.append(ParameterMutation(
                        target="KSampler.cfg",
                        action="set",
                        value=value,
                        magnitude=mag,
                        reason=f"Intent '{word}' -> cfg {cfg_dir}",
                    ))

            # Steps
            steps_dir = tr.get("steps_direction")
            if steps_dir and "steps_direction" not in suppressed_dims:
                value = self._direction_to_value(
                    steps_dir, "steps", param_space, workflow_state.get("steps"),
                )
                if value is not None:
                    _, mag = _DIRECTION_MAGNITUDE.get(
                        steps_dir, ("up", "moderate")
                    )
                    param_mutations.append(ParameterMutation(
                        target="KSampler.steps",
                        action="set",
                        value=value,
                        magnitude=mag,
                        reason=f"Intent '{word}' -> steps {steps_dir}",
                    ))

            # Denoise
            denoise_dir = tr.get("denoise_direction")
            if denoise_dir and "denoise_direction" not in suppressed_dims:
                value = self._direction_to_value(
                    denoise_dir, "denoise", param_space, workflow_state.get("denoise"),
                )
                if value is not None:
                    _, mag = _DIRECTION_MAGNITUDE.get(
                        denoise_dir, ("down", "moderate")
                    )
                    param_mutations.append(ParameterMutation(
                        target="KSampler.denoise",
                        action="set",
                        value=value,
                        magnitude=mag,
                        reason=f"Intent '{word}' -> denoise {denoise_dir}",
                    ))

            # Sampler preference
            sampler_pref = tr.get("sampler_preference")
            if sampler_pref:
                param_mutations.append(ParameterMutation(
                    target="KSampler.sampler_name",
                    action="set",
                    value=sampler_pref,
                    reason=f"Intent '{word}' -> sampler {sampler_pref}",
                ))

            # Prompt additions (positive)
            prompt_add = tr.get("prompt_additions")
            if prompt_add:
                formatted = self._format_prompt_addition(prompt_add, prompt_style)
                prompt_mutations.append(PromptMutation(
                    target="positive_prompt",
                    action="append",
                    value=formatted,
                    reason=f"Intent '{word}'",
                ))

            # Negative prompt additions
            neg_add = tr.get("negative_additions")
            if neg_add and neg_effectiveness >= 0.3:
                formatted = self._format_prompt_addition(neg_add, prompt_style)
                prompt_mutations.append(PromptMutation(
                    target="negative_prompt",
                    action="append",
                    value=formatted,
                    reason=f"Intent '{word}'",
                ))

        # 6. Deduplicate parameter mutations (last one wins per target)
        param_mutations = self._deduplicate_mutations(param_mutations)

        # 7. Compute confidence
        confidence = self._compute_confidence(
            matched, unmatched, conflicts, fallback,
        )

        # 8. Build warnings
        warnings: list[str] = []
        if fallback:
            warnings.append(
                f"No dedicated profile for '{model_id}'. "
                f"Using fallback — results may be less precise."
            )
        for word in unmatched:
            warnings.append(
                f"Unknown intent '{word}' — no translation found in profile."
            )
        if suppressed_dims:
            for dim in sorted(suppressed_dims):
                warnings.append(
                    f"Conflicting intents on {dim} — see conflicts_resolved."
                )

        return IntentSpecification(
            model_id=model_id,
            parameter_mutations=param_mutations,
            prompt_mutations=prompt_mutations,
            confidence=confidence,
            conflicts_resolved=conflicts,
            warnings=warnings,
            using_fallback_profile=fallback,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_intents(user_intent: str) -> list[str]:
        """Split compound intents into individual intent words.

        Handles separators: "and", "but", ",", ";", "&", "+".
        Returns lowercased, stripped, non-empty tokens.
        """
        if not user_intent or not user_intent.strip():
            return []
        # Normalise separators
        text = user_intent.lower().strip()
        # Replace conjunction/separator words with a common delimiter
        text = re.sub(r"\s+(?:and|but|&|\+)\s+", ",", text)
        text = re.sub(r"\s*[;]\s*", ",", text)
        parts = [p.strip() for p in text.split(",")]
        return [p for p in parts if p]

    @staticmethod
    def _lookup_intent(
        intent_word: str, translations: dict[str, dict],
    ) -> dict | None:
        """Look up an intent word in the translations dict.

        Resolution order:
        1. Exact match on the intent word
        2. Synonym mapping -> then exact match
        3. Substring match (intent word contained in a translation key)
        """
        word = intent_word.strip().lower()
        # Exact match
        if word in translations:
            return translations[word]
        # Synonym
        canonical = _INTENT_SYNONYMS.get(word)
        if canonical and canonical in translations:
            return translations[canonical]
        # Substring: check if the word is a substring of any key
        for key, val in sorted(translations.items()):
            if word in key or key in word:
                return val
        return None

    @staticmethod
    def _detect_conflicts(
        matched: list[tuple[str, dict]],
    ) -> list[tuple[str, str, str]]:
        """Find conflicting direction pairs among matched translations.

        Returns list of (intent_word_a, intent_word_b, dimension).
        """
        conflicts: list[tuple[str, str, str]] = []
        for dim in _CONFLICT_DIMENSIONS:
            lower_intents: list[str] = []
            higher_intents: list[str] = []
            for word, tr in matched:
                direction = tr.get(dim)
                if not direction or direction == "default":
                    continue
                if direction in _LOWER_DIRECTIONS:
                    lower_intents.append(word)
                elif direction in _HIGHER_DIRECTIONS:
                    higher_intents.append(word)
            if lower_intents and higher_intents:
                conflicts.append((lower_intents[0], higher_intents[0], dim))
        return conflicts

    @staticmethod
    def _resolve_conflict(
        intent_a: str, intent_b: str, dimension: str,
    ) -> ConflictResolution:
        """Resolve a conflict between two intents on a given dimension."""
        key = (f"{dimension}:lower", f"{dimension}:higher")
        rule = _CONFLICT_RULES.get(key, {
            "strategy": "favor_first_intent",
            "explanation": (
                f"Conflicting intents on {dimension}. "
                f"Favoring first intent as default resolution."
            ),
        })
        return ConflictResolution(
            intent_a=intent_a,
            intent_b=intent_b,
            conflict_dimension=dimension,
            resolution_strategy=rule["strategy"],
            explanation=rule["explanation"],
        )

    @staticmethod
    def _direction_to_value(
        direction: str,
        param_name: str,
        param_space: dict[str, Any],
        current_value: float | None = None,
    ) -> float | None:
        """Convert a direction string to a concrete parameter value.

        Uses the parameter space's range, sweet_spot, and default to
        compute a value.  Never exceeds range bounds.  Prefers staying
        within the sweet_spot.
        """
        spec = param_space.get(param_name)
        if spec is None:
            return None

        default = spec.get("default")
        p_range = spec.get("range")
        sweet = spec.get("sweet_spot") or spec.get("img2img_sweet_spot")

        if p_range is None:
            return None

        range_low, range_high = float(p_range[0]), float(p_range[1])
        range_size = range_high - range_low

        if direction == "default":
            return float(default) if default is not None else None

        # Determine current anchor
        anchor: float
        if current_value is not None:
            anchor = float(current_value)
        elif default is not None:
            anchor = float(default)
        else:
            anchor = (range_low + range_high) / 2.0

        dm = _DIRECTION_MAGNITUDE.get(direction)
        if dm is None:
            # Infer from string
            if direction in _LOWER_DIRECTIONS:
                dm = ("down", "moderate")
            elif direction in _HIGHER_DIRECTIONS:
                dm = ("up", "moderate")
            else:
                return None

        up_or_down, magnitude = dm
        if up_or_down == "hold":
            return float(default) if default is not None else anchor

        fraction = _MAGNITUDE_FRACTION.get(magnitude, 0.25)
        delta = range_size * fraction

        if up_or_down == "down":
            raw = anchor - delta
        else:
            raw = anchor + delta

        # Clamp to range bounds
        raw = max(range_low, min(range_high, raw))

        # Prefer sweet_spot if available
        if sweet:
            sweet_low, sweet_high = float(sweet[0]), float(sweet[1])
            if up_or_down == "down":
                raw = max(raw, sweet_low)
            else:
                raw = min(raw, sweet_high)
            # Re-clamp after sweet_spot adjustment
            raw = max(range_low, min(range_high, raw))

        # Round to reasonable precision
        if param_name == "steps":
            return int(round(raw))
        return round(raw, 2)

    @staticmethod
    def _format_prompt_addition(text: str, style: str) -> str:
        """Format a prompt addition based on the model's prompt style.

        Styles:
        - natural_language: keep as-is (sentence fragments)
        - tag_based: ensure comma-separated tags, no prose
        - hybrid: keep as-is (works for both)
        """
        if not text:
            return ""
        text = text.strip()
        if style == "tag_based":
            # Ensure comma separation, strip excess whitespace
            tags = [t.strip() for t in text.replace(";", ",").split(",")]
            return ", ".join(t for t in tags if t)
        # natural_language and hybrid: return as-is
        return text

    @staticmethod
    def _deduplicate_mutations(
        mutations: list[ParameterMutation],
    ) -> list[ParameterMutation]:
        """Keep only the last mutation per target."""
        seen: dict[str, ParameterMutation] = {}
        for m in mutations:
            seen[m.target] = m
        return list(seen.values())

    @staticmethod
    def _compute_confidence(
        matched: list[tuple[str, dict]],
        unmatched: list[str],
        conflicts: list[ConflictResolution],
        fallback: bool,
    ) -> float:
        """Compute a confidence score 0-1 for the specification."""
        total = len(matched) + len(unmatched)
        if total == 0:
            return 0.3  # empty intent -> low confidence

        # Base confidence from match ratio
        match_ratio = len(matched) / total
        if match_ratio >= 1.0:
            base = 0.95
        elif match_ratio >= 0.5:
            base = 0.7 + (match_ratio - 0.5) * 0.4  # 0.7 - 0.9
        else:
            base = 0.3 + match_ratio * 0.8  # 0.3 - 0.7

        # Penalty for conflicts
        base -= len(conflicts) * 0.1

        # Penalty for fallback profile
        if fallback:
            base -= 0.15

        return max(0.0, min(1.0, round(base, 2)))

    @staticmethod
    def _refinement_to_intent(action: dict[str, Any]) -> str | None:
        """Convert a Verify Agent refinement action to an intent word.

        Recognizes common refinement types and maps them to canonical
        intent words that the translation table understands.
        """
        action_type = action.get("type", "")
        target = action.get("target", "")
        recommendation = action.get("recommendation", "")

        # Direct parameter adjustments
        if action_type == "adjust_params":
            if "cfg" in target.lower():
                if "lower" in recommendation.lower():
                    return "dreamier"
                if "higher" in recommendation.lower():
                    return "sharper"
            if "steps" in target.lower():
                if "higher" in recommendation.lower():
                    return "more detailed"
                if "lower" in recommendation.lower():
                    return "faster"

        # Quality-oriented refinements
        if action_type == "improve_quality":
            return "more detailed"
        if action_type == "reduce_artifacts":
            return "cleaner"
        if action_type == "change_style":
            return recommendation.lower() if recommendation else None

        return None


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _parse_effectiveness(value: Any) -> float:
    """Parse a negative_prompt effectiveness value to a float.

    Profiles may store this as a float (0.65), a string ("low"),
    or a descriptive string ("0.5").
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Try numeric parse first
        try:
            return float(value)
        except ValueError:
            pass
        low_words = {"low", "minimal", "none", "negligible"}
        if value.lower() in low_words:
            return 0.1
        high_words = {"high", "strong", "significant"}
        if value.lower() in high_words:
            return 0.8
    return 0.5  # default moderate
