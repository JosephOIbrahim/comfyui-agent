"""Model Profile Registry — the ACCESS oracle for model-specific knowledge.

Agents never memorize model quirks. They query this registry at runtime.
Every model-specific decision (steps, CFG, resolution, prompt style,
known artifacts) is encoded in YAML profiles that live alongside this
module. Unknown models get a conservative fallback without crashing.

Resolution order for ``load_profile(model_id)``:

1. Exact match: ``<model_id>.yaml`` in PROFILES_DIR
2. Architecture fallback: ``default_dit.yaml`` / ``default_unet.yaml``
   / ``default_video.yaml`` (tried in order)
3. Minimal defaults: hardcoded conservative values that work everywhere

Profiles are cached in memory with thread-safe access so repeated
lookups within a single session are effectively free.
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROFILES_DIR: Path = Path(__file__).parent
"""Directory where YAML profile files are stored."""

_FALLBACK_CANDIDATES: list[str] = [
    "default_dit",
    "default_unet",
    "default_video",
]

# ---------------------------------------------------------------------------
# Thread-safe cache
# ---------------------------------------------------------------------------

_cache: dict[str, dict[str, Any]] = {}
_cache_lock: threading.Lock = threading.Lock()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a YAML file with explicit UTF-8 encoding."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Profile at {path} did not parse to a dict")
    return data


def _minimal_defaults(model_id: str) -> dict[str, Any]:
    """Absolute minimum profile with conservative safe values.

    Returned when no YAML file matches and no architecture fallback
    exists.  Every field that downstream code might access is present
    so callers never hit a ``KeyError``.
    """
    return {
        "meta": {
            "model_id": model_id,
            "display_name": model_id,
            "model_class": "unknown",
            "base_arch": "unknown",
            "modality": "image",
            "_is_fallback": True,
            "_is_minimal": True,
        },
        "prompt_engineering": {
            "style": "hybrid",
            "positive_prompt": {
                "structure": "description_first",
                "keyword_sensitivity": 0.5,
                "effective_patterns": [],
                "token_weighting": "none",
                "max_effective_tokens": 200,
            },
            "negative_prompt": {
                "required_base": "blurry, low quality, distorted",
                "style": "exclusion_list",
                "effectiveness": 0.5,
            },
            "intent_translations": {},
        },
        "parameter_space": {
            "steps": {
                "default": 20,
                "range": [10, 50],
                "sweet_spot": [15, 30],
            },
            "cfg": {
                "default": 7.0,
                "range": [1.0, 15.0],
                "sweet_spot": [5.0, 9.0],
            },
            "sampler": {
                "recommended": ["euler", "dpmpp_2m"],
                "avoid": [],
                "scheduler": "karras",
            },
            "resolution": {
                "native": [512, 512],
                "supported_ratios": ["1:1"],
                "upscale_friendly": True,
            },
            "denoise": {
                "default": 1.0,
                "img2img_sweet_spot": [0.4, 0.7],
            },
            "lora_behavior": {
                "max_simultaneous": 2,
                "strength_range": [0.1, 1.0],
                "default_strength": 0.8,
                "interaction_model": "additive",
                "known_conflicts": [],
            },
        },
        "quality_signatures": {
            "expected_characteristics": [],
            "known_artifacts": [],
            "quality_floor": {
                "description": "Coherent output with recognizable subject matter",
                "reference_score": 0.5,
            },
            "iteration_signals": {
                "needs_more_steps": [],
                "needs_lower_cfg": [],
                "needs_higher_cfg": [],
                "needs_reprompt": [],
                "needs_inpaint": [],
                "model_limitation": [],
            },
        },
    }


def _load_fallback(model_id: str) -> dict[str, Any] | None:
    """Try architecture-level default profiles in priority order.

    Returns ``None`` if no fallback YAML exists on disk.
    """
    for candidate in _FALLBACK_CANDIDATES:
        path = PROFILES_DIR / f"{candidate}.yaml"
        if path.is_file():
            profile = _load_yaml(path)
            profile.setdefault("meta", {})
            profile["meta"]["_is_fallback"] = True
            profile["meta"]["model_id"] = model_id
            return profile
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_profile(model_id: str) -> dict[str, Any]:
    """Load (and cache) a model profile by *model_id*.

    Resolution order:
    1. Exact YAML match in ``PROFILES_DIR``
    2. Architecture fallback (``default_dit`` / ``default_unet`` /
       ``default_video``)
    3. Hardcoded minimal defaults

    The returned dict is a deep copy so callers can mutate freely.
    """
    with _cache_lock:
        if model_id in _cache:
            return copy.deepcopy(_cache[model_id])

    # --- Resolve outside the lock (I/O) --------------------------------
    exact_path = PROFILES_DIR / f"{model_id}.yaml"

    if exact_path.is_file():
        profile = _load_yaml(exact_path)
        profile.setdefault("meta", {})
        profile["meta"].setdefault("model_id", model_id)
    else:
        profile = _load_fallback(model_id)
        if profile is None:
            profile = _minimal_defaults(model_id)

    # --- Store and return ------------------------------------------------
    with _cache_lock:
        _cache[model_id] = profile

    return copy.deepcopy(profile)


def get_intent_section(model_id: str) -> dict[str, Any]:
    """Return the ``prompt_engineering`` section of a profile."""
    return load_profile(model_id).get("prompt_engineering", {})


def get_parameter_section(model_id: str) -> dict[str, Any]:
    """Return the ``parameter_space`` section of a profile."""
    return load_profile(model_id).get("parameter_space", {})


def get_quality_section(model_id: str) -> dict[str, Any]:
    """Return the ``quality_signatures`` section of a profile."""
    return load_profile(model_id).get("quality_signatures", {})


def is_fallback(model_id: str) -> bool:
    """Check whether *model_id* resolved to a fallback profile."""
    profile = load_profile(model_id)
    meta = profile.get("meta", {})
    return bool(meta.get("_is_fallback", False))


def list_profiles() -> list[str]:
    """List model-ids for all concrete YAML profiles on disk.

    Excludes ``_schema.yaml`` and ``default_*.yaml`` helper files.
    Returns a deterministically sorted list.
    """
    profiles: list[str] = []
    for path in PROFILES_DIR.glob("*.yaml"):
        stem = path.stem
        if stem.startswith("_") or stem.startswith("default_"):
            continue
        profiles.append(stem)
    profiles.sort()
    return profiles


def clear_cache() -> None:
    """Drop all cached profiles.  Intended for testing."""
    with _cache_lock:
        _cache.clear()
