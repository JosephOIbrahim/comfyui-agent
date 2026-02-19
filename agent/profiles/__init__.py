"""Model Profile Registry — public API.

Re-exports the loader functions so callers can do::

    from agent.profiles import load_profile, list_profiles
"""

from .loader import (
    clear_cache,
    get_intent_section,
    get_parameter_section,
    get_quality_section,
    is_fallback,
    list_profiles,
    load_profile,
)

__all__ = [
    "clear_cache",
    "get_intent_section",
    "get_parameter_section",
    "get_quality_section",
    "is_fallback",
    "list_profiles",
    "load_profile",
]
