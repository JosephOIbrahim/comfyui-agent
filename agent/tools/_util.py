"""Shared utilities for tool modules.

He2025 alignment: deterministic JSON serialization ensures
same inputs -> same outputs across sessions and runs.
"""

import json as _json


def to_json(obj, **kwargs) -> str:
    """Serialize to JSON with deterministic key ordering."""
    kwargs.setdefault("sort_keys", True)
    return _json.dumps(obj, **kwargs)
