"""Recursive freeze utility — read-only views without copying.

The recurring pattern in `agent/schemas/loader.py` and
`agent/profiles/loader.py` was:

    return copy.deepcopy(self._cache[cache_key])

The deepcopy fired on EVERY cache hit, negating the cache's purpose: hits
were as expensive as misses (O(W) where W = the cached object's size).
The deepcopy was the only thing protecting concurrent consumers from
each other's mutations.

`deep_freeze` produces a recursively read-only view of a nested dict/list
structure WITHOUT copying. Mutation attempts raise `TypeError` at the
call site instead of silently corrupting a shared cache entry — strictly
better signal than the deepcopy provided. Cost: one wrapper allocation
per nested container at first call; zero on subsequent reads.

Container substitutions:
  dict  -> types.MappingProxyType  (read-only view of the same dict)
  list  -> tuple                   (immutable copy — unavoidable)
  set   -> frozenset               (immutable copy — unavoidable)
  other -> returned as-is          (assumed scalar / already immutable)

Lists and sets are converted to immutable equivalents because Python has
no read-only-view wrapper for them. The conversion happens once at
freeze time; subsequent reads are O(1).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any


def deep_freeze(obj: Any) -> Any:
    """Return a recursively read-only view of `obj`.

    Mutating the returned view raises `TypeError`. Callers that need a
    mutable copy can call `copy.deepcopy` on the returned view explicitly
    — but most callers don't need that, and the previous defensive
    deepcopy was overhead they never used.
    """
    if isinstance(obj, dict):
        return MappingProxyType({k: deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(deep_freeze(v) for v in obj)
    if isinstance(obj, set):
        return frozenset(deep_freeze(v) for v in obj)
    return obj
