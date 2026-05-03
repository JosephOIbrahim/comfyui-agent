"""Cozy autonomous harness — long-running self-healing execution loop.

The harness wraps AutoresearchRunner and adds:
  - Checkpoint persistence (atomic stage + ratchet + experience triple)
  - Self-healing error classification via constitution.self_healing_ladder
  - Bounded-failure ladder (TRANSIENT → backoff, RECOVERABLE → repair, TERMINAL → halt)
  - MetaAgent integration (Tier-2 improvements gated by the ratchet)

Entry points:
  CozyLoop(...).run()                  — programmatic
  agent run --autonomous --hours N     — CLI

See .claude/COZY_CONSTITUTION.md for the governing doctrine.
"""

from .cli_callables import (
    make_execute_fn,
    make_propose_fn,
)
from .cozy_loop import (
    CozyLoop,
    CozyLoopConfig,
    CozyLoopResult,
    HealthSnapshot,
)

__all__ = [
    "CozyLoop",
    "CozyLoopConfig",
    "CozyLoopResult",
    "HealthSnapshot",
    "make_execute_fn",
    "make_propose_fn",
]
