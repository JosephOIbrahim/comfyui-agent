"""DeltaLayer — non-destructive mutation layer with LIVRPS opinion.

Each delta captures a set of mutations to workflow node inputs,
tagged with a LIVRPS opinion tier and protected by SHA-256
integrity checking. Deltas are never applied in place — they
are composed by the CognitiveGraphEngine at resolution time.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

Opinion = Literal["P", "R", "V", "I", "L", "S"]

LIVRPS_PRIORITY: dict[str, int] = {
    "P": 1,  # Payloads — deep archive, loaded on demand
    "R": 2,  # References — base templates, prior rules
    "V": 3,  # VariantSets — context-dependent alternatives
    "I": 4,  # Inherits — experience-derived patterns
    "L": 5,  # Local — current session edits (strongest creative opinion)
    "S": 6,  # Safety — structural constraints (INVERTED: always wins)
}


def _compute_hash(opinion: str, mutations: dict[str, dict[str, Any]]) -> str:
    """SHA-256 of opinion + deterministic JSON of mutations."""
    payload = json.dumps(
        {"opinion": opinion, "mutations": mutations},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class DeltaLayer:
    """Non-destructive mutation layer with LIVRPS opinion and integrity check.

    Attributes:
        layer_id: Unique identifier for this layer.
        opinion: LIVRPS tier — determines resolution priority.
        timestamp: UTC timestamp at creation.
        description: Human-readable description of what changed.
        mutations: {node_id: {param_name: value, ...}, ...}
            If a node_id includes "class_type" as a key, it signals
            node injection (new node not in base graph).
        creation_hash: SHA-256 computed at creation for tamper detection.
    """

    layer_id: str
    opinion: Opinion
    timestamp: float
    description: str
    mutations: dict[str, dict[str, Any]]
    creation_hash: str = field(default="", repr=False)

    def __post_init__(self):
        if not self.creation_hash:
            self.creation_hash = _compute_hash(self.opinion, self.mutations)

    @property
    def layer_hash(self) -> str:
        """Recompute hash from current state.

        Compare with creation_hash to detect tampering.
        """
        return _compute_hash(self.opinion, self.mutations)

    @property
    def is_intact(self) -> bool:
        """True if layer has not been tampered with since creation."""
        return self.creation_hash == self.layer_hash

    @property
    def priority(self) -> int:
        """Numeric priority from LIVRPS_PRIORITY."""
        return LIVRPS_PRIORITY[self.opinion]

    @classmethod
    def create(
        cls,
        mutations: dict[str, dict[str, Any]],
        opinion: Opinion = "L",
        description: str = "",
        layer_id: str | None = None,
    ) -> DeltaLayer:
        """Factory method with auto-generated layer_id and timestamp."""
        return cls(
            layer_id=layer_id or uuid.uuid4().hex[:12],
            opinion=opinion,
            timestamp=time.time(),
            description=description,
            mutations=mutations,
        )
