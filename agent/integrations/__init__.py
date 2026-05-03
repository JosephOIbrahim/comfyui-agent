"""External-system integrations for the Cozy agent.

Each integration registers as a subscriber on `CognitiveWorkflowStage` (see
W2.1 in the Cozy plan) and translates between the stage's internal events
and the external system's wire format.
"""

from .moneta import (
    MonetaAdapter,
    MonetaAdapterConfig,
    MonetaConfigError,
    from_env,
)

__all__ = [
    "MonetaAdapter",
    "MonetaAdapterConfig",
    "MonetaConfigError",
    "from_env",
]
