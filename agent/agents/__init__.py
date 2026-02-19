"""MoE Specialist Agents for the ComfyUI SUPER DUPER Agent.

Specialist agents that the Router delegates to:
  IntentAgent   -- translates artistic language into parameter specs
  VerifyAgent   -- model-relative quality judgment and iteration control
  Router        -- authority delegation and loop control
"""

from .intent_agent import (
    ConflictResolution,
    IntentAgent,
    IntentSpecification,
    ParameterMutation,
    PromptMutation,
)
from .router import (
    AUTHORITY_RULES,
    DELEGATION_SEQUENCES,
    Router,
    RouterContext,
)
from .verify_agent import (
    RefinementAction,
    VerificationResult,
    VerifyAgent,
)

__all__ = [
    "AUTHORITY_RULES",
    "ConflictResolution",
    "DELEGATION_SEQUENCES",
    "IntentAgent",
    "IntentSpecification",
    "ParameterMutation",
    "PromptMutation",
    "RefinementAction",
    "Router",
    "RouterContext",
    "VerificationResult",
    "VerifyAgent",
]
