"""Autonomous Pipeline — end-to-end generation from intent to learning.

Wires all cognitive components into a single autonomous pipeline:
intent → compose → predict → execute → evaluate → learn
"""

from .autonomous import AutonomousPipeline, PipelineConfig, PipelineResult, PipelineStage
from ..experience.accumulator import ExperienceAccumulator
from ..prediction.cwm import CognitiveWorldModel
from ..prediction.arbiter import SimulationArbiter
from ..prediction.counterfactual import CounterfactualGenerator
from agent.config import EXPERIENCE_FILE


def create_default_pipeline() -> AutonomousPipeline:
    """Construct an AutonomousPipeline with default singleton components.

    Instantiates all four cognitive components fresh, loading any previously
    saved experience from EXPERIENCE_FILE so learning persists across sessions.
    The caller owns their lifetime — for MCP server use, call once at startup
    and keep the returned pipeline alive for the server's lifetime (Option A).

    Two calls return two independent pipelines with independent
    accumulator state. There is no implicit module-level singleton.

    Returns:
        AutonomousPipeline ready to call .run(PipelineConfig(...)).
    """
    accumulator = ExperienceAccumulator.load(str(EXPERIENCE_FILE))
    cwm = CognitiveWorldModel()
    arbiter = SimulationArbiter()
    cf_gen = CounterfactualGenerator()
    return AutonomousPipeline(
        accumulator=accumulator,
        cwm=cwm,
        arbiter=arbiter,
        counterfactual_gen=cf_gen,
    )


__all__ = [
    "AutonomousPipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
    "create_default_pipeline",
]
