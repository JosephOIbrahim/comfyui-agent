"""Autonomous Pipeline — end-to-end generation from intent to learning.

Wires all cognitive components into a single autonomous pipeline:
intent → compose → predict → execute → evaluate → learn
"""

from .autonomous import AutonomousPipeline, PipelineConfig, PipelineResult, PipelineStage

__all__ = [
    "AutonomousPipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
]
