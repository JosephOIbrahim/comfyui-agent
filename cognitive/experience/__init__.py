"""Experience Accumulator — structured learning from generation outcomes.

Every generation is an experiment. This module captures the full
(params -> outcome) tuple, builds context signatures for fast
matching, and manages three learning phases.
"""

from .chunk import ExperienceChunk, QualityScore
from .signature import GenerationContextSignature
from .accumulator import ExperienceAccumulator, LearningPhase

__all__ = [
    "ExperienceChunk",
    "QualityScore",
    "GenerationContextSignature",
    "ExperienceAccumulator",
    "LearningPhase",
]
