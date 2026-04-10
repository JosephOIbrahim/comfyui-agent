"""ExperienceAccumulator — manages the three learning phases.

Phase 1 (0-30 gens):   Prior rules only
Phase 2 (30-100 gens): Blended prior + experience
Phase 3 (100+ gens):   Experience-dominant

The accumulator stores ExperienceChunks and retrieves relevant
ones by context signature matching.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_save_lock = threading.Lock()

from .chunk import ExperienceChunk, QualityScore
from .signature import GenerationContextSignature


class LearningPhase(Enum):
    """Three learning phases based on accumulated experience."""

    PRIOR = "prior"  # 0-30 generations
    BLENDED = "blended"  # 30-100 generations
    EXPERIENCED = "experienced"  # 100+ generations


@dataclass
class RetrievalResult:
    """Result of an experience retrieval query."""

    matches: list[ExperienceChunk] = field(default_factory=list)
    query_signature: GenerationContextSignature | None = None
    best_quality: float = 0.0
    avg_quality: float = 0.0
    pattern_parameters: dict[str, Any] = field(default_factory=dict)


class ExperienceAccumulator:
    """Manages accumulated generation experience.

    Stores ExperienceChunks, retrieves by similarity, and manages
    the transition through learning phases.

    Thread-safety: all mutations and reads of _chunks are protected by
    _chunks_lock (threading.Lock). _save_lock remains for atomic writes.
    """

    # Phase thresholds
    PHASE_2_THRESHOLD = 30
    PHASE_3_THRESHOLD = 100

    def __init__(self, max_chunks: int = 10000):
        self._chunks: list[ExperienceChunk] = []
        self._max_chunks = max_chunks
        self._chunks_lock = threading.Lock()

    @property
    def generation_count(self) -> int:
        with self._chunks_lock:
            return len(self._chunks)

    @property
    def learning_phase(self) -> LearningPhase:
        count = self.generation_count
        if count < self.PHASE_2_THRESHOLD:
            return LearningPhase.PRIOR
        if count < self.PHASE_3_THRESHOLD:
            return LearningPhase.BLENDED
        return LearningPhase.EXPERIENCED

    @property
    def experience_weight(self) -> float:
        """How much to weight experience vs prior rules (0.0 - 1.0).

        Phase 1: 0.0 (all prior)
        Phase 2: Linear ramp from 0.0 to 0.7
        Phase 3: 0.85 (experience-dominant, but priors still inform)
        """
        count = self.generation_count
        if count < self.PHASE_2_THRESHOLD:
            return 0.0
        if count < self.PHASE_3_THRESHOLD:
            progress = (count - self.PHASE_2_THRESHOLD) / (
                self.PHASE_3_THRESHOLD - self.PHASE_2_THRESHOLD
            )
            return progress * 0.7
        return 0.85

    def record(self, chunk: ExperienceChunk) -> None:
        """Record a new experience chunk.

        Enforces max_chunks by removing oldest low-quality entries.
        """
        with self._chunks_lock:
            self._chunks.append(chunk)

            if len(self._chunks) > self._max_chunks:
                # Remove the lowest-quality (oldest if tied) chunk.
                # O(n) linear scan instead of O(n log n) sort — reduces lock-hold
                # time so concurrent retrieve() calls aren't stalled. (Cycle 33 fix)
                min_idx = min(
                    range(len(self._chunks)),
                    key=lambda i: (self._chunks[i].quality.overall, self._chunks[i].timestamp),
                )
                self._chunks.pop(min_idx)

    def retrieve(
        self,
        signature: GenerationContextSignature,
        top_k: int = 10,
        min_similarity: float = 0.3,
    ) -> RetrievalResult:
        """Retrieve relevant experience by context signature.

        Returns chunks sorted by relevance (similarity * quality * decay).
        """
        result = RetrievalResult(query_signature=signature)

        with self._chunks_lock:
            snapshot = list(self._chunks)

        scored = []
        for chunk in snapshot:
            chunk_sig = GenerationContextSignature.from_workflow(
                _chunk_to_workflow_proxy(chunk)
            )
            sim = signature.similarity(chunk_sig)
            if sim >= min_similarity:
                relevance = sim * chunk.quality.overall * chunk.decay_weight
                scored.append((relevance, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        result.matches = [chunk for _, chunk in scored[:top_k]]

        if result.matches:
            qualities = [c.quality.overall for c in result.matches if c.quality.is_scored]
            if qualities:
                result.best_quality = max(qualities)
                result.avg_quality = sum(qualities) / len(qualities)

            # Extract most common successful parameters
            result.pattern_parameters = _extract_patterns(result.matches)

        return result

    def get_successful_chunks(self, min_quality: float = 0.5) -> list[ExperienceChunk]:
        """Get all chunks above a quality threshold."""
        with self._chunks_lock:
            return [
                c for c in self._chunks
                if c.quality.overall >= min_quality and c.succeeded
            ]

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about accumulated experience."""
        with self._chunks_lock:
            total = len(self._chunks)
            succeeded = sum(1 for c in self._chunks if c.succeeded)
            scored = [c.quality.overall for c in self._chunks if c.quality.is_scored]

        phase = self.learning_phase
        weight = self.experience_weight
        return {
            "total_generations": total,
            "successful": succeeded,
            "failed": total - succeeded,
            "learning_phase": phase.value,
            "experience_weight": round(weight, 3),
            "avg_quality": round(sum(scored) / len(scored), 3) if scored else 0.0,
            "best_quality": round(max(scored), 3) if scored else 0.0,
        }

    # ── Persistence ────────────────────────────────────────────────

    def save(self, path: str) -> int:
        """Save all chunks to a JSONL file. Returns count saved.

        Uses an atomic write (write to .tmp, then os.replace) so an unclean
        shutdown between writes never leaves a truncated file.  A module-level
        lock prevents two concurrent callers from interleaving their writes.
        """
        import json
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(str(p) + ".tmp")
        with self._chunks_lock:
            snapshot = list(self._chunks)
        with _save_lock:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for chunk in snapshot:
                    f.write(json.dumps(chunk.to_dict(), sort_keys=True) + "\n")
            os.replace(tmp_path, p)
        return len(snapshot)

    @classmethod
    def load(cls, path: str, max_chunks: int = 10000) -> ExperienceAccumulator:
        """Load chunks from a JSONL file. Returns a new accumulator."""
        import json
        from pathlib import Path
        acc = cls(max_chunks=max_chunks)
        p = Path(path)
        if not p.exists():
            return acc
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    chunk = ExperienceChunk.from_dict(data)
                    acc._chunks.append(chunk)
                except json.JSONDecodeError:
                    # Malformed JSON line — skip silently.
                    continue
                except Exception as _e:
                    # Deserialization error — log so data corruption is visible. (Cycle 33 fix)
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "Failed to load experience chunk: %s", _e
                    )
                    continue
        # Enforce max_chunks (load() is called at init time; no lock needed here
        # since no other thread has a reference to acc yet)
        if len(acc._chunks) > max_chunks:
            acc._chunks.sort(key=lambda c: (c.quality.overall, c.timestamp))
            acc._chunks = acc._chunks[-max_chunks:]
        return acc


def _chunk_to_workflow_proxy(chunk: ExperienceChunk) -> dict[str, Any]:
    """Convert an ExperienceChunk to a minimal workflow-like dict for signature matching."""
    proxy: dict[str, Any] = {}

    if chunk.checkpoint:
        proxy["loader"] = {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": chunk.checkpoint},
        }

    if chunk.parameters:
        for node_id, params in chunk.parameters.items():
            if isinstance(params, dict):
                proxy[node_id] = {
                    "class_type": params.get("class_type", "KSampler"),
                    "inputs": {k: v for k, v in params.items() if k != "class_type"},
                }

    return proxy


def _extract_patterns(chunks: list[ExperienceChunk]) -> dict[str, Any]:
    """Extract most common parameter values from successful chunks."""
    param_counts: dict[str, dict[str, int]] = {}

    for chunk in chunks:
        if not chunk.succeeded:
            continue
        for node_id, params in chunk.parameters.items():
            if isinstance(params, dict):
                for k, v in params.items():
                    key = f"{node_id}.{k}"
                    val_str = str(v)
                    param_counts.setdefault(key, {})
                    param_counts[key][val_str] = param_counts[key].get(val_str, 0) + 1

    # Pick most common value for each parameter
    patterns = {}
    for key, counts in param_counts.items():
        if counts:
            best_val = max(counts, key=counts.get)
            patterns[key] = best_val

    return patterns
