"""Tests for the Experience Accumulator.

[EXPERIENCE x CRUCIBLE] — Tests for ExperienceChunk, signatures,
context matching, learning phases, and temporal decay.
"""

import time

import pytest

from cognitive.experience.chunk import ExperienceChunk, QualityScore
from cognitive.experience.signature import GenerationContextSignature
from cognitive.experience.accumulator import (
    ExperienceAccumulator,
    LearningPhase,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_workflow():
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned.safetensors"}},
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42, "steps": 20, "cfg": 7.5,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
            },
        },
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512}},
    }


@pytest.fixture
def good_chunk():
    return ExperienceChunk(
        model_family="SD1.5",
        checkpoint="v1-5-pruned.safetensors",
        prompt="a landscape",
        parameters={"4": {"steps": 20, "cfg": 7.5, "sampler_name": "euler"}},
        output_filenames=["output_001.png"],
        quality=QualityScore(overall=0.8, technical=0.7, aesthetic=0.9),
    )


@pytest.fixture
def accumulator():
    return ExperienceAccumulator()


# ---------------------------------------------------------------------------
# ExperienceChunk
# ---------------------------------------------------------------------------

class TestExperienceChunk:

    def test_chunk_creation(self, good_chunk):
        assert good_chunk.chunk_id != ""
        assert good_chunk.timestamp > 0
        assert good_chunk.succeeded is True

    def test_failed_chunk(self):
        chunk = ExperienceChunk(error="CUDA OOM")
        assert chunk.succeeded is False

    def test_no_output_not_succeeded(self):
        chunk = ExperienceChunk()
        assert chunk.succeeded is False

    def test_decay_weight_fresh(self, good_chunk):
        # Just created — decay weight should be ~1.0
        assert good_chunk.decay_weight > 0.99

    def test_decay_weight_old(self):
        chunk = ExperienceChunk(timestamp=time.time() - 14 * 24 * 3600)  # 14 days ago
        assert chunk.decay_weight < 0.3  # 2 half-lives

    def test_context_match_same_family(self):
        a = ExperienceChunk(model_family="SD1.5", checkpoint="test.safetensors")
        b = ExperienceChunk(model_family="SD1.5", checkpoint="test.safetensors")
        assert a.matches_context(b) is True

    def test_context_match_different_family(self):
        a = ExperienceChunk(model_family="SD1.5")
        b = ExperienceChunk(model_family="SDXL")
        assert a.matches_context(b) is False


# ---------------------------------------------------------------------------
# QualityScore
# ---------------------------------------------------------------------------

class TestQualityScore:

    def test_default_not_scored(self):
        q = QualityScore()
        assert q.is_scored is False

    def test_scored(self):
        q = QualityScore(overall=0.5)
        assert q.is_scored is True

    def test_clamps_above_one(self):
        """Cycle 27 fix: overall > 1.0 must be clamped to 1.0, not stored raw."""
        q = QualityScore(overall=1.5, technical=2.0, aesthetic=99.0, prompt_adherence=3.3)
        assert q.overall == 1.0
        assert q.technical == 1.0
        assert q.aesthetic == 1.0
        assert q.prompt_adherence == 1.0

    def test_clamps_below_zero(self):
        """Negative scores must be clamped to 0.0."""
        q = QualityScore(overall=-0.5, technical=-1.0)
        assert q.overall == 0.0
        assert q.technical == 0.0

    def test_valid_range_unchanged(self):
        """Values already in [0, 1] must not be altered."""
        q = QualityScore(overall=0.75, technical=0.6, aesthetic=0.9, prompt_adherence=0.5)
        assert q.overall == pytest.approx(0.75)
        assert q.technical == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# GenerationContextSignature
# ---------------------------------------------------------------------------

class TestSignature:

    def test_from_workflow(self, sample_workflow):
        sig = GenerationContextSignature.from_workflow(sample_workflow)
        assert sig.model_family == "SD1.5"
        assert sig.resolution_bucket == "512x512"
        assert sig.cfg_bucket == "medium"
        assert sig.steps_bucket == "normal"
        assert sig.sampler == "euler"
        assert sig.scheduler == "normal"
        assert sig.denoise_bucket == "high"

    def test_similarity_identical(self, sample_workflow):
        sig1 = GenerationContextSignature.from_workflow(sample_workflow)
        sig2 = GenerationContextSignature.from_workflow(sample_workflow)
        assert sig1.similarity(sig2) == 1.0

    def test_similarity_different(self):
        sig1 = GenerationContextSignature(
            model_family="SD1.5", cfg_bucket="low", steps_bucket="few",
        )
        sig2 = GenerationContextSignature(
            model_family="SDXL", cfg_bucket="high", steps_bucket="many",
        )
        assert sig1.similarity(sig2) == 0.0

    def test_similarity_partial(self):
        sig1 = GenerationContextSignature(
            model_family="SD1.5", cfg_bucket="medium", steps_bucket="normal",
            sampler="euler",
        )
        sig2 = GenerationContextSignature(
            model_family="SD1.5", cfg_bucket="medium", steps_bucket="many",
            sampler="dpmpp_2m",
        )
        sim = sig1.similarity(sig2)
        assert 0.0 < sim < 1.0

    def test_empty_signatures(self):
        sig1 = GenerationContextSignature()
        sig2 = GenerationContextSignature()
        assert sig1.similarity(sig2) == 0.0

    def test_cfg_buckets(self):
        low = GenerationContextSignature.from_workflow({
            "1": {"class_type": "KSampler", "inputs": {"cfg": 3.0}},
        })
        high = GenerationContextSignature.from_workflow({
            "1": {"class_type": "KSampler", "inputs": {"cfg": 12.0}},
        })
        assert low.cfg_bucket == "low"
        assert high.cfg_bucket == "high"

    def test_feature_detection(self):
        wf = {
            "1": {"class_type": "ControlNetApply", "inputs": {}},
            "2": {"class_type": "LoraLoader", "inputs": {}},
            "3": {"class_type": "IPAdapterApply", "inputs": {}},
        }
        sig = GenerationContextSignature.from_workflow(wf)
        assert sig.has_controlnet is True
        assert sig.has_lora is True
        assert sig.has_ipadapter is True


# ---------------------------------------------------------------------------
# ExperienceAccumulator
# ---------------------------------------------------------------------------

class TestAccumulator:

    def test_empty_accumulator(self, accumulator):
        assert accumulator.generation_count == 0
        assert accumulator.learning_phase == LearningPhase.PRIOR
        assert accumulator.experience_weight == 0.0

    def test_record_chunk(self, accumulator, good_chunk):
        accumulator.record(good_chunk)
        assert accumulator.generation_count == 1

    def test_phase_transitions(self, accumulator):
        # Phase 1: Prior (0-29)
        for i in range(29):
            accumulator.record(ExperienceChunk(chunk_id=f"c{i}"))
        assert accumulator.learning_phase == LearningPhase.PRIOR

        # Phase 2: Blended (30-99)
        accumulator.record(ExperienceChunk(chunk_id="c30"))
        assert accumulator.learning_phase == LearningPhase.BLENDED
        # At exactly 30, weight is 0.0 (start of ramp). Add one more to get > 0.
        accumulator.record(ExperienceChunk(chunk_id="c31"))
        assert accumulator.experience_weight > 0.0

        # Phase 3: Experienced (100+)
        for i in range(70):
            accumulator.record(ExperienceChunk(chunk_id=f"c{31+i}"))
        assert accumulator.learning_phase == LearningPhase.EXPERIENCED
        assert accumulator.experience_weight == 0.85

    def test_experience_weight_ramp(self, accumulator):
        """Weight ramps from 0.0 to 0.7 during Phase 2."""
        for i in range(30):
            accumulator.record(ExperienceChunk(chunk_id=f"c{i}"))
        w_start = accumulator.experience_weight
        assert w_start == 0.0

        for i in range(35):
            accumulator.record(ExperienceChunk(chunk_id=f"c{30+i}"))
        w_mid = accumulator.experience_weight
        assert 0.0 < w_mid < 0.7

    def test_retrieval(self, accumulator, sample_workflow):
        # Record some chunks with quality
        for i in range(5):
            chunk = ExperienceChunk(
                chunk_id=f"c{i}",
                model_family="SD1.5",
                checkpoint="v1-5-pruned.safetensors",
                parameters={"4": {"steps": 20, "cfg": 7.5, "sampler_name": "euler", "class_type": "KSampler"}},
                output_filenames=[f"out_{i}.png"],
                quality=QualityScore(overall=0.7 + i * 0.05),
            )
            accumulator.record(chunk)

        sig = GenerationContextSignature.from_workflow(sample_workflow)
        result = accumulator.retrieve(sig, top_k=3)
        assert len(result.matches) <= 3
        assert result.best_quality > 0

    def test_get_successful_chunks(self, accumulator):
        accumulator.record(ExperienceChunk(
            output_filenames=["ok.png"],
            quality=QualityScore(overall=0.8),
        ))
        accumulator.record(ExperienceChunk(error="fail"))
        accumulator.record(ExperienceChunk(
            output_filenames=["low.png"],
            quality=QualityScore(overall=0.3),
        ))

        good = accumulator.get_successful_chunks(min_quality=0.5)
        assert len(good) == 1
        assert good[0].quality.overall == 0.8

    def test_get_stats(self, accumulator, good_chunk):
        accumulator.record(good_chunk)
        stats = accumulator.get_stats()
        assert stats["total_generations"] == 1
        assert stats["successful"] == 1
        assert stats["learning_phase"] == "prior"

    def test_max_chunks_enforced(self):
        acc = ExperienceAccumulator(max_chunks=5)
        for i in range(10):
            acc.record(ExperienceChunk(
                chunk_id=f"c{i}",
                quality=QualityScore(overall=i * 0.1),
            ))
        assert acc.generation_count == 5

    def test_50_mock_generations(self, accumulator):
        """50 mock generations produce valid ExperienceChunks."""
        for i in range(50):
            chunk = ExperienceChunk(
                chunk_id=f"gen_{i:03d}",
                model_family="SD1.5",
                checkpoint="v1-5-pruned.safetensors",
                prompt=f"prompt {i}",
                parameters={"4": {"steps": 20, "cfg": 7.0 + (i % 5)}},
                output_filenames=[f"output_{i:03d}.png"],
                quality=QualityScore(overall=0.5 + (i % 10) * 0.05),
                execution_time_ms=1000 + i * 50,
            )
            accumulator.record(chunk)

        assert accumulator.generation_count == 50
        assert accumulator.learning_phase == LearningPhase.BLENDED
        stats = accumulator.get_stats()
        assert stats["successful"] == 50
        assert stats["avg_quality"] > 0

    def test_save_load_round_trip(self, accumulator, good_chunk, tmp_path):
        """Save to JSONL and load back — data survives."""
        accumulator.record(good_chunk)
        accumulator.record(ExperienceChunk(
            chunk_id="c2", model_family="SDXL",
            output_filenames=["x.png"],
            quality=QualityScore(overall=0.6),
        ))

        path = str(tmp_path / "experience.jsonl")
        saved = accumulator.save(path)
        assert saved == 2

        loaded = ExperienceAccumulator.load(path)
        assert loaded.generation_count == 2
        assert loaded._chunks[0].chunk_id == good_chunk.chunk_id
        assert loaded._chunks[0].quality.overall == 0.8
        assert loaded._chunks[1].model_family == "SDXL"

    def test_load_nonexistent_file(self, tmp_path):
        """Loading from nonexistent file returns empty accumulator."""
        loaded = ExperienceAccumulator.load(str(tmp_path / "nope.jsonl"))
        assert loaded.generation_count == 0

    def test_chunk_to_dict_round_trip(self, good_chunk):
        """ExperienceChunk serializes and deserializes correctly."""
        d = good_chunk.to_dict()
        restored = ExperienceChunk.from_dict(d)
        assert restored.chunk_id == good_chunk.chunk_id
        assert restored.model_family == good_chunk.model_family
        assert restored.quality.overall == good_chunk.quality.overall
        assert restored.quality.aesthetic == good_chunk.quality.aesthetic


# ---------------------------------------------------------------------------
# Cycle 33: accumulator O(n) eviction + exception handling
# ---------------------------------------------------------------------------

class TestAccumulatorEviction:
    """Accumulator must evict the lowest-quality chunk using O(n) scan, not sort."""

    def _make_chunk(self, model_family, quality_score, timestamp=None):
        from cognitive.experience.chunk import QualityScore as QS
        chunk = ExperienceChunk()
        chunk.model_family = model_family
        chunk.parameters = {"cfg": 7.0, "steps": 20}
        chunk.quality = QS(overall=quality_score, aesthetic=quality_score)
        if timestamp is not None:
            chunk.timestamp = timestamp
        return chunk

    def test_evicts_lowest_quality_not_newest(self):
        """When max_chunks exceeded, must remove lowest-quality chunk, not newest."""
        acc = ExperienceAccumulator(max_chunks=3)
        high1 = self._make_chunk("SD1.5", 0.9, timestamp=1.0)
        high2 = self._make_chunk("SD1.5", 0.8, timestamp=2.0)
        high3 = self._make_chunk("SDXL", 0.7, timestamp=3.0)
        low = self._make_chunk("Flux", 0.1, timestamp=0.5)  # lowest quality

        acc.record(high1)
        acc.record(high2)
        acc.record(high3)
        acc.record(low)  # triggers eviction — max_chunks=3, low quality evicted

        # After eviction: should have exactly 3 chunks (generation_count = current count)
        assert acc.generation_count == 3
        # The low quality chunk should be the one evicted
        with acc._chunks_lock:
            qualities = [c.quality.overall for c in acc._chunks]
        assert 0.1 not in qualities, "Lowest-quality chunk should have been evicted"

    def test_evicts_oldest_when_quality_tied(self):
        """When quality is tied, must evict the oldest chunk (lowest timestamp)."""
        acc = ExperienceAccumulator(max_chunks=2)
        old_chunk = self._make_chunk("SD1.5", 0.5, timestamp=1.0)
        new_chunk = self._make_chunk("SDXL", 0.5, timestamp=2.0)

        acc.record(old_chunk)
        acc.record(new_chunk)
        # Record a 3rd to trigger eviction
        newest = self._make_chunk("Flux", 0.5, timestamp=3.0)
        acc.record(newest)

        with acc._chunks_lock:
            timestamps = [c.timestamp for c in acc._chunks]
        # Oldest (timestamp=1.0) should be evicted; newer two remain
        assert 1.0 not in timestamps

    def test_no_eviction_under_max_chunks(self):
        """Under max_chunks, no eviction should occur."""
        acc = ExperienceAccumulator(max_chunks=10)
        for i in range(5):
            acc.record(self._make_chunk("SD1.5", 0.5 + i * 0.05))
        with acc._chunks_lock:
            count = len(acc._chunks)
        assert count == 5


class TestAccumulatorLoadErrorHandling:
    """accumulator.load() must log warnings on deserialization errors."""

    def test_corrupt_jsonl_line_is_skipped(self, tmp_path):
        """A corrupt JSON line in JSONL must be skipped, not crash."""
        jsonl = tmp_path / "chunks.jsonl"
        jsonl.write_text('{"bad json"\n', encoding="utf-8")
        # Should not raise
        acc = ExperienceAccumulator.load(str(jsonl))
        with acc._chunks_lock:
            assert len(acc._chunks) == 0

    def test_partial_chunk_data_is_skipped_with_warning(self, tmp_path, caplog):
        """A JSON line with a bad quality score triggers the warning path."""
        import logging
        import json as _json
        jsonl = tmp_path / "chunks.jsonl"
        # "overall": "not_a_float" causes QualityScore.__post_init__ to call
        # float("not_a_float"), which raises ValueError — caught by the
        # except Exception handler that logs a warning. (Cycle 33 fix)
        jsonl.write_text(_json.dumps({"quality": {"overall": "not_a_float"}}) + "\n",
                         encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            acc = ExperienceAccumulator.load(str(jsonl))
        # Should have logged a warning about the failed chunk
        assert any("Failed" in r.message or "chunk" in r.message.lower() for r in caplog.records)
        with acc._chunks_lock:
            assert len(acc._chunks) == 0


# ---------------------------------------------------------------------------
# Cycle 34: decay_weight clamped to [0.0, 1.0]
# ---------------------------------------------------------------------------

class TestDecayWeightClamp:
    """decay_weight must never exceed 1.0 regardless of timestamp."""

    def test_future_timestamp_clamped_to_one(self):
        """A chunk with a future timestamp (clock skew) must yield weight == 1.0."""
        from cognitive.experience.chunk import ExperienceChunk
        import time
        chunk = ExperienceChunk()
        chunk.timestamp = time.time() + 86400  # 1 day in the future
        assert chunk.decay_weight == 1.0

    def test_current_timestamp_at_most_one(self):
        """A brand-new chunk's weight must be <= 1.0."""
        from cognitive.experience.chunk import ExperienceChunk
        chunk = ExperienceChunk()
        assert 0.0 <= chunk.decay_weight <= 1.0

    def test_old_chunk_weight_non_negative(self):
        """A very old chunk's weight must be >= 0.0 (never negative)."""
        from cognitive.experience.chunk import ExperienceChunk
        chunk = ExperienceChunk()
        chunk.timestamp = 0.0  # Unix epoch — very old
        assert chunk.decay_weight >= 0.0


# ---------------------------------------------------------------------------
# Cycle 42 — accumulator.record() input guards
# ---------------------------------------------------------------------------

class TestAccumulatorRecordGuards:
    """Adversarial tests for None/wrong-type guards in Accumulator.record()."""

    def _make_accumulator(self):
        from cognitive.experience.accumulator import ExperienceAccumulator
        return ExperienceAccumulator()

    def test_record_none_raises_value_error(self):
        """Passing None to record() must raise ValueError."""
        import pytest
        acc = self._make_accumulator()
        with pytest.raises(ValueError, match="None"):
            acc.record(None)

    def test_record_string_raises_type_error(self):
        """Passing a string to record() must raise TypeError."""
        import pytest
        acc = self._make_accumulator()
        with pytest.raises(TypeError, match="ExperienceChunk"):
            acc.record("not a chunk")

    def test_record_valid_chunk_does_not_raise(self):
        """Valid ExperienceChunk must be recorded without error."""
        from cognitive.experience.chunk import ExperienceChunk
        acc = self._make_accumulator()
        chunk = ExperienceChunk()
        acc.record(chunk)
        assert len(acc._chunks) == 1
