"""Tests for the Experience Accumulator.

[EXPERIENCE x CRUCIBLE] — Tests for ExperienceChunk, signatures,
context matching, learning phases, and temporal decay.
"""

import time

import pytest

from src.cognitive.experience.chunk import ExperienceChunk, QualityScore
from src.cognitive.experience.signature import GenerationContextSignature
from src.cognitive.experience.accumulator import (
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
