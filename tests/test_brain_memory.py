"""Tests for brain/memory.py — outcome recording and pattern learning."""

import json

import pytest

from agent.brain import memory
from agent.config import SESSIONS_DIR


@pytest.fixture(autouse=True)
def clean_session_outcomes():
    """Remove test outcome files after each test."""
    yield
    for f in SESSIONS_DIR.glob("test_*_outcomes.jsonl"):
        f.unlink(missing_ok=True)


class TestRecordOutcome:
    def test_record_basic(self):
        result = json.loads(memory.handle("record_outcome", {
            "session": "test_mem",
            "key_params": {"model": "sdxl_base.safetensors", "steps": 20, "cfg": 7.0},
            "model_combo": ["sdxl_base"],
            "render_time_s": 12.5,
            "quality_score": 0.85,
        }))
        assert result["recorded"] is True
        assert result["total_outcomes"] == 1

    def test_record_multiple(self):
        for i in range(5):
            memory.handle("record_outcome", {
                "session": "test_multi",
                "key_params": {"model": "test", "steps": 20 + i},
                "quality_score": 0.7 + i * 0.05,
            })
        result = json.loads(memory.handle("record_outcome", {
            "session": "test_multi",
            "key_params": {"model": "test", "steps": 25},
        }))
        assert result["total_outcomes"] == 6

    def test_record_with_feedback(self):
        result = json.loads(memory.handle("record_outcome", {
            "session": "test_fb",
            "key_params": {"model": "test"},
            "user_feedback": "positive",
            "vision_notes": ["good composition", "slight banding"],
        }))
        assert result["recorded"] is True


class TestGetLearnedPatterns:
    def test_empty_history(self):
        result = json.loads(memory.handle("get_learned_patterns", {
            "session": "test_empty",
        }))
        assert result["outcomes_count"] == 0

    def test_best_models(self):
        # Record outcomes with different model combos
        for score, combo in [(0.9, ["sdxl", "lora_a"]), (0.8, ["sdxl"]), (0.7, ["sd15"])]:
            memory.handle("record_outcome", {
                "session": "test_patterns",
                "key_params": {"model": combo[0]},
                "model_combo": combo,
                "quality_score": score,
            })
        result = json.loads(memory.handle("get_learned_patterns", {
            "session": "test_patterns",
            "query": "best_models",
        }))
        assert len(result["best_model_combos"]) >= 2
        # Best combo should be first
        assert result["best_model_combos"][0]["avg_quality"] >= result["best_model_combos"][1]["avg_quality"]

    def test_optimal_params(self):
        # High quality with steps=25
        for _ in range(3):
            memory.handle("record_outcome", {
                "session": "test_optimal",
                "key_params": {"steps": 25, "cfg": 7.0},
                "quality_score": 0.9,
            })
        # Low quality with steps=10
        for _ in range(3):
            memory.handle("record_outcome", {
                "session": "test_optimal",
                "key_params": {"steps": 10, "cfg": 12.0},
                "quality_score": 0.5,
            })
        result = json.loads(memory.handle("get_learned_patterns", {
            "session": "test_optimal",
            "query": "optimal_params",
        }))
        assert "steps" in result["optimal_params"]
        assert result["optimal_params"]["steps"]["best_value"] == "25"

    def test_speed_analysis(self):
        for t in [8.0, 12.0, 15.0]:
            memory.handle("record_outcome", {
                "session": "test_speed",
                "key_params": {"model": "test"},
                "render_time_s": t,
            })
        result = json.loads(memory.handle("get_learned_patterns", {
            "session": "test_speed",
            "query": "speed_analysis",
        }))
        assert result["speed_analysis"]["fastest_s"] == 8.0
        assert result["speed_analysis"]["total_runs"] == 3

    def test_model_filter(self):
        memory.handle("record_outcome", {
            "session": "test_filter",
            "key_params": {"model": "sdxl_base"},
            "model_combo": ["sdxl_base"],
            "quality_score": 0.9,
        })
        memory.handle("record_outcome", {
            "session": "test_filter",
            "key_params": {"model": "sd15"},
            "model_combo": ["sd15"],
            "quality_score": 0.7,
        })
        result = json.loads(memory.handle("get_learned_patterns", {
            "session": "test_filter",
            "model_filter": "sdxl",
        }))
        assert result["outcomes_count"] == 1


class TestGetRecommendations:
    def test_empty_history(self):
        result = json.loads(memory.handle("get_recommendations", {
            "session": "test_rec_empty",
        }))
        assert len(result["recommendations"]) == 0

    def test_recommends_better_model(self):
        # Record good results with a specific combo
        for _ in range(5):
            memory.handle("record_outcome", {
                "session": "test_rec",
                "key_params": {"model": "sdxl_turbo"},
                "model_combo": ["sdxl_turbo", "detail_lora"],
                "quality_score": 0.9,
            })
        result = json.loads(memory.handle("get_recommendations", {
            "session": "test_rec",
            "current_model": "sd15_base",
        }))
        # Should recommend the better combo
        assert len(result["recommendations"]) > 0
        assert result["based_on"] == 5
