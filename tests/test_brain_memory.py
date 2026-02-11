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

    def test_negative_pattern_avoidance(self):
        # Record bad outcomes with a specific param
        for _ in range(3):
            memory.handle("record_outcome", {
                "session": "test_avoid",
                "key_params": {"cfg": 15.0, "steps": 10},
                "quality_score": 0.2,
                "user_feedback": "negative",
            })
        memory.handle("record_outcome", {
            "session": "test_avoid",
            "key_params": {"cfg": 7.0, "steps": 25},
            "quality_score": 0.9,
        })
        result = json.loads(memory.handle("get_recommendations", {
            "session": "test_avoid",
        }))
        assert result["avoidance_patterns"] > 0
        warnings = [r for r in result["recommendations"] if r["type"] == "warning"]
        assert len(warnings) > 0

    def test_context_step_recommendation(self):
        # Record high-quality at 30 steps
        for _ in range(3):
            memory.handle("record_outcome", {
                "session": "test_ctx_steps",
                "key_params": {"steps": 30, "model": "sdxl"},
                "quality_score": 0.85,
            })
        # Ask with low steps
        result = json.loads(memory.handle("get_recommendations", {
            "session": "test_ctx_steps",
            "current_params": {"steps": 10, "model": "sdxl"},
        }))
        ctx_recs = [r for r in result["recommendations"] if r["type"] == "context"]
        assert len(ctx_recs) > 0
        assert "steps" in ctx_recs[0]["suggestion"].lower()

    def test_goal_speed_recommendation(self):
        memory.handle("record_outcome", {
            "session": "test_goal_speed",
            "key_params": {"steps": 8, "model": "turbo"},
            "render_time_s": 3.0,
            "quality_score": 0.7,
        })
        memory.handle("record_outcome", {
            "session": "test_goal_speed",
            "key_params": {"steps": 30, "model": "sdxl"},
            "render_time_s": 20.0,
            "quality_score": 0.9,
        })
        result = json.loads(memory.handle("get_recommendations", {
            "session": "test_goal_speed",
            "goal": "make it faster",
        }))
        goal_recs = [r for r in result["recommendations"] if r["type"] == "goal"]
        assert len(goal_recs) > 0

    def test_goal_quality_recommendation(self):
        memory.handle("record_outcome", {
            "session": "test_goal_q",
            "key_params": {"steps": 40, "cfg": 7.0},
            "quality_score": 0.95,
        })
        memory.handle("record_outcome", {
            "session": "test_goal_q",
            "key_params": {"steps": 10, "cfg": 12.0},
            "quality_score": 0.4,
        })
        result = json.loads(memory.handle("get_recommendations", {
            "session": "test_goal_q",
            "goal": "maximize quality",
        }))
        goal_recs = [r for r in result["recommendations"] if r["type"] == "goal"]
        assert len(goal_recs) > 0
        assert goal_recs[0]["quality_score"] == 0.95


class TestImplicitFeedback:
    def test_empty_history(self):
        result = json.loads(memory.handle("detect_implicit_feedback", {
            "session": "test_implicit_empty",
        }))
        assert result["signals"] == []
        assert result["summary"] == {}

    def test_detect_reuse(self):
        """Repeated model combo should produce positive reuse signal."""
        for _ in range(3):
            memory.handle("record_outcome", {
                "session": "test_implicit_reuse",
                "key_params": {"model": "sdxl", "steps": 20},
                "model_combo": ["sdxl_base", "detail_lora"],
                "quality_score": 0.8,
            })
        result = json.loads(memory.handle("detect_implicit_feedback", {
            "session": "test_implicit_reuse",
        }))
        reuse_signals = [s for s in result["signals"] if s["type"] == "reuse"]
        assert len(reuse_signals) >= 1
        assert reuse_signals[0]["signal"] == "positive"
        assert reuse_signals[0]["run_count"] == 3

    def test_detect_abandonment(self):
        """Model combo used once then abandoned should produce negative signal."""
        # One run with model A
        memory.handle("record_outcome", {
            "session": "test_implicit_abandon",
            "key_params": {"model": "bad_model"},
            "model_combo": ["bad_model"],
        })
        # Then several runs with model B
        for _ in range(3):
            memory.handle("record_outcome", {
                "session": "test_implicit_abandon",
                "key_params": {"model": "good_model"},
                "model_combo": ["good_model"],
            })
        result = json.loads(memory.handle("detect_implicit_feedback", {
            "session": "test_implicit_abandon",
        }))
        abandon_signals = [s for s in result["signals"] if s["type"] == "abandonment"]
        assert len(abandon_signals) >= 1
        assert abandon_signals[0]["signal"] == "negative"
        assert "bad_model" in abandon_signals[0]["models"]

    def test_detect_refinement_burst(self):
        """Cluster of similar-param runs should detect refinement."""
        base = {"model": "sdxl", "sampler": "euler", "cfg": 7.0}
        for steps in [20, 22, 25, 24, 23]:
            memory.handle("record_outcome", {
                "session": "test_implicit_refine",
                "key_params": {**base, "steps": steps},
                "model_combo": ["sdxl"],
            })
        result = json.loads(memory.handle("detect_implicit_feedback", {
            "session": "test_implicit_refine",
        }))
        burst_signals = [s for s in result["signals"] if s["type"] == "refinement_burst"]
        assert len(burst_signals) >= 1
        assert burst_signals[0]["signal"] == "positive"
        assert burst_signals[0]["burst_length"] >= 3

    def test_detect_parameter_regression(self):
        """Reverting a parameter should produce negative signal."""
        memory.handle("record_outcome", {
            "session": "test_implicit_regress",
            "key_params": {"steps": 20, "cfg": 7.0},
        })
        memory.handle("record_outcome", {
            "session": "test_implicit_regress",
            "key_params": {"steps": 20, "cfg": 12.0},  # changed cfg
        })
        memory.handle("record_outcome", {
            "session": "test_implicit_regress",
            "key_params": {"steps": 20, "cfg": 7.0},  # reverted cfg
        })
        result = json.loads(memory.handle("detect_implicit_feedback", {
            "session": "test_implicit_regress",
        }))
        regression_signals = [s for s in result["signals"] if s["type"] == "parameter_regression"]
        assert len(regression_signals) >= 1
        assert regression_signals[0]["signal"] == "negative"
        assert regression_signals[0]["parameter"] == "cfg"
        assert regression_signals[0]["reverted_from"] == "12.0"

    def test_satisfaction_inference(self):
        """Multiple positive signals should infer likely_satisfied."""
        for i in range(5):
            memory.handle("record_outcome", {
                "session": "test_implicit_sat",
                "key_params": {"model": "sdxl", "steps": 20 + i},
                "model_combo": ["sdxl_base"],
                "quality_score": 0.8,
            })
        result = json.loads(memory.handle("detect_implicit_feedback", {
            "session": "test_implicit_sat",
        }))
        assert result["summary"]["positive"] > result["summary"]["negative"]
        assert result["summary"]["inferred_satisfaction"] in ("likely_satisfied", "mixed")

    def test_window_parameter(self):
        """Window should limit how many outcomes are analyzed."""
        for i in range(10):
            memory.handle("record_outcome", {
                "session": "test_implicit_window",
                "key_params": {"steps": i},
                "model_combo": [f"model_{i % 2}"],
            })
        result = json.loads(memory.handle("detect_implicit_feedback", {
            "session": "test_implicit_window",
            "window": 3,
        }))
        assert result["outcomes_analyzed"] == 3
        assert result["total_outcomes"] == 10

    def test_params_similarity_function(self):
        """Test the similarity helper directly."""
        assert memory._params_similarity({}, {}) == 0.0
        assert memory._params_similarity({"a": 1}, {"a": 1}) == 1.0
        assert memory._params_similarity({"a": 1, "b": 2}, {"a": 1, "b": 3}) == 0.5
