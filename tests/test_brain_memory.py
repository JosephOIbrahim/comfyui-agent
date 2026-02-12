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
    # Also clean up backup files from rotation tests
    for f in SESSIONS_DIR.glob("test_*_outcomes.jsonl.*"):
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


class TestOutcomeSchemaVersioning:
    def test_recorded_outcome_has_version(self):
        """New outcomes should include schema_version."""
        memory.handle("record_outcome", {
            "session": "test_schema_ver",
            "key_params": {"model": "test"},
        })
        outcomes = memory._load_outcomes("test_schema_ver")
        assert len(outcomes) >= 1
        assert outcomes[-1]["schema_version"] == memory.OUTCOME_SCHEMA_VERSION

    def test_migrate_v0_outcome(self):
        """Outcomes without schema_version get migrated on load."""
        v0_outcome = {
            "timestamp": 1700000000,
            "session": "test_migrate_v0",
            "key_params": {"model": "old"},
        }
        migrated = memory._migrate_outcome(v0_outcome)
        assert migrated["schema_version"] == 1


class TestOutcomeFileRotation:
    def test_rotation_creates_backup(self):
        """When file exceeds max size, it should be rotated."""
        path = memory._outcomes_path("test_rotation")
        # Write enough data to trigger rotation
        with open(path, "w", encoding="utf-8") as f:
            # Write a chunk bigger than OUTCOME_MAX_BYTES
            line = json.dumps({"key_params": {}, "data": "x" * 1000}, sort_keys=True) + "\n"
            for _ in range(memory.OUTCOME_MAX_BYTES // len(line) + 10):
                f.write(line)

        # Now append — this should trigger rotation
        memory._append_outcome("test_rotation", {"key_params": {"test": True}})

        # Original file should exist (new one after rotation)
        assert path.exists()
        # Backup should exist
        from pathlib import Path
        backup = Path(f"{path}.1")
        assert backup.exists()

        # Cleanup
        path.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)


class TestTemporalDecay:
    def test_weight_of_brand_new_outcome(self):
        """Brand new outcomes should have weight ~1.0."""
        now = 1700000000
        w = memory._temporal_weight(now, now=now)
        assert abs(w - 1.0) < 0.01

    def test_weight_decays_at_half_life(self):
        """At exactly one half-life, weight should be ~0.5."""
        now = 1700000000
        old_time = now - memory.DECAY_HALF_LIFE_S
        w = memory._temporal_weight(old_time, now=now)
        assert abs(w - 0.5) < 0.02

    def test_weight_never_below_minimum(self):
        """Even ancient outcomes should have minimum weight 0.01."""
        now = 1700000000
        ancient = now - 365 * 24 * 3600 * 10  # 10 years ago
        w = memory._temporal_weight(ancient, now=now)
        assert w >= 0.01

    def test_recent_outcomes_weighted_higher(self):
        """Recent outcomes should dominate aggregation over old ones."""
        # Record old outcomes with LOW quality
        for i in range(3):
            memory._append_outcome("test_decay", {
                "schema_version": 1,
                "timestamp": 1600000000 + i,  # very old
                "session": "test_decay",
                "key_params": {"model": "sdxl"},
                "model_combo": ["sdxl"],
                "quality_score": 0.3,
            })
        # Record recent outcomes with HIGH quality
        import time
        now = time.time()
        for i in range(3):
            memory._append_outcome("test_decay", {
                "schema_version": 1,
                "timestamp": now - i,
                "session": "test_decay",
                "key_params": {"model": "sdxl"},
                "model_combo": ["sdxl"],
                "quality_score": 0.95,
            })
        outcomes = memory._load_outcomes("test_decay")
        combos = memory._best_model_combos(outcomes)
        # Weighted average should be closer to 0.95 than to 0.3
        assert combos[0]["avg_quality"] > 0.7

    def test_temporal_decay_in_optimal_params(self):
        """Temporal decay should discount old bad runs when recent runs are good.

        steps=10: all recent at 0.75 -> weighted avg ~0.75
        steps=30: one old bad run (0.2) + recent good runs (0.85)
          Without decay: simple avg = (0.2+0.85+0.85)/3 = 0.633 -> steps=10 wins
          With decay: weighted avg ≈ 0.85 (old 0.2 nearly zeroed) -> steps=30 wins
        """
        import time
        now = time.time()
        # steps=10: recent, consistent 0.75
        for i in range(3):
            memory._append_outcome("test_decay_params", {
                "schema_version": 1,
                "timestamp": now - i,
                "session": "test_decay_params",
                "key_params": {"steps": 10},
                "quality_score": 0.75,
            })
        # steps=30: one OLD bad run, then recent good runs
        memory._append_outcome("test_decay_params", {
            "schema_version": 1,
            "timestamp": 1600000000,  # very old bad result
            "session": "test_decay_params",
            "key_params": {"steps": 30},
            "quality_score": 0.2,
        })
        for i in range(2):
            memory._append_outcome("test_decay_params", {
                "schema_version": 1,
                "timestamp": now - i,
                "session": "test_decay_params",
                "key_params": {"steps": 30},
                "quality_score": 0.85,
            })
        outcomes = memory._load_outcomes("test_decay_params")
        optimal = memory._optimal_params(outcomes)
        # With decay, steps=30 (weighted avg ~0.85) beats steps=10 (0.75)
        assert optimal["steps"]["best_value"] == "30"
        assert optimal["steps"]["avg_quality"] > 0.8


class TestGoalIdInOutcome:
    def test_record_with_goal_id(self):
        """Outcomes can include a goal_id from the planner."""
        result = json.loads(memory.handle("record_outcome", {
            "session": "test_goal_id",
            "key_params": {"model": "sdxl"},
            "goal_id": "abc123def456",
        }))
        assert result["recorded"] is True
        outcomes = memory._load_outcomes("test_goal_id")
        assert outcomes[-1]["goal_id"] == "abc123def456"

    def test_record_without_goal_id(self):
        """goal_id should be None when not provided."""
        memory.handle("record_outcome", {
            "session": "test_no_goal_id",
            "key_params": {"model": "test"},
        })
        outcomes = memory._load_outcomes("test_no_goal_id")
        assert outcomes[-1]["goal_id"] is None


class TestCrossSessionLearning:
    def test_load_all_outcomes_merges_sessions(self):
        """Global scope should aggregate across all test sessions."""
        # Write to two different sessions
        memory._append_outcome("test_global_a", {
            "schema_version": 1,
            "timestamp": 1700000001,
            "session": "test_global_a",
            "key_params": {"model": "sdxl"},
            "model_combo": ["sdxl"],
            "quality_score": 0.9,
        })
        memory._append_outcome("test_global_b", {
            "schema_version": 1,
            "timestamp": 1700000002,
            "session": "test_global_b",
            "key_params": {"model": "flux"},
            "model_combo": ["flux"],
            "quality_score": 0.85,
        })
        all_outcomes = memory._load_all_outcomes()
        sessions = {o["session"] for o in all_outcomes}
        assert "test_global_a" in sessions
        assert "test_global_b" in sessions

    def test_global_scope_via_get_learned_patterns(self):
        """scope=global in get_learned_patterns should use all sessions."""
        memory._append_outcome("test_scope_x", {
            "schema_version": 1,
            "timestamp": 1700000001,
            "session": "test_scope_x",
            "key_params": {"model": "sdxl"},
            "model_combo": ["sdxl"],
            "quality_score": 0.9,
        })
        memory._append_outcome("test_scope_y", {
            "schema_version": 1,
            "timestamp": 1700000002,
            "session": "test_scope_y",
            "key_params": {"model": "flux"},
            "model_combo": ["flux"],
            "quality_score": 0.8,
        })
        result = json.loads(memory.handle("get_learned_patterns", {
            "scope": "global",
            "query": "best_models",
        }))
        # Global scope should find outcomes from both sessions
        assert result["outcomes_count"] >= 2

    def test_load_all_outcomes_sorted_by_timestamp(self):
        """Merged outcomes should be sorted by timestamp."""
        memory._append_outcome("test_sorted_a", {
            "schema_version": 1,
            "timestamp": 1700000010,
            "session": "test_sorted_a",
            "key_params": {"model": "later"},
        })
        memory._append_outcome("test_sorted_b", {
            "schema_version": 1,
            "timestamp": 1700000001,
            "session": "test_sorted_b",
            "key_params": {"model": "earlier"},
        })
        all_outcomes = memory._load_all_outcomes()
        # Filter to our test sessions
        ours = [o for o in all_outcomes if o.get("session", "").startswith("test_sorted_")]
        if len(ours) >= 2:
            assert ours[0]["timestamp"] <= ours[-1]["timestamp"]
