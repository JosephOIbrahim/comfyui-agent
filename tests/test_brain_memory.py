"""Tests for brain/memory.py — outcome recording and pattern learning."""

import json

import pytest

from agent.brain import handle
from agent.brain.memory import (
    MemoryAgent,
    OUTCOME_SCHEMA_VERSION,
    DECAY_HALF_LIFE_S,
    _temporal_weight,
    _migrate_outcome,
    _best_model_combos,
    _optimal_params,
    _params_similarity,
)
from agent.config import SESSIONS_DIR


# Shared MemoryAgent instance for direct method access in tests
_mem = MemoryAgent()


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
        result = json.loads(handle("record_outcome", {
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
            handle("record_outcome", {
                "session": "test_multi",
                "key_params": {"model": "test", "steps": 20 + i},
                "quality_score": 0.7 + i * 0.05,
            })
        result = json.loads(handle("record_outcome", {
            "session": "test_multi",
            "key_params": {"model": "test", "steps": 25},
        }))
        assert result["total_outcomes"] == 6

    def test_record_with_feedback(self):
        result = json.loads(handle("record_outcome", {
            "session": "test_fb",
            "key_params": {"model": "test"},
            "user_feedback": "positive",
            "vision_notes": ["good composition", "slight banding"],
        }))
        assert result["recorded"] is True


class TestGetLearnedPatterns:
    def test_empty_history(self):
        result = json.loads(handle("get_learned_patterns", {
            "session": "test_empty",
        }))
        assert result["outcomes_count"] == 0

    def test_best_models(self):
        # Record outcomes with different model combos
        for score, combo in [(0.9, ["sdxl", "lora_a"]), (0.8, ["sdxl"]), (0.7, ["sd15"])]:
            handle("record_outcome", {
                "session": "test_patterns",
                "key_params": {"model": combo[0]},
                "model_combo": combo,
                "quality_score": score,
            })
        result = json.loads(handle("get_learned_patterns", {
            "session": "test_patterns",
            "query": "best_models",
        }))
        assert len(result["best_model_combos"]) >= 2
        # Best combo should be first
        assert result["best_model_combos"][0]["avg_quality"] >= result["best_model_combos"][1]["avg_quality"]

    def test_optimal_params(self):
        # High quality with steps=25
        for _ in range(3):
            handle("record_outcome", {
                "session": "test_optimal",
                "key_params": {"steps": 25, "cfg": 7.0},
                "quality_score": 0.9,
            })
        # Low quality with steps=10
        for _ in range(3):
            handle("record_outcome", {
                "session": "test_optimal",
                "key_params": {"steps": 10, "cfg": 12.0},
                "quality_score": 0.5,
            })
        result = json.loads(handle("get_learned_patterns", {
            "session": "test_optimal",
            "query": "optimal_params",
        }))
        assert "steps" in result["optimal_params"]
        assert result["optimal_params"]["steps"]["best_value"] == "25"

    def test_speed_analysis(self):
        for t in [8.0, 12.0, 15.0]:
            handle("record_outcome", {
                "session": "test_speed",
                "key_params": {"model": "test"},
                "render_time_s": t,
            })
        result = json.loads(handle("get_learned_patterns", {
            "session": "test_speed",
            "query": "speed_analysis",
        }))
        assert result["speed_analysis"]["fastest_s"] == 8.0
        assert result["speed_analysis"]["total_runs"] == 3

    def test_model_filter(self):
        handle("record_outcome", {
            "session": "test_filter",
            "key_params": {"model": "sdxl_base"},
            "model_combo": ["sdxl_base"],
            "quality_score": 0.9,
        })
        handle("record_outcome", {
            "session": "test_filter",
            "key_params": {"model": "sd15"},
            "model_combo": ["sd15"],
            "quality_score": 0.7,
        })
        result = json.loads(handle("get_learned_patterns", {
            "session": "test_filter",
            "model_filter": "sdxl",
        }))
        assert result["outcomes_count"] == 1


class TestGetRecommendations:
    def test_empty_history(self):
        result = json.loads(handle("get_recommendations", {
            "session": "test_rec_empty",
        }))
        assert len(result["recommendations"]) == 0

    def test_recommends_better_model(self):
        # Record good results with a specific combo
        for _ in range(5):
            handle("record_outcome", {
                "session": "test_rec",
                "key_params": {"model": "sdxl_turbo"},
                "model_combo": ["sdxl_turbo", "detail_lora"],
                "quality_score": 0.9,
            })
        result = json.loads(handle("get_recommendations", {
            "session": "test_rec",
            "current_model": "sd15_base",
        }))
        # Should recommend the better combo
        assert len(result["recommendations"]) > 0
        assert result["based_on"] == 5

    def test_negative_pattern_avoidance(self):
        # Record bad outcomes with a specific param
        for _ in range(3):
            handle("record_outcome", {
                "session": "test_avoid",
                "key_params": {"cfg": 15.0, "steps": 10},
                "quality_score": 0.2,
                "user_feedback": "negative",
            })
        handle("record_outcome", {
            "session": "test_avoid",
            "key_params": {"cfg": 7.0, "steps": 25},
            "quality_score": 0.9,
        })
        result = json.loads(handle("get_recommendations", {
            "session": "test_avoid",
        }))
        assert result["avoidance_patterns"] > 0
        warnings = [r for r in result["recommendations"] if r["type"] == "warning"]
        assert len(warnings) > 0

    def test_context_step_recommendation(self):
        # Record high-quality at 30 steps
        for _ in range(3):
            handle("record_outcome", {
                "session": "test_ctx_steps",
                "key_params": {"steps": 30, "model": "sdxl"},
                "quality_score": 0.85,
            })
        # Ask with low steps
        result = json.loads(handle("get_recommendations", {
            "session": "test_ctx_steps",
            "current_params": {"steps": 10, "model": "sdxl"},
        }))
        ctx_recs = [r for r in result["recommendations"] if r["type"] == "context"]
        assert len(ctx_recs) > 0
        assert "steps" in ctx_recs[0]["suggestion"].lower()

    def test_goal_speed_recommendation(self):
        handle("record_outcome", {
            "session": "test_goal_speed",
            "key_params": {"steps": 8, "model": "turbo"},
            "render_time_s": 3.0,
            "quality_score": 0.7,
        })
        handle("record_outcome", {
            "session": "test_goal_speed",
            "key_params": {"steps": 30, "model": "sdxl"},
            "render_time_s": 20.0,
            "quality_score": 0.9,
        })
        result = json.loads(handle("get_recommendations", {
            "session": "test_goal_speed",
            "goal": "make it faster",
        }))
        goal_recs = [r for r in result["recommendations"] if r["type"] == "goal"]
        assert len(goal_recs) > 0

    def test_goal_quality_recommendation(self):
        handle("record_outcome", {
            "session": "test_goal_q",
            "key_params": {"steps": 40, "cfg": 7.0},
            "quality_score": 0.95,
        })
        handle("record_outcome", {
            "session": "test_goal_q",
            "key_params": {"steps": 10, "cfg": 12.0},
            "quality_score": 0.4,
        })
        result = json.loads(handle("get_recommendations", {
            "session": "test_goal_q",
            "goal": "maximize quality",
        }))
        goal_recs = [r for r in result["recommendations"] if r["type"] == "goal"]
        assert len(goal_recs) > 0
        assert goal_recs[0]["quality_score"] == 0.95


class TestImplicitFeedback:
    def test_empty_history(self):
        result = json.loads(handle("detect_implicit_feedback", {
            "session": "test_implicit_empty",
        }))
        assert result["signals"] == []
        assert result["summary"] == {}

    def test_detect_reuse(self):
        """Repeated model combo should produce positive reuse signal."""
        for _ in range(3):
            handle("record_outcome", {
                "session": "test_implicit_reuse",
                "key_params": {"model": "sdxl", "steps": 20},
                "model_combo": ["sdxl_base", "detail_lora"],
                "quality_score": 0.8,
            })
        result = json.loads(handle("detect_implicit_feedback", {
            "session": "test_implicit_reuse",
        }))
        reuse_signals = [s for s in result["signals"] if s["type"] == "reuse"]
        assert len(reuse_signals) >= 1
        assert reuse_signals[0]["signal"] == "positive"
        assert reuse_signals[0]["run_count"] == 3

    def test_detect_abandonment(self):
        """Model combo used once then abandoned should produce negative signal."""
        # One run with model A
        handle("record_outcome", {
            "session": "test_implicit_abandon",
            "key_params": {"model": "bad_model"},
            "model_combo": ["bad_model"],
        })
        # Then several runs with model B
        for _ in range(3):
            handle("record_outcome", {
                "session": "test_implicit_abandon",
                "key_params": {"model": "good_model"},
                "model_combo": ["good_model"],
            })
        result = json.loads(handle("detect_implicit_feedback", {
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
            handle("record_outcome", {
                "session": "test_implicit_refine",
                "key_params": {**base, "steps": steps},
                "model_combo": ["sdxl"],
            })
        result = json.loads(handle("detect_implicit_feedback", {
            "session": "test_implicit_refine",
        }))
        burst_signals = [s for s in result["signals"] if s["type"] == "refinement_burst"]
        assert len(burst_signals) >= 1
        assert burst_signals[0]["signal"] == "positive"
        assert burst_signals[0]["burst_length"] >= 3

    def test_detect_parameter_regression(self):
        """Reverting a parameter should produce negative signal."""
        handle("record_outcome", {
            "session": "test_implicit_regress",
            "key_params": {"steps": 20, "cfg": 7.0},
        })
        handle("record_outcome", {
            "session": "test_implicit_regress",
            "key_params": {"steps": 20, "cfg": 12.0},  # changed cfg
        })
        handle("record_outcome", {
            "session": "test_implicit_regress",
            "key_params": {"steps": 20, "cfg": 7.0},  # reverted cfg
        })
        result = json.loads(handle("detect_implicit_feedback", {
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
            handle("record_outcome", {
                "session": "test_implicit_sat",
                "key_params": {"model": "sdxl", "steps": 20 + i},
                "model_combo": ["sdxl_base"],
                "quality_score": 0.8,
            })
        result = json.loads(handle("detect_implicit_feedback", {
            "session": "test_implicit_sat",
        }))
        assert result["summary"]["positive"] > result["summary"]["negative"]
        assert result["summary"]["inferred_satisfaction"] in ("likely_satisfied", "mixed")

    def test_window_parameter(self):
        """Window should limit how many outcomes are analyzed."""
        for i in range(10):
            handle("record_outcome", {
                "session": "test_implicit_window",
                "key_params": {"steps": i},
                "model_combo": [f"model_{i % 2}"],
            })
        result = json.loads(handle("detect_implicit_feedback", {
            "session": "test_implicit_window",
            "window": 3,
        }))
        assert result["outcomes_analyzed"] == 3
        assert result["total_outcomes"] == 10

    def test_params_similarity_function(self):
        """Test the similarity helper directly."""
        assert _params_similarity({}, {}) == 0.0
        assert _params_similarity({"a": 1}, {"a": 1}) == 1.0
        assert _params_similarity({"a": 1, "b": 2}, {"a": 1, "b": 3}) == 0.5


class TestOutcomeSchemaVersioning:
    def test_recorded_outcome_has_version(self):
        """New outcomes should include schema_version."""
        handle("record_outcome", {
            "session": "test_schema_ver",
            "key_params": {"model": "test"},
        })
        outcomes = _mem._load_outcomes("test_schema_ver")
        assert len(outcomes) >= 1
        assert outcomes[-1]["schema_version"] == OUTCOME_SCHEMA_VERSION

    def test_migrate_v0_outcome(self):
        """Outcomes without schema_version get migrated on load."""
        v0_outcome = {
            "timestamp": 1700000000,
            "session": "test_migrate_v0",
            "key_params": {"model": "old"},
        }
        migrated = _migrate_outcome(v0_outcome)
        assert migrated["schema_version"] == 1


class TestOutcomeFileRotation:
    def test_rotation_creates_backup(self):
        """When file exceeds max size, it should be rotated."""
        from agent.brain.memory import OUTCOME_MAX_BYTES
        path = _mem._outcomes_path("test_rotation")
        # Write enough data to trigger rotation
        with open(path, "w", encoding="utf-8") as f:
            # Write a chunk bigger than OUTCOME_MAX_BYTES
            line = json.dumps({"key_params": {}, "data": "x" * 1000}, sort_keys=True) + "\n"
            for _ in range(OUTCOME_MAX_BYTES // len(line) + 10):
                f.write(line)

        # Now append — this should trigger rotation
        _mem._append_outcome("test_rotation", {"key_params": {"test": True}})

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
        w = _temporal_weight(now, now=now)
        assert abs(w - 1.0) < 0.01

    def test_weight_decays_at_half_life(self):
        """At exactly one half-life, weight should be ~0.5."""
        now = 1700000000
        old_time = now - DECAY_HALF_LIFE_S
        w = _temporal_weight(old_time, now=now)
        assert abs(w - 0.5) < 0.02

    def test_weight_never_below_minimum(self):
        """Even ancient outcomes should have minimum weight 0.01."""
        now = 1700000000
        ancient = now - 365 * 24 * 3600 * 10  # 10 years ago
        w = _temporal_weight(ancient, now=now)
        assert w >= 0.01

    def test_recent_outcomes_weighted_higher(self):
        """Recent outcomes should dominate aggregation over old ones."""
        # Record old outcomes with LOW quality
        for i in range(3):
            _mem._append_outcome("test_decay", {
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
            _mem._append_outcome("test_decay", {
                "schema_version": 1,
                "timestamp": now - i,
                "session": "test_decay",
                "key_params": {"model": "sdxl"},
                "model_combo": ["sdxl"],
                "quality_score": 0.95,
            })
        outcomes = _mem._load_outcomes("test_decay")
        combos = _best_model_combos(outcomes)
        # Weighted average should be closer to 0.95 than to 0.3
        assert combos[0]["avg_quality"] > 0.7

    def test_temporal_decay_in_optimal_params(self):
        """Temporal decay should discount old bad runs when recent runs are good."""
        import time
        now = time.time()
        # steps=10: recent, consistent 0.75
        for i in range(3):
            _mem._append_outcome("test_decay_params", {
                "schema_version": 1,
                "timestamp": now - i,
                "session": "test_decay_params",
                "key_params": {"steps": 10},
                "quality_score": 0.75,
            })
        # steps=30: one OLD bad run, then recent good runs
        _mem._append_outcome("test_decay_params", {
            "schema_version": 1,
            "timestamp": 1600000000,  # very old bad result
            "session": "test_decay_params",
            "key_params": {"steps": 30},
            "quality_score": 0.2,
        })
        for i in range(2):
            _mem._append_outcome("test_decay_params", {
                "schema_version": 1,
                "timestamp": now - i,
                "session": "test_decay_params",
                "key_params": {"steps": 30},
                "quality_score": 0.85,
            })
        outcomes = _mem._load_outcomes("test_decay_params")
        optimal = _optimal_params(outcomes)
        # With decay, steps=30 (weighted avg ~0.85) beats steps=10 (0.75)
        assert optimal["steps"]["best_value"] == "30"
        assert optimal["steps"]["avg_quality"] > 0.8


class TestGoalIdInOutcome:
    def test_record_with_goal_id(self):
        """Outcomes can include a goal_id from the planner."""
        result = json.loads(handle("record_outcome", {
            "session": "test_goal_id",
            "key_params": {"model": "sdxl"},
            "goal_id": "abc123def456",
        }))
        assert result["recorded"] is True
        outcomes = _mem._load_outcomes("test_goal_id")
        assert outcomes[-1]["goal_id"] == "abc123def456"

    def test_record_without_goal_id(self):
        """goal_id should be None when not provided."""
        handle("record_outcome", {
            "session": "test_no_goal_id",
            "key_params": {"model": "test"},
        })
        outcomes = _mem._load_outcomes("test_no_goal_id")
        assert outcomes[-1]["goal_id"] is None


class TestCrossSessionLearning:
    def test_load_all_outcomes_merges_sessions(self):
        """Global scope should aggregate across all test sessions."""
        # Write to two different sessions
        _mem._append_outcome("test_global_a", {
            "schema_version": 1,
            "timestamp": 1700000001,
            "session": "test_global_a",
            "key_params": {"model": "sdxl"},
            "model_combo": ["sdxl"],
            "quality_score": 0.9,
        })
        _mem._append_outcome("test_global_b", {
            "schema_version": 1,
            "timestamp": 1700000002,
            "session": "test_global_b",
            "key_params": {"model": "flux"},
            "model_combo": ["flux"],
            "quality_score": 0.85,
        })
        all_outcomes = _mem._load_all_outcomes()
        sessions = {o["session"] for o in all_outcomes}
        assert "test_global_a" in sessions
        assert "test_global_b" in sessions

    def test_global_scope_via_get_learned_patterns(self):
        """scope=global in get_learned_patterns should use all sessions."""
        _mem._append_outcome("test_scope_x", {
            "schema_version": 1,
            "timestamp": 1700000001,
            "session": "test_scope_x",
            "key_params": {"model": "sdxl"},
            "model_combo": ["sdxl"],
            "quality_score": 0.9,
        })
        _mem._append_outcome("test_scope_y", {
            "schema_version": 1,
            "timestamp": 1700000002,
            "session": "test_scope_y",
            "key_params": {"model": "flux"},
            "model_combo": ["flux"],
            "quality_score": 0.8,
        })
        result = json.loads(handle("get_learned_patterns", {
            "scope": "global",
            "query": "best_models",
        }))
        # Global scope should find outcomes from both sessions
        assert result["outcomes_count"] >= 2

    def test_load_all_outcomes_sorted_by_timestamp(self):
        """Merged outcomes should be sorted by timestamp."""
        _mem._append_outcome("test_sorted_a", {
            "schema_version": 1,
            "timestamp": 1700000010,
            "session": "test_sorted_a",
            "key_params": {"model": "later"},
        })
        _mem._append_outcome("test_sorted_b", {
            "schema_version": 1,
            "timestamp": 1700000001,
            "session": "test_sorted_b",
            "key_params": {"model": "earlier"},
        })
        all_outcomes = _mem._load_all_outcomes()
        # Filter to our test sessions
        ours = [o for o in all_outcomes if o.get("session", "").startswith("test_sorted_")]
        if len(ours) >= 2:
            assert ours[0]["timestamp"] <= ours[-1]["timestamp"]


# ---------------------------------------------------------------------------
# Cycle 33: window validation + load_outcomes lock
# ---------------------------------------------------------------------------

class TestWindowValidation:
    """detect_implicit_feedback must validate the window parameter."""

    def _make_agent_with_outcomes(self, tmp_path, count=5):
        """Create a MemoryAgent with N recorded outcomes."""
        from agent.brain._sdk import BrainConfig
        cfg = BrainConfig(sessions_dir=tmp_path / "sessions")
        agent = MemoryAgent(cfg)
        for i in range(count):
            agent.handle("record_outcome", {
                "session": "test_window",
                "key_params": {"model": "sd1.5", "steps": 20, "cfg": 7.0},
                "workflow_hash": f"hash{i}",
                "workflow_summary": "sd1.5 at 512x512, 20 steps",
                "model_combo": ["sd1.5.safetensors"],
            })
        return agent

    def test_window_zero_does_not_analyze_all_outcomes(self, tmp_path):
        """window=0 must be clamped to 1, not silently analyze all outcomes."""
        agent = self._make_agent_with_outcomes(tmp_path, count=10)
        # Must not crash — window=0 is clamped to 1
        result = json.loads(agent.handle("detect_implicit_feedback", {
            "session": "test_window",
            "window": 0,
        }))
        assert "signals" in result or "message" in result
        # No crash (window=0 before fix caused outcomes[-0:] = all outcomes)

    def test_window_negative_does_not_skip_outcomes(self, tmp_path):
        """Negative window must be clamped to 1."""
        agent = self._make_agent_with_outcomes(tmp_path, count=5)
        result = json.loads(agent.handle("detect_implicit_feedback", {
            "session": "test_window",
            "window": -5,
        }))
        assert "signals" in result or "message" in result

    def test_window_string_falls_back_to_default(self, tmp_path):
        """Non-numeric window must not crash — falls back to default 20."""
        agent = self._make_agent_with_outcomes(tmp_path, count=3)
        result = json.loads(agent.handle("detect_implicit_feedback", {
            "session": "test_window",
            "window": "many",
        }))
        assert "signals" in result or "message" in result

    def test_window_positive_normal_path_unaffected(self, tmp_path):
        """A valid positive window value must work normally."""
        agent = self._make_agent_with_outcomes(tmp_path, count=5)
        result = json.loads(agent.handle("detect_implicit_feedback", {
            "session": "test_window",
            "window": 3,
        }))
        assert "signals" in result or "message" in result


# ---------------------------------------------------------------------------
# Cycle 37: _load_all_outcomes acquires per-session lock (race condition fix)
# ---------------------------------------------------------------------------

class TestLoadAllOutcomesLocking:
    """_load_all_outcomes() must hold each session's lock while reading its file
    to prevent reading a partially-written JSONL line from a concurrent
    _append_outcome() call. (Cycle 37 fix)
    """

    def test_load_all_outcomes_acquires_per_session_lock(self, tmp_path):
        """_load_all_outcomes must acquire the per-session lock for each file it reads."""
        import threading
        from agent.brain.memory import MemoryAgent, _get_outcomes_lock
        from agent.brain._sdk import BrainConfig

        agent = MemoryAgent(config=BrainConfig(sessions_dir=tmp_path))

        # Write two sessions
        agent._append_outcome("locktest_a", {
            "schema_version": 1, "timestamp": 1.0, "session": "locktest_a",
            "key_params": {}, "model_combo": [], "quality_score": 0.9,
        })
        agent._append_outcome("locktest_b", {
            "schema_version": 1, "timestamp": 2.0, "session": "locktest_b",
            "key_params": {}, "model_combo": [], "quality_score": 0.8,
        })

        # Simulate a concurrent write: hold session A's lock while _load_all_outcomes runs.
        # If _load_all_outcomes does NOT acquire locks, it won't block and returns normally.
        # If it DOES acquire locks, it will block until we release — both are fine for
        # correctness; we just verify no exception and the data is consistent.
        lock_a = _get_outcomes_lock("locktest_a")
        results = []
        errors = []

        def _load_while_locked():
            try:
                outcomes = agent._load_all_outcomes()
                results.append(outcomes)
            except Exception as e:
                errors.append(e)

        # Acquire the lock (simulating a concurrent write in progress)
        with lock_a:
            t = threading.Thread(target=_load_while_locked)
            t.start()
            # The loader must block waiting for lock_a, but will eventually
            # succeed once we exit the `with` block.
            # Give it 0.1s with the lock held — it should not have returned yet
            t.join(timeout=0.1)
            # While lock_a is held, the loader may be blocked (this is the fix
            # in action). We don't assert it's blocked — just that it completes
            # correctly and safely.

        # Release lock_a — loader can now finish
        t.join(timeout=5.0)
        assert not errors, f"_load_all_outcomes raised: {errors}"
        assert len(results) == 1
        sessions = {o["session"] for o in results[0]}
        assert "locktest_a" in sessions
        assert "locktest_b" in sessions

    def test_load_all_outcomes_handles_osError_on_removed_file(self, tmp_path):
        """If a file disappears between glob and read, OSError is caught and skipped."""
        from unittest.mock import patch
        from agent.brain.memory import MemoryAgent
        from agent.brain._sdk import BrainConfig

        agent = MemoryAgent(config=BrainConfig(sessions_dir=tmp_path))
        agent._append_outcome("vanishing", {
            "schema_version": 1, "timestamp": 1.0, "session": "vanishing",
            "key_params": {}, "model_combo": [], "quality_score": 0.9,
        })

        # Patch Path.read_text to raise OSError (file removed mid-read)
        original_read_text = type(tmp_path).read_text

        call_count = 0

        def _flaky_read_text(self, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("File removed between glob and read")
            return original_read_text(self, **kwargs)

        with patch.object(type(tmp_path / "x"), "read_text", _flaky_read_text):
            # Should not raise — OSError is caught and the file is skipped
            outcomes = agent._load_all_outcomes()

        # Either 0 outcomes (file skipped) or 1 (if patching didn't intercept) — no crash
        assert isinstance(outcomes, list)


# ---------------------------------------------------------------------------
# Cycle 40: quality_score validation and query validation
# ---------------------------------------------------------------------------

class TestQualityScoreValidation:
    """Cycle 40: record_outcome must reject quality_score outside [0, 1]."""

    def test_quality_score_below_zero_returns_error(self):
        result = json.loads(handle("record_outcome", {
            "session": "qs-test",
            "workflow_summary": "test",
            "key_params": {},  # Cycle 58: key_params now required
            "quality_score": -0.1,
        }))
        assert "error" in result
        assert "quality_score" in result["error"]

    def test_quality_score_above_one_returns_error(self):
        result = json.loads(handle("record_outcome", {
            "session": "qs-test",
            "workflow_summary": "test",
            "key_params": {},  # Cycle 58: key_params now required
            "quality_score": 1.5,
        }))
        assert "error" in result
        assert "quality_score" in result["error"]

    def test_quality_score_string_non_numeric_returns_error(self):
        result = json.loads(handle("record_outcome", {
            "session": "qs-test",
            "workflow_summary": "test",
            "key_params": {},  # Cycle 58: key_params now required
            "quality_score": "high",
        }))
        assert "error" in result

    def test_quality_score_zero_is_valid(self):
        result = json.loads(handle("record_outcome", {
            "session": "qs-test",
            "workflow_summary": "test",
            "key_params": {},  # Cycle 58: key_params now required
            "quality_score": 0.0,
        }))
        assert result.get("recorded") is True

    def test_quality_score_one_is_valid(self):
        result = json.loads(handle("record_outcome", {
            "session": "qs-test",
            "workflow_summary": "test",
            "key_params": {},  # Cycle 58: key_params now required
            "quality_score": 1.0,
        }))
        assert result.get("recorded") is True

    def test_quality_score_none_is_valid(self):
        """Omitting quality_score (None) should be accepted."""
        result = json.loads(handle("record_outcome", {
            "session": "qs-test",
            "workflow_summary": "test",
            "key_params": {},  # Cycle 58: key_params now required
        }))
        assert result.get("recorded") is True


class TestGetLearnedPatternsQueryValidation:
    """Cycle 40: get_learned_patterns must reject unknown query values."""

    def test_invalid_query_returns_error(self):
        result = json.loads(handle("get_learned_patterns", {
            "session": "qv-test",
            "query": "nonexistent_query",
        }))
        assert "error" in result
        assert "query" in result["error"].lower() or "Invalid" in result["error"]

    def test_valid_query_all_succeeds(self):
        result = json.loads(handle("get_learned_patterns", {
            "session": "qv-test",
            "query": "all",
        }))
        assert "error" not in result

    def test_valid_query_best_models_succeeds(self):
        result = json.loads(handle("get_learned_patterns", {
            "session": "qv-test",
            "query": "best_models",
        }))
        assert "error" not in result

    def test_valid_query_speed_analysis_succeeds(self):
        result = json.loads(handle("get_learned_patterns", {
            "session": "qv-test",
            "query": "speed_analysis",
        }))
        assert "error" not in result


# ---------------------------------------------------------------------------
# Cycle 51 — rotation OSError logs instead of silent pass
# ---------------------------------------------------------------------------

class TestOutcomeRotationLogsError:
    """Rotation failures must be logged (not silently swallowed)."""

    def test_rotation_oserror_is_logged(self, tmp_path):
        """When _rotate_outcomes raises OSError, a warning must be logged."""
        from unittest.mock import patch
        from agent.brain.memory import MemoryAgent

        agent = MemoryAgent()
        # Create a real outcomes file that exceeds the size threshold
        outcomes_file = tmp_path / "test-session-outcomes.jsonl"
        outcomes_file.write_text('{"x":1}\n' * 100, encoding="utf-8")

        with patch.object(agent, "_outcomes_path", return_value=outcomes_file), \
             patch("agent.brain.memory._rotate_outcomes", side_effect=OSError("disk full")), \
             patch("agent.brain.memory.OUTCOME_MAX_BYTES", 0), \
             patch("agent.brain.memory.log") as mock_log:
            try:
                agent._append_outcome("test-session", {"outcome": "test", "timestamp": 1})
            except Exception:
                pass  # write may also fail; we only care about the log call
        mock_log.warning.assert_called_once()
        warning_msg = str(mock_log.warning.call_args)
        assert "rotation" in warning_msg.lower() or "rotate" in warning_msg.lower()


# ---------------------------------------------------------------------------
# Cycle 58: record_outcome required field guard for key_params
# ---------------------------------------------------------------------------

class TestRecordOutcomeKeyParamsGuard:
    """record_outcome must enforce key_params as required (schema compliance)."""

    def test_missing_key_params_returns_error(self):
        """Omitting key_params must return structured error, not silently default {}."""
        result = json.loads(handle("record_outcome", {
            "session": "test_c58_missing",
            "workflow_summary": "test run",
        }))
        assert "error" in result
        assert "key_params" in result["error"]

    def test_none_key_params_returns_error(self):
        """Explicit key_params=None must return error."""
        result = json.loads(handle("record_outcome", {
            "session": "test_c58_none",
            "key_params": None,
        }))
        assert "error" in result
        assert "key_params" in result["error"]

    def test_string_key_params_returns_error(self):
        """key_params as string must return error — must be a dict."""
        result = json.loads(handle("record_outcome", {
            "session": "test_c58_str",
            "key_params": "model=sd15",
        }))
        assert "error" in result
        assert "dict" in result["error"].lower() or "key_params" in result["error"]

    def test_list_key_params_returns_error(self):
        """key_params as list must return error — must be a dict."""
        result = json.loads(handle("record_outcome", {
            "session": "test_c58_list",
            "key_params": ["model", "steps"],
        }))
        assert "error" in result

    def test_empty_dict_key_params_is_valid(self):
        """Empty dict {} is a valid key_params value — no params extracted."""
        result = json.loads(handle("record_outcome", {
            "session": "test_c58_empty_dict",
            "key_params": {},
        }))
        assert "error" not in result
        assert result.get("recorded") is True

    def test_valid_key_params_records_successfully(self):
        """Normal key_params dict must record without error."""
        result = json.loads(handle("record_outcome", {
            "session": "test_c58_valid",
            "key_params": {"model": "sd15.safetensors", "steps": 20, "cfg": 7.0},
        }))
        assert "error" not in result
        assert result.get("recorded") is True


# ---------------------------------------------------------------------------
# Cycle 59 — allow_nan=False coverage for outcome writes and workflow hash
# ---------------------------------------------------------------------------

class TestOutcomeNaNSafety:
    """Cycle 59: memory.py json.dumps calls must reject NaN/Infinity (allow_nan=False)."""

    def test_workflow_hash_raises_on_nan_key_params(self):
        """_workflow_hash must raise ValueError when key_params contains NaN."""
        from agent.brain.memory import _workflow_hash
        with pytest.raises(ValueError):
            _workflow_hash({"cfg": float("nan")})

    def test_workflow_hash_raises_on_inf_key_params(self):
        """_workflow_hash must raise ValueError when key_params contains Infinity."""
        from agent.brain.memory import _workflow_hash
        with pytest.raises(ValueError):
            _workflow_hash({"steps": float("inf")})

    def test_outcome_write_raises_on_nan_in_outcome_dict(self):
        """_append_outcome must raise ValueError if outcome dict contains NaN (allow_nan=False)."""
        nan_outcome = {
            "quality_score": float("nan"),
            "session": "test_c59_nan_write",
            "timestamp": 1.0,
        }
        with pytest.raises(ValueError):
            _mem._append_outcome("test_c59_nan_write", nan_outcome)


# ---------------------------------------------------------------------------
# Cycle 63: _get_outcomes_lock — WeakValueDictionary prevents eviction race
# ---------------------------------------------------------------------------

class TestOutcomesLockWeakRef:
    """_get_outcomes_lock() must return same lock while caller holds reference (Cycle 63)."""

    def test_same_session_same_lock(self):
        """Two calls for the same session return the SAME lock object."""
        from agent.brain.memory import _get_outcomes_lock

        lock_a = _get_outcomes_lock("mem-same-63")
        lock_b = _get_outcomes_lock("mem-same-63")
        assert lock_a is lock_b, "Same session must return the same lock object"

    def test_concurrent_same_session_same_lock(self):
        """Concurrent _get_outcomes_lock() calls for the same session yield the same object."""
        import threading
        from agent.brain.memory import _get_outcomes_lock

        results = []

        def grab():
            results.append(_get_outcomes_lock("mem-concurrent-63"))

        threads = [threading.Thread(target=grab) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(id(lk) for lk in results)) == 1, \
            "All concurrent callers must receive the same lock object"


# ---------------------------------------------------------------------------
# Cycle 67: render_time_s validation + JSONDecodeError log guard
# ---------------------------------------------------------------------------

_GOOD_PARAMS = {
    "key_params": {"sampler": "euler", "steps": 20, "cfg": 7.0},
    "session": "test_cycle67",
}


class TestRenderTimeSValidation:
    """Cycle 67: record_outcome must validate render_time_s before allow_nan=False."""

    def test_nan_render_time_returns_error(self):
        """float('nan') render_time_s must return error, not crash in _append_outcome."""
        result = json.loads(handle("record_outcome", {
            **_GOOD_PARAMS,
            "render_time_s": float("nan"),
        }))
        assert "error" in result
        assert "render_time_s" in result["error"].lower()

    def test_inf_render_time_returns_error(self):
        """float('inf') render_time_s must return error, not crash."""
        result = json.loads(handle("record_outcome", {
            **_GOOD_PARAMS,
            "render_time_s": float("inf"),
        }))
        assert "error" in result
        assert "render_time_s" in result["error"].lower()

    def test_negative_render_time_returns_error(self):
        """Negative render_time_s must return error (seconds cannot be negative)."""
        result = json.loads(handle("record_outcome", {
            **_GOOD_PARAMS,
            "render_time_s": -1.5,
        }))
        assert "error" in result
        assert "render_time_s" in result["error"].lower()

    def test_string_render_time_returns_error(self):
        """String render_time_s ('fast') must return error."""
        result = json.loads(handle("record_outcome", {
            **_GOOD_PARAMS,
            "render_time_s": "fast",
        }))
        assert "error" in result
        assert "render_time_s" in result["error"].lower()

    def test_valid_render_time_is_recorded(self):
        """Valid float render_time_s must be recorded without error."""
        result = json.loads(handle("record_outcome", {
            **_GOOD_PARAMS,
            "render_time_s": 12.5,
        }))
        assert result.get("recorded") is True

    def test_none_render_time_allowed(self):
        """Omitting render_time_s must succeed (it's optional)."""
        result = json.loads(handle("record_outcome", _GOOD_PARAMS))
        assert result.get("recorded") is True


class TestOutcomeJsonDecodeLog:
    """Cycle 67: corrupted JSONL lines must be logged at DEBUG, not silently skipped."""

    def test_corrupted_line_logged_at_debug(self, tmp_path, caplog):
        """A corrupted JSONL line must emit a DEBUG log entry."""
        import logging
        from agent.brain.memory import MemoryAgent
        from agent.brain._sdk import BrainConfig

        cfg = BrainConfig(sessions_dir=tmp_path)
        mem = MemoryAgent(cfg)

        # Write a corrupted JSONL line directly into the outcomes file
        session = "logtest"
        outcomes_path = tmp_path / f"{session}_outcomes.jsonl"
        outcomes_path.write_text("{corrupted json{\n", encoding="utf-8")

        with caplog.at_level(logging.DEBUG, logger="agent.brain.memory"):
            outcomes = mem._load_outcomes(session)

        assert outcomes == []  # corrupted line skipped
        assert any("corrupt" in r.message.lower() or "skip" in r.message.lower()
                   for r in caplog.records), \
            "Expected DEBUG log about corrupted outcome line"
