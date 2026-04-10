"""Tests for agent/stage/ratchet.py — adversarial coverage, no real I/O."""

from __future__ import annotations

import pytest

from agent.stage.ratchet import (
    DEFAULT_WEIGHTS,
    Ratchet,
    RatchetDecision,
    RatchetError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def usd_stage():
    """Fresh CognitiveWorkflowStage, skipped if usd-core is not installed."""
    pytest.importorskip("pxr", reason="usd-core not installed")
    from agent.stage.cognitive_stage import CognitiveWorkflowStage
    return CognitiveWorkflowStage()


# ---------------------------------------------------------------------------
# DEFAULT_WEIGHTS
# ---------------------------------------------------------------------------

class TestDefaultWeights:
    def test_all_six_axes_present(self):
        for axis in ("aesthetic", "depth", "normals", "camera", "segmentation", "lighting"):
            assert axis in DEFAULT_WEIGHTS, f"Missing axis: {axis}"

    def test_all_weights_positive(self):
        for axis, w in DEFAULT_WEIGHTS.items():
            assert w > 0.0, f"Weight for '{axis}' must be positive"


# ---------------------------------------------------------------------------
# Ratchet.__init__
# ---------------------------------------------------------------------------

class TestRatchetInit:
    def test_default_weights_match_module_constant(self):
        r = Ratchet()
        assert r.weights == DEFAULT_WEIGHTS

    def test_custom_weights_stored(self):
        r = Ratchet({"aesthetic": 2.0, "depth": 0.5})
        assert r.weights == {"aesthetic": 2.0, "depth": 0.5}

    def test_custom_weights_do_not_mutate_input(self):
        w = {"aesthetic": 1.0}
        r = Ratchet(w)
        w["extra"] = 99.0
        assert "extra" not in r.weights

    def test_default_threshold(self):
        r = Ratchet()
        assert r.threshold == 0.5

    def test_custom_threshold_stored(self):
        r = Ratchet(threshold=0.7)
        assert r.threshold == 0.7

    def test_threshold_boundary_zero(self):
        r = Ratchet(threshold=0.0)
        assert r.threshold == 0.0

    def test_threshold_boundary_one(self):
        r = Ratchet(threshold=1.0)
        assert r.threshold == 1.0

    def test_threshold_below_zero_raises(self):
        with pytest.raises(RatchetError, match="threshold"):
            Ratchet(threshold=-0.01)

    def test_threshold_above_one_raises(self):
        with pytest.raises(RatchetError, match="threshold"):
            Ratchet(threshold=1.01)

    def test_history_starts_empty(self):
        r = Ratchet()
        assert r.history == []

    def test_weights_property_returns_copy(self):
        r = Ratchet()
        w = r.weights
        w["hacked"] = 99.0
        assert "hacked" not in r.weights


# ---------------------------------------------------------------------------
# compute_score
# ---------------------------------------------------------------------------

class TestComputeScore:
    def test_equal_weights_equal_scores(self):
        r = Ratchet()
        assert abs(r.compute_score({"aesthetic": 0.8, "depth": 0.8}) - 0.8) < 1e-9

    def test_weighted_average_two_axes(self):
        r = Ratchet({"aesthetic": 2.0, "depth": 1.0})
        # (1.0*2 + 0.0*1) / 3 = 0.6667
        score = r.compute_score({"aesthetic": 1.0, "depth": 0.0})
        assert abs(score - 2.0 / 3.0) < 1e-9

    def test_unknown_axis_ignored(self):
        r = Ratchet({"aesthetic": 1.0})
        score = r.compute_score({"aesthetic": 0.6, "unknown_axis": 0.9})
        assert abs(score - 0.6) < 1e-9

    def test_zero_weight_axis_ignored(self):
        r = Ratchet({"aesthetic": 1.0, "depth": 0.0})
        score = r.compute_score({"aesthetic": 0.4, "depth": 1.0})
        assert abs(score - 0.4) < 1e-9

    def test_no_matching_axes_returns_zero(self):
        r = Ratchet({"aesthetic": 1.0})
        assert r.compute_score({"depth": 0.9}) == 0.0

    def test_empty_axis_scores_returns_zero(self):
        r = Ratchet()
        assert r.compute_score({}) == 0.0

    def test_score_below_zero_raises(self):
        r = Ratchet()
        with pytest.raises(RatchetError, match="Score for 'aesthetic'"):
            r.compute_score({"aesthetic": -0.001})

    def test_score_above_one_raises(self):
        r = Ratchet()
        with pytest.raises(RatchetError, match="Score for 'depth'"):
            r.compute_score({"depth": 1.001})

    def test_boundary_score_zero_accepted(self):
        r = Ratchet()
        assert r.compute_score({"aesthetic": 0.0}) == 0.0

    def test_boundary_score_one_accepted(self):
        r = Ratchet()
        assert r.compute_score({"aesthetic": 1.0}) == 1.0

    def test_single_axis_equals_its_score(self):
        r = Ratchet()
        assert abs(r.compute_score({"aesthetic": 0.73}) - 0.73) < 1e-9

    def test_all_default_axes_equal_score(self):
        r = Ratchet()
        axes = {k: 0.42 for k in DEFAULT_WEIGHTS}
        assert abs(r.compute_score(axes) - 0.42) < 1e-9

    def test_subset_of_axes_averages_only_present(self):
        r = Ratchet()
        # Only aesthetic and lighting provided; others absent — only those two count
        score = r.compute_score({"aesthetic": 1.0, "lighting": 0.0})
        assert abs(score - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# decide (auto-threshold)
# ---------------------------------------------------------------------------

class TestDecide:
    def test_above_threshold_returns_true(self):
        r = Ratchet(threshold=0.5)
        assert r.decide("d1", {"aesthetic": 0.9}) is True

    def test_at_threshold_returns_true(self):
        r = Ratchet(threshold=0.5)
        assert r.decide("d1", {"aesthetic": 0.5}) is True

    def test_below_threshold_returns_false(self):
        r = Ratchet(threshold=0.5)
        assert r.decide("d1", {"aesthetic": 0.499}) is False

    def test_threshold_zero_always_kept(self):
        r = Ratchet(threshold=0.0)
        assert r.decide("d1", {"aesthetic": 0.0}) is True

    def test_threshold_one_only_perfect_score_kept(self):
        r = Ratchet(threshold=1.0)
        assert r.decide("d1", {"aesthetic": 0.999}) is False
        assert r.decide("d2", {"aesthetic": 1.0}) is True

    def test_decision_appended_to_history(self):
        r = Ratchet()
        r.decide("delta_abc", {"aesthetic": 0.9})
        assert len(r.history) == 1
        assert r.history[0].delta_id == "delta_abc"

    def test_composite_stored_accurately(self):
        r = Ratchet({"aesthetic": 1.0})
        r.decide("d1", {"aesthetic": 0.75})
        assert abs(r.history[0].composite - 0.75) < 1e-9

    def test_axis_scores_stored_in_history(self):
        r = Ratchet()
        r.decide("d1", {"aesthetic": 0.9, "depth": 0.7})
        assert r.history[0].axis_scores == {"aesthetic": 0.9, "depth": 0.7}

    def test_kept_flag_matches_return_value(self):
        r = Ratchet(threshold=0.5)
        result = r.decide("d1", {"aesthetic": 0.8})
        assert r.history[0].kept is result

    def test_multiple_decisions_accumulated(self):
        r = Ratchet()
        for i in range(7):
            r.decide(f"d{i}", {"aesthetic": i / 10})
        assert len(r.history) == 7

    def test_invalid_score_raises_before_recording(self):
        r = Ratchet()
        with pytest.raises(RatchetError):
            r.decide("d1", {"aesthetic": 2.0})
        assert len(r.history) == 0  # nothing recorded


# ---------------------------------------------------------------------------
# keep / discard (explicit)
# ---------------------------------------------------------------------------

class TestExplicitKeepDiscard:
    def test_keep_returns_decision_with_kept_true(self):
        r = Ratchet()
        d = r.keep("delta_1")
        assert isinstance(d, RatchetDecision)
        assert d.kept is True
        assert d.delta_id == "delta_1"

    def test_keep_with_scores_computes_composite(self):
        r = Ratchet({"aesthetic": 1.0})
        d = r.keep("d2", {"aesthetic": 0.9})
        assert abs(d.composite - 0.9) < 1e-9

    def test_keep_without_scores_composite_is_one(self):
        r = Ratchet()
        d = r.keep("d3")
        assert d.composite == 1.0
        assert d.axis_scores == {}

    def test_keep_appended_to_history(self):
        r = Ratchet()
        r.keep("d1")
        assert len(r.history) == 1
        assert r.history[0].kept is True

    def test_discard_returns_decision_with_kept_false(self):
        r = Ratchet()
        d = r.discard("delta_x")
        assert d.kept is False

    def test_discard_with_scores_computes_composite(self):
        r = Ratchet({"aesthetic": 1.0})
        d = r.discard("d2", {"aesthetic": 0.2})
        assert abs(d.composite - 0.2) < 1e-9

    def test_discard_without_scores_composite_is_zero(self):
        r = Ratchet()
        d = r.discard("d3")
        assert d.composite == 0.0

    def test_keep_then_discard_ordering_preserved(self):
        r = Ratchet()
        r.keep("d1")
        r.discard("d2")
        assert r.history[0].kept is True
        assert r.history[1].kept is False

    def test_invalid_score_in_keep_raises(self):
        r = Ratchet()
        with pytest.raises(RatchetError):
            r.keep("d1", {"aesthetic": -0.5})

    def test_invalid_score_in_discard_raises(self):
        r = Ratchet()
        with pytest.raises(RatchetError):
            r.discard("d1", {"depth": 1.5})


# ---------------------------------------------------------------------------
# History queries
# ---------------------------------------------------------------------------

class TestHistoryQueries:
    def test_kept_ids_empty_initially(self):
        assert Ratchet().kept_ids() == []

    def test_discarded_ids_empty_initially(self):
        assert Ratchet().discarded_ids() == []

    def test_kept_ids_filters_correctly(self):
        r = Ratchet(threshold=0.5)
        r.decide("d1", {"aesthetic": 0.9})
        r.decide("d2", {"aesthetic": 0.1})
        r.decide("d3", {"aesthetic": 0.8})
        assert r.kept_ids() == ["d1", "d3"]

    def test_discarded_ids_filters_correctly(self):
        r = Ratchet(threshold=0.5)
        r.decide("d1", {"aesthetic": 0.9})
        r.decide("d2", {"aesthetic": 0.1})
        assert r.discarded_ids() == ["d2"]

    def test_explicit_keep_appears_in_kept_ids(self):
        r = Ratchet()
        r.keep("d1")
        assert "d1" in r.kept_ids()

    def test_explicit_discard_appears_in_discarded_ids(self):
        r = Ratchet()
        r.discard("d1")
        assert "d1" in r.discarded_ids()

    def test_best_none_when_empty(self):
        assert Ratchet().best() is None

    def test_worst_none_when_empty(self):
        assert Ratchet().worst() is None

    def test_best_returns_highest_composite(self):
        r = Ratchet({"aesthetic": 1.0})
        r.decide("d1", {"aesthetic": 0.3})
        r.decide("d2", {"aesthetic": 0.9})
        r.decide("d3", {"aesthetic": 0.6})
        assert r.best().delta_id == "d2"

    def test_worst_returns_lowest_composite(self):
        r = Ratchet({"aesthetic": 1.0})
        r.decide("d1", {"aesthetic": 0.3})
        r.decide("d2", {"aesthetic": 0.9})
        assert r.worst().delta_id == "d1"

    def test_best_with_single_entry(self):
        r = Ratchet()
        r.keep("d1", {"aesthetic": 0.5})
        assert r.best().delta_id == "d1"

    def test_summary_counts(self):
        r = Ratchet(threshold=0.5)
        r.decide("d1", {"aesthetic": 0.9})
        r.decide("d2", {"aesthetic": 0.1})
        r.decide("d3", {"aesthetic": 0.7})
        s = r.summary()
        assert s["total"] == 3
        assert s["kept"] == 2
        assert s["discarded"] == 1

    def test_summary_empty(self):
        s = Ratchet().summary()
        assert s["total"] == 0
        assert s["kept"] == 0
        assert s["discarded"] == 0
        assert s["best_composite"] is None
        assert s["worst_composite"] is None

    def test_summary_threshold_reflected(self):
        r = Ratchet(threshold=0.77)
        assert r.summary()["threshold"] == 0.77

    def test_history_property_returns_copy(self):
        r = Ratchet()
        r.keep("d1")
        h = r.history
        h.append(None)       # mutate the copy
        assert len(r.history) == 1  # original unchanged

    def test_duplicate_delta_ids_allowed(self):
        r = Ratchet()
        r.keep("same_id")
        r.discard("same_id")
        assert len(r.history) == 2


# ---------------------------------------------------------------------------
# extract_recipe (requires usd-core)
# ---------------------------------------------------------------------------

class TestExtractRecipe:
    def test_extract_no_kept_raises(self, usd_stage):
        r = Ratchet()
        r.discard("d1")
        with pytest.raises(RatchetError, match="No kept deltas"):
            r.extract_recipe(usd_stage)

    def test_extract_empty_history_raises(self, usd_stage):
        r = Ratchet()
        with pytest.raises(RatchetError, match="No kept deltas"):
            r.extract_recipe(usd_stage)

    def test_extract_kept_id_not_in_stage_raises(self, usd_stage):
        r = Ratchet()
        r.keep("nonexistent_id_xyz")
        with pytest.raises(RatchetError, match="None of the kept delta ids"):
            r.extract_recipe(usd_stage)

    def test_extract_explicit_empty_kept_ids_raises(self, usd_stage):
        r = Ratchet()
        r.keep("d1")
        with pytest.raises(RatchetError, match="No kept deltas"):
            r.extract_recipe(usd_stage, kept_ids=[])

    def test_extract_single_delta_returns_identifier(self, usd_stage):
        delta_id = usd_stage.add_agent_delta(
            "test_agent", {"/workflows/w1:name": "hello"}
        )
        r = Ratchet()
        r.keep(delta_id)
        recipe_id = r.extract_recipe(usd_stage)
        assert isinstance(recipe_id, str)
        assert len(recipe_id) > 0

    def test_extract_recipe_name_in_identifier(self, usd_stage):
        delta_id = usd_stage.add_agent_delta(
            "agent", {"/workflows/w1:name": "v1"}
        )
        r = Ratchet()
        r.keep(delta_id)
        recipe_id = r.extract_recipe(usd_stage, recipe_name="my_recipe")
        assert "my_recipe" in recipe_id

    def test_extract_composed_value_still_readable(self, usd_stage):
        delta_id = usd_stage.add_agent_delta(
            "agent", {"/workflows/test:name": "from_delta"}
        )
        r = Ratchet()
        r.keep(delta_id)
        r.extract_recipe(usd_stage)
        assert usd_stage.read("/workflows/test", "name") == "from_delta"

    def test_extract_with_explicit_kept_ids(self, usd_stage):
        d1 = usd_stage.add_agent_delta("a", {"/workflows/w1:name": "val1"})
        r = Ratchet()
        # Don't call keep() — pass ids explicitly
        recipe_id = r.extract_recipe(usd_stage, kept_ids=[d1])
        assert recipe_id is not None

    def test_extract_filters_discarded_only_merges_kept(self, usd_stage):
        d1 = usd_stage.add_agent_delta("a", {"/workflows/w1:name": "kept"})
        d2 = usd_stage.add_agent_delta("a", {"/workflows/w2:name": "discarded"})
        r = Ratchet()
        r.keep(d1)
        r.discard(d2)
        recipe_id = r.extract_recipe(usd_stage)
        assert recipe_id is not None

    def test_extract_recipe_layer_added_to_stage_deltas(self, usd_stage):
        before = usd_stage.delta_count
        d1 = usd_stage.add_agent_delta("a", {"/workflows/w1:name": "v"})
        r = Ratchet()
        r.keep(d1)
        r.extract_recipe(usd_stage)
        # delta_count should increase by 2: original delta + recipe layer
        assert usd_stage.delta_count == before + 2

    def test_extract_multiple_kept_deltas(self, usd_stage):
        d1 = usd_stage.add_agent_delta("a", {"/workflows/w1:steps": 20})
        d2 = usd_stage.add_agent_delta("a", {"/workflows/w2:steps": 30})
        r = Ratchet()
        r.keep(d1)
        r.keep(d2)
        recipe_id = r.extract_recipe(usd_stage)
        assert recipe_id is not None

    def test_extract_without_usd_raises_when_mocked(self, monkeypatch):
        import agent.stage.ratchet as rmod
        monkeypatch.setattr(rmod, "HAS_USD", False)
        r = Ratchet()
        r.keep("any_id")
        with pytest.raises(RatchetError, match="USD not available"):
            r.extract_recipe(object())  # dummy cws — never reached


# ---------------------------------------------------------------------------
# FORESIGHT integration — CWM, Experience, Counterfactuals, Arbiter
# ---------------------------------------------------------------------------

@pytest.fixture
def foresight_stage():
    """CWS + CWM-ready stage with some experiences pre-recorded."""
    pytest.importorskip("pxr", reason="usd-core not installed")
    from agent.stage.cognitive_stage import CognitiveWorkflowStage
    return CognitiveWorkflowStage()


@pytest.fixture
def workflow_sig():
    from agent.stage.workflow_signature import WorkflowSignature
    return WorkflowSignature(model_family="sdxl", resolution_band="1024")


class TestForesightDegradation:
    """Ratchet works identically without FORESIGHT components."""

    def test_no_foresight_by_default(self):
        r = Ratchet()
        assert r.has_foresight is False

    def test_decide_without_foresight(self):
        r = Ratchet()
        assert r.decide("d1", {"aesthetic": 0.9}) is True
        assert r.history[0].predicted_scores is None
        assert r.history[0].prediction_accuracy is None
        assert r.history[0].arbiter_mode is None

    def test_keep_without_foresight(self):
        r = Ratchet()
        d = r.keep("d1", {"aesthetic": 0.8})
        assert d.predicted_scores is None

    def test_discard_without_foresight(self):
        r = Ratchet()
        d = r.discard("d1", {"aesthetic": 0.2})
        assert d.predicted_scores is None

    def test_close_session_without_foresight(self):
        r = Ratchet()
        r.keep("d1", {"aesthetic": 0.8})
        assert r.close_session() == []

    def test_summary_shows_foresight_disabled(self):
        r = Ratchet()
        s = r.summary()
        assert s["foresight_enabled"] is False
        assert s["predictions_made"] == 0
        assert s["avg_prediction_accuracy"] is None


class TestCWMIntegration:
    """CWM wired into Ratchet — predicts outcomes and logs accuracy."""

    def test_has_foresight_with_cwm_and_cws(self, foresight_stage):
        from agent.stage.cwm import predict
        r = Ratchet(cws=foresight_stage, cwm=predict)
        assert r.has_foresight is True

    def test_cwm_only_no_cws_is_not_foresight(self):
        r = Ratchet(cwm=object())
        assert r.has_foresight is False

    def test_prediction_logged_on_decide(self, foresight_stage, workflow_sig):
        from agent.stage.cwm import predict
        r = Ratchet(
            cws=foresight_stage, cwm=predict,
            workflow_signature=workflow_sig,
        )
        r.decide(
            "d1", {"aesthetic": 0.9, "lighting": 0.8},
            change_context={"param": "steps", "direction": "increase"},
        )
        d = r.history[0]
        assert d.predicted_scores is not None
        assert "aesthetic" in d.predicted_scores

    def test_prediction_accuracy_computed(self, foresight_stage, workflow_sig):
        from agent.stage.cwm import predict
        r = Ratchet(
            cws=foresight_stage, cwm=predict,
            workflow_signature=workflow_sig,
        )
        r.decide(
            "d1", {"aesthetic": 0.9},
            change_context={"param": "steps", "direction": "increase"},
        )
        d = r.history[0]
        assert d.prediction_accuracy is not None
        assert 0.0 <= d.prediction_accuracy <= 1.0

    def test_no_prediction_without_change_context(self, foresight_stage):
        from agent.stage.cwm import predict
        r = Ratchet(cws=foresight_stage, cwm=predict)
        r.decide("d1", {"aesthetic": 0.9})
        assert r.history[0].predicted_scores is None

    def test_summary_shows_prediction_stats(self, foresight_stage, workflow_sig):
        from agent.stage.cwm import predict
        r = Ratchet(
            cws=foresight_stage, cwm=predict,
            workflow_signature=workflow_sig,
        )
        r.decide(
            "d1", {"aesthetic": 0.8},
            change_context={"param": "steps", "direction": "increase"},
        )
        s = r.summary()
        assert s["foresight_enabled"] is True
        assert s["predictions_made"] >= 1
        assert s["avg_prediction_accuracy"] is not None


class TestExperienceRecording:
    """Experience auto-recorded after every decision."""

    def test_experience_recorded_on_decide(self, foresight_stage, workflow_sig):
        from agent.stage.cwm import predict
        from agent.stage.experience import query_experience
        r = Ratchet(
            cws=foresight_stage, cwm=predict,
            workflow_signature=workflow_sig,
        )
        r.decide(
            "d1", {"aesthetic": 0.7, "lighting": 0.6},
            change_context={"param": "cfg", "direction": "increase"},
        )
        exps = query_experience(foresight_stage)
        assert len(exps) >= 1

    def test_experience_recorded_on_keep(self, foresight_stage, workflow_sig):
        from agent.stage.cwm import predict
        from agent.stage.experience import query_experience
        r = Ratchet(
            cws=foresight_stage, cwm=predict,
            workflow_signature=workflow_sig,
        )
        r.keep(
            "d1", {"aesthetic": 0.9},
            change_context={"action": "add_lora"},
        )
        exps = query_experience(foresight_stage)
        assert len(exps) >= 1

    def test_experience_has_signature_hash(self, foresight_stage, workflow_sig):
        from agent.stage.cwm import predict
        from agent.stage.experience import query_experience
        r = Ratchet(
            cws=foresight_stage, cwm=predict,
            workflow_signature=workflow_sig,
        )
        r.decide(
            "d1", {"aesthetic": 0.8},
            change_context={"param": "steps", "direction": "increase"},
        )
        exps = query_experience(foresight_stage)
        assert exps[0].context_signature_hash == workflow_sig.signature_hash()

    def test_no_experience_without_foresight(self, foresight_stage):
        from agent.stage.experience import query_experience
        r = Ratchet()  # no CWS/CWM
        r.decide("d1", {"aesthetic": 0.8})
        exps = query_experience(foresight_stage)
        assert len(exps) == 0


class TestCounterfactualGeneration:
    """Counterfactual generated at session close."""

    def test_close_session_generates_counterfactual(
        self, foresight_stage, workflow_sig,
    ):
        from agent.stage.counterfactuals import list_pending
        from agent.stage.cwm import predict
        r = Ratchet(
            cws=foresight_stage, cwm=predict,
            workflow_signature=workflow_sig,
        )
        r.keep(
            "d1", {"aesthetic": 0.8, "lighting": 0.7},
            change_context={"param": "steps", "direction": "increase"},
        )
        cf_ids = r.close_session()
        assert len(cf_ids) == 1

        pending = list_pending(foresight_stage)
        assert any(p.cf_id == cf_ids[0] for p in pending)

    def test_close_session_no_kept_no_counterfactual(
        self, foresight_stage, workflow_sig,
    ):
        from agent.stage.cwm import predict
        r = Ratchet(
            cws=foresight_stage, cwm=predict,
            workflow_signature=workflow_sig,
        )
        r.discard("d1", {"aesthetic": 0.2})
        cf_ids = r.close_session()
        assert cf_ids == []

    def test_close_session_picks_best_kept(
        self, foresight_stage, workflow_sig,
    ):
        from agent.stage.counterfactuals import list_pending
        from agent.stage.cwm import predict
        r = Ratchet(
            cws=foresight_stage, cwm=predict,
            workflow_signature=workflow_sig,
        )
        r.keep(
            "d1", {"aesthetic": 0.5},
            change_context={"param": "steps", "direction": "increase"},
        )
        r.keep(
            "d2", {"aesthetic": 0.9},
            change_context={"param": "cfg", "direction": "increase"},
        )
        cf_ids = r.close_session()
        assert len(cf_ids) == 1

        pending = list_pending(foresight_stage)
        # The counterfactual should be based on d2 (best kept)
        cf = next(p for p in pending if p.cf_id == cf_ids[0])
        assert cf.source_chunk_id == "d2"


class TestArbiterIntegration:
    """Arbiter consulted for surfacing mode."""

    def test_arbiter_mode_logged(self, foresight_stage, workflow_sig):
        from agent.stage.arbiter import Arbiter
        from agent.stage.cwm import predict
        arbiter = Arbiter()
        r = Ratchet(
            cws=foresight_stage, cwm=predict,
            arbiter=arbiter, workflow_signature=workflow_sig,
        )
        r.decide(
            "d1", {"aesthetic": 0.8},
            change_context={"param": "steps", "direction": "increase"},
        )
        d = r.history[0]
        assert d.arbiter_mode in ("silent", "soft_surface", "explicit")

    def test_no_arbiter_mode_without_arbiter(self, foresight_stage):
        from agent.stage.cwm import predict
        r = Ratchet(cws=foresight_stage, cwm=predict)
        r.decide(
            "d1", {"aesthetic": 0.8},
            change_context={"param": "steps", "direction": "increase"},
        )
        assert r.history[0].arbiter_mode is None

    def test_arbiter_tracks_decisions(self, foresight_stage, workflow_sig):
        from agent.stage.arbiter import Arbiter
        from agent.stage.cwm import predict
        arbiter = Arbiter()
        r = Ratchet(
            cws=foresight_stage, cwm=predict,
            arbiter=arbiter, workflow_signature=workflow_sig,
        )
        r.decide(
            "d1", {"aesthetic": 0.8},
            change_context={"param": "steps", "direction": "increase"},
        )
        r.decide(
            "d2", {"aesthetic": 0.6},
            change_context={"param": "cfg", "direction": "decrease"},
        )
        assert len(arbiter.decisions) == 2


# ---------------------------------------------------------------------------
# Cycle 39: ratchet _history must be capped (FIFO eviction)
# ---------------------------------------------------------------------------

class TestRatchetHistoryCap:
    """_history must never exceed _max_history entries. (Cycle 39 fix)"""

    def test_history_capped_at_max(self):
        """After max+N decide() calls, history length equals max."""
        from agent.stage.ratchet import _MAX_RATCHET_HISTORY
        r = Ratchet()
        r._max_history = 5  # Override for fast test
        for i in range(10):
            r.decide(f"d{i}", {"aesthetic": 0.7})
        assert len(r.history) == 5

    def test_oldest_entry_evicted(self):
        """The oldest decision is evicted when the cap is exceeded."""
        r = Ratchet()
        r._max_history = 3
        r.decide("first", {"aesthetic": 0.9})
        r.decide("second", {"aesthetic": 0.8})
        r.decide("third", {"aesthetic": 0.7})
        r.decide("fourth", {"aesthetic": 0.6})  # pushes "first" out
        ids = [d.delta_id for d in r.history]
        assert "first" not in ids
        assert "second" in ids
        assert "fourth" in ids

    def test_keep_and_discard_also_cap(self):
        """keep() and discard() explicit methods also respect the cap."""
        r = Ratchet()
        r._max_history = 3
        r.keep("k1")
        r.keep("k2")
        r.discard("d1")
        r.discard("d2")  # 4th call — pushes "k1" out
        assert len(r.history) == 3

    def test_default_cap_constant_in_place(self):
        """_MAX_RATCHET_HISTORY module constant must exist and be reasonable."""
        from agent.stage.ratchet import _MAX_RATCHET_HISTORY
        assert _MAX_RATCHET_HISTORY >= 1_000
