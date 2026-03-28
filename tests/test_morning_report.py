"""Tests for agent/stage/morning_report.py — pure logic, no I/O."""

from __future__ import annotations

from agent.stage.morning_report import generate_report


def _make_decision(delta_id, kept, composite, scores=None, pred_acc=None):
    return {
        "delta_id": delta_id,
        "kept": kept,
        "composite": composite,
        "axis_scores": scores or {},
        "prediction_accuracy": pred_acc,
    }


class TestGenerateReport:
    def test_empty_report(self):
        report = generate_report()
        assert "Morning Report" in report
        assert "Total experiments:** 0" in report

    def test_with_history(self):
        history = [
            _make_decision("d1", True, 0.8, {"aesthetic": 0.9}),
            _make_decision("d2", False, 0.3, {"aesthetic": 0.2}),
        ]
        report = generate_report(history)
        assert "Total experiments:** 2" in report
        assert "Kept:** 1" in report

    def test_score_trajectory(self):
        history = [
            _make_decision("d1", True, 0.5),
            _make_decision("d2", True, 0.7),
            _make_decision("d3", True, 0.9),
        ]
        report = generate_report(history)
        assert "First:** 0.500" in report
        assert "Best:** 0.900" in report

    def test_trend_improving(self):
        history = [
            _make_decision(f"d{i}", True, 0.3 + i * 0.05)
            for i in range(9)
        ]
        report = generate_report(history)
        assert "Improving" in report

    def test_best_recipe(self):
        history = [
            _make_decision("d1", True, 0.5, {"aesthetic": 0.5}),
            _make_decision("d2", True, 0.9, {"aesthetic": 0.95}),
        ]
        report = generate_report(history)
        assert "Best Recipe" in report
        assert "d2" in report

    def test_axis_impact(self):
        history = [
            _make_decision("d1", True, 0.9, {"aesthetic": 0.9, "depth": 0.5}),
            _make_decision("d2", False, 0.2, {"aesthetic": 0.1, "depth": 0.4}),
        ]
        report = generate_report(history)
        assert "Most Impactful" in report

    def test_prediction_accuracy(self):
        history = [
            _make_decision("d1", True, 0.8, pred_acc=0.85),
            _make_decision("d2", True, 0.7, pred_acc=0.92),
        ]
        report = generate_report(history)
        assert "Prediction Accuracy" in report
        assert "Predictions made:** 2" in report

    def test_experience_stats(self):
        stats = {
            "total_count": 50,
            "unique_signatures": 3,
            "avg_outcome": {"aesthetic": 0.72, "lighting": 0.68},
        }
        report = generate_report(experience_stats=stats)
        assert "Experience Base" in report
        assert "Total experiences:** 50" in report

    def test_counterfactuals(self):
        report = generate_report(counterfactual_count=5)
        assert "Counterfactuals" in report
        assert "Generated:** 5" in report

    def test_warnings(self):
        report = generate_report(warnings=["Low VRAM detected", "GPU throttling"])
        assert "Warnings" in report
        assert "Low VRAM" in report

    def test_program_objective(self):
        report = generate_report(program_objective="Generate portraits")
        assert "Generate portraits" in report

    def test_session_name(self):
        report = generate_report(session_name="Overnight Run")
        assert "Overnight Run" in report

    def test_no_prediction_section_without_predictions(self):
        history = [_make_decision("d1", True, 0.8)]
        report = generate_report(history)
        assert "Prediction Accuracy" not in report

    def test_keep_rate_calculation(self):
        history = [
            _make_decision("d1", True, 0.8),
            _make_decision("d2", True, 0.7),
            _make_decision("d3", False, 0.3),
            _make_decision("d4", True, 0.6),
        ]
        report = generate_report(history)
        assert "75%" in report
