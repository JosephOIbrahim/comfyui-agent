"""Tests for agent/stage/autoresearch_runner.py — mocked execution."""

from __future__ import annotations

import pytest

from agent.stage.autoresearch_runner import (
    AutoresearchRunner,
    RunnerConfig,
    RunResult,
)


SAMPLE_PROGRAM = """\
# Objective
Test photorealistic portraits.

# Parameters
- steps: 20-50, current=30
- cfg: 5.0-12.0, current=7.5

# Anchors
- checkpoint: test.safetensors

# Strategy
- Prioritize aesthetic quality

# Success Criteria
- aesthetic >= 0.8
"""


@pytest.fixture
def usd_stage():
    pytest.importorskip("pxr", reason="usd-core not installed")
    from agent.stage.cognitive_stage import CognitiveWorkflowStage
    return CognitiveWorkflowStage()


def _mock_execute(change_context):
    """Mock execute that returns scores based on change type."""
    if change_context.get("direction") == "increase":
        return {"aesthetic": 0.8, "lighting": 0.7}
    return {"aesthetic": 0.4, "lighting": 0.5}


def _mock_propose_counter():
    """Propose function that returns None after 3 calls."""
    calls = {"n": 0}
    def propose():
        calls["n"] += 1
        if calls["n"] > 3:
            return None
        return {"param": "steps", "direction": "increase"}
    return propose


class TestRunnerConfig:
    def test_defaults(self):
        c = RunnerConfig()
        assert c.budget_hours == 1.0
        assert c.max_experiments == 100


class TestRunResult:
    def test_to_dict(self):
        r = RunResult(stopped_reason="budget_exhausted")
        d = r.to_dict()
        assert d["experiment_count"] == 0
        assert d["stopped_reason"] == "budget_exhausted"


class TestAutoresearchRunner:
    def test_setup(self):
        config = RunnerConfig(max_experiments=5)
        runner = AutoresearchRunner(config, execute_fn=_mock_execute)
        runner.setup()
        assert runner._ratchet is not None

    def test_run_with_max_experiments(self):
        config = RunnerConfig(max_experiments=5)
        runner = AutoresearchRunner(config, execute_fn=_mock_execute)
        result = runner.run()
        assert len(result.experiments) == 5
        assert result.stopped_reason == "max_experiments"

    def test_run_with_no_more_proposals(self):
        config = RunnerConfig(max_experiments=100)
        runner = AutoresearchRunner(
            config,
            execute_fn=_mock_execute,
            propose_fn=_mock_propose_counter(),
        )
        result = runner.run()
        assert len(result.experiments) == 3
        assert result.stopped_reason == "no_more_proposals"

    def test_experiments_have_scores(self):
        config = RunnerConfig(max_experiments=3)
        runner = AutoresearchRunner(config, execute_fn=_mock_execute)
        result = runner.run()
        for exp in result.experiments:
            assert "aesthetic" in exp.axis_scores
            assert exp.composite > 0

    def test_kept_and_discarded(self):
        config = RunnerConfig(max_experiments=6)
        runner = AutoresearchRunner(config, execute_fn=_mock_execute)
        result = runner.run()
        kept = sum(1 for e in result.experiments if e.kept)
        discarded = sum(1 for e in result.experiments if not e.kept)
        assert kept + discarded == 6

    def test_report_generated(self):
        config = RunnerConfig(max_experiments=3)
        runner = AutoresearchRunner(config, execute_fn=_mock_execute)
        result = runner.run()
        assert len(result.report) > 0
        assert "Morning Report" in result.report

    def test_total_seconds_tracked(self):
        config = RunnerConfig(max_experiments=2)
        runner = AutoresearchRunner(config, execute_fn=_mock_execute)
        result = runner.run()
        assert result.total_seconds >= 0.0

    def test_with_cws(self, usd_stage):
        config = RunnerConfig(max_experiments=3)
        runner = AutoresearchRunner(
            config, execute_fn=_mock_execute, cws=usd_stage,
        )
        result = runner.run()
        assert len(result.experiments) == 3

    def test_with_cws_generates_counterfactuals(self, usd_stage):
        config = RunnerConfig(max_experiments=3)
        def good_scores(ctx):
            return {"aesthetic": 0.9, "lighting": 0.85}
        runner = AutoresearchRunner(
            config, execute_fn=good_scores, cws=usd_stage,
        )
        result = runner.run()
        # Should have counterfactuals since CWS is wired in
        assert isinstance(result.counterfactual_ids, list)

    def test_with_program(self, tmp_path):
        prog = tmp_path / "program.md"
        prog.write_text(SAMPLE_PROGRAM, encoding="utf-8")
        config = RunnerConfig(
            max_experiments=2, program_path=str(prog),
        )
        runner = AutoresearchRunner(config, execute_fn=_mock_execute)
        result = runner.run()
        assert "Test photorealistic" in result.report

    def test_default_proposals_cycle(self):
        config = RunnerConfig(max_experiments=7)
        runner = AutoresearchRunner(config, execute_fn=_mock_execute)
        result = runner.run()
        # Should cycle through default proposals
        contexts = [e.change_context for e in result.experiments]
        assert any(c.get("param") == "steps" for c in contexts)
        assert any(c.get("param") == "cfg" for c in contexts)

    def test_execute_failure_continues(self):
        call_count = {"n": 0}
        def failing_execute(ctx):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("GPU OOM")
            return {"aesthetic": 0.7}

        config = RunnerConfig(max_experiments=3)
        runner = AutoresearchRunner(config, execute_fn=failing_execute)
        result = runner.run()
        # Should have completed despite failure
        assert len(result.experiments) >= 1

    def test_result_to_dict(self):
        config = RunnerConfig(max_experiments=2)
        runner = AutoresearchRunner(config, execute_fn=_mock_execute)
        result = runner.run()
        d = result.to_dict()
        assert d["experiment_count"] == 2
        assert d["has_report"] is True


# ---------------------------------------------------------------------------
# Cycle 62: statistics failure must log at DEBUG
# ---------------------------------------------------------------------------

class TestStatisticsLogging:
    """get_statistics failure → log.debug (Cycle 62)."""

    def test_statistics_failure_logs_debug(self, caplog):
        """When get_statistics raises, a debug message must appear in the report."""
        import logging
        from unittest.mock import MagicMock, patch

        cws_mock = MagicMock()
        config = RunnerConfig(max_experiments=1)
        runner = AutoresearchRunner(config, execute_fn=_mock_execute, cws=cws_mock)

        # get_statistics is a local import inside _generate_report — patch source module
        with patch("agent.stage.experience.get_statistics",
                   side_effect=RuntimeError("USD unavailable")), \
             caplog.at_level(logging.DEBUG, logger="agent.stage.autoresearch_runner"):
            result = runner.run()

        assert any("statistic" in r.message.lower()
                   for r in caplog.records), "Expected debug log on statistics failure"

    def test_statistics_failure_still_produces_report(self):
        """Report generation must complete even when statistics are unavailable."""
        from unittest.mock import MagicMock, patch

        cws_mock = MagicMock()
        config = RunnerConfig(max_experiments=2)
        runner = AutoresearchRunner(config, execute_fn=_mock_execute, cws=cws_mock)

        with patch("agent.stage.experience.get_statistics",
                   side_effect=RuntimeError("stats broken")):
            result = runner.run()

        assert len(result.report) > 0
