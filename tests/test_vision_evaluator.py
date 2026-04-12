"""Tests for the vision-based QualityScore evaluator.

Validates that the vision evaluator correctly maps analyzer output
to QualityScore, falls back gracefully, and integrates with the
pipeline's evaluator selection logic.
"""

import copy
import json

import pytest
from types import SimpleNamespace
from unittest.mock import patch

from cognitive.pipeline.autonomous import (
    AutonomousPipeline,
    PipelineConfig,
)
from cognitive.experience.chunk import QualityScore


# ---------------------------------------------------------------------------
# Mock compose to isolate from template changes
# ---------------------------------------------------------------------------

_KNOWN_WORKFLOW = {
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "test.safetensors"}},
    "2": {"class_type": "KSampler",
          "inputs": {"model": ["1", 0], "seed": 42, "steps": 20, "cfg": 7.0,
                     "sampler_name": "euler", "scheduler": "normal",
                     "denoise": 1.0}},
}


def _mock_compose(*args, **kwargs):
    return SimpleNamespace(
        success=True,
        workflow_data=copy.deepcopy(_KNOWN_WORKFLOW),
        plan=SimpleNamespace(model_family="SD1.5", parameters={"cfg": 7.0, "steps": 20}),
        error=None,
    )


@pytest.fixture(autouse=True)
def _patch_compose(monkeypatch):
    import cognitive.pipeline.autonomous as _mod
    monkeypatch.setattr(_mod, "compose_workflow", _mock_compose)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_exec_result(filenames: list[str] | None = None, success: bool = True):
    """Create a mock execution result."""
    fnames = filenames if filenames is not None else ["output.png"]
    return type("R", (), {"output_filenames": fnames, "success": success})()


def _make_analyzer(response: dict):
    """Return an analyzer callable that returns *response* for any path."""
    def analyzer(image_path: str) -> dict:
        return response
    return analyzer


# ---------------------------------------------------------------------------
# Vision evaluator unit tests
# ---------------------------------------------------------------------------

class TestVisionEvaluator:

    def test_maps_quality_score(self):
        analyzer = _make_analyzer({
            "quality_score": 0.85,
            "technical_score": 0.8,
            "aesthetic_score": 0.9,
            "prompt_adherence": 0.75,
        })
        evaluator = AutonomousPipeline._vision_evaluator(analyzer)
        result = evaluator(_make_exec_result())
        assert isinstance(result, QualityScore)
        assert result.overall == pytest.approx(0.85)
        assert result.technical == pytest.approx(0.8)
        assert result.aesthetic == pytest.approx(0.9)
        assert result.prompt_adherence == pytest.approx(0.75)
        assert result.source == "vision"

    def test_missing_fields_default_to_zero(self):
        analyzer = _make_analyzer({"quality_score": 0.7})
        evaluator = AutonomousPipeline._vision_evaluator(analyzer)
        result = evaluator(_make_exec_result())
        assert result.overall == pytest.approx(0.7)
        assert result.technical == 0.0
        assert result.aesthetic == 0.0

    def test_fallback_when_no_output_files(self):
        analyzer = _make_analyzer({"quality_score": 0.9})
        evaluator = AutonomousPipeline._vision_evaluator(analyzer)
        result = evaluator(_make_exec_result(filenames=[]))
        assert result.source == "vision_fallback"
        assert result.overall == pytest.approx(0.5)

    def test_fallback_on_analyzer_exception(self):
        def broken_analyzer(path):
            raise RuntimeError("API down")

        evaluator = AutonomousPipeline._vision_evaluator(broken_analyzer)
        result = evaluator(_make_exec_result())
        assert result.source == "vision_error"
        assert result.overall == pytest.approx(0.5)

    def test_json_string_response(self):
        """Analyzer returning JSON string should be parsed."""
        response_str = json.dumps({
            "quality_score": 0.6,
            "prompt_adherence": 0.5,
        })

        def str_analyzer(path):
            return response_str

        evaluator = AutonomousPipeline._vision_evaluator(str_analyzer)
        result = evaluator(_make_exec_result())
        assert result.overall == pytest.approx(0.6)
        assert result.source == "vision"

    def test_non_dict_non_string_response(self):
        def bad_analyzer(path):
            return 42

        evaluator = AutonomousPipeline._vision_evaluator(bad_analyzer)
        result = evaluator(_make_exec_result())
        assert result.source == "vision_type_error"

    def test_invalid_json_string(self):
        def bad_json_analyzer(path):
            return "not valid json {{"

        evaluator = AutonomousPipeline._vision_evaluator(bad_json_analyzer)
        result = evaluator(_make_exec_result())
        assert result.source == "vision_parse_error"


# ---------------------------------------------------------------------------
# Pipeline evaluator selection
# ---------------------------------------------------------------------------

class TestEvaluatorSelection:

    def test_explicit_evaluator_wins(self):
        """When config.evaluator is set, it takes priority."""
        pipeline = AutonomousPipeline()
        result = pipeline.run(PipelineConfig(
            intent="test",
            evaluator=lambda _: QualityScore(overall=0.99, source="custom"),
            vision_analyzer=_make_analyzer({"quality_score": 0.1}),
        ))
        assert result.quality.overall == pytest.approx(0.99)
        assert result.quality.source == "custom"

    def test_vision_analyzer_used_when_no_evaluator(self):
        """vision_analyzer is used when evaluator is None."""
        pipeline = AutonomousPipeline()
        result = pipeline.run(PipelineConfig(
            intent="test",
            executor=lambda wf: _make_exec_result(
                filenames=["out.png"], success=True,
            ),
            vision_analyzer=_make_analyzer({
                "quality_score": 0.72,
                "technical_score": 0.6,
            }),
        ))
        assert result.quality.source == "vision"
        assert result.quality.overall == pytest.approx(0.72)

    def test_default_evaluator_without_vision(self):
        """Falls back to rule-based when neither evaluator nor vision_analyzer."""
        pipeline = AutonomousPipeline()
        result = pipeline.run(PipelineConfig(intent="test"))
        assert result.quality.source == "rule"
