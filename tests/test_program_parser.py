"""Tests for agent/stage/program_parser.py — no real I/O except tmp files."""

from __future__ import annotations

import pytest

from agent.stage.program_parser import (
    ParameterAxis,
    ParseError,
    ProgramSpec,
    parse_program,
    parse_program_text,
)


SAMPLE_PROGRAM = """\
# Objective
Generate photorealistic portraits with consistent lighting.

# Parameters
- steps: 20-50, current=30
- cfg: 5.0-12.0, current=7.5
- denoise: 0.3-0.9, current=0.6

# Anchors
- checkpoint: realisticVisionV60.safetensors
- resolution: 1024x1024

# Strategy
- Prioritize aesthetic quality over speed
- Try DPM++ 2M Karras first

# Success Criteria
- aesthetic >= 0.8
- lighting >= 0.7
"""


class TestParameterAxis:
    def test_defaults(self):
        p = ParameterAxis(name="steps")
        assert p.min_value == 0.0
        assert p.max_value == 1.0

    def test_to_dict(self):
        p = ParameterAxis(name="cfg", min_value=5.0, max_value=12.0, current=7.5)
        d = p.to_dict()
        assert d["name"] == "cfg"
        assert d["min"] == 5.0
        assert d["current"] == 7.5

    def test_step_in_dict(self):
        p = ParameterAxis(name="steps", step=5.0)
        d = p.to_dict()
        assert d["step"] == 5.0

    def test_no_step_not_in_dict(self):
        p = ParameterAxis(name="steps")
        assert "step" not in p.to_dict()


class TestProgramSpec:
    def test_defaults(self):
        spec = ProgramSpec()
        assert spec.objective == ""
        assert spec.parameter_axes == []

    def test_to_dict(self):
        spec = ProgramSpec(objective="test", strategy_hints=["hint1"])
        d = spec.to_dict()
        assert d["objective"] == "test"
        assert "hint1" in d["strategy_hints"]

    def test_param_names(self):
        spec = ProgramSpec(parameter_axes=[
            ParameterAxis(name="steps"),
            ParameterAxis(name="cfg"),
        ])
        assert spec.param_names == ["steps", "cfg"]


class TestParseProgramText:
    def test_objective(self):
        spec = parse_program_text(SAMPLE_PROGRAM)
        assert "photorealistic" in spec.objective

    def test_parameters(self):
        spec = parse_program_text(SAMPLE_PROGRAM)
        assert len(spec.parameter_axes) == 3
        steps = next(p for p in spec.parameter_axes if p.name == "steps")
        assert steps.min_value == 20.0
        assert steps.max_value == 50.0
        assert steps.current == 30.0

    def test_cfg_parameter(self):
        spec = parse_program_text(SAMPLE_PROGRAM)
        cfg = next(p for p in spec.parameter_axes if p.name == "cfg")
        assert cfg.min_value == 5.0
        assert cfg.max_value == 12.0
        assert cfg.current == 7.5

    def test_anchors(self):
        spec = parse_program_text(SAMPLE_PROGRAM)
        assert "checkpoint" in spec.anchor_params
        assert spec.anchor_params["checkpoint"] == "realisticVisionV60.safetensors"
        assert spec.anchor_params["resolution"] == "1024x1024"

    def test_strategy_hints(self):
        spec = parse_program_text(SAMPLE_PROGRAM)
        assert len(spec.strategy_hints) == 2
        assert "aesthetic" in spec.strategy_hints[0]

    def test_success_criteria(self):
        spec = parse_program_text(SAMPLE_PROGRAM)
        assert spec.success_criteria["aesthetic"] == 0.8
        assert spec.success_criteria["lighting"] == 0.7

    def test_empty_text(self):
        spec = parse_program_text("")
        assert spec.objective == ""
        assert spec.parameter_axes == []

    def test_partial_program(self):
        text = "# Objective\nJust testing.\n"
        spec = parse_program_text(text)
        assert "testing" in spec.objective
        assert spec.parameter_axes == []

    def test_no_current_defaults_to_midpoint(self):
        text = "# Parameters\n- x: 0.0-10.0\n"
        spec = parse_program_text(text)
        assert len(spec.parameter_axes) == 1
        assert spec.parameter_axes[0].current == 5.0

    def test_step_parameter(self):
        text = "# Parameters\n- steps: 10-50, current=30, step=5\n"
        spec = parse_program_text(text)
        assert spec.parameter_axes[0].step == 5.0

    def test_raw_text_preserved(self):
        spec = parse_program_text(SAMPLE_PROGRAM)
        assert spec.raw_text == SAMPLE_PROGRAM


class TestParseProgram:
    def test_file_not_found(self):
        with pytest.raises(ParseError, match="not found"):
            parse_program("/nonexistent/path.md")

    def test_reads_file(self, tmp_path):
        p = tmp_path / "program.md"
        p.write_text(SAMPLE_PROGRAM, encoding="utf-8")
        spec = parse_program(p)
        assert "photorealistic" in spec.objective
        assert len(spec.parameter_axes) == 3
