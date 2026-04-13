"""Tests for agent/tools/_util.py — to_json, validate_path."""

import pytest

from agent.tools._util import to_json


class TestToJsonNanInfinity:
    """to_json must reject NaN and Infinity (not valid JSON per spec)."""

    def test_nan_raises(self):
        """NaN float must not be silently serialized as invalid JSON."""
        with pytest.raises((ValueError, Exception)):
            to_json({"value": float("nan")})

    def test_positive_infinity_raises(self):
        """Positive infinity must not be silently serialized."""
        with pytest.raises((ValueError, Exception)):
            to_json({"value": float("inf")})

    def test_negative_infinity_raises(self):
        """Negative infinity must not be silently serialized."""
        with pytest.raises((ValueError, Exception)):
            to_json({"value": float("-inf")})

    def test_normal_float_serializes(self):
        """Normal finite floats must still serialize correctly."""
        result = to_json({"cfg": 7.5, "denoise": 0.85})
        assert '"cfg": 7.5' in result

    def test_zero_float_serializes(self):
        """Zero float must serialize."""
        result = to_json({"value": 0.0})
        assert "0.0" in result or '"value": 0' in result

    def test_nested_nan_raises(self):
        """NaN nested inside a dict must also fail."""
        with pytest.raises((ValueError, Exception)):
            to_json({"outer": {"inner": float("nan")}})


class TestToJsonDeterminism:
    """to_json must produce sort_keys=True output by default."""

    def test_keys_sorted(self):
        result = to_json({"z": 1, "a": 2, "m": 3})
        assert result.index('"a"') < result.index('"m"') < result.index('"z"')

    def test_path_serialized_as_str(self):
        from pathlib import Path
        p = Path("test") / "file.json"
        result = to_json({"p": p})
        # Path is serialized as a string (exact separators are platform-dependent)
        assert "file.json" in result
        assert isinstance(result, str)

    def test_set_serialized_sorted(self):
        result = to_json({"s": {3, 1, 2}})
        assert "[1, 2, 3]" in result
