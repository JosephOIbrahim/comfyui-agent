"""Tests for agent/logging_config.py — structured logging, formatters, correlation IDs."""

import json
import logging

import pytest

from agent.logging_config import (
    JSONFormatter,
    HumanFormatter,
    get_correlation_id,
    set_correlation_id,
    setup_logging,
)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging state between tests."""
    # Clear correlation ID
    import agent.logging_config as lc
    if hasattr(lc._local, "correlation_id"):
        del lc._local.correlation_id
    yield
    # Reset root logger
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)


class TestCorrelationId:
    def test_default_none(self):
        assert get_correlation_id() is None

    def test_set_returns_id(self):
        cid = set_correlation_id()
        assert cid is not None
        assert len(cid) == 12

    def test_get_after_set(self):
        set_correlation_id("test-123")
        assert get_correlation_id() == "test-123"

    def test_custom_id(self):
        cid = set_correlation_id("my-session-id")
        assert cid == "my-session-id"
        assert get_correlation_id() == "my-session-id"

    def test_auto_generated_is_hex(self):
        cid = set_correlation_id()
        int(cid, 16)  # Should not raise — valid hex


class TestJSONFormatter:
    def _make_record(self, msg="test message", level=logging.INFO):
        record = logging.LogRecord(
            name="test.logger",
            level=level,
            pathname="test.py",
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )
        return record

    def test_produces_valid_json(self):
        fmt = JSONFormatter()
        record = self._make_record()
        output = fmt.format(record)
        data = json.loads(output)
        assert data["message"] == "test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert "timestamp" in data

    def test_sort_keys(self):
        """JSON output uses sort_keys=True (He2025 compliance)."""
        fmt = JSONFormatter()
        record = self._make_record()
        output = fmt.format(record)
        data = json.loads(output)
        keys = list(data.keys())
        assert keys == sorted(keys)

    def test_includes_correlation_id(self):
        set_correlation_id("corr-abc")
        fmt = JSONFormatter()
        record = self._make_record()
        output = fmt.format(record)
        data = json.loads(output)
        assert data["correlation_id"] == "corr-abc"

    def test_no_correlation_id_when_unset(self):
        fmt = JSONFormatter()
        record = self._make_record()
        output = fmt.format(record)
        data = json.loads(output)
        assert "correlation_id" not in data


class TestHumanFormatter:
    def _make_record(self, msg="test message", level=logging.INFO):
        record = logging.LogRecord(
            name="test.logger",
            level=level,
            pathname="test.py",
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )
        return record

    def test_readable_output(self):
        fmt = HumanFormatter()
        record = self._make_record()
        output = fmt.format(record)
        assert "test.logger" in output
        assert "test message" in output

    def test_includes_correlation_id(self):
        set_correlation_id("sess-xyz")
        fmt = HumanFormatter()
        record = self._make_record()
        output = fmt.format(record)
        assert "[sess-xyz]" in output

    def test_no_prefix_without_correlation_id(self):
        fmt = HumanFormatter()
        record = self._make_record()
        output = fmt.format(record)
        assert not output.startswith("[")


class TestSetupLogging:
    def test_creates_stderr_handler(self):
        setup_logging(level=logging.DEBUG)
        root = logging.getLogger()
        assert len(root.handlers) >= 1
        assert root.level == logging.DEBUG

    def test_creates_file_handler(self, tmp_path):
        log_file = tmp_path / "test.log"
        setup_logging(level=logging.INFO, log_file=log_file)
        root = logging.getLogger()
        assert len(root.handlers) >= 2  # stderr + file

    def test_json_format_flag(self):
        setup_logging(level=logging.INFO, json_format=True)
        root = logging.getLogger()
        assert any(
            isinstance(h.formatter, JSONFormatter) for h in root.handlers
        )

    def test_log_file_auto_creates_dir(self, tmp_path):
        log_file = tmp_path / "subdir" / "test.log"
        setup_logging(level=logging.INFO, log_file=log_file)
        assert log_file.parent.exists()

    def test_file_handler_always_json(self, tmp_path):
        """File handler always uses JSON formatter even in human mode."""
        log_file = tmp_path / "test.log"
        setup_logging(level=logging.DEBUG, log_file=log_file, json_format=False)
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers
            if hasattr(h, "baseFilename")
        ]
        assert len(file_handlers) == 1
        assert isinstance(file_handlers[0].formatter, JSONFormatter)
