"""Structured logging with JSON format, rotation, and correlation IDs.

Provides two formatters:
- JSONFormatter: machine-parseable JSON lines (sort_keys=True for He2025)
- HumanFormatter: clean human-readable output for interactive CLI

Correlation IDs use thread-local storage to tag all log entries from
a single agent session, making it easy to trace a conversation across
interleaved log output.
"""

import json
import logging
import logging.handlers
import threading
import uuid
from pathlib import Path


# ---------------------------------------------------------------------------
# Correlation ID (thread-local)
# ---------------------------------------------------------------------------

_local = threading.local()


def set_correlation_id(corr_id: str | None = None) -> str:
    """Set (or generate) a correlation ID for the current thread.

    Returns the correlation ID that was set.
    """
    cid = corr_id or uuid.uuid4().hex[:12]
    _local.correlation_id = cid
    return cid


def get_correlation_id() -> str | None:
    """Get the correlation ID for the current thread, or None."""
    return getattr(_local, "correlation_id", None)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects.

    Fields: timestamp, level, logger, message, correlation_id (if set).
    sort_keys=True for He2025 deterministic serialization.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
        }
        corr_id = get_correlation_id()
        if corr_id:
            entry["correlation_id"] = corr_id
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, sort_keys=True)


class HumanFormatter(logging.Formatter):
    """Clean human-readable log format for interactive CLI use."""

    def format(self, record: logging.LogRecord) -> str:
        corr_id = get_correlation_id()
        prefix = f"[{corr_id}] " if corr_id else ""
        base = f"{prefix}{record.name}: {record.getMessage()}"
        if record.exc_info and record.exc_info[1]:
            base += "\n" + self.formatException(record.exc_info)
        return base


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(
    *,
    level: int = logging.WARNING,
    log_file: Path | str | None = None,
    json_format: bool = False,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> None:
    """Configure logging for the agent.

    Args:
        level: Logging level (e.g. logging.DEBUG, logging.WARNING).
        log_file: Optional path to a log file (enables rotation).
        json_format: If True, use JSONFormatter; otherwise HumanFormatter.
        max_bytes: Max log file size before rotation (default 10 MB).
        backup_count: Number of rotated log files to keep.
    """
    root = logging.getLogger()

    # Remove existing handlers to avoid duplicate output
    root.handlers.clear()
    root.setLevel(level)

    formatter = JSONFormatter() if json_format else HumanFormatter()

    # stderr handler (always)
    import sys
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    # File handler with rotation (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        root.addHandler(file_handler)
