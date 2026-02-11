"""Shared utilities for tool modules.

He2025 alignment: deterministic JSON serialization ensures
same inputs -> same outputs across sessions and runs.
"""

import json as _json
from pathlib import Path

# Directories that tools are allowed to read/write within.
# Populated lazily from config to avoid circular imports.
_SAFE_DIRS: list[Path] | None = None

_BLOCKED_PREFIXES = (
    "C:\\Windows", "C:\\Program Files", "C:\\ProgramData",
    "/etc", "/usr", "/bin", "/sbin", "/var", "/root",
)


def _get_safe_dirs() -> list[Path]:
    """Lazily load safe directories from config."""
    global _SAFE_DIRS
    if _SAFE_DIRS is None:
        from ..config import (
            COMFYUI_DATABASE, PROJECT_DIR, SESSIONS_DIR, WORKFLOWS_DIR,
        )
        _SAFE_DIRS = [
            COMFYUI_DATABASE.resolve(),
            PROJECT_DIR.resolve(),
            SESSIONS_DIR.resolve(),
            WORKFLOWS_DIR.resolve(),
        ]
    return _SAFE_DIRS


def validate_path(path_str: str, *, must_exist: bool = False) -> str | None:
    """Validate a file path is safe for tool access.

    Returns None if valid, or an error message string if invalid.
    Rejects path traversal attacks and access to system directories.
    """
    try:
        p = Path(path_str).resolve()
    except (OSError, ValueError) as e:
        return f"Invalid path: {e}"

    # Block system directories
    p_str = str(p)
    for prefix in _BLOCKED_PREFIXES:
        if p_str.startswith(prefix):
            return "Access denied: path is in a protected system directory"

    # Check against allowed directories
    safe_dirs = _get_safe_dirs()
    in_safe_dir = any(
        p_str.startswith(str(sd)) for sd in safe_dirs
    )
    # Also allow temp directories (pytest tmp_path, system temp)
    import tempfile
    temp_dir = Path(tempfile.gettempdir()).resolve()
    if p_str.startswith(str(temp_dir)):
        in_safe_dir = True

    if not in_safe_dir:
        return (
            f"Access denied: path '{path_str}' is outside allowed directories. "
            f"Allowed: ComfyUI database, project dir, sessions, workflows."
        )

    if must_exist and not p.exists():
        return f"File not found: {path_str}"

    return None


def to_json(obj, **kwargs) -> str:
    """Serialize to JSON with deterministic key ordering."""
    kwargs.setdefault("sort_keys", True)
    return _json.dumps(obj, **kwargs)
