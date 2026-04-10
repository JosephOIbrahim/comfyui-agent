"""Shared utilities for tool modules.

He2025 alignment: deterministic JSON serialization ensures
same inputs -> same outputs across sessions and runs.
"""

import datetime as _datetime
import json as _json
import os
import tempfile
import uuid as _uuid
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
            COMFYUI_DATABASE, COMFYUI_OUTPUT_DIR, PROJECT_DIR, SESSIONS_DIR,
            WORKFLOWS_DIR, COMFYUI_INSTALL_DIR,
        )
        _SAFE_DIRS = [
            COMFYUI_DATABASE.resolve(),
            COMFYUI_OUTPUT_DIR.resolve(),
            PROJECT_DIR.resolve(),
            SESSIONS_DIR.resolve(),
            WORKFLOWS_DIR.resolve(),
        ]
        # Add install dir if it differs from database (split-directory setups)
        install_resolved = COMFYUI_INSTALL_DIR.resolve()
        if install_resolved not in _SAFE_DIRS:
            _SAFE_DIRS.append(install_resolved)
    return _SAFE_DIRS


def validate_path(path_str: str, *, must_exist: bool = False) -> str | None:
    """Validate a file path is safe for tool access.

    Returns None if valid, or an error message string if invalid.
    Rejects path traversal attacks and access to system directories.
    """
    # Block Windows-specific attack vectors before Path.resolve(), which may
    # silently normalise them in ways that bypass safe-dir checks.
    if os.name == "nt":
        _norm = path_str.replace("/", "\\")
        # UNC paths (\\server\share) — potential SSRF / network-share access
        if _norm.startswith("\\\\"):
            return "Access denied: UNC/network paths are not allowed"
        # Drive-relative paths (C:file.txt with no leading slash) — resolve()
        # maps them to the CWD of that drive, potentially escaping safe_dirs
        if len(path_str) >= 2 and path_str[1] == ":" and (
            len(path_str) < 3 or path_str[2] not in ("\\", "/")
        ):
            return "Access denied: drive-relative paths are not allowed"
        # NTFS alternate data streams (file.txt:stream_name) — any colon after
        # the drive-letter position (index 1) indicates a stream identifier
        if ":" in path_str[2:]:
            return "Access denied: NTFS alternate data streams are not allowed"

    try:
        p = Path(path_str).resolve()
    except (OSError, ValueError) as e:
        return f"Invalid path: {e}"

    # Block system directories.
    # Normalize case on Windows because the filesystem is case-insensitive but
    # Path.resolve() may preserve the caller's casing for non-existent paths.
    p_str = str(p)
    if os.name == "nt":
        p_str_cmp = p_str.lower()
        blocked = any(p_str_cmp.startswith(pfx.lower()) for pfx in _BLOCKED_PREFIXES)
    else:
        blocked = any(p_str.startswith(pfx) for pfx in _BLOCKED_PREFIXES)
    if blocked:
        return "Access denied: path is in a protected system directory"

    # Check against allowed directories
    safe_dirs = _get_safe_dirs()
    in_safe_dir = any(p.is_relative_to(sd) for sd in safe_dirs)
    # Also allow temp directories (pytest tmp_path, system temp)
    temp_dir = Path(tempfile.gettempdir()).resolve()
    if p.is_relative_to(temp_dir):
        in_safe_dir = True

    if not in_safe_dir:
        return (
            f"Access denied: path '{path_str}' is outside allowed directories. "
            f"Allowed: ComfyUI database, project dir, sessions, workflows."
        )

    if must_exist and not p.exists():
        return f"File not found: {path_str}"

    return None


def _json_default(obj):
    """Fallback encoder for common non-serializable types.

    Handles Path (→ str), set/frozenset (→ sorted list, He2025 determinism),
    datetime/date (→ ISO-8601 str), and UUID (→ str). (Cycle 35 fix)
    All other types raise TypeError with a descriptive message.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)  # sorted for deterministic output (He2025)
    if isinstance(obj, (_datetime.datetime, _datetime.date)):
        return obj.isoformat()
    if isinstance(obj, _uuid.UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def to_json(obj, **kwargs) -> str:
    """Serialize to JSON with deterministic key ordering.

    Uses _json_default as fallback encoder so Path and set/frozenset values
    don't crash the serializer. Callers can override default= if needed.

    allow_nan=False (Cycle 51): NaN/Infinity are not valid JSON per spec.
    Raising ValueError here is strictly better than silently producing invalid
    JSON that downstream parsers will reject or misinterpret.
    """
    kwargs.setdefault("sort_keys", True)
    kwargs.setdefault("default", _json_default)
    kwargs.setdefault("allow_nan", False)
    return _json.dumps(obj, **kwargs)


from ..errors import error_json  # noqa: F401, E402 -- re-exported for tool modules
