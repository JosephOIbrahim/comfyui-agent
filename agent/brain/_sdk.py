"""SDK foundation for standalone brain agents.

Provides BrainConfig (dependency injection container) and BrainAgent (base class).
When running inside the full agent package, get_integrated_config() auto-populates
from agent.config / agent.tools._util / agent.rate_limiter. When standalone,
callers provide their own BrainConfig with sensible defaults.
"""

import json as _json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..workflow_session import WorkflowSession

_registration_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Defaults for standalone mode
# ---------------------------------------------------------------------------

def _default_to_json(obj: Any, **kwargs) -> str:
    """Deterministic JSON serialization (He2025 pattern)."""
    kwargs.setdefault("sort_keys", True)
    kwargs.setdefault("allow_nan", False)  # Cycle 53: NaN/Infinity raises ValueError
    return _json.dumps(obj, **kwargs)


def _default_validate_path(path_str: str, *, must_exist: bool = False) -> str | None:
    """Permissive path validation for standalone use."""
    try:
        p = Path(path_str).resolve()
    except (OSError, ValueError) as e:
        return f"Invalid path: {e}"
    if must_exist and not p.exists():
        return f"File not found: {path_str}"
    return None


class _NullLimiter:
    """No-op rate limiter for standalone use."""

    def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        return True


_NULL_LIMITER = _NullLimiter()


def _null_limiter_factory() -> _NullLimiter:
    return _NULL_LIMITER


# ---------------------------------------------------------------------------
# BrainConfig — dependency injection container
# ---------------------------------------------------------------------------

@dataclass
class BrainConfig:
    """Configuration container for brain agents.

    When integrated with the full agent package, auto-populated via
    get_integrated_config(). When standalone, caller provides values
    or uses defaults.
    """

    to_json: Callable[..., str] = field(default_factory=lambda: _default_to_json)
    validate_path: Callable[..., str | None] = field(default_factory=lambda: _default_validate_path)
    sessions_dir: Path = field(default_factory=lambda: Path("./sessions"))
    comfyui_url: str = "http://127.0.0.1:8188"
    custom_nodes_dir: Path = field(default_factory=lambda: Path("./Custom_Nodes"))
    models_dir: Path = field(default_factory=lambda: Path("./models"))
    agent_model: str = "claude-sonnet-4-20250514"
    vision_limiter: Callable = field(default_factory=lambda: _null_limiter_factory)
    tool_dispatcher: Callable | None = None
    get_workflow_state: Callable | None = None
    patch_handle: Callable | None = None


# ---------------------------------------------------------------------------
# BrainAgent — base class
# ---------------------------------------------------------------------------

class BrainAgent:
    """Base class for SDK-ready brain agents.

    Subclasses override TOOLS and handle(). Config is injected via
    __init__; if None, falls back to get_integrated_config().

    Auto-registration: subclasses are collected via __init_subclass__.
    Use get_all_tools() and dispatch() for centralized routing.
    """

    TOOLS: list[dict] = []
    _registry: dict[str, 'BrainAgent'] = {}
    _all_tools: list[dict] = []
    _registered: bool = False

    def __init__(self, config: BrainConfig | None = None):
        if config is None:
            config = get_integrated_config()
        self.cfg = config
        self.to_json = config.to_json

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def handle(self, name: str, tool_input: dict) -> str:
        raise NotImplementedError

    @classmethod
    def _register_all(cls):
        if cls._registered:  # Fast path — no lock needed after first registration
            return
        with _registration_lock:
            if cls._registered:  # Double-check after acquiring lock
                return
            cls._registered = True
            for subcls in cls.__subclasses__():
                if not subcls.TOOLS:
                    continue
                instance = subcls()
                for tool in subcls.TOOLS:
                    cls._registry[tool["name"]] = instance
                    cls._all_tools.append(tool)

    @classmethod
    def get_all_tools(cls) -> list[dict]:
        cls._register_all()
        return list(cls._all_tools)

    @classmethod
    def dispatch(cls, name: str, tool_input: dict) -> str:
        cls._register_all()
        agent = cls._registry.get(name)
        if agent is None:
            return _json.dumps({"error": f"Unknown brain tool: {name}"}, sort_keys=True, allow_nan=False)  # Cycle 59
        try:
            return agent.handle(name, tool_input)
        except Exception as exc:  # Cycle 36: defense-in-depth — isolate agent crashes
            return _json.dumps({"error": f"Brain tool '{name}' raised: {exc}"}, sort_keys=True, allow_nan=False)  # Cycle 59

    @classmethod
    def _reset_registry(cls):
        """Reset for testing."""
        with _registration_lock:
            cls._registry.clear()
            cls._all_tools.clear()
            cls._registered = False


# ---------------------------------------------------------------------------
# Integrated config — lazy singleton
# ---------------------------------------------------------------------------

_integrated_config: BrainConfig | None = None
_config_lock = threading.Lock()


def _lazy_tool_dispatcher(name: str, tool_input: dict) -> str:
    from ..tools import handle
    return handle(name, tool_input)


def _lazy_get_workflow_state() -> "WorkflowSession":
    from ..tools.workflow_patch import _get_state
    return _get_state()


def _lazy_patch_handle(name: str, tool_input: dict) -> str:
    """Dispatch to workflow_patch internal handlers, bypassing the outer lock.

    ``workflow_patch.handle`` wraps every call in ``with _state_lock:`` but
    ``WorkflowSession.__getitem__`` also acquires the *same* non-reentrant
    lock, causing a deadlock.  Import the internal handlers directly.
    """
    from ..tools.workflow_patch import (
        _handle_apply_patch,
        _handle_set_input,
    )
    if name == "apply_workflow_patch":
        return _handle_apply_patch(tool_input)
    elif name == "set_input":
        return _handle_set_input(tool_input)
    # Fallback for other tool names (undo, diff, etc.)
    from ..tools.workflow_patch import handle
    return handle(name, tool_input)


def get_integrated_config() -> BrainConfig:
    """Build a BrainConfig from the full agent package. Cached singleton."""
    global _integrated_config
    if _integrated_config is not None:
        return _integrated_config

    with _config_lock:
        if _integrated_config is not None:
            return _integrated_config

        from ..config import (
            AGENT_MODEL, COMFYUI_URL, CUSTOM_NODES_DIR, MODELS_DIR, SESSIONS_DIR,
        )
        from ..rate_limiter import VISION_LIMITER
        from ..tools._util import to_json, validate_path

        _integrated_config = BrainConfig(
            to_json=to_json,
            validate_path=validate_path,
            sessions_dir=SESSIONS_DIR,
            comfyui_url=COMFYUI_URL,
            custom_nodes_dir=CUSTOM_NODES_DIR,
            models_dir=MODELS_DIR,
            agent_model=AGENT_MODEL,
            vision_limiter=VISION_LIMITER,
            tool_dispatcher=_lazy_tool_dispatcher,
            get_workflow_state=_lazy_get_workflow_state,
            patch_handle=_lazy_patch_handle,
        )
        return _integrated_config


def reset_integrated_config() -> None:
    """Reset the cached config. For testing only."""
    global _integrated_config
    with _config_lock:
        _integrated_config = None
