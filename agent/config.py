"""Configuration and environment handling."""

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root regardless of working directory (supports MCP server launch)
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# LLM Provider selection — anthropic (default), openai, gemini, ollama
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")

# Provider API keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# API-key validation is deferred to a callable (T5 from the 5x review).
# Pre-fix this printed at import time, leaking the warning into every
# `agent --help`, `agent inspect`, etc. — confusing because those
# commands don't need an API key. Callers who DO need the LLM call
# `warn_on_missing_api_key()` explicitly at their entry point.
_api_key_warn_emitted = False


def warn_on_missing_api_key() -> None:
    """Emit the API-key-missing warning once per process, IF needed.

    Idempotent: only emits on first call. Safe to call from any command
    that requires the LLM. Commands that DON'T need the LLM (--help,
    inspect, parse, autonomous --execute-mode {mock,dry-run}) should
    NOT call this.
    """
    global _api_key_warn_emitted
    if _api_key_warn_emitted:
        return
    _api_key_warn_emitted = True
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        print(
            "WARNING: ANTHROPIC_API_KEY is not set. "
            "Set it in your .env file or environment. "
            "Get your key at https://console.anthropic.com/",
            file=sys.stderr,
        )
    # Validate format (warn, don't block — key may be valid in other formats)
    elif ANTHROPIC_API_KEY and not re.match(r"^sk-ant-", ANTHROPIC_API_KEY):
        print(
            "WARNING: ANTHROPIC_API_KEY doesn't match expected format (sk-ant-...). "
            "Verify your key at https://console.anthropic.com/",
            file=sys.stderr,
        )

# MCP auth token (optional — for future HTTP/SSE transport auth)
MCP_AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN")

# Third-party API keys (optional — improve rate limits for external services)
CIVITAI_API_KEY = os.getenv("CIVITAI_API_KEY")    # Optional — improves CivitAI rate limits
GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")   # Optional — improves GitHub API rate limits
HF_TOKEN = os.getenv("HF_TOKEN")                  # Cycle 58: required for gated HF models (Flux, SD3, etc.)

# Model selection — if not set, uses the default for the active LLM_PROVIDER.
# Defaults per provider: anthropic=claude-sonnet-4-20250514, openai=gpt-4o,
# gemini=gemini-2.5-flash, ollama=llama3.1
_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.1",
}
AGENT_MODEL = os.getenv("AGENT_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, "claude-sonnet-4-20250514"))
# ^ Only affects CLI mode (agent run). MCP mode inherits the model from Claude Code.
# Override in .env: AGENT_MODEL=gpt-4o or AGENT_MODEL=claude-opus-4-6-20250929
MAX_TOKENS = 16384
MAX_AGENT_TURNS = 30

# Context management
COMPACT_THRESHOLD = 120_000  # tokens — start compacting at this level

# API resilience
API_MAX_RETRIES = 3
API_RETRY_DELAY = 1.0  # seconds — base delay, doubles each retry

# ComfyUI connection
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1").strip().rstrip("/")
_port_raw = os.getenv("COMFYUI_PORT", "8188")
try:
    COMFYUI_PORT = int(_port_raw)
    if not (1 <= COMFYUI_PORT <= 65535):
        print(
            f"WARNING: COMFYUI_PORT={COMFYUI_PORT} out of range (1-65535). Using 8188.",
            file=sys.stderr,
        )
        COMFYUI_PORT = 8188
except ValueError:
    print(
        f"WARNING: COMFYUI_PORT='{_port_raw}' is not a valid integer. "
        "Falling back to default port 8188.",
        file=sys.stderr,
    )
    COMFYUI_PORT = 8188
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

# Kill switches — independently disable subsystems (all default ON)
BRAIN_ENABLED = os.getenv("BRAIN_ENABLED", "1") == "1"
OBSERVATION_ENABLED = os.getenv("OBSERVATION_ENABLED", "1") == "1"
DAG_ENABLED = os.getenv("DAG_ENABLED", "1") == "1"
GATE_ENABLED = os.getenv("GATE_ENABLED", "1") == "1"

# Paths — cross-platform defaults for ComfyUI database location
def _default_comfyui_database() -> str:
    """Sensible default ComfyUI database path per platform."""
    return str(Path.home() / "ComfyUI")


COMFYUI_DATABASE = Path(os.getenv("COMFYUI_DATABASE", _default_comfyui_database()))
CUSTOM_NODES_DIR = COMFYUI_DATABASE / "Custom_Nodes"
MODELS_DIR = COMFYUI_DATABASE / "models"
WORKFLOWS_DIR = COMFYUI_DATABASE / "Workflows"

# Output directory — may differ from COMFYUI_DATABASE when using extra_model_paths
# or symlinked setups. Override with COMFYUI_OUTPUT_DIR in .env.
def _default_comfyui_output() -> str:
    """Default output directory. Checks COMFYUI_OUTPUT_DIR env var first."""
    env = os.getenv("COMFYUI_OUTPUT_DIR")
    if env:
        return env
    return str(COMFYUI_DATABASE / "output")


COMFYUI_OUTPUT_DIR = Path(_default_comfyui_output())

# ComfyUI installation directory — auto-detected or overridden via env.
# This is the actual ComfyUI repo (with /blueprints, /comfy, etc.),
# which may differ from COMFYUI_DATABASE on split-directory setups.
def _default_comfyui_install() -> str:
    """Auto-detect ComfyUI installation path."""
    env = os.getenv("COMFYUI_INSTALL_DIR")
    if env:
        return env
    # Auto-detect: check common locations for the actual ComfyUI install
    candidates = [
        COMFYUI_DATABASE / "ComfyUI",
        Path.home() / "ComfyUI",
    ]
    for candidate in candidates:
        if (candidate / "comfy").is_dir() or (candidate / "main.py").exists():
            return str(candidate)
    # Fallback: custom_nodes symlink may point back to the install
    custom_nodes_link = CUSTOM_NODES_DIR
    if custom_nodes_link.is_symlink():
        resolved = custom_nodes_link.resolve().parent
        if (resolved / "main.py").exists():
            return str(resolved)
    return str(COMFYUI_DATABASE)


COMFYUI_INSTALL_DIR = Path(_default_comfyui_install())
COMFYUI_BLUEPRINTS_DIR = COMFYUI_INSTALL_DIR / "blueprints"

# Model catalog (rich metadata about installed models)
MODEL_CATALOG_PATH = COMFYUI_DATABASE / "model_catalog.json"

# Experience accumulator persistence — JSONL file that survives between sessions
EXPERIENCE_FILE = COMFYUI_DATABASE / "comfy-cozy-experience.jsonl"

# Auto-initialization (see startup.py)
AUTO_SCAN_MODELS = os.getenv("AUTO_SCAN_MODELS", "false").lower() == "true"
AUTO_SCAN_WORKFLOWS = os.getenv("AUTO_SCAN_WORKFLOWS", "false").lower() == "true"
AUTO_LOAD_WORKFLOW = os.getenv("AUTO_LOAD_WORKFLOW", "")
AUTO_LOAD_SESSION = os.getenv("AUTO_LOAD_SESSION", "")

# Stage persistence — durable USD checkpoint across sessions.
# STAGE_DEFAULT_PATH: if set, ensure_stage() loads from this .usda file on cold
# start and uses it as the default flush target. Empty string = in-memory only.
STAGE_DEFAULT_PATH = os.getenv("STAGE_DEFAULT_PATH", "")
# STAGE_AUTOSAVE_SECONDS: interval for the daemon flush timer. 0 disables.
STAGE_AUTOSAVE_SECONDS = int(os.getenv("STAGE_AUTOSAVE_SECONDS", "300"))
# STAGE_AUTOLOAD_EXPERIENCE: when "true", the cognitive ExperienceAccumulator
# is loaded from EXPERIENCE_FILE on first ensure_stage(). Wires the dormant
# create_default_pipeline() into the live runtime.
STAGE_AUTOLOAD_EXPERIENCE = os.getenv("STAGE_AUTOLOAD_EXPERIENCE", "false").lower() == "true"

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
SESSIONS_DIR = PROJECT_DIR / "sessions"
LOCAL_WORKFLOWS_DIR = PROJECT_DIR / "workflows"
LOG_DIR = PROJECT_DIR / "logs"
