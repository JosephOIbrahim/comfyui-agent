"""Configuration and environment handling."""

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Validate API key format (warn, don't block — key may be valid in other formats)
if ANTHROPIC_API_KEY and not re.match(r"^sk-ant-", ANTHROPIC_API_KEY):
    print(
        "WARNING: ANTHROPIC_API_KEY doesn't match expected format (sk-ant-...). "
        "Verify your key at https://console.anthropic.com/",
        file=sys.stderr,
    )

# MCP auth token (optional — for future HTTP/SSE transport auth)
MCP_AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN")
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-4-20250514")
# ^ Only affects CLI mode (agent run). MCP mode inherits the model from Claude Code.
# Override in .env: AGENT_MODEL=claude-opus-4-6-20250929 for higher quality CLI sessions.
MAX_TOKENS = 16384
MAX_AGENT_TURNS = 30

# Context management
COMPACT_THRESHOLD = 120_000  # tokens — start compacting at this level

# API resilience
API_MAX_RETRIES = 3
API_RETRY_DELAY = 1.0  # seconds — base delay, doubles each retry

# ComfyUI connection
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
_port_raw = os.getenv("COMFYUI_PORT", "8188")
try:
    COMFYUI_PORT = int(_port_raw)
except ValueError:
    print(
        f"WARNING: COMFYUI_PORT='{_port_raw}' is not a valid integer. "
        "Falling back to default port 8188.",
        file=sys.stderr,
    )
    COMFYUI_PORT = 8188
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

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

# ComfyUI installation directory — defaults to COMFYUI_DATABASE unless overridden.
# Override with COMFYUI_INSTALL_DIR in .env when install and data dirs differ.
def _default_comfyui_install() -> str:
    """Default ComfyUI installation path. May differ from COMFYUI_DATABASE."""
    env = os.getenv("COMFYUI_INSTALL_DIR")
    if env:
        return env
    return str(COMFYUI_DATABASE)


COMFYUI_INSTALL_DIR = Path(_default_comfyui_install())
COMFYUI_BLUEPRINTS_DIR = COMFYUI_INSTALL_DIR / "blueprints"

# Model catalog (rich metadata about installed models)
MODEL_CATALOG_PATH = COMFYUI_DATABASE / "model_catalog.json"

# Auto-initialization (see startup.py)
AUTO_SCAN_MODELS = os.getenv("AUTO_SCAN_MODELS", "false").lower() == "true"
AUTO_SCAN_WORKFLOWS = os.getenv("AUTO_SCAN_WORKFLOWS", "false").lower() == "true"
AUTO_LOAD_WORKFLOW = os.getenv("AUTO_LOAD_WORKFLOW", "")
AUTO_LOAD_SESSION = os.getenv("AUTO_LOAD_SESSION", "")

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
SESSIONS_DIR = PROJECT_DIR / "sessions"
LOCAL_WORKFLOWS_DIR = PROJECT_DIR / "workflows"
LOG_DIR = PROJECT_DIR / "logs"
