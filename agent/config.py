"""Configuration and environment handling."""

import os
import platform
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
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-opus-4-6-20250929")
MAX_TOKENS = 16384
MAX_AGENT_TURNS = 30

# Context management
COMPACT_THRESHOLD = 120_000  # tokens — start compacting at this level

# API resilience
API_MAX_RETRIES = 3
API_RETRY_DELAY = 1.0  # seconds — base delay, doubles each retry

# ComfyUI connection
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

# Paths — cross-platform defaults for ComfyUI database location
def _default_comfyui_database() -> str:
    """Sensible default ComfyUI database path per platform."""
    _sys = platform.system()
    if _sys == "Windows":
        return "G:/COMFYUI_Database"
    elif _sys == "Darwin":
        return str(Path.home() / "ComfyUI")
    else:
        return str(Path.home() / "ComfyUI")


COMFYUI_DATABASE = Path(os.getenv("COMFYUI_DATABASE", _default_comfyui_database()))
CUSTOM_NODES_DIR = COMFYUI_DATABASE / "Custom_Nodes"
MODELS_DIR = COMFYUI_DATABASE / "models"
WORKFLOWS_DIR = COMFYUI_DATABASE / "Workflows"

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
SESSIONS_DIR = PROJECT_DIR / "sessions"
LOCAL_WORKFLOWS_DIR = PROJECT_DIR / "workflows"
LOG_DIR = PROJECT_DIR / "logs"
