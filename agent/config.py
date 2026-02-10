"""Configuration and environment handling."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
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
COMFYUI_WS_URL = f"ws://{COMFYUI_HOST}:{COMFYUI_PORT}"

# Paths — note: Custom_Nodes has capital C and N on this installation
COMFYUI_DATABASE = Path(os.getenv("COMFYUI_DATABASE", "G:/COMFYUI_Database"))
CUSTOM_NODES_DIR = COMFYUI_DATABASE / "Custom_Nodes"
MODELS_DIR = COMFYUI_DATABASE / "models"
WORKFLOWS_DIR = COMFYUI_DATABASE / "Workflows"

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
SESSIONS_DIR = PROJECT_DIR / "sessions"
LOCAL_WORKFLOWS_DIR = PROJECT_DIR / "workflows"
