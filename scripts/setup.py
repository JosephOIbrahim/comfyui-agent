#!/usr/bin/env python3
"""Interactive setup for new users.

Run once after cloning:
    python scripts/setup.py
"""

import os
import sys
from pathlib import Path

# Force UTF-8 on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
BAT_FILE = PROJECT_ROOT / "scripts" / "comfyui_with_agent.bat"


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"  {prompt}{suffix}: ").strip()
    return value if value else default


def main():
    print()
    print("  ╔═══════════════════════════════════════════╗")
    print("  ║   ComfyUI Agent — First-Time Setup        ║")
    print("  ╚═══════════════════════════════════════════╝")
    print()

    # 1. API key
    print("  1. Anthropic API Key")
    print("     Get one at: https://console.anthropic.com/")
    api_key = ask("Paste your key (sk-ant-...)")
    if not api_key:
        print("\n  [!] No API key provided. You can add it to .env later.")
        api_key = "sk-ant-PASTE-YOUR-KEY-HERE"
    print()

    # 2. ComfyUI database path
    print("  2. ComfyUI Database Path")
    print("     This is where your models/, Custom_Nodes/, and output/ folders live.")
    print("     Common locations:")
    print("       - G:/COMFYUI_Database")
    print("       - C:/ComfyUI")
    print("       - ~/ComfyUI (Linux/Mac)")

    default_db = "G:/COMFYUI_Database"
    comfyui_db = ask("Path to your ComfyUI database", default_db)

    # Validate
    db_path = Path(comfyui_db)
    if db_path.exists():
        models = db_path / "models"
        nodes = db_path / "Custom_Nodes"
        if models.exists() and nodes.exists():
            print(f"     [+] Found models/ and Custom_Nodes/")
        elif models.exists():
            print(f"     [+] Found models/ (Custom_Nodes/ not found -- that's OK if you don't have custom nodes yet)")
        else:
            print(f"     [!] Path exists but models/ not found. Double-check this is your ComfyUI folder.")
    else:
        print(f"     [!] Path doesn't exist yet. Make sure it's correct before running the agent.")
    print()

    # 3. ComfyUI startup script (for Windows bat)
    comfyui_bat = ""
    if sys.platform == "win32":
        print("  3. ComfyUI Startup Script (optional)")
        print("     If you have a .bat file that starts ComfyUI, the launcher can use it.")
        comfyui_bat = ask("Path to your ComfyUI .bat file (or press Enter to skip)")
        print()

    # 4. Output dir (if different from database)
    print("  4. ComfyUI Output Directory (optional)")
    print("     Only set this if your output/ folder is NOT inside your database path.")
    output_dir = ask("Output directory (or press Enter for default)")
    print()

    # 5. Model choice
    print("  5. Claude Model for CLI mode")
    print("     Sonnet = faster + cheaper. Opus = higher quality.")
    print("     This only affects 'agent run' (CLI mode). Claude Code picks its own model.")
    model = ask("Model", "claude-sonnet-4-20250514")
    print()

    # --- Write .env ---
    lines = [
        f"ANTHROPIC_API_KEY={api_key}",
        f"COMFYUI_DATABASE={comfyui_db}",
    ]
    if output_dir:
        lines.append(f"COMFYUI_OUTPUT_DIR={output_dir}")
    if model != "claude-sonnet-4-20250514":
        lines.append(f"AGENT_MODEL={model}")

    lines.append("")
    lines.append("# Defaults (uncomment to override):")
    lines.append("# COMFYUI_HOST=127.0.0.1")
    lines.append("# COMFYUI_PORT=8188")

    ENV_FILE.write_text("\n".join(lines) + "\n")
    print(f"  [+] Created {ENV_FILE}")

    # --- Update startup script paths ---
    if comfyui_bat and BAT_FILE.exists():
        bat_content = BAT_FILE.read_text()
        bat_content = bat_content.replace(
            "set COMFYUI_BAT=G:\\COMFY\\ComfyUI\\comfyui_zen.bat",
            f"set COMFYUI_BAT={comfyui_bat}",
        )
        bat_content = bat_content.replace(
            "set AGENT_DIR=C:\\Users\\User\\comfyui-agent",
            f"set AGENT_DIR={PROJECT_ROOT}",
        )
        BAT_FILE.write_text(bat_content)
        print(f"  [+] Updated {BAT_FILE}")

    print()
    print("  ╔═══════════════════════════════════════════╗")
    print("  ║   Setup complete!                          ║")
    print("  ╚═══════════════════════════════════════════╝")
    print()
    print("  Next steps:")
    print("    1. Start ComfyUI")
    print("    2. Run: agent run")
    print("    3. Or use with Claude Code (MCP auto-configured)")
    print()
    if comfyui_bat:
        print(f"  Or double-click: scripts/comfyui_with_agent.bat")
        print()


if __name__ == "__main__":
    main()
