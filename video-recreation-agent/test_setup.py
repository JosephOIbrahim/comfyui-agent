"""
Quick Setup Test — Run this to verify everything is connected.

Usage:
    python test_setup.py

Checks:
    1. ffmpeg is available
    2. ComfyUI is running and reachable
    3. Required node types are installed
    4. Workspace directories exist
    5. Agent team files are in place
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Load bridge.env if it exists
env_path = Path("config/bridge.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


def check(name, ok, detail=""):
    status = "✓" if ok else "✗"
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    print("=" * 56)
    print("  Video Recreation Agent — Setup Check")
    print("=" * 56)
    all_ok = True

    # ── 1. ffmpeg ──────────────────────────────────────
    print("\n1. Tools")
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    all_ok &= check("ffmpeg", ffmpeg_ok,
                     "installed" if ffmpeg_ok else "NOT FOUND — install ffmpeg")

    ffprobe_ok = shutil.which("ffprobe") is not None
    all_ok &= check("ffprobe", ffprobe_ok,
                     "installed" if ffprobe_ok else "NOT FOUND — comes with ffmpeg")

    python_ok = sys.version_info >= (3, 10)
    all_ok &= check("Python 3.10+", python_ok,
                     f"Python {sys.version_info.major}.{sys.version_info.minor}")

    # ── 2. ComfyUI Connection ─────────────────────────
    print("\n2. ComfyUI Connection")
    try:
        from agent.tools.comfyui_bridge import ComfyUIBridge
        bridge = ComfyUIBridge()
        connected = bridge.is_connected()
        all_ok &= check("ComfyUI reachable", connected,
                         f"at {bridge.base_url}" if connected else
                         f"NOT running at {bridge.base_url}")

        if connected:
            stats = bridge.get_system_stats()
            devices = stats.get("devices", [{}])
            if devices:
                d = devices[0]
                vram = d.get("vram_total", 0) / (1024 ** 3)
                check("GPU detected", True, f"{d.get('name', '?')} ({vram:.0f}GB)")

            # Check node types
            schemas = bridge.get_node_schemas()
            check("Node schemas loaded", len(schemas) > 0, f"{len(schemas)} types")

            # Check for video generation nodes
            kling = bridge.find_nodes_by_category("kling")
            check("Kling nodes", len(kling) > 0,
                  f"{len(kling)} found" if kling else "NOT installed — see Step 5")

            # Check for models
            ckpts = bridge.list_models("checkpoints")
            check("Checkpoints", len(ckpts) > 0, f"{len(ckpts)} installed")

    except ImportError:
        all_ok &= check("Bridge module", False,
                         "can't import — make sure you're in the agent repo")
    except Exception as e:
        all_ok &= check("ComfyUI reachable", False, str(e))

    # ── 3. Workspace ──────────────────────────────────
    print("\n3. Workspace")
    workspace = Path(os.environ.get("WORKSPACE", "workspace"))
    dirs = ["reference", "keyframes", "workflows", "outputs", "qa", "qa/frames"]
    for d in dirs:
        p = workspace / d
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        check(f"workspace/{d}/", p.exists(), "created" if p.exists() else "FAILED")

    # ── 4. Agent Team Files ───────────────────────────
    print("\n4. Agent Team Files")
    files = {
        "CLAUDE.md": "Orchestrator",
        ".claude/commands/analyze-video.md": "Analyst expert",
        ".claude/commands/build-workflow.md": "Architect expert",
        ".claude/commands/generate-shots.md": "Generator expert",
        ".claude/commands/assemble-montage.md": "Editor expert",
        ".claude/commands/qa-compare.md": "QA expert",
        "agent/tools/comfyui_bridge.py": "Bridge module",
        "config/bridge.env": "Config",
    }
    for filepath, desc in files.items():
        exists = Path(filepath).exists()
        all_ok &= check(filepath, exists,
                         desc if exists else f"MISSING — copy from package")

    # ── Summary ───────────────────────────────────────
    print("\n" + "=" * 56)
    if all_ok:
        print("  ALL CHECKS PASSED — Ready to recreate videos!")
        print("  Try: /project:analyze-video [video_url]")
    else:
        print("  SOME CHECKS FAILED — Fix the issues above")
        print("  See SETUP_GUIDE.md for detailed instructions")
    print("=" * 56)


if __name__ == "__main__":
    main()
