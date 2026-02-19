"""Build the agent's system prompt from knowledge files and context."""

from .config import KNOWLEDGE_DIR, COMFYUI_DATABASE, CUSTOM_NODES_DIR, MODELS_DIR

_RULES = """\
RULES:
1. NEVER claim to know about specific models from memory. ALWAYS use tools.
2. When asked "what model should I use for X?" — search first, recommend after.
3. When modifying workflows, ALWAYS show the proposed patch and get confirmation before applying.
4. When something fails, read the error, check node compatibility, suggest fixes.
5. Use extended thinking for: workflow design from scratch, debugging, architecture decisions.
6. Use standard responses for: simple queries, status checks, model listings.
7. If ComfyUI is not running, say so immediately. Most tools require it.
8. Prefer /object_info over memory for node interfaces. It's always current.
9. When suggesting new nodes/models, check if they're already installed first.
10. Log key decisions and preferences to session notes (add_note) for continuity.
11. At conversation end, save the session so the user can resume later.
12. Use format='names_only' or format='summary' for large tool queries; drill down with specific tools.
13. For workflow creation, prefer loading a template (list_workflow_templates) and patching it.
14. Before executing, use validate_before_execute to catch errors early.
15. Use add_node/connect_nodes/set_input for building workflows instead of raw JSON patches when possible.
16. When past outcomes exist, proactively mention relevant patterns without overwhelming.

TOOL OVERVIEW:

Inspection (live API):
  is_comfyui_running, get_all_nodes, get_node_info, get_system_stats,
  get_queue_status, get_history

Filesystem:
  list_custom_nodes, list_models, get_models_summary, read_node_source

Workflow Understanding:
  load_workflow, validate_workflow, get_editable_fields

Workflow Editing (RFC6902 JSON Patch):
  apply_workflow_patch, preview_workflow_patch, undo_workflow_patch,
  get_workflow_diff, save_workflow, reset_workflow

Semantic Workflow Building:
  add_node, connect_nodes, set_input

Execution:
  validate_before_execute, execute_workflow, get_execution_status

Discovery:
  discover (unified search across registry, CivitAI, HuggingFace for nodes and models),
  find_missing_nodes (dependency analysis with install suggestions),
  check_node_updates (GitHub release tracking for installed packs),
  get_repo_releases (release history for a specific GitHub repo)

Templates:
  list_workflow_templates, get_workflow_template

Session Memory:
  save_session, load_session, list_sessions, add_note
"""

# Keyword -> knowledge file mapping for dynamic loading
_KNOWLEDGE_TRIGGERS = {
    "controlnet_patterns": [
        "controlnet", "control_net", "depth_map", "canny", "openpose",
        "lineart", "scribble", "ControlNetLoader", "ControlNetApply",
    ],
    "video_workflows": [
        "animatediff", "animate", "video", "svd", "stable_video",
        "frame", "motion", "AnimateDiffLoader", "VHS_",
    ],
    "flux_specifics": [
        "flux", "FLUX", "FluxGuidance", "FluxLoader",
    ],
    "common_recipes": [
        "create workflow", "build workflow", "from scratch",
        "new workflow", "make a workflow", "set up a pipeline",
    ],
    "3d_workflows": [
        "3d", "3D", "mesh", "hunyuan3d", "Hunyuan3D", "triplane",
        "gaussian", "splat", "3dgs", "3DGS", "glb", "GLB", "ply",
        "point_cloud", "Hunyuan3DLoader", "SaveGLB", "Preview3D",
        "Load3DGaussian", "MeshDecoder", "3d_gen",
        "3d generation", "Wan2.1", "wan2", "Wan-T2V", "Wan-I2V",
        "vnccs", "VNCCS", "action director", "ActionDirector",
        "pose", "camera angle", "3d viewport", "3dview",
        "depth map from 3d", "controlnet 3d",
    ],
    "3d_partner_nodes": [
        "mesh generation", "3d model", "generate 3d", "best 3d",
        "hunyuan3d", "meshy", "tripo", "rodin", "trellis",
        "partner node", "partner", "which 3d", "compare 3d",
        "splat to mesh", "gaussian to mesh", "convert splat",
        "export 3d", "NeRF to mesh", "get GLB", "marching cubes",
    ],
    "audio_workflows": [
        "tts", "TTS", "text to speech", "audio", "voice",
        "cosyvoice", "CosyVoice", "CosyVoiceLoader",
        "qwen audio", "QwenAudio", "bark", "tortoise",
        "xtts", "XTTS", "fish speech", "chattts", "ChatTTS",
        "SaveAudio", "LoadAudio", "AudioSave",
        "text-to-speech", "narration", "speech synthesis",
    ],
}


def _detect_relevant_knowledge(session_context: dict | None) -> set[str]:
    """Detect which optional knowledge files to load based on context."""
    triggers = set()

    if not session_context:
        return triggers

    # Check workflow node types
    wf = session_context.get("workflow", {})
    workflow_text = ""
    if wf.get("current_workflow") and isinstance(wf["current_workflow"], dict):
        # He2025: sort to ensure deterministic keyword detection
        class_types = sorted([
            n.get("class_type", "")
            for n in wf["current_workflow"].values()
            if isinstance(n, dict)
        ])
        workflow_text = " ".join(class_types)

    # Check notes
    notes_text = " ".join(
        n.get("text", "") if isinstance(n, dict) else str(n)
        for n in session_context.get("notes", [])
    )

    combined = f"{workflow_text} {notes_text}".lower()

    for knowledge_file, keywords in _KNOWLEDGE_TRIGGERS.items():
        if any(kw.lower() in combined for kw in keywords):
            triggers.add(knowledge_file)

    return triggers


def build_system_prompt(session_context: dict | None = None) -> str:
    """Assemble the full system prompt from knowledge files, session context, and rules.

    Args:
        session_context: Optional session data dict with 'notes', 'workflow', 'name' keys.
                        When provided, injects session memory and loads relevant knowledge.
    """
    parts = [
        "You are a ComfyUI co-pilot — an expert assistant that helps artists "
        "inspect, understand, discover, modify, and execute ComfyUI workflows "
        "through natural conversation.\n",
        "You have tools to query the live ComfyUI API, scan the local filesystem, "
        "search for custom node packs and models in the ComfyUI Manager registry, "
        "and search HuggingFace Hub for broader model discovery.\n",
        f"ComfyUI installation: {COMFYUI_DATABASE}\n"
        f"Custom nodes: {CUSTOM_NODES_DIR}\n"
        f"Models: {MODELS_DIR}\n",
    ]

    # Inject session context (privileged position — before knowledge)
    if session_context:
        parts.append("\n--- Session Context ---")
        session_name = session_context.get("name", "")
        if session_name:
            parts.append(f"Active session: {session_name}")

        # Loaded workflow summary
        wf = session_context.get("workflow", {})
        if wf.get("loaded_path"):
            parts.append(f"Loaded workflow: {wf['loaded_path']} (format: {wf.get('format', '?')})")
            if wf.get("history_depth", 0) > 0:
                parts.append(f"  Patches applied: {wf['history_depth']}")

        # Session notes (most recent 10)
        notes = session_context.get("notes", [])
        if notes:
            parts.append("User preferences and notes from previous sessions:")
            for note in notes[-10:]:
                text = note.get("text", "") if isinstance(note, dict) else str(note)
                parts.append(f"  - {text}")
        parts.append("")

    # Proactive recommendations from memory (if outcomes exist)
    if session_context and session_context.get("name"):
        try:
            from .brain.memory import handle as memory_handle
            import json as _json
            recs_raw = memory_handle("get_recommendations", {
                "session": session_context["name"],
            })
            recs = _json.loads(recs_raw)
            top_recs = [
                r for r in recs.get("recommendations", [])
                if r.get("confidence", 0) >= 0.7
            ][:3]
            if top_recs:
                parts.append("\n--- Recommendations from Past Sessions ---")
                for rec in top_recs:
                    parts.append(
                        f"  - [{rec.get('category', '?')}] "
                        f"{rec.get('recommendation', '')}"
                    )
                parts.append("")
        except Exception:
            pass  # Memory unavailable — skip silently

    # Load core knowledge files (always)
    relevant_extras = _detect_relevant_knowledge(session_context)
    if KNOWLEDGE_DIR.exists():
        for md_file in sorted(KNOWLEDGE_DIR.glob("*.md")):
            stem = md_file.stem
            # Always load core knowledge; load extras only if relevant
            is_core = stem == "comfyui_core"
            is_relevant_extra = stem in relevant_extras
            if is_core or is_relevant_extra:
                try:
                    content = md_file.read_text(encoding="utf-8")
                    parts.append(f"\n--- {stem} ---\n{content}\n")
                except Exception:
                    pass

    parts.append(f"\n{_RULES}")

    return "\n".join(parts)
