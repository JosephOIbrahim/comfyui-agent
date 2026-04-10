"""Build the agent's system prompt from knowledge files and context."""

import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)

from .config import (
    KNOWLEDGE_DIR, COMFYUI_DATABASE, CUSTOM_NODES_DIR, MODELS_DIR,
    COMFYUI_INSTALL_DIR, COMFYUI_BLUEPRINTS_DIR, WORKFLOWS_DIR,
)

_RULES = """\
RULES:
1. NEVER claim to know about specific models from memory. ALWAYS use tools.
2. When asked "what model should I use for X?" -- search first, recommend after.
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
15. Use add_node/connect_nodes/set_input for building workflows instead of raw patches when possible.
16. When past outcomes exist, proactively mention relevant patterns without overwhelming.

TOOL OVERVIEW:

Inspection (live API):
  is_comfyui_running, get_all_nodes, get_node_info, get_system_stats,
  get_queue_status, get_history

Filesystem:
  list_custom_nodes, list_models, get_models_summary, read_node_source

Workflow Understanding:
  load_workflow, validate_workflow, get_editable_fields

Workflow Editing:
  apply_workflow_patch, preview_workflow_patch, undo_workflow_patch,
  get_workflow_diff, save_workflow, reset_workflow

Semantic Workflow Building:
  add_node, connect_nodes, set_input

Execution:
  validate_before_execute, execute_workflow, get_execution_status

Discovery:
  discover (unified search across local catalog, registry, CivitAI, HuggingFace for nodes and models),
  find_missing_nodes (dependency analysis with install suggestions),
  check_node_updates (GitHub release tracking for installed packs),
  get_repo_releases (release history for a specific GitHub repo)

Provisioning (install & download):
  install_node_pack (git clone a custom node pack into Custom_Nodes),
  download_model (download a model file to the correct models subdirectory),
  uninstall_node_pack (disable a node pack non-destructively)

Templates & Workflows:
  list_workflow_templates (built-in + user workflows + ComfyUI blueprints),
  get_workflow_template (load any template/workflow/blueprint for editing)

Session Memory:
  save_session, load_session, list_sessions, add_note
"""

# Keyword -> knowledge file mapping loaded from YAML
_TRIGGERS_PATH = Path(__file__).parent / "knowledge" / "triggers.yaml"
_triggers_lock = threading.Lock()
_triggers_cache: dict | None = None


def _load_triggers() -> dict:
    """Load knowledge triggers from YAML. Cached after first load (thread-safe)."""
    global _triggers_cache
    if _triggers_cache is not None:
        return _triggers_cache
    with _triggers_lock:
        if _triggers_cache is not None:
            return _triggers_cache
        if _TRIGGERS_PATH.exists():
            import yaml
            try:
                with open(_TRIGGERS_PATH, encoding="utf-8") as f:
                    _triggers_cache = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "triggers.yaml contains invalid YAML — knowledge triggers disabled: %s", exc
                )
                _triggers_cache = {}
        else:
            _triggers_cache = {}
        return _triggers_cache


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

    trigger_defs = _load_triggers()
    for knowledge_file, config in trigger_defs.items():
        keywords = config.get("keywords", [])
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
        "You are a ComfyUI co-pilot -- an expert assistant that helps artists "
        "inspect, understand, discover, modify, and execute ComfyUI workflows "
        "through natural conversation.\n",
        "You have tools to query the live ComfyUI API, scan the local filesystem, "
        "search for custom node packs and models in the ComfyUI Manager registry, "
        "and search HuggingFace Hub for broader model discovery.\n",
        f"ComfyUI installation: {COMFYUI_INSTALL_DIR}\n"
        f"ComfyUI database: {COMFYUI_DATABASE}\n"
        f"Custom nodes: {CUSTOM_NODES_DIR}\n"
        f"Models: {MODELS_DIR}\n"
        f"User workflows: {WORKFLOWS_DIR}\n"
        f"ComfyUI blueprints: {COMFYUI_BLUEPRINTS_DIR}\n",
    ]

    # Inject session context (privileged position -- before knowledge)
    if session_context:
        parts.append("\n--- Session Context ---")
        session_name = session_context.get("name", "")
        if session_name:
            parts.append(f"Active session: {session_name}")

        # Loaded workflow summary
        wf = session_context.get("workflow", {})
        if wf.get("loaded_path"):
            parts.append(
                f"Loaded workflow: {wf['loaded_path']} "
                f"(format: {wf.get('format', '?')})"
            )
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
            from .brain.memory import MemoryAgent
            import json as _json
            _mem = MemoryAgent()
            recs_raw = _mem.handle("get_recommendations", {
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
            pass  # Memory unavailable -- skip silently

    # Auto-read creative metadata from last output (if available)
    if session_context and session_context.get("last_output_path"):
        try:
            from .tools import handle as _tools_handle
            import json as _json
            meta_raw = _tools_handle("reconstruct_context", {
                "image_path": session_context["last_output_path"],
            })
            meta = _json.loads(meta_raw)
            if meta.get("has_context"):
                parts.append("\n--- Last Output Context ---")
                parts.append(meta.get("summary", ""))
                ctx = meta.get("context", {})
                if ctx.get("intent"):
                    parts.append(
                        f"  Artist wanted: {ctx['intent'].get('what_artist_wanted', '')}"
                    )
                    parts.append(
                        f"  Interpretation: {ctx['intent'].get('how_agent_interpreted', '')}"
                    )
                if ctx.get("session", {}).get("key_params"):
                    kp = ctx["session"]["key_params"]
                    parts.append(
                        f"  Last params: {', '.join(f'{k}={v}' for k, v in sorted(kp.items()))}"
                    )
                parts.append("")
        except Exception:
            pass  # Metadata unavailable -- skip silently

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
                except Exception as _e:  # Cycle 62: log instead of silently swallow
                    log.debug("Could not read knowledge file %s: %s", md_file.name, _e)

    parts.append(f"\n{_RULES}")

    return "\n".join(parts)
