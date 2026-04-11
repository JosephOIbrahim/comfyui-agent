"""COMFY COZY UI -- aiohttp routes mounted on ComfyUI's PromptServer.

Provides:
  POST /superduper/chat   -- synchronous chat (request/response)
  GET  /superduper/status  -- agent and connection state
  GET  /superduper/ws      -- WebSocket for real-time streaming
"""

import asyncio
import json
import logging
import queue
import sys
import threading
import uuid
from pathlib import Path

from aiohttp import web

log = logging.getLogger("superduper.routes")

# ---------------------------------------------------------------------------
# Agent brain import (lazy — ComfyUI loads us before agent is on sys.path)
# ---------------------------------------------------------------------------

_brain_lock = threading.Lock()
_brain_ready = False
_client = None


def _ensure_brain():
    """Lazily import the agent brain and create the Anthropic client."""
    global _brain_ready, _client
    if _brain_ready:
        return True

    with _brain_lock:
        if _brain_ready:
            return True

        try:
            # Add project root to path so `agent` package is importable
            project_root = str(Path(__file__).resolve().parent.parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from agent.main import create_client
            _client = create_client()
            _brain_ready = True
            log.info("Agent brain loaded, Anthropic client ready")
            return True
        except Exception as e:
            log.error("Failed to load agent brain: %s", e)
            return False


# ---------------------------------------------------------------------------
# Session + correlation propagation
# ---------------------------------------------------------------------------
# The sidebar previously trampled MCP's "default" workflow slot because no
# code path set agent._conn_ctx._conn_session before invoking tools.  These
# helpers spawn worker threads / executor tasks with:
#   1. _conn_session ContextVar set → workflow_patch._get_state() picks the
#      right session (isolation between sidebar tabs and MCP connections).
#   2. set_correlation_id() called → all log entries from this conversation
#      get tagged with the conv.id, so a single chat is greppable end-to-end.
#      threading.local() is per-thread, so the corr ID must be set inside
#      the worker — not the parent — to take effect.

def _spawn_with_session(target, args, session_id: str, *, daemon: bool = True):
    """threading.Thread wrapper that sets _conn_session + correlation ID."""
    import contextvars
    from agent._conn_ctx import _conn_session
    from agent.logging_config import set_correlation_id

    def runner():
        _conn_session.set(session_id)
        set_correlation_id(session_id)
        return target(*args)

    ctx = contextvars.copy_context()
    return threading.Thread(
        target=ctx.run,
        args=(runner,),
        daemon=daemon,
    )


def _run_in_executor_with_session(loop, target, *args, session_id: str):
    """run_in_executor wrapper that sets _conn_session + correlation ID."""
    import contextvars
    from agent._conn_ctx import _conn_session
    from agent.logging_config import set_correlation_id

    def runner():
        _conn_session.set(session_id)
        set_correlation_id(session_id)
        return target(*args)

    ctx = contextvars.copy_context()
    return loop.run_in_executor(None, lambda: ctx.run(runner))


# ---------------------------------------------------------------------------
# Per-connection conversation state
# ---------------------------------------------------------------------------

class ConversationState:
    """Tracks one sidebar conversation."""

    def __init__(self):
        self.id = str(uuid.uuid4())[:8]
        self.messages: list[dict] = []
        self.system_prompt: str | None = None
        self.busy = False
        self.lock = threading.Lock()
        self.workflow_summary: dict | None = None
        self.missing_nodes: list[str] | None = None
        self._workflow_hash: int | None = None  # skip reload if unchanged
        self._first_workflow = True  # run missing-nodes check on first inject
        self._missing_nodes_full: list[dict] | None = None  # full data for panel

    def _build_system(self):
        from agent.system_prompt import build_system_prompt
        base = build_system_prompt()

        # Append workflow context if available
        extras = []
        if self.workflow_summary:
            s = self.workflow_summary
            extras.append(
                "\n\n## Current Workflow (auto-injected from ComfyUI canvas)\n\n"
                f"Format: {s['format']} | Nodes: {s['node_count']} | "
                f"Connections: {s['connection_count']}\n\n"
                f"{s['summary']}\n\n"
                "The workflow is already loaded in the PILOT engine. "
                "Use PILOT tools (add_node, set_input, connect_nodes, "
                "apply_workflow_patch) directly — do NOT ask for a file path."
            )
        if self.missing_nodes:
            extras.append(
                "\n\n**Missing nodes detected:** "
                + ", ".join(self.missing_nodes)
                + "\nCall repair_workflow(auto_install=true) to fix these automatically before executing."
            )

        self.system_prompt = base + "".join(extras)


# Active conversations keyed by connection ID
_conversations: dict[str, ConversationState] = {}


# ---------------------------------------------------------------------------
# Workflow injection (sidebar -> agent state)
# ---------------------------------------------------------------------------

def _inject_workflow(conv: ConversationState, workflow_data: dict) -> None:
    """Inject live workflow data into agent state and conversation context.

    Skips reload if the workflow hasn't changed (preserves undo history).
    Runs find_missing_nodes on first injection (best-effort).
    """
    # Quick change detection via hash of sorted JSON
    from agent.tools._util import to_json as _to_json
    wf_hash = hash(_to_json(workflow_data))

    if wf_hash == conv._workflow_hash:
        # Workflow unchanged — rebuild system prompt (keeps context fresh)
        # but don't reload (preserves undo history)
        conv._build_system()
        return

    conv._workflow_hash = wf_hash

    # Load into PILOT engine
    from agent.tools.workflow_patch import load_workflow_from_data
    err = load_workflow_from_data(workflow_data, source="<sidebar>")
    if err:
        log.warning("Workflow injection failed: %s", err)
        return

    # Summarize for system prompt
    from agent.tools.workflow_parse import summarize_workflow_data
    conv.workflow_summary = summarize_workflow_data(workflow_data)

    # On first workflow injection, check for missing nodes (best-effort)
    # Store full missing data for panel emission (not just class_types)
    if conv._first_workflow:
        conv._first_workflow = False
        try:
            from agent.tools.comfy_discover import handle as discover_handle
            result_json = discover_handle("find_missing_nodes", {})
            import json as _json
            result = _json.loads(result_json)
            missing = result.get("missing", [])
            if missing:
                conv.missing_nodes = [m.get("class_type", "?") for m in missing]
                # Store full data for panel emission
                conv._missing_nodes_full = missing
            else:
                conv.missing_nodes = None
                conv._missing_nodes_full = None
        except Exception as e:
            log.debug("Missing nodes check skipped: %s", e)

    # Force system prompt rebuild with new workflow context
    conv._build_system()


# ---------------------------------------------------------------------------
# Panel builders (tool result -> structured panel data for frontend)
# ---------------------------------------------------------------------------

# Ordered slot classification rules (checked first to last; first match wins).
# More specific patterns before generic ones to avoid false matches.
_SLOT_RULES = [
    ("controlnet", ["controlnet", "control_net", "preprocessor", "canny", "openpose"]),
    ("vae",        ["vae"]),
    ("clip",       ["clip", "textencode", "t5"]),
    ("latent",     ["latent", "emptylatent"]),
    ("sampler",    ["sampler", "ksampler"]),
    ("conditioning", ["conditioning", "prompt"]),
    ("sigmas",     ["sigmas", "scheduler"]),
    ("guider",     ["guider"]),
    ("noise",      ["randomnoise"]),
    ("mask",       ["mask"]),
    ("image",      ["image", "preview", "save", "video", "combine"]),
    ("model",      ["checkpoint", "model", "unet", "lora", "loader"]),
]


def _classify_slot(class_type: str) -> str:
    """Keyword-based class_type -> slot type string."""
    ct_lower = class_type.lower()
    for slot_type, keywords in _SLOT_RULES:
        for kw in keywords:
            if kw in ct_lower:
                return slot_type
    return "model"  # fallback


def _extract_flow_chain(nodes: dict, max_nodes: int = 8) -> list[dict]:
    """BFS from root nodes to build a signal flow chain.

    Args:
        nodes: API format {node_id: {class_type, inputs}}
        max_nodes: Maximum nodes in chain

    Returns:
        List of {label, slotType} dicts
    """
    if not nodes:
        return []

    # Find nodes that receive input from other nodes (have upstream connections)
    has_upstream = set()
    for nid, ndata in nodes.items():
        for val in (ndata.get("inputs") or {}).values():
            if isinstance(val, list) and len(val) == 2:
                # This node (nid) receives input from val[0]
                has_upstream.add(nid)

    # Roots: nodes with no upstream connections (source nodes)
    all_ids = set(nodes.keys())
    roots = all_ids - has_upstream
    if not roots:
        roots = {sorted(all_ids)[0]} if all_ids else set()

    # BFS from roots
    visited = []
    queue_bfs = list(sorted(roots))
    seen = set()

    while queue_bfs and len(visited) < max_nodes:
        nid = queue_bfs.pop(0)
        if nid in seen:
            continue
        seen.add(nid)

        ndata = nodes.get(nid)
        if not ndata:
            continue

        ct = ndata.get("class_type", "?")
        visited.append({"label": ct, "slotType": _classify_slot(ct)})

        # Find downstream nodes (nodes that reference this nid)
        for other_id, other_data in nodes.items():
            if other_id in seen:
                continue
            for val in (other_data.get("inputs") or {}).values():
                if isinstance(val, list) and len(val) == 2 and str(val[0]) == nid:
                    queue_bfs.append(other_id)
                    break

    return visited


# Slot type -> hex color for panel dot colors
_SLOT_HEX = {
    "clip": "#FFD500", "clip_vision": "#A8DADC", "conditioning": "#FFA931",
    "controlnet": "#6EE7B7", "image": "#64B5F6", "latent": "#FF9CF9",
    "mask": "#81C784", "model": "#B39DDB", "style_model": "#C2FFAE",
    "vae": "#FF6E6E", "noise": "#B0B0B0", "guider": "#66FFFF",
    "sampler": "#ECB4B4", "sigmas": "#CDFFCD",
}


def _panel_workflow_analysis(result: dict) -> dict | None:
    """Build panel data from load_workflow / validate_workflow result."""
    # Extract nodes from result
    nodes = result.get("nodes") or result.get("workflow") or {}
    if isinstance(nodes, str):
        return None

    # If result has API-format workflow data embedded
    if not nodes and "node_count" in result:
        # Minimal header-only panel
        return {
            "type": "workflow_analysis",
            "header": {
                "label": "workflow \u00b7 analysis",
                "badge": result.get("format", "workflow"),
                "title": result.get("loaded_path", "Workflow loaded").split("/")[-1].split("\\")[-1],
                "summary": f"{result.get('node_count', '?')} nodes, "
                           f"{result.get('connection_count', '?')} connections. "
                           f"Format: {result.get('format', 'unknown')}.",
                "stats": [
                    {"value": str(result.get("node_count", "?")), "label": "nodes"},
                    {"value": str(result.get("connection_count", "?")), "label": "connections"},
                ],
            },
            "sections": [],
            "footer": {"status": "loaded", "actions": [
                {"label": "Modify", "variant": "secondary"},
                {"label": "Run", "variant": "primary"},
            ]},
        }

    # Full analysis with node classification
    # Group nodes by slot type
    groups = {}
    for nid, ndata in nodes.items():
        ct = ndata.get("class_type", "?")
        slot = _classify_slot(ct)
        groups.setdefault(slot, []).append({"label": ct, "slotType": slot})

    # Count connections
    conn_count = 0
    for ndata in nodes.values():
        for val in (ndata.get("inputs") or {}).values():
            if isinstance(val, list) and len(val) == 2:
                conn_count += 1

    # Build flow chain
    flow = _extract_flow_chain(nodes)

    # Determine workflow type from nodes
    class_types = {n.get("class_type", "").lower() for n in nodes.values()}
    wf_type = "workflow"
    if any("video" in c or "animatediff" in c or "svd" in c for c in class_types):
        wf_type = "video gen"
    elif any("controlnet" in c for c in class_types):
        wf_type = "controlnet"
    elif any("img2img" in c or "inpaint" in c for c in class_types):
        wf_type = "img2img"
    else:
        wf_type = "txt2img"

    sections = []

    # Signal Flow section (open by default)
    if flow:
        sections.append({
            "title": "Signal Flow",
            "dotColor": "#00BB81",
            "count": len(flow),
            "defaultOpen": True,
            "type": "flow_chain",
            "data": {"chains": [{"nodes": flow}]},
        })

    # Node group sections (collapsed)
    # Priority order for display
    group_order = ["model", "clip", "conditioning", "sampler", "vae",
                   "controlnet", "image", "latent", "mask", "noise",
                   "guider", "sigmas"]
    group_labels = {
        "model": "Models & Loaders", "clip": "Text Encoding",
        "conditioning": "Conditioning", "sampler": "Sampling",
        "vae": "VAE", "controlnet": "ControlNet", "image": "Image I/O",
        "latent": "Latent Space", "mask": "Masks", "noise": "Noise",
        "guider": "Guidance", "sigmas": "Noise Schedule",
    }

    for slot in group_order:
        if slot not in groups:
            continue
        tags = groups[slot]
        # Deduplicate tags
        seen = {}
        deduped = []
        for t in tags:
            key = t["label"]
            if key in seen:
                seen[key] += 1
            else:
                seen[key] = 1
                deduped.append(t)
        # Add counts to labels
        final_tags = []
        for t in deduped:
            label = t["label"]
            count = seen[label]
            if count > 1:
                label = f"{label} \u00d7{count}"
            final_tags.append({"label": label, "slotType": t["slotType"]})

        sections.append({
            "title": group_labels.get(slot, slot.title()),
            "dotColor": _SLOT_HEX.get(slot, "#9a9caa"),
            "count": len(tags),
            "defaultOpen": False,
            "type": "slot_tags",
            "data": {"tags": final_tags},
        })

    return {
        "type": "workflow_analysis",
        "header": {
            "label": "workflow \u00b7 analysis",
            "badge": wf_type,
            "title": "Workflow Analysis",
            "summary": f"{len(nodes)} nodes, {conn_count} connections.",
            "stats": [
                {"value": str(len(nodes)), "label": "nodes"},
                {"value": str(conn_count), "label": "connections"},
                {"value": wf_type, "label": "type"},
            ],
        },
        "sections": sections,
        "footer": {"status": "loaded", "actions": [
            {"label": "Modify", "variant": "secondary"},
            {"label": "Run", "variant": "primary"},
        ]},
    }


def _panel_discovery(result: dict, tool_input: dict) -> dict | None:
    """Build panel data from discover tool results."""
    results_list = result.get("results", [])
    if not results_list:
        return None

    query = tool_input.get("query", "search")
    category = tool_input.get("category", "all")

    sections = []
    max_results = 5
    for item in results_list[:max_results]:
        name = item.get("name", item.get("title", "Unknown"))
        source = item.get("source", "")
        installed = item.get("installed", False)

        tags = []
        if item.get("model_type") or item.get("type"):
            t = item.get("model_type") or item.get("type")
            slot = _classify_slot(str(t))
            tags.append({"label": str(t), "slotType": slot})
        if installed:
            tags.append({"label": "installed", "slotType": "controlnet"})

        rows = []
        if source:
            rows.append({"label": "Source", "value": source})
        if item.get("downloads"):
            rows.append({"label": "Downloads", "value": str(item["downloads"])})
        if item.get("rating"):
            rows.append({"label": "Rating", "value": str(item["rating"])})

        section_data = {}
        section_type = "detail_rows"
        if tags:
            section_type = "slot_tags"
            section_data["tags"] = tags
        if rows:
            section_type = "detail_rows"
            section_data["rows"] = rows

        sections.append({
            "title": name,
            "dotColor": "#00BB81",
            "count": None,
            "defaultOpen": len(sections) == 0,  # first result open
            "type": section_type,
            "data": section_data,
        })

    return {
        "type": "discovery",
        "header": {
            "label": f"discovery \u00b7 {category}",
            "badge": str(len(results_list)),
            "title": f'Results for "{query}"',
            "summary": f"Found {len(results_list)} result{'s' if len(results_list) != 1 else ''}. "
                       f"Top result: {results_list[0].get('name', '?')}.",
            "stats": [
                {"value": str(len(results_list)), "label": "found"},
            ],
        },
        "sections": sections,
        "footer": {
            "status": "ready",
            "actions": [{"label": "Details", "variant": "secondary"}],
        },
    }


# ---------------------------------------------------------------------------
# Missing nodes panel
# ---------------------------------------------------------------------------

def _panel_missing_nodes(missing: list[dict]) -> dict:
    """Build a panel showing missing nodes with install actions."""
    sections = []
    install_actions = []

    for m in missing[:10]:
        class_type = m.get("class_type", "?")
        pack_name = m.get("pack_name", "")
        pack_url = m.get("pack_url", "")

        row = {"label": class_type, "value": pack_name or "Unknown pack"}
        sections.append(row)

        if pack_url and pack_url not in [a.get("url") for a in install_actions]:
            install_actions.append({
                "label": f"Install {pack_name or class_type}",
                "variant": "primary",
                "action": "install_node_pack",
                "url": pack_url,
                "name": pack_name,
            })

    actions = install_actions[:3]  # max 3 install buttons
    if not actions:
        actions = [{"label": "Search for nodes", "variant": "secondary",
                     "action": "agent_message",
                     "message": "Find and install the missing custom nodes for my workflow"}]

    return {
        "type": "missing_nodes",
        "header": {
            "label": "workflow \u00b7 repair",
            "badge": str(len(missing)),
            "title": "Missing Nodes Detected",
            "summary": (
                f"{len(missing)} node type{'s' if len(missing) != 1 else ''} "
                f"not found. Install the required packs to fix."
            ),
            "stats": [
                {"value": str(len(missing)), "label": "missing"},
            ],
        },
        "sections": [{
            "title": "Missing Node Types",
            "dotColor": "#FF6E6E",
            "count": len(sections),
            "defaultOpen": True,
            "type": "detail_rows",
            "data": {"rows": sections},
        }],
        "footer": {
            "status": "needs_repair",
            "actions": actions,
        },
    }


# ---------------------------------------------------------------------------
# Install/download result panels
# ---------------------------------------------------------------------------

def _panel_install_result(result: dict) -> dict | None:
    """Panel for install_node_pack or download_model results."""
    if result.get("error"):
        return None

    if result.get("installed"):
        return {
            "type": "install_result",
            "header": {
                "label": "provision \u00b7 install",
                "badge": "OK",
                "title": f"Installed: {result['installed']}",
                "summary": result.get("message", ""),
                "stats": [],
            },
            "sections": [],
            "footer": {
                "status": "restart_required" if result.get("restart_required") else "ready",
                "actions": [{"label": "Restart ComfyUI", "variant": "primary",
                             "action": "agent_message",
                             "message": "How do I restart ComfyUI to load the new nodes?"}],
            },
        }

    if result.get("downloaded"):
        return {
            "type": "download_result",
            "header": {
                "label": "provision \u00b7 download",
                "badge": "OK",
                "title": f"Downloaded: {result['downloaded']}",
                "summary": result.get("message", ""),
                "stats": [
                    {"value": f"{result.get('size_gb', '?')} GB", "label": "size"},
                    {"value": f"{result.get('speed_mbps', '?')} MB/s", "label": "speed"},
                ],
            },
            "sections": [],
            "footer": {"status": "ready", "actions": []},
        }

    return None


# Tools that produce panelable results
_PANEL_TOOLS = {
    "load_workflow", "validate_workflow",
    "discover",
    "find_missing_nodes",
    "install_node_pack", "download_model",
    "repair_workflow", "reconfigure_workflow",
}


def _build_panel_for_tool(name: str, tool_input: dict, result_json: str) -> dict | None:
    """Dispatch tool result to appropriate panel builder. Returns panel dict or None."""
    if name not in _PANEL_TOOLS:
        return None

    try:
        result = json.loads(result_json) if isinstance(result_json, str) else result_json
    except (json.JSONDecodeError, TypeError):
        return None

    if isinstance(result, dict) and result.get("error"):
        return None

    if name in ("load_workflow", "validate_workflow"):
        return _panel_workflow_analysis(result)
    elif name == "discover":
        return _panel_discovery(result, tool_input)
    elif name == "find_missing_nodes":
        missing = result.get("missing", [])
        if missing:
            return _panel_missing_nodes(missing)
        return None
    elif name in ("install_node_pack", "download_model"):
        return _panel_install_result(result)
    elif name == "repair_workflow":
        return _panel_repair_report(result)
    elif name == "reconfigure_workflow":
        return _panel_reconfigure_report(result)

    return None


def _panel_repair_report(result: dict) -> dict | None:
    """Panel for repair_workflow results."""
    if result.get("error"):
        return None
    status = result.get("status", "report")
    sections = []
    install_results = result.get("install_results", [])
    if install_results:
        rows = []
        for ir in install_results:
            icon = "OK" if ir.get("success") else "FAIL"
            rows.append({
                "label": f"[{icon}] {ir.get('pack', '?')}",
                "value": ", ".join(ir.get("nodes_provided", [])[:3]),
            })
        sections.append({
            "title": "Installed Packs",
            "dotColor": "#00BB81" if any(ir["success"] for ir in install_results) else "#FF6E6E",
            "count": len(install_results),
            "defaultOpen": True,
            "type": "detail_rows",
            "data": {"rows": rows},
        })
    unresolved = result.get("unresolved_nodes", [])
    if unresolved:
        sections.append({
            "title": "Unresolved Nodes",
            "dotColor": "#FF6E6E",
            "count": len(unresolved),
            "defaultOpen": True,
            "type": "slot_tags",
            "data": {"tags": [{"label": n, "slotType": "model"} for n in unresolved]},
        })
    return {
        "type": "repair_report",
        "header": {
            "label": "workflow \u00b7 repair",
            "badge": "OK" if status == "repaired" else str(result.get("missing_count", 0)),
            "title": "Workflow Repair Report",
            "summary": result.get("message", ""),
            "stats": [
                {"value": str(result.get("missing_count", 0)), "label": "missing"},
                {"value": str(result.get("packs_installed", 0)), "label": "installed"},
            ],
        },
        "sections": sections,
        "footer": {
            "status": "restart_required" if result.get("restart_required") else "ready",
            "actions": (
                [{"label": "Restart ComfyUI", "variant": "primary",
                  "action": "agent_message",
                  "message": "How do I restart ComfyUI to load the new nodes?"}]
                if result.get("restart_required") else []
            ),
        },
    }


def _panel_reconfigure_report(result: dict) -> dict | None:
    """Panel for reconfigure_workflow results."""
    if result.get("error"):
        return None
    details = result.get("details", {})
    sections = []
    # Fixes applied
    fixes = details.get("fixes", [])
    if fixes:
        rows = [{"label": f"{f['class_type']}.{f['field']}", "value": f"{f['old']} -> {f['new']}"}
                for f in fixes]
        sections.append({
            "title": "Substitutions Applied",
            "dotColor": "#00BB81",
            "count": len(fixes),
            "defaultOpen": True,
            "type": "detail_rows",
            "data": {"rows": rows},
        })
    # Still missing
    missing = details.get("missing", [])
    still_missing = [m for m in missing if not any(
        f["node_id"] == m.get("node_id") and f["field"] == m.get("field") for f in fixes)]  # noqa: E501
    if still_missing:
        rows = [{"label": f"{m['node']}.{m['field']}", "value": m["model"]}
                for m in still_missing]
        sections.append({
            "title": "Models Needing Download",
            "dotColor": "#FF6E6E",
            "count": len(still_missing),
            "defaultOpen": True,
            "type": "detail_rows",
            "data": {"rows": rows},
        })
    return {
        "type": "reconfigure_report",
        "header": {
            "label": "workflow \u00b7 reconfigure",
            "badge": str(result.get("total_references", 0)),
            "title": "Model Reference Report",
            "summary": result.get("message", ""),
            "stats": [
                {"value": str(result.get("found", 0)), "label": "found"},
                {"value": str(result.get("missing", 0)), "label": "missing"},
                {"value": str(result.get("fixes_applied", 0)), "label": "fixed"},
            ],
        },
        "sections": sections,
        "footer": {
            "status": "ready" if not still_missing else "needs_download",
            "actions": (
                [{"label": "Download missing models", "variant": "primary",
                  "action": "agent_message",
                  "message": "Download the missing models for my workflow"}]
                if still_missing else []
            ),
        },
    }


# ---------------------------------------------------------------------------
# Agent dispatch helpers (stage -> agent identity card)
# ---------------------------------------------------------------------------

_STAGE_TO_AGENT = {
    "UNDERSTAND": "router",
    "DISCOVER":   "intent",
    "PILOT":      "execution",
    "VERIFY":     "verify",
    "BRAIN":      "router",
}

_AGENT_ROSTER = [
    {"key": "router",    "name": "ROUTER",    "role": "Pipeline Orchestrator"},
    {"key": "intent",    "name": "INTENT",    "role": "Artistic Translator"},
    {"key": "execution", "name": "EXECUTION", "role": "Workflow Patcher"},
    {"key": "verify",    "name": "VERIFY",    "role": "Quality Judge"},
]


def _emit_agent_dispatch(msg_queue: queue.Queue, prompt_text: str,
                         node_count: int | None = None) -> None:
    """Push an agent_dispatch event to the WebSocket queue."""
    agents = [dict(a, status="waiting") for a in _AGENT_ROSTER]
    msg_queue.put({
        "type": "agent_dispatch",
        "prompt": prompt_text,
        "agents": agents,
        "estimate_seconds": None,
        "node_count": node_count,
    })


def _emit_agent_status(msg_queue: queue.Queue, agent_key: str, status: str,
                        message: str | None = None,
                        duration: str | None = None) -> None:
    """Push an agent_status event to the WebSocket queue."""
    msg_queue.put({
        "type": "agent_status",
        "agent_key": agent_key,
        "status": status,
        "message": message,
        "duration": duration,
    })


# ---------------------------------------------------------------------------
# Synchronous agent runner (called from thread)
# ---------------------------------------------------------------------------

class _QueueStreamHandler:
    """StreamHandler that pushes events to a thread-safe queue.

    Bridges the StreamHandler protocol (expected by run_agent_turn)
    to the event queue consumed by the WebSocket/REST forwarder.
    """

    def __init__(self, msg_queue, conv, active_agents):
        self._q = msg_queue
        self._conv = conv
        self._active = active_agents

    def on_text(self, text):
        self._q.put({"type": "text_delta", "text": text})

    def on_thinking(self, text):
        pass  # not forwarded to frontend

    def on_tool_call(self, name, inp):
        stage = _infer_stage(name)
        agent_key = _STAGE_TO_AGENT.get(stage, "router")

        if agent_key not in self._active:
            for prev in list(self._active):
                _emit_agent_status(self._q, prev, "complete")
            self._active.clear()
            self._active.add(agent_key)
            _emit_agent_status(self._q, agent_key, "active",
                               message=f"Using {name}...")

        nodes = _extract_nodes_touched(name, inp, self._conv)
        self._q.put({
            "type": "tool_call",
            "tool": name,
            "stage": stage,
            "nodes_touched": nodes,
        })

    def on_tool_result(self, name, inp, result_json):
        panel = _build_panel_for_tool(name, inp, result_json)
        if panel:
            self._q.put({"type": "panel", "panel": panel})

    def on_stream_end(self):
        pass

    def on_input(self):
        return None  # non-interactive


def _run_agent_sync(conv: ConversationState, user_text: str, msg_queue: queue.Queue):
    """Run the agent loop synchronously, pushing events to msg_queue.

    Events are dicts: {"type": "text_delta"|"tool_call"|"stage"|"done"|"error", ...}
    """
    from agent.main import run_agent_turn

    conv._build_system()
    conv.messages.append({"role": "user", "content": user_text})

    # Emit dispatch card before first agent turn
    node_count = None
    if conv.workflow_summary:
        node_count = conv.workflow_summary.get("node_count")
    _emit_agent_dispatch(msg_queue, user_text, node_count)

    max_turns = 15  # safety limit per user message
    _active_agents: set[str] = set()  # track which agents are currently active
    handler = _QueueStreamHandler(msg_queue, conv, _active_agents)

    for turn in range(max_turns):
        try:
            conv.messages, done = run_agent_turn(
                _client,
                conv.messages,
                conv.system_prompt,
                handler=handler,
            )

            if done:
                # Complete any remaining active agents
                for a in list(_active_agents):
                    _emit_agent_status(msg_queue, a, "complete")
                _active_agents.clear()
                msg_queue.put({"type": "done"})
                return

        except Exception as e:
            log.error("Agent turn error: %s", e, exc_info=True)
            # Mark active agents as errored
            for a in list(_active_agents):
                _emit_agent_status(msg_queue, a, "error", message=str(e))
            _active_agents.clear()
            msg_queue.put({"type": "error", "message": str(e)})
            return

    # Complete remaining agents on turn limit
    for a in list(_active_agents):
        _emit_agent_status(msg_queue, a, "complete")
    _active_agents.clear()
    msg_queue.put({"type": "done"})


def _infer_stage(tool_name: str) -> str:
    """Map tool name to intelligence layer stage."""
    understand = {
        "load_workflow", "validate_workflow", "get_editable_fields",
        "is_comfyui_running", "get_all_nodes", "get_node_info",
        "get_system_stats", "get_queue_status", "get_history",
        "list_custom_nodes", "list_models", "get_models_summary",
        "read_node_source",
    }
    discover = {
        "discover", "find_missing_nodes", "check_registry_freshness",
        "get_install_instructions", "get_civitai_model", "get_trending_models",
        "list_workflow_templates", "get_workflow_template",
        "check_node_updates", "get_repo_releases",
        "identify_model_family", "check_model_compatibility",
        "install_node_pack", "download_model", "uninstall_node_pack",
        "repair_workflow", "reconfigure_workflow",
    }
    pilot = {
        "apply_workflow_patch", "preview_workflow_patch", "undo_workflow_patch",
        "get_workflow_diff", "save_workflow", "reset_workflow",
        "add_node", "connect_nodes", "set_input",
        "save_session", "load_session", "list_sessions", "add_note",
    }
    verify = {
        "validate_before_execute", "execute_workflow",
        "get_execution_status", "execute_with_progress",
        "get_output_path", "verify_execution",
    }

    if tool_name in understand:
        return "UNDERSTAND"
    elif tool_name in discover:
        return "DISCOVER"
    elif tool_name in pilot:
        return "PILOT"
    elif tool_name in verify:
        return "VERIFY"
    else:
        return "BRAIN"


def _extract_nodes_touched(tool_name: str, tool_input: dict, conv: ConversationState) -> list[dict]:
    """Extract which workflow nodes a tool call is interacting with.

    Returns list of {"node_id": str|None, "class_type": str, "slot_type": str}.
    """
    nodes_touched = []

    # Tools that directly reference nodes by ID or class_type
    _NODE_TOOLS = {
        "set_input": ["node_id"],
        "connect_nodes": ["source_node_id", "target_node_id", "from_node", "to_node"],
        "add_node": ["class_type"],
        "get_node_info": ["class_type", "node_type"],
    }

    # Only process tools that interact with specific nodes
    if tool_name not in _NODE_TOOLS and tool_name not in (
        "apply_workflow_patch", "preview_workflow_patch",
        "load_workflow", "validate_workflow",
        "get_editable_fields", "validate_before_execute",
    ):
        return nodes_touched

    # Get current workflow nodes from PILOT state (best-effort)
    wf_nodes = {}
    try:
        from agent.tools.workflow_patch import _get_state
        wf = _get_state()["current_workflow"]
        if isinstance(wf, dict):
            wf_nodes = wf
    except Exception:
        pass

    # Extract node references from tool input parameters
    node_id_keys = _NODE_TOOLS.get(tool_name, [])
    for key in node_id_keys:
        val = tool_input.get(key)
        if val is None:
            continue

        if key == "class_type":
            nodes_touched.append({
                "node_id": None,
                "class_type": str(val),
                "slot_type": _classify_slot(str(val)),
            })
        else:
            nid = str(val)
            ndata = wf_nodes.get(nid, {})
            ct = ndata.get("class_type", f"Node {nid}")
            nodes_touched.append({
                "node_id": nid,
                "class_type": ct,
                "slot_type": _classify_slot(ct),
            })

    # For patch tools, extract node_ids from RFC6902 paths
    if tool_name in ("apply_workflow_patch", "preview_workflow_patch"):
        patches = tool_input.get("patches") or tool_input.get("patch") or []
        if isinstance(patches, list):
            seen_ids = set()
            for op in patches:
                path = op.get("path", "")
                parts = path.strip("/").split("/")
                if parts and parts[0] and parts[0] not in seen_ids:
                    nid = parts[0]
                    seen_ids.add(nid)
                    ndata = wf_nodes.get(nid, {})
                    ct = ndata.get("class_type", f"Node {nid}")
                    nodes_touched.append({
                        "node_id": nid,
                        "class_type": ct,
                        "slot_type": _classify_slot(ct),
                    })

    return nodes_touched


# ---------------------------------------------------------------------------
# Environment snapshot (sent on WebSocket connect)
# ---------------------------------------------------------------------------

def _build_environment() -> dict:
    """Build a snapshot of the ComfyUI installation for the panel."""
    try:
        from agent.config import (
            COMFYUI_DATABASE, COMFYUI_INSTALL_DIR, MODELS_DIR,
            CUSTOM_NODES_DIR, WORKFLOWS_DIR, COMFYUI_BLUEPRINTS_DIR,
            COMFYUI_OUTPUT_DIR,
        )

        # Count models by type
        model_counts = {}
        if MODELS_DIR.exists():
            for subdir in sorted(MODELS_DIR.iterdir()):
                if subdir.is_dir() and not subdir.name.startswith("."):
                    count = sum(1 for f in subdir.iterdir()
                                if f.is_file() and f.suffix.lower()
                                in (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"))
                    if count > 0:
                        model_counts[subdir.name] = count

        # Count custom nodes
        node_count = 0
        if CUSTOM_NODES_DIR.exists():
            node_count = sum(
                1 for d in CUSTOM_NODES_DIR.iterdir()
                if d.is_dir() and not d.name.startswith((".", "_", "__"))
            )

        # Count workflows
        workflow_count = 0
        if WORKFLOWS_DIR.exists():
            workflow_count = sum(1 for f in WORKFLOWS_DIR.glob("*.json"))

        return {
            "database": str(COMFYUI_DATABASE),
            "install_dir": str(COMFYUI_INSTALL_DIR),
            "models_dir": str(MODELS_DIR.resolve()),
            "custom_nodes_dir": str(CUSTOM_NODES_DIR.resolve()),
            "workflows_dir": str(WORKFLOWS_DIR),
            "blueprints_dir": str(COMFYUI_BLUEPRINTS_DIR),
            "output_dir": str(COMFYUI_OUTPUT_DIR),
            "model_counts": model_counts,
            "total_models": sum(model_counts.values()),
            "node_packs": node_count,
            "workflows": workflow_count,
        }
    except Exception as e:
        log.warning("Environment snapshot failed: %s", e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Direct action handler (executes tools without Claude API round-trip)
# ---------------------------------------------------------------------------

# Actions that execute tools directly (no Claude needed)
_DIRECT_ACTIONS = {
    "install_node_pack", "download_model", "uninstall_node_pack",
    "repair_workflow", "reconfigure_workflow",
    "validate", "repair", "reconfigure",
}


async def _handle_panel_action(ws, conv, loop, action, data):
    """Handle panel action buttons by executing tools directly."""

    if action == "agent_message":
        # This one still routes through Claude (for open-ended questions)
        message = data.get("message", "").strip()
        if not message or conv.busy:
            return
        conv.busy = True
        await ws.send_json({"type": "stage", "stage": "THINKING", "detail": ""})
        msg_queue = queue.Queue()
        thread = _spawn_with_session(
            _run_agent_sync,
            (conv, message, msg_queue),
            session_id=conv.id,
        )
        thread.start()
        accumulated_text = []
        while True:
            try:
                event = await loop.run_in_executor(
                    None, lambda: msg_queue.get(timeout=0.1)
                )
            except queue.Empty:
                if not thread.is_alive():
                    while not msg_queue.empty():
                        event = msg_queue.get_nowait()
                        await _forward_event(ws, event, accumulated_text)
                    break
                continue
            await _forward_event(ws, event, accumulated_text)
            if event["type"] in ("done", "error"):
                break
        if accumulated_text:
            await ws.send_json({
                "type": "message", "role": "agent",
                "content": "".join(accumulated_text),
            })
        conv.busy = False
        return

    # --- Direct tool execution (no Claude) ---

    # Map shorthand actions to tool calls
    tool_name = action
    tool_input = {}

    if action == "repair":
        tool_name = "repair_workflow"
        tool_input = {"auto_install": True}
    elif action == "reconfigure":
        tool_name = "reconfigure_workflow"
        tool_input = {"auto_fix": True}
    elif action == "validate":
        tool_name = "validate_before_execute"
        tool_input = {}
    elif action == "install_node_pack":
        tool_name = "install_node_pack"
        tool_input = {"url": data.get("url", ""), "name": data.get("name", "")}
    elif action == "download_model":
        tool_name = "download_model"
        tool_input = {
            "url": data.get("url", ""),
            "model_type": data.get("model_type", "checkpoints"),
            "filename": data.get("filename", ""),
        }

    if tool_name not in _DIRECT_ACTIONS:
        await ws.send_json({
            "type": "error",
            "message": f"Unknown action: {action}",
        })
        return

    # Execute tool directly in executor (no API call)
    await ws.send_json({
        "type": "stage",
        "stage": "DISCOVER" if "repair" in tool_name or "reconfigure" in tool_name else "PILOT",
        "detail": f"Running {tool_name}...",
    })

    try:
        from agent.tools.comfy_provision import handle as provision_handle
        from agent.tools.comfy_execute import handle as execute_handle

        if tool_name == "validate_before_execute":
            result_json = await _run_in_executor_with_session(
                loop, execute_handle, tool_name, tool_input,
                session_id=conv.id,
            )
        else:
            result_json = await _run_in_executor_with_session(
                loop, provision_handle, tool_name, tool_input,
                session_id=conv.id,
            )

        # Build and send panel
        panel = _build_panel_for_tool(tool_name, tool_input, result_json)
        if panel:
            await ws.send_json({"type": "panel", "panel": panel})

        # Also send the result as a text message for context
        result = json.loads(result_json)
        msg = result.get("message", result.get("error", "Done."))
        await ws.send_json({
            "type": "message", "role": "agent",
            "content": msg,
        })

    except Exception as e:
        log.error("Direct action failed: %s", e, exc_info=True)
        await ws.send_json({
            "type": "error",
            "message": f"Action failed: {e}",
        })

    await ws.send_json({"type": "stage", "stage": "DONE", "detail": ""})


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def websocket_handler(request):
    """Bidirectional WebSocket for real-time sidebar <-> agent communication."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    if not _ensure_brain():
        await ws.send_json({
            "type": "error",
            "message": "Agent brain failed to load. Check logs.",
        })
        await ws.close()
        return ws

    conv = ConversationState()
    _conversations[conv.id] = conv

    # Send environment info on connect so panel has native awareness
    try:
        env = _build_environment()
    except Exception:
        env = {}
    await ws.send_json({
        "type": "connected",
        "conversation_id": conv.id,
        "environment": env,
    })

    log.info("WebSocket connected: %s", conv.id)

    try:
        async for raw_msg in ws:
            if raw_msg.type in (web.WSMsgType.TEXT,):
                try:
                    data = json.loads(raw_msg.data)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = data.get("type", "")

                if msg_type == "chat":
                    content = data.get("content", "").strip()
                    if not content:
                        continue

                    if conv.busy:
                        await ws.send_json({
                            "type": "error",
                            "message": "Agent is still processing. Please wait.",
                        })
                        continue

                    # Inject workflow if provided by frontend.  The executor
                    # call sets _conn_session = conv.id so load_workflow_from_data
                    # writes into THIS conversation's WorkflowSession instead of
                    # the shared "default" slot.
                    loop = asyncio.get_running_loop()
                    workflow_data = data.get("workflow")
                    if workflow_data and isinstance(workflow_data, dict):
                        try:
                            await _run_in_executor_with_session(
                                loop, _inject_workflow, conv, workflow_data,
                                session_id=conv.id,
                            )
                        except Exception as e:
                            log.warning("Workflow injection error: %s", e)

                    # Emit missing nodes panel if detected on this injection
                    if conv._missing_nodes_full:
                        panel = _panel_missing_nodes(conv._missing_nodes_full)
                        await ws.send_json({"type": "panel", "panel": panel})

                    conv.busy = True
                    await ws.send_json({"type": "stage", "stage": "THINKING", "detail": ""})

                    # Run agent in background thread, stream results back.
                    # The thread sets _conn_session = conv.id so every tool call
                    # made by the agent loop sees this conversation's session.
                    msg_queue = queue.Queue()

                    thread = _spawn_with_session(
                        _run_agent_sync,
                        (conv, content, msg_queue),
                        session_id=conv.id,
                    )
                    thread.start()

                    # Drain the queue and forward events to WebSocket
                    accumulated_text = []
                    while True:
                        try:
                            event = await loop.run_in_executor(
                                None, lambda: msg_queue.get(timeout=0.1)
                            )
                        except queue.Empty:
                            if not thread.is_alive():
                                # Thread finished — drain remaining
                                while not msg_queue.empty():
                                    event = msg_queue.get_nowait()
                                    await _forward_event(ws, event, accumulated_text)
                                break
                            continue

                        await _forward_event(ws, event, accumulated_text)

                        if event["type"] in ("done", "error"):
                            break

                    # Send final assembled message
                    if accumulated_text:
                        full_text = "".join(accumulated_text)
                        await ws.send_json({
                            "type": "message",
                            "role": "agent",
                            "content": full_text,
                        })

                    conv.busy = False

                elif msg_type == "approve":
                    # Phase 4: patch approval
                    pass

                elif msg_type == "reject":
                    # Phase 4: patch rejection
                    pass

                elif msg_type == "action":
                    # Panel action — execute tools DIRECTLY (no Claude round-trip)
                    action = data.get("action", "")
                    await _handle_panel_action(ws, conv, loop, action, data)

            elif raw_msg.type == web.WSMsgType.ERROR:
                log.error("WebSocket error: %s", ws.exception())

    finally:
        _conversations.pop(conv.id, None)
        log.info("WebSocket disconnected: %s", conv.id)

    return ws


async def _forward_event(ws, event, accumulated_text):
    """Forward an agent event to the WebSocket client."""
    etype = event["type"]

    if etype == "text_delta":
        # Accumulate text (we send the full message at the end)
        accumulated_text.append(event["text"])
        # Also send delta for live streaming
        await ws.send_json({
            "type": "text_delta",
            "text": event["text"],
        })

    elif etype == "tool_call":
        event_data = {
            "type": "stage",
            "stage": event.get("stage", "THINKING"),
            "detail": f"Using {event['tool']}...",
        }
        nodes = event.get("nodes_touched", [])
        if nodes:
            event_data["nodes_touched"] = nodes
        await ws.send_json(event_data)

    elif etype == "panel":
        await ws.send_json({
            "type": "panel",
            "panel": event["panel"],
        })

    elif etype == "agent_dispatch":
        await ws.send_json({
            "type": "agent_dispatch",
            "prompt": event.get("prompt", ""),
            "agents": event.get("agents", []),
            "estimate_seconds": event.get("estimate_seconds"),
            "node_count": event.get("node_count"),
        })

    elif etype == "agent_status":
        await ws.send_json({
            "type": "agent_status",
            "agent_key": event.get("agent_key", ""),
            "status": event.get("status", "waiting"),
            "message": event.get("message"),
            "duration": event.get("duration"),
        })

    elif etype == "error":
        await ws.send_json({
            "type": "error",
            "message": event.get("message", "Unknown error"),
            "recoverable": True,
        })

    elif etype == "done":
        await ws.send_json({"type": "stage", "stage": "DONE", "detail": ""})


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

async def handle_chat(request):
    """POST /superduper/chat -- simple request/response chat."""
    if not _ensure_brain():
        return web.json_response(
            {"error": "Agent brain not available"}, status=503
        )

    data = await request.json()
    content = data.get("content", "").strip()
    if not content:
        return web.json_response({"error": "Empty message"}, status=400)

    # Use a temporary conversation for REST endpoint
    conv = ConversationState()
    msg_queue = queue.Queue()

    loop = asyncio.get_running_loop()
    await _run_in_executor_with_session(
        loop, _run_agent_sync, conv, content, msg_queue,
        session_id=conv.id,
    )

    # Collect all text
    text_parts = []
    stage = "DONE"
    while not msg_queue.empty():
        event = msg_queue.get_nowait()
        if event["type"] == "text_delta":
            text_parts.append(event["text"])
        elif event["type"] == "tool_call":
            stage = event.get("stage", stage)

    return web.json_response({
        "role": "agent",
        "content": "".join(text_parts),
        "stage": stage,
    })


async def handle_status(request):
    """GET /superduper/status -- agent and connection state."""
    brain_ok = _ensure_brain()
    return web.json_response({
        "brain": "ready" if brain_ok else "unavailable",
        "active_conversations": len(_conversations),
        "conversations": [
            {"id": c.id, "busy": c.busy, "messages": len(c.messages)}
            for c in _conversations.values()
        ],
    })


# ---------------------------------------------------------------------------
# Route setup (called from ui/__init__.py at import time)
# ---------------------------------------------------------------------------

def setup_routes():
    """Mount all COMFY COZY routes on ComfyUI's PromptServer."""
    try:
        from server import PromptServer
        routes = PromptServer.instance.routes
        routes.post("/superduper/chat")(handle_chat)
        routes.get("/superduper/status")(handle_status)
        routes.get("/superduper/ws")(websocket_handler)
        log.info("COMFY COZY routes mounted: /superduper/chat, /superduper/status, /superduper/ws")
    except Exception as e:
        log.error("Failed to mount COMFY COZY routes: %s", e)
