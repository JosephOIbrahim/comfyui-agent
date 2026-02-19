"""SUPER DUPER UI -- aiohttp routes mounted on ComfyUI's PromptServer.

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
                + "\nWarn the artist if they ask about issues or try to execute."
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
    import hashlib
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
            else:
                conv.missing_nodes = None
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
        desc = item.get("description", "")[:120]
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


# Tools that produce panelable results
_PANEL_TOOLS = {
    "load_workflow", "validate_workflow",
    "discover",
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

    return None


# ---------------------------------------------------------------------------
# Synchronous agent runner (called from thread)
# ---------------------------------------------------------------------------

def _run_agent_sync(conv: ConversationState, user_text: str, msg_queue: queue.Queue):
    """Run the agent loop synchronously, pushing events to msg_queue.

    Events are dicts: {"type": "text_delta"|"tool_call"|"stage"|"done"|"error", ...}
    """
    from agent.main import run_agent_turn
    from agent.tools import ALL_TOOLS

    conv._build_system()
    conv.messages.append({"role": "user", "content": user_text})

    max_turns = 15  # safety limit per user message

    for turn in range(max_turns):
        try:
            def on_text(text):
                msg_queue.put({"type": "text_delta", "text": text})

            def on_tool(name, inp):
                # Infer stage from tool name
                stage = _infer_stage(name)
                msg_queue.put({
                    "type": "tool_call",
                    "tool": name,
                    "stage": stage,
                })

            def on_result(name, inp, result_json):
                panel = _build_panel_for_tool(name, inp, result_json)
                if panel:
                    msg_queue.put({"type": "panel", "panel": panel})

            conv.messages, done = run_agent_turn(
                _client,
                conv.messages,
                conv.system_prompt,
                on_text_delta=on_text,
                on_tool_call=on_tool,
                on_tool_result=on_result,
            )

            if done:
                msg_queue.put({"type": "done"})
                return

        except Exception as e:
            log.error("Agent turn error: %s", e, exc_info=True)
            msg_queue.put({"type": "error", "message": str(e)})
            return

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

    await ws.send_json({
        "type": "connected",
        "conversation_id": conv.id,
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

                    # Inject workflow if provided by frontend
                    loop = asyncio.get_running_loop()
                    workflow_data = data.get("workflow")
                    if workflow_data and isinstance(workflow_data, dict):
                        try:
                            await loop.run_in_executor(
                                None, _inject_workflow, conv, workflow_data
                            )
                        except Exception as e:
                            log.warning("Workflow injection error: %s", e)

                    conv.busy = True
                    await ws.send_json({"type": "stage", "stage": "THINKING", "detail": ""})

                    # Run agent in background thread, stream results back
                    msg_queue = queue.Queue()

                    thread = threading.Thread(
                        target=_run_agent_sync,
                        args=(conv, content, msg_queue),
                        daemon=True,
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
        await ws.send_json({
            "type": "stage",
            "stage": event.get("stage", "THINKING"),
            "detail": f"Using {event['tool']}...",
        })

    elif etype == "panel":
        await ws.send_json({
            "type": "panel",
            "panel": event["panel"],
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
    await loop.run_in_executor(
        None, _run_agent_sync, conv, content, msg_queue
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
    """Mount all SUPER DUPER routes on ComfyUI's PromptServer."""
    try:
        from server import PromptServer
        routes = PromptServer.instance.routes
        routes.post("/superduper/chat")(handle_chat)
        routes.get("/superduper/status")(handle_status)
        routes.get("/superduper/ws")(websocket_handler)
        log.info("SUPER DUPER routes mounted: /superduper/chat, /superduper/status, /superduper/ws")
    except Exception as e:
        log.error("Failed to mount SUPER DUPER routes: %s", e)
