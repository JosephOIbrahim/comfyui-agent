"""WebSocket chat handler for the Comfy Cozy Panel.

Bridges the panel frontend to the LLM agent via a persistent WebSocket.
Adapted from ui/server/routes.py but simplified for the panel context.
"""

import json
import logging
import queue
import sys
import threading
import uuid
from pathlib import Path

from aiohttp import web

log = logging.getLogger("comfy-cozy.chat")

# ---------------------------------------------------------------------------
# Lazy brain loading
# ---------------------------------------------------------------------------

_brain_lock = threading.Lock()
_brain_ready = False
_client = None


def _ensure_brain():
    """Lazily import the agent brain and create the LLM client."""
    global _brain_ready, _client
    if _brain_ready:
        return True

    with _brain_lock:
        if _brain_ready:
            return True

        try:
            project_root = str(Path(__file__).resolve().parent.parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from agent.main import create_client
            _client = create_client()
            _brain_ready = True
            log.info("Panel chat: agent brain loaded, LLM client ready")
            return True
        except Exception as e:
            log.error("Panel chat: failed to load agent brain: %s", e)
            return False


# ---------------------------------------------------------------------------
# Per-connection conversation state
# ---------------------------------------------------------------------------

class ConversationState:
    """Tracks one panel chat conversation."""

    def __init__(self):
        self.id = str(uuid.uuid4())[:8]
        self.messages: list[dict] = []
        self.system_prompt: str | None = None
        self.busy = False
        self.lock = threading.Lock()
        self.workflow_summary: dict | None = None
        self.missing_nodes: list[str] | None = None
        self._workflow_hash: int | None = None

    def _build_system(self):
        from agent.system_prompt import build_system_prompt
        base = build_system_prompt()

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
# Inject workflow from the panel's loaded workflow
# ---------------------------------------------------------------------------

def _inject_current_workflow(conv: ConversationState) -> None:
    """Load the current agent-side workflow into conversation context."""
    from agent.tools.workflow_patch import get_current_workflow
    workflow = get_current_workflow()
    if workflow is None:
        return

    from agent.tools._util import to_json as _to_json
    wf_hash = hash(_to_json(workflow))
    if wf_hash == conv._workflow_hash:
        conv._build_system()
        return

    conv._workflow_hash = wf_hash

    from agent.tools.workflow_parse import summarize_workflow_data
    conv.workflow_summary = summarize_workflow_data(workflow)
    conv._build_system()


def _inject_workflow_data(conv: ConversationState, workflow_data: dict) -> None:
    """Load explicit workflow data into the agent and conversation context."""
    from agent.tools._util import to_json as _to_json
    wf_hash = hash(_to_json(workflow_data))
    if wf_hash == conv._workflow_hash:
        conv._build_system()
        return

    conv._workflow_hash = wf_hash

    from agent.tools.workflow_patch import load_workflow_from_data
    err = load_workflow_from_data(workflow_data, source="<panel>")
    if err:
        log.warning("Workflow injection failed: %s", err)
        return

    from agent.tools.workflow_parse import summarize_workflow_data
    conv.workflow_summary = summarize_workflow_data(workflow_data)
    conv._build_system()


# ---------------------------------------------------------------------------
# Queue-based stream handler
# ---------------------------------------------------------------------------

_EXECUTION_TOOLS = frozenset({"execute_workflow", "execute_with_progress"})


class _QueueStreamHandler:
    """StreamHandler that pushes events to a thread-safe queue."""

    def __init__(self, msg_queue):
        self._q = msg_queue

    def on_text(self, text):
        self._q.put({"type": "text_delta", "text": text})

    def on_thinking(self, text):
        pass

    def on_tool_call(self, name, inp):
        self._q.put({"type": "tool_call", "name": name})
        if name in _EXECUTION_TOOLS:
            self._q.put({
                "type": "executing",
                "message": f"Running workflow via {name}...",
            })

    def on_tool_result(self, name, inp, result_json):
        pass

    def on_stream_end(self):
        pass

    def on_input(self):
        return None


# ---------------------------------------------------------------------------
# Synchronous agent runner
# ---------------------------------------------------------------------------

def _run_agent_sync(conv: ConversationState, user_text: str, msg_queue: queue.Queue):
    """Run the agent loop synchronously, pushing events to msg_queue."""
    from agent.main import run_agent_turn
    from agent.queue_progress import QueueProgressReporter

    conv._build_system()
    conv.messages.append({"role": "user", "content": user_text})

    max_turns = 15
    handler = _QueueStreamHandler(msg_queue)
    progress = QueueProgressReporter(msg_queue)

    # Patch handle_tool inside agent.main to forward the progress reporter.
    # run_agent_turn imports handle as handle_tool from agent.tools; we
    # temporarily replace agent.main.handle_tool so execution tools can
    # stream progress back through the queue.
    import agent.main as _main_mod
    _original_handle = _main_mod.handle_tool

    def _handle_with_progress(name, tool_input, **kw):
        return _original_handle(name, tool_input, progress=progress, **kw)

    _main_mod.handle_tool = _handle_with_progress

    try:
        for _turn in range(max_turns):
            try:
                conv.messages, done = run_agent_turn(
                    _client,
                    conv.messages,
                    conv.system_prompt,
                    handler=handler,
                )

                if done:
                    msg_queue.put({"type": "done"})
                    return

            except Exception as e:
                log.error("Agent turn error: %s", e, exc_info=True)
                msg_queue.put({"type": "error", "message": str(e)})
                return

        msg_queue.put({"type": "done"})
    finally:
        _main_mod.handle_tool = _original_handle


# ---------------------------------------------------------------------------
# Forward events from queue to WebSocket
# ---------------------------------------------------------------------------

async def _forward_event(ws, event, accumulated_text):
    """Forward an agent event to the WebSocket client."""
    etype = event["type"]

    if etype == "text_delta":
        accumulated_text.append(event["text"])
        await ws.send_json({"type": "text_delta", "text": event["text"]})

    elif etype == "tool_call":
        await ws.send_json({"type": "tool_call", "name": event.get("name", "")})

    elif etype == "progress":
        await ws.send_json({
            "type": "progress",
            "progress": event.get("progress", 0),
            "total": event.get("total"),
            "message": event.get("message", ""),
        })

    elif etype == "executing":
        await ws.send_json({
            "type": "executing",
            "message": event.get("message", "Executing workflow..."),
        })

    elif etype == "error":
        await ws.send_json({
            "type": "error",
            "message": event.get("message", "Unknown error"),
        })

    elif etype == "done":
        await ws.send_json({"type": "done"})


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def websocket_handler(request):
    """Bidirectional WebSocket for panel chat <-> agent communication."""
    import asyncio

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

    log.info("Panel WebSocket connected: %s", conv.id)

    try:
        async for raw_msg in ws:
            if raw_msg.type == web.WSMsgType.TEXT:
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

                    # Inject workflow context
                    loop = asyncio.get_running_loop()

                    workflow_data = data.get("workflow")
                    if workflow_data and isinstance(workflow_data, dict):
                        try:
                            await loop.run_in_executor(
                                None, _inject_workflow_data, conv, workflow_data,
                            )
                        except Exception as e:
                            log.warning("Workflow injection error: %s", e)
                    else:
                        try:
                            await loop.run_in_executor(
                                None, _inject_current_workflow, conv,
                            )
                        except Exception as e:
                            log.warning("Current workflow injection error: %s", e)

                    conv.busy = True

                    msg_queue = queue.Queue()
                    thread = threading.Thread(
                        target=_run_agent_sync,
                        args=(conv, content, msg_queue),
                        daemon=True,
                    )
                    thread.start()

                    accumulated_text = []
                    while True:
                        try:
                            event = await loop.run_in_executor(
                                None, lambda: msg_queue.get(timeout=0.1),
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

                    conv.busy = False

                elif msg_type == "workflow":
                    workflow_data = data.get("data")
                    if workflow_data and isinstance(workflow_data, dict):
                        loop = asyncio.get_running_loop()
                        try:
                            await loop.run_in_executor(
                                None, _inject_workflow_data, conv, workflow_data,
                            )
                            await ws.send_json({
                                "type": "workflow_ack",
                                "node_count": conv.workflow_summary.get(
                                    "node_count", 0
                                ) if conv.workflow_summary else 0,
                            })
                        except Exception as e:
                            log.warning("Workflow update error: %s", e)
                            await ws.send_json({
                                "type": "error",
                                "message": f"Workflow update failed: {e}",
                            })

            elif raw_msg.type == web.WSMsgType.ERROR:
                log.error("Panel WebSocket error: %s", ws.exception())

    finally:
        _conversations.pop(conv.id, None)
        log.info("Panel WebSocket disconnected: %s", conv.id)

    return ws
