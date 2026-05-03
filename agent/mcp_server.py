"""MCP (Model Context Protocol) server for the ComfyUI Agent.

Exposes all agent tools (intelligence + brain layers) as MCP tools,
allowing Claude Desktop, Claude Code, or any MCP-compatible client to
use ComfyUI agent capabilities directly.

Supports stdio transport (default) and SSE transport (--sse flag).

Usage:
    # stdio (for Claude Desktop / Claude Code config)
    agent mcp

    # Or run directly
    python -m agent.mcp_server
"""

import asyncio
import atexit
import functools
import logging
import signal
import threading
import uuid

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from .progress import ProgressReporter

log = logging.getLogger(__name__)

# Idempotency guard for the shutdown flush (atexit + SIGTERM both fire).
_shutdown_flushed = threading.Event()

# One stable session namespace per MCP server process.  Each ``agent mcp``
# launch gets its own UUID so multiple Claude Code instances don't share
# workflow/session state.  Tool handlers read this via current_conn_session()
# after _handler sets the ContextVar in the executor thread.
_SERVER_SESSION_ID: str = f"conn_{uuid.uuid4().hex[:8]}"

# ---------------------------------------------------------------------------
# Server instructions — tells MCP clients how to use the tools
# ---------------------------------------------------------------------------

_SERVER_INSTRUCTIONS = """\
# ComfyUI Agent

AI co-pilot for VFX artists using ComfyUI. 100+ tools across four phases:
UNDERSTAND -> DISCOVER -> PILOT -> VERIFY.

## Tool Usage Guide

1. **Always start** with `comfyui_agent_ping` to verify both the MCP server \
and ComfyUI are reachable.
2. **Before modifying workflows**: load with `load_workflow`, inspect with \
`get_editable_fields`, then patch with `add_node`/`connect_nodes`/`set_input` \
or `apply_workflow_patch`.
3. **Before executing**: call `validate_before_execute` to catch errors early.
4. **Use `get_node_info`** for node interfaces — never guess from memory.
5. **Use `discover`** to search for models and nodes — ecosystems change daily.
6. **Never generate entire workflows from scratch.** Make surgical, validated \
modifications.
7. **Vision tools** (`analyze_image`, `compare_outputs`) require the brain \
layer and an Anthropic API key.

## Quick Reference

| Need | Tool |
|------|------|
| Check connection | `comfyui_agent_ping` |
| Load a workflow | `load_workflow` |
| See editable fields | `get_editable_fields` |
| Change a value | `set_input` |
| Add a node | `add_node` |
| Wire nodes | `connect_nodes` |
| Find models/nodes | `discover` |
| Run workflow | `execute_workflow` |
| Check GPU/VRAM | `get_system_stats` |

## Model Families (never mix)

- **SD 1.5**: 512x512, CFG 7-12, negative prompts important
- **SDXL**: 1024x1024, CFG 5-9, base + refiner
- **Flux**: 512-1024, guidance ~1.0, FluxGuidance node, no negatives
- **SD3**: 1024x1024, CFG 5-7, triple text encoder
"""


def _convert_schema(tool_def: dict) -> dict:
    """Convert an Anthropic tool schema to MCP-compatible JSON Schema.

    Anthropic tool schemas use `input_schema` with a JSON Schema object.
    MCP uses `inputSchema` with the same JSON Schema format, so the
    conversion is mainly about ensuring the schema is well-formed.
    """
    schema = dict(tool_def.get("input_schema", {}))
    # Ensure type is present
    if "type" not in schema:
        schema["type"] = "object"
    # Ensure properties exists
    if "properties" not in schema:
        schema["properties"] = {}
    return schema


def _check_comfyui_reachable() -> dict:
    """Quick connectivity check to ComfyUI. Returns status dict."""
    from .config import COMFYUI_URL
    try:
        with httpx.Client() as client:
            resp = client.get(f"{COMFYUI_URL}/system_stats", timeout=5.0)
            resp.raise_for_status()
            stats = resp.json()
            devices = stats.get("devices", [])
            gpu = devices[0].get("name", "unknown") if devices else "no GPU"
            version = stats.get("system", {}).get("comfyui_version", "unknown")
            return {
                "reachable": True,
                "url": COMFYUI_URL,
                "version": version,
                "gpu": gpu,
            }
    except Exception as e:
        return {
            "reachable": False,
            "url": COMFYUI_URL,
            "error": str(e),
        }


def create_mcp_server() -> "Server":
    """Create and configure the MCP server with all agent tools.

    Returns a configured mcp.server.Server instance ready to connect
    to a transport.
    """
    from . import __version__
    from .tools import ALL_TOOLS, handle as handle_tool

    server = Server("comfyui-agent")

    @server.list_tools()
    async def list_tools() -> list:
        """Return all agent tools in MCP format, plus the built-in ping."""
        mcp_tools = [
            types.Tool(
                name="comfyui_agent_ping",
                description=(
                    "Health check for the ComfyUI Agent MCP server. "
                    "Returns server version, tool count, and ComfyUI "
                    "connection status. Call this first to verify everything "
                    "is working."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
        ]
        for tool_def in ALL_TOOLS:
            mcp_tools.append(types.Tool(
                name=tool_def["name"],
                description=tool_def.get("description", ""),
                inputSchema=_convert_schema(tool_def),
            ))
        return mcp_tools

    # ------------------------------------------------------------------
    # W2.2 — MCP resources for the USD stage
    # ------------------------------------------------------------------
    # Expose the four top-level prim trees in the cognitive stage as MCP
    # resources so external consumers (e.g., Moneta) can read and subscribe
    # to stage state without polling tools. URIs use the stage:// scheme
    # with the prim path as the path component.
    #
    # SESSION SCOPE: resources reflect the SERVER session
    # (`_SERVER_SESSION_ID`) — the same session that tool calls default to
    # when no explicit `session_id` argument is passed. MCP clients that
    # invoke tools with a custom `session_id` will see those mutations on
    # a separate SessionContext that is NOT visible through resources.
    # This is a deliberate constraint of the MCP resource protocol (no
    # per-call ContextVar surface for resource reads). For per-session
    # resources, future work would add a `stage://session/<id>/<prim>`
    # URI scheme.
    _STAGE_RESOURCE_PRIMS = ("/workflows", "/experience", "/agents", "/scenes")

    try:
        @server.list_resources()
        async def list_resources() -> list:
            """Advertise the four stage prim trees as MCP resources."""
            try:
                resources = []
                for prim_path in _STAGE_RESOURCE_PRIMS:
                    resources.append(types.Resource(
                        uri=f"stage://{prim_path.lstrip('/')}",
                        name=f"Stage {prim_path}",
                        description=(
                            f"USD prim subtree at {prim_path} from the "
                            f"CognitiveWorkflowStage. Returns a JSON snapshot "
                            f"of all attributes; updates are pushed to "
                            f"subscribers when stage state changes."
                        ),
                        mimeType="application/json",
                    ))
                return resources
            except Exception as exc:
                log.warning("list_resources failed: %s", exc)
                return []

        @server.read_resource()
        async def read_resource(uri) -> str:
            """Return a JSON snapshot of the requested stage subtree."""
            from .session_context import get_session_context
            from .tools._util import to_json
            uri_str = str(uri)
            if not uri_str.startswith("stage://"):
                raise ValueError(f"Unknown resource URI: {uri_str}")
            prim_path = "/" + uri_str[len("stage://"):].lstrip("/")

            ctx = get_session_context(_SERVER_SESSION_ID)
            stage = ctx.ensure_stage()
            if stage is None:
                return to_json({"error": "stage unavailable (usd-core missing)"})

            # Walk the subtree under prim_path collecting attributes.
            result: dict = {}
            try:
                root_prim = stage.stage.GetPrimAtPath(prim_path)
                if not root_prim.IsValid():
                    return to_json({"prim_path": prim_path, "exists": False})
                for prim in root_prim.GetAllChildren():
                    attrs = {}
                    for attr in prim.GetAttributes():
                        val = attr.Get()
                        if val is None:
                            continue
                        if isinstance(val, (bool, int, float, str)):
                            attrs[attr.GetName()] = val
                        else:
                            attrs[attr.GetName()] = str(val)
                    if attrs:
                        result[str(prim.GetPath())] = attrs
                return to_json({
                    "prim_path": prim_path,
                    "exists": True,
                    "children": result,
                })
            except Exception as exc:
                return to_json({"error": str(exc), "prim_path": prim_path})
    except (AttributeError, TypeError) as exc:
        # MCP SDK version may not expose @list_resources / @read_resource.
        # Don't fail server creation — just log and continue with tools only.
        log.warning(
            "MCP resources not registered (SDK lacks decorators): %s", exc
        )

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> list | types.CallToolResult:
        """Execute a tool call and return the result.

        Note: stdio transport provides process-level isolation but cannot
        enforce token-based auth. For production deployments needing auth,
        use HTTP/SSE transport with an auth proxy instead.
        """
        # Built-in ping tool — no delegation needed
        if name == "comfyui_agent_ping":
            from .tools._util import to_json
            try:
                from . import tool_count
                intel_count, brain_count, total_count = tool_count()
            except Exception:
                log.debug("tool_count() unavailable during ping", exc_info=True)
                intel_count, brain_count, total_count = "?", "?", "?"
            _loop = asyncio.get_running_loop()
            comfyui = await _loop.run_in_executor(None, _check_comfyui_reachable)
            return [types.TextContent(type="text", text=to_json({
                "status": "ok",
                "server": "comfyui-agent",
                "version": __version__,
                "tools": {
                    "intelligence": intel_count,
                    "brain": brain_count,
                    "total": total_count,
                },
                "comfyui": comfyui,
            }))]

        from .config import MCP_AUTH_TOKEN
        if MCP_AUTH_TOKEN:
            log.warning(
                "MCP_AUTH_TOKEN is set but stdio transport cannot enforce auth. "
                "For authenticated access, use HTTP/SSE transport with a reverse proxy."
            )

        arguments = arguments or {}

        # Extract optional session_id for future multi-session support
        session_id = arguments.pop("_session_id", None)

        # Build progress reporter so handlers can send live updates
        loop = asyncio.get_running_loop()
        progress_token = None
        session = None
        request_id = None
        try:
            ctx = server.request_context
            if ctx:
                session = ctx.session
                request_id = ctx.request_id
                if ctx.meta:
                    progress_token = ctx.meta.progressToken
        except Exception as _e:  # Cycle 62: log instead of silently swallow
            log.debug("Request context unavailable — progress will be noop: %s", _e)

        if session and progress_token is not None:
            progress = ProgressReporter(loop, session, progress_token, request_id)
        else:
            progress = ProgressReporter.noop()

        # Resolve session context for session-scoped state isolation.
        # _SERVER_SESSION_ID is generated once at process start — all tool
        # calls from this MCP server instance share the same session namespace.
        # The _handler wrapper below sets the ContextVar inside the executor
        # thread so session_tools.current_conn_session() returns the right name.
        from ._conn_ctx import _conn_session as _cs
        from .session_context import get_session_context
        conn_name = session_id or _SERVER_SESSION_ID
        ctx = get_session_context(conn_name)

        # Our tool handlers are synchronous — run in thread executor
        # to avoid blocking the async event loop.
        # _handler sets the ContextVar inside the worker thread so that
        # any tool (e.g. session_tools) that calls current_conn_session()
        # from within the thread gets the correct connection-scoped name.
        _conn_name_local = conn_name  # avoid late-binding in closure
        _partial = functools.partial(
            handle_tool, name, arguments,
            session_id=conn_name, progress=progress, ctx=ctx,
        )

        def _handler():
            _cs.set(_conn_name_local)
            return _partial()

        try:
            # Wrap in asyncio.wait_for so a hung tool handler (e.g. ComfyUI
            # unreachable, stuck lock) cannot block the MCP event loop
            # indefinitely. 120 s covers the slowest legitimate operations
            # (large model downloads are handled separately with their own
            # streaming timeout). (Cycle 31 fix)
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _handler),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            log.error("Tool %s timed out after 120 s", name)
            return types.CallToolResult(
                isError=True,
                content=[types.TextContent(
                    type="text",
                    text=(
                        f"Tool '{name}' timed out after 120 s. "
                        "ComfyUI may be unresponsive — check that it is running."
                    ),
                )],
            )
        except Exception as e:
            log.error("Tool %s failed: %s", name, e, exc_info=True)
            return types.CallToolResult(
                isError=True,
                content=[types.TextContent(
                    type="text",
                    text=f"Error executing {name}: {e}",
                )],
            )

        return [types.TextContent(type="text", text=result)]

    return server


async def run_stdio():
    """Run the MCP server using stdio transport."""
    server = create_mcp_server()

    # Startup health check — run sync HTTP call in executor to avoid
    # blocking the event loop during startup.
    loop = asyncio.get_running_loop()
    comfyui_status = await loop.run_in_executor(None, _check_comfyui_reachable)
    if comfyui_status["reachable"]:
        log.info(
            "ComfyUI Agent MCP server starting (stdio) — "
            "ComfyUI %s reachable at %s (%s)",
            comfyui_status.get("version", "?"),
            comfyui_status["url"],
            comfyui_status.get("gpu", "?"),
        )
    else:
        log.warning(
            "ComfyUI Agent MCP server starting (stdio) — "
            "ComfyUI NOT reachable at %s: %s",
            comfyui_status["url"],
            comfyui_status.get("error", "unknown"),
        )

    init_options = server.create_initialization_options()
    # Inject server instructions for MCP clients
    init_options.instructions = _SERVER_INSTRUCTIONS

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


def _flush_all_sessions() -> None:
    """Flush every session's stage to disk on shutdown.

    Unlike cli.py's atexit which only fires when --session NAME was set, this
    runs unconditionally for every active session. Each session flushes to its
    stage's _root_path (set from STAGE_DEFAULT_PATH at ensure_stage time) so
    callers without an explicit session name still get crash-safety.
    """
    if _shutdown_flushed.is_set():
        return
    _shutdown_flushed.set()
    try:
        from .session_context import iter_sessions
    except ImportError:
        # iter_sessions added below; if missing fall back to no-op.
        return
    for ctx in iter_sessions():
        try:
            ctx.stop_autosave()
        except Exception:
            pass
        stage = getattr(ctx, "_stage", None)
        if stage is None:
            continue
        root_path = getattr(stage, "_root_path", None)
        if root_path is None:
            continue
        try:
            stage.flush()
            log.info("Stage flushed to %s on shutdown", root_path)
        except Exception as exc:
            log.warning("Stage flush on shutdown failed: %s", exc)


def _sigterm_handler(_signum, _frame):
    """SIGTERM handler — flush stages then exit cleanly."""
    _flush_all_sessions()
    # Don't call sys.exit here; let the asyncio loop unwind naturally.


def main():
    """Entry point for running the MCP server."""
    # Configure logging to stderr (stdio transport uses stdout for JSON-RPC)
    from .logging_config import setup_logging
    from .config import LOG_DIR
    setup_logging(
        level=logging.INFO,
        log_file=LOG_DIR / "mcp.log",
        json_format=True,
    )

    # A4 — visible cold-start warning when usd-core is absent. Without this,
    # autosave/event-registry/Moneta-adapter all silently no-op and users
    # think the persistence layer is wired when it isn't.
    try:
        from .stage import HAS_USD
        if not HAS_USD:
            log.warning(
                "usd-core is NOT installed — Cozy persistence (stage flush, "
                "autosave, event subscribe registry, Moneta adapter) is "
                "DISABLED. Install with: pip install usd-core"
            )
    except ImportError:
        log.warning(
            "agent.stage module failed to import — persistence layer "
            "disabled."
        )

    # Register shutdown hooks for stage flush. These fire on:
    #   - normal interpreter exit (atexit)
    #   - SIGTERM from MCP client / OS
    # Note: SIGINT (Ctrl-C) is handled by asyncio's KeyboardInterrupt path,
    # which in turn triggers atexit during normal unwind.
    atexit.register(_flush_all_sessions)
    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except ValueError:
        # signal.signal() can only be called from the main thread; in test
        # harnesses or embedded contexts this may not be possible.
        log.debug("SIGTERM handler not registered (non-main thread)")

    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
