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
import functools
import logging

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from .progress import ProgressReporter

log = logging.getLogger(__name__)

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

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> list:
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
        except Exception:
            pass  # No request context available — progress will be noop

        if session and progress_token is not None:
            progress = ProgressReporter(loop, session, progress_token, request_id)
        else:
            progress = ProgressReporter.noop()

        # Resolve session context for session-scoped state isolation
        from .session_context import get_session_context
        ctx = get_session_context(session_id or "default")

        # Our tool handlers are synchronous — run in thread executor
        # to avoid blocking the async event loop
        try:
            handler = functools.partial(
                handle_tool, name, arguments,
                session_id=session_id, progress=progress, ctx=ctx,
            )
            result = await loop.run_in_executor(None, handler)
        except Exception as e:
            log.error("Tool %s failed: %s", name, e, exc_info=True)
            return [types.TextContent(
                type="text",
                text=f"Error executing {name}: {e}",
            )]

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

    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
