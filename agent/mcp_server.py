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
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

log = logging.getLogger(__name__)


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


def create_mcp_server() -> "Server":
    """Create and configure the MCP server with all agent tools.

    Returns a configured mcp.server.Server instance ready to connect
    to a transport.
    """
    from .tools import ALL_TOOLS, handle as handle_tool

    server = Server("comfyui-agent")

    @server.list_tools()
    async def list_tools() -> list:
        """Return all agent tools in MCP format."""
        mcp_tools = []
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
        from .config import MCP_AUTH_TOKEN
        if MCP_AUTH_TOKEN:
            log.warning(
                "MCP_AUTH_TOKEN is set but stdio transport cannot enforce auth. "
                "For authenticated access, use HTTP/SSE transport with a reverse proxy."
            )

        arguments = arguments or {}

        # Extract optional session_id for future multi-session support
        session_id = arguments.pop("_session_id", None)

        # Our tool handlers are synchronous — run in thread executor
        # to avoid blocking the async event loop
        loop = asyncio.get_running_loop()
        try:
            import functools
            handler = functools.partial(
                handle_tool, name, arguments, session_id=session_id
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

    async with stdio_server() as (read_stream, write_stream):
        log.info("ComfyUI Agent MCP server starting (stdio transport)")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


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
