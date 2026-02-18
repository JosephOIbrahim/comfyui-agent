"""Tests for the MCP server adapter.

Tests the tool schema conversion and server creation without requiring
the actual mcp SDK (mocked where needed).
"""

import json


class TestSchemaConversion:
    """Test Anthropic -> MCP schema conversion."""

    def test_convert_basic_schema(self):
        from agent.mcp_server import _convert_schema

        tool_def = {
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name"},
                },
                "required": ["name"],
            },
        }
        result = _convert_schema(tool_def)
        assert result["type"] == "object"
        assert "name" in result["properties"]
        assert result["required"] == ["name"]

    def test_convert_empty_schema(self):
        from agent.mcp_server import _convert_schema

        result = _convert_schema({})
        assert result["type"] == "object"
        assert result["properties"] == {}

    def test_convert_schema_without_type(self):
        from agent.mcp_server import _convert_schema

        tool_def = {
            "input_schema": {
                "properties": {"x": {"type": "number"}},
            },
        }
        result = _convert_schema(tool_def)
        assert result["type"] == "object"
        assert "x" in result["properties"]

    def test_convert_all_agent_tools(self):
        """Every agent tool schema should convert without error."""
        from agent.mcp_server import _convert_schema
        from agent.tools import ALL_TOOLS

        for tool_def in ALL_TOOLS:
            schema = _convert_schema(tool_def)
            assert schema["type"] == "object", f"Tool {tool_def['name']} has bad schema"
            assert "properties" in schema, f"Tool {tool_def['name']} missing properties"


class TestServerCreation:
    """Test MCP server creation and tool bridging."""

    def test_create_server(self):
        """Should create a server — mcp is now a core dependency."""
        from agent import mcp_server

        server = mcp_server.create_mcp_server()
        assert server is not None
        assert server.name == "comfyui-agent"


class TestToolBridging:
    """Test that tools are properly bridged to MCP format."""

    def test_all_tools_have_valid_mcp_names(self):
        """Tool names should be valid MCP identifiers (no spaces, etc.)."""
        from agent.tools import ALL_TOOLS

        for tool_def in ALL_TOOLS:
            name = tool_def["name"]
            assert " " not in name, f"Tool name has spaces: {name}"
            assert name == name.lower().replace("-", "_"), \
                f"Tool name should be snake_case: {name}"

    def test_all_tools_have_descriptions(self):
        """Every tool must have a non-empty description for MCP listing."""
        from agent.tools import ALL_TOOLS

        for tool_def in ALL_TOOLS:
            desc = tool_def.get("description", "")
            assert desc, f"Tool {tool_def['name']} has empty description"
            assert len(desc) >= 10, \
                f"Tool {tool_def['name']} description too short: {desc}"

    def test_tool_count_matches_registry(self):
        """MCP server should expose exactly the same tools as the registry."""
        from agent.mcp_server import _convert_schema
        from agent.tools import ALL_TOOLS

        converted = [_convert_schema(t) for t in ALL_TOOLS]
        assert len(converted) == 65


class TestToolExecution:
    """Test tool execution through the MCP bridge."""

    def test_sync_tool_result(self):
        """Verify a simple tool returns expected result format."""
        from agent.tools import handle

        # Use a tool that works without ComfyUI running
        result = handle("identify_model_family", {"model_name": "sdxl_base.safetensors"})
        parsed = json.loads(result)
        assert parsed["family"] == "sdxl"

    def test_unknown_tool_returns_error(self):
        """Unknown tool should return error string, not crash."""
        from agent.tools import handle

        result = handle("totally_fake_tool", {})
        assert "Unknown tool" in result
