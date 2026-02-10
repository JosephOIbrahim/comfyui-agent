"""Test the tool registry and dispatch."""

import json
from agent.tools import ALL_TOOLS, handle


class TestToolRegistry:
    def test_all_tools_have_required_fields(self):
        """Every tool must have name, description, input_schema."""
        for tool in ALL_TOOLS:
            assert "name" in tool, f"Tool missing name: {tool}"
            assert "description" in tool, f"Tool {tool['name']} missing description"
            assert "input_schema" in tool, f"Tool {tool['name']} missing input_schema"
            assert tool["input_schema"]["type"] == "object"

    def test_no_duplicate_names(self):
        names = [t["name"] for t in ALL_TOOLS]
        assert len(names) == len(set(names)), f"Duplicate tool names: {names}"

    def test_unknown_tool_returns_error(self):
        result = handle("nonexistent_tool", {})
        assert "Unknown tool" in result

    def test_expected_tools_present(self):
        """Phase 1 + Phase 2 + Phase 3 tools should all be registered."""
        names = {t["name"] for t in ALL_TOOLS}
        expected = {
            # Phase 1: ComfyUI API
            "is_comfyui_running",
            "get_all_nodes",
            "get_node_info",
            "get_system_stats",
            "get_queue_status",
            "get_history",
            # Phase 1: Filesystem inspection
            "list_custom_nodes",
            "list_models",
            "get_models_summary",
            "read_node_source",
            # Phase 2: Workflow parsing
            "load_workflow",
            "validate_workflow",
            "get_editable_fields",
            # Phase 3: Workflow patching
            "apply_workflow_patch",
            "preview_workflow_patch",
            "undo_workflow_patch",
            "get_workflow_diff",
            "save_workflow",
            "reset_workflow",
            # Phase 3: Execution
            "execute_workflow",
            "get_execution_status",
            # Phase 4: Discovery
            "search_custom_nodes",
            "search_models",
            "find_missing_nodes",
            # Phase 5: Session memory
            "save_session",
            "load_session",
            "list_sessions",
            "add_note",
        }
        assert expected.issubset(names), f"Missing tools: {expected - names}"

    def test_total_tool_count(self):
        assert len(ALL_TOOLS) == 28, f"Expected 28 tools, got {len(ALL_TOOLS)}"
