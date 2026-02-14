"""Test the tool registry and dispatch."""

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
        """All tools should be registered."""
        names = {t["name"] for t in ALL_TOOLS}
        expected = {
            # ComfyUI API
            "is_comfyui_running",
            "get_all_nodes",
            "get_node_info",
            "get_system_stats",
            "get_queue_status",
            "get_history",
            # Filesystem inspection
            "list_custom_nodes",
            "list_models",
            "get_models_summary",
            "read_node_source",
            # Workflow parsing
            "load_workflow",
            "validate_workflow",
            "get_editable_fields",
            # Workflow patching (RFC6902)
            "apply_workflow_patch",
            "preview_workflow_patch",
            "undo_workflow_patch",
            "get_workflow_diff",
            "save_workflow",
            "reset_workflow",
            # Semantic composition
            "add_node",
            "connect_nodes",
            "set_input",
            # Execution
            "validate_before_execute",
            "execute_workflow",
            "get_execution_status",
            # Discovery
            "search_custom_nodes",
            "search_models",
            "find_missing_nodes",
            # Templates
            "list_workflow_templates",
            "get_workflow_template",
            # Session memory
            "save_session",
            "load_session",
            "list_sessions",
            "add_note",
            # Brain: Vision
            "analyze_image",
            "compare_outputs",
            "suggest_improvements",
            # Brain: Planner
            "plan_goal",
            "get_plan",
            "complete_step",
            "replan",
            # Brain: Memory
            "record_outcome",
            "get_learned_patterns",
            "get_recommendations",
            # Brain: Orchestrator
            "spawn_subtask",
            "check_subtasks",
            # Brain: Optimizer
            "profile_workflow",
            "suggest_optimizations",
            "check_tensorrt_status",
            "apply_optimization",
            # Brain: Demo
            "start_demo",
            "demo_checkpoint",
            # CivitAI discovery
            "search_civitai",
            "get_civitai_model",
            "get_trending_models",
            # Model compatibility
            "check_model_compatibility",
            "identify_model_family",
            # WebSocket execution
            "execute_with_progress",
            # Brain: Vision hash compare
            "hash_compare_images",
            # Freshness tracking
            "check_registry_freshness",
            # Implicit feedback
            "detect_implicit_feedback",
        }
        assert expected.issubset(names), f"Missing tools: {expected - names}"

    def test_total_tool_count(self):
        assert len(ALL_TOOLS) == 62, f"Expected 62 tools, got {len(ALL_TOOLS)}"
