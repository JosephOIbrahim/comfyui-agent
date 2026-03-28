"""Test the tool registry and dispatch."""

import json
from unittest.mock import patch

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
            "classify_workflow",
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
            "discover",
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
            # Verify loop
            "get_output_path",
            "verify_execution",
            # Brain: Iterative refine
            "iterative_refine",
            # GitHub releases
            "check_node_updates",
            "get_repo_releases",
            # Pipeline chaining
            "create_pipeline",
            "run_pipeline",
            "get_pipeline_status",
            # Discovery (additional)
            "get_install_instructions",
            # Node replacement
            "get_node_replacements",
            "check_workflow_deprecations",
            "migrate_deprecated_nodes",
            # Image metadata
            "write_image_metadata",
            "read_image_metadata",
            "reconstruct_context",
            # Brain: Intent collector
            "capture_intent",
            "get_current_intent",
            # Brain: Iteration tracking
            "start_iteration_tracking",
            "record_iteration_step",
            "finalize_iterations",
            # Brain: MoE classify
            "classify_intent",
            # Stage: Provisioner tools
            "provision_download",
            "provision_verify",
            "provision_status",
            # Stage: Stage tools
            "stage_read",
            "stage_write",
            "stage_add_delta",
            "stage_rollback",
            "stage_reconstruct_clean",
            "stage_list_deltas",
            # Stage: FORESIGHT tools
            "predict_experiment",
            "record_experience",
            "get_experience_stats",
            "list_counterfactuals",
            "get_prediction_accuracy",
            # Stage: Compositor tools
            "compose_scene",
            "validate_scene",
            "extract_conditioning",
            "export_scene",
            # Stage: Hyperagent tools
            "propose_improvement",
            "check_evolution_tier",
            "get_meta_history",
            "get_calibration_stats",
        }
        assert expected == names, (
            f"Tool mismatch!\n"
            f"  Missing from registry: {expected - names}\n"
            f"  Missing from expected: {names - expected}"
        )

    def test_total_tool_count(self):
        assert len(ALL_TOOLS) == 103, f"Expected 103 tools, got {len(ALL_TOOLS)}"

    def test_brain_tool_error_returns_json(self):
        """Brain tool exceptions should be caught and returned as JSON errors."""
        with patch("agent.brain.handle", side_effect=RuntimeError("test boom")):
            result = handle("plan_goal", {})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "plan_goal" in parsed["error"]

    def test_intelligence_tool_error_returns_json(self):
        """Intelligence layer tool exceptions should be caught and returned as JSON."""
        with patch(
            "agent.tools.comfy_api.handle",
            side_effect=RuntimeError("api boom"),
        ):
            result = handle("is_comfyui_running", {})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "is_comfyui_running" in parsed["error"]
