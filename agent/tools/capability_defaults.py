"""Static default capabilities for every tool in the agent.

Provides ``build_default_capabilities()`` which returns a complete list of
``ToolCapability`` instances covering all ~103 tools across intelligence,
brain, stage, and utility layers.

Grouped by source module for maintainability.  Add new tools to the
appropriate group when extending the agent.
"""

from __future__ import annotations

from .capability_registry import ToolCapability


def _cap(name: str, **kw) -> ToolCapability:
    """Shorthand constructor — fills tool_name automatically."""
    return ToolCapability(tool_name=name, **kw)


def build_default_capabilities() -> list[ToolCapability]:
    """Return capability descriptors for every registered tool."""
    caps: list[ToolCapability] = []

    # ------------------------------------------------------------------
    # comfy_api — live ComfyUI HTTP queries
    # ------------------------------------------------------------------
    for name in (
        "is_comfyui_running",
        "get_all_nodes",
        "get_node_info",
        "get_system_stats",
        "get_queue_status",
        "get_history",
    ):
        caps.append(_cap(
            name, requires_comfyui=True, phase="understand", risk_level=0,
        ))

    # ------------------------------------------------------------------
    # comfy_inspect — filesystem / server inspection
    # ------------------------------------------------------------------
    caps.append(_cap(
        "list_custom_nodes", requires_comfyui=False, phase="understand",
        risk_level=0,
    ))
    caps.append(_cap(
        "list_models", requires_comfyui=False, phase="understand",
        risk_level=0,
    ))
    caps.append(_cap(
        "get_models_summary", requires_comfyui=False, phase="understand",
        risk_level=0,
    ))
    caps.append(_cap(
        "read_node_source", requires_comfyui=False, phase="understand",
        risk_level=0,
    ))

    # ------------------------------------------------------------------
    # workflow_parse — read-only workflow analysis
    # ------------------------------------------------------------------
    for name in (
        "load_workflow", "validate_workflow",
        "get_editable_fields", "classify_workflow",
    ):
        caps.append(_cap(name, risk_level=0, phase="understand"))

    # ------------------------------------------------------------------
    # workflow_patch — workflow mutation
    # ------------------------------------------------------------------
    caps.append(_cap(
        "apply_workflow_patch", mutates_workflow=True, risk_level=1,
        phase="pilot",
    ))
    caps.append(_cap(
        "preview_workflow_patch", mutates_workflow=False, risk_level=0,
        phase="pilot",
    ))
    caps.append(_cap(
        "undo_workflow_patch", mutates_workflow=True, risk_level=1,
        phase="pilot",
    ))
    caps.append(_cap(
        "get_workflow_diff", mutates_workflow=False, risk_level=0,
        phase="pilot",
    ))
    caps.append(_cap(
        "save_workflow", mutates_workflow=True, risk_level=1, phase="pilot",
    ))
    caps.append(_cap(
        "reset_workflow", mutates_workflow=True, risk_level=1, phase="pilot",
    ))

    # ------------------------------------------------------------------
    # workflow_templates — read-only template access
    # ------------------------------------------------------------------
    caps.append(_cap("list_workflow_templates", risk_level=0, phase="discover"))
    caps.append(_cap("get_workflow_template", risk_level=0, phase="discover"))

    # ------------------------------------------------------------------
    # comfy_execute — workflow execution
    # ------------------------------------------------------------------
    caps.append(_cap(
        "validate_before_execute", requires_comfyui=True, risk_level=0,
        phase="verify",
    ))
    caps.append(_cap(
        "execute_workflow", requires_comfyui=True, risk_level=2,
        phase="pilot", latency_class="batch",
    ))
    caps.append(_cap(
        "get_execution_status", requires_comfyui=True, risk_level=0,
        phase="pilot",
    ))
    caps.append(_cap(
        "execute_with_progress", requires_comfyui=True, risk_level=2,
        phase="pilot", latency_class="batch",
    ))

    # ------------------------------------------------------------------
    # comfy_discover — node/model discovery
    # ------------------------------------------------------------------
    for name in (
        "discover", "find_missing_nodes",
        "check_registry_freshness", "get_install_instructions",
    ):
        caps.append(_cap(name, risk_level=0, phase="discover"))

    # ------------------------------------------------------------------
    # session_tools — session persistence
    # ------------------------------------------------------------------
    caps.append(_cap("save_session", risk_level=1))
    caps.append(_cap("load_session", risk_level=1))
    caps.append(_cap("list_sessions", risk_level=0))
    caps.append(_cap("add_note", risk_level=1))

    # ------------------------------------------------------------------
    # civitai_api — CivitAI model discovery
    # ------------------------------------------------------------------
    caps.append(_cap("get_civitai_model", risk_level=0, phase="discover"))
    caps.append(_cap("get_trending_models", risk_level=0, phase="discover"))

    # ------------------------------------------------------------------
    # model_compat — model family identification
    # ------------------------------------------------------------------
    caps.append(_cap("identify_model_family", risk_level=0))
    caps.append(_cap("check_model_compatibility", risk_level=0))

    # ------------------------------------------------------------------
    # verify_execution — post-run verification
    # ------------------------------------------------------------------
    caps.append(_cap("verify_execution", risk_level=0, phase="verify"))

    # ------------------------------------------------------------------
    # github_releases — update checking
    # ------------------------------------------------------------------
    caps.append(_cap("check_node_updates", risk_level=0))
    caps.append(_cap("get_repo_releases", risk_level=0))

    # ------------------------------------------------------------------
    # pipeline — multi-step pipeline execution
    # ------------------------------------------------------------------
    caps.append(_cap("create_pipeline", risk_level=1, phase="pilot"))
    caps.append(_cap("get_pipeline_status", risk_level=0, phase="pilot"))
    caps.append(_cap(
        "run_pipeline", risk_level=2, phase="pilot", latency_class="batch",
    ))

    # ------------------------------------------------------------------
    # image_metadata — embedded image metadata
    # ------------------------------------------------------------------
    caps.append(_cap("write_image_metadata", risk_level=1))
    caps.append(_cap("read_image_metadata", risk_level=0))
    caps.append(_cap("reconstruct_context", risk_level=0))

    # ------------------------------------------------------------------
    # node_replacement — deprecated node migration
    # ------------------------------------------------------------------
    caps.append(_cap("get_node_replacements", risk_level=0))
    caps.append(_cap("check_workflow_deprecations", risk_level=0))
    caps.append(_cap(
        "migrate_deprecated_nodes", risk_level=1, mutates_workflow=True,
    ))

    # ------------------------------------------------------------------
    # comfy_provision — install / uninstall (high risk)
    # ------------------------------------------------------------------
    caps.append(_cap(
        "install_node_pack", requires_comfyui=True, risk_level=3,
        phase="discover", latency_class="batch",
    ))
    caps.append(_cap(
        "download_model", risk_level=3, phase="discover",
        latency_class="batch",
    ))
    caps.append(_cap(
        "uninstall_node_pack", requires_comfyui=True, risk_level=4,
        phase="discover", latency_class="batch",
    ))

    # ------------------------------------------------------------------
    # provision_tools — stage-based provisioning
    # ------------------------------------------------------------------
    caps.append(_cap(
        "provision_download", risk_level=3, requires_stage=True,
        latency_class="batch",
    ))
    caps.append(_cap(
        "provision_status", risk_level=0, requires_stage=True,
    ))
    caps.append(_cap(
        "provision_verify", risk_level=0, requires_stage=True,
    ))

    # ------------------------------------------------------------------
    # stage_tools — staging area for workflow edits
    # ------------------------------------------------------------------
    caps.append(_cap(
        "stage_read", risk_level=0, requires_stage=True,
    ))
    caps.append(_cap(
        "stage_write", risk_level=1, requires_stage=True,
    ))
    caps.append(_cap(
        "stage_add_delta", risk_level=1, requires_stage=True,
    ))
    caps.append(_cap(
        "stage_list_deltas", risk_level=0, requires_stage=True,
    ))
    caps.append(_cap(
        "stage_rollback", risk_level=1, requires_stage=True,
    ))
    caps.append(_cap(
        "stage_reconstruct_clean", risk_level=0, requires_stage=True,
    ))

    # ------------------------------------------------------------------
    # foresight_tools — predictive experiment engine
    # ------------------------------------------------------------------
    caps.append(_cap(
        "predict_experiment", risk_level=0, requires_stage=True,
    ))
    caps.append(_cap(
        "propose_improvement", risk_level=0, requires_stage=True,
    ))
    caps.append(_cap(
        "record_experience", risk_level=1, requires_stage=True,
    ))
    caps.append(_cap(
        "get_experience_stats", risk_level=0, requires_stage=True,
    ))
    caps.append(_cap(
        "list_counterfactuals", risk_level=0, requires_stage=True,
    ))
    caps.append(_cap(
        "get_prediction_accuracy", risk_level=0, requires_stage=True,
    ))
    caps.append(_cap(
        "check_evolution_tier", risk_level=0, requires_stage=True,
    ))
    caps.append(_cap(
        "get_calibration_stats", risk_level=0, requires_stage=True,
    ))
    caps.append(_cap(
        "get_meta_history", risk_level=0, requires_stage=True,
    ))

    # ------------------------------------------------------------------
    # compositor_tools — scene composition
    # ------------------------------------------------------------------
    caps.append(_cap(
        "compose_scene", risk_level=1, requires_stage=True,
        mutates_workflow=True,
    ))
    caps.append(_cap(
        "validate_scene", risk_level=0, requires_stage=True,
    ))
    caps.append(_cap(
        "export_scene", risk_level=1, requires_stage=True,
    ))
    caps.append(_cap(
        "extract_conditioning", risk_level=0, requires_stage=True,
    ))

    # ------------------------------------------------------------------
    # hyperagent_tools — autonomous workflow repair / reconfigure
    # ------------------------------------------------------------------
    caps.append(_cap(
        "reconfigure_workflow", risk_level=1, mutates_workflow=True,
        requires_stage=True,
    ))
    caps.append(_cap(
        "repair_workflow", risk_level=1, mutates_workflow=True,
        requires_stage=True,
    ))

    # ------------------------------------------------------------------
    # Semantic build tools (from workflow_patch or dedicated module)
    # ------------------------------------------------------------------
    for name in ("add_node", "connect_nodes", "set_input"):
        caps.append(_cap(
            name, risk_level=1, mutates_workflow=True, phase="pilot",
        ))

    # ------------------------------------------------------------------
    # Brain tools — all require the brain layer
    # ------------------------------------------------------------------

    # Vision
    caps.append(_cap(
        "analyze_image", requires_brain=True, risk_level=2,
        phase="verify", latency_class="batch",
        output_type="structured",
    ))
    caps.append(_cap(
        "compare_outputs", requires_brain=True, risk_level=2,
        phase="verify", latency_class="batch",
        output_type="structured",
    ))
    caps.append(_cap(
        "suggest_improvements", requires_brain=True, risk_level=0,
        phase="verify", latency_class="batch",
    ))
    caps.append(_cap(
        "hash_compare_images", requires_brain=True, risk_level=0,
        phase="verify",
    ))

    # Planner
    caps.append(_cap(
        "plan_goal", requires_brain=True, risk_level=0,
        phase="understand",
    ))
    caps.append(_cap(
        "get_plan", requires_brain=True, risk_level=0,
        phase="understand",
    ))
    caps.append(_cap(
        "complete_step", requires_brain=True, risk_level=0,
        phase="understand",
    ))
    caps.append(_cap(
        "replan", requires_brain=True, risk_level=0, phase="understand",
    ))

    # Memory
    caps.append(_cap(
        "record_outcome", requires_brain=True, risk_level=1,
    ))
    caps.append(_cap(
        "get_learned_patterns", requires_brain=True, risk_level=0,
    ))
    caps.append(_cap(
        "get_recommendations", requires_brain=True, risk_level=0,
    ))
    caps.append(_cap(
        "detect_implicit_feedback", requires_brain=True, risk_level=0,
    ))

    # Orchestrator
    caps.append(_cap(
        "spawn_subtask", requires_brain=True, risk_level=1,
        latency_class="batch",
    ))
    caps.append(_cap(
        "check_subtasks", requires_brain=True, risk_level=0,
        latency_class="batch",
    ))

    # Optimizer
    caps.append(_cap(
        "profile_workflow", requires_brain=True, risk_level=0,
        requires_comfyui=True,
    ))
    caps.append(_cap(
        "suggest_optimizations", requires_brain=True, risk_level=0,
    ))
    caps.append(_cap(
        "check_tensorrt_status", requires_brain=True, risk_level=0,
        requires_comfyui=True,
    ))
    caps.append(_cap(
        "apply_optimization", requires_brain=True, risk_level=2,
        mutates_workflow=True,
    ))

    # Demo
    caps.append(_cap(
        "start_demo", requires_brain=True, risk_level=1,
    ))
    caps.append(_cap(
        "demo_checkpoint", requires_brain=True, risk_level=1,
    ))

    # Intent
    caps.append(_cap(
        "capture_intent", requires_brain=True, risk_level=1,
    ))
    caps.append(_cap(
        "get_current_intent", requires_brain=True, risk_level=0,
    ))

    # Iteration
    caps.append(_cap(
        "start_iteration_tracking", requires_brain=True, risk_level=0,
    ))
    caps.append(_cap(
        "record_iteration_step", requires_brain=True, risk_level=1,
    ))
    caps.append(_cap(
        "finalize_iterations", requires_brain=True, risk_level=1,
    ))

    # Iterative refine (MoE)
    caps.append(_cap(
        "iterative_refine", requires_brain=True, risk_level=2,
        latency_class="batch",
    ))

    # Ping
    caps.append(_cap(
        "comfyui_agent_ping", requires_brain=True, risk_level=0,
    ))

    # ------------------------------------------------------------------
    # Utility / standalone tools
    # ------------------------------------------------------------------
    caps.append(_cap("classify_intent", risk_level=0))
    caps.append(_cap("get_output_path", risk_level=0))

    return caps
