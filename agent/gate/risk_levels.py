"""Risk level classification for all tools in the system.

Every tool is assigned a static risk level that governs how the
pre-dispatch gate handles it.  Unknown tools default to REVERSIBLE
(safe-side default: not free-pass, not locked).

Risk Levels:
  READ_ONLY   (0)  Always pass, zero overhead.
  REVERSIBLE  (1)  Workflow mutations (undoable via undo_workflow_patch).
  EXECUTION   (2)  GPU execution or API calls that produce output.
  PROVISION   (3)  Filesystem modifications (downloads, installs).
  DESTRUCTIVE (4)  Uninstall, delete, reset -- never auto-opens.
"""

from __future__ import annotations

from enum import IntEnum


class RiskLevel(IntEnum):
    READ_ONLY = 0
    REVERSIBLE = 1
    EXECUTION = 2
    PROVISION = 3
    DESTRUCTIVE = 4


# ---------------------------------------------------------------------------
# Static classification: tool_name -> RiskLevel
# ---------------------------------------------------------------------------

TOOL_RISK_LEVELS: dict[str, RiskLevel] = {
    # ------------------------------------------------------------------
    # READ_ONLY (0) — pure reads, zero side-effects
    # ------------------------------------------------------------------
    # Live API reads
    "is_comfyui_running": RiskLevel.READ_ONLY,
    "get_all_nodes": RiskLevel.READ_ONLY,
    "get_node_info": RiskLevel.READ_ONLY,
    "get_system_stats": RiskLevel.READ_ONLY,
    "get_queue_status": RiskLevel.READ_ONLY,
    "get_history": RiskLevel.READ_ONLY,
    # Filesystem reads
    "list_custom_nodes": RiskLevel.READ_ONLY,
    "list_models": RiskLevel.READ_ONLY,
    "get_models_summary": RiskLevel.READ_ONLY,
    "read_node_source": RiskLevel.READ_ONLY,
    # Workflow reads
    "load_workflow": RiskLevel.READ_ONLY,
    "validate_workflow": RiskLevel.READ_ONLY,
    "get_editable_fields": RiskLevel.READ_ONLY,
    "classify_workflow": RiskLevel.READ_ONLY,
    "get_workflow_diff": RiskLevel.READ_ONLY,
    "preview_workflow_patch": RiskLevel.READ_ONLY,
    # Discovery reads
    "discover": RiskLevel.READ_ONLY,
    "find_missing_nodes": RiskLevel.READ_ONLY,
    "check_registry_freshness": RiskLevel.READ_ONLY,
    "get_install_instructions": RiskLevel.READ_ONLY,
    # CivitAI reads
    "get_civitai_model": RiskLevel.READ_ONLY,
    "get_trending_models": RiskLevel.READ_ONLY,
    # Model compatibility reads
    "identify_model_family": RiskLevel.READ_ONLY,
    "check_model_compatibility": RiskLevel.READ_ONLY,
    # Node replacement / deprecation reads
    "get_node_replacements": RiskLevel.READ_ONLY,
    "check_workflow_deprecations": RiskLevel.READ_ONLY,
    "check_node_updates": RiskLevel.READ_ONLY,
    "get_repo_releases": RiskLevel.READ_ONLY,
    # Template reads
    "list_workflow_templates": RiskLevel.READ_ONLY,
    "get_workflow_template": RiskLevel.READ_ONLY,
    # Session reads
    "list_sessions": RiskLevel.READ_ONLY,
    "load_session": RiskLevel.READ_ONLY,
    # Planner reads
    "get_plan": RiskLevel.READ_ONLY,
    # Memory reads
    "get_learned_patterns": RiskLevel.READ_ONLY,
    "get_recommendations": RiskLevel.READ_ONLY,
    "get_experience_stats": RiskLevel.READ_ONLY,
    "list_counterfactuals": RiskLevel.READ_ONLY,
    "get_prediction_accuracy": RiskLevel.READ_ONLY,
    # Intent reads
    "get_current_intent": RiskLevel.READ_ONLY,
    # Execution status reads
    "get_execution_status": RiskLevel.READ_ONLY,
    "get_output_path": RiskLevel.READ_ONLY,
    # Stage reads
    "stage_read": RiskLevel.READ_ONLY,
    "stage_list_deltas": RiskLevel.READ_ONLY,
    "stage_reconstruct_clean": RiskLevel.READ_ONLY,
    # Provision reads
    "provision_status": RiskLevel.READ_ONLY,  # Iter 13: reverted Cycle 64 backwards rename
    "provision_verify": RiskLevel.READ_ONLY,  # Iter 13: reverted Cycle 64 backwards rename
    "suggest_wiring": RiskLevel.READ_ONLY,  # Cycle 64: was missing
    # Metadata reads
    "read_image_metadata": RiskLevel.READ_ONLY,
    "reconstruct_context": RiskLevel.READ_ONLY,
    # Optimizer reads
    "profile_workflow": RiskLevel.READ_ONLY,
    "suggest_optimizations": RiskLevel.READ_ONLY,
    "check_tensorrt_status": RiskLevel.READ_ONLY,
    # Subtask reads
    "check_subtasks": RiskLevel.READ_ONLY,
    # Ping
    "comfyui_agent_ping": RiskLevel.READ_ONLY,
    # Calibration / meta reads
    "classify_intent": RiskLevel.READ_ONLY,
    "get_calibration_stats": RiskLevel.READ_ONLY,
    "check_evolution_tier": RiskLevel.READ_ONLY,
    "get_meta_history": RiskLevel.READ_ONLY,
    # Pipeline reads
    "get_pipeline_status": RiskLevel.READ_ONLY,
    # Validate (read-only check)
    "validate_before_execute": RiskLevel.READ_ONLY,
    "validate_scene": RiskLevel.READ_ONLY,
    # ------------------------------------------------------------------
    # REVERSIBLE (1) — workflow mutations, undoable
    # ------------------------------------------------------------------
    "apply_workflow_patch": RiskLevel.REVERSIBLE,
    "add_node": RiskLevel.REVERSIBLE,
    "connect_nodes": RiskLevel.REVERSIBLE,
    "set_input": RiskLevel.REVERSIBLE,
    "save_workflow": RiskLevel.REVERSIBLE,
    "undo_workflow_patch": RiskLevel.REVERSIBLE,
    "stage_write": RiskLevel.REVERSIBLE,
    "stage_add_delta": RiskLevel.REVERSIBLE,
    # Session writes (additive, non-destructive)
    "save_session": RiskLevel.REVERSIBLE,
    "add_note": RiskLevel.REVERSIBLE,
    # Planner writes (additive)
    "plan_goal": RiskLevel.REVERSIBLE,
    "complete_step": RiskLevel.REVERSIBLE,
    "replan": RiskLevel.REVERSIBLE,
    # Memory writes (additive)
    "record_outcome": RiskLevel.REVERSIBLE,
    "detect_implicit_feedback": RiskLevel.REVERSIBLE,
    "record_experience": RiskLevel.REVERSIBLE,
    # Intent writes
    "capture_intent": RiskLevel.REVERSIBLE,
    # Iteration tracking (additive)
    "start_iteration_tracking": RiskLevel.REVERSIBLE,
    "record_iteration_step": RiskLevel.REVERSIBLE,
    "finalize_iterations": RiskLevel.REVERSIBLE,
    # Metadata writes
    "write_image_metadata": RiskLevel.REVERSIBLE,
    # Reconfigure / repair (undoable via patch history)
    "reconfigure_workflow": RiskLevel.REVERSIBLE,
    "repair_workflow": RiskLevel.REVERSIBLE,
    # Scene composition (reversible, additive)
    "compose_scene": RiskLevel.REVERSIBLE,
    "export_scene": RiskLevel.REVERSIBLE,
    "extract_conditioning": RiskLevel.REVERSIBLE,
    # Pipeline creation (reversible)
    "create_pipeline": RiskLevel.REVERSIBLE,
    # Experiment / improvement (additive)
    "predict_experiment": RiskLevel.REVERSIBLE,
    "propose_improvement": RiskLevel.REVERSIBLE,
    # Demo (state management, reversible)
    "start_demo": RiskLevel.REVERSIBLE,
    "demo_checkpoint": RiskLevel.REVERSIBLE,
    # Subtask spawn (reversible)
    "spawn_subtask": RiskLevel.REVERSIBLE,
    # Optimizer apply (reversible via undo)
    "apply_optimization": RiskLevel.REVERSIBLE,
    # Iterative refine (reversible)
    "iterative_refine": RiskLevel.REVERSIBLE,
    "wire_model": RiskLevel.REVERSIBLE,  # Cycle 64: was missing
    # ------------------------------------------------------------------
    # EXECUTION (2) — GPU execution, API calls that produce output
    # ------------------------------------------------------------------
    "execute_workflow": RiskLevel.EXECUTION,
    "execute_with_progress": RiskLevel.EXECUTION,
    "analyze_image": RiskLevel.EXECUTION,
    "compare_outputs": RiskLevel.EXECUTION,
    "suggest_improvements": RiskLevel.EXECUTION,
    "hash_compare_images": RiskLevel.EXECUTION,
    "run_pipeline": RiskLevel.EXECUTION,
    "verify_execution": RiskLevel.EXECUTION,  # Cycle 64: was missing (triggers GPU output analysis)
    # ------------------------------------------------------------------
    # PROVISION (3) — filesystem modifications (downloads, installs)
    # ------------------------------------------------------------------
    "install_node_pack": RiskLevel.PROVISION,
    "download_model": RiskLevel.PROVISION,
    "provision_download": RiskLevel.PROVISION,
    "provision_model": RiskLevel.PROVISION,  # Cycle 64: was missing (downloads + wires a model)
    # ------------------------------------------------------------------
    # DESTRUCTIVE (4) — never auto-opens
    # ------------------------------------------------------------------
    "uninstall_node_pack": RiskLevel.DESTRUCTIVE,
    "reset_workflow": RiskLevel.DESTRUCTIVE,
    "stage_rollback": RiskLevel.DESTRUCTIVE,
    "migrate_deprecated_nodes": RiskLevel.DESTRUCTIVE,
}


def get_risk_level(tool_name: str) -> RiskLevel:
    """Get risk level for a tool.

    Defaults to REVERSIBLE for unknown tools -- safe default that
    requires gate checks without being overly restrictive.
    """
    return TOOL_RISK_LEVELS.get(tool_name, RiskLevel.REVERSIBLE)
