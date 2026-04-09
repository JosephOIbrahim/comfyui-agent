"""Server routes for the Comfy Cozy Panel.

Thin REST wrappers mounted on ComfyUI's PromptServer via aiohttp.
Each route translates HTTP requests into agent tool calls.
Covers the full tool surface: load, edit, execute, discover.
"""

import logging

from aiohttp import web

from .middleware import check_auth, check_rate_limit, check_size

log = logging.getLogger("comfy-cozy")


def _tool_call(tool_name, tool_input):
    """Call an agent tool and return the JSON string result."""
    from agent.tools import handle
    return handle(tool_name, tool_input)


def _guard(request, category, *, post=False):
    """Combined auth + rate limit + size guard. Returns rejection response or None."""
    rejected = check_auth(request) or check_rate_limit(category)
    if rejected:
        return rejected
    if post:
        rejected = check_size(request)
        if rejected:
            return rejected
    return None


def setup_routes():
    """Mount panel routes on PromptServer."""
    try:
        from server import PromptServer
        routes = PromptServer.instance.routes
    except Exception:
        log.debug("PromptServer not available — routes not mounted")
        return

    # ── Health ─────────────────────────────────────────────────────

    @routes.get("/comfy-cozy/health")
    async def health(request):
        try:
            from agent.health import check_health

            result = check_health()
            status_code = 200 if result["status"] == "ok" else 503
            return web.json_response(result, status=status_code)
        except Exception as e:
            log.error("Health check error: %s", e, exc_info=True)
            return web.json_response(
                {"status": "error", "error": "Health check failed"},
                status=503,
            )

    # ── Graph State (CognitiveGraphEngine) ─────────────────────────

    @routes.get("/comfy-cozy/graph-state")
    async def graph_state(request):
        """Read CognitiveGraphEngine state."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            from agent.tools.workflow_patch import get_current_workflow, get_engine
            workflow = get_current_workflow()
            engine = get_engine()

            result = {
                "has_workflow": workflow is not None,
                "node_count": len(workflow) if workflow else 0,
                "has_engine": engine is not None,
                "delta_count": len(engine.delta_stack) if engine else 0,
                "integrity": None,
                "deltas": [],
                "nodes": {},
            }

            if engine:
                ok, errors = engine.verify_stack_integrity()
                result["integrity"] = {"intact": ok, "errors": errors}
                for delta in engine.delta_stack:
                    result["deltas"].append({
                        "layer_id": delta.layer_id,
                        "opinion": delta.opinion,
                        "description": delta.description,
                        "timestamp": delta.timestamp,
                        "mutations": delta.mutations,
                    })

            if workflow:
                for nid, ndata in sorted(workflow.items()):
                    if isinstance(ndata, dict) and "class_type" in ndata:
                        result["nodes"][nid] = {
                            "class_type": ndata["class_type"],
                            "inputs": ndata.get("inputs", {}),
                        }

            return web.json_response(result)
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    # ── Workflow API (Reverse Bridge) ─────────────────────────────

    @routes.get("/comfy-cozy/get-workflow-api")
    async def get_workflow_api(request):
        """Return the current agent workflow in API format.

        This is the reverse bridge: the frontend polls this to push
        agent-side mutations back onto the ComfyUI canvas.
        """
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            from agent.tools.workflow_patch import get_current_workflow
            workflow = get_current_workflow()
            if workflow is None:
                return web.json_response({"error": "No workflow loaded"}, status=404)
            return web.json_response({"workflow": workflow})
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    # ── Workflow Loading ───────────────────────────────────────────

    @routes.post("/comfy-cozy/load-workflow")
    async def load_workflow(request):
        """Load a workflow from a file path."""
        try:
            rejected = _guard(request, "mutation", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("load_workflow", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.post("/comfy-cozy/load-workflow-data")
    async def load_workflow_data(request):
        """Load a workflow from raw JSON data (canvas injection).

        This is the critical bridge: the frontend sends the live canvas
        graph here on every change. The agent can then repair, modify,
        and execute the workflow.
        """
        try:
            rejected = _guard(request, "mutation", post=True)
            if rejected:
                return rejected
            body = await request.json()
            workflow_data = body.get("data", {})
            source = body.get("source", "<panel>")

            from agent.tools.workflow_patch import load_workflow_from_data
            err = load_workflow_from_data(workflow_data, source=source)
            if err:
                return web.json_response({"error": err}, status=400)

            result = {"loaded": True}

            # Count nodes for context
            nodes = {
                k: v for k, v in workflow_data.items()
                if isinstance(v, dict) and "class_type" in v
            }
            result["node_count"] = len(nodes)

            # Check for missing nodes (best-effort, non-blocking)
            try:
                missing_json = _tool_call("find_missing_nodes", {})
                import json as _json
                missing_data = _json.loads(missing_json)
                missing = missing_data.get("missing", [])
                if missing:
                    result["missing_nodes"] = [
                        {"class_type": m.get("class_type", "?"), "pack": m.get("pack", "")}
                        for m in missing
                    ]
            except Exception:
                pass  # Missing node check is best-effort

            return web.json_response(result)
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    # ── Workflow Mutation ──────────────────────────────────────────

    @routes.post("/comfy-cozy/set-input")
    async def set_input(request):
        """Push a delta layer via set_input tool."""
        try:
            rejected = _guard(request, "mutation", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("set_input", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.post("/comfy-cozy/add-node")
    async def add_node(request):
        """Add a new node to the workflow."""
        try:
            rejected = _guard(request, "mutation", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("add_node", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.post("/comfy-cozy/connect-nodes")
    async def connect_nodes(request):
        """Connect two nodes in the workflow."""
        try:
            rejected = _guard(request, "mutation", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("connect_nodes", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.post("/comfy-cozy/apply-patch")
    async def apply_patch(request):
        """Apply RFC6902 patches to the workflow."""
        try:
            rejected = _guard(request, "mutation", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("apply_workflow_patch", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.post("/comfy-cozy/rollback")
    async def rollback(request):
        """Undo the last delta layer."""
        try:
            rejected = _guard(request, "mutation")
            if rejected:
                return rejected
            result = _tool_call("undo_workflow_patch", {})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.post("/comfy-cozy/reset")
    async def reset(request):
        """Reset workflow to base state."""
        try:
            rejected = _guard(request, "mutation")
            if rejected:
                return rejected
            result = _tool_call("reset_workflow", {})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/diff")
    async def diff(request):
        """Get diff from base workflow."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            result = _tool_call("get_workflow_diff", {})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/editable-fields")
    async def editable_fields(request):
        """Get editable fields of the loaded workflow."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            result = _tool_call("get_editable_fields", {})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    # ── Execution ─────────────────────────────────────────────────

    @routes.post("/comfy-cozy/validate")
    async def validate(request):
        """Pre-execution validation."""
        try:
            rejected = _guard(request, "execute")
            if rejected:
                return rejected
            result = _tool_call("validate_before_execute", {})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.post("/comfy-cozy/execute")
    async def execute(request):
        """Execute the loaded workflow on ComfyUI."""
        try:
            rejected = _guard(request, "execute", post=True)
            if rejected:
                return rejected
            body = await request.json() if request.content_length else {}
            result = _tool_call("execute_workflow", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/execution-status")
    async def execution_status(request):
        """Check execution status."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            prompt_id = request.query.get("prompt_id", "")
            result = _tool_call("get_execution_status", {"prompt_id": prompt_id})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    # ── Discovery ─────────────────────────────────────────────────

    @routes.get("/comfy-cozy/node-info")
    async def node_info(request):
        """Get info for a specific node type."""
        try:
            rejected = _guard(request, "discover")
            if rejected:
                return rejected
            node_type = request.query.get("node_type", "")
            result = _tool_call("get_node_info", {"node_type": node_type})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/models")
    async def models(request):
        """List models by type."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            model_type = request.query.get("model_type", "checkpoints")
            result = _tool_call("list_models", {"model_type": model_type, "format": "summary"})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/system-stats")
    async def system_stats(request):
        """Get ComfyUI system stats."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            result = _tool_call("get_system_stats", {})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    # ── Cognitive Layer ────────────────────────────────────────────

    @routes.get("/comfy-cozy/experience")
    async def experience(request):
        """Read ExperienceAccumulator stats."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            from cognitive.experience.accumulator import ExperienceAccumulator
            acc = ExperienceAccumulator()
            return web.json_response(acc.get_stats())
        except ImportError:
            return web.json_response({
                "total_generations": 0,
                "learning_phase": "prior",
                "experience_weight": 0,
                "message": "Cognitive module not available",
            })
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/autoresearch")
    async def autoresearch(request):
        """Read autoresearch results."""
        rejected = _guard(request, "read")
        if rejected:
            return rejected
        return web.json_response({
            "status": "idle",
            "message": "No autoresearch run active",
        })

    # ── Discovery ─────────────────────────────────────────────────

    @routes.post("/comfy-cozy/discover")
    async def discover(request):
        """Search for models, nodes, or workflows."""
        try:
            rejected = _guard(request, "discover", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("discover", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/list-custom-nodes")
    async def list_custom_nodes(request):
        """List installed custom node packs."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            result = _tool_call("list_custom_nodes", {})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/models-summary")
    async def models_summary(request):
        """Get summary of all installed models."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            result = _tool_call("get_models_summary", {})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/queue-status")
    async def queue_status(request):
        """Get ComfyUI queue status."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            result = _tool_call("get_queue_status", {})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/history")
    async def history(request):
        """Get execution history."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            max_items = int(request.query.get("max_items", "10"))
            result = _tool_call("get_history", {"max_items": max_items})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    # ── Provisioning ──────────────────────────────────────────────

    @routes.post("/comfy-cozy/install-node-pack")
    async def install_node_pack(request):
        """Install a custom node pack from a URL."""
        try:
            rejected = _guard(request, "download", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("install_node_pack", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.post("/comfy-cozy/download-model")
    async def download_model(request):
        """Download a model from a URL."""
        try:
            rejected = _guard(request, "download", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("download_model", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.post("/comfy-cozy/uninstall-node-pack")
    async def uninstall_node_pack(request):
        """Uninstall a custom node pack."""
        try:
            rejected = _guard(request, "download", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("uninstall_node_pack", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    # ── Repair ────────────────────────────────────────────────────

    @routes.post("/comfy-cozy/repair-workflow")
    async def repair_workflow(request):
        """Repair workflow by finding and installing missing nodes."""
        try:
            rejected = _guard(request, "mutation", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("repair_workflow", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.post("/comfy-cozy/reconfigure-workflow")
    async def reconfigure_workflow(request):
        """Reconfigure workflow to fix compatibility issues."""
        try:
            rejected = _guard(request, "mutation", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("reconfigure_workflow", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/check-deprecations")
    async def check_deprecations(request):
        """Check for deprecated nodes in the loaded workflow."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            result = _tool_call("check_workflow_deprecations", {})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.post("/comfy-cozy/migrate-deprecated")
    async def migrate_deprecated(request):
        """Migrate deprecated nodes to replacements."""
        try:
            rejected = _guard(request, "mutation", post=True)
            if rejected:
                return rejected
            body = await request.json() if request.content_length else {}
            result = _tool_call("migrate_deprecated_nodes", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    # ── Session ───────────────────────────────────────────────────

    @routes.post("/comfy-cozy/save-session")
    async def save_session(request):
        """Save current session state."""
        try:
            rejected = _guard(request, "mutation", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("save_session", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.post("/comfy-cozy/load-session-data")
    async def load_session_data(request):
        """Load a saved session."""
        try:
            rejected = _guard(request, "mutation", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("load_session", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/list-sessions")
    async def list_sessions(request):
        """List all saved sessions."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            result = _tool_call("list_sessions", {})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    # ── Workflow Persistence ──────────────────────────────────────

    @routes.post("/comfy-cozy/save-workflow")
    async def save_workflow(request):
        """Save workflow to a file path."""
        try:
            rejected = _guard(request, "mutation", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("save_workflow", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.post("/comfy-cozy/preview-patch")
    async def preview_patch(request):
        """Preview a patch without applying it."""
        try:
            rejected = _guard(request, "read", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("preview_workflow_patch", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/classify-workflow")
    async def classify_workflow(request):
        """Classify the loaded workflow type."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            result = _tool_call("classify_workflow", {})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/workflow-templates")
    async def workflow_templates(request):
        """List available workflow templates."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            result = _tool_call("list_workflow_templates", {})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    # ── CivitAI ───────────────────────────────────────────────────

    @routes.get("/comfy-cozy/civitai-model")
    async def civitai_model(request):
        """Get CivitAI model details."""
        try:
            rejected = _guard(request, "discover")
            if rejected:
                return rejected
            model_id = request.query.get("model_id", "")
            result = _tool_call("get_civitai_model", {"model_id": model_id})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/trending-models")
    async def trending_models(request):
        """Get trending models from CivitAI."""
        try:
            rejected = _guard(request, "discover")
            if rejected:
                return rejected
            params = {}
            for key in ("model_type", "base_model", "period", "max_results"):
                val = request.query.get(key)
                if val is not None:
                    params[key] = int(val) if key == "max_results" else val
            result = _tool_call("get_trending_models", params)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    # ── Model Compatibility ───────────────────────────────────────

    @routes.post("/comfy-cozy/check-compatibility")
    async def check_compatibility(request):
        """Check compatibility between models."""
        try:
            rejected = _guard(request, "read", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("check_model_compatibility", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    # ── Auto-Wire ─────────────────────────────────────────────────

    @routes.post("/comfy-cozy/wire-model")
    async def wire_model(request):
        """Wire a downloaded model into the loaded workflow."""
        try:
            rejected = _guard(request, "mutation", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("wire_model", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/suggest-wiring")
    async def suggest_wiring(request):
        """Analyze workflow and suggest model wiring."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            result = _tool_call("suggest_wiring", {})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    # ── Provision Pipeline ──────────────────────────────────────

    @routes.post("/comfy-cozy/provision-model")
    async def provision_model(request):
        """One-step model provisioning: discover, download, verify, wire."""
        try:
            rejected = _guard(request, "download", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("provision_model", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.get("/comfy-cozy/provision-status")
    async def provision_pipeline_status(request):
        """Check what the workflow needs vs what is installed."""
        try:
            rejected = _guard(request, "read")
            if rejected:
                return rejected
            result = _tool_call("provision_pipeline_status", {})
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    @routes.post("/comfy-cozy/provision-verify")
    async def provision_pipeline_verify(request):
        """Verify a model file exists and check compatibility."""
        try:
            rejected = _guard(request, "read", post=True)
            if rejected:
                return rejected
            body = await request.json()
            result = _tool_call("provision_pipeline_verify", body)
            return web.Response(text=result, content_type="application/json")
        except Exception as e:
            log.error("Route %s error: %s", request.path, e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

    # ── Chat WebSocket ─────────────────────────────────────────────
    try:
        from .chat import websocket_handler as chat_ws_handler
        routes.get("/comfy-cozy/ws")(chat_ws_handler)
    except Exception as e:
        log.debug("Chat WebSocket not available: %s", e)

    log.info("Comfy Cozy Panel routes mounted (%d routes)", 50)
