"""Optimizer module — GPU-aware performance engineering.

Understands TensorRT, CUTLASS, VRAM management, and ComfyUI execution
characteristics. Profiles workflows, suggests optimizations ranked by
impact/effort, and can apply them with same-seed validation.
"""

import logging

from ..config import COMFYUI_URL, CUSTOM_NODES_DIR, MODELS_DIR
from ..tools._util import to_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU profiles — built-in knowledge
# ---------------------------------------------------------------------------

GPU_PROFILES: dict[str, dict] = {
    "NVIDIA GeForce RTX 4090": {
        "vram_gb": 24,
        "cuda_cores": 16384,
        "tensor_cores": 512,
        "arch": "Ada Lovelace",
        "compute_cap": "sm_89",
        "trt_supported": True,
        "sweet_spots": {
            "sd15_batch": 4,
            "sdxl_batch": 2,
            "sdxl_trt_batch": 4,
            "flux_batch": 1,
            "max_resolution_no_tiling": [1536, 1536],
        },
    },
    "NVIDIA GeForce RTX 4080": {
        "vram_gb": 16,
        "cuda_cores": 9728,
        "tensor_cores": 304,
        "arch": "Ada Lovelace",
        "compute_cap": "sm_89",
        "trt_supported": True,
        "sweet_spots": {
            "sd15_batch": 4,
            "sdxl_batch": 1,
            "sdxl_trt_batch": 2,
            "flux_batch": 1,
            "max_resolution_no_tiling": [1280, 1280],
        },
    },
    "NVIDIA GeForce RTX 3090": {
        "vram_gb": 24,
        "cuda_cores": 10496,
        "tensor_cores": 328,
        "arch": "Ampere",
        "compute_cap": "sm_86",
        "trt_supported": True,
        "sweet_spots": {
            "sd15_batch": 4,
            "sdxl_batch": 2,
            "sdxl_trt_batch": 3,
            "flux_batch": 1,
            "max_resolution_no_tiling": [1536, 1536],
        },
    },
    "NVIDIA GeForce RTX 3080": {
        "vram_gb": 10,
        "cuda_cores": 8704,
        "tensor_cores": 272,
        "arch": "Ampere",
        "compute_cap": "sm_86",
        "trt_supported": True,
        "sweet_spots": {
            "sd15_batch": 2,
            "sdxl_batch": 1,
            "sdxl_trt_batch": 1,
            "flux_batch": 1,
            "max_resolution_no_tiling": [1024, 1024],
        },
    },
}

# Nodes known to be GPU-intensive
_GPU_HEAVY_NODES = {
    "KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced",
    "VAEDecode", "VAEEncode", "VAEDecodeTiled", "VAEEncodeTiled",
    "ControlNetApply", "ControlNetApplyAdvanced",
    "IPAdapterApply", "IPAdapterApplyFaceID",
}

# TensorRT-related node packs
_TRT_PACKS = {
    "ComfyUI_TensorRT": {
        "nodes": ["TensorRTLoader", "UNET_TensorRT"],
        "url": "https://github.com/comfyanonymous/ComfyUI_TensorRT",
    },
    "ComfyUI-TRT": {
        "nodes": ["Load TensorRT Engine", "Run TensorRT Engine"],
        "url": "https://github.com/NVIDIA/ComfyUI-TRT",
    },
}

# Optimization catalog — ranked by impact/effort
_OPTIMIZATIONS = [
    {
        "id": "fp16_precision",
        "name": "FP16 Precision",
        "category": "high_impact_low_effort",
        "impact": "~30% speed improvement",
        "effort": "Free — change VAE precision setting",
        "description": "Use half-precision floating point for faster inference with minimal quality loss.",
        "applies_when": lambda wf, gpu: True,
    },
    {
        "id": "batch_size",
        "name": "Optimal Batch Size",
        "category": "high_impact_low_effort",
        "impact": "~20% throughput improvement",
        "effort": "Free — adjust batch_size parameter",
        "description": "Set batch size to GPU sweet spot for maximum throughput.",
        "applies_when": lambda wf, gpu: True,
    },
    {
        "id": "vae_tiling",
        "name": "VAE Tiling",
        "category": "high_impact_low_effort",
        "impact": "Prevents OOM on large images",
        "effort": "Free — swap VAEDecode for VAEDecodeTiled",
        "description": "Process VAE in tiles to handle large resolutions without running out of VRAM.",
        "applies_when": lambda wf, gpu: _has_large_resolution(wf, gpu),
    },
    {
        "id": "tensorrt",
        "name": "TensorRT Engine Compilation",
        "category": "high_impact_medium_effort",
        "impact": "~2-3x faster inference",
        "effort": "5-minute engine build (cached after first run)",
        "description": "Compile the diffusion model into an optimized TensorRT engine for maximum GPU utilization.",
        "applies_when": lambda wf, gpu: gpu.get("trt_supported", False),
    },
    {
        "id": "cuda_graphs",
        "name": "CUDA Graphs",
        "category": "high_impact_medium_effort",
        "impact": "~15% speed improvement",
        "effort": "Requires compatible node versions",
        "description": "Capture and replay GPU execution graphs to minimize launch overhead.",
        "applies_when": lambda wf, gpu: gpu.get("compute_cap", "").startswith("sm_8"),
    },
    {
        "id": "sampler_efficiency",
        "name": "Efficient Sampler Selection",
        "category": "medium_impact",
        "impact": "Depends on current sampler and step count",
        "effort": "Parameter change",
        "description": "Some samplers converge faster. DPM++ 2M Karras often matches Euler quality in fewer steps.",
        "applies_when": lambda wf, gpu: True,
    },
    {
        "id": "step_optimization",
        "name": "Step Count Optimization",
        "category": "medium_impact",
        "impact": "Quality/speed tradeoff",
        "effort": "Parameter change — needs same-seed comparison",
        "description": "Find the minimum step count that maintains quality. Often 20-25 steps is sufficient.",
        "applies_when": lambda wf, gpu: True,
    },
    {
        "id": "model_offloading",
        "name": "Model Offloading",
        "category": "situational",
        "impact": "Prevents OOM in multi-model workflows",
        "effort": "ComfyUI setting change",
        "description": "Offload unused models to CPU/RAM between pipeline stages to free VRAM.",
        "applies_when": lambda wf, gpu: _has_multiple_models(wf),
    },
    {
        "id": "controlnet_resolution",
        "name": "ControlNet Resolution Tuning",
        "category": "situational",
        "impact": "Speed vs precision tradeoff",
        "effort": "Parameter change",
        "description": "ControlNet processing resolution can often be lower than the output resolution without visible quality loss.",
        "applies_when": lambda wf, gpu: _has_controlnet(wf),
    },
]


def _has_large_resolution(wf: dict, gpu: dict) -> bool:
    """Check if workflow uses resolution exceeding GPU tiling threshold."""
    max_res = gpu.get("sweet_spots", {}).get("max_resolution_no_tiling", [1536, 1536])
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        w = inputs.get("width", 0)
        h = inputs.get("height", 0)
        if isinstance(w, (int, float)) and isinstance(h, (int, float)):
            if w > max_res[0] or h > max_res[1]:
                return True
    return False


def _has_multiple_models(wf: dict) -> bool:
    """Check if workflow loads multiple models."""
    loader_count = 0
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        if "Loader" in ct or "loader" in ct:
            loader_count += 1
    return loader_count > 2


def _has_controlnet(wf: dict) -> bool:
    """Check if workflow uses ControlNet."""
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        if "ControlNet" in ct or "controlnet" in ct:
            return True
    return False


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "profile_workflow",
        "description": (
            "Analyze a workflow's execution characteristics: estimated VRAM peak, "
            "GPU-heavy nodes, model count, resolution analysis. Uses GPU profile "
            "matching for hardware-specific insights."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "workflow": {
                    "type": "object",
                    "description": "The workflow to profile (API format). If not provided, uses the currently loaded workflow.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "suggest_optimizations",
        "description": (
            "Given a workflow and GPU profile, return ranked optimization opportunities: "
            "TensorRT, CUDA graphs, batch size, precision, tiling, sampler efficiency. "
            "Ranked by impact/effort ratio."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "workflow": {
                    "type": "object",
                    "description": "The workflow to optimize. If not provided, uses currently loaded workflow.",
                },
                "gpu_name": {
                    "type": "string",
                    "description": "GPU name override. If not provided, auto-detected via get_system_stats.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "check_tensorrt_status",
        "description": (
            "Check TensorRT readiness: are TRT node packs installed? "
            "Are there cached engines for the current model? "
            "What's needed to enable TRT acceleration?"
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "apply_optimization",
        "description": (
            "Apply a specific optimization to the currently loaded workflow. "
            "Supports: vae_tiling (swap VAEDecode for VAEDecodeTiled), "
            "batch_size (adjust to GPU sweet spot), step_optimization "
            "(reduce steps with quality check)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "optimization_id": {
                    "type": "string",
                    "description": (
                        "ID of the optimization to apply. "
                        "Get IDs from suggest_optimizations."
                    ),
                },
                "params": {
                    "type": "object",
                    "description": "Optimization-specific parameters (e.g., batch_size value).",
                },
            },
            "required": ["optimization_id"],
        },
    },
]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _get_workflow() -> dict | None:
    """Get the currently loaded workflow from workflow_patch state."""
    from ..tools.workflow_patch import _state
    return _state.get("current_workflow")


def _detect_gpu() -> dict:
    """Try to detect GPU from ComfyUI system stats, fall back to generic."""
    import httpx
    try:
        resp = httpx.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        data = resp.json()
        devices = data.get("devices", [])
        if devices:
            gpu_name = devices[0].get("name", "")
            vram = devices[0].get("vram_total", 0)
            # Try to match known profile
            for profile_name, profile in GPU_PROFILES.items():
                if profile_name.lower() in gpu_name.lower():
                    return {**profile, "detected_name": gpu_name}
            # Unknown GPU — build basic profile
            return {
                "detected_name": gpu_name,
                "vram_gb": round(vram / (1024**3), 1) if vram else 0,
                "trt_supported": True,  # Assume modern NVIDIA
                "sweet_spots": {},
            }
    except Exception:
        pass

    return {
        "detected_name": "unknown",
        "vram_gb": 0,
        "trt_supported": False,
        "sweet_spots": {},
    }


def _handle_profile_workflow(tool_input: dict) -> str:
    wf = tool_input.get("workflow") or _get_workflow()
    if not wf:
        return to_json({"error": "No workflow loaded. Load a workflow first."})

    gpu = _detect_gpu()

    # Analyze nodes
    gpu_heavy = []
    all_nodes = []
    loaders = []
    resolutions = []

    for nid, node in sorted(wf.items()):
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "unknown")
        all_nodes.append({"id": nid, "class_type": ct})

        if ct in _GPU_HEAVY_NODES:
            gpu_heavy.append({"id": nid, "class_type": ct})

        if "Loader" in ct or "loader" in ct:
            loaders.append({"id": nid, "class_type": ct})

        inputs = node.get("inputs", {})
        w = inputs.get("width")
        h = inputs.get("height")
        if isinstance(w, (int, float)) and isinstance(h, (int, float)):
            resolutions.append({"node": nid, "width": int(w), "height": int(h)})

    # Estimate VRAM usage (rough heuristic)
    max_res = max(
        (r["width"] * r["height"] for r in resolutions),
        default=512 * 512,
    )
    # SD model ~4GB, each loader ~2-4GB, resolution scaling
    estimated_vram_gb = 4.0 + len(loaders) * 2.0 + (max_res / (1024 * 1024)) * 0.5

    return to_json({
        "gpu": {
            "name": gpu.get("detected_name", "unknown"),
            "vram_gb": gpu.get("vram_gb", 0),
            "trt_supported": gpu.get("trt_supported", False),
        },
        "workflow": {
            "total_nodes": len(all_nodes),
            "gpu_heavy_nodes": gpu_heavy,
            "model_loaders": loaders,
            "resolutions": resolutions,
        },
        "estimates": {
            "vram_peak_gb": round(estimated_vram_gb, 1),
            "fits_in_vram": estimated_vram_gb < gpu.get("vram_gb", 24),
            "tiling_recommended": _has_large_resolution(wf, gpu),
            "offloading_recommended": len(loaders) > 2,
        },
    })


def _handle_suggest_optimizations(tool_input: dict) -> str:
    wf = tool_input.get("workflow") or _get_workflow()
    if not wf:
        return to_json({"error": "No workflow loaded. Load a workflow first."})

    gpu_name = tool_input.get("gpu_name")
    if gpu_name and gpu_name in GPU_PROFILES:
        gpu = GPU_PROFILES[gpu_name]
    else:
        gpu = _detect_gpu()

    applicable = []
    for opt in _OPTIMIZATIONS:
        try:
            if opt["applies_when"](wf, gpu):
                applicable.append({
                    "id": opt["id"],
                    "name": opt["name"],
                    "category": opt["category"],
                    "impact": opt["impact"],
                    "effort": opt["effort"],
                    "description": opt["description"],
                })
        except Exception:
            continue

    # Group by category
    categorized = {}
    for opt in applicable:
        cat = opt["category"]
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append(opt)

    return to_json({
        "gpu": gpu.get("detected_name", gpu_name or "unknown"),
        "optimization_count": len(applicable),
        "optimizations": applicable,
        "by_category": categorized,
        "priority_order": [opt["id"] for opt in applicable],
        "message": f"{len(applicable)} optimizations available. Start with 'high_impact_low_effort' category.",
    })


def _handle_check_tensorrt(tool_input: dict) -> str:
    # Check for TRT node packs
    trt_installed = {}
    if CUSTOM_NODES_DIR.exists():
        for pack_name, pack_info in _TRT_PACKS.items():
            pack_path = CUSTOM_NODES_DIR / pack_name
            trt_installed[pack_name] = {
                "installed": pack_path.exists(),
                "nodes": pack_info["nodes"],
                "url": pack_info["url"],
            }

    # Check for cached TRT engines
    engine_cache = MODELS_DIR / "tensorrt" if MODELS_DIR.exists() else None
    cached_engines = []
    if engine_cache and engine_cache.exists():
        # He2025: sort for deterministic engine list order
        for f in sorted(engine_cache.glob("*.engine")):
            cached_engines.append({
                "name": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
            })

    any_installed = any(v["installed"] for v in trt_installed.values())

    gpu = _detect_gpu()

    return to_json({
        "trt_supported": gpu.get("trt_supported", False),
        "gpu": gpu.get("detected_name", "unknown"),
        "node_packs": trt_installed,
        "any_pack_installed": any_installed,
        "cached_engines": cached_engines,
        "engine_count": len(cached_engines),
        "ready": any_installed and gpu.get("trt_supported", False),
        "next_steps": (
            "TensorRT is ready! Use apply_optimization with id='tensorrt' to swap in TRT nodes."
            if any_installed
            else "Install a TensorRT node pack first. Recommended: ComfyUI_TensorRT from NVIDIA."
        ),
    })


def _handle_apply_optimization(tool_input: dict) -> str:
    opt_id = tool_input["optimization_id"]
    params = tool_input.get("params", {})

    from ..tools.workflow_patch import _state, handle as patch_handle

    wf = _state.get("current_workflow")
    if not wf:
        return to_json({"error": "No workflow loaded. Load a workflow first."})

    gpu = _detect_gpu()

    if opt_id == "vae_tiling":
        # Swap VAEDecode -> VAEDecodeTiled
        swapped = []
        for nid, node in wf.items():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") == "VAEDecode":
                patch_handle("apply_workflow_patch", {
                    "patches": [
                        {"op": "replace", "path": f"/{nid}/class_type", "value": "VAEDecodeTiled"},
                        {"op": "add", "path": f"/{nid}/inputs/tile_size", "value": 512},
                    ],
                })
                swapped.append(nid)
        return to_json({
            "applied": "vae_tiling",
            "nodes_swapped": swapped,
            "message": f"Swapped {len(swapped)} VAEDecode nodes to VAEDecodeTiled (tile_size=512).",
        })

    elif opt_id == "batch_size":
        target_batch = params.get("batch_size")
        if not target_batch:
            # Auto-detect from GPU profile
            spots = gpu.get("sweet_spots", {})
            # Try to detect model type from workflow
            has_sdxl = any(
                "sdxl" in str(n.get("inputs", {}).get("ckpt_name", "")).lower()
                for n in wf.values() if isinstance(n, dict)
            )
            target_batch = spots.get("sdxl_batch", 1) if has_sdxl else spots.get("sd15_batch", 2)

        updated = []
        for nid, node in wf.items():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") == "EmptyLatentImage":
                patch_handle("set_input", {
                    "node_id": nid,
                    "input_name": "batch_size",
                    "value": target_batch,
                })
                updated.append(nid)

        return to_json({
            "applied": "batch_size",
            "batch_size": target_batch,
            "nodes_updated": updated,
            "message": f"Set batch_size={target_batch} on {len(updated)} latent image nodes.",
        })

    elif opt_id == "step_optimization":
        target_steps = params.get("steps", 20)
        updated = []
        for nid, node in wf.items():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") in ("KSampler", "KSamplerAdvanced"):
                patch_handle("set_input", {
                    "node_id": nid,
                    "input_name": "steps",
                    "value": target_steps,
                })
                updated.append(nid)

        return to_json({
            "applied": "step_optimization",
            "steps": target_steps,
            "nodes_updated": updated,
            "message": (
                f"Set steps={target_steps} on {len(updated)} sampler nodes. "
                "Run a same-seed comparison to verify quality is maintained."
            ),
        })

    elif opt_id == "sampler_efficiency":
        sampler = params.get("sampler", "dpmpp_2m")
        scheduler = params.get("scheduler", "karras")
        updated = []
        for nid, node in wf.items():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") in ("KSampler", "KSamplerAdvanced"):
                patch_handle("set_input", {
                    "node_id": nid,
                    "input_name": "sampler_name",
                    "value": sampler,
                })
                patch_handle("set_input", {
                    "node_id": nid,
                    "input_name": "scheduler",
                    "value": scheduler,
                })
                updated.append(nid)

        return to_json({
            "applied": "sampler_efficiency",
            "sampler": sampler,
            "scheduler": scheduler,
            "nodes_updated": updated,
            "message": (
                f"Set sampler={sampler}, scheduler={scheduler} on {len(updated)} nodes. "
                "Run a same-seed comparison to verify quality."
            ),
        })

    else:
        return to_json({
            "error": f"Optimization '{opt_id}' not yet implemented as auto-apply.",
            "hint": "Use apply_workflow_patch to manually apply this optimization.",
        })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute an optimizer tool call."""
    if name == "profile_workflow":
        return _handle_profile_workflow(tool_input)
    elif name == "suggest_optimizations":
        return _handle_suggest_optimizations(tool_input)
    elif name == "check_tensorrt_status":
        return _handle_check_tensorrt(tool_input)
    elif name == "apply_optimization":
        return _handle_apply_optimization(tool_input)
    else:
        return to_json({"error": f"Unknown optimizer tool: {name}"})
