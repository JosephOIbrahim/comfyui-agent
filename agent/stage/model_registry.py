"""Model registry — track models in the USD stage under /models/.

Each model is a prim under /models/{model_type}/{safe_name} with attributes
tracking its status, filesystem path, hash, and source URL.

Status lifecycle:
  available   → Known to exist (CivitAI, HuggingFace, registry) but not on disk
  downloading → Download in progress
  materialized → On disk, verified (or verification skipped)
  failed      → Download or verification failed

The registry is the source of truth for what the Provisioner has done.
The filesystem is the source of truth for what's actually on disk.
reconcile() cross-references the two.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .cognitive_stage import CognitiveWorkflowStage, StageError

# Valid model types (matches ComfyUI directory structure)
MODEL_TYPES = (
    "checkpoints",
    "clip",
    "clip_vision",
    "controlnet",
    "diffusion_models",
    "embeddings",
    "gligen",
    "hypernetworks",
    "loras",
    "style_models",
    "text_encoders",
    "unet",
    "upscale_models",
    "vae",
    "vae_approx",
)

VALID_STATUSES = ("available", "downloading", "materialized", "failed")


def _safe_name(name: str) -> str:
    """Convert a model filename to a valid USD prim name."""
    # Strip extension, replace non-alphanumeric with underscore
    stem = Path(name).stem
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", stem)
    # Ensure doesn't start with digit
    if safe and safe[0].isdigit():
        safe = f"m_{safe}"
    return safe or "unnamed"


def register_model(
    cws: CognitiveWorkflowStage,
    model_type: str,
    filename: str,
    *,
    status: str = "available",
    source_url: str = "",
    sha256: str = "",
    file_path: str = "",
    size_bytes: int = 0,
    base_model: str = "",
    description: str = "",
) -> str:
    """Register a model in the USD stage.

    Args:
        cws: CognitiveWorkflowStage instance.
        model_type: Model category (e.g., "checkpoints", "loras").
        filename: Model filename (e.g., "v1-5-pruned-emaonly.safetensors").
        status: One of: available, downloading, materialized, failed.
        source_url: Download URL (CivitAI, HuggingFace, etc.).
        sha256: Expected SHA256 hash for verification.
        file_path: Absolute path on disk (if materialized).
        size_bytes: File size in bytes.
        base_model: Model family (e.g., "SD 1.5", "SDXL", "Flux").
        description: Human-readable description.

    Returns:
        USD prim path of the registered model.
    """
    if model_type not in MODEL_TYPES:
        raise StageError(
            f"Unknown model type '{model_type}'. "
            f"Valid types: {', '.join(MODEL_TYPES)}"
        )
    if status not in VALID_STATUSES:
        raise StageError(
            f"Invalid status '{status}'. "
            f"Valid: {', '.join(VALID_STATUSES)}"
        )

    safe = _safe_name(filename)
    prim_path = f"/models/{model_type}/{safe}"

    cws.write(prim_path, "filename", filename)
    cws.write(prim_path, "status", status)
    cws.write(prim_path, "model_type", model_type)

    if source_url:
        cws.write(prim_path, "source_url", source_url)
    if sha256:
        cws.write(prim_path, "sha256", sha256)
    if file_path:
        cws.write(prim_path, "file_path", file_path)
    if size_bytes:
        cws.write(prim_path, "size_bytes", size_bytes)
    if base_model:
        cws.write(prim_path, "base_model", base_model)
    if description:
        cws.write(prim_path, "description", description)

    return prim_path


def update_status(
    cws: CognitiveWorkflowStage,
    prim_path: str,
    status: str,
    **extra_attrs: Any,
) -> None:
    """Update a model's status and optional extra attributes.

    Args:
        cws: CognitiveWorkflowStage instance.
        prim_path: USD prim path of the model.
        status: New status.
        **extra_attrs: Additional attributes to set (e.g., file_path, sha256).
    """
    if status not in VALID_STATUSES:
        raise StageError(f"Invalid status '{status}'")
    if not cws.prim_exists(prim_path):
        raise StageError(f"Model not registered: {prim_path}")

    cws.write(prim_path, "status", status)
    for key, value in extra_attrs.items():
        if isinstance(value, (str, int, float, bool)):
            cws.write(prim_path, key, value)


def get_model(
    cws: CognitiveWorkflowStage, prim_path: str
) -> dict[str, Any] | None:
    """Get all attributes of a registered model.

    Returns:
        Dict of model attributes, or None if not registered.
    """
    if not cws.prim_exists(prim_path):
        return None
    return cws.get_prim_attrs(prim_path)


def list_models_by_type(
    cws: CognitiveWorkflowStage, model_type: str
) -> list[dict[str, Any]]:
    """List all registered models of a given type.

    Returns:
        List of model attribute dicts.
    """
    type_path = f"/models/{model_type}"
    if not cws.prim_exists(type_path):
        return []

    results = []
    for name in cws.list_children(type_path):
        prim_path = f"{type_path}/{name}"
        attrs = cws.get_prim_attrs(prim_path)
        if attrs:
            attrs["_prim_path"] = prim_path
            results.append(attrs)
    return results


def list_models_by_status(
    cws: CognitiveWorkflowStage, status: str
) -> list[dict[str, Any]]:
    """List all registered models with a given status across all types.

    Returns:
        List of model attribute dicts.
    """
    results = []
    for model_type in MODEL_TYPES:
        for model in list_models_by_type(cws, model_type):
            if model.get("status") == status:
                results.append(model)
    return results


def find_model(
    cws: CognitiveWorkflowStage, filename: str
) -> dict[str, Any] | None:
    """Find a model by filename across all types.

    Returns:
        Model attribute dict with _prim_path, or None.
    """
    for model_type in MODEL_TYPES:
        type_path = f"/models/{model_type}"
        for name in cws.list_children(type_path):
            prim_path = f"{type_path}/{name}"
            stored_name = cws.read(prim_path, "filename")
            if stored_name == filename:
                attrs = cws.get_prim_attrs(prim_path)
                attrs["_prim_path"] = prim_path
                return attrs
    return None


def reconcile(
    cws: CognitiveWorkflowStage, models_dir: Path
) -> dict[str, list[str]]:
    """Cross-reference registry with filesystem.

    Finds:
    - registered_missing: In registry as materialized, but not on disk
    - unregistered_on_disk: On disk, but not in registry
    - status_mismatch: Status doesn't match filesystem reality

    Args:
        cws: CognitiveWorkflowStage instance.
        models_dir: Path to the ComfyUI models directory.

    Returns:
        Dict with lists of discrepancies.
    """
    model_extensions = {
        ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"
    }

    registered_missing = []
    status_mismatch = []
    registered_files: set[str] = set()

    for model_type in MODEL_TYPES:
        for model in list_models_by_type(cws, model_type):
            filename = model.get("filename", "")
            status = model.get("status", "")
            prim_path = model.get("_prim_path", "")

            if not filename:
                continue

            registered_files.add(f"{model_type}/{filename}")

            # Check if materialized model exists on disk
            file_on_disk = (models_dir / model_type / filename).exists()

            if status == "materialized" and not file_on_disk:
                registered_missing.append(prim_path)
            elif status == "available" and file_on_disk:
                status_mismatch.append(prim_path)

    # Find unregistered files on disk
    unregistered = []
    for model_type in MODEL_TYPES:
        type_dir = models_dir / model_type
        if not type_dir.exists():
            continue
        for f in type_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in model_extensions:
                rel = f"{model_type}/{f.relative_to(type_dir)}"
                if rel not in registered_files:
                    unregistered.append(str(f))

    return {
        "registered_missing": registered_missing,
        "unregistered_on_disk": unregistered,
        "status_mismatch": status_mismatch,
    }
