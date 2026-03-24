"""Edge case generators for adversarial testing of workflows.

Produces deliberately malformed or pathological workflow dicts in the
ComfyUI API format::

    {"node_id": {"class_type": "NodeName", "inputs": {...}}}

Every function is pure — no side effects, no external dependencies.
"""

from __future__ import annotations


def empty_workflow() -> dict:
    """Workflow with zero nodes."""
    return {}


def disconnected_nodes() -> dict:
    """Three nodes with no connections between them.

    A CheckpointLoader, a CLIPTextEncode with a literal (not linked)
    clip value, and a SaveImage with no image input.
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "v1-5-pruned-emaonly.safetensors",
            },
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "a cat",
                "clip": "NOT_A_LINK",
            },
        },
        "3": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "ComfyUI",
            },
        },
    }


def cycle_workflow() -> dict:
    """Circular dependency: A -> B -> C -> A.

    Node 1 takes input from node 3, node 2 from node 1, node 3 from
    node 2 — forming a cycle that should be rejected by any DAG
    validator.
    """
    return {
        "1": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "seed": 42,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "2": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["1", 0],
                "vae": ["1", 2],
            },
        },
        "3": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["2", 0],
                "vae": ["2", 1],
            },
        },
    }


def type_mismatch_workflow() -> dict:
    """IMAGE output (SaveImage slot 0) connected to MODEL input (KSampler).

    SaveImage does not produce a MODEL output, so this is a type
    mismatch that should be caught by type-aware validation.
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "v1-5-pruned-emaonly.safetensors",
            },
        },
        "2": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": ["1", 0],
            },
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "seed": 42,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
    }


def missing_node_type() -> dict:
    """References a class_type that does not exist in any ComfyUI install."""
    return {
        "1": {
            "class_type": "TotallyFakeNodeThatDoesNotExist_XYZ9999",
            "inputs": {
                "value": 42,
            },
        },
        "2": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": ["1", 0],
            },
        },
    }


def oversized_workflow(n: int = 1000) -> dict:
    """Chain of *n* KSampler nodes for stress testing.

    Each node (except the first) takes its model input from the
    previous node, forming a long linear chain.
    """
    if n < 1:
        n = 1
    workflow: dict = {}
    for i in range(1, n + 1):
        node_id = str(i)
        inputs: dict = {
            "seed": i,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
        }
        if i > 1:
            inputs["model"] = [str(i - 1), 0]
        workflow[node_id] = {
            "class_type": "KSampler",
            "inputs": inputs,
        }
    return workflow


def mixed_model_family() -> dict:
    """SD1.5 checkpoint loaded alongside an SDXL LoRA.

    Mixing model families is a common artist mistake that produces
    garbage output or crashes.
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "v1-5-pruned-emaonly.safetensors",
            },
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": "sdxl_offset_lora.safetensors",
                "strength_model": 1.0,
                "strength_clip": 1.0,
                "model": ["1", 0],
                "clip": ["1", 1],
            },
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "seed": 42,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
    }


def invalid_cfg_range() -> dict:
    """KSampler with CFG = 500, far outside any sane range."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "v1-5-pruned-emaonly.safetensors",
            },
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "a cat",
                "clip": ["1", 1],
            },
        },
        "3": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1,
            },
        },
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["2", 0],
                "latent_image": ["3", 0],
                "seed": 42,
                "steps": 20,
                "cfg": 500,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
    }


def negative_dimensions() -> dict:
    """EmptyLatentImage with width and height of -512."""
    return {
        "1": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": -512,
                "height": -512,
                "batch_size": 1,
            },
        },
        "2": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": ["1", 0],
            },
        },
    }


def empty_prompt() -> dict:
    """CLIPTextEncode with an empty string prompt."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "v1-5-pruned-emaonly.safetensors",
            },
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "",
                "clip": ["1", 1],
            },
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "seed": 42,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
    }


# ---------------------------------------------------------------------------
# Convenience: all generators in a single dict for parametrized testing
# ---------------------------------------------------------------------------

ALL_GENERATORS: dict[str, callable] = {
    "empty_workflow": empty_workflow,
    "disconnected_nodes": disconnected_nodes,
    "cycle_workflow": cycle_workflow,
    "type_mismatch_workflow": type_mismatch_workflow,
    "missing_node_type": missing_node_type,
    "oversized_workflow": oversized_workflow,
    "mixed_model_family": mixed_model_family,
    "invalid_cfg_range": invalid_cfg_range,
    "negative_dimensions": negative_dimensions,
    "empty_prompt": empty_prompt,
}
