"""Anchor parameter immunity — constitutional protection for critical workflow parameters.

Anchor parameters are structurally immune to agent modification. Writing to an
anchor raises AnchorViolationError. Hard stop, not a soft warning.
"""


class AnchorViolationError(Exception):
    """Raised when an agent attempts to modify a constitutionally protected parameter."""

    def __init__(self, node_type: str, param_name: str):
        self.node_type = node_type
        self.param_name = param_name
        super().__init__(
            f"Constitutional violation: '{param_name}' on '{node_type}' is an anchor "
            f"parameter. Anchor parameters require human approval to modify."
        )


# Parameters that agents cannot modify without human gate.
# Protects model selection, resolution, and safety-critical settings.
ANCHOR_PARAMS: dict[str, list[str]] = {
    "CheckpointLoaderSimple": ["ckpt_name"],
    "UNETLoader": ["unet_name"],
    "DualCLIPLoader": ["clip_name1", "clip_name2"],
    "TripleCLIPLoader": ["clip_name1", "clip_name2", "clip_name3"],
    "CLIPLoader": ["clip_name"],
    "VAELoader": ["vae_name"],
    "LoraLoader": ["lora_name"],
    "LoraLoaderModelOnly": ["lora_name"],
    "EmptyLatentImage": ["width", "height"],
    "EmptySD3LatentImage": ["width", "height"],
}


def is_anchor(node_type: str, param_name: str) -> bool:
    """Returns True if this parameter is constitutionally protected."""
    return param_name in ANCHOR_PARAMS.get(node_type, [])


def check_anchor(node_type: str, param_name: str) -> None:
    """Raise AnchorViolationError if the parameter is protected."""
    if is_anchor(node_type, param_name):
        raise AnchorViolationError(node_type, param_name)
