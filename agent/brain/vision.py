"""Vision module — gives the agent eyes.

Uses Claude Vision API to analyze generated images, compare outputs,
and suggest parameter improvements. Vision calls use a separate API call
to keep images out of the main conversation context window.
"""

import base64
import json
import logging
from pathlib import Path

import anthropic

from ..config import AGENT_MODEL
from ..tools._util import to_json

log = logging.getLogger(__name__)

# Vision analysis uses a smaller max_tokens since responses are structured
_VISION_MAX_TOKENS = 4096

TOOLS: list[dict] = [
    {
        "name": "analyze_image",
        "description": (
            "Analyze a generated image using Claude Vision. Returns structured "
            "assessment: quality score (0-1), detected artifacts, composition notes, "
            "prompt adherence, and improvement suggestions. Use after execute_workflow "
            "to evaluate output quality."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the image file to analyze.",
                },
                "prompt_used": {
                    "type": "string",
                    "description": (
                        "The text prompt used to generate the image. "
                        "Helps assess prompt adherence."
                    ),
                },
                "workflow_context": {
                    "type": "string",
                    "description": (
                        "Brief description of the workflow (e.g., 'SDXL txt2img, "
                        "20 steps, DPM++ 2M Karras, CFG 7.0'). Optional."
                    ),
                },
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "compare_outputs",
        "description": (
            "Compare two images (same-seed A/B test). Returns what changed, "
            "whether it improved, and specific differences. Use after modifying "
            "a workflow to verify the change had the intended effect."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_a": {
                    "type": "string",
                    "description": "Path to the 'before' image.",
                },
                "image_b": {
                    "type": "string",
                    "description": "Path to the 'after' image.",
                },
                "change_description": {
                    "type": "string",
                    "description": (
                        "What was changed between A and B "
                        "(e.g., 'increased CFG from 7 to 12')."
                    ),
                },
            },
            "required": ["image_a", "image_b"],
        },
    },
    {
        "name": "suggest_improvements",
        "description": (
            "Given an image and the workflow that produced it, suggest specific "
            "parameter tweaks to improve quality. Returns actionable suggestions "
            "with expected impact."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image to improve.",
                },
                "workflow_summary": {
                    "type": "string",
                    "description": (
                        "Summary of current workflow parameters: model, steps, "
                        "CFG, sampler, resolution, etc."
                    ),
                },
                "goal": {
                    "type": "string",
                    "description": (
                        "What the artist wants (e.g., 'more detail in faces', "
                        "'reduce noise in background', 'match reference style')."
                    ),
                },
            },
            "required": ["image_path", "workflow_summary"],
        },
    },
]


def _read_image_as_base64(path: str) -> tuple[str, str]:
    """Read an image file and return (base64_data, media_type)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = p.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "image/png")
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return data, media_type


def _call_vision(system_prompt: str, user_content: list) -> str:
    """Make a separate Claude Vision API call. Returns the text response."""
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=AGENT_MODEL,
            max_tokens=_VISION_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        # Extract text from response
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""
    except anthropic.APIError as e:
        log.error("Vision API error: %s", e)
        return to_json({"error": f"Vision API error: {e}"})


def _handle_analyze_image(tool_input: dict) -> str:
    image_path = tool_input["image_path"]
    prompt_used = tool_input.get("prompt_used", "")
    workflow_context = tool_input.get("workflow_context", "")

    try:
        img_data, media_type = _read_image_as_base64(image_path)
    except FileNotFoundError as e:
        return to_json({"error": str(e)})
    except Exception as e:
        return to_json({"error": f"Failed to read image: {e}"})

    system = (
        "You are an expert image quality analyst for AI-generated images. "
        "Analyze the image and return a JSON object with these fields:\n"
        '- "quality_score": float 0-1 (overall quality)\n'
        '- "artifacts": list of detected issues (banding, noise, anatomy errors, '
        "color shifts, blurriness, etc.)\n"
        '- "composition": brief composition assessment\n'
        '- "prompt_adherence": float 0-1 if prompt provided, null otherwise\n'
        '- "strengths": list of what looks good\n'
        '- "suggestions": list of actionable improvement ideas\n'
        "Return ONLY valid JSON, no markdown formatting."
    )

    user_content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": img_data},
        },
        {
            "type": "text",
            "text": (
                f"Analyze this AI-generated image.\n"
                f"Prompt used: {prompt_used or 'not provided'}\n"
                f"Workflow: {workflow_context or 'not provided'}\n"
                f"Return structured JSON analysis."
            ),
        },
    ]

    raw = _call_vision(system, user_content)

    # Try to parse as JSON, wrap if needed
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"raw_analysis": raw, "parse_error": "Vision response was not valid JSON"}

    result["image_path"] = image_path
    return to_json(result)


def _handle_compare_outputs(tool_input: dict) -> str:
    image_a = tool_input["image_a"]
    image_b = tool_input["image_b"]
    change_desc = tool_input.get("change_description", "")

    try:
        data_a, type_a = _read_image_as_base64(image_a)
        data_b, type_b = _read_image_as_base64(image_b)
    except FileNotFoundError as e:
        return to_json({"error": str(e)})
    except Exception as e:
        return to_json({"error": f"Failed to read images: {e}"})

    system = (
        "You are comparing two AI-generated images (A = before, B = after a change). "
        "Return a JSON object with:\n"
        '- "improved": bool (overall, did B improve over A?)\n'
        '- "differences": list of specific changes observed\n'
        '- "quality_delta": float -1 to +1 (negative = worse, positive = better)\n'
        '- "recommendation": one-sentence verdict\n'
        "Return ONLY valid JSON, no markdown formatting."
    )

    user_content = [
        {"type": "text", "text": "Image A (before):"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": type_a, "data": data_a},
        },
        {"type": "text", "text": "Image B (after):"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": type_b, "data": data_b},
        },
        {
            "type": "text",
            "text": (
                f"Change made: {change_desc or 'not specified'}\n"
                f"Compare these images and return structured JSON analysis."
            ),
        },
    ]

    raw = _call_vision(system, user_content)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"raw_analysis": raw, "parse_error": "Vision response was not valid JSON"}

    result["image_a"] = image_a
    result["image_b"] = image_b
    return to_json(result)


def _handle_suggest_improvements(tool_input: dict) -> str:
    image_path = tool_input["image_path"]
    workflow_summary = tool_input["workflow_summary"]
    goal = tool_input.get("goal", "")

    try:
        img_data, media_type = _read_image_as_base64(image_path)
    except FileNotFoundError as e:
        return to_json({"error": str(e)})
    except Exception as e:
        return to_json({"error": f"Failed to read image: {e}"})

    system = (
        "You are an expert ComfyUI workflow optimizer. Given an image and the "
        "workflow that produced it, suggest specific parameter changes. "
        "Return a JSON object with:\n"
        '- "suggestions": list of objects, each with:\n'
        '  - "parameter": what to change (e.g., "steps", "cfg", "sampler")\n'
        '  - "current_value": current setting\n'
        '  - "suggested_value": recommended change\n'
        '  - "reason": why this helps\n'
        '  - "expected_impact": what will change (quality, speed, etc.)\n'
        '  - "confidence": float 0-1\n'
        '- "priority_order": list of parameter names in order of impact\n'
        "Return ONLY valid JSON, no markdown formatting."
    )

    user_content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": img_data},
        },
        {
            "type": "text",
            "text": (
                f"Current workflow: {workflow_summary}\n"
                f"Artist goal: {goal or 'general improvement'}\n"
                f"Suggest specific parameter changes to improve this output."
            ),
        },
    ]

    raw = _call_vision(system, user_content)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"raw_analysis": raw, "parse_error": "Vision response was not valid JSON"}

    result["image_path"] = image_path
    return to_json(result)


def handle(name: str, tool_input: dict) -> str:
    """Execute a vision tool call."""
    if name == "analyze_image":
        return _handle_analyze_image(tool_input)
    elif name == "compare_outputs":
        return _handle_compare_outputs(tool_input)
    elif name == "suggest_improvements":
        return _handle_suggest_improvements(tool_input)
    else:
        return to_json({"error": f"Unknown vision tool: {name}"})
