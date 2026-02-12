"""Vision module — gives the agent eyes.

Uses Claude Vision API to analyze generated images, compare outputs,
and suggest parameter improvements. Vision calls use a separate API call
to keep images out of the main conversation context window.

Also provides fast perceptual hash comparison via Pillow for instant
A/B regression testing without API calls.
"""

import base64
import json
import logging
from pathlib import Path

import anthropic

from ..config import AGENT_MODEL
from ..rate_limiter import VISION_LIMITER
from ..tools._util import to_json
from ._protocol import brain_message

log = logging.getLogger(__name__)

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# Vision analysis uses a smaller max_tokens since responses are structured
_VISION_MAX_TOKENS = 4096
_VISION_TIMEOUT = 120  # seconds — max wait for Vision API response

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
        "name": "hash_compare_images",
        "description": (
            "Fast pixel-level comparison of two images using perceptual hashing. "
            "Returns similarity score, pixel difference percentage, and average "
            "color delta. Instant (no API call) — use as pre-check before "
            "compare_outputs to detect identical or near-identical images."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_a": {
                    "type": "string",
                    "description": "Path to the first image.",
                },
                "image_b": {
                    "type": "string",
                    "description": "Path to the second image.",
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
    from ..tools._util import validate_path
    path_err = validate_path(path, must_exist=True)
    if path_err:
        raise FileNotFoundError(path_err)
    p = Path(path)

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
    if not VISION_LIMITER().acquire(timeout=10.0):
        return to_json({"error": "Rate limited — too many Vision API calls. Try again shortly."})

    try:
        client = anthropic.Anthropic(timeout=_VISION_TIMEOUT)
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
    except Exception as e:
        # Catch httpx.TimeoutException and other transport errors
        log.error("Vision API transport error: %s", e)
        return to_json({"error": f"Vision API timeout or transport error: {e}"})


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

    # Emit BrainMessage so memory module can track vision assessments
    msg = brain_message(
        source="vision",
        target="memory",
        msg_type="result",
        payload={
            "action": "image_analyzed",
            "image_path": image_path,
            "quality_score": result.get("quality_score"),
            "prompt_adherence": result.get("prompt_adherence"),
        },
    )
    log.debug("BrainMessage: vision->memory: %s", msg["correlation_id"])

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

    # Emit BrainMessage for A/B comparison tracking
    msg = brain_message(
        source="vision",
        target="memory",
        msg_type="result",
        payload={
            "action": "images_compared",
            "improved": result.get("improved"),
            "quality_delta": result.get("quality_delta"),
        },
    )
    log.debug("BrainMessage: vision->memory comparison: %s", msg["correlation_id"])

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


def _compute_average_hash(img, hash_size: int = 8) -> int:
    """Compute average hash (aHash) — resize to hash_size, compare to mean."""
    resized = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(resized.getdata())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p > mean:
            bits |= 1 << i
    return bits


def _hamming_distance(h1: int, h2: int) -> int:
    """Count differing bits between two hashes."""
    x = h1 ^ h2
    count = 0
    while x:
        count += 1
        x &= x - 1
    return count


def _pixel_diff(img_a, img_b) -> dict:
    """Compare pixels between two images. Returns diff stats."""
    # Resize both to same dimensions for comparison
    size = (min(img_a.width, img_b.width), min(img_a.height, img_b.height))
    a = img_a.resize(size).convert("RGB")
    b = img_b.resize(size).convert("RGB")

    pixels_a = list(a.getdata())
    pixels_b = list(b.getdata())
    total = len(pixels_a)

    if total == 0:
        return {"diff_pixels": 0, "diff_pct": 0.0, "avg_color_delta": 0.0}

    diff_count = 0
    total_delta = 0.0
    threshold = 10  # Per-channel difference threshold to count as "different"

    for pa, pb in zip(pixels_a, pixels_b):
        channel_diff = sum(abs(ca - cb) for ca, cb in zip(pa, pb))
        avg_diff = channel_diff / 3
        if avg_diff > threshold:
            diff_count += 1
        total_delta += avg_diff

    return {
        "diff_pixels": diff_count,
        "diff_pct": round(diff_count / total * 100, 2),
        "avg_color_delta": round(total_delta / total, 2),
    }


def _handle_hash_compare(tool_input: dict) -> str:
    image_a = tool_input["image_a"]
    image_b = tool_input["image_b"]

    if not _HAS_PIL:
        return to_json({
            "error": "Pillow not installed. Install with: pip install Pillow",
            "fallback": "Use compare_outputs for Vision API comparison.",
        })

    path_a = Path(image_a)
    path_b = Path(image_b)

    if not path_a.exists():
        return to_json({"error": f"Image not found: {image_a}"})
    if not path_b.exists():
        return to_json({"error": f"Image not found: {image_b}"})

    try:
        img_a = Image.open(path_a)
        img_b = Image.open(path_b)
    except Exception as e:
        return to_json({"error": f"Failed to open images: {e}"})

    # Compute average hash
    hash_a = _compute_average_hash(img_a)
    hash_b = _compute_average_hash(img_b)
    hamming = _hamming_distance(hash_a, hash_b)
    hash_similarity = round(1.0 - hamming / 64, 4)  # 64 bits for 8x8 hash

    # Pixel-level diff
    diff = _pixel_diff(img_a, img_b)

    # Classification
    if hamming == 0 and diff["diff_pct"] == 0:
        verdict = "identical"
    elif hamming <= 2 and diff["diff_pct"] < 1.0:
        verdict = "near_identical"
    elif hamming <= 8 and diff["diff_pct"] < 10.0:
        verdict = "similar"
    elif hamming <= 16:
        verdict = "different"
    else:
        verdict = "very_different"

    return to_json({
        "verdict": verdict,
        "hash_similarity": hash_similarity,
        "hamming_distance": hamming,
        "pixel_diff_pct": diff["diff_pct"],
        "avg_color_delta": diff["avg_color_delta"],
        "diff_pixels": diff["diff_pixels"],
        "resolution_a": f"{img_a.width}x{img_a.height}",
        "resolution_b": f"{img_b.width}x{img_b.height}",
        "image_a": image_a,
        "image_b": image_b,
    })


def handle(name: str, tool_input: dict) -> str:
    """Execute a vision tool call."""
    if name == "analyze_image":
        return _handle_analyze_image(tool_input)
    elif name == "compare_outputs":
        return _handle_compare_outputs(tool_input)
    elif name == "hash_compare_images":
        return _handle_hash_compare(tool_input)
    elif name == "suggest_improvements":
        return _handle_suggest_improvements(tool_input)
    else:
        return to_json({"error": f"Unknown vision tool: {name}"})
