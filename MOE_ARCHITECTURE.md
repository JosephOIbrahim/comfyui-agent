# MoE Agent Architecture: Implementation Guide

> **Paste this document into a Claude Code session to guide implementation.**
> **Read CLAUDE.md first for project context, then use this as the architectural blueprint.**

---

## Executive Summary

Evolve the comfyui-agent from a single monolithic agent with 62 tools into a Mixture of Experts (MoE) architecture where a lightweight Router delegates domain authority to three specialist agents — Intent, Execution, and Verify — each with model-awareness baked in via a shared Model Profile Registry.

**The critical innovation:** Model Communication Profiles. Different models (Flux, SDXL, LoRA checkpoints, video models) require fundamentally different prompting strategies, parameter ranges, and quality criteria. The profiles are the ACCESS oracle for model-specific knowledge — agents query them at runtime instead of memorizing model quirks.

**The second innovation:** Agent Output Schemas. Each specialist agent can define, expose, and generate schemas for its output. A dedicated `SchemaGenerator` node creates schemas from example outputs — users can customize what Intent produces, what Verify returns, and how the pipeline communicates. This makes the agent pipeline user-shapeable, not locked to developer-defined contracts.

**What changes:** Tool calls are mediated by specialist agents. Quality judgment becomes model-relative. The iterative_refine loop gets formal structure. Agent output contracts become user-configurable via schemas.

**What doesn't change:** The 62 existing tools stay where they are. The four-layer taxonomy (UNDERSTAND, DISCOVER, PILOT, VERIFY) stays. The HTTP/WebSocket interface stays. RFC6902 patching stays.

---

## Architecture Diagram

```
                    ┌─────────────┐
                    │   ROUTER    │
                    │ (sequencer) │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌───▼────┐ ┌────▼─────┐
        │  INTENT   │ │ EXEC   │ │  VERIFY  │
        │  Agent    │ │ Agent  │ │  Agent   │
        │ [schema]  │ │[schema]│ │ [schema] │
        └─────┬─────┘ └───┬────┘ └────┬─────┘
              │            │            │
    ┌─────────▼────────────▼────────────▼─────────┐
    │          EXISTING TOOL LAYERS                │
    │                                              │
    │  UNDERSTAND    DISCOVER    PILOT    VERIFY   │
    │  (15 tools)   (12 tools) (20 tools)(15 tools)│
    └──────────────────────────────────────────────┘
              │                                    │
    ┌─────────▼────────────────────────────────────┤
    │          MODEL PROFILE REGISTRY              │
    │          profiles/*.yaml                     │
    ├──────────────────────────────────────────────┤
    │          SCHEMA REGISTRY                     │
    │   schemas/{agent}/default.yaml + custom/     │
    └──────────────────────────────────────────────┘
```

### Tool Ownership

| Specialist Agent | Primarily Uses | Also Reads From |
|---|---|---|
| Intent Agent | No tools directly — pure reasoning layer | UNDERSTAND (current workflow state) |
| Execution Agent | PILOT tools (workflow mutation, execution) | UNDERSTAND (inspect before mutating) |
| Verify Agent | VERIFY tools (vision analysis, execution verification) | UNDERSTAND (read outputs) |
| Router | None directly — delegates everything | DISCOVER (if model/node lookup needed) |

---

## Agent Output Schemas & Schema Generation

### The Problem

The default agent output contracts (`IntentSpecification`, `VerificationResult`) encode the developer's assumptions about what information matters. But artists have different workflows, different priorities, different downstream consumers. One artist wants the Intent Agent to also output mood-board keywords for reference. Another wants the Verify Agent to score composition separately from technical quality. A pipeline TD wants the Execution Agent to emit timing telemetry.

Hardcoding these contracts means every customization is a code change. That's antithetical to the project's philosophy — the user shouldn't need to be a developer to shape how their co-pilot communicates.

### The Solution: Schema-Defined Agent Outputs

Each specialist agent has a **default output schema** (the dataclasses defined in Phases 2-4) but can also accept a **user-provided schema** that reshapes its output. A `SchemaGenerator` node creates new schemas from example outputs — the user shows the system what they want, and the system infers the structure.

```
┌──────────────────────────────────────────────────────────┐
│                    SCHEMA FLOW                           │
│                                                          │
│  Example Output ──→ SchemaGenerator Node ──→ Schema YAML │
│                                                          │
│  Schema YAML ──→ Agent ──→ Structured Output             │
│                    ↑                                     │
│              (validates against schema)                   │
└──────────────────────────────────────────────────────────┘
```

### Schema Structure

A schema defines: what fields an agent outputs, their types, which are required vs optional, and what each field means (so the agent can reason about populating it).

```yaml
# schemas/intent/default.yaml
# Default output schema for the Intent Agent
# Users can create custom schemas that extend or reshape this.

schema:
  name: "IntentSpecification"
  version: "1.0"
  agent: "intent"
  description: "Translates artistic intent into parameter mutations"

fields:
  model_id:
    type: "string"
    required: true
    description: "Active model identifier"

  parameter_mutations:
    type: "list[ParameterMutation]"
    required: true
    description: "Concrete parameter changes to apply"
    item_schema:
      target:
        type: "string"
        description: "Node.parameter path, e.g. 'KSampler.cfg'"
      action:
        type: "enum"
        values: ["set", "adjust_up", "adjust_down"]
      value:
        type: "any"
        description: "Concrete value for 'set' actions"
      magnitude:
        type: "enum"
        values: ["slight", "moderate", "large"]
        required: false
        description: "For adjust actions — how much to change"
      reason:
        type: "string"
        description: "Why this change, traceable to profile or intent"

  prompt_mutations:
    type: "list[PromptMutation]"
    required: true
    description: "Changes to prompt text"
    item_schema:
      target:
        type: "enum"
        values: ["positive_prompt", "negative_prompt"]
      action:
        type: "enum"
        values: ["append", "prepend", "replace", "remove"]
      value:
        type: "string"
      reason:
        type: "string"

  confidence:
    type: "float"
    range: [0.0, 1.0]
    required: true
    description: "How confident the agent is in this translation"

  conflicts_resolved:
    type: "list[string]"
    required: false
    description: "Explanations of how competing intents were reconciled"

  warnings:
    type: "list[string]"
    required: false
    description: "Issues the user should be aware of"

  # ---- EXTENSION POINT ----
  # Users can add custom fields below this line.
  # The agent will attempt to populate any field that has a description.
```

### User-Custom Schema Example

An artist who works heavily with LoRA stacking might want the Intent Agent to also output LoRA-specific reasoning:

```yaml
# schemas/intent/custom/lora_workflow.yaml
# Custom Intent schema for LoRA-heavy workflows

schema:
  name: "LoRAIntentSpecification"
  version: "1.0"
  agent: "intent"
  extends: "default"               # inherits all default fields
  description: "Intent spec with LoRA-specific reasoning"

# All fields from default are inherited. These are ADDITIONS:
fields:
  lora_reasoning:
    type: "object"
    required: false
    description: "How this intent affects active LoRAs"
    fields:
      affected_loras:
        type: "list[string]"
        description: "Which LoRAs are impacted by this intent change"
      strength_adjustments:
        type: "list[object]"
        description: "Recommended LoRA strength changes"
        item_schema:
          lora_name:
            type: "string"
          current_strength:
            type: "float"
          recommended_strength:
            type: "float"
          reason:
            type: "string"
      interaction_warnings:
        type: "list[string]"
        description: "Known LoRA interaction issues relevant to this intent"

  mood_keywords:
    type: "list[string]"
    required: false
    description: "Mood-board reference keywords that capture the artistic intent"
```

### SchemaGenerator Node

The `SchemaGenerator` is a ComfyUI custom node and a standalone tool that infers a schema from an example output. The user provides a sample of what they want the agent to produce, and the node generates the YAML schema.

```python
# src/schemas/generator.py

"""
SchemaGenerator — infer output schemas from examples.

The user shows the system what they want an agent to output.
The generator infers the structure, types, and descriptions.

This can operate as:
  1. A ComfyUI custom node (for visual workflow integration)
  2. A CLI tool (for pipeline TDs)
  3. An agent tool (for in-conversation schema creation)
"""

from dataclasses import dataclass, field
from typing import Any
import yaml


@dataclass
class InferredField:
    """A field inferred from an example value."""
    name: str
    type: str                          # inferred from Python type
    required: bool = True
    description: str = ""              # LLM-generated from context
    example_value: Any = None
    nested_fields: list["InferredField"] = field(default_factory=list)


def infer_schema_from_example(
    example: dict,
    agent: str,
    schema_name: str,
    extends: str | None = "default",
) -> dict:
    """
    Infer a YAML schema from an example output dict.

    Args:
        example: A sample output that represents what the user wants
        agent: Which agent this schema is for ("intent", "execution", "verify")
        schema_name: Human-readable name for the schema
        extends: Base schema to extend (None for standalone)

    Returns:
        Schema dict ready to be written as YAML

    Example:
        >>> example = {
        ...     "parameter_mutations": [...],
        ...     "mood_keywords": ["ethereal", "warm"],
        ...     "lora_reasoning": {
        ...         "affected_loras": ["detail_enhancer"],
        ...         "strength_adjustments": [...]
        ...     }
        ... }
        >>> schema = infer_schema_from_example(example, "intent", "MyCustomIntent")
        >>> # Returns schema YAML with mood_keywords and lora_reasoning as new fields
    """
    fields = {}
    for key, value in example.items():
        fields[key] = _infer_field(key, value)

    schema = {
        "schema": {
            "name": schema_name,
            "version": "1.0",
            "agent": agent,
            "description": f"Custom {agent} output schema",
        },
        "fields": fields,
    }

    if extends:
        schema["schema"]["extends"] = extends

    return schema


def _infer_field(name: str, value: Any) -> dict:
    """Infer field definition from a name-value pair."""
    field_def = {"required": True, "description": f"[auto-inferred from example]"}

    if isinstance(value, str):
        field_def["type"] = "string"
        field_def["example"] = value
    elif isinstance(value, bool):
        field_def["type"] = "boolean"
    elif isinstance(value, int):
        field_def["type"] = "integer"
    elif isinstance(value, float):
        field_def["type"] = "float"
        if 0.0 <= value <= 1.0:
            field_def["range"] = [0.0, 1.0]
    elif isinstance(value, list):
        field_def["type"] = f"list"
        if value:
            if isinstance(value[0], dict):
                field_def["item_schema"] = {
                    k: _infer_field(k, v) for k, v in value[0].items()
                }
            elif isinstance(value[0], str):
                field_def["type"] = "list[string]"
                field_def["example"] = value[:3]
            else:
                field_def["type"] = f"list[{type(value[0]).__name__}]"
    elif isinstance(value, dict):
        field_def["type"] = "object"
        field_def["fields"] = {
            k: _infer_field(k, v) for k, v in value.items()
        }
    else:
        field_def["type"] = "any"

    return field_def


def write_schema(schema: dict, path: str) -> None:
    """Write schema to YAML file."""
    with open(path, "w") as f:
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)


# ============================================================
# ComfyUI Node Interface
# ============================================================

class SchemaGeneratorNode:
    """
    ComfyUI custom node that generates agent output schemas from examples.

    INPUTS:
      - example_output: JSON string of an example agent output
      - agent_type: enum ["intent", "execution", "verify"]
      - schema_name: string name for the schema
      - extends_default: boolean — inherit from default schema?

    OUTPUTS:
      - schema_yaml: the generated schema as YAML string
      - schema_path: path where the schema was saved
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "example_output": ("STRING", {"multiline": True}),
                "agent_type": (["intent", "execution", "verify"],),
                "schema_name": ("STRING", {"default": "CustomSchema"}),
            },
            "optional": {
                "extends_default": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("schema_yaml", "schema_path")
    FUNCTION = "generate"
    CATEGORY = "comfyui-agent/schemas"

    def generate(self, example_output, agent_type, schema_name, extends_default=True):
        import json
        example = json.loads(example_output)
        extends = "default" if extends_default else None
        schema = infer_schema_from_example(example, agent_type, schema_name, extends)
        schema_yaml = yaml.dump(schema, default_flow_style=False, sort_keys=False)

        # Save to schema registry
        schema_dir = f"schemas/{agent_type}/custom"
        safe_name = schema_name.lower().replace(" ", "_")
        schema_path = f"{schema_dir}/{safe_name}.yaml"
        # (actual file write handled by agent infrastructure)

        return (schema_yaml, schema_path)
```

### Schema Loader & Validation

```python
# src/schemas/loader.py

"""
Schema Registry — loads and validates agent output schemas.

Each agent has a default schema. Users can override with custom schemas.
The loader resolves inheritance (extends), merges fields, and provides
validation for agent outputs.
"""

from pathlib import Path
from typing import Any
import yaml

SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"
_cache: dict[str, dict] = {}


def load_schema(agent: str, schema_name: str = "default") -> dict:
    """
    Load a schema for a given agent.

    Resolution order:
      1. schemas/{agent}/custom/{schema_name}.yaml
      2. schemas/{agent}/{schema_name}.yaml
      3. schemas/{agent}/default.yaml (fallback)

    If the schema has 'extends', fields from the base are merged in.
    """
    cache_key = f"{agent}:{schema_name}"
    if cache_key in _cache:
        return _cache[cache_key]

    schema = _resolve_schema(agent, schema_name)

    # Handle inheritance
    extends = schema.get("schema", {}).get("extends")
    if extends and extends != schema_name:
        base = load_schema(agent, extends)
        schema = _merge_schemas(base, schema)

    _cache[cache_key] = schema
    return schema


def _resolve_schema(agent: str, schema_name: str) -> dict:
    """Find and load a schema file."""
    # Try custom first
    custom_path = SCHEMAS_DIR / agent / "custom" / f"{schema_name}.yaml"
    if custom_path.exists():
        return _load_yaml(custom_path)

    # Then standard location
    standard_path = SCHEMAS_DIR / agent / f"{schema_name}.yaml"
    if standard_path.exists():
        return _load_yaml(standard_path)

    # Fallback to default
    default_path = SCHEMAS_DIR / agent / "default.yaml"
    if default_path.exists():
        return _load_yaml(default_path)

    raise FileNotFoundError(f"No schema found for agent={agent}, name={schema_name}")


def _merge_schemas(base: dict, extension: dict) -> dict:
    """Merge extension fields into base schema. Extension wins on conflict."""
    merged = {**base}
    merged["schema"] = {**base.get("schema", {}), **extension.get("schema", {})}
    merged["fields"] = {**base.get("fields", {}), **extension.get("fields", {})}
    return merged


def validate_output(output: dict, agent: str, schema_name: str = "default") -> list[str]:
    """
    Validate an agent output against its schema.
    Returns list of validation errors (empty = valid).
    """
    schema = load_schema(agent, schema_name)
    errors = []
    fields = schema.get("fields", {})

    for field_name, field_def in fields.items():
        if field_def.get("required", False) and field_name not in output:
            errors.append(f"Missing required field: {field_name}")
        if field_name in output:
            type_errors = _validate_type(field_name, output[field_name], field_def)
            errors.extend(type_errors)

    return errors


def _validate_type(name: str, value: Any, field_def: dict) -> list[str]:
    """Basic type validation."""
    errors = []
    expected_type = field_def.get("type", "any")

    type_map = {
        "string": str, "integer": int, "float": (int, float),
        "boolean": bool, "object": dict,
    }

    if expected_type in type_map:
        if not isinstance(value, type_map[expected_type]):
            errors.append(f"{name}: expected {expected_type}, got {type(value).__name__}")

    if expected_type.startswith("list") and not isinstance(value, list):
        errors.append(f"{name}: expected list, got {type(value).__name__}")

    if "range" in field_def and isinstance(value, (int, float)):
        lo, hi = field_def["range"]
        if not (lo <= value <= hi):
            errors.append(f"{name}: {value} outside range [{lo}, {hi}]")

    if "values" in field_def and expected_type == "enum":
        if value not in field_def["values"]:
            errors.append(f"{name}: '{value}' not in {field_def['values']}")

    return errors


def list_schemas(agent: str) -> list[str]:
    """List all available schemas for an agent."""
    schemas = []
    agent_dir = SCHEMAS_DIR / agent
    if agent_dir.exists():
        for f in agent_dir.glob("*.yaml"):
            schemas.append(f.stem)
        custom_dir = agent_dir / "custom"
        if custom_dir.exists():
            for f in custom_dir.glob("*.yaml"):
                schemas.append(f"custom/{f.stem}")
    return schemas


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
```

### How Each Agent Consumes Schemas

**Intent Agent:** Receives the active schema as part of its input context. When populating the output, it iterates over all fields in the schema and attempts to populate each one that has a `description`. Required fields must be populated; optional fields are best-effort. Custom fields (like `mood_keywords` or `lora_reasoning`) get populated because the agent can reason about what the description asks for.

```python
# In intent_agent.py — schema-aware output generation

async def translate(
    self,
    user_intent: str,
    model_id: str,
    workflow_state: dict,
    refinement_context: list | None = None,
    output_schema: str = "default",         # ← schema selection
) -> dict:
    """
    Translate intent using the specified output schema.
    The schema defines what fields to populate.
    """
    schema = load_schema("intent", output_schema)
    profile = get_intent_section(model_id)

    # Core translation logic (unchanged)
    mutations = self._translate_intent(user_intent, profile, workflow_state)

    # Build output according to schema
    output = {}
    for field_name, field_def in schema.get("fields", {}).items():
        output[field_name] = self._populate_field(
            field_name, field_def, mutations, user_intent, profile
        )

    # Validate before returning
    errors = validate_output(output, "intent", output_schema)
    if errors:
        output["_validation_warnings"] = errors

    return output
```

**Execution Agent:** Schema defines what metadata to emit alongside the execution result. Default: just patches and status. Custom: could include timing telemetry, node-level execution traces, or resource usage.

**Verify Agent:** Schema defines the scoring dimensions. Default: `overall_score`, `intent_alignment`, `technical_quality`. Custom: could add `composition_score`, `color_harmony`, `style_conformance`, or any other quality axis the artist cares about.

```yaml
# schemas/verify/custom/cinematographer.yaml
# A cinematographer wants composition-focused quality judgment

schema:
  name: "CinematographerVerification"
  version: "1.0"
  agent: "verify"
  extends: "default"

fields:
  composition_score:
    type: "float"
    range: [0.0, 1.0]
    required: true
    description: "Rule of thirds, leading lines, visual weight balance"

  lighting_continuity:
    type: "float"
    range: [0.0, 1.0]
    required: true
    description: "Consistency of light direction and color temperature"

  depth_of_field:
    type: "object"
    required: false
    description: "Depth of field assessment"
    fields:
      bokeh_quality:
        type: "float"
        range: [0.0, 1.0]
        description: "Quality of out-of-focus areas"
      focal_plane:
        type: "string"
        description: "Where the focal plane appears to be"

  cinematic_grade:
    type: "enum"
    values: ["A", "B", "C", "D", "F"]
    required: true
    description: "Overall cinematic quality grade for this frame"
```

### Schema Selection at Runtime

The Router manages which schema each agent uses for a given pipeline run:

```python
# In router.py — schema-aware delegation

@dataclass
class RouterContext:
    user_intent: str
    intent_type: str
    active_model: str
    workflow_state: str
    iteration_count: int = 0
    max_iterations: int = 3
    history: list[dict] = field(default_factory=list)
    # Schema configuration per agent
    schemas: dict[str, str] = field(default_factory=lambda: {
        "intent": "default",
        "execution": "default",
        "verify": "default",
    })
```

Users set schemas via:
1. **Conversation:** "Use my cinematographer verification schema"
2. **Workflow metadata:** A custom property on the workflow JSON specifying schemas
3. **Profile linkage:** A model profile can recommend schemas via a new `recommended_schemas` section

---

## Implementation Phases

> **This is an AND-node task.** All phases are interdependent and the architecture only works if they're coherent. Phase 1 (Model Profiles) is the hardest branch — everything downstream consumes this structure. Address it first.

### Phase 1: Model Profile Registry

**Priority: HIGHEST — this is the foundation everything else consumes.**

**What:** A YAML-based registry of model communication profiles that encode how to prompt, parameterize, and evaluate outputs for each model.

**Where:** `src/profiles/` directory with one YAML file per model, plus a `default_{arch}.yaml` fallback for unknown models.

**Why YAML over JSON:** Profiles will be human-editable by artists. YAML comments let you annotate WHY a particular sweet spot exists ("cfg >7 causes banding on this checkpoint because...").

#### Profile Schema

Each profile serves three consumers (Intent, Execution, Verify) from a single source of truth:

```yaml
# profiles/flux1-dev.yaml
# Model Communication Profile — Flux.1 Dev
# ============================================================
# This file encodes model-specific knowledge that agents query
# at runtime. It is the ACCESS oracle for Flux behavior.
# ============================================================

meta:
  model_id: "flux1-dev"            # unique key, matches ComfyUI model name
  model_class: "flux"              # family grouping for fallback behavior
  base_arch: "dit"                 # diffusion transformer vs unet
  modality: "image"                # image | video | audio
  version_hash: ""                 # optional — ties profile to specific checkpoint

# ============================================================
# INTENT AGENT CONSUMES THIS SECTION
# How to translate artistic language into parameters for this model
# ============================================================
prompt_engineering:
  style: "natural_language"        # natural_language | tag_based | hybrid
  positive_prompt:
    structure: "description_first"  # description_first | tags_first | weighted_blocks
    keyword_sensitivity: 0.8        # how much exact wording matters vs. semantic meaning (0-1)
    effective_patterns:             # maps artistic language → effective prompt fragments
      - pattern: "cinematic lighting, dramatic shadows"
        maps_to_intent: ["dramatic", "moody", "cinematic"]
      - pattern: "soft diffused light, pastel tones"
        maps_to_intent: ["dreamy", "gentle", "ethereal"]
      - pattern: "sharp detail, high contrast"
        maps_to_intent: ["crisp", "detailed", "punchy"]
    token_weighting: "parenthetical" # parenthetical | numeric | none
    max_effective_tokens: 256        # prompt length beyond which quality degrades
  negative_prompt:
    required_base: "blurry, low quality, distorted"
    style: "exclusion_list"        # exclusion_list | anti_description | minimal
    effectiveness: 0.6              # how much negative prompt actually matters for this model (0-1)

  # THE CRITICAL MAPPING: artistic intent → parameter space
  # Each key is an intent word/phrase an artist might use.
  # Values describe DIRECTION of parameter change, not absolute values.
  # The Intent Agent resolves directions into concrete values using parameter_space ranges.
  intent_translations:
    "dreamier":
      cfg_direction: "lower"
      sampler_preference: "euler_ancestral"
      prompt_additions: ["soft focus", "ethereal glow"]
      denoise_direction: "lower"
    "sharper":
      cfg_direction: "higher"
      sampler_preference: "dpmpp_2m"
      prompt_additions: ["crisp details", "sharp focus"]
      steps_direction: "higher"
    "more stylized":
      cfg_direction: "higher"
      prompt_strategy: "emphasize_style_tokens"
    "more photorealistic":
      cfg_direction: "moderate"     # not too high, not too low
      prompt_additions: ["photorealistic", "detailed skin texture", "natural lighting"]
      sampler_preference: "dpmpp_2m"
    "moodier":
      prompt_additions: ["low key lighting", "deep shadows", "atmospheric"]
      cfg_direction: "slightly_higher"
    "more abstract":
      cfg_direction: "lower"
      steps_direction: "lower"
      denoise_direction: "higher"

# ============================================================
# EXECUTION AGENT CONSUMES THIS SECTION
# Concrete parameter ranges, sweet spots, and failure modes
# ============================================================
parameter_space:
  steps:
    default: 20
    range: [8, 50]
    sweet_spot: [18, 28]           # where quality/speed tradeoff is optimal
    diminishing_returns: 35         # above this, marginal quality gain per step
  cfg:
    default: 3.5
    range: [1.0, 10.0]
    sweet_spot: [2.5, 4.5]
    failure_modes:
      too_high: "oversaturation, color banding, artifacts at >7"
      too_low: "incoherent composition, prompt ignored at <1.5"
  sampler:
    recommended: ["euler", "dpmpp_2m", "euler_ancestral"]
    avoid: ["ddim"]                 # known poor results for this model
    scheduler: "normal"
  resolution:
    native: [1024, 1024]
    supported_ratios: ["1:1", "16:9", "9:16", "4:3", "3:4"]
    upscale_friendly: true
  denoise:
    default: 1.0
    img2img_sweet_spot: [0.4, 0.7]

  lora_behavior:
    max_simultaneous: 3
    strength_range: [0.3, 1.2]
    default_strength: 0.8
    interaction_model: "additive"   # additive | multiplicative | complex
    known_conflicts: []             # list of LoRA pairs that interfere

# ============================================================
# VERIFY AGENT CONSUMES THIS SECTION
# What "good" looks like for this model, and how to diagnose problems
# ============================================================
quality_signatures:
  expected_characteristics:
    - "photorealistic skin texture"
    - "coherent lighting direction"
    - "stable hand/finger geometry at cfg 3-4"
    - "smooth gradients without banding"
  known_artifacts:
    - condition: "cfg > 7"
      artifact: "color banding in gradients"
    - condition: "steps < 12"
      artifact: "soft/mushy details, incomplete structures"
    - condition: "resolution != native"
      artifact: "tiling or stretching at non-native resolutions"
    - condition: "prompt > max_effective_tokens"
      artifact: "later prompt elements ignored"
  quality_floor:
    description: "minimum acceptable output at optimal settings"
    reference_score: 0.7            # normalized quality baseline (0-1)

  # Maps visual problems → actionable parameter adjustments
  # The Verify Agent uses this to produce refinement instructions
  iteration_signals:
    needs_more_steps: ["mushy details", "incomplete structures", "noise visible"]
    needs_lower_cfg: ["oversaturated", "harsh edges", "color artifacts", "banding"]
    needs_higher_cfg: ["incoherent composition", "prompt not reflected", "random output"]
    needs_reprompt: ["wrong subject", "missing elements", "style mismatch"]
    needs_inpaint: ["hand/finger issues", "face distortion", "text artifacts"]
    model_limitation: ["consistent failure across parameter ranges"]
```

#### Fallback Profiles

Create these defaults for unknown models:

```
profiles/
├── flux1-dev.yaml
├── sdxl-base.yaml
├── default_dit.yaml          # fallback for unknown DiT models
├── default_unet.yaml         # fallback for unknown UNet models
├── default_video.yaml        # fallback for video models
└── _schema.yaml              # canonical schema reference
```

#### Profile Loader

```python
# src/profiles/loader.py

"""
Model Profile Registry — the ACCESS oracle for model-specific knowledge.

Agents never memorize model quirks. They query this registry at runtime.
Unknown models get a default_{arch} fallback with conservative parameters.
"""

import yaml
from pathlib import Path
from typing import Optional

PROFILES_DIR = Path(__file__).parent
_cache: dict[str, dict] = {}


def load_profile(model_id: str) -> dict:
    """Load a model profile by ID. Falls back to arch default if not found."""
    if model_id in _cache:
        return _cache[model_id]

    profile_path = PROFILES_DIR / f"{model_id}.yaml"
    if profile_path.exists():
        profile = _load_yaml(profile_path)
    else:
        profile = _load_fallback(model_id)

    _cache[model_id] = profile
    return profile


def _load_fallback(model_id: str) -> dict:
    """Attempt arch-based fallback, then absolute default."""
    # Try to infer arch from model_id naming conventions
    for arch in ["dit", "unet", "video"]:
        fallback_path = PROFILES_DIR / f"default_{arch}.yaml"
        if fallback_path.exists():
            profile = _load_yaml(fallback_path)
            profile["meta"]["model_id"] = model_id
            profile["meta"]["_is_fallback"] = True
            return profile

    # Last resort: return minimal safe defaults
    return _minimal_defaults(model_id)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _minimal_defaults(model_id: str) -> dict:
    """Absolute minimum profile — conservative and safe."""
    return {
        "meta": {
            "model_id": model_id,
            "model_class": "unknown",
            "base_arch": "unknown",
            "modality": "image",
            "_is_fallback": True,
            "_is_minimal": True,
        },
        "prompt_engineering": {
            "style": "hybrid",
            "positive_prompt": {
                "structure": "description_first",
                "keyword_sensitivity": 0.5,
                "effective_patterns": [],
                "token_weighting": "none",
                "max_effective_tokens": 200,
            },
            "negative_prompt": {
                "required_base": "blurry, low quality, distorted",
                "style": "exclusion_list",
                "effectiveness": 0.5,
            },
            "intent_translations": {},
        },
        "parameter_space": {
            "steps": {"default": 20, "range": [10, 50], "sweet_spot": [15, 30]},
            "cfg": {"default": 7.0, "range": [1.0, 15.0], "sweet_spot": [5.0, 9.0]},
            "sampler": {"recommended": ["euler", "dpmpp_2m"], "avoid": []},
            "resolution": {"native": [512, 512], "supported_ratios": ["1:1"]},
            "denoise": {"default": 1.0, "img2img_sweet_spot": [0.4, 0.7]},
        },
        "quality_signatures": {
            "expected_characteristics": [],
            "known_artifacts": [],
            "quality_floor": {"reference_score": 0.5},
            "iteration_signals": {},
        },
    }


def get_intent_section(model_id: str) -> dict:
    """Convenience: return only what the Intent Agent needs."""
    profile = load_profile(model_id)
    return profile.get("prompt_engineering", {})


def get_parameter_section(model_id: str) -> dict:
    """Convenience: return only what the Execution Agent needs."""
    profile = load_profile(model_id)
    return profile.get("parameter_space", {})


def get_quality_section(model_id: str) -> dict:
    """Convenience: return only what the Verify Agent needs."""
    profile = load_profile(model_id)
    return profile.get("quality_signatures", {})


def is_fallback(model_id: str) -> bool:
    """Check if a profile is a fallback (not model-specific)."""
    profile = load_profile(model_id)
    return profile.get("meta", {}).get("_is_fallback", False)
```

#### Phase 1 Success Criteria

- [ ] `_schema.yaml` defines the canonical profile structure with comments
- [ ] `flux1-dev.yaml` is complete and covers all three consumer sections
- [ ] `sdxl-base.yaml` is complete (validates schema works across architectures)
- [ ] `default_dit.yaml` and `default_unet.yaml` provide safe fallbacks
- [ ] `loader.py` loads profiles, caches them, falls back correctly
- [ ] Test: `load_profile("flux1-dev")` returns complete profile
- [ ] Test: `load_profile("unknown-model-xyz")` returns a fallback without crashing
- [ ] Test: `is_fallback()` correctly identifies fallback vs specific profiles
- [ ] Test: Convenience accessors return correct subsections

#### Too-Easy Check

If writing the profiles feels trivially fast, you're not capturing enough model-specific behavior. A good Flux profile should encode things that took hours of experimentation to learn. If it's just defaults, it's not doing its job. The `intent_translations` section especially — every entry should reflect real artistic→parameter knowledge.

---

### Phase 1.5: Schema System

**Priority: HIGH — agents need schemas before they can produce schema-aware output.**

**What:** The schema registry, loader, validator, and SchemaGenerator node. This provides the infrastructure that allows each agent to produce user-customizable structured output.

**Where:** `src/schemas/` directory plus `schemas/` YAML files.

**Dependency:** Profiles (Phase 1) must exist first because schemas reference profile-derived concepts (parameter names, quality dimensions). But schemas must exist before agents (Phases 2-4) because agents produce schema-validated output.

#### Default Schema Files to Create

```
schemas/
├── intent/
│   ├── default.yaml               # IntentSpecification schema
│   └── custom/                    # user-created schemas go here
│       └── .gitkeep
├── execution/
│   ├── default.yaml               # ExecutionResult schema
│   └── custom/
│       └── .gitkeep
├── verify/
│   ├── default.yaml               # VerificationResult schema
│   └── custom/
│       └── .gitkeep
└── _meta/
    └── field_types.yaml           # canonical type definitions
```

Each default schema should be a YAML representation of the dataclass defined in the corresponding agent's phase. This makes the dataclass and the schema two views of the same contract — the dataclass is the Python-side interface, the schema is the user-facing configuration.

#### SchemaGenerator as ComfyUI Node

The `SchemaGenerator` node should be registered as a ComfyUI custom node in the project's node pack. This allows users to:

1. Run a workflow, get an output
2. Feed that output (or a hand-crafted example) into the SchemaGenerator node
3. Select which agent the schema is for
4. Get a YAML schema file that can be saved and reused

The node should also be callable as an agent tool (`generate_schema`) for in-conversation schema creation:

```
User: "I want the verify agent to also score composition and lighting separately"

Agent: Uses generate_schema tool with example:
  {
    "overall_score": 0.85,
    "intent_alignment": 0.9,
    "technical_quality": 0.8,
    "composition_score": 0.7,
    "lighting_continuity": 0.9,
    "decision": "accept"
  }

→ Generates schemas/verify/custom/composition_focused.yaml
→ "Created 'composition_focused' schema for the Verify Agent. It now
   scores composition and lighting as separate dimensions. Want me to
   use it for the next run?"
```

#### Phase 1.5 Success Criteria

- [ ] Default schemas exist for all three agents (intent, execution, verify)
- [ ] Schema loader resolves: custom → standard → default fallback
- [ ] Schema inheritance (`extends`) merges fields correctly
- [ ] Schema validator catches missing required fields
- [ ] Schema validator catches type mismatches
- [ ] SchemaGenerator infers correct types from example values
- [ ] SchemaGenerator handles nested objects and lists
- [ ] SchemaGenerator node registers in ComfyUI
- [ ] `generate_schema` agent tool works in conversation context
- [ ] Test: roundtrip — example dict → generated schema → validate same dict = no errors
- [ ] Test: custom schema extends default without losing required fields
- [ ] Test: `list_schemas("verify")` returns both default and custom schemas

---

### Phase 2: Intent Agent

**Priority: HIGH — this is the reasoning layer that consumes profiles.**

**What:** A specialist agent that translates natural language artistic intent into structured parameter specifications. It is a PURE REASONING agent — it does NOT touch ComfyUI. It does NOT call tools. Its output is a specification that the Execution Agent implements.

**Why tool-less:** Separation of concerns. Intent can be tested independently of ComfyUI being running. It also means Intent's system prompt stays focused on artistic translation, not API mechanics.

#### Intent Agent System Prompt (Core Logic)

```python
# src/agents/intent_agent.py

"""
Intent Agent — translates artistic language into parameter specifications.

This agent is tool-less. It consumes a model profile and user intent,
then produces a structured IntentSpecification that the Execution Agent
implements. It never touches ComfyUI directly.

The separation allows:
- Testing Intent without a running ComfyUI instance
- Swapping models without changing the Intent logic
- Composing conflicting intents through explicit negotiation
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ParameterMutation:
    """A single parameter change."""
    target: str              # e.g. "KSampler.cfg", "KSampler.sampler_name"
    action: Literal["set", "adjust_up", "adjust_down"]
    value: float | str | None = None       # concrete value for "set"
    magnitude: str | None = None           # "slight", "moderate", "large" for adjust
    reason: str = ""                        # why this change, traceable to profile


@dataclass
class PromptMutation:
    """A change to the prompt text."""
    target: Literal["positive_prompt", "negative_prompt"]
    action: Literal["append", "prepend", "replace", "remove"]
    value: str = ""
    reason: str = ""


@dataclass
class IntentSpecification:
    """
    The output contract of the Intent Agent.
    The Execution Agent consumes this to produce RFC6902 patches.
    """
    model_id: str
    parameter_mutations: list[ParameterMutation] = field(default_factory=list)
    prompt_mutations: list[PromptMutation] = field(default_factory=list)
    confidence: float = 0.0                # 0-1, how confident in this translation
    conflicts_resolved: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    using_fallback_profile: bool = False   # flag if profile was a fallback


@dataclass
class ConflictResolution:
    """How the Intent Agent resolves competing intents."""
    intent_a: str
    intent_b: str
    conflict_dimension: str    # e.g. "cfg_direction"
    resolution_strategy: str   # e.g. "hold_current, adjust_via_prompt_and_sampler"
    explanation: str
```

#### Intent Translation Logic

The Intent Agent follows this process:

```
1. Receive user intent string ("make it dreamier")
2. Get active model_id from current workflow state (provided by Router)
3. Load model profile via loader.get_intent_section(model_id)
4. Look up intent in profile.intent_translations
   - If exact match: use the translation directly
   - If no match: decompose intent into component words, try partial matches
   - If still no match: use LLM reasoning with parameter_space constraints
5. Check for conflicts between multiple intents
   - "dreamier AND more detailed" → cfg wants to go both down AND up
   - Apply conflict resolution: hold cfg, adjust via sampler + prompt instead
6. Convert directional instructions to concrete values
   - "cfg_direction: lower" + current cfg 3.5 + sweet_spot [2.5, 4.5] → cfg 2.8
   - Respect sweet_spot ranges — never go outside unless explicitly asked
7. Check prompt style
   - If model is "natural_language": integrate additions as natural phrases
   - If model is "tag_based": add as comma-separated tags
   - If model is "hybrid": match the style of the existing prompt
8. Output IntentSpecification
```

#### Conflict Resolution Rules

```python
# Built into the Intent Agent's reasoning, NOT in the profile.
# The profile says WHAT each intent wants. The agent decides HOW to reconcile.

CONFLICT_RULES = {
    # When two intents pull cfg in opposite directions
    ("cfg_direction:lower", "cfg_direction:higher"): {
        "strategy": "hold_current_cfg_adjust_via_prompt_and_sampler",
        "explanation": "Conflicting cfg demands. Holding cfg, using prompt tokens "
                       "and sampler selection to achieve both intents.",
    },
    # When two intents pull steps in opposite directions
    ("steps_direction:lower", "steps_direction:higher"): {
        "strategy": "favor_higher_for_quality",
        "explanation": "Conflicting step demands. Favoring higher steps since "
                       "quality is the safer bet.",
    },
    # When denoise conflicts
    ("denoise_direction:lower", "denoise_direction:higher"): {
        "strategy": "favor_lower_for_preservation",
        "explanation": "Conflicting denoise demands. Favoring lower to preserve "
                       "more of the original image.",
    },
}
```

#### Phase 2 Success Criteria

- [ ] `IntentSpecification` dataclass is defined and serializable
- [ ] Intent Agent can translate "dreamier" → concrete mutations for flux1-dev
- [ ] Intent Agent can translate "dreamier" → different mutations for sdxl-base
- [ ] Intent Agent handles unknown intents gracefully (falls back to LLM reasoning with constraints)
- [ ] Conflict resolution works: "dreamier and sharper" produces sensible output
- [ ] Fallback profile flag propagates to warnings in the spec
- [ ] **Schema-aware:** Intent Agent accepts `output_schema` parameter and populates custom fields
- [ ] **Schema-aware:** Custom fields with descriptions get populated via LLM reasoning
- [ ] **Schema-aware:** Output validates against the specified schema with no errors
- [ ] Test: roundtrip — intent string → IntentSpecification → verify all fields populated
- [ ] Test: same intent + different model → different parameter values
- [ ] Test: conflicting intents → conflicts_resolved is populated with explanation
- [ ] Test: custom schema with `mood_keywords` field → field populated from intent context

---

### Phase 3: Verify Agent

**Priority: HIGH — second hardest branch. "Good" is model-relative.**

**What:** A specialist agent that performs vision analysis on generated outputs, judges quality relative to the model's expected characteristics, and decides whether the iterative_refine loop continues or exits.

#### Verification Result Contract

```python
# src/agents/verify_agent.py

"""
Verify Agent — model-aware quality judgment and iteration control.

This agent consumes quality_signatures from the model profile and
the output image(s) from execution. It produces a VerificationResult
that the Router uses to decide: accept, refine, reprompt, or escalate.

Critical design: "good" is model-relative. A Flux output at cfg 3.5
and an SDXL output at cfg 7 look fundamentally different, and both
can be correct. The profile defines the quality baseline.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RefinementAction:
    """A single refinement instruction for the next iteration."""
    type: Literal["adjust_params", "reprompt", "inpaint", "upscale", "retry"]
    target: str              # what to change (e.g. "hand region", "cfg", "prompt")
    reason: str              # why this refinement is needed
    priority: int = 1        # 1=highest, used to order multiple actions


@dataclass
class VerificationResult:
    """
    The output contract of the Verify Agent.
    The Router uses this to control the iterative_refine loop.
    """
    overall_score: float              # 0-1 composite quality score
    intent_alignment: float           # 0-1 did we achieve what the user asked for?
    technical_quality: float          # 0-1 model-relative quality assessment
    decision: Literal["accept", "refine", "reprompt", "escalate"]
    refinement_actions: list[RefinementAction] = field(default_factory=list)
    iteration_count: int = 0          # which iteration is this?
    max_iterations: int = 3           # prevent infinite loops
    diagnosed_issues: list[str] = field(default_factory=list)
    model_limitations: list[str] = field(default_factory=list)  # issues that can't be fixed by params
    using_fallback_profile: bool = False
```

#### Verification Process

```
1. Receive output image + execution context from Router
2. Load model profile via loader.get_quality_section(model_id)
3. Perform vision analysis on the output:
   a. Check expected_characteristics — is each one present?
   b. Check against known_artifacts — given the parameters used, are expected artifacts present?
   c. Compare against user's original intent — did the requested change land?
4. Score the output:
   - technical_quality: based on expected_characteristics hit rate + artifact detection
   - intent_alignment: based on whether the user's request was achieved
   - overall_score: weighted combination (intent_alignment * 0.6 + technical_quality * 0.4)
5. Make decision:
   - overall_score >= quality_floor AND intent_alignment > 0.7 → "accept"
   - intent_alignment < 0.4 → "reprompt" (fundamentally wrong)
   - technical_quality < quality_floor AND iteration_count < max_iterations → "refine"
   - iteration_count >= max_iterations → "escalate" (return to user with best attempt)
6. If "refine", map diagnosed issues to iteration_signals:
   - "mushy details" → needs_more_steps → RefinementAction(type="adjust_params", target="steps")
   - "oversaturated" → needs_lower_cfg → RefinementAction(type="adjust_params", target="cfg")
   - "hand issues" → needs_inpaint → RefinementAction(type="inpaint", target="hand region")
7. If issue maps to model_limitation, add to model_limitations list (don't waste iterations on it)
8. Output VerificationResult
```

#### Model-Class Calibration

Different architectures have different quality ceilings. The Verify Agent calibrates expectations based on the profile:

```
DiT models (Flux):
  - High coherence expected at 20 steps
  - Hands/fingers are the primary failure mode
  - Quality ceiling is high

UNet models (SDXL):
  - More artifacts expected at same step count
  - Prompt adherence requires higher cfg
  - Quality ceiling varies significantly with checkpoint

Video models:
  - Temporal coherence is primary concern, not per-frame quality
  - Per-frame artifacts are acceptable if motion is smooth

LoRA-heavy checkpoints:
  - Style conformance matters more than photorealism
  - Evaluate against the LoRA's intended aesthetic, not general quality
```

This calibration comes from `quality_signatures.expected_characteristics` and `quality_signatures.quality_floor`, not from the agent's training data.

#### Phase 3 Success Criteria

- [ ] `VerificationResult` dataclass is defined and serializable
- [ ] Verify Agent can evaluate a Flux output against flux1-dev quality_signatures
- [ ] Verify Agent correctly identifies known artifacts from profile conditions
- [ ] Verify Agent maps visual problems to actionable iteration_signals
- [ ] Verify Agent separates model limitations from parameter problems
- [ ] Decision logic respects max_iterations and quality_floor
- [ ] **Schema-aware:** Verify Agent accepts `output_schema` parameter and scores custom dimensions
- [ ] **Schema-aware:** Custom quality dimensions (e.g. `composition_score`) populated via vision analysis
- [ ] **Schema-aware:** Decision logic accounts for custom required fields in scoring
- [ ] Test: high-quality output → "accept" with score above quality_floor
- [ ] Test: low-quality output → "refine" with populated refinement_actions
- [ ] Test: fundamentally wrong output → "reprompt"
- [ ] Test: max iterations reached → "escalate"
- [ ] Test: custom schema with `composition_score` → field populated from vision analysis

---

### Phase 4: Router

**Priority: MEDIUM — the Router is lighter than it seems.**

**What:** A sequencer that delegates authority to specialist agents and controls the iterative_refine loop. The Router does NOT understand models, judge quality, or translate intent. It understands sequencing and authority boundaries.

#### Router Logic

```python
# src/agents/router.py

"""
Router — authority delegation and loop control.

The Router is NOT a god-agent. It doesn't understand models, judge quality,
or translate intent. It understands:
- SEQUENCING: which agent runs in which order
- AUTHORITY: what each agent is allowed to decide
- LOOP CONTROL: when to iterate, when to stop, when to escalate

Authority boundaries are HARD:
- Intent Agent OWNS parameter decisions → Execution never changes params on its own
- Execution Agent OWNS workflow mutation → Intent never touches ComfyUI
- Verify Agent OWNS quality judgment → only it can say accept/refine/reprompt
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RouterContext:
    """State the Router maintains across the agent pipeline."""
    user_intent: str                       # original user request
    intent_type: Literal[
        "generation",      # create from scratch
        "modification",    # change existing output
        "evaluation",      # judge what we have
        "exploration",     # just translate, don't execute yet
    ]
    active_model: str                      # model_id from current workflow
    workflow_state: Literal[
        "empty",           # no workflow loaded
        "configured",      # workflow ready, not yet executed
        "executing",       # currently generating
        "has_output",      # output available
    ]
    iteration_count: int = 0
    max_iterations: int = 3
    history: list[dict] = field(default_factory=list)  # audit trail


# Delegation sequences by intent type
DELEGATION_SEQUENCES = {
    "generation":    ["intent", "execution", "verify"],
    "modification":  ["intent", "execution", "verify"],
    "evaluation":    ["verify"],           # skip intent and execution
    "exploration":   ["intent"],           # just translate, don't execute
}
```

#### Router Flow (Modification — Primary Use Case)

```
User: "make it dreamier"

ROUTER: Classify intent_type → "modification"
ROUTER: Get active_model from workflow → "flux1-dev"
ROUTER: Delegation sequence → [intent, execution, verify]

ROUTER → INTENT AGENT:
  Input: { user_intent: "make it dreamier", model_id: "flux1-dev", current_workflow_state }
  Output: IntentSpecification

ROUTER: Check IntentSpecification.confidence
  - If < 0.5: ask user for clarification before proceeding
  - If >= 0.5: pass to Execution

ROUTER → EXECUTION AGENT:
  Input: { intent_spec: IntentSpecification, workflow_json: current_workflow }
  Output: { patches: [RFC6902 patches], execution_result: { output_images, status } }

ROUTER → VERIFY AGENT:
  Input: { output_images, original_intent: "make it dreamier", model_id: "flux1-dev", iteration: 0 }
  Output: VerificationResult

ROUTER: Check VerificationResult.decision
  - "accept": return result to user ✓
  - "refine": loop back → re-enter at INTENT with refinement_actions as additional context
  - "reprompt": loop back → re-enter at INTENT with reprompt guidance
  - "escalate": return best attempt to user with explanation of what couldn't be fixed
```

#### Authority Boundary Enforcement

These are HARD rules the Router enforces:

```python
AUTHORITY_RULES = {
    "intent": {
        "owns": ["parameter_decisions", "prompt_modifications", "intent_translation"],
        "cannot": ["execute_workflows", "call_comfyui_api", "modify_workflow_json_directly"],
    },
    "execution": {
        "owns": ["workflow_mutation", "comfyui_communication", "rfc6902_patching"],
        "cannot": ["change_parameters_without_intent_spec", "judge_output_quality"],
    },
    "verify": {
        "owns": ["quality_judgment", "iteration_decisions", "issue_diagnosis"],
        "cannot": ["modify_prompts", "change_parameters", "execute_workflows"],
        "can_recommend": ["reprompt", "param_adjustments", "inpaint"],  # but Intent/Exec implement
    },
}
```

#### Exception Handling (Router-Owned)

```python
ROUTER_EXCEPTIONS = {
    "timeout": {
        "condition": "execution exceeds timeout threshold",
        "action": "abort execution, return partial result to user",
    },
    "loop_limit": {
        "condition": "verify keeps saying 'refine' past max_iterations",
        "action": "escalate to user with best attempt + diagnosis",
    },
    "model_not_found": {
        "condition": "intent references model that doesn't exist",
        "action": "catch before execution, ask user to select model",
    },
    "profile_fallback": {
        "condition": "model profile is a fallback (not model-specific)",
        "action": "warn user that results may be suboptimal, proceed with caution",
    },
    "low_confidence": {
        "condition": "IntentSpecification.confidence < 0.5",
        "action": "ask user for clarification before proceeding",
    },
}
```

#### Phase 4 Success Criteria

- [ ] Router correctly classifies intent types
- [ ] Router follows correct delegation sequence for each intent type
- [ ] Authority boundaries are enforced — Execution can't run without an IntentSpec
- [ ] Iterative loop works: Intent → Execute → Verify → (refine) → Intent → ...
- [ ] Loop exits correctly on "accept", "escalate", and max_iterations
- [ ] Exception handling: timeout, loop limit, model not found all work
- [ ] **Schema-aware:** Router carries `schemas` dict and passes correct schema to each agent
- [ ] **Schema-aware:** User can set schemas via conversation ("use my cinematographer schema")
- [ ] **Schema-aware:** Router validates that requested schemas exist before delegating
- [ ] Test: full pipeline — user intent → IntentSpec → execution → VerificationResult → decision
- [ ] Test: refinement loop — verify returns "refine" → router loops back correctly
- [ ] Test: escalation — max_iterations reached → user gets best attempt
- [ ] Test: custom schemas propagate through the full pipeline without errors

---

### Phase 5: Integration & iterative_refine

**Priority: This is the capstone that wires everything together.**

**What:** Connect the MoE architecture to the existing tool layers and implement the `iterative_refine` brain tool — the strategic centerpiece that transforms the agent from a command executor into a co-pilot.

#### Integration Points

```python
# How specialist agents connect to existing tools

TOOL_LAYER_BINDINGS = {
    # Intent Agent is tool-less — it's pure reasoning
    # But it READS workflow state via these tools:
    "intent_reads": [
        "inspect_workflow",         # UNDERSTAND: current workflow structure
        "get_node_params",          # UNDERSTAND: current parameter values
        "get_active_model",         # UNDERSTAND: which model is loaded
    ],

    # Execution Agent WRITES via these tools:
    "execution_uses": [
        "patch_workflow",           # PILOT: RFC6902 JSON patching
        "execute_workflow",         # PILOT: trigger ComfyUI execution
        "execute_with_progress",    # PILOT: execution with progress tracking
        "set_node_param",           # PILOT: direct parameter setting
    ],

    # Verify Agent EVALUATES via these tools:
    "verify_uses": [
        "verify_execution",         # VERIFY: check execution completed
        "analyze_output",           # VERIFY: vision analysis of output
        "compare_outputs",          # VERIFY: compare before/after
    ],

    # Router uses DISCOVER layer when needed:
    "router_uses": [
        "search_models",            # DISCOVER: find models
        "search_nodes",             # DISCOVER: find nodes
        "generate_schema",          # DISCOVER: create output schemas from examples
        "list_schemas",             # DISCOVER: list available schemas per agent
    ],
}
```

#### iterative_refine Brain Tool

```python
# src/tools/brain/iterative_refine.py

"""
iterative_refine — the co-pilot loop.

This is the brain tool that orchestrates the MoE pipeline:
  Intent → Execute → Verify → (loop or exit)

It is the strategic centerpiece of the agent. Without it, we have
individual components. With it, we have an autonomous quality
improvement loop that iterates toward artistic intent.
"""

async def iterative_refine(
    user_intent: str,
    workflow_state: dict,
    model_id: str,
    max_iterations: int = 3,
    schemas: dict[str, str] | None = None,   # per-agent schema overrides
) -> dict:
    """
    Execute the full MoE pipeline with iterative refinement.

    Args:
        user_intent: Natural language artistic intent from user
        workflow_state: Current ComfyUI workflow JSON
        model_id: Active model identifier
        max_iterations: Safety limit on refinement loops
        schemas: Optional per-agent schema names, e.g.
                 {"intent": "lora_workflow", "verify": "cinematographer"}
                 Defaults to "default" for each agent if not specified.

    Returns:
        {
            "status": "accepted" | "escalated" | "error",
            "final_output": image_path | None,
            "iterations": int,
            "history": [VerificationResult, ...],
            "final_verdict": VerificationResult,
            "schemas_used": {"intent": str, "verify": str, "execution": str},
        }
    """
    schema_config = schemas or {"intent": "default", "execution": "default", "verify": "default"}

    router = Router(
        user_intent=user_intent,
        model_id=model_id,
        workflow_state=workflow_state,
        max_iterations=max_iterations,
        schemas=schema_config,
    )

    # Determine delegation sequence
    sequence = router.get_delegation_sequence()

    # Main loop
    refinement_context = None  # accumulates across iterations
    for iteration in range(max_iterations):
        # INTENT PHASE
        if "intent" in sequence:
            intent_spec = await intent_agent.translate(
                user_intent=user_intent,
                model_id=model_id,
                workflow_state=workflow_state,
                refinement_context=refinement_context,  # None on first pass
            )
            router.check_confidence(intent_spec)

        # EXECUTION PHASE
        if "execution" in sequence:
            execution_result = await execution_agent.execute(
                intent_spec=intent_spec,
                workflow_json=workflow_state,
            )
            if execution_result.error:
                return router.handle_execution_error(execution_result)

        # VERIFY PHASE
        if "verify" in sequence:
            verification = await verify_agent.evaluate(
                output=execution_result.output,
                original_intent=user_intent,
                model_id=model_id,
                iteration=iteration,
            )
            router.record_history(verification)

            # DECISION
            if verification.decision == "accept":
                return {
                    "status": "accepted",
                    "final_output": execution_result.output,
                    "iterations": iteration + 1,
                    "history": router.history,
                    "final_verdict": verification,
                }

            if verification.decision == "escalate" or iteration == max_iterations - 1:
                return {
                    "status": "escalated",
                    "final_output": execution_result.output,  # best attempt
                    "iterations": iteration + 1,
                    "history": router.history,
                    "final_verdict": verification,
                }

            # REFINE or REPROMPT: loop continues
            refinement_context = verification.refinement_actions
            # Update workflow_state with latest output for next iteration
            workflow_state = execution_result.updated_workflow
```

#### Phase 5 Success Criteria

- [ ] iterative_refine brain tool is implemented and callable
- [ ] Full pipeline works: intent string → refined output
- [ ] Refinement loop converges (quality improves across iterations)
- [ ] Loop exits correctly on accept, escalate, and max_iterations
- [ ] Error handling works for execution failures
- [ ] History is recorded for debugging/user transparency
- [ ] Test: end-to-end with mocked ComfyUI backend
- [ ] Test: end-to-end with live ComfyUI (integration test)
- [ ] Existing tools continue to work independently of MoE layer

---

## Key Design Decisions (Reference)

### Why the Intent Agent is Tool-Less

The Intent Agent is a pure reasoning layer. It consumes a model profile and produces a structured specification. This separation:
- Enables testing without a running ComfyUI instance
- Keeps the system prompt focused on artistic translation, not API mechanics
- Allows swapping the reasoning model independently of the execution model
- Makes the intent→parameter mapping auditable and debuggable

### Why Profiles are YAML, Not Database

- Human-editable by artists who want to tune model behavior
- Comments document WHY sweet spots exist (institutional knowledge)
- Version-controllable alongside the codebase
- Community-shareable as standalone files
- No database dependency for what is essentially static configuration

### Why the Router is Lightweight

The Router is a sequencer, not a god-agent. It delegates authority, doesn't exercise it. This prevents the common failure mode where a "smart router" becomes a bottleneck that second-guesses specialists. The specialists have domain authority. The Router has control flow authority. These don't overlap.

### Why Authority Boundaries are Hard

If the Execution Agent can decide "I'll change CFG on my own because it seems right," you've lost the architecture's primary value — traceable, auditable intent-to-parameter translation. Every parameter change traces back to either the user's intent (via Intent Agent) or a quality diagnosis (via Verify Agent). Never from an agent acting unilaterally.

### Model Profile as ACCESS Oracle

This is the project's core principle applied to model knowledge: ACCESS > LEARN. The agents don't memorize that Flux works best at CFG 3.5. They query the profile. When a new model drops, you write a profile — you don't retrain agents. When you learn something new about a model, you update one YAML file — not three agent system prompts.

### Why Schemas are User-Facing, Not Just Internal Contracts

The dataclasses (`IntentSpecification`, `VerificationResult`) are the developer-side interface. The YAML schemas are the user-side interface to the same contracts. This dual representation exists because:

- **Artists don't read Python.** They can read YAML. A schema file that says `composition_score: { type: float, description: "Rule of thirds and visual weight balance" }` is self-documenting.
- **Every workflow is different.** A character artist cares about face quality. A matte painter cares about horizon alignment. A motion designer cares about temporal coherence. Hardcoding all possible quality dimensions is impossible — letting users define their own is extensible.
- **The SchemaGenerator closes the loop.** Users don't have to write YAML from scratch. They show the system an example of what they want, and it generates the schema. This is the same "show don't tell" interaction pattern that makes the whole agent accessible to non-developers.
- **Schemas make agents composable.** A pipeline TD can define schemas that ensure agent outputs interoperate with their studio's existing tooling. The agent doesn't need to know about the pipeline — it just populates the schema.

### Why SchemaGenerator is Both a Node and a Tool

As a **ComfyUI node**, it integrates into visual workflows — artists can wire it into their existing workflow graphs. As an **agent tool**, it enables conversational schema creation — "I want the verify agent to also check X." Both entry points produce the same YAML artifact. The node serves artists who think visually; the tool serves artists who think conversationally. Both are valid and both produce the same result.

---

## File Structure (Target)

```
src/
├── profiles/
│   ├── __init__.py
│   ├── loader.py                  # Profile registry and caching
│   ├── _schema.yaml               # Canonical schema reference
│   ├── flux1-dev.yaml             # Flux.1 Dev profile
│   ├── sdxl-base.yaml             # SDXL Base profile
│   ├── default_dit.yaml           # DiT architecture fallback
│   ├── default_unet.yaml          # UNet architecture fallback
│   └── default_video.yaml         # Video model fallback
├── schemas/
│   ├── __init__.py
│   ├── loader.py                  # Schema registry, inheritance, validation
│   ├── generator.py               # SchemaGenerator (infer from examples)
│   ├── intent/
│   │   ├── default.yaml           # IntentSpecification schema
│   │   └── custom/                # User-created intent schemas
│   │       └── .gitkeep
│   ├── execution/
│   │   ├── default.yaml           # ExecutionResult schema
│   │   └── custom/                # User-created execution schemas
│   │       └── .gitkeep
│   ├── verify/
│   │   ├── default.yaml           # VerificationResult schema
│   │   └── custom/                # User-created verify schemas
│   │       └── .gitkeep
│   └── _meta/
│       └── field_types.yaml       # Canonical type definitions
├── agents/
│   ├── __init__.py
│   ├── intent_agent.py            # Intent translation + conflict resolution
│   ├── execution_agent.py         # Workflow mutation + ComfyUI communication
│   ├── verify_agent.py            # Quality judgment + iteration control
│   └── router.py                  # Authority delegation + loop control
├── nodes/
│   └── schema_generator_node.py   # ComfyUI custom node for schema generation
├── tools/
│   ├── brain/
│   │   └── iterative_refine.py    # The co-pilot loop (brain tool)
│   └── schema/
│       └── generate_schema.py     # Agent tool for in-conversation schema creation
└── tests/
    ├── test_profiles/
    │   ├── test_loader.py
    │   ├── test_schema_validation.py
    │   └── test_profile_completeness.py
    ├── test_schemas/
    │   ├── test_schema_loader.py
    │   ├── test_schema_generator.py
    │   ├── test_schema_inheritance.py
    │   └── test_schema_validation.py
    ├── test_agents/
    │   ├── test_intent_agent.py
    │   ├── test_verify_agent.py
    │   ├── test_router.py
    │   ├── test_authority_boundaries.py
    │   └── test_schema_aware_output.py
    └── test_integration/
        ├── test_moe_pipeline_mocked.py
        └── test_moe_pipeline_live.py
```

---

## Bootstrap Order (Summary)

This is AND-node. All pieces must cohere. Work in dependency order:

```
Phase 1:   Model Profiles        ← FOUNDATION. Everything reads from this.
  │
Phase 1.5: Schema System         ← Agent output contracts. Agents produce schema-validated output.
  │
Phase 2:   Intent Agent          ← Consumes profiles + schemas. Pure reasoning, testable in isolation.
  │
Phase 3:   Verify Agent          ← Consumes profiles + schemas. Testable against saved outputs.
  │
Phase 4:   Router                ← Sequences Intent → Execution → Verify. Carries schema config.
  │                                Execution Agent is existing PILOT tools with formal input contract.
  │
Phase 5:   Integration           ← Wire it all together. iterative_refine brain tool.
```

**The Execution Agent is NOT a separate phase** because it's essentially the existing PILOT layer with a formal input contract (IntentSpecification instead of raw user messages). It's the least new work — just add the contract interface on top of existing tools.

**Phase 1.5 is before agents** because agents need to import the schema loader and validator. The schemas themselves are YAML configs, not heavy infrastructure — this phase is lightweight but must exist before agents can produce schema-aware output.

---

## Testing Strategy

### Unit Tests (Per Phase)

Each phase has its own success criteria above. All unit tests should pass before moving to the next phase.

### Integration Tests (Phase 5)

**Mocked Backend:**
- Full pipeline with mocked ComfyUI responses
- Verify refinement loop converges
- Verify escalation works at max_iterations
- Verify authority boundaries aren't violated

**Live Backend:**
- Full pipeline against running ComfyUI
- Real workflow, real model, real output
- Vision analysis on actual generated images
- Measure: does iterative refinement actually improve output quality?

### Regression

- All 518 existing tests continue to pass
- Existing tool functionality is not broken by MoE layer
- MoE layer is additive, not a replacement

---

## Notes for Claude Code

- Read `CLAUDE.md` first — it has project conventions, test patterns, and linting rules
- Existing codebase has 518 passing tests and zero lint warnings. Keep it that way.
- Use existing test patterns (check `tests/` for conventions)
- The existing `execute_with_progress` and `verify_execution` tools are the foundation for Execution and Verify agents respectively
- `dispatch_brain_message` routing may need to be extended for MoE delegation
- The `auto_verify` integration in `execute_with_progress` is a precursor to the Verify Agent
- Profile YAML files should be committed alongside code, not generated at runtime
- Default schema YAML files should also be committed — they ARE the contract definition
- Custom schema files in `schemas/{agent}/custom/` are user-generated and should be `.gitignore`d in the project repo but preserved in user installations
- The `SchemaGenerator` node needs to be registered in the ComfyUI node pack's `__init__.py`
- The `generate_schema` agent tool should be registered as a DISCOVER-layer tool (it helps users discover/create output formats)
- When implementing schema-aware output in agents, the LLM needs the field descriptions in its context to populate custom fields — include the schema fields in the agent's prompt, not just the profile
