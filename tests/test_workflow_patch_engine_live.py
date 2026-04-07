"""Regression test for Phase 0.5 packaging fix.

Asserts that the CognitiveGraphEngine is reachable through the
production import path used by `agent mcp` — not just the test-only
`from src.cognitive...` form that pytest's rootdir injection enabled
prior to the Phase 0.5 fix.

If this test fails, the packaging fix has regressed and the MCP
server's PILOT engine is silently dead in production.
"""

import subprocess


def test_workflow_patch_imports_engine_from_top_level():
    """workflow_patch must import CognitiveGraphEngine via `cognitive`,
    not via `src.cognitive`."""
    import agent.tools.workflow_patch as wp

    assert wp.CognitiveGraphEngine is not None
    assert wp.CognitiveGraphEngine.__module__ == "cognitive.core.graph"


def test_create_engine_instantiates_against_minimal_workflow():
    """_create_engine should produce a real engine for a 1-node workflow."""
    from agent.tools.workflow_patch import _create_engine

    minimal = {
        "1": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
        }
    }
    engine = _create_engine(minimal)

    assert engine is not None
    assert engine.__class__.__name__ == "CognitiveGraphEngine"
    assert engine.delta_stack == []
    assert engine.to_api_json() == minimal


def test_no_legacy_src_cognitive_imports_remain():
    """No source file should import from `src.cognitive`.

    This is the adversarial sweep that prevents anyone from accidentally
    adding a new `from src.cognitive` import in the future. Tests are
    excluded from the sweep target — the rename of the test files is
    part of Phase 0.5 and there's no reason a new test file can't use
    the post-fix `from cognitive...` form.
    """
    result = subprocess.run(
        ["grep", "-rln", "from src.cognitive", "agent/", "panel/", "cognitive/"],
        capture_output=True,
        text=True,
    )
    # grep exit code 1 = no matches (the success case)
    # grep exit code 0 = matches found (the failure case)
    assert result.returncode == 1, (
        f"Legacy `from src.cognitive` imports still present:\n{result.stdout}"
    )
