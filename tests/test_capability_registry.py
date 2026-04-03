"""CRUCIBLE tests for capability registry (agent/tools/capability_registry.py).

Adversarial: tests empty states, duplicate registration, immutability,
and the default capabilities population.
"""

from __future__ import annotations

import threading

import pytest

from agent.tools.capability_registry import ToolCapability, ToolCapabilityRegistry
from agent.tools.capability_defaults import build_default_capabilities


# ---------------------------------------------------------------------------
# Registry basics
# ---------------------------------------------------------------------------


class TestRegistryBasics:
    def test_registry_empty_initially(self):
        reg = ToolCapabilityRegistry()
        assert reg.all_tools() == []

    def test_register_and_get(self):
        reg = ToolCapabilityRegistry()
        cap = ToolCapability(tool_name="test_tool", risk_level=0, phase="understand")
        reg.register(cap)
        result = reg.get("test_tool")
        assert result is not None
        assert result.tool_name == "test_tool"

    def test_get_missing_returns_none(self):
        reg = ToolCapabilityRegistry()
        assert reg.get("nonexistent") is None

    def test_register_overwrites(self):
        reg = ToolCapabilityRegistry()
        cap1 = ToolCapability(tool_name="t", risk_level=0)
        cap2 = ToolCapability(tool_name="t", risk_level=3)
        reg.register(cap1)
        reg.register(cap2)
        result = reg.get("t")
        assert result is not None
        assert result.risk_level == 3

    def test_register_batch(self):
        reg = ToolCapabilityRegistry()
        caps = [
            ToolCapability(tool_name=f"t{i}", risk_level=i % 5)
            for i in range(10)
        ]
        reg.register_batch(caps)
        assert len(reg.all_tools()) == 10

    def test_all_tools_sorted_by_name(self):
        reg = ToolCapabilityRegistry()
        reg.register(ToolCapability(tool_name="zebra"))
        reg.register(ToolCapability(tool_name="alpha"))
        reg.register(ToolCapability(tool_name="mid"))
        names = [c.tool_name for c in reg.all_tools()]
        assert names == ["alpha", "mid", "zebra"]


# ---------------------------------------------------------------------------
# Query / selection tests
# ---------------------------------------------------------------------------


class TestRegistrySelect:
    @pytest.fixture()
    def populated_registry(self) -> ToolCapabilityRegistry:
        reg = ToolCapabilityRegistry()
        reg.register_batch([
            ToolCapability(
                tool_name="read_tool", risk_level=0, phase="understand",
                requires_comfyui=True,
            ),
            ToolCapability(
                tool_name="edit_tool", risk_level=1, phase="pilot",
                mutates_workflow=True,
            ),
            ToolCapability(
                tool_name="execute_tool", risk_level=2, phase="pilot",
                requires_comfyui=True, latency_class="batch",
            ),
            ToolCapability(
                tool_name="provision_tool", risk_level=3, phase="discover",
            ),
            ToolCapability(
                tool_name="destroy_tool", risk_level=4, phase="discover",
            ),
            ToolCapability(
                tool_name="brain_tool", risk_level=0, phase="verify",
                requires_brain=True,
            ),
            ToolCapability(
                tool_name="any_phase_tool", risk_level=0, phase="any",
            ),
        ])
        return reg

    def test_select_by_phase(self, populated_registry: ToolCapabilityRegistry):
        results = populated_registry.select({"phase": "pilot"})
        names = {c.tool_name for c in results}
        assert "edit_tool" in names
        assert "execute_tool" in names
        # "any" phase tools should also match
        assert "any_phase_tool" in names
        assert "read_tool" not in names

    def test_select_by_risk(self, populated_registry: ToolCapabilityRegistry):
        results = populated_registry.select({"max_risk": 1})
        for cap in results:
            assert cap.risk_level <= 1
        names = {c.tool_name for c in results}
        assert "execute_tool" not in names
        assert "provision_tool" not in names
        assert "destroy_tool" not in names

    def test_select_requires_comfyui(self, populated_registry: ToolCapabilityRegistry):
        results = populated_registry.select({"requires_comfyui": True})
        assert all(c.requires_comfyui for c in results)
        names = {c.tool_name for c in results}
        assert "read_tool" in names
        assert "execute_tool" in names
        assert "edit_tool" not in names

    def test_select_best_returns_lowest_risk(self, populated_registry: ToolCapabilityRegistry):
        best = populated_registry.select_best({"requires_comfyui": True})
        assert best is not None
        assert best.risk_level == 0
        assert best.tool_name == "read_tool"

    def test_select_no_matches_returns_empty(self, populated_registry: ToolCapabilityRegistry):
        results = populated_registry.select({
            "requires_comfyui": True,
            "requires_brain": True,
        })
        assert results == []

    def test_select_best_no_match_returns_none(self, populated_registry: ToolCapabilityRegistry):
        best = populated_registry.select_best({
            "requires_comfyui": True,
            "requires_brain": True,
        })
        assert best is None

    def test_select_sorted_by_risk_then_latency(self, populated_registry: ToolCapabilityRegistry):
        results = populated_registry.select({})
        risks = [c.risk_level for c in results]
        assert risks == sorted(risks)

    def test_by_phase_returns_only_matching(self, populated_registry: ToolCapabilityRegistry):
        results = populated_registry.by_phase("verify")
        for cap in results:
            assert cap.phase in ("verify", "any"), f"Unexpected phase: {cap.phase}"

    def test_by_phase_includes_any(self, populated_registry: ToolCapabilityRegistry):
        results = populated_registry.by_phase("understand")
        names = {c.tool_name for c in results}
        assert "any_phase_tool" in names

    def test_by_risk_returns_bounded(self, populated_registry: ToolCapabilityRegistry):
        results = populated_registry.by_risk(2)
        for cap in results:
            assert cap.risk_level <= 2

    def test_select_mutates_workflow(self, populated_registry: ToolCapabilityRegistry):
        results = populated_registry.select({"mutates_workflow": True})
        assert all(c.mutates_workflow for c in results)


# ---------------------------------------------------------------------------
# Capability immutability
# ---------------------------------------------------------------------------


class TestCapabilityImmutability:
    def test_capability_frozen(self):
        cap = ToolCapability(tool_name="t", risk_level=0)
        with pytest.raises(AttributeError):
            cap.tool_name = "hacked"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            cap.risk_level = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Default capabilities
# ---------------------------------------------------------------------------


class TestDefaultCapabilities:
    def test_default_capabilities_count(self):
        caps = build_default_capabilities()
        assert len(caps) >= 100, f"Only {len(caps)} default capabilities (expected 100+)"

    def test_default_capabilities_no_duplicates(self):
        caps = build_default_capabilities()
        names = [c.tool_name for c in caps]
        dupes = [n for n in names if names.count(n) > 1]
        assert len(dupes) == 0, f"Duplicate tool names: {set(dupes)}"

    def test_default_capabilities_all_have_names(self):
        caps = build_default_capabilities()
        for cap in caps:
            assert cap.tool_name, f"Empty tool_name in capability: {cap}"

    def test_default_capabilities_valid_phases(self):
        valid = {"understand", "discover", "pilot", "verify", "any"}
        caps = build_default_capabilities()
        for cap in caps:
            assert cap.phase in valid, f"{cap.tool_name} has invalid phase: {cap.phase}"

    def test_default_capabilities_valid_risk_levels(self):
        caps = build_default_capabilities()
        for cap in caps:
            assert 0 <= cap.risk_level <= 4, (
                f"{cap.tool_name} has invalid risk: {cap.risk_level}"
            )

    def test_default_capabilities_populate_registry(self):
        reg = ToolCapabilityRegistry()
        caps = build_default_capabilities()
        reg.register_batch(caps)
        assert len(reg.all_tools()) == len(caps)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestRegistryThreadSafety:
    def test_concurrent_register_and_select(self):
        reg = ToolCapabilityRegistry()
        errors: list[str] = []

        def register_many(start: int) -> None:
            try:
                for i in range(50):
                    cap = ToolCapability(
                        tool_name=f"tool_{start}_{i}",
                        risk_level=i % 5,
                    )
                    reg.register(cap)
            except Exception as exc:
                errors.append(str(exc))

        def select_many() -> None:
            try:
                for _ in range(50):
                    reg.select({"max_risk": 2})
                    reg.all_tools()
            except Exception as exc:
                errors.append(str(exc))

        threads = [
            threading.Thread(target=register_many, args=(tid * 100,))
            for tid in range(5)
        ] + [
            threading.Thread(target=select_many)
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        # All 250 registrations should have landed
        assert len(reg.all_tools()) == 250
