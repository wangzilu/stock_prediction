"""Tests for Phase D / SC-A3: supply_chain_edges.yaml tier classification
+ tier-filtered loading."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _add_project_root_to_path():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    yield


def _mod():
    import importlib
    return importlib.import_module("scripts.build_global_chain_factors")


# ── Tier classifier ──────────────────────────────────────────────────

@pytest.mark.parametrize("source,tier", [
    ("公告", "A"),
    ("年报", "A"),
    ("公告/年报", "A"),
    ("政策", "A"),
    ("订单", "B"),
    ("合作", "B"),
    ("认证", "B"),
    ("公司互动", "B"),
    ("供应链公开信息", "B"),
    ("行业研报", "C"),
    ("研报", "C"),
    ("机构调研", "C"),
    ("调研", "C"),
    ("行业逻辑", "D"),
    ("行业常识", "D"),
    ("主题", "D"),
    ("公开信息", "D"),
])
def test_classify_known_sources(source, tier):
    assert _mod().classify_edge_tier(source) == tier


def test_empty_source_defaults_to_d():
    """No source string = fall back to D so production overlays don't
    silently accept an unlabelled edge."""
    assert _mod().classify_edge_tier("") == "D"
    assert _mod().classify_edge_tier("   ") == "D"
    assert _mod().classify_edge_tier(None) == "D"


def test_unknown_source_defaults_to_d():
    assert _mod().classify_edge_tier("foo bar baz") == "D"


# ── load_edges ──────────────────────────────────────────────────────

def test_load_edges_default_b_returns_a_and_b_only():
    """Default min_tier='B' includes A + B but not C / D."""
    edges = _mod().load_edges("B")
    assert edges
    tiers = {e.get("tier") for e in edges}
    assert tiers.issubset({"A", "B"}), (
        f"min_tier=B returned tiers {tiers}; expected only A/B"
    )


def test_load_edges_d_returns_everything():
    edges_all = _mod().load_edges("D")
    edges_b = _mod().load_edges("B")
    assert len(edges_all) >= len(edges_b)


def test_load_edges_explicit_a_only():
    edges = _mod().load_edges(explicit_tiers=frozenset({"A"}))
    assert all(e["tier"] == "A" for e in edges)


def test_load_edges_annotate_default_true():
    edges = _mod().load_edges("D")
    assert all("tier" in e for e in edges)


def test_load_edges_annotate_off_strips_tier_field():
    """When annotate=False the original YAML keys come through unchanged."""
    edges = _mod().load_edges("D", annotate=False)
    assert all("tier" not in e for e in edges)


def test_production_edge_tiers_is_a_and_b():
    """Production overlay opts into A/B only by default."""
    mod = _mod()
    assert mod.PRODUCTION_EDGE_TIERS == frozenset({"A", "B"})
