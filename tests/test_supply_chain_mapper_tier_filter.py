"""Integration test for SC-A3 tier filter — the real path.

cx review 2026-06-06 (P1): the SC-A1/A3 unit tests monkeypatched the
mapper, so they did NOT catch the case where ``SupplyChainMapper``
read the YAML directly and re-injected C/D edges that the top-level
``load_edges`` had filtered out. This test exercises the full
``SupplyChainMapper`` path (no monkeypatch on the mapper class itself)
against a temp YAML that contains both A and D edges.

The contract under test: when the mapper is initialised with
``min_tier="B"`` or ``explicit_tiers={"A","B"}``, only the A/B edges
appear in ``get_all_affected_stocks``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


@pytest.fixture(autouse=True)
def _add_project_root_to_path():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    yield


def _patch_edges_path(monkeypatch, tmp_yaml):
    """Redirect the YAML path that both
    ``scripts.build_global_chain_factors`` and
    ``factors.supply_chain_mapper`` read.
    """
    import scripts.build_global_chain_factors as builder
    import factors.supply_chain_mapper as mapper
    monkeypatch.setattr(builder, "EDGES_PATH", tmp_yaml)
    monkeypatch.setattr(mapper, "EDGES_PATH", tmp_yaml)


def _write_yaml(tmp_path, rows):
    p = tmp_path / "supply_chain_edges.yaml"
    p.write_text(yaml.safe_dump(rows, allow_unicode=True), encoding="utf-8")
    return p


@pytest.fixture
def edge_yaml(tmp_path):
    """Two tier-A (公告 / 年报) edges and two tier-D (主题 / 行业逻辑)
    edges, all pointing at different stocks so we can assert which ones
    survive the filter."""
    rows = [
        # A — should pass production filter
        {"src_entity": "Nvidia", "topic": "AI_server",
         "dst_stock": "sz300308", "dst_name": "中际旭创",
         "direction": 1, "weight": 0.9, "confidence": 0.9,
         "source": "公告"},
        {"src_entity": "Nvidia", "topic": "AI_server",
         "dst_stock": "sh601138", "dst_name": "工业富联",
         "direction": 1, "weight": 0.8, "confidence": 0.9,
         "source": "年报"},
        # D — must be dropped by production filter
        {"src_entity": "Nvidia", "topic": "AI_server",
         "dst_stock": "sz000001", "dst_name": "平安银行",
         "direction": 1, "weight": 0.2, "confidence": 0.3,
         "source": "行业逻辑"},
        {"src_entity": "Nvidia", "topic": "AI_server",
         "dst_stock": "sh600519", "dst_name": "贵州茅台",
         "direction": 1, "weight": 0.1, "confidence": 0.2,
         "source": "主题"},
    ]
    return rows


def test_supply_chain_mapper_drops_d_tier_edges(edge_yaml, tmp_path, monkeypatch):
    """The real mapper must NOT carry tier-D edges through to its
    company-level company_edges list when initialised with the
    production tier set."""
    p = _write_yaml(tmp_path, edge_yaml)
    _patch_edges_path(monkeypatch, p)

    from factors.supply_chain_mapper import SupplyChainMapper

    m = SupplyChainMapper(explicit_tiers=frozenset({"A", "B"}))

    # Only the two A-tier edges should survive.
    kept_stocks = {e["dst_stock"] for e in m._company_edges}
    assert kept_stocks == {"sz300308", "sh601138"}, (
        f"production mapper carried unexpected edges: {kept_stocks}"
    )


def test_supply_chain_mapper_default_min_tier_is_b():
    """Production default: ``min_tier='B'`` (= A + B)."""
    from factors.supply_chain_mapper import SupplyChainMapper
    m = SupplyChainMapper.__new__(SupplyChainMapper)
    # Inspect default arg without invoking __init__'s YAML read.
    import inspect
    sig = inspect.signature(SupplyChainMapper.__init__)
    assert sig.parameters["min_tier"].default == "B"


def test_supply_chain_mapper_d_tier_opt_in(edge_yaml, tmp_path, monkeypatch):
    """Research / shadow callers can opt into D explicitly."""
    p = _write_yaml(tmp_path, edge_yaml)
    _patch_edges_path(monkeypatch, p)

    from factors.supply_chain_mapper import SupplyChainMapper

    m = SupplyChainMapper(min_tier="D")
    kept_stocks = {e["dst_stock"] for e in m._company_edges}
    # All 4 stocks survive when D is allowed.
    assert kept_stocks == {"sz300308", "sh601138", "sz000001", "sh600519"}
