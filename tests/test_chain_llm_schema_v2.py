"""Tests for Phase D / SC-A2: global supply chain LLM extractor v2 schema.

The v2 schema replaces direction/confidence with explicit RELATIONS so
the LLM never predicts stock direction. These tests assert:

  (a) the JSON schema invariants the parser enforces (mandatory fields,
      enum values for relation_type / evidence_strength / factuality)
  (b) the system prompt explicitly forbids direction prediction
  (c) unknown ``evidence_strength`` downgrades to ``"D"``
  (d) rows missing ``src_entity`` or ``relations`` are refused

Plus an integration test for ``build_global_chain_factors --schema v2``
that exercises the v2 → propagation path end-to-end against a tmp YAML
edge graph and asserts parquet rows land at the right stocks.
"""
from __future__ import annotations

import json
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


# ─── (a) Schema validation ───────────────────────────────────────────

def test_parse_response_v2_happy_path():
    """A well-formed v2 LLM response round-trips through the parser."""
    from factors.global_chain_llm_extractor import parse_response_v2
    raw = json.dumps({
        "is_supply_chain_event": True,
        "src_entity": "Nvidia",
        "relations": [
            {"target_entity": "TSMC", "relation_type": "supplier",
             "evidence_strength": "A"},
            {"target_entity": "Samsung", "relation_type": "customer",
             "evidence_strength": "C"},
        ],
        "topic": "AI_server",
        "factuality": "confirmed",
        "summary": "Nvidia Blackwell ramps demand on TSMC.",
    })
    parsed = parse_response_v2(raw)
    assert parsed is not None
    assert parsed["src_entity"] == "Nvidia"
    assert parsed["schema_version"] == "v2"
    assert parsed["factuality"] == "confirmed"
    assert parsed["topic"] == "AI_server"
    assert len(parsed["relations"]) == 2
    assert parsed["relations"][0]["target_entity"] == "TSMC"
    assert parsed["relations"][0]["relation_type"] == "supplier"
    assert parsed["relations"][0]["evidence_strength"] == "A"
    # MUST NOT contain a direction field — that's the whole point of v2.
    assert "direction" not in parsed
    assert "confidence" not in parsed


def test_parse_response_v2_rejects_unknown_relation_type():
    """Unknown ``relation_type`` values are dropped (we won't invent
    an enum value). When that drops the row to zero relations the
    whole event is refused."""
    from factors.global_chain_llm_extractor import parse_response_v2
    raw = json.dumps({
        "is_supply_chain_event": True,
        "src_entity": "Nvidia",
        "relations": [
            {"target_entity": "TSMC",
             "relation_type": "spaghetti_friend",   # not in enum
             "evidence_strength": "A"},
        ],
        "factuality": "confirmed",
        "summary": "x",
    })
    assert parse_response_v2(raw) is None


def test_parse_response_v2_factuality_enum():
    """``factuality`` outside the allowed bucket gets the conservative
    default ``"speculation"``."""
    from factors.global_chain_llm_extractor import parse_response_v2
    raw = json.dumps({
        "is_supply_chain_event": True,
        "src_entity": "Apple",
        "relations": [
            {"target_entity": "Foxconn", "relation_type": "supplier",
             "evidence_strength": "B"},
        ],
        "factuality": "definitely-true-trust-me",   # not in enum
        "summary": "x",
    })
    parsed = parse_response_v2(raw)
    assert parsed is not None
    assert parsed["factuality"] == "speculation"


# ─── (b) Prompt forbids direction prediction ─────────────────────────

def test_v2_system_prompt_forbids_direction():
    """The v2 system prompt MUST tell the LLM not to output direction.
    This is the load-bearing constraint of SC-A2 and a regression here
    would silently re-introduce stance prediction."""
    from factors.global_chain_llm_extractor import SYSTEM_PROMPT_V2
    p = SYSTEM_PROMPT_V2.lower()
    assert "do not predict stock direction" in p, (
        "v2 system prompt must explicitly forbid stock-direction prediction"
    )
    # And no instruction to OUTPUT a direction field
    assert "do not include a 'direction' field" in p


def test_v2_system_prompt_demands_evidence_default_d():
    """SC-A3 evidence tiers must default to D under uncertainty.
    The prompt has to say so explicitly so the LLM doesn't invent a tier."""
    from factors.global_chain_llm_extractor import SYSTEM_PROMPT_V2
    p = SYSTEM_PROMPT_V2.lower()
    assert "assign d" in p or "assign d " in p, (
        "v2 system prompt must instruct the LLM to default evidence_strength=D "
        "when it cannot judge"
    )


def test_v2_system_prompt_mentions_yaml_vocab():
    """Spec demands the prompt steer the LLM toward names the
    supply_chain_edges.yaml graph already knows about."""
    from factors.global_chain_llm_extractor import SYSTEM_PROMPT_V2
    # A handful of canonical names should appear verbatim.
    for name in ("Nvidia", "TSMC", "Apple"):
        assert name in SYSTEM_PROMPT_V2, (
            f"v2 system prompt should hint at the YAML vocabulary "
            f"({name!r} missing)"
        )


# ─── (c) Unknown evidence_strength downgrades to D ───────────────────

def test_parse_response_v2_unknown_evidence_strength_downgrades_to_d():
    """Spec rule (c): unknown ``evidence_strength`` becomes ``"D"``."""
    from factors.global_chain_llm_extractor import parse_response_v2
    raw = json.dumps({
        "is_supply_chain_event": True,
        "src_entity": "Tesla",
        "relations": [
            {"target_entity": "CATL", "relation_type": "supplier",
             "evidence_strength": "Z"},   # unknown
            {"target_entity": "BYD", "relation_type": "competitor",
             "evidence_strength": ""},    # empty
        ],
        "factuality": "speculation",
        "summary": "x",
    })
    parsed = parse_response_v2(raw)
    assert parsed is not None
    for rel in parsed["relations"]:
        assert rel["evidence_strength"] == "D"


# ─── (d) Mandatory-field refusal ─────────────────────────────────────

def test_parse_response_v2_refuses_missing_src_entity():
    """Spec rule (d): no ``src_entity`` ⇒ refuse the row."""
    from factors.global_chain_llm_extractor import parse_response_v2
    raw = json.dumps({
        "is_supply_chain_event": True,
        # src_entity missing
        "relations": [
            {"target_entity": "TSMC", "relation_type": "supplier",
             "evidence_strength": "A"},
        ],
        "factuality": "confirmed",
        "summary": "x",
    })
    assert parse_response_v2(raw) is None


def test_parse_response_v2_refuses_missing_relations():
    """Spec rule (d): no ``relations`` ⇒ refuse the row."""
    from factors.global_chain_llm_extractor import parse_response_v2
    raw = json.dumps({
        "is_supply_chain_event": True,
        "src_entity": "Apple",
        # relations missing
        "factuality": "confirmed",
        "summary": "x",
    })
    assert parse_response_v2(raw) is None

    raw_empty = json.dumps({
        "is_supply_chain_event": True,
        "src_entity": "Apple",
        "relations": [],   # explicitly empty
        "factuality": "confirmed",
        "summary": "x",
    })
    assert parse_response_v2(raw_empty) is None


def test_parse_response_v2_refuses_non_supply_chain():
    """When ``is_supply_chain_event=false`` the row is filtered out
    regardless of other fields being well-formed."""
    from factors.global_chain_llm_extractor import parse_response_v2
    raw = json.dumps({
        "is_supply_chain_event": False,
        "src_entity": "Nvidia",
        "relations": [
            {"target_entity": "TSMC", "relation_type": "supplier",
             "evidence_strength": "A"},
        ],
        "factuality": "confirmed",
        "summary": "generic commentary",
    })
    assert parse_response_v2(raw) is None


# ─── Integration: v2 → build_global_chain_factors ────────────────────

def _write_yaml(tmp_path, rows):
    p = tmp_path / "supply_chain_edges.yaml"
    p.write_text(yaml.safe_dump(rows, allow_unicode=True), encoding="utf-8")
    return p


@pytest.fixture
def edge_yaml(tmp_path):
    """A 2-edge A-tier graph: Nvidia → 中际旭创, TSMC → 中芯国际.

    The v2 adapter emits a v1-shaped event for BOTH ends of every
    relation (src + target) so the propagation should hit at least
    one of these stocks when the v2 event is "Nvidia supplier TSMC".
    """
    rows = [
        {"src_entity": "Nvidia", "topic": "AI_server",
         "dst_stock": "sz300308", "dst_name": "中际旭创",
         "direction": 1, "weight": 0.9, "confidence": 0.9,
         "source": "公告"},
        {"src_entity": "TSMC", "topic": "semiconductor",
         "dst_stock": "sh688981", "dst_name": "中芯国际",
         "direction": 1, "weight": 0.7, "confidence": 0.8,
         "source": "公告"},
    ]
    return rows


def test_v2_event_to_v1_shape_emits_both_ends():
    """A v2 row with ``{src=Nvidia, relations=[{target=TSMC}]}`` should
    produce v1 events keyed on BOTH Nvidia and TSMC so the propagation
    can hit either side of the YAML graph."""
    from scripts.build_global_chain_factors import _v2_event_to_v1_shape
    v2 = {
        "src_entity": "Nvidia",
        "relations": [
            {"target_entity": "TSMC", "relation_type": "supplier",
             "evidence_strength": "A"},
        ],
        "factuality": "confirmed",
        "topic": "AI_server",
        "date": "2026-06-07",
    }
    out = _v2_event_to_v1_shape(v2)
    src_entities = {e["source_entity"] for e in out}
    assert src_entities == {"Nvidia", "TSMC"}
    # Direction is derived from relation_type — NOT extracted by LLM
    for e in out:
        assert e["direction"] == +1   # 'supplier' is +1 in the table
        assert "confidence" in e and e["confidence"] > 0
        assert e["schema_version"] == "v2"


def test_v2_event_to_v1_shape_competitor_inverts():
    """``competitor`` relation_type inverts direction to -1."""
    from scripts.build_global_chain_factors import _v2_event_to_v1_shape
    v2 = {
        "src_entity": "Nvidia",
        "relations": [
            {"target_entity": "AMD", "relation_type": "competitor",
             "evidence_strength": "B"},
        ],
        "factuality": "confirmed",
    }
    out = _v2_event_to_v1_shape(v2)
    assert all(e["direction"] == -1 for e in out)


def test_build_factors_schema_v2_end_to_end(edge_yaml, tmp_path, monkeypatch):
    """Integration: write a v2-shaped JSONL, call ``build_factors`` with
    ``schema='v2'``, assert the propagated parquet contains the
    expected stocks. Also asserts the parquet schema (column set) is
    unchanged from v1 so FeatureMerger does not need a code change."""
    import scripts.build_global_chain_factors as builder
    import factors.supply_chain_mapper as mapper_mod

    # Repoint the edge YAML used by both the builder and the mapper.
    p = _write_yaml(tmp_path, edge_yaml)
    monkeypatch.setattr(builder, "EDGES_PATH", p)
    monkeypatch.setattr(mapper_mod, "EDGES_PATH", p)

    # Set up an isolated v2 events dir + parquet output dir.
    v2_dir = tmp_path / "events_v2"
    v2_dir.mkdir()
    out_parquet = tmp_path / "factors_llm.parquet"
    monkeypatch.setattr(builder, "EVENTS_DIR_LLM_V2", v2_dir)
    monkeypatch.setattr(builder, "OUTPUT_PATH_LLM", out_parquet)

    target_date = "2026-06-07"
    v2_event = {
        "src_entity": "Nvidia",
        "relations": [
            {"target_entity": "TSMC", "relation_type": "supplier",
             "evidence_strength": "A"},
        ],
        "topic": "AI_server",
        "factuality": "confirmed",
        "summary": "Nvidia ramps Blackwell on TSMC capacity.",
        "schema_version": "v2",
        "source": "llm",
        "date": target_date,
        "news_title": "Nvidia Blackwell demand strong",
    }
    (v2_dir / f"{target_date}.jsonl").write_text(
        json.dumps(v2_event, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    df = builder.build_factors(target_date, schema="v2", lookback_days=2)
    assert not df.empty, "v2 path should produce factor rows"

    # Parquet must have the v1-compatible column set.
    expected_cols = {
        "global_chain_alpha",
        "global_chain_event_count",
        "global_chain_pos_score",
        "global_chain_neg_score",
        "company_level_alpha",
        "industry_level_alpha",
        "level",
    }
    assert expected_cols.issubset(set(df.columns)), (
        f"v2 parquet schema diverged from v1; got {set(df.columns)}"
    )

    instruments = {inst.lower() for _, inst in df.index}
    # The Nvidia→中际旭创 edge AND the TSMC→中芯国际 edge should both
    # fire because the v2 adapter expands the relation into events
    # keyed on both endpoints.
    assert "sz300308" in instruments, (
        f"expected 中际旭创 (Nvidia supplier) in instruments, got {instruments}"
    )
    assert "sh688981" in instruments, (
        f"expected 中芯国际 (TSMC supplier) in instruments, got {instruments}"
    )

    # Output parquet was actually written to disk.
    assert out_parquet.exists()
