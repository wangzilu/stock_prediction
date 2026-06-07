"""Tests for ``scripts/event_study.py`` — PE-6 (task #145).

Step 1 covers the CLI shape only:
  - --source must be one of the 7 known sources
  - --window parses 'lo,hi' with lo <= hi
  - defaults: --benchmark sh000300, --window -5,5, --out-dir under
    data/storage/event_study

Step 2 covers the event loader for each source, including the
PE-4 / chain XWLB stock-broadcast logic (theme → basket).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import event_study as es


def test_parse_window_default_5_5():
    lo, hi = es.parse_window("-5,5")
    assert (lo, hi) == (-5, 5)


def test_parse_window_asymmetric():
    lo, hi = es.parse_window("-2,10")
    assert (lo, hi) == (-2, 10)


def test_parse_window_rejects_swapped():
    with pytest.raises(ValueError):
        es.parse_window("5,-5")


def test_parse_window_rejects_bad_format():
    with pytest.raises(ValueError):
        es.parse_window("not-a-window")


def test_parse_args_minimum():
    cfg = es.parse_args(
        ["--source", "llm", "--start", "2026-04-01", "--end", "2026-04-30"]
    )
    assert cfg.source == "llm"
    assert cfg.start == "2026-04-01"
    assert cfg.end == "2026-04-30"
    assert cfg.window_lo == -5
    assert cfg.window_hi == 5
    assert cfg.benchmark == es.DEFAULT_BENCHMARK
    assert cfg.top_n is None


def test_parse_args_window_override():
    # Argparse treats a leading "-" in an optional value as a new flag;
    # using "=" makes the assignment explicit.
    cfg = es.parse_args(
        [
            "--source", "pe1",
            "--start", "2024-01-01", "--end", "2024-12-31",
            "--window=-3,10",
        ]
    )
    assert (cfg.window_lo, cfg.window_hi) == (-3, 10)


def test_parse_args_rejects_unknown_source():
    with pytest.raises(SystemExit):
        es.parse_args(
            ["--source", "what", "--start", "2026-01-01", "--end", "2026-01-31"]
        )


def test_supported_sources_match_phase_doc():
    # The 7 sources called out in the task spec.
    assert set(es.SUPPORTED_SOURCES) == {
        "pe1", "pe2", "pe3", "pe4", "llm", "chain_rule", "chain_llm",
    }


# ─────────────────────────────────────────────────────────────────────
# Step 2 — event loaders
# ─────────────────────────────────────────────────────────────────────
def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_load_events_pe1_market_keyed(tmp_path):
    # Two PBC events on different dates inside the requested window.
    _write_jsonl(
        tmp_path / "2024-02-20.jsonl",
        [{"publish_date": "2024-02-20", "policy_stance": "easing"}],
    )
    _write_jsonl(
        tmp_path / "2024-02-21.jsonl",
        [{"publish_date": "2024-02-21", "policy_stance": "tightening"}],
    )
    # One event outside the window — must be filtered.
    _write_jsonl(
        tmp_path / "2023-12-31.jsonl",
        [{"publish_date": "2023-12-31", "policy_stance": "easing"}],
    )
    events = es.load_events(
        source="pe1",
        start="2024-02-01",
        end="2024-02-28",
        events_root=tmp_path,
    )
    # Both in-window events; market-keyed instrument; event_type derived
    # from policy_stance.
    assert len(events) == 2
    assert set(events["instrument"].unique()) == {es.MARKET_INSTRUMENT}
    assert set(events["event_type"].unique()) == {"easing", "tightening"}
    assert set(events["event_date"].dt.strftime("%Y-%m-%d")) == {
        "2024-02-20", "2024-02-21",
    }


def test_load_events_llm_stock_keyed(tmp_path):
    # LLM company event — qlib_code drives instrument.
    _write_jsonl(
        tmp_path / "2026-04-27.jsonl",
        [
            {
                "qlib_code": "sh600519",
                "stock_code": "600519",
                "extract_date": "2026-04-27",
                "event_type": "earnings_beat",
            },
            {
                "qlib_code": "sz000858",
                "stock_code": "000858",
                "extract_date": "2026-04-27",
                "event_type": "regulatory_penalty",
            },
        ],
    )
    events = es.load_events(
        source="llm",
        start="2026-04-01",
        end="2026-04-30",
        events_root=tmp_path,
    )
    assert len(events) == 2
    insts = set(events["instrument"].unique())
    assert insts == {"SH600519", "SZ000858"}
    assert set(events["event_type"].unique()) == {
        "earnings_beat", "regulatory_penalty",
    }


def test_load_events_chain_rule_market_keyed(tmp_path):
    # Chain events have no A-share attribution → market-keyed
    _write_jsonl(
        tmp_path / "2026-05-25.jsonl",
        [
            {
                "date": "2026-05-25",
                "event_type": "capacity_expansion",
                "source_entity": "Nvidia",
                "topic": "ai_server",
            },
        ],
    )
    events = es.load_events(
        source="chain_rule",
        start="2026-05-01",
        end="2026-05-31",
        events_root=tmp_path,
    )
    assert len(events) == 1
    assert events.iloc[0]["instrument"] == es.MARKET_INSTRUMENT
    assert events.iloc[0]["event_type"] == "capacity_expansion"


def test_load_events_filters_window(tmp_path):
    _write_jsonl(
        tmp_path / "2026-04-27.jsonl",
        [
            {"qlib_code": "sh600519", "extract_date": "2026-04-27",
             "event_type": "x"},
        ],
    )
    out = es.load_events(
        source="llm",
        start="2026-05-01",
        end="2026-05-31",
        events_root=tmp_path,
    )
    assert out.empty


def test_load_events_unknown_source_raises(tmp_path):
    with pytest.raises(ValueError):
        es.load_events(
            source="not_a_real_source",
            start="2026-01-01",
            end="2026-01-31",
            events_root=tmp_path,
        )


def test_load_events_missing_dir_returns_empty(tmp_path):
    # Should not raise, just return an empty frame with the right cols.
    out = es.load_events(
        source="pe1",
        start="2026-01-01",
        end="2026-01-31",
        events_root=tmp_path / "does_not_exist",
    )
    assert out.empty
    assert {"event_id", "event_date", "instrument", "event_type"}.issubset(
        out.columns
    )
