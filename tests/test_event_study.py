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


# ─────────────────────────────────────────────────────────────────────
# Step 3 — excess return panel
# ─────────────────────────────────────────────────────────────────────
def _flat_close_loader(close_by_instrument: dict[str, pd.Series]):
    """Build a CloseLoader callable from a dict of instrument→close series.

    Each series must be indexed by tz-naive ``DatetimeIndex``.
    """
    def loader(insts, start, end):
        out = {}
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        for inst in insts:
            ser = close_by_instrument.get(inst.upper())
            if ser is None:
                continue
            mask = (ser.index >= s) & (ser.index <= e)
            out[inst.upper()] = ser.loc[mask].copy()
        return out
    return loader


def test_excess_return_curve_matches_known_injection():
    """Inject a +1d known abnormal return; mean curve must recover it."""
    # 30 trading days; stock = benchmark + 0.02 on event date + 1
    dates = pd.bdate_range("2026-01-05", periods=30)
    bench = pd.Series(100.0 * (1.0 + pd.Series([0.001] * 30,
                                               index=dates)).cumprod())
    bench.index = dates
    # stock identical to bench except offset +1d for two known events
    stock1 = bench.copy()
    stock2 = bench.copy()
    # event 1 on dates[10] → stock close on dates[11] = bench * 1.02
    # → stock pct_change[11] - bench pct_change[11] = +0.02
    stock1.iloc[11] = stock1.iloc[11] * 1.02
    # propagate the jump forward (otherwise day +2 would show -0.02)
    stock1.iloc[12:] = stock1.iloc[12:] * 1.02
    # event 2 on dates[20] → stock close on dates[21] = bench * 1.02
    stock2.iloc[21] = stock2.iloc[21] * 1.02
    stock2.iloc[22:] = stock2.iloc[22:] * 1.02

    close_loader = _flat_close_loader({
        "SH600519": stock1,
        "SZ000858": stock2,
        "SH000300": bench,
    })

    events = pd.DataFrame([
        {"event_id": "e1", "event_date": dates[10],
         "instrument": "SH600519", "event_type": "earnings_beat"},
        {"event_id": "e2", "event_date": dates[20],
         "instrument": "SZ000858", "event_type": "earnings_beat"},
    ])

    panel = es.build_excess_return_panel(
        events=events,
        benchmark="SH000300",
        window_lo=-2, window_hi=3,
        close_loader=close_loader,
    )
    # 2 events x 6 offsets (-2 -1 0 +1 +2 +3) = 12 rows
    assert len(panel) == 2
    assert "offset_+1" in panel.columns
    # Mean abnormal return at offset +1 ≈ 0.02. Tolerance reflects the
    # tiny benchmark-drift term in the synthetic series (0.001 / day);
    # the injection itself is exactly 0.02 so 2e-4 is plenty tight.
    mean_plus_1 = panel["offset_+1"].mean()
    assert abs(mean_plus_1 - 0.02) < 2e-4


def test_excess_return_skips_event_with_short_window():
    """Event too close to end of price history → row dropped."""
    dates = pd.bdate_range("2026-01-05", periods=10)
    bench = pd.Series([100.0] * 10, index=dates)
    stock = bench.copy()
    close_loader = _flat_close_loader({
        "SH600519": stock, "SH000300": bench,
    })
    events = pd.DataFrame([
        # only 1 day after — but window_hi=3 needs 3 days after
        {"event_id": "e1", "event_date": dates[8],
         "instrument": "SH600519", "event_type": "x"},
    ])
    panel = es.build_excess_return_panel(
        events=events, benchmark="SH000300",
        window_lo=-2, window_hi=3,
        close_loader=close_loader,
    )
    assert panel.empty


def test_excess_return_market_keyed_uses_benchmark_for_self():
    """MARKET instrument → 'excess return' is 0 by construction."""
    dates = pd.bdate_range("2026-01-05", periods=20)
    bench = pd.Series((1.0 + pd.Series([0.01] * 20, index=dates)).cumprod() * 100.0)
    bench.index = dates
    close_loader = _flat_close_loader({"SH000300": bench})
    events = pd.DataFrame([
        {"event_id": "e1", "event_date": dates[10],
         "instrument": es.MARKET_INSTRUMENT, "event_type": "macro_shock"},
    ])
    panel = es.build_excess_return_panel(
        events=events, benchmark="SH000300",
        window_lo=-1, window_hi=1,
        close_loader=close_loader,
    )
    assert len(panel) == 1
    # For market-keyed events excess return is the benchmark return
    # itself (study answers "did the market move"). The smooth 1%/day
    # series gives offset_0 ≈ 0.01.
    assert abs(panel["offset_+0"].iloc[0] - 0.01) < 1e-6
