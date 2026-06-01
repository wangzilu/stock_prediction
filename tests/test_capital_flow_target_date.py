"""PIT-safety tests for _load_capital_flow_signals(target_date=...).

Per cx code review 2026-05-31 P2 #18: the method previously took no
target_date parameter and always returned the latest 5 trading days
of capital flow / northbound. Any backfill or snapshot-roundtrip path
that called it on a historical morning_recommendation reconstruction
would leak FUTURE capital flow into the HISTORICAL window.

This file proves two contracts after the fix:
  1. **Backwards-compat**: target_date=None preserves the original
     behavior (latest 5 trading days).
  2. **PIT-safe**: when target_date is passed, only rows with
     trade_date <= target_date are considered, and the cache is
     bypassed so live and backfill paths don't corrupt each other.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from scheduler.jobs import DailyPipeline


@pytest.fixture
def pipeline_with_flow_fixture(tmp_path, monkeypatch):
    """A DailyPipeline plus a fund_flow_history.parquet covering 10 trading
    days, each day has 3 stocks. Northbound parquet absent (we test only
    the fund-flow path; northbound uses the same target_date guard so the
    contract is symmetric)."""
    # Build a fixture parquet with 10 days × 3 stocks
    rows = []
    dates = [
        "2026-05-20", "2026-05-21", "2026-05-22", "2026-05-23", "2026-05-26",
        "2026-05-27", "2026-05-28", "2026-05-29", "2026-05-30", "2026-06-02",
    ]
    stocks = ["SH600519", "SZ000001", "SZ300750"]
    for d in dates:
        for s in stocks:
            # net_mf_amount differs per day so we can distinguish which
            # window the function actually used.
            #   2026-05-20 → 100000 (oldest, distinct value)
            #   2026-06-02 → 1000000 (latest, distinct value)
            day_num = dates.index(d)
            rows.append({
                "qlib_code": s,
                "trade_date": d,
                "net_mf_amount": (day_num + 1) * 100000.0,
            })
    df = pd.DataFrame(rows)
    flow_path = tmp_path / "fund_flow_history.parquet"
    df.to_parquet(flow_path)

    # Patch DATA_DIR to point at our fixture
    monkeypatch.setattr("scheduler.jobs.DATA_DIR", tmp_path)

    # Build a minimal pipeline (skip __init__ which touches networks)
    pipeline = DailyPipeline.__new__(DailyPipeline)
    pipeline.market_collector = MagicMock()
    pipeline.crypto_collector = MagicMock()
    pipeline.gold_collector = MagicMock()
    pipeline.sentiment_collector = MagicMock()
    pipeline.macro_collector = MagicMock()
    pipeline.sentiment_scorer = MagicMock()
    pipeline.signal_scorer = MagicMock()
    pipeline.global_indices = MagicMock()
    pipeline.risk_monitor = MagicMock()
    pipeline.pusher = MagicMock()
    pipeline.verifier = MagicMock()
    pipeline.market_judge = MagicMock()
    pipeline.llm_analyst = MagicMock()
    pipeline.index_predictor = MagicMock()
    pipeline._geo_factors = None
    pipeline._headlines = None
    pipeline._capital_flow_signals = None
    pipeline._lgb_predictions = None
    pipeline._lgb_status = {"status": "unknown", "count": 0, "error": ""}
    pipeline._rl_agent = None
    pipeline._mid_model = None
    pipeline._mid_model_checked = False
    return pipeline


def test_target_date_none_preserves_latest_5_days(pipeline_with_flow_fixture):
    """Backwards-compat: target_date=None returns the same windowing
    behavior the live cron path always had — latest 5 trading days."""
    pipeline = pipeline_with_flow_fixture

    signals = pipeline._load_capital_flow_signals(target_date=None)

    # Latest 5 distinct trading dates in fixture:
    #   2026-05-28, 2026-05-29, 2026-05-30, 2026-06-02 → only 4 weekday
    # actually 10 fixture dates, latest 5 = 5-28/29/30, 6-02 plus 5-27
    # Sum of net_mf for stocks across days 6..10 (0-indexed) =
    #   (6+7+8+9+10) * 100000 = 4_000_000 per stock
    assert len(signals) == 3
    for code in ("SH600519", "SZ000001", "SZ300750"):
        assert code in signals
        assert signals[code]["net_mf_5d"] == pytest.approx(4_000_000.0)
        # net_mf = latest day = 2026-06-02 = day 10 = 1_000_000
        assert signals[code]["net_mf"] == pytest.approx(1_000_000.0)


def test_target_date_historical_excludes_future_days(pipeline_with_flow_fixture):
    """PIT-safe: target_date=2026-05-26 must NOT see fund flow from
    2026-05-27 onwards. Latest 5 dates within window = 2026-05-20/21/22/
    23/26 → sum = (1+2+3+4+5)*100000 = 1_500_000 per stock; net_mf =
    day 5 = 500_000."""
    pipeline = pipeline_with_flow_fixture

    signals = pipeline._load_capital_flow_signals(target_date="2026-05-26")

    assert len(signals) == 3
    for code in ("SH600519", "SZ000001", "SZ300750"):
        assert signals[code]["net_mf_5d"] == pytest.approx(1_500_000.0), (
            f"PIT LEAK: {code} sum {signals[code]['net_mf_5d']:,.0f} != expected "
            "1,500,000 (sum of fund flow for 5-20 through 5-26). Likely the "
            "function read trade_date > 2026-05-26."
        )
        assert signals[code]["net_mf"] == pytest.approx(500_000.0), (
            f"PIT LEAK: {code} latest net_mf {signals[code]['net_mf']:,.0f} != "
            "500,000 (2026-05-26 fund flow). Future day leaked in."
        )


def test_target_date_does_not_pollute_live_cache(pipeline_with_flow_fixture):
    """Backfill calls must NOT populate the live cache (which would
    corrupt subsequent target_date=None calls). Verify by calling
    historical first, then live, and asserting live still returns the
    latest 5 days regardless of the historical call."""
    pipeline = pipeline_with_flow_fixture

    # Backfill call (historical)
    hist = pipeline._load_capital_flow_signals(target_date="2026-05-22")
    assert hist["SH600519"]["net_mf"] == pytest.approx(300_000.0)  # 2026-05-22 = day 3
    # Cache slot must STILL be None — backfill shouldn't write it
    assert pipeline._capital_flow_signals is None, (
        "Backfill call corrupted the live cache slot — subsequent live "
        "cron calls would return stale historical data."
    )

    # Live call: latest 5 days (untouched by the backfill above)
    live = pipeline._load_capital_flow_signals(target_date=None)
    assert live["SH600519"]["net_mf"] == pytest.approx(1_000_000.0)
    # Now the cache IS populated
    assert pipeline._capital_flow_signals is live


def test_target_date_in_future_returns_all_data(pipeline_with_flow_fixture):
    """target_date later than all available data acts like target_date=None
    (returns latest 5 days). Useful for tests that pin a future date."""
    pipeline = pipeline_with_flow_fixture

    signals = pipeline._load_capital_flow_signals(target_date="2099-12-31")

    # Same result as target_date=None
    assert signals["SH600519"]["net_mf"] == pytest.approx(1_000_000.0)
    assert signals["SH600519"]["net_mf_5d"] == pytest.approx(4_000_000.0)


def test_target_date_before_all_data_returns_empty(pipeline_with_flow_fixture):
    """target_date earlier than all available data returns empty dict
    (no signals for stocks with no flow data in window)."""
    pipeline = pipeline_with_flow_fixture

    signals = pipeline._load_capital_flow_signals(target_date="2020-01-01")

    # No data → empty signals
    assert signals == {}
