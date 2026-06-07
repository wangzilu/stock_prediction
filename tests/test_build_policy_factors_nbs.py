"""Tests for scripts/build_policy_factors.py --source nbs.

Phase E.3 (PE-3) step 3 — mirror of test_build_policy_factors_state_council.py
for the NBS macro-surprise factor builder. Covers:

  1. Per-date MARKET keying — output rows are (datetime, "MARKET"),
     same as PE-1 PBC (macro releases are market-wide).
  2. PIT safety — factor value at date D only uses events whose
     ``publish_date <= D``.
  3. 3-month rolling windows — events older than 90 days drop out of
     the surprise sums and the retail mean.
  4. PMI dummy threshold at 50 — the latest PMI > 50 lights the dummy,
     <50 turns it off, missing PMI keeps it at 0.
  5. Multi-release aggregation — multiple CPI releases in the window
     sum their (consensus - headline) gaps.
  6. Empty-input behavior — no events → zero-valued rows per date,
     no crash.

No LLM calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_policy_factors as bpf


# ─────────────────────────────────────────────────────────────────────
# Fixtures — synthesise extracted NBS macro events
# ─────────────────────────────────────────────────────────────────────
def _event(
    *,
    publish_date: str,
    series_name: str = "cpi",
    release_period: str | None = None,
    headline_value: float | None = 0.3,
    prior_value: float | None = 0.4,
    consensus_value: float | None = 0.5,
    mom_change: float | None = 0.1,
    yoy_change: float | None = 0.3,
    surprise_direction: str = "downside",
    policy_type: str = "cpi_monthly",
    title: str = "NBS macro release",
) -> dict:
    return {
        "publish_date": publish_date,
        "policy_type": policy_type,
        "title": f"{title} {publish_date}",
        "url": f"http://www.stats.gov.cn/{publish_date}/{series_name}",
        "series_name": series_name,
        "release_period": release_period,
        "headline_value": headline_value,
        "prior_value": prior_value,
        "consensus_value": consensus_value,
        "mom_change": mom_change,
        "yoy_change": yoy_change,
        "surprise_direction": surprise_direction,
        "extracted_at": "2026-06-03T16:00:00Z",
    }


def _write_events(root: Path, events: list[dict]) -> None:
    by_date: dict[str, list[dict]] = {}
    for ev in events:
        by_date.setdefault(ev["publish_date"], []).append(ev)
    root.mkdir(parents=True, exist_ok=True)
    for d, rows in by_date.items():
        with open(root / f"{d}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────
# Test 1 — Per-date MARKET keying. One CPI release on a single day
# yields one MARKET-keyed row with the surprise filled.
# ─────────────────────────────────────────────────────────────────────
def test_market_instrument_keying_per_date(tmp_path: Path):
    """NBS factor rows are keyed (datetime, MARKET); a CPI release fills
    nbs_cpi_surprise_3m as (consensus - headline)."""
    events_root = tmp_path / "policy_events" / "nbs"
    _write_events(events_root, [
        _event(
            publish_date="2026-06-03",
            series_name="cpi",
            headline_value=0.3,
            consensus_value=0.5,
        ),
    ])
    df = bpf.build_nbs_factors_range(
        start_date="2026-06-03", end_date="2026-06-03",
        input_dir=events_root,
    )
    assert not df.empty
    assert (df["instrument"] == "MARKET").all(), df["instrument"].unique()
    row = df.iloc[0]
    # consensus (0.5) - headline (0.3) = 0.2 (inflation undershoots → positive)
    assert abs(row["nbs_cpi_surprise_3m"] - 0.2) < 1e-9
    # No PPI / PMI / retail event → those factors are zero.
    assert row["nbs_ppi_surprise_3m"] == 0.0
    assert row["nbs_pmi_above_50_dummy"] == 0.0
    assert row["nbs_retail_growth_yoy_3m"] == 0.0


# ─────────────────────────────────────────────────────────────────────
# Test 2 — PIT safety: a CPI release on 2026-06-10 must NOT contribute
# to the 2026-06-01 row.
# ─────────────────────────────────────────────────────────────────────
def test_pit_safety_no_future_leak(tmp_path: Path):
    events_root = tmp_path / "policy_events" / "nbs"
    _write_events(events_root, [
        _event(
            publish_date="2026-06-10",
            series_name="cpi",
            headline_value=0.3,
            consensus_value=2.5,  # would make surprise = +2.2
        ),
    ])
    df = bpf.build_nbs_factors_range(
        start_date="2026-06-01", end_date="2026-06-12",
        input_dir=events_root,
    )
    df = df.set_index("datetime")
    # On 2026-06-01 the event is not yet visible — factor must be 0.
    assert df.loc["2026-06-01", "nbs_cpi_surprise_3m"] == 0.0
    # On 2026-06-10 the event hits — factor = 2.5 - 0.3 = 2.2.
    assert abs(df.loc["2026-06-10", "nbs_cpi_surprise_3m"] - 2.2) < 1e-9


# ─────────────────────────────────────────────────────────────────────
# Test 3 — 3-month rolling window: an event older than 90 days drops
# out of the surprise sum and the retail mean.
# ─────────────────────────────────────────────────────────────────────
def test_three_month_rolling_window_drops_old_events(tmp_path: Path):
    events_root = tmp_path / "policy_events" / "nbs"
    _write_events(events_root, [
        # Day 1: CPI release. (consensus 0.5 - headline 0.3) = +0.2.
        _event(
            publish_date="2026-03-01",
            series_name="cpi",
            headline_value=0.3, consensus_value=0.5,
        ),
        # Day 90 from above: still in window (boundary check).
        _event(
            publish_date="2026-05-29",
            series_name="cpi",
            headline_value=0.4, consensus_value=0.6,
        ),
    ])
    df = bpf.build_nbs_factors_range(
        start_date="2026-05-29", end_date="2026-07-01",
        input_dir=events_root,
    ).set_index("datetime")

    # On 2026-05-29 (the day the second event lands): both events visible
    # if within 90 days. 2026-03-01 → 2026-05-29 is exactly 89 days,
    # so still in window. Sum = 0.2 + 0.2 = 0.4.
    val_first_day = df.loc["2026-05-29", "nbs_cpi_surprise_3m"]
    assert abs(val_first_day - 0.4) < 1e-9, (
        f"on the second-event day both should be in window. got {val_first_day}"
    )

    # On 2026-07-01 (older than 90 days after 2026-03-01 but within 90
    # days of 2026-05-29): only the second event contributes.
    val_later = df.loc["2026-07-01", "nbs_cpi_surprise_3m"]
    assert abs(val_later - 0.2) < 1e-9, (
        f"only the second event must remain in window. got {val_later}"
    )


# ─────────────────────────────────────────────────────────────────────
# Test 4 — PMI dummy threshold: latest PMI > 50 → dummy=1; <=50 → 0.
# ─────────────────────────────────────────────────────────────────────
def test_pmi_above_50_dummy(tmp_path: Path):
    events_root = tmp_path / "policy_events" / "nbs"
    _write_events(events_root, [
        # Latest PMI = 50.4 → dummy=1
        _event(
            publish_date="2026-06-01",
            series_name="pmi",
            headline_value=50.4,
            consensus_value=50.0,
            policy_type="pmi_monthly",
        ),
    ])
    df = bpf.build_nbs_factors_range(
        start_date="2026-06-01", end_date="2026-06-01",
        input_dir=events_root,
    )
    assert df.iloc[0]["nbs_pmi_above_50_dummy"] == 1.0

    # Override with a PMI = 49.5 release later.
    _write_events(events_root, [
        _event(
            publish_date="2026-06-01",
            series_name="pmi",
            headline_value=50.4,
            policy_type="pmi_monthly",
        ),
        _event(
            publish_date="2026-07-01",
            series_name="pmi",
            headline_value=49.5,
            policy_type="pmi_monthly",
        ),
    ])
    df2 = bpf.build_nbs_factors_range(
        start_date="2026-07-01", end_date="2026-07-01",
        input_dir=events_root,
    )
    # Latest = 49.5 → dummy=0
    assert df2.iloc[0]["nbs_pmi_above_50_dummy"] == 0.0

    # Exactly 50.0 is NOT above 50 — dummy=0 (strict >).
    _write_events(events_root, [
        _event(
            publish_date="2026-06-01",
            series_name="pmi",
            headline_value=50.0,
            policy_type="pmi_monthly",
        ),
    ])
    df3 = bpf.build_nbs_factors_range(
        start_date="2026-06-01", end_date="2026-06-01",
        input_dir=events_root,
    )
    assert df3.iloc[0]["nbs_pmi_above_50_dummy"] == 0.0


# ─────────────────────────────────────────────────────────────────────
# Test 5 — Multi-release aggregation: 3 CPI releases in the 3-month
# window stack their surprise gaps; retail yoy averages across the window.
# ─────────────────────────────────────────────────────────────────────
def test_multi_release_aggregation(tmp_path: Path):
    events_root = tmp_path / "policy_events" / "nbs"
    _write_events(events_root, [
        # CPI: 3 releases, each with (consensus - headline) = +0.1
        _event(
            publish_date="2026-04-10",
            series_name="cpi",
            headline_value=0.2, consensus_value=0.3,
        ),
        _event(
            publish_date="2026-05-10",
            series_name="cpi",
            headline_value=0.3, consensus_value=0.4,
        ),
        _event(
            publish_date="2026-06-10",
            series_name="cpi",
            headline_value=0.4, consensus_value=0.5,
        ),
        # Retail: 2 releases with yoy 3.0% and 3.4% → mean 3.2%.
        _event(
            publish_date="2026-05-15",
            series_name="retail_sales",
            yoy_change=3.0,
            headline_value=None, consensus_value=None,
            policy_type="retail_sales_monthly",
        ),
        _event(
            publish_date="2026-06-15",
            series_name="retail_sales",
            yoy_change=3.4,
            headline_value=None, consensus_value=None,
            policy_type="retail_sales_monthly",
        ),
    ])
    df = bpf.build_nbs_factors_range(
        start_date="2026-06-15", end_date="2026-06-15",
        input_dir=events_root,
    )
    row = df.iloc[0]
    # CPI surprise = +0.1 * 3 = +0.3 (sum of consensus - headline over window)
    assert abs(row["nbs_cpi_surprise_3m"] - 0.3) < 1e-9, row.to_dict()
    # Retail yoy = (3.0 + 3.4) / 2 = 3.2
    assert abs(row["nbs_retail_growth_yoy_3m"] - 3.2) < 1e-9, row.to_dict()


# ─────────────────────────────────────────────────────────────────────
# Test 6 — Empty input: zero rows for every date, no crash.
# ─────────────────────────────────────────────────────────────────────
def test_empty_event_dir_returns_zero_valued_rows(tmp_path: Path):
    events_root = tmp_path / "policy_events" / "nbs"
    events_root.mkdir(parents=True, exist_ok=True)
    df = bpf.build_nbs_factors_range(
        start_date="2026-06-01", end_date="2026-06-05",
        input_dir=events_root,
    )
    # Empty input → one row per date with zero factors (unlike PE-2,
    # PE-3 is MARKET-keyed dense so a future broadcast doesn't see holes).
    assert len(df) == 5
    assert (df["instrument"] == "MARKET").all()
    for col in (
        "nbs_cpi_surprise_3m", "nbs_ppi_surprise_3m",
        "nbs_pmi_above_50_dummy", "nbs_retail_growth_yoy_3m",
    ):
        assert (df[col] == 0.0).all(), col


# ─────────────────────────────────────────────────────────────────────
# Bonus — PPI surprise direction sanity. Restrictive PPI undershoot
# (headline below consensus) → positive surprise.
# ─────────────────────────────────────────────────────────────────────
def test_ppi_surprise_independent_of_cpi(tmp_path: Path):
    events_root = tmp_path / "policy_events" / "nbs"
    _write_events(events_root, [
        _event(
            publish_date="2026-06-04",
            series_name="ppi",
            headline_value=-0.6,
            consensus_value=-0.4,
            policy_type="ppi_monthly",
        ),
    ])
    df = bpf.build_nbs_factors_range(
        start_date="2026-06-04", end_date="2026-06-04",
        input_dir=events_root,
    )
    row = df.iloc[0]
    # consensus -0.4 - headline -0.6 = +0.2 (PPI undershoots → positive)
    assert abs(row["nbs_ppi_surprise_3m"] - 0.2) < 1e-9
    # No CPI event → cpi_surprise stays 0.
    assert row["nbs_cpi_surprise_3m"] == 0.0
