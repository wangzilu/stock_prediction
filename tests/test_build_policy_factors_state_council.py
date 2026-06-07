"""Tests for scripts/build_policy_factors.py --source state_council.

Phase E.2 (PE-2) step 3 — mirror of test_build_policy_factors.py for
the per-industry industry-policy factor builder. Covers:

  1. PIT safety — factor value at date D only uses events whose
     ``publish_date <= D``.
  2. Per-industry keying — output rows are (datetime, INDUSTRY_<NAME>),
     NOT (datetime, "MARKET"). A doc tagging two industries produces
     two rows on the publish date.
  3. 5d / 20d signed-strength sums and novelty decay window.
  4. Health gate — success requires BOTH n_rows>0 AND n_events>0
     (mirrors PE-1 PBC discipline).

No LLM calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_policy_factors as bpf


# ─────────────────────────────────────────────────────────────────────
# Fixtures — synthesise extracted SC policy events
# ─────────────────────────────────────────────────────────────────────
def _event(
    *,
    publish_date: str,
    target_industries: list[str] | None = None,
    policy_direction: str = "supportive",
    policy_strength: float = 0.8,
    fiscal_support: float | None = 500.0,
    subsidy_or_tax: str = "subsidy",
    regulatory_tightening: bool = False,
    implementation_deadline: str | None = None,
    title: str = "国务院政策",
    policy_type: str = "state_council_doc",
) -> dict:
    return {
        "publish_date": publish_date,
        "policy_type": policy_type,
        "title": f"{title} on {publish_date}",
        "url": f"http://www.gov.cn/{publish_date}/{policy_type}",
        "target_industries": target_industries or ["semiconductor"],
        "policy_direction": policy_direction,
        "policy_strength": policy_strength,
        "fiscal_support": fiscal_support,
        "subsidy_or_tax": subsidy_or_tax,
        "regulatory_tightening": regulatory_tightening,
        "implementation_deadline": implementation_deadline,
        "extracted_at": "2026-06-05T15:30:00Z",
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
# Test 1 — Per-industry keying. A doc tagging 2 industries lands on
# both on the publish date.
# ─────────────────────────────────────────────────────────────────────
def test_industry_instrument_keying(tmp_path: Path):
    """Two target industries → two factor rows on the publish date."""
    events_root = tmp_path / "policy_events" / "state_council"
    _write_events(events_root, [
        _event(
            publish_date="2026-06-05",
            target_industries=["semiconductor", "renewable_energy"],
            policy_direction="supportive",
            policy_strength=0.7,
        ),
    ])

    df = bpf.build_sc_factors_range(
        start_date="2026-06-05", end_date="2026-06-05",
        input_dir=events_root,
    )
    assert not df.empty, "factor build must emit per-industry rows"
    assert set(df["instrument"].unique()) == {
        "INDUSTRY_SEMICONDUCTOR", "INDUSTRY_RENEWABLE_ENERGY",
    }, df["instrument"].unique()
    # Both rows must have positive 5d support since policy_direction was
    # supportive and strength was 0.7.
    for inst in ("INDUSTRY_SEMICONDUCTOR", "INDUSTRY_RENEWABLE_ENERGY"):
        row = df[df["instrument"] == inst].iloc[0]
        assert row["industry_policy_support_5d"] > 0, (inst, row)
        assert row["industry_policy_support_20d"] > 0
        # Novelty: just landed today → 1.0
        assert row["industry_policy_novelty"] == 1.0


# ─────────────────────────────────────────────────────────────────────
# Test 2 — PIT safety. An industry's first mention on 2026-06-10
# must NOT contribute to its 2026-06-01 factor row.
# ─────────────────────────────────────────────────────────────────────
def test_pit_safety_no_future_leak(tmp_path: Path):
    events_root = tmp_path / "policy_events" / "state_council"
    _write_events(events_root, [
        _event(
            publish_date="2026-06-10",
            target_industries=["semiconductor"],
            policy_direction="supportive",
            policy_strength=0.9,
        ),
    ])

    df = bpf.build_sc_factors_range(
        start_date="2026-06-01", end_date="2026-06-12",
        input_dir=events_root,
    )
    # On 2026-06-01: no events visible → 0 rows for that date.
    early = df[df["datetime"] == pd.Timestamp("2026-06-01")]
    assert early.empty, (
        "PE-2 must NOT emit a row for a date that has no visible events. "
        "Saw rows: %s" % early.to_dict("records")
    )
    # On 2026-06-10: semiconductor row appears.
    on_day = df[df["datetime"] == pd.Timestamp("2026-06-10")]
    assert not on_day.empty
    assert (on_day["instrument"] == "INDUSTRY_SEMICONDUCTOR").any()


# ─────────────────────────────────────────────────────────────────────
# Test 3 — 5d vs 20d window. A supportive event on 2026-06-01
# contributes to the 5d sum on 2026-06-03 but not on 2026-06-10
# (out of 5-day window, still in 20-day window).
# ─────────────────────────────────────────────────────────────────────
def test_short_and_long_window_sums(tmp_path: Path):
    events_root = tmp_path / "policy_events" / "state_council"
    _write_events(events_root, [
        _event(
            publish_date="2026-06-01",
            target_industries=["semiconductor"],
            policy_direction="supportive",
            policy_strength=1.0,
        ),
    ])
    df = bpf.build_sc_factors_range(
        start_date="2026-06-01", end_date="2026-06-25",
        input_dir=events_root,
    )

    df_by_date = df[df["instrument"] == "INDUSTRY_SEMICONDUCTOR"].set_index(
        "datetime"
    )

    # 2026-06-03: still in both 5d and 20d windows
    row_03 = df_by_date.loc[pd.Timestamp("2026-06-03")]
    assert row_03["industry_policy_support_5d"] == 1.0
    assert row_03["industry_policy_support_20d"] == 1.0

    # 2026-06-10: out of 5d (event was 9 days ago), still in 20d
    row_10 = df_by_date.loc[pd.Timestamp("2026-06-10")]
    assert row_10["industry_policy_support_5d"] == 0.0
    assert row_10["industry_policy_support_20d"] == 1.0

    # 2026-06-22: out of both 5d and 20d (21 days later); but still
    # in 60d novelty window so the row exists with zero support sums.
    row_22 = df_by_date.loc[pd.Timestamp("2026-06-22")]
    assert row_22["industry_policy_support_5d"] == 0.0
    assert row_22["industry_policy_support_20d"] == 0.0
    # Novelty decayed but >0 since still within 60d.
    assert 0.0 < row_22["industry_policy_novelty"] < 1.0


# ─────────────────────────────────────────────────────────────────────
# Test 4 — Restrictive policy contributes negatively to the sum.
# ─────────────────────────────────────────────────────────────────────
def test_restrictive_policy_negative_contribution(tmp_path: Path):
    events_root = tmp_path / "policy_events" / "state_council"
    _write_events(events_root, [
        _event(
            publish_date="2026-06-05",
            target_industries=["real_estate"],
            policy_direction="restrictive",
            policy_strength=0.6,
        ),
    ])
    df = bpf.build_sc_factors_range(
        start_date="2026-06-05", end_date="2026-06-05",
        input_dir=events_root,
    )
    assert not df.empty
    row = df[df["instrument"] == "INDUSTRY_REAL_ESTATE"].iloc[0]
    # Restrictive: signed strength = -0.6
    assert row["industry_policy_support_5d"] == -0.6
    assert row["industry_policy_support_20d"] == -0.6


# ─────────────────────────────────────────────────────────────────────
# Test 5 — Empty input dir → empty output. Health publishes the
# no_events branch and exits cleanly (no crash).
# ─────────────────────────────────────────────────────────────────────
def test_empty_event_dir_returns_empty_factor_frame(tmp_path: Path):
    events_root = tmp_path / "policy_events" / "state_council"
    events_root.mkdir(parents=True, exist_ok=True)
    df = bpf.build_sc_factors_range(
        start_date="2026-06-01", end_date="2026-06-05",
        input_dir=events_root,
    )
    assert df.empty


# ─────────────────────────────────────────────────────────────────────
# Test 6 — Two events on different days for the same industry: the
# 5d sum aggregates BOTH signed strengths.
# ─────────────────────────────────────────────────────────────────────
def test_two_events_same_industry_aggregate(tmp_path: Path):
    events_root = tmp_path / "policy_events" / "state_council"
    _write_events(events_root, [
        _event(
            publish_date="2026-06-01",
            target_industries=["semiconductor"],
            policy_direction="supportive",
            policy_strength=0.5,
        ),
        _event(
            publish_date="2026-06-03",
            target_industries=["semiconductor"],
            policy_direction="supportive",
            policy_strength=0.4,
        ),
    ])
    df = bpf.build_sc_factors_range(
        start_date="2026-06-03", end_date="2026-06-03",
        input_dir=events_root,
    )
    row = df[df["instrument"] == "INDUSTRY_SEMICONDUCTOR"].iloc[0]
    # 5d window: 0.5 + 0.4 = 0.9
    assert abs(row["industry_policy_support_5d"] - 0.9) < 1e-9
    assert abs(row["industry_policy_support_20d"] - 0.9) < 1e-9
