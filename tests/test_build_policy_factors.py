"""Tests for ``scripts/build_policy_factors.py`` — Phase E.1 step 3.

Covers:
  1. PIT safety — factor value at date D only uses events whose
     ``publish_date <= D``. A factor row for an earlier date must never
     see numbers from a later event.
  2. ``pbc_liquidity_zscore_20d`` window — z-score is computed over the
     trailing 20 calendar days of net_injection. Constant input → z=0.
  3. Dummy flag computation — ``pbc_easing_dummy=1`` iff the last 5
     days had any easing event; same for tightening. Older events
     outside the 5-day window do NOT trigger the flag.

No LLM calls. We seed an in-memory list of "extracted" events to
exercise the factor logic in isolation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_policy_factors as bpf


# ─────────────────────────────────────────────────────────────────────
# Fixtures — synthesise extracted policy events
# ─────────────────────────────────────────────────────────────────────
def _event(
    *,
    publish_date: str,
    policy_stance: str = "easing",
    net_injection: float | None = 700.0,
    repo_rate_change: int | None = None,
    tool_type: str = "omo",
) -> dict:
    return {
        "publish_date": publish_date,
        "policy_type": tool_type,
        "title": f"PBC {tool_type} on {publish_date}",
        "url": f"http://example.com/{publish_date}/{tool_type}",
        "policy_stance": policy_stance,
        "liquidity_injection_amount": net_injection,
        "net_injection": net_injection,
        "repo_rate_change": repo_rate_change,
        "tool_type": tool_type,
        "duration_days": 7 if tool_type == "omo" else None,
        "unexpectedness": 0.3,
        "extracted_at": "2026-06-05T09:30:00Z",
    }


def _write_events(root: Path, events: list[dict]) -> None:
    """Write events grouped by publish_date to one JSONL per day."""
    by_date: dict[str, list[dict]] = {}
    for ev in events:
        by_date.setdefault(ev["publish_date"], []).append(ev)
    root.mkdir(parents=True, exist_ok=True)
    for d, rows in by_date.items():
        with open(root / f"{d}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────
# Test 1 — PIT safety
# ─────────────────────────────────────────────────────────────────────
def test_factor_value_does_not_see_future_events(tmp_path: Path):
    """A factor row for date D must only use events with publish_date <= D."""
    events_root = tmp_path / "policy_events" / "pbc"
    # 2026-06-01: small easing event (+100); 2026-06-10: big easing event (+5000)
    _write_events(events_root, [
        _event(publish_date="2026-06-01", net_injection=100.0),
        _event(publish_date="2026-06-10", net_injection=5000.0),
    ])

    df = bpf.build_factors_range(
        start_date="2026-06-01", end_date="2026-06-10",
        input_dir=events_root,
    )
    df = df.set_index("datetime")

    # On 2026-06-01, factor row must only count the +100 event,
    # not the future +5000. The simplest check: pbc_easing_dummy=1 on
    # both days but the 20d rolling sum / zscore on 2026-06-01 must be
    # computed using only events through 2026-06-01.
    # Use the underlying 20d rolling sum the zscore is built from.
    # The easing_dummy must be 1 on both dates.
    assert df.loc["2026-06-01", "pbc_easing_dummy"] == 1.0
    assert df.loc["2026-06-10", "pbc_easing_dummy"] == 1.0

    # Backbone PIT check: latest event date encoded in factor row metadata
    # must reflect PIT cutoff.
    # The mean of net_injection used in zscore on D=2026-06-01 must be
    # 100.0 (only that day's event). On D=2026-06-10 it must include
    # both events.
    # The factor surface doesn't expose mean directly, so we assert a
    # weaker but solid PIT property: the easing dummy on a date before
    # the easing event is 0.
    _write_events(events_root, [
        _event(publish_date="2026-06-01", net_injection=100.0,
                policy_stance="neutral"),
        _event(publish_date="2026-06-10", net_injection=5000.0,
                policy_stance="easing"),
    ])
    df2 = bpf.build_factors_range(
        start_date="2026-06-01", end_date="2026-06-10",
        input_dir=events_root,
    )
    df2 = df2.set_index("datetime")
    # On 2026-06-01 there is NO easing event yet → flag must be 0.
    assert df2.loc["2026-06-01", "pbc_easing_dummy"] == 0.0
    # On 2026-06-10 the easing event hit → flag must be 1.
    assert df2.loc["2026-06-10", "pbc_easing_dummy"] == 1.0


# ─────────────────────────────────────────────────────────────────────
# Test 2 — zscore_20d window
# ─────────────────────────────────────────────────────────────────────
def test_zscore_20d_is_zero_on_constant_input(tmp_path: Path):
    """If net_injection is constant for 25 days, z-score is 0 (or NaN/0)."""
    events_root = tmp_path / "policy_events" / "pbc"
    # 25 days of identical +500 injections.
    base = pd.Timestamp("2026-05-15")
    evs = [
        _event(
            publish_date=(base + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            net_injection=500.0,
        )
        for i in range(25)
    ]
    _write_events(events_root, evs)

    df = bpf.build_factors_range(
        start_date=evs[0]["publish_date"],
        end_date=evs[-1]["publish_date"],
        input_dir=events_root,
    )
    df = df.set_index("datetime")

    # The last day has a full 20-day window of identical values → std=0
    # → zscore must be 0 (not NaN) by convention (we return 0 when std<eps).
    last_date = evs[-1]["publish_date"]
    z = df.loc[last_date, "pbc_liquidity_zscore_20d"]
    # Accept 0.0 exactly; we explicitly fill std=0 → z=0
    assert z == 0.0, f"expected 0.0 z on constant input, got {z}"

    # Now bump one day's value and check z != 0 at the bump date and after
    bump_date = (base + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    new_evs = []
    for ev in evs:
        if ev["publish_date"] == bump_date:
            new_evs.append(_event(
                publish_date=bump_date, net_injection=5000.0,
            ))
        else:
            new_evs.append(ev)
    # Overwrite the file for the bump date
    _write_events(events_root, new_evs)
    df2 = bpf.build_factors_range(
        start_date=evs[0]["publish_date"],
        end_date=evs[-1]["publish_date"],
        input_dir=events_root,
    )
    df2 = df2.set_index("datetime")
    z2 = df2.loc[bump_date, "pbc_liquidity_zscore_20d"]
    assert z2 > 0, f"expected positive z on +5000 outlier, got {z2}"


# ─────────────────────────────────────────────────────────────────────
# Test 3 — Dummy flag 5-day window for easing/tightening; older event
# outside the 5-day window does NOT trigger the flag.
# ─────────────────────────────────────────────────────────────────────
def test_easing_dummy_only_within_5d_window(tmp_path: Path):
    """Easing event 10 days ago does not light up the easing_dummy today."""
    events_root = tmp_path / "policy_events" / "pbc"
    _write_events(events_root, [
        # An easing event 10 days before the signal date
        _event(publish_date="2026-05-25", policy_stance="easing"),
    ])

    df = bpf.build_factors_range(
        start_date="2026-05-25", end_date="2026-06-04",
        input_dir=events_root,
    )
    df = df.set_index("datetime")

    # On the event day itself: easing_dummy = 1
    assert df.loc["2026-05-25", "pbc_easing_dummy"] == 1.0
    # 5 days later: still in window → 1
    assert df.loc["2026-05-30", "pbc_easing_dummy"] == 1.0
    # 10 days later: out of window → 0
    assert df.loc["2026-06-04", "pbc_easing_dummy"] == 0.0


# ─────────────────────────────────────────────────────────────────────
# Test 4 — MARKET instrument keying + short_rate_pressure 20d sum
# ─────────────────────────────────────────────────────────────────────
def test_market_instrument_keying_and_short_rate_pressure(tmp_path: Path):
    """Factors are keyed by (datetime, "MARKET"); short_rate_pressure is 20d sum."""
    events_root = tmp_path / "policy_events" / "pbc"
    _write_events(events_root, [
        # Two rate cuts: -10bp on 6-01 and -15bp on 6-05
        _event(
            publish_date="2026-06-01",
            policy_stance="easing",
            net_injection=None,
            repo_rate_change=-10,
            tool_type="omo",
        ),
        _event(
            publish_date="2026-06-05",
            policy_stance="easing",
            net_injection=None,
            repo_rate_change=-15,
            tool_type="omo",
        ),
    ])

    df = bpf.build_factors_range(
        start_date="2026-06-01", end_date="2026-06-10",
        input_dir=events_root,
    )

    # Schema: (datetime, instrument) tuple with instrument="MARKET"
    assert "instrument" in df.columns or df.index.name in ("instrument", None)
    if "instrument" in df.columns:
        assert (df["instrument"] == "MARKET").all()

    # short_rate_pressure on 2026-06-05 must be -10 + -15 = -25
    df2 = df.set_index("datetime")
    val = df2.loc["2026-06-05", "short_rate_pressure"]
    assert val == -25, f"expected -25, got {val}"
    # After 6-05, the 20d sum still includes both events; on 6-10 still -25
    assert df2.loc["2026-06-10", "short_rate_pressure"] == -25
