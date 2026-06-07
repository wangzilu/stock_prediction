"""Tests for scripts/build_policy_factors.py --source xinwen_lianbo.

Phase E.4 (PE-4) step 3 — mirror of test_build_policy_factors_state_council.py
for the XWLB theme-attention factor builder. Covers:

  1. Per-theme THEME_<UPPER> keying — output rows are
     (datetime, "THEME_<UPPER>"), one row per (date, theme).
  2. PIT safety — factor at date D uses only events with
     publish_date <= D.
  3. 5-day rolling window — themes whose last mention is >5 days old
     drop out of the output (sparse-by-theme output).
  4. consecutive_days computation — counts consecutive day streaks
     and caps at XWLB_CONSECUTIVE_DAYS_CAP.
  5. max-of-priority aggregation across the 5-day window.
  6. Empty-input behavior — no events → zero output rows, no crash.

No LLM calls.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts import build_policy_factors as bpf


# ─────────────────────────────────────────────────────────────────────
# Fixtures — synthesise extracted XWLB events.
# ─────────────────────────────────────────────────────────────────────
def _event(
    *,
    publish_date: str,
    themes: list[str],
    theme_mention_counts: dict | None = None,
    policy_priority_signal: float = 0.5,
    regions_mentioned: list[str] | None = None,
    policy_type: str = "xinwen_lianbo_daily",
    title: str = "XWLB transcript",
    url: str | None = None,
) -> dict:
    if theme_mention_counts is None:
        theme_mention_counts = {t: 1 for t in themes}
    return {
        "publish_date": publish_date,
        "policy_type": policy_type,
        "title": f"{title} {publish_date}",
        "url": url or f"https://news.sina.com.cn/xwlb/{publish_date}/x.html",
        "themes": themes,
        "theme_mention_counts": theme_mention_counts,
        "policy_priority_signal": policy_priority_signal,
        "regions_mentioned": regions_mentioned or [],
        "extracted_at": "2026-06-05T16:25:00Z",
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
# Test 1 — Per-theme THEME_<UPPER> keying.
# ─────────────────────────────────────────────────────────────────────
def test_per_theme_keying_emits_theme_upper_instrument(tmp_path: Path):
    """A broadcast covering 2 themes on a single day yields 2 rows, each
    keyed THEME_<UPPER>."""
    events_root = tmp_path / "policy_events" / "xinwen_lianbo"
    _write_events(events_root, [
        _event(
            publish_date="2026-06-05",
            themes=["semiconductor_self_reliance", "robotics_ai"],
            theme_mention_counts={
                "semiconductor_self_reliance": 3,
                "robotics_ai": 1,
            },
            policy_priority_signal=0.8,
        ),
    ])
    df = bpf.build_xinwen_lianbo_factors_range(
        start_date="2026-06-05", end_date="2026-06-05",
        input_dir=events_root,
    )
    assert not df.empty
    instruments = set(df["instrument"].tolist())
    assert "THEME_SEMICONDUCTOR_SELF_RELIANCE" in instruments
    assert "THEME_ROBOTICS_AI" in instruments
    # All rows must be THEME_-prefixed.
    assert all(s.startswith("THEME_") for s in instruments), instruments
    # The semi row carries the 3-count.
    semi = df[df["instrument"] == "THEME_SEMICONDUCTOR_SELF_RELIANCE"].iloc[0]
    assert semi["theme_mention_count_1d"] == 3.0
    assert semi["theme_mention_count_5d"] == 3.0
    assert semi["theme_consecutive_days"] == 1.0
    assert semi["theme_priority_5d_max"] == 0.8


# ─────────────────────────────────────────────────────────────────────
# Test 2 — PIT safety: a 2026-06-10 broadcast does NOT influence the
# 2026-06-01 row.
# ─────────────────────────────────────────────────────────────────────
def test_pit_safety_no_future_leak(tmp_path: Path):
    events_root = tmp_path / "policy_events" / "xinwen_lianbo"
    _write_events(events_root, [
        _event(
            publish_date="2026-06-10",
            themes=["semiconductor_self_reliance"],
            theme_mention_counts={"semiconductor_self_reliance": 5},
        ),
    ])
    df = bpf.build_xinwen_lianbo_factors_range(
        start_date="2026-06-01", end_date="2026-06-12",
        input_dir=events_root,
    )
    # On 2026-06-01 the event is not yet visible — no row for the theme.
    dates_with_rows = df["datetime"].dt.strftime("%Y-%m-%d").unique().tolist()
    assert "2026-06-01" not in dates_with_rows, dates_with_rows
    # On 2026-06-10 the theme lights up.
    row_10 = df[df["datetime"] == "2026-06-10"]
    assert not row_10.empty
    assert row_10.iloc[0]["theme_mention_count_1d"] == 5.0


# ─────────────────────────────────────────────────────────────────────
# Test 3 — 5-day window for theme_mention_count_5d, sparse drop-out.
# ─────────────────────────────────────────────────────────────────────
def test_five_day_window_sums_and_drops_old_themes(tmp_path: Path):
    """Themes mentioned within the trailing 5d sum into count_5d; themes
    whose last mention is older than 5 days drop out of the output
    (sparse-by-theme)."""
    events_root = tmp_path / "policy_events" / "xinwen_lianbo"
    _write_events(events_root, [
        _event(
            publish_date="2026-06-01",
            themes=["real_estate"],
            theme_mention_counts={"real_estate": 2},
            policy_priority_signal=0.6,
        ),
        _event(
            publish_date="2026-06-03",
            themes=["real_estate"],
            theme_mention_counts={"real_estate": 1},
            policy_priority_signal=0.4,
        ),
        # OLD theme — its last mention is 10 days before 2026-06-13.
        _event(
            publish_date="2026-06-03",
            themes=["old_topic"],
            theme_mention_counts={"old_topic": 1},
            policy_priority_signal=0.2,
        ),
    ])
    # Signal date 2026-06-05: both real_estate mentions in window (2 + 1).
    df = bpf.build_xinwen_lianbo_factors_range(
        start_date="2026-06-05", end_date="2026-06-05",
        input_dir=events_root,
    )
    re_row = df[df["instrument"] == "THEME_REAL_ESTATE"].iloc[0]
    assert re_row["theme_mention_count_5d"] == 3.0, re_row.to_dict()
    # Max priority over the window: max(0.6, 0.4) = 0.6.
    assert re_row["theme_priority_5d_max"] == 0.6

    # Signal date 2026-06-13: every mention is older than 5 days → 0 rows.
    df_late = bpf.build_xinwen_lianbo_factors_range(
        start_date="2026-06-13", end_date="2026-06-13",
        input_dir=events_root,
    )
    assert df_late.empty, df_late


# ─────────────────────────────────────────────────────────────────────
# Test 4 — consecutive_days computation.
# ─────────────────────────────────────────────────────────────────────
def test_consecutive_days_streak_and_cap(tmp_path: Path):
    """A theme mentioned on 3 consecutive days has streak=3; a gap
    breaks the streak; the streak caps at XWLB_CONSECUTIVE_DAYS_CAP."""
    import shutil
    # Case A: 3 consecutive days. signal_date is day 3.
    events_a = tmp_path / "case_a"
    _write_events(events_a, [
        _event(publish_date="2026-06-03", themes=["belt_and_road"]),
        _event(publish_date="2026-06-04", themes=["belt_and_road"]),
        _event(publish_date="2026-06-05", themes=["belt_and_road"]),
    ])
    df_a = bpf.build_xinwen_lianbo_factors_range(
        start_date="2026-06-05", end_date="2026-06-05",
        input_dir=events_a,
    )
    assert df_a.iloc[0]["theme_consecutive_days"] == 3.0, df_a.to_dict()

    # Case B: gap on day 2 breaks streak.
    events_b = tmp_path / "case_b"
    _write_events(events_b, [
        _event(publish_date="2026-06-03", themes=["belt_and_road"]),
        # gap on 06-04
        _event(publish_date="2026-06-05", themes=["belt_and_road"]),
    ])
    df_b = bpf.build_xinwen_lianbo_factors_range(
        start_date="2026-06-05", end_date="2026-06-05",
        input_dir=events_b,
    )
    # Only today (06-05) is consecutive (06-04 has no mention).
    assert df_b.iloc[0]["theme_consecutive_days"] == 1.0, df_b.to_dict()

    # Case C: 20-day run, cap at XWLB_CONSECUTIVE_DAYS_CAP (14).
    events_c = tmp_path / "case_c"
    long_run = []
    import datetime as _dt
    base = _dt.date(2026, 5, 17)
    for i in range(20):
        d = (base + _dt.timedelta(days=i)).isoformat()
        long_run.append(_event(publish_date=d, themes=["belt_and_road"]))
    _write_events(events_c, long_run)
    df_c = bpf.build_xinwen_lianbo_factors_range(
        start_date="2026-06-05", end_date="2026-06-05",
        input_dir=events_c,
    )
    assert df_c.iloc[0]["theme_consecutive_days"] == float(
        bpf.XWLB_CONSECUTIVE_DAYS_CAP
    ), df_c.to_dict()


# ─────────────────────────────────────────────────────────────────────
# Test 5 — max-of-priority aggregation across 5d window.
# ─────────────────────────────────────────────────────────────────────
def test_priority_5d_max_aggregation(tmp_path: Path):
    """theme_priority_5d_max is the MAX of policy_priority_signal across
    the trailing 5d window (= "did the theme get a lead-story spot at
    least once in the last week?")."""
    events_root = tmp_path / "policy_events" / "xinwen_lianbo"
    _write_events(events_root, [
        _event(
            publish_date="2026-06-01",
            themes=["renewable_energy"],
            policy_priority_signal=0.2,
        ),
        _event(
            publish_date="2026-06-03",
            themes=["renewable_energy"],
            policy_priority_signal=0.9,           # the lead-story day
        ),
        _event(
            publish_date="2026-06-05",
            themes=["renewable_energy"],
            policy_priority_signal=0.3,
        ),
    ])
    df = bpf.build_xinwen_lianbo_factors_range(
        start_date="2026-06-05", end_date="2026-06-05",
        input_dir=events_root,
    )
    row = df[df["instrument"] == "THEME_RENEWABLE_ENERGY"].iloc[0]
    # max of 0.9 and 0.3 (within the trailing 5d window from 06-05;
    # 06-01 is exactly 4 days back so is in the [signal-4, signal]
    # window).
    assert row["theme_priority_5d_max"] == 0.9, row.to_dict()

    # If we shift signal_date forward past the 0.9 event, the max drops.
    df_far = bpf.build_xinwen_lianbo_factors_range(
        start_date="2026-06-09", end_date="2026-06-09",
        input_dir=events_root,
    )
    # 06-09 - 5d window = [06-05, 06-09]. Only 06-05 (0.3) in window.
    row_far = df_far[df_far["instrument"] == "THEME_RENEWABLE_ENERGY"].iloc[0]
    assert row_far["theme_priority_5d_max"] == 0.3, row_far.to_dict()


# ─────────────────────────────────────────────────────────────────────
# Test 6 — Empty input.
# ─────────────────────────────────────────────────────────────────────
def test_empty_event_dir_returns_empty_frame(tmp_path: Path):
    """No events → empty DataFrame, no crash. (PE-4 is sparse by theme
    like PE-2 — there is no "MARKET" placeholder row.)"""
    events_root = tmp_path / "policy_events" / "xinwen_lianbo"
    events_root.mkdir(parents=True, exist_ok=True)
    df = bpf.build_xinwen_lianbo_factors_range(
        start_date="2026-06-01", end_date="2026-06-05",
        input_dir=events_root,
    )
    assert df.empty, df

    # Same when the dir doesn't exist at all.
    missing_root = tmp_path / "missing"
    df2 = bpf.build_xinwen_lianbo_factors_range(
        start_date="2026-06-01", end_date="2026-06-05",
        input_dir=missing_root,
    )
    assert df2.empty


# ─────────────────────────────────────────────────────────────────────
# Bonus — Empty themes-list rows are dropped at load time and never
# emit a per-(date, theme) factor row.
# ─────────────────────────────────────────────────────────────────────
def test_empty_themes_list_is_dropped_at_load_time(tmp_path: Path):
    """An XWLB broadcast with no themes is a 'filler' day — it must
    not generate any factor row."""
    events_root = tmp_path / "policy_events" / "xinwen_lianbo"
    _write_events(events_root, [
        _event(publish_date="2026-06-05", themes=[]),
    ])
    df = bpf.build_xinwen_lianbo_factors_range(
        start_date="2026-06-05", end_date="2026-06-05",
        input_dir=events_root,
    )
    assert df.empty, df
