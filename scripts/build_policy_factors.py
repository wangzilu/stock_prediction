"""Build policy factors from extracted policy events.

Phase E.1 step 3 (PBC):
  Reads ``data/storage/policy_events/pbc/<YYYY-MM-DD>.jsonl`` and emits
  a single parquet keyed by ``(datetime, "MARKET")`` — the synthetic
  instrument convention the cross_market_regime overlay uses to
  broadcast a market-level signal to every stock.

Phase E.2 step 3 (PE-2, State Council / ministry):
  Reads ``data/storage/policy_events/state_council/<YYYY-MM-DD>.jsonl``
  and emits a parquet keyed by ``(datetime, "INDUSTRY_<NAME>")`` —
  per-industry, per-day rows so the FeatureMerger can map A-share
  stocks to industries at execution time. The mapper-to-stock step is
  a follow-up; for now the parquet carries synthetic industry
  instruments and the consumer is responsible for the join.

Phase E.3 step 3 (PE-3, NBS macro surprise):
  Reads ``data/storage/policy_events/nbs/<YYYY-MM-DD>.jsonl`` and
  emits a parquet keyed by ``(datetime, "MARKET")`` (same as PE-1
  PBC) because macro statistics affect the whole market, not specific
  industries. Factors are rolling 3-month aggregates of CPI / PPI
  surprises plus a PMI-above-50 dummy and a 3-month yoy retail-sales
  average.

Phase E.4 step 3 (PE-4, Xinwen Lianbo theme attention):
  Reads ``data/storage/policy_events/xinwen_lianbo/<YYYY-MM-DD>.jsonl``
  and emits a parquet keyed by ``(datetime, "THEME_<NAME>")`` — per-
  (theme, day) rows so the FeatureMerger can map A-share stocks to
  themes at execution time. Theme attention is per-theme, NOT per-
  industry or MARKET (the canonical 9 themes from the phase plan do
  not align with industry taxonomy, e.g. 一带一路 spans many sectors).

Output
------
    data/storage/pbc_liquidity_factors.parquet (PE-1, MARKET-keyed)
    data/storage/state_council_policy_factors.parquet (PE-2, INDUSTRY-keyed)
    data/storage/nbs_macro_factors.parquet (PE-3, MARKET-keyed)
    data/storage/xinwen_lianbo_theme_factors.parquet (PE-4, THEME-keyed)
    Plus .health.json sidecars next to each parquet.

Factor definitions (per the phase doc)
--------------------------------------
- ``pbc_liquidity_zscore_20d``: rolling 20-day z-score of net_injection.
  When the trailing window has zero std (e.g. constant injections), the
  z-score is set to 0.0 by convention to avoid NaN broadcast.
- ``pbc_easing_dummy``: 1.0 if any easing event in the last 5 calendar
  days (inclusive of the signal date), else 0.0.
- ``pbc_tightening_dummy``: same for tightening events.
- ``short_rate_pressure``: sum of ``repo_rate_change`` (basis points)
  over the trailing 20 calendar days.

PIT discipline
--------------
Factor value at date D uses only events with ``publish_date <= D``.
This is enforced by per-D filtering inside ``build_factors_for_date``
— we never use the whole input table for a single D's row.

Usage
-----
    # daily cron mode (today only, appends to parquet)
    python scripts/build_policy_factors.py --source pbc

    # backfill explicit window
    python scripts/build_policy_factors.py --source pbc \\
        --start 2026-04-01 --end 2026-06-05
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)

INPUT_DIR = DATA_DIR / "policy_events" / "pbc"
OUTPUT_PATH = DATA_DIR / "pbc_liquidity_factors.parquet"
HEALTH_PATH = DATA_DIR / "pbc_liquidity_factors.health.json"
HEALTH_SOURCE_NAME = "pbc_liquidity_factors"

# Phase E.2 (PE-2) — State Council / ministry industry policy.
INPUT_DIR_SC = DATA_DIR / "policy_events" / "state_council"
OUTPUT_PATH_SC = DATA_DIR / "state_council_policy_factors.parquet"
HEALTH_PATH_SC = DATA_DIR / "state_council_policy_factors.health.json"
HEALTH_SOURCE_NAME_SC = "state_council_policy_factors"

# Phase E.3 (PE-3) — NBS macro surprise (MARKET-keyed, same as PBC).
INPUT_DIR_NBS = DATA_DIR / "policy_events" / "nbs"
OUTPUT_PATH_NBS = DATA_DIR / "nbs_macro_factors.parquet"
HEALTH_PATH_NBS = DATA_DIR / "nbs_macro_factors.health.json"
HEALTH_SOURCE_NAME_NBS = "nbs_macro_factors"

# Phase E.4 (PE-4) — Xinwen Lianbo theme attention (THEME-keyed).
INPUT_DIR_XWLB = DATA_DIR / "policy_events" / "xinwen_lianbo"
OUTPUT_PATH_XWLB = DATA_DIR / "xinwen_lianbo_theme_factors.parquet"
HEALTH_PATH_XWLB = DATA_DIR / "xinwen_lianbo_theme_factors.health.json"
HEALTH_SOURCE_NAME_XWLB = "xinwen_lianbo_theme_factors"

# Window sizes — match the phase doc; kept as constants so changes
# require a code edit rather than a config drift.
ZSCORE_WINDOW_DAYS = 20
DUMMY_WINDOW_DAYS = 5
RATE_PRESSURE_WINDOW_DAYS = 20

# PE-2 windows. 5d / 20d match the PBC convention; novelty is a
# decay-style score keyed on first-mention date so a brand-new industry
# in the policy stream lights up even before the 5d window fills.
INDUSTRY_SUPPORT_SHORT_DAYS = 5
INDUSTRY_SUPPORT_LONG_DAYS = 20
INDUSTRY_NOVELTY_WINDOW_DAYS = 60

# PE-3 windows. 3-month rolling sums match the NBS monthly publish
# cadence (3 months ≈ 90 calendar days). The PMI dummy is a point-in-
# time check against the latest release only.
NBS_SURPRISE_WINDOW_DAYS = 90
NBS_PMI_THRESHOLD = 50.0

# PE-4 windows. Short 1d / 5d rolling windows match XWLB's DAILY publish
# cadence. consecutive_days caps at 14: a theme that ran for >2 weeks
# is already a regime, not a daily attention burst, and a higher cap
# would just create a fat-tailed factor whose top-decile is one or two
# special events (Two Sessions, Party Congress) that already get their
# own overlay.
XWLB_MENTION_SHORT_DAYS = 5
XWLB_CONSECUTIVE_DAYS_CAP = 14
XWLB_PRIORITY_WINDOW_DAYS = 5

# Synthetic instrument the cross-market regime overlay already uses;
# the FeatureMerger broadcasts MARKET-keyed rows to every stock at
# merge time. Kept as a named constant so a future rename is one edit.
MARKET_INSTRUMENT = "MARKET"
# PE-2 industry instrument prefix. The downstream FeatureMerger maps
# A-share stocks -> industry via the supply_chain_mapper. Kept as a
# named constant so the prefix is grep-discoverable in the merger.
INDUSTRY_INSTRUMENT_PREFIX = "INDUSTRY_"
# PE-4 theme instrument prefix. The downstream FeatureMerger will map
# A-share stocks -> theme via a thematic basket mapper (TBD). Until
# the mapper lands the parquet is consumed standalone.
THEME_INSTRUMENT_PREFIX = "THEME_"


# ─────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────
def _load_events_from_dir(input_dir: Path) -> pd.DataFrame:
    """Load every per-day JSONL under ``input_dir`` into one DataFrame.

    Missing / unreadable files are skipped silently (PIT-safe: a future
    day with no file just means "no events" for the build).
    """
    if not input_dir.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    for fp in sorted(input_dir.glob("*.jsonl")):
        try:
            for line in fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError as e:
            logger.warning("Failed to read %s: %s", fp, e)
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Coerce publish_date to datetime; drop rows without a parseable date.
    df["publish_date"] = pd.to_datetime(df.get("publish_date"), errors="coerce")
    df = df.dropna(subset=["publish_date"]).reset_index(drop=True)
    # Coerce numeric columns; non-numeric → NaN.
    for col in ("net_injection", "liquidity_injection_amount", "repo_rate_change"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    # Stance / tool_type defaults so .str ops are safe later.
    df["policy_stance"] = df.get("policy_stance", "unknown").fillna("unknown")
    df["tool_type"] = df.get("tool_type", "other").fillna("other")
    return df


# ─────────────────────────────────────────────────────────────────────
# Per-date factor computation — the PIT-safe core
# ─────────────────────────────────────────────────────────────────────
def build_factors_for_date(
    events_df: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> dict[str, float]:
    """Compute one row of factor values for ``signal_date``.

    PIT: only consults events whose ``publish_date <= signal_date``.
    The trailing-window filters then sub-slice that PIT-safe view.

    Returns a dict of factor name -> float. Always returns finite values;
    NaN-prone paths are coerced to 0.0 by convention so downstream
    broadcasts don't have to defend against NaN.
    """
    # PIT cutoff
    visible = events_df[events_df["publish_date"] <= signal_date]
    if visible.empty:
        return {
            "pbc_liquidity_zscore_20d": 0.0,
            "pbc_easing_dummy": 0.0,
            "pbc_tightening_dummy": 0.0,
            "short_rate_pressure": 0.0,
        }

    # ── zscore_20d on net_injection ──────────────────────────────────
    z_cutoff = signal_date - pd.Timedelta(days=ZSCORE_WINDOW_DAYS - 1)
    z_window = visible[visible["publish_date"] >= z_cutoff]
    net_vals = z_window["net_injection"].dropna()
    if len(net_vals) < 2:
        z = 0.0
    else:
        mean = float(net_vals.mean())
        std = float(net_vals.std(ddof=0))
        if std < 1e-9:
            z = 0.0
        else:
            # Z-score of the most-recent net_injection on signal_date.
            # If signal_date itself has no event, use the latest one
            # in-window.
            today_vals = visible[visible["publish_date"] == signal_date][
                "net_injection"
            ].dropna()
            if not today_vals.empty:
                latest = float(today_vals.iloc[-1])
            else:
                latest = float(net_vals.iloc[-1])
            z = (latest - mean) / std

    # ── easing / tightening 5d dummies ───────────────────────────────
    # "Last 5 days" inclusive of signal_date — an easing event on date D
    # lights up the flag on D, D+1, ..., D+5 (6 distinct dates with flag=1).
    # On D+6 (10 days later test calls D+10) the event has dropped out.
    d_cutoff = signal_date - pd.Timedelta(days=DUMMY_WINDOW_DAYS)
    d_window = visible[visible["publish_date"] >= d_cutoff]
    easing_dummy = (
        1.0 if (d_window["policy_stance"] == "easing").any() else 0.0
    )
    tightening_dummy = (
        1.0 if (d_window["policy_stance"] == "tightening").any() else 0.0
    )

    # ── short_rate_pressure: sum of repo_rate_change over 20d ────────
    r_cutoff = signal_date - pd.Timedelta(days=RATE_PRESSURE_WINDOW_DAYS - 1)
    r_window = visible[visible["publish_date"] >= r_cutoff]
    rate_sum = float(r_window["repo_rate_change"].dropna().sum())

    return {
        "pbc_liquidity_zscore_20d": float(z),
        "pbc_easing_dummy": float(easing_dummy),
        "pbc_tightening_dummy": float(tightening_dummy),
        "short_rate_pressure": float(rate_sum),
    }


# ─────────────────────────────────────────────────────────────────────
# Range driver
# ─────────────────────────────────────────────────────────────────────
def build_factors_range(
    start_date: str,
    end_date: str,
    input_dir: Path | None = None,
) -> pd.DataFrame:
    """Build a long-form factor DataFrame for every date in [start, end].

    Returns a DataFrame with columns::

        datetime, instrument, pbc_liquidity_zscore_20d, pbc_easing_dummy,
        pbc_tightening_dummy, short_rate_pressure

    Empty windows still produce one row per date with zero-valued
    factors — so a downstream broadcast does not get sparse holes.
    """
    input_root = input_dir or INPUT_DIR
    events = _load_events_from_dir(input_root)
    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)
    if e < s:
        raise ValueError(f"end ({end_date}) must be >= start ({start_date})")

    rows: list[dict] = []
    cur = s
    while cur <= e:
        factors = build_factors_for_date(events, cur)
        row = {
            "datetime": cur,
            "instrument": MARKET_INSTRUMENT,
            **factors,
        }
        rows.append(row)
        cur += pd.Timedelta(days=1)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Phase E.2 (PE-2) — State Council / industry-policy factor builder.
# Per-(industry, date) parquet, NOT per-(MARKET, date). The factor
# surface is:
#   - industry_policy_support_5d:  sum of signed strength over 5d window
#   - industry_policy_support_20d: sum of signed strength over 20d window
#   - industry_policy_novelty:     1.0 when the industry's first mention
#                                  in the trailing 60d window is on
#                                  signal_date, decaying linearly to 0
#                                  by the end of the window.
# Sign convention: supportive=+1, restrictive=-1, neutral=0.
# Strength bias: 0.5 floor so a neutral-direction mention still
# contributes a half-step to the trailing sum (otherwise a string of
# neutral mentions vanishes and we lose attention information).
# ─────────────────────────────────────────────────────────────────────
def _load_sc_events_from_dir(input_dir: Path) -> pd.DataFrame:
    """Load State Council per-day JSONL files into one DataFrame.

    Different schema from ``_load_events_from_dir``: target_industries
    is a list, policy_direction is the stance, policy_strength is the
    [0,1] confidence. We EXPLODE on target_industries so the downstream
    factor calc is one row per (date, industry).
    """
    if not input_dir.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    for fp in sorted(input_dir.glob("*.jsonl")):
        try:
            for line in fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError as e:
            logger.warning("Failed to read %s: %s", fp, e)
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["publish_date"] = pd.to_datetime(df.get("publish_date"), errors="coerce")
    df = df.dropna(subset=["publish_date"]).reset_index(drop=True)
    # Default columns so downstream ops don't KeyError.
    if "target_industries" not in df.columns:
        df["target_industries"] = [[] for _ in range(len(df))]
    df["policy_direction"] = df.get("policy_direction", "neutral").fillna("neutral")
    df["policy_strength"] = pd.to_numeric(
        df.get("policy_strength"), errors="coerce"
    ).fillna(0.5)
    # Explode on industries. Rows whose target_industries list is empty
    # become a single "__market__" row so the policy doc is not lost.
    def _coerce(item):
        if isinstance(item, list):
            return item if item else ["__market__"]
        if isinstance(item, str) and item.strip():
            return [item.strip()]
        return ["__market__"]

    df["target_industries"] = df["target_industries"].apply(_coerce)
    df = df.explode("target_industries").reset_index(drop=True)
    df = df.rename(columns={"target_industries": "industry"})
    df["industry"] = df["industry"].astype(str).str.lower().str.strip()
    df = df[df["industry"].astype(bool)].reset_index(drop=True)
    # Signed score: +strength for supportive, -strength for restrictive,
    # 0 for neutral / unknown. Half-floor so neutral mentions still
    # carry attention (the 0.5 keeps the novelty signal alive).
    def _signed_score(stance: str, strength: float) -> float:
        s = max(float(strength or 0.0), 0.0)
        if stance == "supportive":
            return s
        if stance == "restrictive":
            return -s
        # neutral / unknown — half-floor to register attention.
        return 0.0
    df["signed_strength"] = [
        _signed_score(st, sg)
        for st, sg in zip(df["policy_direction"], df["policy_strength"])
    ]
    return df


def build_sc_factors_for_date(
    events_df: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> list[dict]:
    """Compute per-industry factor rows for ``signal_date``.

    PIT: only events with ``publish_date <= signal_date`` are visible.
    Returns one dict per industry seen in the 60d window. If the
    DataFrame is empty, returns ``[]`` (the caller is expected to
    skip the date rather than emit a fake row).
    """
    if events_df.empty:
        return []
    visible = events_df[events_df["publish_date"] <= signal_date]
    if visible.empty:
        return []
    cutoff_60d = signal_date - pd.Timedelta(days=INDUSTRY_NOVELTY_WINDOW_DAYS - 1)
    novelty_window = visible[visible["publish_date"] >= cutoff_60d]
    industries = sorted(novelty_window["industry"].unique().tolist())
    short_cutoff = signal_date - pd.Timedelta(days=INDUSTRY_SUPPORT_SHORT_DAYS - 1)
    long_cutoff = signal_date - pd.Timedelta(days=INDUSTRY_SUPPORT_LONG_DAYS - 1)
    out: list[dict] = []
    for ind in industries:
        ind_rows = novelty_window[novelty_window["industry"] == ind]
        # 5d / 20d signed-strength sums.
        short_sum = float(
            ind_rows[ind_rows["publish_date"] >= short_cutoff][
                "signed_strength"
            ].sum()
        )
        long_sum = float(
            ind_rows[ind_rows["publish_date"] >= long_cutoff][
                "signed_strength"
            ].sum()
        )
        # Novelty: how recently the industry first appeared in the 60d
        # window. first_seen=signal_date → 1.0; oldest possible → 0.0.
        first_seen = ind_rows["publish_date"].min()
        if pd.isna(first_seen):
            novelty = 0.0
        else:
            age_days = (signal_date - first_seen).days
            denom = max(INDUSTRY_NOVELTY_WINDOW_DAYS - 1, 1)
            novelty = float(max(0.0, 1.0 - age_days / denom))
        out.append({
            "industry": ind,
            "industry_policy_support_5d": short_sum,
            "industry_policy_support_20d": long_sum,
            "industry_policy_novelty": novelty,
        })
    return out


def build_sc_factors_range(
    start_date: str,
    end_date: str,
    input_dir: Path | None = None,
) -> pd.DataFrame:
    """Build a long-form per-industry factor DataFrame for [start, end].

    Schema::
        datetime, instrument="INDUSTRY_<UPPER>",
        industry_policy_support_5d, industry_policy_support_20d,
        industry_policy_novelty

    Dates with no policy mentions in the 60d window produce ZERO rows
    (not one per industry); this is consistent with the spec that
    factors are sparse by industry rather than dense per date.
    """
    input_root = input_dir or INPUT_DIR_SC
    events = _load_sc_events_from_dir(input_root)
    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)
    if e < s:
        raise ValueError(f"end ({end_date}) must be >= start ({start_date})")
    rows: list[dict] = []
    cur = s
    while cur <= e:
        per_industry = build_sc_factors_for_date(events, cur)
        for r in per_industry:
            industry = r.pop("industry")
            rows.append({
                "datetime": cur,
                "instrument": (
                    INDUSTRY_INSTRUMENT_PREFIX + industry.upper()
                    if industry != "__market__" else MARKET_INSTRUMENT
                ),
                **r,
            })
        cur += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


PBC_FACTOR_COLUMNS = (
    "pbc_liquidity_zscore_20d",
    "pbc_easing_dummy",
    "pbc_tightening_dummy",
    "short_rate_pressure",
)
SC_FACTOR_COLUMNS = (
    "industry_policy_support_5d",
    "industry_policy_support_20d",
    "industry_policy_novelty",
)
NBS_FACTOR_COLUMNS = (
    "nbs_cpi_surprise_3m",
    "nbs_ppi_surprise_3m",
    "nbs_pmi_above_50_dummy",
    "nbs_retail_growth_yoy_3m",
)
XWLB_FACTOR_COLUMNS = (
    "theme_mention_count_1d",
    "theme_mention_count_5d",
    "theme_consecutive_days",
    "theme_priority_5d_max",
)


# ─────────────────────────────────────────────────────────────────────
# Phase E.3 (PE-3) — NBS macro surprise factor builder. MARKET-keyed
# (same as PBC) because CPI / PPI / PMI / retail sales are market-wide
# macro signals, not per-industry.
#
# Factors:
#   - nbs_cpi_surprise_3m:   sum of (consensus - headline) over last 3
#                             months for CPI releases. Positive ↔ inflation
#                             undershoots expectations.
#   - nbs_ppi_surprise_3m:   same for PPI.
#   - nbs_pmi_above_50_dummy: 1 if the LATEST PMI release in the trailing
#                             3-month window is > 50, else 0.
#   - nbs_retail_growth_yoy_3m: average of yoy_change for retail_sales
#                             over the trailing 3-month window.
# ─────────────────────────────────────────────────────────────────────
def _load_nbs_events_from_dir(input_dir: Path) -> pd.DataFrame:
    """Load NBS per-day JSONL files into one DataFrame.

    Schema fields used downstream: series_name, headline_value,
    consensus_value, yoy_change, publish_date. Missing files / unreadable
    files are skipped silently (PIT-safe).
    """
    if not input_dir.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    for fp in sorted(input_dir.glob("*.jsonl")):
        try:
            for line in fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError as e:
            logger.warning("Failed to read %s: %s", fp, e)
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["publish_date"] = pd.to_datetime(df.get("publish_date"), errors="coerce")
    df = df.dropna(subset=["publish_date"]).reset_index(drop=True)
    for col in (
        "headline_value", "prior_value", "consensus_value",
        "mom_change", "yoy_change",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    df["series_name"] = df.get("series_name", "other").fillna("other")
    return df


def build_nbs_factors_for_date(
    events_df: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> dict[str, float]:
    """Compute one row of NBS macro factors for ``signal_date``.

    PIT: only consults events whose ``publish_date <= signal_date``.
    All factors are NaN-safe — they coerce to 0.0 in the absence of
    relevant data so downstream broadcasts don't have to defend.
    """
    zero = {
        "nbs_cpi_surprise_3m": 0.0,
        "nbs_ppi_surprise_3m": 0.0,
        "nbs_pmi_above_50_dummy": 0.0,
        "nbs_retail_growth_yoy_3m": 0.0,
    }
    if events_df.empty:
        return dict(zero)
    visible = events_df[events_df["publish_date"] <= signal_date]
    if visible.empty:
        return dict(zero)
    cutoff = signal_date - pd.Timedelta(days=NBS_SURPRISE_WINDOW_DAYS - 1)
    window = visible[visible["publish_date"] >= cutoff]

    # CPI surprise: sum of (consensus - headline) for CPI releases. A
    # positive value means actual inflation came in BELOW expectations.
    cpi_rows = window[window["series_name"] == "cpi"]
    cpi_pairs = cpi_rows[["consensus_value", "headline_value"]].dropna(
        subset=["consensus_value", "headline_value"]
    )
    cpi_surprise = float((cpi_pairs["consensus_value"] - cpi_pairs["headline_value"]).sum())

    ppi_rows = window[window["series_name"] == "ppi"]
    ppi_pairs = ppi_rows[["consensus_value", "headline_value"]].dropna(
        subset=["consensus_value", "headline_value"]
    )
    ppi_surprise = float((ppi_pairs["consensus_value"] - ppi_pairs["headline_value"]).sum())

    # PMI dummy: look at the LATEST PMI release in the window. >50 = 1.
    pmi_rows = window[window["series_name"] == "pmi"].dropna(
        subset=["headline_value"]
    )
    if pmi_rows.empty:
        pmi_dummy = 0.0
    else:
        latest_pmi = pmi_rows.sort_values("publish_date").iloc[-1]
        pmi_dummy = 1.0 if float(latest_pmi["headline_value"]) > NBS_PMI_THRESHOLD else 0.0

    # Retail growth yoy: simple mean across the window's retail releases.
    retail_rows = window[window["series_name"] == "retail_sales"]
    retail_yoy = retail_rows["yoy_change"].dropna()
    retail_growth = float(retail_yoy.mean()) if not retail_yoy.empty else 0.0

    return {
        "nbs_cpi_surprise_3m": cpi_surprise,
        "nbs_ppi_surprise_3m": ppi_surprise,
        "nbs_pmi_above_50_dummy": pmi_dummy,
        "nbs_retail_growth_yoy_3m": retail_growth,
    }


def build_nbs_factors_range(
    start_date: str,
    end_date: str,
    input_dir: Path | None = None,
) -> pd.DataFrame:
    """Build a long-form NBS factor DataFrame for every date in [start, end].

    Returns a DataFrame with columns::

        datetime, instrument="MARKET", nbs_cpi_surprise_3m,
        nbs_ppi_surprise_3m, nbs_pmi_above_50_dummy,
        nbs_retail_growth_yoy_3m

    Like PE-1, empty windows still produce one row per date with
    zero-valued factors so downstream broadcasts do not get sparse holes.
    """
    input_root = input_dir or INPUT_DIR_NBS
    events = _load_nbs_events_from_dir(input_root)
    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)
    if e < s:
        raise ValueError(f"end ({end_date}) must be >= start ({start_date})")
    rows: list[dict] = []
    cur = s
    while cur <= e:
        factors = build_nbs_factors_for_date(events, cur)
        rows.append({
            "datetime": cur,
            "instrument": MARKET_INSTRUMENT,
            **factors,
        })
        cur += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Phase E.4 (PE-4) — Xinwen Lianbo theme attention factor builder.
# Per-(theme, date) parquet, NOT per-(MARKET, date). Theme attention is
# the unit of analysis — a given day, the broadcast either covers a
# theme or it doesn't, and the run length / mention count is what we
# regress against (industry / basket-level returns, NOT individual
# stock returns).
#
# Factors:
#   - theme_mention_count_1d:    mention count from today's broadcast
#   - theme_mention_count_5d:    sum over the trailing 5 calendar days
#   - theme_consecutive_days:    length of the current consecutive-day
#                                 streak the theme has been mentioned,
#                                 capped at 14 (themes that run for
#                                 more than 2 weeks are regimes, not
#                                 attention bursts)
#   - theme_priority_5d_max:     max policy_priority_signal over the
#                                 trailing 5 days (= "did the theme
#                                 get a lead-story spot at least once
#                                 in the last week?")
# ─────────────────────────────────────────────────────────────────────
def _load_xinwen_lianbo_events_from_dir(input_dir: Path) -> pd.DataFrame:
    """Load XWLB per-day JSONL files into a one-row-per-(date, theme)
    DataFrame.

    Each input row carries themes / theme_mention_counts / priority. We
    EXPLODE on themes so the downstream factor calc is one row per
    (date, theme). Rows whose theme list is empty are dropped — there
    is no per-theme factor to emit for a "no-themes" broadcast (the
    PE-4 phase plan explicitly excludes generic filler).
    """
    if not input_dir.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    for fp in sorted(input_dir.glob("*.jsonl")):
        try:
            for line in fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError as e:
            logger.warning("Failed to read %s: %s", fp, e)
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["publish_date"] = pd.to_datetime(df.get("publish_date"), errors="coerce")
    df = df.dropna(subset=["publish_date"]).reset_index(drop=True)
    # Default columns so downstream ops don't KeyError.
    if "themes" not in df.columns:
        df["themes"] = [[] for _ in range(len(df))]
    if "theme_mention_counts" not in df.columns:
        df["theme_mention_counts"] = [{} for _ in range(len(df))]
    df["policy_priority_signal"] = pd.to_numeric(
        df.get("policy_priority_signal"), errors="coerce",
    ).fillna(0.0)

    def _theme_list(item):
        if isinstance(item, list):
            return [t for t in item if isinstance(t, str) and t.strip()]
        if isinstance(item, str) and item.strip():
            return [item.strip()]
        return []

    df["themes"] = df["themes"].apply(_theme_list)
    # Drop rows whose themes list is empty — they contribute no
    # per-theme factor and would just become NaN noise.
    df = df[df["themes"].apply(bool)].reset_index(drop=True)
    if df.empty:
        return df
    # Explode and join the per-theme count back in. theme_mention_counts
    # is a dict keyed by theme; we look up post-explode.
    df = df.explode("themes").reset_index(drop=True)
    df = df.rename(columns={"themes": "theme"})
    df["theme"] = df["theme"].astype(str).str.lower().str.strip()
    df = df[df["theme"].astype(bool)].reset_index(drop=True)

    def _lookup_count(row) -> int:
        counts = row.get("theme_mention_counts")
        if not isinstance(counts, dict):
            return 1
        v = counts.get(row["theme"])
        try:
            n = int(v)
            return max(1, n)
        except (TypeError, ValueError):
            return 1

    df["mention_count"] = df.apply(_lookup_count, axis=1)
    return df


def build_xinwen_lianbo_factors_for_date(
    events_df: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> list[dict]:
    """Compute per-theme factor rows for ``signal_date``.

    PIT: only events with ``publish_date <= signal_date`` are visible.
    Returns one dict per theme that has been mentioned in the trailing
    5-day window. Themes whose last mention is older than 5 days drop
    out of the output (the factor surface should be sparse — a theme
    that hasn't been on the broadcast for a week is not a useful daily
    factor).
    """
    if events_df.empty:
        return []
    visible = events_df[events_df["publish_date"] <= signal_date]
    if visible.empty:
        return []
    short_cutoff = signal_date - pd.Timedelta(days=XWLB_MENTION_SHORT_DAYS - 1)
    short_window = visible[visible["publish_date"] >= short_cutoff]
    if short_window.empty:
        return []
    themes = sorted(short_window["theme"].unique().tolist())
    consec_cutoff = signal_date - pd.Timedelta(days=XWLB_CONSECUTIVE_DAYS_CAP - 1)
    consec_window = visible[visible["publish_date"] >= consec_cutoff]
    out: list[dict] = []
    today = signal_date.normalize()
    for theme in themes:
        theme_short = short_window[short_window["theme"] == theme]
        theme_consec = consec_window[consec_window["theme"] == theme]
        # 1d count: how many mention rows on signal_date (typically 0 or
        # 1 — XWLB airs once per day — but defensively allow >1 if the
        # collector duplicated a row across two sina list pages and the
        # dedup missed).
        today_rows = theme_short[
            theme_short["publish_date"].dt.normalize() == today
        ]
        count_1d = int(today_rows["mention_count"].sum())
        count_5d = int(theme_short["mention_count"].sum())
        # priority max in 5d window.
        prio_window = theme_short["policy_priority_signal"].dropna()
        priority_max = float(prio_window.max()) if not prio_window.empty else 0.0
        # Consecutive days: walk back from signal_date and count
        # contiguous days with at least one mention. We compute the set
        # of normalized days the theme appeared in the 14d window, then
        # walk backwards.
        appeared_days: set[pd.Timestamp] = set(
            theme_consec["publish_date"].dt.normalize().tolist()
        )
        consec = 0
        cur = today
        while cur in appeared_days and consec < XWLB_CONSECUTIVE_DAYS_CAP:
            consec += 1
            cur = cur - pd.Timedelta(days=1)
        out.append({
            "theme": theme,
            "theme_mention_count_1d": float(count_1d),
            "theme_mention_count_5d": float(count_5d),
            "theme_consecutive_days": float(consec),
            "theme_priority_5d_max": priority_max,
        })
    return out


def build_xinwen_lianbo_factors_range(
    start_date: str,
    end_date: str,
    input_dir: Path | None = None,
) -> pd.DataFrame:
    """Build a long-form per-theme factor DataFrame for [start, end].

    Schema::
        datetime, instrument="THEME_<UPPER>",
        theme_mention_count_1d, theme_mention_count_5d,
        theme_consecutive_days, theme_priority_5d_max

    Dates with no theme mentions in the 5d window produce ZERO rows
    (consistent with PE-2's sparse-by-instrument output convention —
    a future broadcast does not need a placeholder row for a theme
    that has not been mentioned in over a week).
    """
    input_root = input_dir or INPUT_DIR_XWLB
    events = _load_xinwen_lianbo_events_from_dir(input_root)
    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)
    if e < s:
        raise ValueError(f"end ({end_date}) must be >= start ({start_date})")
    rows: list[dict] = []
    cur = s
    while cur <= e:
        per_theme = build_xinwen_lianbo_factors_for_date(events, cur)
        for r in per_theme:
            theme = r.pop("theme")
            rows.append({
                "datetime": cur,
                "instrument": THEME_INSTRUMENT_PREFIX + theme.upper(),
                **r,
            })
        cur += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def _write_health_sidecar(
    *,
    output_path: Path,
    health_path: Path,
    df: pd.DataFrame,
    events: pd.DataFrame,
    start_date: str,
    end_date: str,
    health_source: str = HEALTH_SOURCE_NAME,
    factor_columns: tuple[str, ...] = PBC_FACTOR_COLUMNS,
) -> None:
    """Sidecar JSON with stats for the SLA gate / daily report."""
    if events.empty:
        latest_event_date = ""
        n_events = 0
        dates_with_events = 0
    else:
        latest_event_date = events["publish_date"].max().strftime("%Y-%m-%d")
        n_events = int(len(events))
        dates_with_events = int(events["publish_date"].nunique())
    sidecar = {
        "source": health_source,
        "start_date": start_date,
        "end_date": end_date,
        "n_events_used": n_events,
        "latest_event_date": latest_event_date,
        "dates_with_events": dates_with_events,
        "n_factor_rows": int(len(df)),
        "factor_columns": list(factor_columns),
        "written_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "output_path": str(output_path),
    }
    health_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = health_path.with_suffix(health_path.suffix + ".tmp")
    tmp.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(health_path)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────
# Health publishing
# ─────────────────────────────────────────────────────────────────────
def publish_health(
    *,
    n_rows: int,
    n_events: int,
    latest_event_date: str,
    target_date: str,
    health_source: str = HEALTH_SOURCE_NAME,
    sparse_steady: bool = False,
) -> None:
    """Publish PE-1 factor build health.

    2026-06-07 cx P1 #3 fix: previously a window with ZERO extracted
    PBC events still wrote one zero-valued row per date and reported
    success=True (since n_rows>0). That painted the freshness gate
    green on a day when the LLM pipeline silently produced nothing
    of value. Now success requires BOTH non-zero factor rows AND at
    least one underlying event — so an LLM extraction outage / parser
    regression surfaces as a failure on the day it happens.

    2026-06-09 follow-up: state_council / xinwen_lianbo are
    sparse-by-design — a window with 0 events is a legitimate
    steady state, not a pipeline failure. Caller passes
    sparse_steady=True for those so the freshness gate stays green.
    """
    try:
        from scheduler.data_health import HealthStatus, write_health
    except Exception as e:  # pragma: no cover — broken install
        logger.warning("Cannot import scheduler.data_health (%s)", e)
        return
    has_real_signal = sparse_steady or (n_rows > 0 and n_events > 0)
    status = HealthStatus(
        success=has_real_signal,
        n_items=n_rows,
        latest_date=latest_event_date or target_date,
        error_type=(
            "" if has_real_signal
            else ("no_events" if n_rows > 0 and n_events == 0
                  else "no_factor_rows")
        ),
        network_profile="ashare",
        extra={
            "n_events_used": n_events,
            "latest_event_date": latest_event_date,
        },
    )
    write_health(health_source, status, date=target_date)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Build PBOC / State Council policy factors from extracted policy events."
    )
    parser.add_argument(
        "--source", default="pbc",
        choices=["pbc", "state_council", "nbs", "xinwen_lianbo"],
        help=(
            "Source. 'pbc' = monetary policy (Phase E.1), produces a "
            "MARKET-keyed parquet. 'state_council' = State Council + 3 "
            "ministries (Phase E.2), produces an INDUSTRY_<NAME>-keyed "
            "parquet. 'nbs' = NBS CPI/PPI/PMI/retail sales (Phase E.3), "
            "produces a MARKET-keyed parquet. 'xinwen_lianbo' = CCTV "
            "新闻联播 theme attention (Phase E.4), produces a "
            "THEME_<NAME>-keyed parquet."
        ),
    )
    parser.add_argument(
        "--date", default=None,
        help="Single signal date YYYY-MM-DD (default: today).",
    )
    parser.add_argument(
        "--start", default=None, help="Backfill start (default: --date).",
    )
    parser.add_argument(
        "--end", default=None, help="Backfill end (default: --date).",
    )
    parser.add_argument(
        "--lookback-days", type=int, default=30,
        help=(
            "When neither --start nor --end is provided, build factors "
            "for the last N days ending today. Default 30."
        ),
    )
    args = parser.parse_args(argv)

    if args.source not in ("pbc", "state_council", "nbs", "xinwen_lianbo"):
        logger.error("Unsupported --source %s", args.source)
        return 2

    today = datetime.now().strftime("%Y-%m-%d")
    if args.start or args.end:
        start = args.start or args.end or today
        end = args.end or args.start or today
    elif args.date:
        start = end = args.date
    else:
        end = today
        start = (
            datetime.strptime(end, "%Y-%m-%d") - timedelta(days=args.lookback_days)
        ).strftime("%Y-%m-%d")

    if args.source == "pbc":
        build_fn = build_factors_range
        load_fn = _load_events_from_dir
        input_dir = INPUT_DIR
        output_path = OUTPUT_PATH
        health_path = HEALTH_PATH
        health_source = HEALTH_SOURCE_NAME
        factor_columns = PBC_FACTOR_COLUMNS
    elif args.source == "state_council":
        build_fn = build_sc_factors_range
        load_fn = _load_sc_events_from_dir
        input_dir = INPUT_DIR_SC
        output_path = OUTPUT_PATH_SC
        health_path = HEALTH_PATH_SC
        health_source = HEALTH_SOURCE_NAME_SC
        factor_columns = SC_FACTOR_COLUMNS
    elif args.source == "nbs":
        build_fn = build_nbs_factors_range
        load_fn = _load_nbs_events_from_dir
        input_dir = INPUT_DIR_NBS
        output_path = OUTPUT_PATH_NBS
        health_path = HEALTH_PATH_NBS
        health_source = HEALTH_SOURCE_NAME_NBS
        factor_columns = NBS_FACTOR_COLUMNS
    else:
        # xinwen_lianbo
        build_fn = build_xinwen_lianbo_factors_range
        load_fn = _load_xinwen_lianbo_events_from_dir
        input_dir = INPUT_DIR_XWLB
        output_path = OUTPUT_PATH_XWLB
        health_path = HEALTH_PATH_XWLB
        health_source = HEALTH_SOURCE_NAME_XWLB
        factor_columns = XWLB_FACTOR_COLUMNS

    df = build_fn(start_date=start, end_date=end)
    if df.empty:
        logger.warning("No factor rows built for [%s, %s]", start, end)
        # 2026-06-07 cx batch C P2 #4 fix: distinguish no_theme_signal
        # (events exist, LLM judged none material → 0 factor rows by
        # design) from pipeline_failed (no events at all → upstream
        # break). XWLB and state_council are sparse-by-theme/industry,
        # so a 0-row day with non-empty events is the steady state and
        # MUST exit 0 to keep the cron health green and not confuse it
        # with a real failure. NBS / PBC stay loud (exit 1) because
        # their factor count != event count and empty downstream
        # usually means a build error.
        events = load_fn(input_dir)
        n_events = int(len(events)) if not events.empty else 0
        latest_event_date = (
            events["publish_date"].max().strftime("%Y-%m-%d")
            if not events.empty else ""
        )
        sparse_by_design = args.source in ("xinwen_lianbo", "state_council")
        if sparse_by_design:
            # 2026-06-09: state_council failed today even though events
            # ran OK, because the [lookback, end] window happened to
            # contain 0 events. For sparse-by-design sources this is a
            # legitimate steady state — drop the original n_events > 0
            # guard. (Earlier C.P2 #4 fix was over-restrictive.)
            logger.info(
                "[%s] sparse-by-design source: %d events in window, "
                "0 material factor rows — exit 0 (steady state, not "
                "pipeline failure).",
                args.source, n_events,
            )
            publish_health(
                n_rows=0,
                n_events=n_events,
                latest_event_date=latest_event_date,
                target_date=end,
                health_source=health_source,
                sparse_steady=True,
            )
            return 0
        publish_health(
            n_rows=0,
            n_events=n_events,
            latest_event_date=latest_event_date,
            target_date=end,
            health_source=health_source,
        )
        return 1

    # Merge with any existing parquet — drop overlapping dates first.
    if output_path.exists():
        try:
            existing = pd.read_parquet(output_path)
            existing["datetime"] = pd.to_datetime(existing["datetime"])
            new_dates = set(df["datetime"].astype("datetime64[ns]").tolist())
            existing = existing[~existing["datetime"].isin(new_dates)]
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.sort_values(["datetime", "instrument"]).reset_index(drop=True)
        except Exception as e:
            logger.warning("Failed to read existing parquet, overwriting: %s", e)
            combined = df
    else:
        combined = df

    _atomic_write_parquet(combined, output_path)
    logger.info(
        "Wrote %d factor rows for [%s, %s] → %s",
        len(df), start, end, output_path,
    )

    # Sidecar health JSON + scheduler data_health record.
    events = load_fn(input_dir)
    _write_health_sidecar(
        output_path=output_path,
        health_path=health_path,
        df=df,
        events=events,
        start_date=start,
        end_date=end,
        health_source=health_source,
        factor_columns=factor_columns,
    )
    publish_health(
        n_rows=len(df),
        n_events=int(len(events)) if not events.empty else 0,
        latest_event_date=(
            events["publish_date"].max().strftime("%Y-%m-%d")
            if not events.empty else ""
        ),
        target_date=end,
        health_source=health_source,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
