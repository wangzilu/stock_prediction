"""Build PBOC liquidity factors from extracted policy events — Phase E.1 step 3.

Reads ``data/storage/policy_events/pbc/<YYYY-MM-DD>.jsonl`` produced by
``scripts/extract_policy_events.py`` and emits a single parquet keyed
by ``(datetime, "MARKET")`` — the synthetic instrument convention the
cross_market_regime overlay uses to broadcast a market-level signal to
every stock.

Output
------
    data/storage/pbc_liquidity_factors.parquet
        Columns: datetime, instrument="MARKET", pbc_liquidity_zscore_20d,
        pbc_easing_dummy, pbc_tightening_dummy, short_rate_pressure
    data/storage/pbc_liquidity_factors.health.json
        Sidecar with n_events_used, latest_event_date, dates_with_events,
        etc. Lets the SLA gate / daily report see what landed.

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

# Window sizes — match the phase doc; kept as constants so changes
# require a code edit rather than a config drift.
ZSCORE_WINDOW_DAYS = 20
DUMMY_WINDOW_DAYS = 5
RATE_PRESSURE_WINDOW_DAYS = 20

# Synthetic instrument the cross-market regime overlay already uses;
# the FeatureMerger broadcasts MARKET-keyed rows to every stock at
# merge time. Kept as a named constant so a future rename is one edit.
MARKET_INSTRUMENT = "MARKET"


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


def _write_health_sidecar(
    *,
    output_path: Path,
    health_path: Path,
    df: pd.DataFrame,
    events: pd.DataFrame,
    start_date: str,
    end_date: str,
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
        "source": HEALTH_SOURCE_NAME,
        "start_date": start_date,
        "end_date": end_date,
        "n_events_used": n_events,
        "latest_event_date": latest_event_date,
        "dates_with_events": dates_with_events,
        "n_factor_rows": int(len(df)),
        "factor_columns": [
            "pbc_liquidity_zscore_20d", "pbc_easing_dummy",
            "pbc_tightening_dummy", "short_rate_pressure",
        ],
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
) -> None:
    try:
        from scheduler.data_health import HealthStatus, write_health
    except Exception as e:  # pragma: no cover — broken install
        logger.warning("Cannot import scheduler.data_health (%s)", e)
        return
    status = HealthStatus(
        success=n_rows > 0,
        n_items=n_rows,
        latest_date=latest_event_date or target_date,
        error_type="" if n_rows > 0 else "no_factor_rows",
        network_profile="ashare",
        extra={
            "n_events_used": n_events,
            "latest_event_date": latest_event_date,
        },
    )
    write_health(HEALTH_SOURCE_NAME, status, date=target_date)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Build PBOC liquidity factors from extracted policy events."
    )
    parser.add_argument(
        "--source", default="pbc", choices=["pbc"],
        help="Source. Only 'pbc' is implemented today.",
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

    if args.source != "pbc":
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

    df = build_factors_range(start_date=start, end_date=end)
    if df.empty:
        logger.warning("No factor rows built for [%s, %s]", start, end)
        return 1

    # Merge with any existing parquet — drop overlapping dates first.
    if OUTPUT_PATH.exists():
        try:
            existing = pd.read_parquet(OUTPUT_PATH)
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

    _atomic_write_parquet(combined, OUTPUT_PATH)
    logger.info(
        "Wrote %d factor rows for [%s, %s] → %s",
        len(df), start, end, OUTPUT_PATH,
    )

    # Sidecar health JSON + scheduler data_health record.
    events = _load_events_from_dir(INPUT_DIR)
    _write_health_sidecar(
        output_path=OUTPUT_PATH,
        health_path=HEALTH_PATH,
        df=df,
        events=events,
        start_date=start,
        end_date=end,
    )
    publish_health(
        n_rows=len(df),
        n_events=int(len(events)) if not events.empty else 0,
        latest_event_date=(
            events["publish_date"].max().strftime("%Y-%m-%d")
            if not events.empty else ""
        ),
        target_date=end,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
