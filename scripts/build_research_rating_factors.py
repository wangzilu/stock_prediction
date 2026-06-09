"""Build per-stock research-rating factors from LLM-extracted reports.

SPIKE 2026-06-09 — LLM Channel #1 factor builder.

Reads:  ``data/storage/research_rating_extracted/<YYYY-MM-DD>.jsonl``
Writes: ``data/storage/research_rating_factors.parquet``
        ``data/storage/research_rating_factors.health.json``

Output schema (long-form, instrument-keyed)::

    datetime    instrument  research_rating_change_score
                            research_eps_revision_pct
                            research_target_upside_pct
                            research_attention_score
                            research_broker_quality_score

where ``instrument`` is the LOWERCASE qlib_code (e.g. ``sh600519``) per
the case-bug fix documented in ``factors/feature_cache_utils.py``.

PIT discipline — CRITICAL
-------------------------
Signal date = ``collected_at`` + 1 BDay. The collector writes its
harvest timestamp into every row; we shift that forward one business
day before keying the parquet. The original ``report_date`` is NEVER
used as a signal date because:

  1. Brokers frequently embargo PDFs until the next morning, so the
     publish-date is forward-leaking when our collector ran EOD.
  2. The collector itself runs after 16:00, by which time same-day
     reports would already have moved the price — we cannot use a
     report published at 09:00 to predict the 11:00 close.

The +1 BDay shift mirrors the pattern used by
``scheduler/data_health.py:_expected_latest_trading_date`` for the
freshness gate.

Factor definitions
------------------
- ``research_rating_change_score``
    Per (instrument, date) average of +2 (upgrade), 0 (reiterate),
    -2 (downgrade), +0.5 (initiate from a high-tier broker), summed
    over a trailing 5-trading-day window. Captures Womack-style
    rating-revision alpha.

- ``research_eps_revision_pct``
    Per (instrument, date) average of ``eps_revision_pct`` over a
    trailing 5-trading-day window. The classic consensus-EPS-revision
    factor (Stickel 1991 / Givoly-Lakonishok). Positive ↔ analysts
    are revising up.

- ``research_target_upside_pct``
    Average of (target_price / spot_close - 1) over the trailing 5d.
    Requires a spot-price join — for the spike we stub this to 0.0
    and document the join (see TODO blocks). Full impl needs to
    read the qlib feature cache's close column.

- ``research_attention_score``
    Count of distinct brokers covering the name in the trailing 20d
    window. Proxy for analyst attention growth — secondary alpha
    documented in Chen-Cheng-Lo (2010).

- ``research_broker_quality_score``
    Weighted sum of rating_change_score by broker-tier quality. Tier
    weights are spike-stubbed (中信/中金/海通 = 1.0, regional = 0.5)
    — production needs a Womack-style historical-accuracy weighting.

Usage
-----
    # daily cron mode
    python scripts/build_research_rating_factors.py

    # backfill window
    python scripts/build_research_rating_factors.py \\
        --start 2026-05-01 --end 2026-06-08
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

INPUT_DIR = DATA_DIR / "research_rating_extracted"
OUTPUT_PATH = DATA_DIR / "research_rating_factors.parquet"
HEALTH_PATH = DATA_DIR / "research_rating_factors.health.json"
HEALTH_SOURCE_NAME = "research_rating_factors"

# Trailing-window sizes (in calendar days for simplicity; matches the
# PE-* factor convention. A future revision can move to trading days
# via qlib's calendar — see scheduler/data_health._expected_latest_trading_date).
RATING_WINDOW_DAYS = 5
EPS_WINDOW_DAYS = 5
TARGET_UPSIDE_WINDOW_DAYS = 5
ATTENTION_WINDOW_DAYS = 20

# Rating-change score map. "initiate" gets a modest +0.5 because
# coverage initiation by a Tier-1 house is a documented positive
# event but smaller than a true upgrade.
RATING_CHANGE_SCORE = {
    "upgrade": 2.0,
    "downgrade": -2.0,
    "reiterate": 0.0,
    "initiate": 0.5,
}

# Broker-tier weighting — STUB. Production needs a Womack-style
# rolling backward-looking accuracy score, NOT a hand-rolled list.
# The values here are for scaffold-only and SHOULD NOT ship as-is.
BROKER_TIER_WEIGHT = {
    "中信证券": 1.0, "中金公司": 1.0, "海通证券": 1.0, "国泰君安": 1.0,
    "招商证券": 0.9, "申万宏源": 0.9, "广发证券": 0.9, "华泰证券": 0.9,
    "兴业证券": 0.8, "国信证券": 0.8, "光大证券": 0.8,
    "_default": 0.5,
}

FACTOR_COLUMNS = (
    "research_rating_change_score",
    "research_eps_revision_pct",
    "research_target_upside_pct",
    "research_attention_score",
    "research_broker_quality_score",
)


def _broker_weight(broker: str) -> float:
    if not broker:
        return BROKER_TIER_WEIGHT["_default"]
    for k, v in BROKER_TIER_WEIGHT.items():
        if k == "_default":
            continue
        if k in broker:
            return v
    return BROKER_TIER_WEIGHT["_default"]


def _load_extracted(input_dir: Path) -> pd.DataFrame:
    """Load all per-day extracted JSONLs into one DataFrame.

    Adds two derived columns up front:
      * ``signal_date`` := collected_at + 1 BDay  (PIT lag)
      * ``broker_weight`` from BROKER_TIER_WEIGHT
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

    # Canonical lowercase instrument key — see
    # factors/feature_cache_utils.normalize_instrument_index. The
    # B.6.3 LLM verdict was PRNG drift precisely because the producer
    # wrote UPPERCASE qlib_code. Enforce here at ingest, NOT in the
    # base-cache joiner, so the parquet on disk is canonical.
    df["instrument"] = df.get("qlib_code", "").astype(str).str.lower()

    # PIT lag: signal_date = collected_at + 1 BDay
    collected = pd.to_datetime(df.get("collected_at"), errors="coerce")
    # If collected_at is NaT (legacy row), fall back to report_date+1B
    # — still PIT-safe but coarser.
    fallback = pd.to_datetime(df.get("report_date"), errors="coerce")
    collected = collected.fillna(fallback)
    df["signal_date"] = collected + pd.tseries.offsets.BDay(1)
    df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.normalize()

    # Coerce numerics
    for col in ("eps_revision_pct", "target_price", "eps_current_year",
                "eps_next_year", "confidence"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["rating_change"] = df.get("rating_change", "reiterate").fillna("reiterate")
    df["rating_change_score"] = df["rating_change"].map(RATING_CHANGE_SCORE).fillna(0.0)
    df["broker"] = df.get("broker", "").fillna("")
    df["broker_weight"] = df["broker"].apply(_broker_weight)
    df = df.dropna(subset=["instrument", "signal_date"]).reset_index(drop=True)
    df = df[df["instrument"].astype(bool)].reset_index(drop=True)
    return df


def _target_upside_pct(target_price: float, instrument: str,
                       signal_date: pd.Timestamp) -> float:
    """Compute (target_price / latest_close - 1).

    SPIKE: returns NaN — full impl requires reading the qlib feature
    cache's close column at signal_date. The join is non-trivial in a
    scaffold (the cache lives at ``data/storage/feature_cache_209_*.parquet``
    with a different schema convention than the LLM event tables) so
    we leave it as a TODO and let the factor degrade gracefully.
    """
    # TODO(spike): join with qlib feature cache:
    #   close = qlib.D.features([instrument], ["$close"],
    #                           start_time=signal_date, end_time=signal_date)
    # then pct = target_price / close - 1
    return float("nan")


def build_factors_for_date(events_df: pd.DataFrame,
                           signal_date: pd.Timestamp) -> pd.DataFrame:
    """Compute per-instrument factor rows for ``signal_date``.

    PIT: only events whose ``signal_date <= signal_date`` are visible.
    Returns a DataFrame keyed by instrument with the 5 factor columns.
    Empty input → empty output (sparse-by-design, like PE-2/PE-4).
    """
    if events_df.empty:
        return pd.DataFrame()
    visible = events_df[events_df["signal_date"] <= signal_date]
    if visible.empty:
        return pd.DataFrame()

    rating_cutoff = signal_date - pd.Timedelta(days=RATING_WINDOW_DAYS - 1)
    eps_cutoff = signal_date - pd.Timedelta(days=EPS_WINDOW_DAYS - 1)
    upside_cutoff = signal_date - pd.Timedelta(days=TARGET_UPSIDE_WINDOW_DAYS - 1)
    attn_cutoff = signal_date - pd.Timedelta(days=ATTENTION_WINDOW_DAYS - 1)

    rating_w = visible[visible["signal_date"] >= rating_cutoff]
    eps_w = visible[visible["signal_date"] >= eps_cutoff]
    upside_w = visible[visible["signal_date"] >= upside_cutoff]
    attn_w = visible[visible["signal_date"] >= attn_cutoff]

    instruments = sorted(rating_w["instrument"].unique().tolist())
    if not instruments:
        # Fall back to attention-window instruments — a stock might
        # have analyst attention but no rating moves in the 5d window.
        instruments = sorted(attn_w["instrument"].unique().tolist())
    if not instruments:
        return pd.DataFrame()

    out_rows: list[dict] = []
    for inst in instruments:
        r_rows = rating_w[rating_w["instrument"] == inst]
        e_rows = eps_w[eps_w["instrument"] == inst]
        u_rows = upside_w[upside_w["instrument"] == inst]
        a_rows = attn_w[attn_w["instrument"] == inst]

        rating_score = float(r_rows["rating_change_score"].sum())
        eps_pct = float(e_rows["eps_revision_pct"].dropna().mean()) if not e_rows.empty else 0.0
        if not np.isfinite(eps_pct):
            eps_pct = 0.0

        # target_upside: STUB — see _target_upside_pct docstring.
        upside_vals = []
        for _, row in u_rows.iterrows():
            tp = row.get("target_price")
            if tp is None or not np.isfinite(tp):
                continue
            v = _target_upside_pct(float(tp), inst, signal_date)
            if np.isfinite(v):
                upside_vals.append(v)
        target_upside = float(np.mean(upside_vals)) if upside_vals else 0.0

        attention = float(a_rows["broker"].nunique())
        broker_quality = float(
            (r_rows["rating_change_score"] * r_rows["broker_weight"]).sum()
        )

        out_rows.append({
            "datetime": signal_date,
            "instrument": inst,
            "research_rating_change_score": rating_score,
            "research_eps_revision_pct": eps_pct,
            "research_target_upside_pct": target_upside,
            "research_attention_score": attention,
            "research_broker_quality_score": broker_quality,
        })
    return pd.DataFrame(out_rows)


def build_factors_range(start_date: str, end_date: str,
                        input_dir: Path | None = None) -> pd.DataFrame:
    """Build a long-form per-instrument factor DataFrame for [start, end]."""
    input_root = input_dir or INPUT_DIR
    events = _load_extracted(input_root)
    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)
    if e < s:
        raise ValueError(f"end ({end_date}) must be >= start ({start_date})")
    pieces: list[pd.DataFrame] = []
    cur = s
    while cur <= e:
        chunk = build_factors_for_date(events, cur)
        if not chunk.empty:
            pieces.append(chunk)
        cur += pd.Timedelta(days=1)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _write_health_sidecar(*, output_path: Path, health_path: Path,
                          df: pd.DataFrame, events: pd.DataFrame,
                          start_date: str, end_date: str) -> None:
    if events.empty:
        latest_event_date = ""
        n_events = 0
    else:
        latest_event_date = events["signal_date"].max().strftime("%Y-%m-%d")
        n_events = int(len(events))
    sidecar = {
        "source": HEALTH_SOURCE_NAME,
        "start_date": start_date,
        "end_date": end_date,
        "n_events_used": n_events,
        "latest_event_date": latest_event_date,
        "n_factor_rows": int(len(df)),
        "factor_columns": list(FACTOR_COLUMNS),
        "written_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "output_path": str(output_path),
    }
    health_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = health_path.with_suffix(health_path.suffix + ".tmp")
    tmp.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(health_path)


def publish_health(*, n_rows: int, n_events: int, latest_event_date: str,
                   target_date: str) -> None:
    """Publish factor-build health via the scheduler.data_health gate.

    Mirrors ``build_policy_factors.publish_health``. Research-rating
    is sparse-by-design (most stock-days have ZERO new ratings) so a
    0-row day with non-empty events is the steady state — we mark
    sparse_steady=True so the freshness gate stays green.
    """
    try:
        from scheduler.data_health import HealthStatus, write_health
    except Exception as e:  # pragma: no cover
        logger.warning("Cannot import scheduler.data_health (%s)", e)
        return
    sparse_steady = n_rows == 0 and n_events > 0
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
    write_health(HEALTH_SOURCE_NAME, status, date=target_date)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Build research-rating factors from extracted LLM JSONLs.",
    )
    parser.add_argument("--date", default=None, help="Single signal date (default: today)")
    parser.add_argument("--start", default=None, help="Backfill start (default: --date)")
    parser.add_argument("--end", default=None, help="Backfill end (default: --date)")
    parser.add_argument("--lookback-days", type=int, default=30,
                        help="When no --start/--end, build last N days (default 30).")
    args = parser.parse_args(argv)

    today = datetime.now().strftime("%Y-%m-%d")
    if args.start or args.end:
        start = args.start or args.end or today
        end = args.end or args.start or today
    elif args.date:
        start = end = args.date
    else:
        end = today
        start = (datetime.strptime(end, "%Y-%m-%d") -
                 timedelta(days=args.lookback_days)).strftime("%Y-%m-%d")

    df = build_factors_range(start, end)
    events = _load_extracted(INPUT_DIR)

    if df.empty:
        logger.warning("No factor rows built for [%s, %s]", start, end)
        n_events = int(len(events)) if not events.empty else 0
        latest = (events["signal_date"].max().strftime("%Y-%m-%d")
                  if not events.empty else "")
        # Sparse-by-design: research_rating is per-(stock, day) — most
        # (stock, day) pairs have ZERO new rating moves. A zero-row
        # window with non-empty events is steady state, not failure.
        if n_events > 0:
            logger.info(
                "research_rating sparse-by-design: %d events in store, "
                "0 material factor rows in window — exit 0 (steady state).",
                n_events,
            )
            publish_health(n_rows=0, n_events=n_events,
                           latest_event_date=latest, target_date=end)
            return 0
        publish_health(n_rows=0, n_events=0,
                       latest_event_date="", target_date=end)
        return 1

    # Merge with existing parquet — drop overlapping (datetime, instrument).
    if OUTPUT_PATH.exists():
        try:
            existing = pd.read_parquet(OUTPUT_PATH)
            existing["datetime"] = pd.to_datetime(existing["datetime"])
            # Drop overlapping rows so the new window wins.
            merge_keys = list(zip(df["datetime"], df["instrument"]))
            existing_keys = list(zip(existing["datetime"], existing["instrument"]))
            mask = [k not in set(merge_keys) for k in existing_keys]
            existing = existing[mask]
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.sort_values(["datetime", "instrument"]).reset_index(drop=True)
        except Exception as e:
            logger.warning("Failed to read existing parquet, overwriting: %s", e)
            combined = df
    else:
        combined = df

    _atomic_write_parquet(combined, OUTPUT_PATH)
    logger.info("Wrote %d factor rows for [%s, %s] -> %s",
                len(df), start, end, OUTPUT_PATH)

    _write_health_sidecar(
        output_path=OUTPUT_PATH, health_path=HEALTH_PATH,
        df=df, events=events, start_date=start, end_date=end,
    )
    publish_health(
        n_rows=len(df),
        n_events=int(len(events)) if not events.empty else 0,
        latest_event_date=(events["signal_date"].max().strftime("%Y-%m-%d")
                            if not events.empty else ""),
        target_date=end,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
