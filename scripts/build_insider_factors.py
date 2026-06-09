"""SPIKE — Build insider trading factors (LLM Channel 3).

Reads LLM-extracted insider events from ``data/storage/llm_events_insider/``
and emits a per-(stock, date) parquet of insider factors.

PIT contract
------------
Signal day = ``announce_date`` (公告日). Execution day = announce_date + 1 BDay.
Most insider announcements are batched post-15:00 by the exchange feed,
so a +1 BDay lag is the conservative default. The PIT cutoff inside
``build_factors_for_date`` only consults events with
``announce_date <= signal_date`` and the FeatureMerger applies the BDay
shift downstream — matching the convention in
``scripts/build_event_factors.py`` (forecast: ann_date + 1 BDay).

Factor surface
--------------
Per (qlib_code, date) row:

    insider_net_buy_5d_pct
        Sum over last 5 calendar days of ``pct_of_company`` with sign
        from action (+1 增持 / 新进 / 协议受让, -1 减持 / 协议出让,
        0 被动稀释). Scaled to %, e.g. +0.5 = "net insiders bought
        0.5 % of total share cap over 5 days".

    insider_net_buy_20d_pct
        Same over 20 calendar days. Smoother — catches multi-tranche
        减持 plans without the 5d-window noise.

    insider_buy_count_20d
        Count of ``action ∈ {增持, 新进}`` events in last 20 days, with
        a confidence floor (>= 0.7) so LLM downgraded "其他" rows do
        not pollute the count.

    insider_sell_count_20d
        Mirror — action ∈ {减持, 协议转让出让}.

    has_controlling_holder_sell_20d
        1.0 if any ``holder_type ∈ {实控人, 控股股东}`` & action="减持"
        in last 20d, else 0.0. The literature (Lakonishok-Lee 2001)
        treats controlling-shareholder sells as a much stronger negative
        signal than 董监高 sells.

    has_strategic_buy_20d
        1.0 if ``holder_type=战略投资者`` & action ∈ {增持, 新进} in
        last 20d. New strategic-investor stakes are a stable positive
        signal (Seyhun 1986).

    has_committed_no_sell_20d
        1.0 if any ``is_committed_no_sell=True`` event in 20d. This is
        weaker than an actual buy but signals management confidence.

    insider_event_count_5d
        Total Channel-3 events in last 5 days regardless of direction —
        the "anything happening?" attention factor.

Output
------
    data/storage/insider_trading_factors.parquet
        MultiIndex: (datetime, instrument) — instrument is qlib lower-case
        (sh600000 / sz000001 / bj430047) per
        ``factors.feature_cache_utils.CANONICAL_INSTRUMENT_CASE``.
    data/storage/insider_trading_factors.health.json

Cost
----
Pure CPU. No LLM calls. Bounded by event count × date count, with the
PIT cutoff dominated by 20d windows. ~50 events/day × 250 days = 12.5 k
rows of raw events; per-date factor build is O(n_events_in_window) ≈
O(1000) per date. Full-year backfill: < 30 seconds on M2.

SPIKE status
------------
SCAFFOLD ONLY. The aggregation logic is wired but has NOT been run end-
to-end (the LLM extractor has not produced events yet). The IC backtest
gate (see ``docs/llm_channel_3_insider_trading_spike_20260609.md``)
must pass before this builder is wired into the daily cron + the 209
feature cache joiner.

Usage
-----
    # daily cron (today)
    python scripts/build_insider_factors.py

    # explicit backfill window
    python scripts/build_insider_factors.py --start 2026-04-01 --end 2026-06-09
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR = DATA_DIR / "llm_events_insider"
OUTPUT_PATH = DATA_DIR / "insider_trading_factors.parquet"
HEALTH_PATH = DATA_DIR / "insider_trading_factors.health.json"

# Action → direction sign for net-buy aggregation.
DIRECTION_BY_ACTION = {
    "增持": 1,
    "新进": 1,
    "协议转让受让": 1,
    "减持": -1,
    "协议转让出让": -1,
    "被动稀释": 0,
    "其他": 0,
}

WINDOW_SHORT_DAYS = 5
WINDOW_LONG_DAYS = 20
CONFIDENCE_FLOOR_COUNT = 0.7   # ≥ this confidence to count toward count factors


def _load_events(input_dir: Path) -> pd.DataFrame:
    """Load every llm_events_insider/<date>.jsonl into one DataFrame.

    Returns empty DataFrame when the producer has not run yet — the
    caller must handle this rather than 0-fill all factors silently
    (see feature_cache_utils.assert_join_coverage rationale).
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
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Coerce announce_date → datetime. Drop NaT-row (publish date is the
    # whole PIT contract; an event without one is unusable).
    df["announce_date"] = pd.to_datetime(df.get("announce_date"), errors="coerce")
    df = df.dropna(subset=["announce_date"]).reset_index(drop=True)

    # qlib_code → lowercase per CANONICAL_INSTRUMENT_CASE.
    if "qlib_code" in df.columns:
        df["instrument"] = df["qlib_code"].astype(str).str.lower()
    elif "ts_code" in df.columns:
        df["instrument"] = df["ts_code"].astype(str).str.lower()
    else:
        logger.error("insider events missing qlib_code/ts_code — cannot build factors")
        return pd.DataFrame()
    df = df[df["instrument"].str.len() == 8].copy()  # sh600000 / sz000001

    for col in ("shares_changed", "pct_of_company", "confidence"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["action"] = df.get("action", "其他").fillna("其他")
    df["holder_type"] = df.get("holder_type", "其他").fillna("其他")
    df["direction"] = df["action"].map(DIRECTION_BY_ACTION).fillna(0).astype(int)
    # signed_pct: + for buy, - for sell. The LLM emits pct_of_company as a
    # magnitude — we apply sign here, not at the LLM (which is unreliable
    # at signs — see V2 direction handling).
    df["signed_pct"] = df["direction"] * df["pct_of_company"].fillna(0.0)
    return df


def build_factors_for_date(
    events_df: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute insider factors for every (instrument) that has any event
    in the trailing window. Returns long-form DataFrame indexed by
    instrument with factor columns. Rows with zero events in window are
    NOT emitted — the merger 0-fills (NaN-handles) at join time.

    PIT-safe: filters by ``announce_date <= signal_date``.
    """
    visible = events_df[events_df["announce_date"] <= signal_date]
    if visible.empty:
        return pd.DataFrame()

    short_cutoff = signal_date - pd.Timedelta(days=WINDOW_SHORT_DAYS)
    long_cutoff = signal_date - pd.Timedelta(days=WINDOW_LONG_DAYS)

    in_short = visible[visible["announce_date"] >= short_cutoff]
    in_long = visible[visible["announce_date"] >= long_cutoff]

    if in_long.empty:
        return pd.DataFrame()

    # Confidence-floored views for count factors.
    in_long_conf = in_long[in_long["confidence"].fillna(0.0) >= CONFIDENCE_FLOOR_COUNT]

    # Per-instrument aggregations.
    short_net = (
        in_short.groupby("instrument")["signed_pct"].sum().rename("insider_net_buy_5d_pct")
    )
    long_net = (
        in_long.groupby("instrument")["signed_pct"].sum().rename("insider_net_buy_20d_pct")
    )

    buy_mask = in_long_conf["action"].isin({"增持", "新进", "协议转让受让"})
    sell_mask = in_long_conf["action"].isin({"减持", "协议转让出让"})

    buy_count = (
        in_long_conf[buy_mask]
        .groupby("instrument")
        .size()
        .rename("insider_buy_count_20d")
    )
    sell_count = (
        in_long_conf[sell_mask]
        .groupby("instrument")
        .size()
        .rename("insider_sell_count_20d")
    )

    controlling_sell_mask = (
        in_long_conf["holder_type"].isin({"实控人", "控股股东"}) & sell_mask
    )
    has_ctrl_sell = (
        in_long_conf[controlling_sell_mask]
        .groupby("instrument")
        .size()
        .gt(0)
        .astype(float)
        .rename("has_controlling_holder_sell_20d")
    )

    strategic_buy_mask = (
        (in_long_conf["holder_type"] == "战略投资者") & buy_mask
    )
    has_strat_buy = (
        in_long_conf[strategic_buy_mask]
        .groupby("instrument")
        .size()
        .gt(0)
        .astype(float)
        .rename("has_strategic_buy_20d")
    )

    has_committed_no_sell = (
        in_long_conf[in_long_conf.get("is_committed_no_sell", False).fillna(False)]
        .groupby("instrument")
        .size()
        .gt(0)
        .astype(float)
        .rename("has_committed_no_sell_20d")
    )

    short_count = (
        in_short.groupby("instrument").size().rename("insider_event_count_5d")
    )

    out = pd.concat(
        [
            short_net, long_net,
            buy_count, sell_count,
            has_ctrl_sell, has_strat_buy, has_committed_no_sell,
            short_count,
        ],
        axis=1,
    )
    # 0-fill discrete counters; net_pct also 0-fills (zero net activity is
    # the correct read for "no insider movement in window").
    out = out.fillna(0.0)
    out["datetime"] = signal_date
    return out.reset_index()


def build_factors_range(
    start_date: str,
    end_date: str,
    input_dir: Path | None = None,
) -> pd.DataFrame:
    events = _load_events(input_dir or INPUT_DIR)
    if events.empty:
        logger.warning(
            "no insider events found in %s — output will be empty parquet",
            input_dir or INPUT_DIR,
        )
        return pd.DataFrame()

    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)
    if e < s:
        raise ValueError(f"end ({end_date}) must be >= start ({start_date})")

    frames = []
    cur = s
    while cur <= e:
        day_frame = build_factors_for_date(events, cur)
        if not day_frame.empty:
            frames.append(day_frame)
        cur += pd.Timedelta(days=1)
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # Conform to the parquet contract of every other Channel-N factor
    # parquet on disk: MultiIndex (datetime, instrument).
    out = out.set_index(["datetime", "instrument"]).sort_index()
    return out


def write_health(df: pd.DataFrame, output_path: Path) -> None:
    n_rows = len(df)
    n_instruments = df.index.get_level_values("instrument").nunique() if n_rows else 0
    if n_rows:
        latest = df.index.get_level_values("datetime").max()
        latest_iso = pd.Timestamp(latest).strftime("%Y-%m-%d")
    else:
        latest_iso = ""
    health = {
        "source": "insider_trading_factors",
        "n_rows": n_rows,
        "n_instruments": n_instruments,
        "latest_date": latest_iso,
        "factor_cols": list(df.columns) if n_rows else [],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "spike_status": "SCAFFOLD_ONLY — see docs/llm_channel_3_insider_trading_spike_20260609.md",
    }
    HEALTH_PATH.write_text(json.dumps(health, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build insider trading factors (LLM Channel 3)")
    p.add_argument("--start", default=None, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", default=None, help="YYYY-MM-DD inclusive")
    p.add_argument("--lookback-days", type=int, default=30, help="When --start/--end omitted")
    args = p.parse_args(argv)

    today = datetime.now().strftime("%Y-%m-%d")
    if args.start is None and args.end is None:
        end = today
        start = (datetime.now() - timedelta(days=args.lookback_days)).strftime("%Y-%m-%d")
    elif args.start is None:
        start = (datetime.strptime(args.end, "%Y-%m-%d") - timedelta(days=args.lookback_days)).strftime("%Y-%m-%d")
        end = args.end
    elif args.end is None:
        start = args.start
        end = today
    else:
        start, end = args.start, args.end

    logger.info("Building insider factors for [%s, %s]", start, end)
    df = build_factors_range(start, end)
    if df.empty:
        logger.warning("SPIKE: empty factor table — writing empty parquet + health for trace")
        # Write an empty parquet anyway so the downstream cache joiner
        # gets a deterministic "no events yet" signal instead of an
        # ENOENT.
        empty = pd.DataFrame(
            columns=[
                "insider_net_buy_5d_pct", "insider_net_buy_20d_pct",
                "insider_buy_count_20d", "insider_sell_count_20d",
                "has_controlling_holder_sell_20d", "has_strategic_buy_20d",
                "has_committed_no_sell_20d", "insider_event_count_5d",
            ]
        )
        empty.index = pd.MultiIndex.from_tuples([], names=["datetime", "instrument"])
        empty.to_parquet(OUTPUT_PATH)
        write_health(empty, OUTPUT_PATH)
        return 0

    df.to_parquet(OUTPUT_PATH)
    write_health(df, OUTPUT_PATH)
    logger.info(
        "wrote %d rows × %d cols, %d unique instruments → %s",
        len(df),
        df.shape[1],
        df.index.get_level_values("instrument").nunique(),
        OUTPUT_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
