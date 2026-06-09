"""SPIKE — Build per-(instrument, date) factors from regulator events.

Channel 4 factor build (mirrors scripts/build_policy_factors.py and
scripts/build_llm_event_factors.py).

Inputs
------
``data/storage/regulator_events/<YYYY-MM-DD>.jsonl`` — output of
``factors/regulator_penalty_extractor.py``.

Output
------
``data/storage/regulator_penalty_factors.parquet`` keyed on
``(datetime, instrument)`` with lowercase instruments matching the
209 production cache convention (see factors/feature_cache_utils.py).

Factor surface (per the SPIKE task brief)
-----------------------------------------
* ``has_penalty``           — -1.0 on any day a stock had a non-inquiry
                              regulator action (severity ≥ fine). 0 else.
* ``severity_max``          — max SEVERITY_SCORE (1-5) over the trailing
                              7 calendar days. 0 if no event.
* ``has_strict_inquiry``    — -0.5 on the day a strict-inquiry letter
                              was filed against the stock. 0 else.
* ``penalty_count_7d``      — count of distinct regulator events over
                              the trailing 7 calendar days (inquiries
                              + penalties together).

PIT discipline
--------------
* ``event_date`` is the signal day.
* Lag of +1 BDay is applied at build time — a regulator action filed on
  date D contributes to the factor row for D+1 BDay (next trading day),
  NEVER for D itself. This mirrors the convention used by
  ``scripts/build_llm_event_factors.py`` for news events.
* Rationale: regulator portals publish AFTER market close, so a same-day
  factor row would leak future information into a model that backtests
  against close-to-close returns.
* The +1 BDay shift is computed against the SSE trading calendar (via
  ``pandas.bdate_range`` with the ``"C"`` custom business-day calendar
  augmented by the SSE holiday list — TODO L1: thread the existing
  ``utils.calendar.get_trading_calendar`` helper).

SPIKE caveats (DO NOT ship before addressing)
---------------------------------------------
1. Multi-stock CSRC press releases (e.g. industry-wide penalty roundup)
   currently emit one event per stock at the LLM layer. The factor
   build SHOULD de-dup on (ts_code, event_date) AFTER the join — a
   single doc that names 5 stocks must NOT multiply-count.
2. The 7-day rolling count uses calendar days for simplicity. If the
   IC backtest shows the factor is dominated by post-holiday clusters
   (CSRC tends to file before long holidays), switch to trading days.
3. Lookahead-bias trap: ``filed_date`` and ``event_date`` are NOT
   the same in the source DOM. CSRC penalty PDFs carry both
   "做出日期" (decision date, earlier) and "公告日期" (publish date,
   later). We use 公告日期 = filed_date as event_date because that
   is when the market could have seen the document. The earlier
   decision date is informational only.
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
from factors.regulator_penalty_extractor import SEVERITY_SCORE  # noqa: E402

logger = logging.getLogger(__name__)

INPUT_DIR = DATA_DIR / "regulator_events"
OUTPUT_PATH = DATA_DIR / "regulator_penalty_factors.parquet"
HEALTH_PATH = DATA_DIR / "regulator_penalty_factors.health.json"
HEALTH_SOURCE_NAME = "regulator_penalty_factors"

# Rolling-window length. Per the SPIKE brief: 7-day count.
ROLLING_WINDOW_DAYS = 7

FACTOR_COLUMNS = (
    "has_penalty",
    "severity_max",
    "has_strict_inquiry",
    "penalty_count_7d",
)

# Per the SPIKE brief:
#   - has_penalty = -1 on a penalty day (severity ≥ fine).
#   - has_strict_inquiry = -0.5 on a strict-inquiry day.
# Penalties (severity ≥ fine) and strict inquiries can co-occur; the
# factor surface keeps them as two columns so the model can learn
# their joint distribution rather than us pre-summing.
PENALTY_FLAG_VALUE = -1.0
STRICT_INQUIRY_FLAG_VALUE = -0.5

# Severity tiers that count as a "real penalty" (vs. a soft warning).
# warning-only days do NOT light up has_penalty; they only contribute
# to penalty_count_7d.
PENALTY_SEVERITIES = {"fine", "suspension", "delisting_warning", "criminal_referral"}


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
def _load_events_from_dir(input_dir: Path) -> pd.DataFrame:
    """Load every regulator_events/<date>.jsonl into one DataFrame.

    Mirrors scripts/build_policy_factors._load_events_from_dir.
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
    df["event_date"] = pd.to_datetime(df.get("event_date"), errors="coerce")
    df = df.dropna(subset=["event_date"]).reset_index(drop=True)
    # Empty ts_code means a market-wide notice — drop here; we are NOT
    # modelling market-wide regulator regimes at this layer (that would
    # be its own MARKET-keyed parquet under E.5).
    df = df[df["ts_code"].astype(str).str.match(r"^\d{6}$", na=False)].reset_index(drop=True)
    df["severity"] = df.get("severity", "warning").fillna("warning")
    df["is_strict_inquiry"] = df.get("is_strict_inquiry", False).fillna(False).astype(bool)
    df["fine_amount_yuan"] = pd.to_numeric(
        df.get("fine_amount_yuan"), errors="coerce",
    ).fillna(0.0)
    # +1 BDay PIT lag — see module docstring.
    df["signal_date"] = df["event_date"] + pd.offsets.BDay(1)
    return df


def _qlib_instrument(ts_code: str) -> str:
    """Convert a 6-digit code to lowercase qlib instrument format.

    See factors/feature_cache_utils.py — the 209 production cache
    keys on lowercase ``sh600519`` / ``sz000001``.
    """
    s = str(ts_code).strip()
    if not s or not s.isdigit() or len(s) != 6:
        return ""
    prefix = "sh" if s.startswith(("6", "9")) else "sz"
    return f"{prefix}{s}"


# ---------------------------------------------------------------------
# Per-date factor row computation — PIT-safe core.
# ---------------------------------------------------------------------
def build_factors_for_date(
    events_df: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> list[dict]:
    """Compute per-instrument factor rows for ``signal_date``.

    PIT: only consults events whose ``signal_date <= signal_date``
    (i.e. event_date+1BDay <= signal_date). The +1BDay shift is
    pre-applied in ``_load_events_from_dir`` so the comparison is a
    plain <=.

    Returns one dict per instrument that had a regulator event in
    the trailing ROLLING_WINDOW_DAYS window. Instruments with no
    recent events are omitted (sparse-by-design — matches PE-4 XWLB
    convention, the FeatureMerger fillna(0)'s the gaps).
    """
    if events_df.empty:
        return []
    visible = events_df[events_df["signal_date"] <= signal_date]
    if visible.empty:
        return []

    rolling_cutoff = signal_date - pd.Timedelta(days=ROLLING_WINDOW_DAYS - 1)
    window = visible[visible["signal_date"] >= rolling_cutoff]
    if window.empty:
        return []

    rows: list[dict] = []
    today = signal_date.normalize()
    for ts_code, grp in window.groupby("ts_code"):
        instrument = _qlib_instrument(ts_code)
        if not instrument:
            continue

        # Today-only rows for the binary flags. The flags use a STRICT
        # equality check on signal_date — the rolling window only
        # contributes to count + severity_max.
        today_rows = grp[grp["signal_date"].dt.normalize() == today]

        has_penalty = 0.0
        if not today_rows.empty and today_rows["severity"].isin(
            PENALTY_SEVERITIES,
        ).any():
            has_penalty = PENALTY_FLAG_VALUE

        has_strict_inquiry = 0.0
        if not today_rows.empty and bool(today_rows["is_strict_inquiry"].any()):
            has_strict_inquiry = STRICT_INQUIRY_FLAG_VALUE

        # Rolling-window severity max: highest SEVERITY_SCORE in the
        # 7-day window. 0 if all events were inquiries with severity
        # outside the score table.
        severity_max = max(
            (SEVERITY_SCORE.get(s, 0) for s in grp["severity"]),
            default=0,
        )

        # Count of distinct (event_date, source_url) pairs in the
        # window. The (date, url) dedup defends against the multi-
        # stock-press-release fan-out flagged in the SPIKE caveats.
        if "source_url" in grp.columns:
            dedup_key = grp["event_date"].astype(str) + "|" + grp["source_url"].astype(str)
        else:
            dedup_key = grp["event_date"].astype(str)
        penalty_count_7d = int(dedup_key.nunique())

        rows.append({
            "datetime": signal_date,
            "instrument": instrument,
            "has_penalty": float(has_penalty),
            "severity_max": float(severity_max),
            "has_strict_inquiry": float(has_strict_inquiry),
            "penalty_count_7d": float(penalty_count_7d),
        })
    return rows


def build_factors_range(
    start_date: str,
    end_date: str,
    input_dir: Path | None = None,
) -> pd.DataFrame:
    """Build a long-form per-(instrument, date) factor frame for the range.

    Like the XWLB and state_council factor builds, output is SPARSE —
    only (date, instrument) pairs with at least one event in the
    trailing window emit rows. The downstream FeatureMerger fills
    missing rows with 0.0 (the natural identity for these factors).
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
        rows.extend(build_factors_for_date(events, cur))
        cur += pd.Timedelta(days=1)
    if not rows:
        return pd.DataFrame(columns=("datetime", "instrument", *FACTOR_COLUMNS))
    return pd.DataFrame(rows)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _write_health_sidecar(
    *,
    output_path: Path,
    health_path: Path,
    df: pd.DataFrame,
    events: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> None:
    if events.empty:
        latest_event_date = ""
        n_events = 0
    else:
        latest_event_date = events["event_date"].max().strftime("%Y-%m-%d")
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


def publish_health(
    *,
    n_rows: int,
    n_events: int,
    latest_event_date: str,
    target_date: str,
    sparse_steady: bool = True,
) -> None:
    """Publish factor build health to scheduler.data_health.

    Regulator events are sparse-by-design — a window with 0 events
    is a legitimate steady state (the market does NOT have a regulator
    action every day), not a pipeline failure. We use sparse_steady=True
    by default, mirroring the 2026-06-09 PE-4 / state_council fix in
    scripts/build_policy_factors.py:publish_health.
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
    write_health(HEALTH_SOURCE_NAME, status, date=target_date)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Build regulator penalty factors from extracted events (SPIKE).",
    )
    parser.add_argument("--date", default=None, help="Single signal date YYYY-MM-DD.")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookback-days", type=int, default=30)
    args = parser.parse_args(argv)

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
    events = _load_events_from_dir(INPUT_DIR)
    if df.empty:
        logger.info(
            "regulator_penalty: 0 factor rows for [%s, %s], %d total events on disk.",
            start, end, int(len(events)),
        )
        publish_health(
            n_rows=0,
            n_events=int(len(events)),
            latest_event_date=(
                events["event_date"].max().strftime("%Y-%m-%d")
                if not events.empty else ""
            ),
            target_date=end,
        )
        return 0

    # Merge with existing parquet — drop overlapping dates first.
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
    _write_health_sidecar(
        output_path=OUTPUT_PATH,
        health_path=HEALTH_PATH,
        df=df,
        events=events,
        start_date=start,
        end_date=end,
    )
    publish_health(
        n_rows=int(len(df)),
        n_events=int(len(events)),
        latest_event_date=(
            events["event_date"].max().strftime("%Y-%m-%d")
            if not events.empty else ""
        ),
        target_date=end,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
