"""Build per-stock, per-date factors from extracted IRM Q&A.

SPIKE STATUS (2026-06-09): scaffold. Builds the per-(datetime, qlib_code)
parquet so the LLM factor reviewers can score the schema against existing
LLM event factors. NOT yet integrated into the daily run.

Input
-----
``data/storage/irm_qa_extracted/<YYYY-MM-DD>.jsonl`` — one row per Q&A,
fields per ``factors/irm_qa_extractor.IRMQAExtractor.extract_from_file``.

Output
------
``data/storage/irm_qa_factors.parquet`` (keyed by ``(datetime, instrument)``
where ``instrument`` is lowercase ``qlib_code`` and ``datetime`` is the
SIGNAL DATE = answer_date + 1 business day).

Per-stock per-date factors
--------------------------
* ``irm_qa_count_5d``                 — total Q&A in trailing 5 calendar days
* ``irm_qa_substantive_count_5d``     — only substantive=True
* ``irm_qa_mean_info_value_5d``       — mean information_value_score (1-5)
* ``irm_qa_net_forward_signal_5d``    — sum(forward_signal_direction * confidence)
                                         over the 5d window
* ``irm_qa_guidance_change_count_5d`` — count of contains_guidance_change=True
* ``irm_qa_dodge_rate_5d``            — (count - substantive_count) / count
                                         (= "IR template share"; high = stonewall)
* ``irm_qa_topic_concentration_5d``   — Herfindahl over topic distribution
                                         (= "all questions about ONE thing")

PIT discipline
--------------
Signal date = ``answer_date + 1 BDay``. Mirrors the event pipeline rule
that any same-day post-15:00 reply only enters the signal stream on the
next trading day. Since IRM answers do NOT carry hour-of-day in the
collector's current schema (we only normalize the timestamp to ISO but
the BDay shift treats them uniformly as "today", giving the
conservative-late convention), the shift is unconditional. If we later
preserve hour-of-day from ``answer_time`` we can branch on it.

Aggregation window
------------------
5 calendar days. Q&A is a low-frequency channel — a single stock
typically receives 0-2 substantive answers per day. A wider window
(20-30d) was considered but rejected:

* IRM Q&A is REACTIVE — a question stays open ~3 days before answer,
  and the "freshness" of the signal half-lives fast.
* Wider window dilutes the dodge_rate signal (a stock that suddenly
  starts giving template answers should light up the factor within a
  week).

A second roll-up at 20d may be added later for the "topic attention"
factor; deferred to round 2.
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
from pandas.tseries.offsets import BDay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)

INPUT_DIR = DATA_DIR / "irm_qa_extracted"
OUTPUT_PATH = DATA_DIR / "irm_qa_factors.parquet"
HEALTH_PATH = DATA_DIR / "irm_qa_factors.health.json"
HEALTH_SOURCE_NAME = "irm_qa_factors"

WINDOW_DAYS = 5

FACTOR_COLUMNS = (
    "irm_qa_count_5d",
    "irm_qa_substantive_count_5d",
    "irm_qa_mean_info_value_5d",
    "irm_qa_net_forward_signal_5d",
    "irm_qa_guidance_change_count_5d",
    "irm_qa_dodge_rate_5d",
    "irm_qa_topic_concentration_5d",
)


def _load_extracted_qa(input_dir: Path) -> pd.DataFrame:
    """Load every per-day JSONL from ``input_dir`` into one DataFrame.

    Schema columns we require downstream: ``qlib_code``, ``answer_date``,
    ``is_substantive``, ``information_value_score``,
    ``forward_signal_direction``, ``confidence``,
    ``contains_guidance_change``, ``question_topic``.

    Missing / unreadable files are skipped silently.
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
    df["answer_date"] = pd.to_datetime(df.get("answer_date"), errors="coerce")
    df = df.dropna(subset=["answer_date"]).reset_index(drop=True)
    # qlib_code must be lowercase to match base cache. We don't trust
    # the collector to have already lowercased — paranoia per the
    # 2026-06-08 case-bug note (factors/feature_cache_utils.py).
    df["qlib_code"] = df["qlib_code"].astype(str).str.lower().str.strip()
    df = df[df["qlib_code"].astype(bool)].reset_index(drop=True)
    df["information_value_score"] = pd.to_numeric(
        df.get("information_value_score"), errors="coerce",
    ).fillna(1).clip(1, 5)
    df["forward_signal_direction"] = pd.to_numeric(
        df.get("forward_signal_direction"), errors="coerce",
    ).fillna(0).clip(-1, 1)
    df["confidence"] = pd.to_numeric(
        df.get("confidence"), errors="coerce",
    ).fillna(0.5).clip(0.0, 1.0)
    for col in ("is_substantive", "contains_guidance_change"):
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
        else:
            df[col] = False
    df["question_topic"] = df.get("question_topic", "other").fillna("other")
    return df


def _signal_date_from_answer(answer_date: pd.Timestamp) -> pd.Timestamp:
    """answer_date + 1 BDay. PIT: post-close convention applies uniformly."""
    return (answer_date + BDay(1)).normalize()


def build_factors_for_date(
    qa_df: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute per-(qlib_code) factor rows for ``signal_date``.

    Returns a DataFrame with columns ``[qlib_code, *FACTOR_COLUMNS]``.
    Stocks with no Q&A in the trailing window simply do not appear —
    the output parquet is SPARSE (only stocks with events have rows).
    Downstream FeatureMerger fills missing keys with 0.
    """
    if qa_df.empty:
        return pd.DataFrame(columns=["qlib_code", *FACTOR_COLUMNS])

    # signal_date - 1 BDay = latest answer_date that can be seen.
    answer_cutoff_latest = (signal_date - BDay(1)).normalize()
    # 5 calendar day trailing window on answer_date.
    answer_cutoff_earliest = (
        answer_cutoff_latest - pd.Timedelta(days=WINDOW_DAYS - 1)
    ).normalize()

    window = qa_df[
        (qa_df["answer_date"].dt.normalize() >= answer_cutoff_earliest)
        & (qa_df["answer_date"].dt.normalize() <= answer_cutoff_latest)
    ]
    if window.empty:
        return pd.DataFrame(columns=["qlib_code", *FACTOR_COLUMNS])

    rows: list[dict] = []
    for code, g in window.groupby("qlib_code"):
        n = int(len(g))
        n_subst = int(g["is_substantive"].sum())
        mean_iv = float(g["information_value_score"].mean())
        # confidence-weighted forward direction. Multiplying by confidence
        # keeps the LLM's own self-uncertainty out of the cross-section
        # without an external calibration step.
        net_fwd = float(
            (g["forward_signal_direction"] * g["confidence"]).sum()
        )
        n_guidance = int(g["contains_guidance_change"].sum())
        dodge = float((n - n_subst) / n) if n > 0 else 0.0
        # Herfindahl on topic share = sum(p_i^2). Close to 1 = all Qs on
        # one topic ("everyone's asking the same thing"); close to 1/k =
        # uniform.
        topic_counts = g["question_topic"].value_counts(normalize=True)
        hhi = float((topic_counts ** 2).sum()) if not topic_counts.empty else 0.0
        rows.append({
            "qlib_code": code,
            "irm_qa_count_5d": float(n),
            "irm_qa_substantive_count_5d": float(n_subst),
            "irm_qa_mean_info_value_5d": mean_iv,
            "irm_qa_net_forward_signal_5d": net_fwd,
            "irm_qa_guidance_change_count_5d": float(n_guidance),
            "irm_qa_dodge_rate_5d": dodge,
            "irm_qa_topic_concentration_5d": hhi,
        })
    return pd.DataFrame(rows)


def build_factors_range(
    start_date: str,
    end_date: str,
    input_dir: Path | None = None,
) -> pd.DataFrame:
    """Build a long-form factor DataFrame for every signal_date in [start, end].

    Returns columns ``[datetime, instrument, *FACTOR_COLUMNS]`` where
    ``datetime`` = signal_date and ``instrument`` = lowercase qlib_code.
    """
    qa = _load_extracted_qa(input_dir or INPUT_DIR)
    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)
    if e < s:
        raise ValueError(f"end ({end_date}) must be >= start ({start_date})")

    out: list[pd.DataFrame] = []
    cur = s
    while cur <= e:
        per_stock = build_factors_for_date(qa, cur)
        if not per_stock.empty:
            per_stock = per_stock.rename(columns={"qlib_code": "instrument"})
            per_stock.insert(0, "datetime", cur)
            out.append(per_stock)
        cur += pd.Timedelta(days=1)

    if not out:
        return pd.DataFrame(columns=["datetime", "instrument", *FACTOR_COLUMNS])
    return pd.concat(out, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────
# IO helpers (parquet atomic write + health sidecar)
# ─────────────────────────────────────────────────────────────────────
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
    qa: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> None:
    if qa.empty:
        latest_answer = ""
        n_qa = 0
    else:
        latest_answer = qa["answer_date"].max().strftime("%Y-%m-%d")
        n_qa = int(len(qa))
    sidecar = {
        "source": HEALTH_SOURCE_NAME,
        "start_date": start_date,
        "end_date": end_date,
        "n_qa_used": n_qa,
        "latest_answer_date": latest_answer,
        "n_factor_rows": int(len(df)),
        "n_unique_instruments": int(df["instrument"].nunique()) if not df.empty else 0,
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
    n_qa: int,
    latest_qa_date: str,
    target_date: str,
) -> None:
    try:
        from scheduler.data_health import HealthStatus, write_health
    except Exception as e:  # pragma: no cover
        logger.warning("Cannot import scheduler.data_health (%s)", e)
        return
    success = n_rows > 0 and n_qa > 0
    status = HealthStatus(
        success=success,
        n_items=n_rows,
        latest_date=latest_qa_date or target_date,
        error_type=(
            "" if success
            else ("no_qa" if n_rows > 0 and n_qa == 0 else "no_factor_rows")
        ),
        network_profile="ashare",
        extra={"n_qa_used": n_qa, "latest_qa_date": latest_qa_date},
    )
    write_health(HEALTH_SOURCE_NAME, status, date=target_date)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(
        description="Build per-stock IRM Q&A factors from extracted Q&A JSONL."
    )
    p.add_argument("--date", default=None,
                   help="Single signal date YYYY-MM-DD (default: today).")
    p.add_argument("--start", default=None, help="Backfill start.")
    p.add_argument("--end", default=None, help="Backfill end.")
    p.add_argument("--lookback-days", type=int, default=30,
                   help="When neither --start nor --end is provided, build for the last N days.")
    args = p.parse_args(argv)

    today = datetime.now().strftime("%Y-%m-%d")
    if args.start or args.end:
        start = args.start or args.end or today
        end = args.end or args.start or today
    elif args.date:
        start = end = args.date
    else:
        end = today
        start = (datetime.strptime(end, "%Y-%m-%d")
                 - timedelta(days=args.lookback_days)).strftime("%Y-%m-%d")

    df = build_factors_range(start_date=start, end_date=end)
    qa = _load_extracted_qa(INPUT_DIR)

    if df.empty:
        logger.warning("No factor rows built for [%s, %s]", start, end)
        publish_health(
            n_rows=0,
            n_qa=int(len(qa)) if not qa.empty else 0,
            latest_qa_date=(
                qa["answer_date"].max().strftime("%Y-%m-%d")
                if not qa.empty else ""
            ),
            target_date=end,
        )
        return 1

    # Merge with existing parquet — drop overlapping signal dates first.
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
        "Wrote %d factor rows (%d unique instruments) for [%s, %s] → %s",
        len(df), df["instrument"].nunique(), start, end, OUTPUT_PATH,
    )

    _write_health_sidecar(
        output_path=OUTPUT_PATH,
        health_path=HEALTH_PATH,
        df=df,
        qa=qa,
        start_date=start,
        end_date=end,
    )
    publish_health(
        n_rows=len(df),
        n_qa=int(len(qa)),
        latest_qa_date=qa["answer_date"].max().strftime("%Y-%m-%d") if not qa.empty else "",
        target_date=end,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
