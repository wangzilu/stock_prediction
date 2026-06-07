"""Build per-stock sentiment factor parquet from the cron-collected
sentiment JSONL series.

Closes the task #164 follow-up that the project lead's 2026-06-06
critique flagged: ``collect_sentiment_daily.py`` writes
``data/storage/sentiment/<YYYY-MM-DD>.jsonl`` but no per-stock factor
build ever existed, so the data sat on disk unused.

Sources in the JSONL:
  - ``xueqiu_hot`` — Xueqiu hot stocks list (per-stock, with stock_code)
  - ``xueqiu`` — Xueqiu trending discussions (per-stock, with stock_code)
  - ``ths_hot`` — Tongdaxin hot stocks (per-stock, with stock_code)
  - ``ths_concept`` — Tongdaxin hot concepts (per-concept, NO stock_code)

Per-stock factors emitted (per date):
  - ``sentiment_hot_rank`` — best (lowest) rank across xueqiu_hot / ths_hot
  - ``sentiment_n_sources`` — count of distinct sources mentioning the stock
  - ``sentiment_heat_score`` — sum of (51 − rank) over per-stock sources,
    clipped at zero; bigger = hotter

Concept rows (``ths_concept``) are NOT mapped to stocks — that's a
follow-up (would need an A-share concept→stock map). They are ignored
here so we don't fabricate signal.

**Data quality caveat (2026-06-07)**: as of 2026-05-30 the
``xueqiu_hot`` HTTP path returns 0 items (API changed); the daily
JSONL files since then contain only ``ths_concept`` rows that have no
``stock_code``. So this builder produces near-empty factor rows for
the recent period. The xueqiu API fix is filed as a follow-up.

Output: ``data/storage/sentiment_factors.parquet``, indexed
(datetime, instrument) with the three factor columns above.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SENTIMENT_DIR = DATA_DIR / "sentiment"
OUTPUT_PATH = DATA_DIR / "sentiment_factors.parquet"
HEALTH_SOURCE_NAME = "sentiment_factors"

PER_STOCK_SOURCES = ("xueqiu_hot", "xueqiu", "ths_hot")
MAX_RANK = 50  # heat_score uses MAX_RANK + 1 − rank so 1 → 50 points


def _stock_code_to_qlib_code(stock_code: str) -> str:
    """Convert 6-digit stock_code to qlib instrument key (sh600519)."""
    sc = (stock_code or "").strip().upper()
    if len(sc) != 6 or not sc.isdigit():
        return ""
    prefix = "sh" if sc.startswith("6") else "sz"
    return f"{prefix}{sc}"


def load_rows(date_path: Path) -> list[dict]:
    rows = []
    for ln in date_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return rows


def per_day_factors(rows: list[dict]) -> pd.DataFrame:
    """Aggregate one day's rows into per-stock factor rows."""
    per_stock: dict[str, dict] = {}
    for r in rows:
        src = r.get("source", "")
        if src not in PER_STOCK_SOURCES:
            continue
        qlib_code = _stock_code_to_qlib_code(r.get("stock_code", ""))
        if not qlib_code:
            continue
        rank = int(r.get("rank") or MAX_RANK + 1)
        bucket = per_stock.setdefault(qlib_code, {
            "best_rank": rank, "n_sources": 0, "heat_sum": 0
        })
        bucket["best_rank"] = min(bucket["best_rank"], rank)
        bucket["n_sources"] += 1
        bucket["heat_sum"] += max(0, MAX_RANK + 1 - rank)
    out = []
    for code, b in per_stock.items():
        out.append({
            "instrument": code,
            "sentiment_hot_rank": float(b["best_rank"]),
            "sentiment_n_sources": int(b["n_sources"]),
            "sentiment_heat_score": float(b["heat_sum"]),
        })
    return pd.DataFrame(out)


def build_factors(
    sentiment_dir: Path = SENTIMENT_DIR,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    if not sentiment_dir.exists():
        raise FileNotFoundError(
            f"sentiment dir missing: {sentiment_dir}. "
            f"Has collect_sentiment_daily.py run yet?"
        )
    daily_frames: list[pd.DataFrame] = []
    n_files = 0
    n_with_signal = 0
    for path in sorted(sentiment_dir.glob("*.jsonl")):
        n_files += 1
        date = path.stem
        try:
            dt = pd.to_datetime(date)
        except (ValueError, TypeError):
            logger.warning("Skipping non-date file: %s", path)
            continue
        rows = load_rows(path)
        df = per_day_factors(rows)
        if df.empty:
            continue
        df["datetime"] = dt
        daily_frames.append(df)
        n_with_signal += 1
    if not daily_frames:
        logger.warning(
            "No per-stock sentiment rows found across %d files. "
            "Xueqiu API may be broken — check collect_sentiment_daily.py.",
            n_files,
        )
        empty = pd.DataFrame(columns=[
            "sentiment_hot_rank", "sentiment_n_sources", "sentiment_heat_score"
        ])
        empty.index = pd.MultiIndex.from_tuples([], names=["datetime", "instrument"])
        return empty
    combined = pd.concat(daily_frames, ignore_index=True)
    combined = combined.set_index(["datetime", "instrument"]).sort_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp.parquet")
    combined.to_parquet(tmp, compression="snappy")
    tmp.replace(output_path)
    logger.info(
        "Wrote %s: %d rows × %d cols (signal-bearing days: %d / %d files)",
        output_path, len(combined), combined.shape[1],
        n_with_signal, n_files,
    )
    # Health write.
    try:
        from scheduler.data_health import HealthStatus, write_health
        # success requires actually-non-empty signal AND at least one
        # day of real per-stock data (not just ths_concept fallback).
        write_health(HEALTH_SOURCE_NAME, HealthStatus(
            success=len(combined) > 0,
            n_items=len(combined),
            latest_date=combined.index.get_level_values("datetime").max()
                .strftime("%Y-%m-%d") if not combined.empty else "",
            error_type="" if not combined.empty else "no_per_stock_data",
            network_profile="none",
            extra={"signal_days": n_with_signal, "total_files": n_files},
        ))
    except Exception as e:
        logger.warning("write_health failed: %s", e)
    return combined


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUTPUT_PATH))
    args = ap.parse_args()
    build_factors(output_path=Path(args.out))


if __name__ == "__main__":
    main()
