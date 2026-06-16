"""B.9 shadow Sp20 tracker — snapshot daily LGB cache + compute realized Sp20.

After each post-close smoke writes ``lgb_latest_predictions.json``, run this
to (a) snapshot the cache so we can attribute picks to a specific date later
and (b) compute realized Sp20 once 5d forward returns are available.

Outputs:
    data/storage/b9_shadow_snapshots/cache_snapshot_<date>.json     (raw cache copy)
    data/storage/b9_shadow_snapshots/sp20_realized_<date>.json      (top20-bot20 5d return)
    data/storage/b9_shadow_snapshots/sp20_running.csv               (rolling window log)

Sp20 definition (per b9_shadow_monitoring_plan_20260613.md):
    Sp20 = mean(top-20 5d forward return) - mean(bottom-20 5d forward return)

Usage:
    python scripts/track_b9_shadow_sp20.py            # snapshot today, compute backfill
"""
from __future__ import annotations

import csv
import json
import logging
import shutil
import sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_PATH = PROJECT_ROOT / "data" / "storage" / "lgb_latest_predictions.json"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "storage" / "b9_shadow_snapshots"
ROLLING_CSV = SNAPSHOT_DIR / "sp20_running.csv"
TOP_N = 20
FWD_HORIZON_DAYS = 5  # 5 trading days


def snapshot_cache() -> str | None:
    """Copy today's cache to snapshot dir keyed by latest_date."""
    if not CACHE_PATH.exists():
        logger.error("cache not found: %s", CACHE_PATH)
        return None
    data = json.loads(CACHE_PATH.read_text())
    date = data.get("latest_date")
    if not date:
        logger.error("cache missing latest_date")
        return None
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    target = SNAPSHOT_DIR / f"cache_snapshot_{date}.json"
    if target.exists():
        logger.info("snapshot already exists for %s (skip): %s", date, target)
    else:
        shutil.copy(CACHE_PATH, target)
        n_preds = len(data.get("predictions", {}))
        model = (data.get("model_path") or "").split("/")[-1]
        logger.info("saved snapshot for %s: %s (model=%s n=%d)", date, target, model, n_preds)
    return date


def _load_qlib_close(symbols: list[str], start_date: str, end_date: str) -> dict:
    """Load daily close prices for symbols from qlib parquet.

    Returns dict[symbol -> dict[date_str -> close]].
    """
    try:
        import qlib
        from qlib.data import D
        from config.settings import QLIB_PROVIDER_URI
        qlib.init(provider_uri=QLIB_PROVIDER_URI, region="cn")
        df = D.features(symbols, ["$close"], start_time=start_date, end_time=end_date, freq="day")
    except Exception as e:
        logger.error("qlib load failed: %s", e)
        return {}

    out: dict[str, dict[str, float]] = {}
    if df is None or df.empty:
        return out
    # MultiIndex (instrument, datetime)
    for (sym, dt), row in df.iterrows():
        d = dt.strftime("%Y-%m-%d")
        out.setdefault(sym, {})[d] = float(row["$close"])
    return out


def compute_sp20_for_date(snapshot_date: str) -> dict | None:
    """Compute realized Sp20 for a snapshot — needs 5 trading days of post-snapshot data."""
    snap_path = SNAPSHOT_DIR / f"cache_snapshot_{snapshot_date}.json"
    if not snap_path.exists():
        logger.warning("snapshot missing for %s", snapshot_date)
        return None
    out_path = SNAPSHOT_DIR / f"sp20_realized_{snapshot_date}.json"
    if out_path.exists():
        logger.info("sp20 already computed for %s, skip", snapshot_date)
        return json.loads(out_path.read_text())

    data = json.loads(snap_path.read_text())
    preds = data.get("predictions", {})
    if not preds:
        logger.warning("snapshot empty predictions for %s", snapshot_date)
        return None

    # Top-20 / Bottom-20
    items = sorted(preds.items(), key=lambda kv: kv[1], reverse=True)
    top = [code for code, _ in items[:TOP_N]]
    bot = [code for code, _ in items[-TOP_N:]]
    all_syms = top + bot

    # Forward window: snapshot_date close → snapshot_date+5tdays close
    start = snapshot_date
    end_dt = datetime.strptime(snapshot_date, "%Y-%m-%d") + timedelta(days=FWD_HORIZON_DAYS * 2)
    end = end_dt.strftime("%Y-%m-%d")
    closes = _load_qlib_close(all_syms, start, end)
    if not closes:
        logger.warning("no closes loaded for %s", snapshot_date)
        return None

    # Compute fwd return per symbol — first business date after snapshot,
    # and the 5th trading day after that.
    def _fwd_return(sym: str) -> float | None:
        rows = sorted(closes.get(sym, {}).items())
        if len(rows) < FWD_HORIZON_DAYS + 1:
            return None
        # rows[0] is the snapshot date close, rows[FWD_HORIZON_DAYS] is 5td later
        # Skip the snapshot date itself; use next bday open=close approx
        c0 = rows[0][1]
        cN = rows[min(FWD_HORIZON_DAYS, len(rows) - 1)][1]
        if c0 <= 0:
            return None
        return (cN / c0) - 1.0

    top_rets = [_fwd_return(s) for s in top]
    bot_rets = [_fwd_return(s) for s in bot]
    top_rets = [r for r in top_rets if r is not None]
    bot_rets = [r for r in bot_rets if r is not None]
    if not top_rets or not bot_rets:
        logger.warning("insufficient fwd returns for %s: top=%d bot=%d",
                       snapshot_date, len(top_rets), len(bot_rets))
        return None

    top_mean = sum(top_rets) / len(top_rets)
    bot_mean = sum(bot_rets) / len(bot_rets)
    sp20_bps = (top_mean - bot_mean) * 10000.0

    result = {
        "snapshot_date": snapshot_date,
        "horizon_days": FWD_HORIZON_DAYS,
        "top_n": TOP_N,
        "top_count": len(top_rets),
        "bot_count": len(bot_rets),
        "top_mean_return": top_mean,
        "bot_mean_return": bot_mean,
        "sp20_bps": sp20_bps,
        "model_path": data.get("model_path", ""),
        "computed_at": end,
    }
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Sp20 for %s: %.2f bps (top mean %.4f, bot mean %.4f)",
                snapshot_date, sp20_bps, top_mean, bot_mean)

    # Append to rolling csv
    write_header = not ROLLING_CSV.exists()
    with open(ROLLING_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["snapshot_date", "top_mean_return", "bot_mean_return", "sp20_bps", "model"])
        w.writerow([
            snapshot_date, f"{top_mean:.6f}", f"{bot_mean:.6f}",
            f"{sp20_bps:.2f}", (data.get("model_path") or "").split("/")[-1],
        ])
    return result


def main():
    today_date = snapshot_cache()
    if not today_date:
        return 1

    # Try to compute Sp20 for any past snapshot that has 5+ trading days of close data.
    for snap in sorted(SNAPSHOT_DIR.glob("cache_snapshot_*.json")):
        date = snap.stem.replace("cache_snapshot_", "")
        verdict_file = SNAPSHOT_DIR / f"sp20_realized_{date}.json"
        if verdict_file.exists():
            continue
        compute_sp20_for_date(date)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
