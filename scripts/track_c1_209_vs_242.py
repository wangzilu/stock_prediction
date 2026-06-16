"""C1 209 vs 242 paper-shadow tracker — dual snapshot + dual Sp20.

Sibling to ``track_b9_shadow_sp20.py``. Runs after smoke (which writes
the production xgb_209_chain_llm cache) and:
  1. Snapshots that production cache as the 209 leg.
  2. Spawns a second inference pass with PRODUCTION_MODEL_PROFILE=xgb_242,
     writes ``data/storage/lgb_xgb_242_predictions.json``.
  3. Snapshots the 242 cache as the 242 leg.
  4. Computes Sp20 (top20 - bot20 5d fwd return) for any snapshot whose
     5 trading days of forward closes have landed.
  5. Appends to running.csv with both legs side-by-side.

See ``docs/paper_trade_209_vs_242_design_20260616.md`` §3 for metrics.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA = PROJECT_ROOT / "data" / "storage"
SNAP_DIR = DATA / "c1_209_vs_242_snapshots"
PROD_CACHE = DATA / "lgb_latest_predictions.json"
CAND_CACHE = DATA / "lgb_xgb_242_predictions.json"
ROLLING_CSV = SNAP_DIR / "c1_running.csv"
SMOKE_SCRIPT = PROJECT_ROOT / "scripts" / "smoke_lgb_predict.py"
TOP_N = 20
FWD_HORIZON_DAYS = 5

PROFILES = {
    "xgb_209_chain_llm": PROD_CACHE,
    "xgb_242": CAND_CACHE,
}


def run_xgb_242_inference(date: str | None = None) -> bool:
    """Run smoke with PRODUCTION_MODEL_PROFILE=xgb_242 → CAND_CACHE."""
    env = os.environ.copy()
    env["PRODUCTION_MODEL_PROFILE"] = "xgb_242"
    env.setdefault("JOBLIB_MULTIPROCESSING", "0")
    cmd = [
        sys.executable, str(SMOKE_SCRIPT),
        "--output", str(CAND_CACHE),
    ]
    if date:
        cmd += ["--date", date]
    logger.info("xgb_242 inference → %s", CAND_CACHE.name)
    try:
        r = subprocess.run(cmd, env=env, timeout=1800, capture_output=True, text=True)
        if r.returncode != 0:
            logger.error("xgb_242 smoke rc=%d stderr=%s", r.returncode, r.stderr[-300:])
            return False
        logger.info("xgb_242 smoke OK")
        return True
    except subprocess.TimeoutExpired:
        logger.error("xgb_242 smoke timed out at 1800s")
        return False


def snapshot(profile: str, src: Path) -> str | None:
    if not src.exists():
        logger.warning("cache missing for %s: %s", profile, src)
        return None
    data = json.loads(src.read_text())
    date = data.get("latest_date")
    if not date:
        logger.warning("%s cache missing latest_date", profile)
        return None
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    target = SNAP_DIR / f"{profile}_snapshot_{date}.json"
    if target.exists():
        logger.info("%s snapshot for %s already exists", profile, date)
    else:
        shutil.copy(src, target)
        n = len(data.get("predictions", {}))
        logger.info("snapshot %s/%s (n=%d)", profile, date, n)
    return date


def _load_qlib_close(symbols: list[str], start_date: str, end_date: str) -> dict:
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
    for (sym, dt), row in df.iterrows():
        d = dt.strftime("%Y-%m-%d")
        out.setdefault(sym, {})[d] = float(row["$close"])
    return out


def compute_sp20(profile: str, snap_date: str) -> dict | None:
    snap = SNAP_DIR / f"{profile}_snapshot_{snap_date}.json"
    if not snap.exists():
        return None
    out = SNAP_DIR / f"{profile}_sp20_{snap_date}.json"
    if out.exists():
        return json.loads(out.read_text())
    data = json.loads(snap.read_text())
    preds = data.get("predictions", {})
    if not preds:
        return None
    items = sorted(preds.items(), key=lambda kv: kv[1], reverse=True)
    top = [c for c, _ in items[:TOP_N]]
    bot = [c for c, _ in items[-TOP_N:]]
    end_dt = datetime.strptime(snap_date, "%Y-%m-%d") + timedelta(days=FWD_HORIZON_DAYS * 2)
    closes = _load_qlib_close(top + bot, snap_date, end_dt.strftime("%Y-%m-%d"))
    if not closes:
        return None

    def _fwd(sym: str) -> float | None:
        rows = sorted(closes.get(sym, {}).items())
        if len(rows) < FWD_HORIZON_DAYS + 1:
            return None
        c0 = rows[0][1]
        cN = rows[min(FWD_HORIZON_DAYS, len(rows) - 1)][1]
        if c0 <= 0:
            return None
        return cN / c0 - 1.0

    top_r = [r for r in (_fwd(s) for s in top) if r is not None]
    bot_r = [r for r in (_fwd(s) for s in bot) if r is not None]
    if not top_r or not bot_r:
        return None
    top_m = sum(top_r) / len(top_r)
    bot_m = sum(bot_r) / len(bot_r)
    sp20_bps = (top_m - bot_m) * 10000.0
    result = {
        "profile": profile,
        "snapshot_date": snap_date,
        "horizon_days": FWD_HORIZON_DAYS,
        "top_n": TOP_N,
        "top_count": len(top_r),
        "bot_count": len(bot_r),
        "top_mean_return": top_m,
        "bot_mean_return": bot_m,
        "sp20_bps": sp20_bps,
    }
    out.write_text(json.dumps(result, indent=2))
    logger.info("[%s] %s Sp20=%.2f bps (top %.4f bot %.4f)",
                profile, snap_date, sp20_bps, top_m, bot_m)
    return result


def _top20_overlap(snap_date: str) -> int:
    snap_209 = SNAP_DIR / f"xgb_209_chain_llm_snapshot_{snap_date}.json"
    snap_242 = SNAP_DIR / f"xgb_242_snapshot_{snap_date}.json"
    if not (snap_209.exists() and snap_242.exists()):
        return -1
    d209 = json.loads(snap_209.read_text())
    d242 = json.loads(snap_242.read_text())
    t209 = {c for c, _ in sorted(d209.get("predictions", {}).items(),
                                  key=lambda kv: kv[1], reverse=True)[:TOP_N]}
    t242 = {c for c, _ in sorted(d242.get("predictions", {}).items(),
                                  key=lambda kv: kv[1], reverse=True)[:TOP_N]}
    return len(t209 & t242)


def main():
    snap_dates = {}
    # Step 1 — snapshot the production cache as 209 leg
    snap_dates["xgb_209_chain_llm"] = snapshot("xgb_209_chain_llm", PROD_CACHE)

    # Step 2 — run xgb_242 inference and snapshot
    if run_xgb_242_inference():
        snap_dates["xgb_242"] = snapshot("xgb_242", CAND_CACHE)
    else:
        logger.error("xgb_242 inference failed — skipping 242 snapshot for this run")

    # Step 3 — compute Sp20 for both legs across any snapshot eligible
    for profile in PROFILES:
        for snap in sorted(SNAP_DIR.glob(f"{profile}_snapshot_*.json")):
            date = snap.stem.split("_snapshot_")[-1]
            if (SNAP_DIR / f"{profile}_sp20_{date}.json").exists():
                continue
            compute_sp20(profile, date)

    # Step 4 — append rolling csv with both legs and overlap
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not ROLLING_CSV.exists()
    with open(ROLLING_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "snapshot_date", "top20_overlap",
                "sp20_209_bps", "sp20_242_bps", "delta_242_minus_209_bps",
            ])
        # Write a row for any snap_date for which BOTH legs have a sp20 file
        dates_both = sorted(
            {p.stem.split("_sp20_")[-1] for p in SNAP_DIR.glob("xgb_209_chain_llm_sp20_*.json")}
            & {p.stem.split("_sp20_")[-1] for p in SNAP_DIR.glob("xgb_242_sp20_*.json")}
        )
        for d in dates_both:
            sp_209 = json.loads((SNAP_DIR / f"xgb_209_chain_llm_sp20_{d}.json").read_text())["sp20_bps"]
            sp_242 = json.loads((SNAP_DIR / f"xgb_242_sp20_{d}.json").read_text())["sp20_bps"]
            ov = _top20_overlap(d)
            w.writerow([d, ov, f"{sp_209:.2f}", f"{sp_242:.2f}", f"{sp_242 - sp_209:.2f}"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
