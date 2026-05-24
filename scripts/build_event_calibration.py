"""Build event calibration table — replace LLM impact with historical return.

CX principle: LLM extracts facts, historical data calibrates weights.

For each event_type × direction bucket:
  - Compute average actual next-day return
  - Shrink low-sample buckets toward 0
  - Output calibrated_alpha per event

Usage:
    python scripts/build_event_calibration.py
"""
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"


def main():
    from config.qlib_runtime import init_qlib
    from qlib.data import D
    from scipy import stats

    init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

    # Load all events
    events_dir = DATA_DIR / "llm_events"
    all_events = []
    for f in sorted(events_dir.glob("*.jsonl")):
        date = f.stem
        for line in open(f):
            e = json.loads(line)
            e["event_date"] = date
            all_events.append(e)

    logger.info(f"Total events: {len(all_events)}")

    # Build qlib map
    qlib_map = {}
    for e in all_events:
        c = e.get("stock_code", "")
        if c.startswith("6"):
            qlib_map[c] = f"sh{c}"
        elif c.startswith(("0", "3")):
            qlib_map[c] = f"sz{c}"

    # Load returns
    ret = D.features(list(set(qlib_map.values())),
                     ["Ref($close, -1) / $close - 1"],
                     start_time="2026-04-25", end_time="2026-05-22")
    ret.columns = ["ret"]
    ret_lookup = {}
    for idx, row in ret.iterrows():
        if np.isfinite(row["ret"]):
            ret_lookup[(idx[0], idx[1].strftime("%Y-%m-%d"))] = float(row["ret"])

    # Match events with returns
    matched = []
    for e in all_events:
        code = e.get("stock_code", "")
        date = e.get("event_date", "")
        qlib = qlib_map.get(code)
        if qlib and (qlib, date) in ret_lookup:
            matched.append({
                "event_type": e.get("event_type", "other"),
                "direction": 1 if e.get("impact_1d", 0) > 0 else (-1 if e.get("impact_1d", 0) < 0 else 0),
                "source": e.get("source", "unknown"),
                "actual_ret": ret_lookup[(qlib, date)],
            })

    logger.info(f"Matched: {len(matched)} events with returns")

    # Build calibration table: event_type × direction → avg_return
    buckets = defaultdict(list)
    for m in matched:
        key = (m["event_type"], m["direction"])
        buckets[key].append(m["actual_ret"])

    # Shrinkage: low sample → shrink toward 0
    MIN_SAMPLES = 30
    SHRINK_SAMPLES = 100  # full weight at 100+ samples

    calibration = {}
    logger.info(f"\n{'type × direction':<35} {'N':>6} {'raw_avg':>8} {'shrunk':>8} {'use':>5}")
    logger.info("-" * 70)

    for (etype, direction), returns in sorted(buckets.items(), key=lambda x: -len(x[1])):
        n = len(returns)
        raw_avg = np.mean(returns)

        # Bayesian shrinkage: weight = min(n / SHRINK_SAMPLES, 1)
        shrink_weight = min(n / SHRINK_SAMPLES, 1.0)
        shrunk = raw_avg * shrink_weight

        use = n >= MIN_SAMPLES
        calibration[(etype, direction)] = {
            "n": n,
            "raw_avg_return": round(raw_avg, 6),
            "shrunk_return": round(shrunk, 6),
            "shrink_weight": round(shrink_weight, 3),
            "use": use,
        }

        dir_str = {1: "+", -1: "-", 0: "0"}.get(direction, "?")
        logger.info(f"  {etype} ({dir_str}){'':<15} {n:>6} {raw_avg*100:>+7.3f}% {shrunk*100:>+7.3f}% {'✅' if use else '❌'}")

    # Save calibration table
    output = {
        "calibrated_at": pd.Timestamp.now().isoformat(),
        "n_events": len(matched),
        "n_buckets": len(calibration),
        "min_samples": MIN_SAMPLES,
        "shrink_samples": SHRINK_SAMPLES,
        "buckets": {f"{k[0]}_{k[1]}": v for k, v in calibration.items()},
    }

    out_path = DATA_DIR / "llm_event_calibration.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved to {out_path}")

    # Summary: which event types have real signal
    logger.info(f"\n=== Signal Summary ===")
    for (etype, direction), cal in sorted(calibration.items(),
                                           key=lambda x: abs(x[1]["shrunk_return"]), reverse=True):
        if not cal["use"]:
            continue
        dir_str = {1: "positive", -1: "negative", 0: "neutral"}.get(direction, "?")
        sr = cal["shrunk_return"]
        if abs(sr) > 0.001:
            logger.info(f"  {etype} ({dir_str}): calibrated alpha = {sr*100:+.3f}% (N={cal['n']})")


if __name__ == "__main__":
    main()
