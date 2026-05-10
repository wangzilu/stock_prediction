"""Factor decay monitoring: track daily IC and warn on degradation.

Reads lgb_eval_latest.json history and monitors IC trends.
Warns when IC is negative for 3+ consecutive days.

Usage:
    python scripts/monitor_factor_decay.py
    python scripts/monitor_factor_decay.py --json
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
EVAL_HISTORY_PATH = DATA_DIR / "lgb_eval_history.json"
DECAY_PATH = DATA_DIR / "factor_decay_status.json"


def monitor(warn_consecutive_negative: int = 3) -> dict:
    """Monitor factor decay from evaluation history.

    Returns:
        Status dict with health, warnings, and trend data.
    """
    if not EVAL_HISTORY_PATH.exists():
        return {"status": "no_data", "message": "No evaluation history found"}

    try:
        history = json.loads(EVAL_HISTORY_PATH.read_text())
    except Exception:
        return {"status": "error", "message": "Cannot read eval history"}

    if len(history) < 2:
        return {"status": "insufficient", "message": f"Only {len(history)} eval records, need 2+"}

    # Extract IC time series
    ic_series = []
    rank_ic_series = []
    spread_series = []

    for entry in history:
        m = entry.get("metrics", {})
        ic_series.append(m.get("ic_mean", 0))
        rank_ic_series.append(m.get("rank_ic_mean", 0))
        spread_series.append(m.get("top20_bot20_spread", 0))

    # Check for consecutive negative IC
    warnings = []
    status = "healthy"

    # IC trend
    recent_ic = ic_series[-warn_consecutive_negative:]
    if all(ic <= 0 for ic in recent_ic) and len(recent_ic) >= warn_consecutive_negative:
        warnings.append(f"IC negative for {len(recent_ic)} consecutive evaluations")
        status = "degraded"

    # RankIC trend
    recent_ric = rank_ic_series[-warn_consecutive_negative:]
    if all(ric <= 0 for ric in recent_ric) and len(recent_ric) >= warn_consecutive_negative:
        warnings.append(f"RankIC negative for {len(recent_ric)} consecutive evaluations")
        if status != "degraded":
            status = "warning"

    # Spread trend
    recent_spread = spread_series[-warn_consecutive_negative:]
    if all(s <= 0 for s in recent_spread) and len(recent_spread) >= warn_consecutive_negative:
        warnings.append(f"Top20 spread negative for {len(recent_spread)} consecutive evaluations")
        status = "degraded"

    # IC declining trend (last 5 vs previous 5)
    if len(ic_series) >= 10:
        recent_avg = np.mean(ic_series[-5:])
        previous_avg = np.mean(ic_series[-10:-5])
        if recent_avg < previous_avg * 0.5 and previous_avg > 0:
            warnings.append(f"IC declined >50%: {previous_avg:.4f} → {recent_avg:.4f}")
            if status == "healthy":
                status = "warning"

    result = {
        "status": status,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "n_evaluations": len(history),
        "latest_ic": round(ic_series[-1], 6) if ic_series else 0,
        "latest_rank_ic": round(rank_ic_series[-1], 6) if rank_ic_series else 0,
        "latest_spread": round(spread_series[-1], 6) if spread_series else 0,
        "ic_trend_5d": [round(x, 6) for x in ic_series[-5:]],
        "warnings": warnings,
        "recommendation": _recommendation(status),
    }
    return result


def _recommendation(status: str) -> str:
    if status == "healthy":
        return "模型信号正常，继续使用当前模型"
    elif status == "warning":
        return "模型信号减弱，建议降低推荐仓位权重"
    elif status == "degraded":
        return "模型信号严重衰退，建议暂停模型推荐，使用保守策略"
    return "数据不足，无法判断"


def main():
    parser = argparse.ArgumentParser(description="Factor decay monitor")
    parser.add_argument("--warn-days", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = monitor(warn_consecutive_negative=args.warn_days)

    # Save
    DECAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DECAY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    os.replace(tmp, DECAY_PATH)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Factor Health: {result['status'].upper()}")
        print(f"  IC: {result['latest_ic']:.4f}  RankIC: {result['latest_rank_ic']:.4f}  Spread: {result['latest_spread']*100:.3f}%")
        print(f"  IC trend (last 5): {result['ic_trend_5d']}")
        if result["warnings"]:
            for w in result["warnings"]:
                print(f"  ⚠️  {w}")
        print(f"  建议: {result['recommendation']}")

    sys.exit(0 if result["status"] != "degraded" else 1)


if __name__ == "__main__":
    main()
