"""Audit factor merge: verify join alignment, coverage, extremes, and spot-check.

Must pass before any factor enters model training.

Usage:
    python scripts/audit_factor_merge.py
    python scripts/audit_factor_merge.py --json
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config.qlib_runtime import init_qlib
from config.settings import PREDICTION_HORIZON_DAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
AUDIT_PATH = DATA_DIR / "factor_merge_audit.json"

CUSTOM_EXPRS = [
    "($close - Min($close, 60)) / (Max($close, 60) - Min($close, 60) + 1e-8)",
    "1.0 / If(Abs($pe) > 0.01, $pe, 1.0)",
    "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20) + 1e-8)",
    "$pb / Ref($pb, 5) - 1",
]
CUSTOM_NAMES = ["price_pos60", "ep", "price_pos20", "pb_mom5"]


def audit(test_days: int = 30, universe: str = "csi300", n_spot_check: int = 10) -> dict:
    from qlib.utils import init_instance_by_config
    from qlib.data import D

    init_qlib(QLIB_DATA)

    today = datetime.now()
    start = (today - timedelta(days=test_days)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    logger.info(f"Auditing factor merge: {start}~{end}, universe={universe}")

    # 1. Load Alpha158
    handler_config = {
        "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
        "kwargs": {"start_time": start, "end_time": end, "instruments": universe},
    }
    dataset_config = {
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {"handler": handler_config, "segments": {"test": (start, end)}},
    }
    dataset = init_instance_by_config(dataset_config)
    X = dataset.prepare("test", col_set="feature")
    logger.info(f"Alpha158: {X.shape}, index names={X.index.names}")

    # 2. Load custom factors
    instruments = list(set(str(c) for c in X.index.get_level_values(1)))
    dates = sorted(X.index.get_level_values(0).unique())
    start_d, end_d = str(min(dates))[:10], str(max(dates))[:10]

    custom_raw = D.features(instruments, CUSTOM_EXPRS, start_time=start_d, end_time=end_d)
    custom_raw.columns = CUSTOM_NAMES
    logger.info(f"D.features raw: {custom_raw.shape}, index names={custom_raw.index.names}")

    # 3. Swaplevel + reindex
    custom = custom_raw.swaplevel().sort_index()
    custom_aligned = custom.reindex(X.index)

    # 4. Audit metrics
    audit_result = {
        "audit_at": datetime.now().isoformat(timespec="seconds"),
        "universe": universe,
        "test_start": start,
        "test_end": end,
    }

    # 4a. Index info
    audit_result["alpha158"] = {
        "shape": list(X.shape),
        "n_dates": len(dates),
        "n_instruments": len(instruments),
        "index_level0_name": str(X.index.names[0]),
        "index_level1_name": str(X.index.names[1]),
        "index_level0_dtype": str(X.index.get_level_values(0).dtype),
        "index_level1_sample": str(X.index.get_level_values(1)[0]),
    }
    audit_result["custom_raw"] = {
        "shape": list(custom_raw.shape),
        "index_level0_name": str(custom_raw.index.names[0]),
        "index_level1_name": str(custom_raw.index.names[1]),
        "index_level0_sample": str(custom_raw.index.get_level_values(0)[0]),
        "index_level1_sample": str(custom_raw.index.get_level_values(1)[0]),
    }

    # 4b. Coverage after reindex
    per_col_missing = {}
    per_col_inf = {}
    per_col_stats = {}
    for col in CUSTOM_NAMES:
        vals = custom_aligned[col].values
        missing = float(np.isnan(vals).mean())
        inf_count = float(np.isinf(vals).sum())
        finite = vals[np.isfinite(vals)]
        per_col_missing[col] = round(missing, 4)
        per_col_inf[col] = int(inf_count)
        if len(finite) > 0:
            per_col_stats[col] = {
                "min": round(float(np.min(finite)), 4),
                "p1": round(float(np.percentile(finite, 1)), 4),
                "p25": round(float(np.percentile(finite, 25)), 4),
                "median": round(float(np.median(finite)), 4),
                "p75": round(float(np.percentile(finite, 75)), 4),
                "p99": round(float(np.percentile(finite, 99)), 4),
                "max": round(float(np.max(finite)), 4),
                "mean": round(float(np.mean(finite)), 4),
                "std": round(float(np.std(finite)), 4),
            }

    audit_result["coverage"] = {
        "total_missing_ratio": round(float(custom_aligned.isna().mean().mean()), 4),
        "per_column_missing": per_col_missing,
        "per_column_inf": per_col_inf,
    }
    audit_result["distributions"] = per_col_stats

    # 4c. Per-date coverage
    daily_coverage = []
    for date in dates[:5]:  # Sample first 5 dates
        day_data = custom_aligned.loc[date]
        daily_coverage.append({
            "date": str(date)[:10],
            "n_stocks": len(day_data),
            "missing_ratio": round(float(day_data.isna().mean().mean()), 4),
        })
    audit_result["daily_coverage_sample"] = daily_coverage

    # 4d. Spot check: random sample 10 (date, instrument) pairs
    spot_checks = []
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(X.index), size=min(n_spot_check, len(X.index)), replace=False)

    for idx in sample_indices:
        dt, inst = X.index[idx]
        # Get value from aligned custom
        aligned_vals = {col: _safe_val(custom_aligned.loc[(dt, inst), col]) for col in CUSTOM_NAMES}

        # Get value directly from D.features for cross-check
        try:
            direct = D.features([inst], CUSTOM_EXPRS, start_time=str(dt)[:10], end_time=str(dt)[:10])
            if direct is not None and len(direct) > 0:
                direct_vals = {CUSTOM_NAMES[i]: _safe_val(direct.iloc[0, i]) for i in range(len(CUSTOM_NAMES))}
            else:
                direct_vals = {col: None for col in CUSTOM_NAMES}
        except Exception:
            direct_vals = {col: "ERROR" for col in CUSTOM_NAMES}

        match = all(
            _vals_match(aligned_vals.get(col), direct_vals.get(col))
            for col in CUSTOM_NAMES
        )

        spot_checks.append({
            "date": str(dt)[:10],
            "instrument": str(inst),
            "aligned": aligned_vals,
            "direct": direct_vals,
            "match": match,
        })

    audit_result["spot_checks"] = spot_checks
    n_match = sum(1 for s in spot_checks if s["match"])
    audit_result["spot_check_match_ratio"] = f"{n_match}/{len(spot_checks)}"

    # 4e. Overall verdict
    total_missing = audit_result["coverage"]["total_missing_ratio"]
    all_match = all(s["match"] for s in spot_checks)

    if total_missing > 0.5:
        verdict = "FAIL: >50% missing after reindex"
    elif not all_match:
        verdict = f"WARN: {len(spot_checks)-n_match}/{len(spot_checks)} spot checks mismatched"
    elif total_missing > 0.1:
        verdict = "WARN: >10% missing"
    else:
        verdict = "PASS"

    audit_result["verdict"] = verdict
    logger.info(f"Verdict: {verdict}")
    return audit_result


def _safe_val(v):
    try:
        f = float(v)
        return round(f, 6) if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _vals_match(a, b, rtol=1e-3):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a == 0 and b == 0:
        return True
    try:
        return abs(a - b) / (abs(a) + 1e-10) < rtol
    except (TypeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description="Audit factor merge alignment")
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--universe", default="csi300")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = audit(test_days=args.test_days, universe=args.universe)

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = AUDIT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    os.replace(tmp, AUDIT_PATH)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Verdict: {result['verdict']}")
        print(f"Missing: {result['coverage']['total_missing_ratio']:.2%}")
        print(f"Spot checks: {result['spot_check_match_ratio']}")
        for col, stats in result.get("distributions", {}).items():
            print(f"  {col}: median={stats['median']:.4f} range=[{stats['p1']:.4f}, {stats['p99']:.4f}]")

    sys.exit(0 if result["verdict"].startswith("PASS") else 1)


if __name__ == "__main__":
    main()
