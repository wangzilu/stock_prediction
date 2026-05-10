"""Single-factor IC test for all candidate factors.

Tests each factor's predictive power independently before adding to model.
Uses Qlib's threading backend to avoid macOS multiprocessing spawn issues.

Usage:
    python scripts/evaluate_factor_ic.py
    python scripts/evaluate_factor_ic.py --json
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

from config.settings import PREDICTION_HORIZON_DAYS
from config.qlib_runtime import init_qlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
FACTOR_IC_PATH = DATA_DIR / "factor_ic_test.json"

LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"

# All candidate factors to test
FACTORS = {
    # Raw valuation
    "$pe": "PE(市盈率)",
    "$pb": "PB(市净率)",
    "1.0 / If(Abs($pe) > 0.01, $pe, 1.0)": "EP(盈利收益率)",
    "1.0 / If(Abs($pb) > 0.01, $pb, 1.0)": "BP(净资产收益率)",
    # Valuation momentum
    "$pe / Ref($pe, 5) - 1": "PE动量5日",
    "$pe / Ref($pe, 20) - 1": "PE动量20日",
    "$pb / Ref($pb, 5) - 1": "PB动量5日",
    "$pb / Ref($pb, 20) - 1": "PB动量20日",
    # Valuation relative
    "$pe / Mean($pe, 60)": "PE相对60日",
    "$pb / Mean($pb, 60)": "PB相对60日",
    # Turnover
    "$turn": "换手率",
    "$turn / Mean($turn, 20)": "换手异常20日",
    "$turn / Mean($turn, 60)": "换手异常60日",
    "Mean($turn, 5) / Mean($turn, 20)": "换手动量",
    "Std($turn, 20)": "换手波动",
    # Amount
    "$amount / Mean($amount, 20)": "成交额异常20日",
    # Position
    "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20) + 1e-8)": "价格位置20日",
    "($close - Min($close, 60)) / (Max($close, 60) - Min($close, 60) + 1e-8)": "价格位置60日",
    # Existing Alpha158 baseline for comparison
    "$close / Ref($close, 5) - 1": "5日收益率(Alpha158基线)",
    "Mean($close, 5) / Mean($close, 20) - 1": "MA5/MA20(Alpha158基线)",
}


def test_all_factors(test_days: int = 60, universe: str = "all") -> list:
    from qlib.data import D
    from qlib.contrib.eva.alpha import calc_ic

    init_qlib(QLIB_DATA)

    today = datetime.now()
    test_start = (today - timedelta(days=test_days)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")

    logger.info(f"Testing {len(FACTORS)} factors on {test_start} ~ {test_end}")
    logger.info(f"Label: {LABEL_EXPR}, Universe: {universe}")

    # Get instruments
    instruments = D.instruments(universe)

    # Fetch label
    logger.info("Fetching labels...")
    t0 = time.time()
    label_df = D.features(instruments, [LABEL_EXPR], start_time=test_start, end_time=test_end)
    label_df.columns = ["label"]
    # D.features returns (instrument, datetime), swap to (datetime, instrument)
    label_df = label_df.swaplevel().sort_index()
    logger.info(f"  Labels: {len(label_df)} samples, {time.time()-t0:.0f}s")

    results = []
    for expr, name in FACTORS.items():
        logger.info(f"Testing: {name}...")
        t1 = time.time()
        try:
            factor_df = D.features(instruments, [expr], start_time=test_start, end_time=test_end)
            factor_df.columns = ["factor"]
            factor_df = factor_df.swaplevel().sort_index()

            common = factor_df.index.intersection(label_df.index)
            f = factor_df.loc[common, "factor"]
            l = label_df.loc[common, "label"]

            mask = f.notna() & l.notna() & np.isfinite(f) & np.isfinite(l)
            f, l = f[mask], l[mask]

            if len(f) < 1000:
                logger.info(f"  SKIP (only {len(f)} samples)")
                continue

            ic, ric = calc_ic(f, l)
            ic_mean = float(ic.mean())
            ric_mean = float(ric.mean())
            icir = ic_mean / (float(ic.std()) + 1e-8)
            ric_pos = float((ric > 0).mean())

            # TopK spread
            df = pd.DataFrame({"pred": f, "label": l})
            spreads = []
            for d, g in df.groupby(level=0):
                if len(g) < 40:
                    continue
                s = g.sort_values("pred", ascending=False)
                spreads.append(s.head(20)["label"].mean() - s.tail(20)["label"].mean())

            spread = float(np.mean(spreads)) if spreads else 0.0
            elapsed = time.time() - t1

            spread_pos = float(np.mean([s > 0 for s in spreads])) if spreads else 0.0
            coverage = len(f) / max(len(label_df), 1)

            # Joint verdict: must satisfy ALL conditions for STRONG
            # (cx-corrected: not just Pearson IC, but RankIC + TopK spread + coverage)
            if ric_mean > 0.01 and spread > 0 and ic_mean > 0 and ric_pos > 0.5:
                verdict = "STRONG"
            elif (ric_mean > 0 or spread > 0) and ic_mean > 0:
                verdict = "OK"
            elif ic_mean > 0:
                verdict = "WEAK"
            else:
                verdict = "NEGATIVE"

            elapsed = time.time() - t1
            logger.info(
                f"  IC={ic_mean:+.4f} RankIC={ric_mean:+.4f} Spread={spread*100:+.3f}% "
                f"RIC>0={ric_pos:.0%} Cov={coverage:.0%} → {verdict} ({elapsed:.0f}s)"
            )

            results.append({
                "name": name,
                "expr": expr,
                "ic_mean": round(ic_mean, 6),
                "rank_ic_mean": round(ric_mean, 6),
                "icir": round(icir, 4),
                "rank_ic_pos_ratio": round(ric_pos, 4),
                "top20_spread": round(spread, 6),
                "spread_pos_ratio": round(spread_pos, 4),
                "coverage": round(coverage, 4),
                "n_samples": len(f),
                "verdict": verdict,
            })
        except Exception as e:
            logger.error(f"  ERROR: {e}")

    # Sort by composite score: RankIC * spread_pos_ratio (what matters for TopK trading)
    results.sort(key=lambda r: r.get("rank_ic_mean", 0) * r.get("spread_pos_ratio", 0), reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Single-factor IC test")
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--universe", default="all")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = test_all_factors(test_days=args.test_days, universe=args.universe)

    # Save
    output = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "label": LABEL_EXPR,
        "factors": results,
    }
    FACTOR_IC_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = FACTOR_IC_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    os.replace(tmp, FACTOR_IC_PATH)

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*85}")
        print(f"{'Factor':<22} {'IC':>8} {'RankIC':>8} {'Spread%':>9} {'RIC>0':>7} {'Cov':>6} {'Verdict':>10}")
        print(f"{'-'*85}")
        for r in results:
            print(
                f"{r['name']:<22} {r['ic_mean']:>+8.4f} {r['rank_ic_mean']:>+8.4f} "
                f"{r['top20_spread']*100:>+8.3f}% {r['rank_ic_pos_ratio']:>6.0%} "
                f"{r.get('coverage', 0):>5.0%} {r['verdict']:>10}"
            )
        print(f"{'='*85}")
        print(f"STRONG = RankIC>0.01 AND Spread>0 AND IC>0 AND RankIC_pos>50%")

        strong = [r for r in results if r["verdict"] == "STRONG"]
        ok = [r for r in results if r["verdict"] == "OK"]
        weak = [r for r in results if r["verdict"] == "WEAK"]
        neg = [r for r in results if r["verdict"] == "NEGATIVE"]
        print(f"\nSTRONG: {len(strong)}  OK: {len(ok)}  WEAK: {len(weak)}  NEGATIVE: {len(neg)}")
        if strong:
            print(f"推荐加入模型: {', '.join(r['name'] for r in strong)}")
        else:
            print("暂无因子满足联合标准，建议用 rolling 多窗口进一步验证")


if __name__ == "__main__":
    main()
