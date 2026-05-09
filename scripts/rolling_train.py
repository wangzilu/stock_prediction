"""Rolling train and evaluation for LGB model.

Trains LGB on expanding windows, evaluates out-of-sample IC/RankIC/spread
across multiple periods to test signal stability.

Usage:
    python scripts/rolling_train.py
    python scripts/rolling_train.py --n-splits 12 --test-days 20
"""
import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config.settings import PREDICTION_HORIZON_DAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
ROLLING_PATH = DATA_DIR / "lgb_rolling_results.json"

LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"


def rolling_train(
    n_splits: int = 12,
    test_days: int = 20,
    train_years: int = 3,
    universe: str = "all",
):
    import qlib
    from qlib.constant import REG_CN
    from qlib.utils import init_instance_by_config
    from qlib.contrib.eva.alpha import calc_ic

    qlib.init(provider_uri=QLIB_DATA, region=REG_CN)

    today = datetime.now()
    results = []

    for split_idx in range(n_splits):
        # Walk backward: most recent split first
        test_end = today - timedelta(days=split_idx * test_days)
        test_start = test_end - timedelta(days=test_days)
        valid_end = test_start - timedelta(days=1)
        valid_start = valid_end - timedelta(days=60)
        train_end = valid_start - timedelta(days=1)
        train_start = train_end - timedelta(days=365 * train_years)

        dates = {
            "train": (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
            "valid": (valid_start.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")),
            "test": (test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")),
        }

        logger.info(f"Split {split_idx+1}/{n_splits}: test {dates['test'][0]}~{dates['test'][1]}")

        try:
            handler_config = {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": dates["train"][0],
                    "end_time": dates["test"][1],
                    "instruments": universe,
                    "label": [LABEL_EXPR],
                },
            }
            dataset_config = {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": handler_config,
                    "segments": {
                        "train": dates["train"],
                        "valid": dates["valid"],
                        "test": dates["test"],
                    },
                },
            }

            dataset = init_instance_by_config(dataset_config)

            model_config = {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {
                    "loss": "mse",
                    "colsample_bytree": 0.8879,
                    "learning_rate": 0.05,
                    "subsample": 0.8789,
                    "lambda_l1": 205.6999,
                    "lambda_l2": 580.9768,
                    "max_depth": 8,
                    "num_leaves": 210,
                    "num_threads": 4,
                },
            }

            model = init_instance_by_config(model_config)
            model.fit(dataset)

            # Evaluate on test
            pred = model.predict(dataset=dataset)
            if isinstance(pred, pd.Series):
                pred = pred.to_frame("score")

            label = dataset.prepare("test", col_set="label")
            if isinstance(label, pd.DataFrame):
                label = label.iloc[:, 0]

            common = pred.index.intersection(label.index)
            pred_s = pred.loc[common, "score"] if "score" in pred.columns else pred.loc[common].iloc[:, 0]
            label_s = label.loc[common]

            mask = pred_s.notna() & label_s.notna() & np.isfinite(pred_s) & np.isfinite(label_s)
            pred_clean = pred_s[mask]
            label_clean = label_s[mask]

            if len(pred_clean) < 50:
                logger.warning(f"  Split {split_idx+1}: too few samples ({len(pred_clean)}), skipping")
                continue

            ic, rank_ic = calc_ic(pred_clean, label_clean)

            # TopK spread
            df = pd.DataFrame({"pred": pred_clean, "label": label_clean})
            spreads = []
            for date, g in df.groupby(level=0):
                if len(g) < 40:
                    continue
                s = g.sort_values("pred", ascending=False)
                spreads.append(s.head(20)["label"].mean() - s.tail(20)["label"].mean())

            result = {
                "split": split_idx + 1,
                "test_start": dates["test"][0],
                "test_end": dates["test"][1],
                "n_samples": len(pred_clean),
                "n_dates": len(ic),
                "ic_mean": round(float(ic.mean()), 6),
                "rank_ic_mean": round(float(rank_ic.mean()), 6),
                "rank_ic_pos": round(float((rank_ic > 0).mean()), 4),
                "top20_spread": round(float(np.mean(spreads)), 6) if spreads else 0.0,
                "spread_pos": round(float(np.mean([s > 0 for s in spreads])), 4) if spreads else 0.0,
            }
            results.append(result)

            logger.info(
                f"  IC={result['ic_mean']:.4f} RankIC={result['rank_ic_mean']:.4f} "
                f"Spread={result['top20_spread']*100:.3f}% ({result['n_dates']} dates)"
            )

        except Exception as e:
            logger.error(f"  Split {split_idx+1} failed: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(description="Rolling LGB training")
    parser.add_argument("--n-splits", type=int, default=12)
    parser.add_argument("--test-days", type=int, default=20)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--universe", default="all")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = rolling_train(
        n_splits=args.n_splits,
        test_days=args.test_days,
        train_years=args.train_years,
        universe=args.universe,
    )

    if not results:
        logger.error("No valid rolling splits completed")
        sys.exit(1)

    # Summary
    ic_values = [r["ic_mean"] for r in results]
    rank_ic_values = [r["rank_ic_mean"] for r in results]
    spread_values = [r["top20_spread"] for r in results]

    summary = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "n_splits": len(results),
        "test_days_per_split": args.test_days,
        "label_expression": LABEL_EXPR,
        "aggregate": {
            "ic_mean": round(float(np.mean(ic_values)), 6),
            "ic_std": round(float(np.std(ic_values)), 6),
            "rank_ic_mean": round(float(np.mean(rank_ic_values)), 6),
            "rank_ic_pos_splits": round(float(np.mean([r > 0 for r in rank_ic_values])), 4),
            "top20_spread_mean": round(float(np.mean(spread_values)), 6),
            "spread_pos_splits": round(float(np.mean([s > 0 for s in spread_values])), 4),
        },
        "splits": results,
    }

    # Save
    ROLLING_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = ROLLING_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    os.replace(tmp, ROLLING_PATH)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Rolling LGB: {len(results)} splits, {args.test_days} days each")
        print(f"{'='*60}")
        agg = summary["aggregate"]
        print(f"IC mean:       {agg['ic_mean']:.4f} +/- {agg['ic_std']:.4f}")
        print(f"RankIC mean:   {agg['rank_ic_mean']:.4f} (>0 in {agg['rank_ic_pos_splits']:.0%} splits)")
        print(f"Top20 spread:  {agg['top20_spread_mean']*100:.3f}% (>0 in {agg['spread_pos_splits']:.0%} splits)")
        print(f"{'='*60}")
        for r in results:
            print(
                f"  Split {r['split']:2d}: {r['test_start']}~{r['test_end']} "
                f"IC={r['ic_mean']:+.4f} RankIC={r['rank_ic_mean']:+.4f} "
                f"Spread={r['top20_spread']*100:+.3f}%"
            )


if __name__ == "__main__":
    main()
