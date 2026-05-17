"""Rolling validation of backtest configs: verify biweekly+dropout across 12 splits.

Tests whether the +35.7% annual / Sharpe 1.31 result is stable or lucky.

Usage:
    python scripts/rolling_backtest_configs.py
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
from models.feature_merger import FeatureMerger
from models.feature_pipeline import (
    prepare_features_174, train_xgb, load_daily_returns,
)
from backtest.cost_model import CostModel
from backtest.portfolio_backtest import PortfolioBacktest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"


def main():
    import xgboost as xgb
    from qlib.utils import init_instance_by_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=12)
    parser.add_argument("--test-days", type=int, default=40,
                        help="Trading days per test split (longer for backtest)")
    parser.add_argument("--train-years", type=int, default=3)
    args = parser.parse_args()

    init_qlib(QLIB_DATA)
    merger = FeatureMerger(DATA_DIR)
    cost = CostModel()

    # Configs to compare
    configs = [
        {"name": "daily", "mode": "fixed", "rebalance_freq": 1, "dropout_k": 0, "hold_bonus": 0},
        {"name": "buffered_partial", "mode": "buffered_partial", "rebalance_freq": 1,
         "dropout_k": 0, "hold_bonus": 0},
        {"name": "buffered+stop8%", "mode": "buffered_partial", "rebalance_freq": 1,
         "dropout_k": 0, "hold_bonus": 0,
         "extra": {"buffer": 5, "trade_rate": 0.35, "max_daily_turnover": 0.15, "drawdown_stop": 0.08}},
    ]

    from qlib.data import D
    # Get trading calendar
    cal = D.calendar(start_time="2020-01-01", end_time=datetime.now().strftime("%Y-%m-%d"))
    trade_dates = sorted(cal)
    today_idx = len(trade_dates) - 1

    all_split_results = []

    for split_idx in range(args.n_splits):
        test_end_idx = today_idx - split_idx * args.test_days
        test_start_idx = test_end_idx - args.test_days
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - 60
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - (args.train_years * 250)

        if train_start_idx < 0:
            break

        test_end = str(trade_dates[test_end_idx])[:10]
        test_start = str(trade_dates[test_start_idx])[:10]
        valid_end = str(trade_dates[valid_end_idx])[:10]
        valid_start = str(trade_dates[valid_start_idx])[:10]
        train_end = str(trade_dates[train_end_idx])[:10]
        train_start = str(trade_dates[train_start_idx])[:10]

        logger.info(f"\nSplit {split_idx+1}/{args.n_splits}: test {test_start}~{test_end}")

        try:
            dataset = init_instance_by_config({
                "class": "DatasetH", "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
                        "kwargs": {"start_time": train_start, "end_time": test_end,
                                   "instruments": "all", "label": [LABEL_EXPR]},
                    },
                    "segments": {
                        "train": (train_start, train_end),
                        "valid": (valid_start, valid_end),
                        "test": (test_start, test_end),
                    },
                },
            })

            # Prepare features
            X_train_df, y_train_s = prepare_features_174(dataset, "train", merger)
            X_valid_df, y_valid_s = prepare_features_174(dataset, "valid", merger)
            X_test_df, y_test_s = prepare_features_174(dataset, "test", merger)

            y_train = y_train_s.values.astype(np.float32)
            mask_train = np.isfinite(y_train)
            y_valid = y_valid_s.values.astype(np.float32)
            mask_valid = np.isfinite(y_valid)

            # Train
            model = train_xgb(
                X_train_df.values.astype(np.float32)[mask_train], y_train[mask_train],
                X_valid_df.values.astype(np.float32)[mask_valid], y_valid[mask_valid])

            # Predict
            pred = model.predict(xgb.DMatrix(X_test_df.values.astype(np.float32)))
            predictions = pd.Series(pred, index=X_test_df.index, name="score")
            predictions = predictions[np.isfinite(predictions)]

            # Load daily returns
            daily_returns = load_daily_returns(X_test_df.index)
            if isinstance(daily_returns, pd.DataFrame):
                daily_returns = daily_returns.rename(columns={"pnl_return_1d": "return"})

            # Run each config
            split_result = {"split": split_idx + 1, "test": f"{test_start}~{test_end}"}
            for cfg in configs:
                extra = cfg.get("extra", {})
                bt = PortfolioBacktest(
                    top_k=20, cost_model=cost,
                    mode=cfg.get("mode", "fixed"),
                    rebalance_freq=cfg["rebalance_freq"],
                    dropout_k=cfg["dropout_k"],
                    hold_bonus=cfg["hold_bonus"],
                    **extra,
                )
                r = bt.run(predictions=predictions.to_frame("score"),
                           returns=daily_returns, return_horizon_days=1)

                split_result[cfg["name"]] = {
                    "annual": round(r.annual_return * 100, 2),
                    "sharpe": round(r.sharpe_ratio, 3),
                    "maxdd": round(r.max_drawdown * 100, 2),
                    "turnover": round(r.avg_turnover * 100, 1),
                    "cost_ratio": round(r.cost_to_return_ratio * 100, 1),
                }
                logger.info(f"  {cfg['name']:<20} annual={r.annual_return*100:+.1f}% "
                            f"sharpe={r.sharpe_ratio:+.3f} dd={r.max_drawdown*100:.1f}%")

            all_split_results.append(split_result)

        except Exception as e:
            logger.error(f"  Split {split_idx+1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_split_results:
        logger.error("No valid splits")
        sys.exit(1)

    # Summary
    n = len(all_split_results)
    logger.info(f"\n{'='*80}")
    logger.info(f"ROLLING BACKTEST CONFIGS: {n} splits × {args.test_days} trading days")
    logger.info(f"{'='*80}")

    for cfg in configs:
        name = cfg["name"]
        annuals = [r[name]["annual"] for r in all_split_results if name in r]
        sharpes = [r[name]["sharpe"] for r in all_split_results if name in r]
        pos_pct = sum(1 for a in annuals if a > 0) / len(annuals) if annuals else 0

        logger.info(f"\n  {name}:")
        logger.info(f"    avg annual:  {np.mean(annuals):+.1f}%")
        logger.info(f"    avg sharpe:  {np.mean(sharpes):+.3f}")
        logger.info(f"    annual>0:    {pos_pct:.0%} ({sum(1 for a in annuals if a > 0)}/{len(annuals)})")
        logger.info(f"    per-split:   {['%+.1f%%' % a for a in annuals]}")

    # Save
    out_path = DATA_DIR / "rolling_backtest_configs.json"
    with open(str(out_path), "w") as f:
        json.dump({"evaluated_at": datetime.now().isoformat(timespec="seconds"),
                   "n_splits": n, "test_days": args.test_days,
                   "splits": all_split_results}, f, indent=2)
    logger.info(f"\nSaved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
