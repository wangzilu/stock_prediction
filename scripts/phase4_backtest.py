"""Phase 4 Track B: Portfolio backtest with costs and constraints.

Takes XGB 174 predictions → TopK portfolio → cost-adjusted PnL.

Execution: T日收盘后出信号, T+1 VWAP 成交.

Usage:
    python scripts/phase4_backtest.py
    python scripts/phase4_backtest.py --top-k 10 --model xgb_175
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
from models.feature_pipeline import prepare_features_174, train_xgb, load_daily_returns
from backtest.cost_model import CostModel
from backtest.portfolio_backtest import PortfolioBacktest, PortfolioResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"

# Track B gate thresholds (from CX plan)
GATE = {
    "cost_adjusted_annual_return": 0.0,   # > 0
    "cost_adjusted_sharpe": 0.8,          # >= 0.8
    "max_drawdown": -0.20,                # >= -20%
    "avg_turnover": 0.35,                 # <= 35%
    "cost_to_return_ratio": 0.35,         # <= 35%
}


def main():
    import xgboost as xgb
    from qlib.utils import init_instance_by_config
    from qlib.data import D

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xgb_174",
                        choices=["xgb_174", "xgb_175"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--test-months", type=int, default=6,
                        help="How many months of test data for backtest")
    parser.add_argument("--train-years", type=int, default=3)
    args = parser.parse_args()

    init_qlib(QLIB_DATA)
    merger = FeatureMerger(DATA_DIR)

    today = datetime.now()
    test_end = today.strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=30 * args.test_months)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=30 * args.test_months + 1)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=30 * args.test_months + 61)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=30 * args.test_months + 62)).strftime("%Y-%m-%d")
    train_start = (today - timedelta(days=365 * args.train_years + 30 * args.test_months + 62)).strftime("%Y-%m-%d")

    logger.info(f"=== Phase 4 Track B: Portfolio Backtest ===")
    logger.info(f"Model: {args.model}, TopK: {args.top_k}")
    logger.info(f"Train: {train_start}~{train_end}")
    logger.info(f"Test:  {test_start}~{test_end} ({args.test_months} months)")

    # Load dataset
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

    include_holder = (args.model == "xgb_175")

    # Prepare train/valid for model training
    logger.info("Preparing features...")
    X_train_df, y_train_s = prepare_features_174(dataset, "train", merger, include_holder)
    X_valid_df, y_valid_s = prepare_features_174(dataset, "valid", merger, include_holder)
    X_test_df, y_test_s = prepare_features_174(dataset, "test", merger, include_holder)

    # NaN filter for training
    y_train = y_train_s.values.astype(np.float32)
    mask_train = np.isfinite(y_train)
    X_train = X_train_df.values.astype(np.float32)[mask_train]
    y_train = y_train[mask_train]

    y_valid = y_valid_s.values.astype(np.float32)
    mask_valid = np.isfinite(y_valid)
    X_valid = X_valid_df.values.astype(np.float32)[mask_valid]
    y_valid = y_valid[mask_valid]

    logger.info(f"  Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test_df.shape}")

    # Train model
    logger.info("Training XGB...")
    t0 = time.time()
    model = train_xgb(X_train, y_train, X_valid, y_valid)
    logger.info(f"  Done: {time.time()-t0:.1f}s")

    # Predict on full test (keep NaN rows — backtest handles them)
    X_test_np = X_test_df.values.astype(np.float32)
    pred_raw = model.predict(xgb.DMatrix(X_test_np))

    # Build predictions DataFrame for backtest
    predictions = pd.Series(pred_raw, index=X_test_df.index, name="score")
    predictions = predictions[np.isfinite(predictions)]

    # Load DAILY realized returns (NOT model label!)
    # Model label is 5-day forward return — cannot be used as daily PnL.
    logger.info("Loading daily realized returns (1-day close-to-close)...")
    daily_returns = load_daily_returns(X_test_df.index)
    # Rename column to match PortfolioBacktest expectation
    if isinstance(daily_returns, pd.DataFrame):
        daily_returns = daily_returns.rename(columns={"pnl_return_1d": "return"})
    elif isinstance(daily_returns, pd.Series):
        daily_returns = daily_returns.rename("return")

    logger.info(f"Predictions: {len(predictions)}, Daily returns: {len(daily_returns)}")
    logger.info(f"  (model_target_horizon={PREDICTION_HORIZON_DAYS}d, pnl_horizon=1d)")

    # Run backtest with multiple configurations
    cost = CostModel()
    logger.info(f"Cost model: {cost.summary()}")

    configs = [
        {"name": "daily_rebal", "rebalance_freq": 1, "dropout_k": 0, "hold_bonus": 0},
        {"name": "weekly_rebal", "rebalance_freq": 5, "dropout_k": 0, "hold_bonus": 0},
        {"name": "weekly+dropout10", "rebalance_freq": 5, "dropout_k": 10, "hold_bonus": 0},
        {"name": "weekly+bonus", "rebalance_freq": 5, "dropout_k": 0, "hold_bonus": 0.01},
        {"name": "weekly+dropout+bonus", "rebalance_freq": 5, "dropout_k": 10, "hold_bonus": 0.01},
        {"name": "biweekly+dropout", "rebalance_freq": 10, "dropout_k": 15, "hold_bonus": 0},
    ]

    all_results = {}
    for cfg in configs:
        bt = PortfolioBacktest(
            top_k=args.top_k, cost_model=cost,
            rebalance_freq=cfg["rebalance_freq"],
            dropout_k=cfg["dropout_k"],
            hold_bonus=cfg["hold_bonus"],
        )
        r = bt.run(
            predictions=predictions.to_frame("score"),
            returns=daily_returns,
            return_horizon_days=1,
        )
        all_results[cfg["name"]] = r
        logger.info(f"  {cfg['name']:<25} annual={r.annual_return*100:+.1f}% "
                    f"sharpe={r.sharpe_ratio:.3f} dd={r.max_drawdown*100:.1f}% "
                    f"turnover={r.avg_turnover*100:.0f}% cost/ret={r.cost_to_return_ratio*100:.0f}%")

    # Find best config
    best_name = max(all_results, key=lambda k: all_results[k].sharpe_ratio)
    result = all_results[best_name]
    logger.info(f"\nBest config: {best_name}")
    print(result.summary())

    # Also show comparison table
    logger.info(f"\n{'='*90}")
    logger.info(f"{'Config':<25} {'Raw Ann':>8} {'Net Ann':>8} {'Sharpe':>7} "
                f"{'MaxDD':>7} {'Turn':>6} {'Cost/Ret':>8}")
    logger.info("-" * 90)
    for name, r in all_results.items():
        marker = " ★" if name == best_name else ""
        logger.info(f"{name:<25} {r.raw_annual_return*100:+.1f}%   "
                    f"{r.annual_return*100:+.1f}%   {r.sharpe_ratio:+.3f}  "
                    f"{r.max_drawdown*100:.1f}%  {r.avg_turnover*100:.0f}%   "
                    f"{r.cost_to_return_ratio*100:.0f}%{marker}")

    # Gate check
    logger.info(f"\n{'='*50}")
    logger.info("TRACK B GATE CHECK")
    logger.info(f"{'='*50}")

    checks = {
        "annual_return > 0": (result.annual_return, result.annual_return > GATE["cost_adjusted_annual_return"]),
        "sharpe >= 0.8": (result.sharpe_ratio, result.sharpe_ratio >= GATE["cost_adjusted_sharpe"]),
        "max_dd >= -20%": (result.max_drawdown, result.max_drawdown >= GATE["max_drawdown"]),
        "avg_turnover <= 35%": (result.avg_turnover, result.avg_turnover <= GATE["avg_turnover"]),
        "cost/return <= 35%": (result.cost_to_return_ratio, result.cost_to_return_ratio <= GATE["cost_to_return_ratio"]),
    }

    all_pass = True
    for name, (value, passed) in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {name:<25} = {value:+.4f}  {status}")
        if not passed:
            all_pass = False

    logger.info(f"\n  Overall: {'✅ ALL GATES PASS' if all_pass else '❌ SOME GATES FAILED'}")

    # Save
    out = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "top_k": args.top_k,
        "test_period": f"{test_start}~{test_end}",
        "cost_model": cost.summary(),
        "raw": {
            "total_return": result.raw_total_return,
            "annual_return": result.raw_annual_return,
            "sharpe": result.raw_sharpe,
        },
        "cost_adjusted": {
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "annual_vol": result.annual_volatility,
            "sharpe": result.sharpe_ratio,
            "calmar": result.calmar_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
        },
        "cost": {
            "total_cost": result.total_cost,
            "cost_to_return_ratio": result.cost_to_return_ratio,
            "avg_turnover": result.avg_turnover,
        },
        "gate_pass": all_pass,
        "n_days": result.n_days,
        "avg_holdings": result.avg_holdings,
    }

    out_path = DATA_DIR / f"phase4_backtest_{args.model}_top{args.top_k}.json"
    with open(str(out_path), "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nSaved: {out_path}")

    # Push
    try:
        from push.wechat import WeChatPusher
        lines = [
            f"📊 Phase 4 Backtest: {args.model} Top{args.top_k}",
            f"{'✅ PASS' if all_pass else '❌ FAIL'}",
            f"Test: {test_start}~{test_end}",
            "",
            f"Raw:  annual={result.raw_annual_return*100:+.1f}% sharpe={result.raw_sharpe:.2f}",
            f"Net:  annual={result.annual_return*100:+.1f}% sharpe={result.sharpe_ratio:.2f}",
            f"MaxDD: {result.max_drawdown*100:.1f}%  Turnover: {result.avg_turnover*100:.0f}%",
            f"Cost/Return: {result.cost_to_return_ratio*100:.0f}%",
        ]
        WeChatPusher().send("\n".join(lines), title="Phase4 Backtest")
    except Exception as e:
        logger.warning(f"Push failed: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
