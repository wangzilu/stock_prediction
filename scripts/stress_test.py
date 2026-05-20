"""Stress test: evaluate model performance on known crisis/bull periods.

Tests whether the model survives known market regimes by:
1. Training on data BEFORE each stress period
2. Predicting DURING the stress period
3. Running buffered_partial backtest on those predictions
4. Comparing per-period return, sharpe, maxdd with full-period baseline

Stress periods:
- 2024_quant_crash: 2024-02-05 ~ 2024-02-29 (量化踩踏)
- 2022_bear: 2022-03-01 ~ 2022-05-31
- 2021_bull: 2021-01-01 ~ 2021-03-31 (牛市验证)
- recent: last 60 trading days

Usage:
    python scripts/stress_test.py
    python scripts/stress_test.py --cache feature_cache_174_holder_regime_ma.parquet
"""
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
SEED = 42


# ---------------------------------------------------------------------------
# Stress period definitions
# ---------------------------------------------------------------------------
STRESS_PERIODS = {
    "2024_quant_crash": {
        "start": "2024-02-05",
        "end": "2024-02-29",
        "desc": "量化踩踏 (quant deleveraging crash)",
    },
    "2022_bear": {
        "start": "2022-03-01",
        "end": "2022-05-31",
        "desc": "熊市 (bear market)",
    },
    "2021_bull": {
        "start": "2021-01-01",
        "end": "2021-03-31",
        "desc": "牛市验证 (bull market validation)",
    },
    "recent": {
        "start": None,  # dynamically set to last 60 trading days
        "end": None,
        "desc": "最近60个交易日 (last 60 trading days)",
    },
}

TRAIN_DAYS = 750  # ~3 years training window
VALID_DAYS = 60   # validation window


def train_xgb(X_train, y_train, X_valid, y_valid, nthread=12, max_rounds=400):
    """Train XGB model with standard params."""
    import xgboost as xgb
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_valid, label=y_valid)
    params = {
        "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
        "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
        "objective": "reg:squarederror", "nthread": nthread, "verbosity": 0, "seed": SEED,
    }
    model = xgb.train(
        params, dt, num_boost_round=max_rounds,
        evals=[(dv, "valid")], early_stopping_rounds=30, verbose_eval=0,
    )
    return model


def run_backtest_on_period(predictions_df, returns_df, top_k=20, full_history_index=None):
    """Run buffered_partial backtest on a given period's predictions and returns."""
    from backtest.cost_model import CostModel
    from backtest.portfolio_backtest import PortfolioBacktest

    cost = CostModel()
    bt = PortfolioBacktest(
        top_k=top_k, cost_model=cost,
        mode="buffered_partial",
        rebalance_freq=1,
        buffer=5,
        trade_rate=0.35,
        max_daily_turnover=0.15,
    )
    result = bt.run(
        predictions=predictions_df,
        returns=returns_df,
        return_horizon_days=1,
        full_history_index=full_history_index,
    )
    return result


def result_to_dict(result):
    """Extract key metrics from PortfolioResult."""
    return {
        "total_return": round(float(result.total_return), 6),
        "annual_return": round(float(result.annual_return), 6),
        "annual_volatility": round(float(result.annual_volatility), 6),
        "sharpe_ratio": round(float(result.sharpe_ratio), 4),
        "max_drawdown": round(float(result.max_drawdown), 6),
        "calmar_ratio": round(float(result.calmar_ratio), 4),
        "win_rate": round(float(result.win_rate), 4),
        "avg_turnover": round(float(result.avg_turnover), 4),
        "total_cost": round(float(result.total_cost), 6),
        "cost_to_return_ratio": round(float(result.cost_to_return_ratio), 4),
        "n_days": int(result.n_days),
        "avg_holdings": round(float(result.avg_holdings), 1),
        "raw_total_return": round(float(result.raw_total_return), 6),
        "raw_annual_return": round(float(result.raw_annual_return), 6),
        "raw_sharpe": round(float(result.raw_sharpe), 4),
    }


def main():
    import xgboost as xgb
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="feature_cache_174_holder_regime_ma.parquet")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--nthread", type=int, default=12)
    parser.add_argument("--max-rounds", type=int, default=400)
    args = parser.parse_args()

    # Load feature cache
    cache_path = DATA_DIR / args.cache
    if not cache_path.exists():
        logger.error(f"Cache not found: {cache_path}")
        logger.error("Run: python scripts/build_feature_cache.py --all")
        sys.exit(1)

    logger.info(f"Loading cache: {cache_path}")
    t0 = time.time()
    cache = pd.read_parquet(str(cache_path))
    logger.info(f"  Loaded: {cache.shape}, {time.time() - t0:.1f}s")

    # Separate features, labels, metadata
    feature_cols = [c for c in cache.columns if not c.startswith("__")]
    label_col = "__label_5d"
    pnl_col = "__pnl_return_1d"

    if label_col not in cache.columns:
        logger.error(f"Label column {label_col} not found")
        sys.exit(1)
    if pnl_col not in cache.columns:
        logger.error(f"PnL column {pnl_col} not found")
        sys.exit(1)

    trade_dates = sorted(cache.index.get_level_values(0).unique())
    dates_level = cache.index.get_level_values(0)
    logger.info(f"  Trading dates: {len(trade_dates)} ({str(trade_dates[0])[:10]} ~ {str(trade_dates[-1])[:10]})")

    # Resolve "recent" period
    STRESS_PERIODS["recent"]["end"] = str(trade_dates[-1])[:10]
    recent_start_idx = max(0, len(trade_dates) - 60)
    STRESS_PERIODS["recent"]["start"] = str(trade_dates[recent_start_idx])[:10]

    # -----------------------------------------------------------------------
    # Run full-period baseline first (train on first 80%, test on last 20%)
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("FULL-PERIOD BASELINE (for comparison)")
    logger.info("=" * 70)

    n_total = len(trade_dates)
    baseline_test_start_idx = int(n_total * 0.8)
    baseline_train_end_idx = baseline_test_start_idx - VALID_DAYS - 1
    baseline_valid_start_idx = baseline_train_end_idx + 1
    baseline_valid_end_idx = baseline_test_start_idx - 1
    baseline_train_start_idx = max(0, baseline_train_end_idx - TRAIN_DAYS)

    bl_train_start = trade_dates[baseline_train_start_idx]
    bl_train_end = trade_dates[baseline_train_end_idx]
    bl_valid_start = trade_dates[baseline_valid_start_idx]
    bl_valid_end = trade_dates[baseline_valid_end_idx]
    bl_test_start = trade_dates[baseline_test_start_idx]
    bl_test_end = trade_dates[-1]

    logger.info(f"  Train: {str(bl_train_start)[:10]} ~ {str(bl_train_end)[:10]}")
    logger.info(f"  Valid: {str(bl_valid_start)[:10]} ~ {str(bl_valid_end)[:10]}")
    logger.info(f"  Test:  {str(bl_test_start)[:10]} ~ {str(bl_test_end)[:10]}")

    # Train baseline model
    tr_mask = (dates_level >= bl_train_start) & (dates_level <= bl_train_end)
    va_mask = (dates_level >= bl_valid_start) & (dates_level <= bl_valid_end)
    te_mask = (dates_level >= bl_test_start) & (dates_level <= bl_test_end)

    X_tr = cache.loc[tr_mask, feature_cols].values.astype(np.float32)
    y_tr = cache.loc[tr_mask, label_col].values.astype(np.float32)
    X_va = cache.loc[va_mask, feature_cols].values.astype(np.float32)
    y_va = cache.loc[va_mask, label_col].values.astype(np.float32)
    X_te = cache.loc[te_mask, feature_cols].values.astype(np.float32)

    m_tr = np.isfinite(y_tr)
    m_va = np.isfinite(y_va)

    logger.info(f"  Training baseline XGB ({X_tr[m_tr].shape[0]} samples)...")
    t1 = time.time()
    baseline_model = train_xgb(
        X_tr[m_tr], y_tr[m_tr], X_va[m_va], y_va[m_va],
        nthread=args.nthread, max_rounds=args.max_rounds,
    )
    logger.info(f"  Done: {time.time() - t1:.1f}s")

    pred_bl = baseline_model.predict(xgb.DMatrix(X_te))
    pred_bl_series = pd.Series(pred_bl, index=cache.index[te_mask], name="score")
    pred_bl_series = pred_bl_series[np.isfinite(pred_bl_series)]

    returns_bl = cache.loc[te_mask, pnl_col].rename("return")
    returns_bl = returns_bl.replace([np.inf, -np.inf], np.nan).dropna()

    logger.info(f"  Running baseline backtest ({pred_bl_series.index.get_level_values(0).nunique()} dates)...")
    baseline_result = run_backtest_on_period(
        pred_bl_series.to_frame("score"), returns_bl.to_frame("return"), top_k=args.top_k,
        full_history_index=cache.index,
    )
    baseline_metrics = result_to_dict(baseline_result)
    logger.info(f"  Baseline: annual={baseline_result.annual_return * 100:+.1f}% "
                f"sharpe={baseline_result.sharpe_ratio:.3f} maxdd={baseline_result.max_drawdown * 100:.1f}%")

    # -----------------------------------------------------------------------
    # Stress test each period
    # -----------------------------------------------------------------------
    stress_results = {}

    for period_name, period_cfg in STRESS_PERIODS.items():
        logger.info(f"\n{'=' * 70}")
        logger.info(f"STRESS PERIOD: {period_name} ({period_cfg['desc']})")
        logger.info(f"  {period_cfg['start']} ~ {period_cfg['end']}")
        logger.info("=" * 70)

        # Find test date range in trading calendar
        test_start = pd.Timestamp(period_cfg["start"])
        test_end = pd.Timestamp(period_cfg["end"])

        # Get actual trading dates within the period
        test_dates_in_period = [d for d in trade_dates if test_start <= d <= test_end]
        if len(test_dates_in_period) < 5:
            logger.warning(f"  Only {len(test_dates_in_period)} trading dates in period, skipping")
            stress_results[period_name] = {
                "status": "skipped",
                "reason": f"only {len(test_dates_in_period)} trading dates",
                **period_cfg,
            }
            continue

        actual_test_start = test_dates_in_period[0]
        actual_test_end = test_dates_in_period[-1]

        # Find training window: train on data BEFORE the stress period
        test_start_idx = trade_dates.index(actual_test_start)
        valid_end_idx = test_start_idx - 1
        if valid_end_idx < VALID_DAYS:
            logger.warning(f"  Not enough history for training, skipping")
            stress_results[period_name] = {"status": "skipped", "reason": "insufficient history", **period_cfg}
            continue

        valid_start_idx = valid_end_idx - VALID_DAYS
        train_end_idx = valid_start_idx - 1
        train_start_idx = max(0, train_end_idx - TRAIN_DAYS)

        s_train_start = trade_dates[train_start_idx]
        s_train_end = trade_dates[train_end_idx]
        s_valid_start = trade_dates[valid_start_idx]
        s_valid_end = trade_dates[valid_end_idx]

        logger.info(f"  Train: {str(s_train_start)[:10]} ~ {str(s_train_end)[:10]} ({train_end_idx - train_start_idx} days)")
        logger.info(f"  Valid: {str(s_valid_start)[:10]} ~ {str(s_valid_end)[:10]}")
        logger.info(f"  Test:  {str(actual_test_start)[:10]} ~ {str(actual_test_end)[:10]} ({len(test_dates_in_period)} days)")

        # Slice data
        s_tr_mask = (dates_level >= s_train_start) & (dates_level <= s_train_end)
        s_va_mask = (dates_level >= s_valid_start) & (dates_level <= s_valid_end)
        s_te_mask = (dates_level >= actual_test_start) & (dates_level <= actual_test_end)

        X_str = cache.loc[s_tr_mask, feature_cols].values.astype(np.float32)
        y_str = cache.loc[s_tr_mask, label_col].values.astype(np.float32)
        X_sva = cache.loc[s_va_mask, feature_cols].values.astype(np.float32)
        y_sva = cache.loc[s_va_mask, label_col].values.astype(np.float32)
        X_ste = cache.loc[s_te_mask, feature_cols].values.astype(np.float32)

        m_str = np.isfinite(y_str)
        m_sva = np.isfinite(y_sva)

        logger.info(f"  Training XGB ({X_str[m_str].shape[0]} train samples, {X_sva[m_sva].shape[0]} valid)...")
        t1 = time.time()
        model = train_xgb(
            X_str[m_str], y_str[m_str], X_sva[m_sva], y_sva[m_sva],
            nthread=args.nthread, max_rounds=args.max_rounds,
        )
        train_time = time.time() - t1
        logger.info(f"  Trained in {train_time:.1f}s")

        # Predict
        pred = model.predict(xgb.DMatrix(X_ste))
        pred_series = pd.Series(pred, index=cache.index[s_te_mask], name="score")
        pred_series = pred_series[np.isfinite(pred_series)]

        returns_period = cache.loc[s_te_mask, pnl_col].rename("return")
        returns_period = returns_period.replace([np.inf, -np.inf], np.nan).dropna()

        logger.info(f"  Predictions: {len(pred_series)}, Returns: {len(returns_period)}")

        # Run backtest
        t2 = time.time()
        result = run_backtest_on_period(
            pred_series.to_frame("score"),
            returns_period.to_frame("return"),
            top_k=args.top_k,
            full_history_index=cache.index,
        )
        bt_time = time.time() - t2

        metrics = result_to_dict(result)
        metrics["train_time_s"] = round(train_time, 1)
        metrics["backtest_time_s"] = round(bt_time, 1)
        metrics["status"] = "ok"
        metrics["desc"] = period_cfg["desc"]
        metrics["period"] = f"{period_cfg['start']} ~ {period_cfg['end']}"
        metrics["actual_test_dates"] = len(test_dates_in_period)
        metrics["train_period"] = f"{str(s_train_start)[:10]} ~ {str(s_train_end)[:10]}"

        # Compare with baseline
        if baseline_metrics["sharpe_ratio"] != 0:
            metrics["sharpe_vs_baseline"] = round(
                metrics["sharpe_ratio"] - baseline_metrics["sharpe_ratio"], 4
            )
        else:
            metrics["sharpe_vs_baseline"] = None

        stress_results[period_name] = metrics

        logger.info(f"  Result: return={result.total_return * 100:+.2f}% "
                    f"annual={result.annual_return * 100:+.1f}% "
                    f"sharpe={result.sharpe_ratio:.3f} "
                    f"maxdd={result.max_drawdown * 100:.1f}%")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info(f"\n{'=' * 90}")
    logger.info("STRESS TEST SUMMARY")
    logger.info(f"{'=' * 90}")
    logger.info(f"{'Period':<22} {'Return':>8} {'AnnRet':>8} {'Sharpe':>7} {'MaxDD':>7} {'WinRate':>8} {'Days':>5}")
    logger.info("-" * 90)

    # Print baseline first
    bl = baseline_metrics
    logger.info(f"{'BASELINE':<22} {bl['total_return']*100:+.1f}%   {bl['annual_return']*100:+.1f}%   "
                f"{bl['sharpe_ratio']:+.3f}  {bl['max_drawdown']*100:.1f}%  "
                f"{bl['win_rate']*100:.0f}%     {bl['n_days']:>4}")
    logger.info("-" * 90)

    for name, res in stress_results.items():
        if res.get("status") == "skipped":
            logger.info(f"{name:<22} SKIPPED: {res.get('reason', 'unknown')}")
            continue
        logger.info(f"{name:<22} {res['total_return']*100:+.1f}%   {res['annual_return']*100:+.1f}%   "
                    f"{res['sharpe_ratio']:+.3f}  {res['max_drawdown']*100:.1f}%  "
                    f"{res['win_rate']*100:.0f}%     {res['n_days']:>4}")

    # Survival assessment
    logger.info(f"\n{'=' * 50}")
    logger.info("SURVIVAL ASSESSMENT")
    logger.info(f"{'=' * 50}")
    survived_count = 0
    total_count = 0
    for name, res in stress_results.items():
        if res.get("status") == "skipped":
            continue
        total_count += 1
        # Survival criteria: sharpe > 0 (positive risk-adjusted return) or maxdd > -15%
        survived = res["sharpe_ratio"] > 0 or res["max_drawdown"] > -0.15
        if survived:
            survived_count += 1
        status = "SURVIVED" if survived else "DISTRESSED"
        logger.info(f"  {name:<22} {status} (sharpe={res['sharpe_ratio']:.3f}, maxdd={res['max_drawdown']*100:.1f}%)")

    logger.info(f"\n  Overall: {survived_count}/{total_count} periods survived")

    # Save
    output = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "cache": args.cache,
        "top_k": args.top_k,
        "xgb_params": {"nthread": args.nthread, "max_rounds": args.max_rounds},
        "train_days": TRAIN_DAYS,
        "valid_days": VALID_DAYS,
        "baseline": baseline_metrics,
        "stress_periods": stress_results,
        "survival": {
            "survived": survived_count,
            "total": total_count,
            "rate": round(survived_count / max(total_count, 1), 2),
        },
    }

    out_path = DATA_DIR / "phase4" / "stress_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
