"""Phase 4K: Compare optimizer_v2 vs baseline across turnover/cost/net-excess.

Grid search over max_turnover and top_k to find optimal trade-off.

Usage:
    python scripts/phase4k_optimizer_backtest.py
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
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
OUTPUT_DIR = DATA_DIR / "phase4k"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Rolling config
N_SPLITS = 12  # 12 splits for speed (vs 24 for final gate)
TEST_DAYS = 20
VALID_DAYS = 60
TRAIN_DAYS = 750

XGB_PARAMS = {
    "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
    "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
    "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": 42,
}

# Configs to compare
OPTIMIZER_CONFIGS = [
    # Baseline: equal-weight top-20 (current champion)
    {"name": "baseline_top20_eq", "mode": "fixed", "top_k": 20, "rebalance_freq": 1},
    # Buffered partial (current best)
    {"name": "buffered_partial", "mode": "buffered_partial", "top_k": 20,
     "buffer": 5, "trade_rate": 0.35, "min_hold_days": 2, "max_daily_turnover": 0.15},
    # Optimizer V2: various turnover caps and portfolio sizes
    {"name": "opt_top50_to20", "mode": "optimizer_v2", "top_k": 50,
     "max_turnover": 0.20, "weight_method": "alpha_proportional"},
    {"name": "opt_top100_to20", "mode": "optimizer_v2", "top_k": 100,
     "max_turnover": 0.20, "weight_method": "alpha_proportional"},
    {"name": "opt_top100_to10", "mode": "optimizer_v2", "top_k": 100,
     "max_turnover": 0.10, "weight_method": "alpha_proportional"},
    {"name": "opt_top100_to30", "mode": "optimizer_v2", "top_k": 100,
     "max_turnover": 0.30, "weight_method": "alpha_proportional"},
    {"name": "opt_top100_eq_to20", "mode": "optimizer_v2", "top_k": 100,
     "max_turnover": 0.20, "weight_method": "equal"},
    {"name": "opt_top200_to20", "mode": "optimizer_v2", "top_k": 200,
     "max_turnover": 0.20, "weight_method": "alpha_proportional"},
]

EXCLUDE_PREFIXES = ("__", "_", "hsi_", "hstech_", "nasdaq_")
EXCLUDE_EXACT = {"holder_num"}
LABEL_COL = "__label_5d"
PNL_COL = "__pnl_return_1d"


def get_feature_cols(all_cols):
    result = []
    for c in all_cols:
        if any(c.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if c in EXCLUDE_EXACT:
            continue
        result.append(c)
    return result


def run_portfolio_metrics(predictions: pd.Series, pnl_returns: pd.Series,
                          config: dict, full_index: pd.MultiIndex) -> dict:
    """Run backtest for a given config and return metrics."""
    from backtest.cost_model import CostModel
    from backtest.portfolio_backtest import PortfolioBacktest

    cost = CostModel()
    mode = config["mode"]
    top_k = config.get("top_k", 20)

    if mode == "optimizer_v2":
        from backtest.optimizer_v2 import TurnoverConstrainedOptimizer
        optimizer = TurnoverConstrainedOptimizer(
            top_k=top_k,
            max_turnover=config.get("max_turnover", 0.20),
            max_single_weight=config.get("max_single_weight", 0.05),
            weight_method=config.get("weight_method", "alpha_proportional"),
            cost_model=cost,
        )
        bt = PortfolioBacktest(
            top_k=top_k, cost_model=cost, mode="optimizer_v2",
            optimizer=optimizer, min_listing_days=60,
            min_hold_days=config.get("min_hold_days", 2),
        )
    elif mode == "buffered_partial":
        bt = PortfolioBacktest(
            top_k=top_k, cost_model=cost, mode="buffered_partial",
            buffer=config.get("buffer", 5),
            trade_rate=config.get("trade_rate", 0.35),
            min_hold_days=config.get("min_hold_days", 2),
            max_daily_turnover=config.get("max_daily_turnover", 0.15),
            min_listing_days=60,
        )
    else:
        bt = PortfolioBacktest(
            top_k=top_k, cost_model=cost, mode="fixed",
            rebalance_freq=config.get("rebalance_freq", 1),
            min_listing_days=60,
        )

    pred_df = predictions.to_frame("score")
    if isinstance(pnl_returns, pd.Series):
        ret_df = pnl_returns.to_frame("return")
    else:
        ret_df = pnl_returns

    result = bt.run(
        predictions=pred_df,
        returns=ret_df,
        return_horizon_days=1,
        full_history_index=full_index,
    )

    return {
        "annual_return": float(result.annual_return),
        "sharpe": float(result.sharpe_ratio),
        "max_drawdown": float(result.max_drawdown),
        "avg_turnover": float(result.avg_turnover),
        "cost_to_return": float(result.cost_to_return_ratio),
        "n_days": result.n_days,
        "avg_holdings": float(np.mean(result.daily_holdings_count)) if hasattr(result, 'daily_holdings_count') and result.daily_holdings_count else top_k,
    }


def main():
    t_start = time.time()

    # Load data
    logger.info("Loading feature cache...")
    cache = pd.read_parquet(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet")
    feature_cols = get_feature_cols(cache.columns)
    logger.info(f"Cache: {cache.shape}, {len(feature_cols)} features")

    trade_dates = sorted(cache.index.get_level_values(0).unique())
    dates_level = cache.index.get_level_values(0)

    all_results = {cfg["name"]: [] for cfg in OPTIMIZER_CONFIGS}

    for split_idx in range(N_SPLITS):
        test_end_idx = len(trade_dates) - 1 - split_idx * TEST_DAYS
        test_start_idx = test_end_idx - TEST_DAYS
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - VALID_DAYS
        train_end_idx = valid_start_idx - 1
        train_start_idx = max(0, train_end_idx - TRAIN_DAYS)

        if train_start_idx >= train_end_idx:
            break

        test_start = trade_dates[test_start_idx]
        test_end = trade_dates[test_end_idx]
        valid_start = trade_dates[valid_start_idx]
        train_start = trade_dates[train_start_idx]
        train_end = trade_dates[train_end_idx]

        logger.info(f"\nSplit {split_idx+1}/{N_SPLITS}: test {str(test_start)[:10]}~{str(test_end)[:10]}")

        # Slice data
        train_mask = (dates_level >= train_start) & (dates_level <= train_end)
        valid_mask = (dates_level >= valid_start) & (dates_level <= trade_dates[valid_end_idx])
        test_mask = (dates_level >= test_start) & (dates_level <= test_end)

        X_tr = cache.loc[train_mask, feature_cols].values.astype(np.float32)
        y_tr = cache.loc[train_mask, LABEL_COL].values.astype(np.float32)
        X_va = cache.loc[valid_mask, feature_cols].values.astype(np.float32)
        y_va = cache.loc[valid_mask, LABEL_COL].values.astype(np.float32)
        X_te = cache.loc[test_mask, feature_cols].values.astype(np.float32)
        y_te = cache.loc[test_mask, LABEL_COL].values.astype(np.float32)
        test_idx = cache.index[test_mask]

        # Filter NaN labels
        m_tr = np.isfinite(y_tr); X_tr, y_tr = X_tr[m_tr], y_tr[m_tr]
        m_va = np.isfinite(y_va); X_va, y_va = X_va[m_va], y_va[m_va]
        m_te = np.isfinite(y_te); X_te, y_te = X_te[m_te], y_te[m_te]
        test_idx = test_idx[m_te]

        if len(X_tr) < 100 or len(X_te) < 100:
            continue

        # Train XGB
        t1 = time.time()
        dt = xgb.DMatrix(X_tr, label=y_tr)
        dv = xgb.DMatrix(X_va, label=y_va)
        model = xgb.train(XGB_PARAMS, dt, num_boost_round=400,
                          evals=[(dv, "valid")], early_stopping_rounds=50,
                          verbose_eval=0)
        pred = model.predict(xgb.DMatrix(X_te))
        train_time = time.time() - t1

        predictions = pd.Series(pred, index=test_idx, name="score")
        predictions = predictions[np.isfinite(predictions)]

        # Get PnL returns
        pnl = cache.loc[test_mask, PNL_COL]
        pnl = pnl.loc[test_idx[np.isfinite(y_te)]].replace([np.inf, -np.inf], np.nan).dropna()

        # Run each config
        for cfg in OPTIMIZER_CONFIGS:
            name = cfg["name"]
            try:
                t2 = time.time()
                metrics = run_portfolio_metrics(predictions, pnl, cfg, cache.index)
                elapsed = time.time() - t2
                metrics["split"] = split_idx + 1
                all_results[name].append(metrics)
                logger.info(
                    f"  {name:<25} annual={metrics['annual_return']*100:+.1f}% "
                    f"sharpe={metrics['sharpe']:+.3f} "
                    f"turnover={metrics['avg_turnover']*100:.1f}% "
                    f"cost/ret={metrics['cost_to_return']*100:.0f}% "
                    f"({elapsed:.1f}s)"
                )
            except Exception as e:
                logger.error(f"  {name}: FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    logger.info(f"\n{'='*110}")
    logger.info(f"PHASE 4K OPTIMIZER COMPARISON: {N_SPLITS} splits")
    logger.info(f"{'='*110}")
    logger.info(
        f"{'Config':<28} {'AvgAnn':>8} {'MedAnn':>8} {'AvgShp':>7} {'AvgTO':>7} "
        f"{'CostDrag':>9} {'AvgDD':>7} {'Pos%':>5} {'#Split':>6}"
    )
    logger.info("-" * 110)

    summary = {}
    for cfg in OPTIMIZER_CONFIGS:
        name = cfg["name"]
        splits = all_results[name]
        if not splits:
            continue
        annuals = [s["annual_return"] for s in splits]
        sharpes = [s["sharpe"] for s in splits]
        turnovers = [s["avg_turnover"] for s in splits]
        costs = [s["cost_to_return"] for s in splits]
        dds = [s["max_drawdown"] for s in splits]
        pos_pct = sum(1 for a in annuals if a > 0) / len(annuals)

        summary[name] = {
            "avg_annual": round(float(np.mean(annuals)) * 100, 2),
            "med_annual": round(float(np.median(annuals)) * 100, 2),
            "avg_sharpe": round(float(np.mean(sharpes)), 3),
            "avg_turnover": round(float(np.mean(turnovers)) * 100, 1),
            "avg_cost_drag": round(float(np.mean(costs)) * 100, 1),
            "avg_maxdd": round(float(np.mean(dds)) * 100, 1),
            "positive_split_pct": round(pos_pct * 100, 0),
            "n_splits": len(splits),
            "per_split": splits,
        }

        logger.info(
            f"{name:<28} {np.mean(annuals)*100:>+7.1f}% {np.median(annuals)*100:>+7.1f}% "
            f"{np.mean(sharpes):>+7.3f} {np.mean(turnovers)*100:>6.1f}% "
            f"{np.mean(costs)*100:>8.1f}% {np.mean(dds)*100:>6.1f}% "
            f"{pos_pct*100:>4.0f}% {len(splits):>6}"
        )

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save
    output = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "n_splits": N_SPLITS,
        "configs": [cfg["name"] for cfg in OPTIMIZER_CONFIGS],
        "summary": summary,
    }
    out_path = OUTPUT_DIR / "optimizer_comparison.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                   default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o)))
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
