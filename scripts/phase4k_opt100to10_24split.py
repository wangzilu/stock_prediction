"""Phase 4K: Full 24-split validation of opt_top100_to10 (the winner config).

Compares against baseline_top20_eq and buffered_partial.

Usage:
    python scripts/phase4k_opt100to10_24split.py
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

N_SPLITS = 24
TEST_DAYS = 20
VALID_DAYS = 60
TRAIN_DAYS = 750

XGB_PARAMS = {
    "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
    "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
    "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": 42,
}

CONFIGS = [
    {"name": "baseline_top20_eq", "mode": "fixed", "top_k": 20, "rebalance_freq": 1},
    {"name": "buffered_partial", "mode": "buffered_partial", "top_k": 20,
     "buffer": 5, "trade_rate": 0.35, "min_hold_days": 2, "max_daily_turnover": 0.15},
    {"name": "opt_top100_to10", "mode": "optimizer_v2", "top_k": 100,
     "max_turnover": 0.10, "weight_method": "alpha_proportional"},
]

EXCLUDE_PREFIXES = ("__", "_", "hsi_", "hstech_", "nasdaq_")
EXCLUDE_EXACT = {"holder_num"}
LABEL_COL = "__label_5d"
PNL_COL = "__pnl_return_1d"


def get_feature_cols(all_cols):
    return [c for c in all_cols
            if not any(c.startswith(p) for p in EXCLUDE_PREFIXES) and c not in EXCLUDE_EXACT]


def run_backtest(predictions, pnl_returns, config, full_index):
    from backtest.cost_model import CostModel
    from backtest.portfolio_backtest import PortfolioBacktest

    cost = CostModel()
    mode = config["mode"]
    top_k = config.get("top_k", 20)

    if mode == "optimizer_v2":
        from backtest.optimizer_v2 import TurnoverConstrainedOptimizer
        optimizer = TurnoverConstrainedOptimizer(
            top_k=top_k,
            max_turnover=config.get("max_turnover", 0.10),
            max_single_weight=0.05,
            weight_method=config.get("weight_method", "alpha_proportional"),
            cost_model=cost,
        )
        bt = PortfolioBacktest(
            top_k=top_k, cost_model=cost, mode="optimizer_v2",
            optimizer=optimizer, min_listing_days=60, min_hold_days=2,
        )
    elif mode == "buffered_partial":
        bt = PortfolioBacktest(
            top_k=top_k, cost_model=cost, mode="buffered_partial",
            buffer=config.get("buffer", 5), trade_rate=config.get("trade_rate", 0.35),
            min_hold_days=config.get("min_hold_days", 2),
            max_daily_turnover=config.get("max_daily_turnover", 0.15),
            min_listing_days=60,
        )
    else:
        bt = PortfolioBacktest(
            top_k=top_k, cost_model=cost, mode="fixed",
            rebalance_freq=config.get("rebalance_freq", 1), min_listing_days=60,
        )

    pred_df = predictions.to_frame("score")
    ret_df = pnl_returns.to_frame("return") if isinstance(pnl_returns, pd.Series) else pnl_returns

    result = bt.run(predictions=pred_df, returns=ret_df,
                    return_horizon_days=1, full_history_index=full_index)

    return {
        "annual_return": float(result.annual_return),
        "sharpe": float(result.sharpe_ratio),
        "max_drawdown": float(result.max_drawdown),
        "avg_turnover": float(result.avg_turnover),
        "cost_to_return": float(result.cost_to_return_ratio),
        "n_days": result.n_days,
    }


def main():
    t_start = time.time()

    logger.info("Loading feature cache...")
    cache = pd.read_parquet(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet")
    feature_cols = get_feature_cols(cache.columns)
    logger.info(f"Cache: {cache.shape}, {len(feature_cols)} features")

    trade_dates = sorted(cache.index.get_level_values(0).unique())
    dates_level = cache.index.get_level_values(0)

    all_results = {cfg["name"]: [] for cfg in CONFIGS}

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

        logger.info(f"\nSplit {split_idx+1}/{N_SPLITS}: test {str(test_start)[:10]}~{str(test_end)[:10]}")

        train_mask = (dates_level >= trade_dates[train_start_idx]) & (dates_level <= trade_dates[train_end_idx])
        valid_mask = (dates_level >= trade_dates[valid_start_idx]) & (dates_level <= trade_dates[valid_end_idx])
        test_mask = (dates_level >= test_start) & (dates_level <= test_end)

        X_tr = cache.loc[train_mask, feature_cols].values.astype(np.float32)
        y_tr = cache.loc[train_mask, LABEL_COL].values.astype(np.float32)
        X_va = cache.loc[valid_mask, feature_cols].values.astype(np.float32)
        y_va = cache.loc[valid_mask, LABEL_COL].values.astype(np.float32)
        X_te = cache.loc[test_mask, feature_cols].values.astype(np.float32)
        y_te = cache.loc[test_mask, LABEL_COL].values.astype(np.float32)
        test_idx = cache.index[test_mask]

        m_tr = np.isfinite(y_tr); X_tr, y_tr = X_tr[m_tr], y_tr[m_tr]
        m_va = np.isfinite(y_va); X_va, y_va = X_va[m_va], y_va[m_va]
        m_te = np.isfinite(y_te)

        dt = xgb.DMatrix(X_tr, label=y_tr)
        dv = xgb.DMatrix(X_va, label=y_va)
        model = xgb.train(XGB_PARAMS, dt, num_boost_round=400,
                          evals=[(dv, "valid")], early_stopping_rounds=50, verbose_eval=0)
        pred = model.predict(xgb.DMatrix(X_te))

        predictions = pd.Series(pred, index=test_idx, name="score")
        predictions = predictions[np.isfinite(predictions)]

        pnl = cache.loc[test_mask, PNL_COL]
        pnl = pnl.loc[test_idx[m_te]].replace([np.inf, -np.inf], np.nan).dropna()

        for cfg in CONFIGS:
            try:
                metrics = run_backtest(predictions, pnl, cfg, cache.index)
                metrics["split"] = split_idx + 1
                all_results[cfg["name"]].append(metrics)
                logger.info(
                    f"  {cfg['name']:<22} ann={metrics['annual_return']*100:+.1f}% "
                    f"shp={metrics['sharpe']:+.3f} to={metrics['avg_turnover']*100:.1f}% "
                    f"cost={metrics['cost_to_return']*100:.0f}%"
                )
            except Exception as e:
                logger.error(f"  {cfg['name']}: FAILED: {e}")

    # Summary
    logger.info(f"\n{'='*110}")
    logger.info(f"PHASE 4K FULL GATE: 24-split validation")
    logger.info(f"{'='*110}")
    logger.info(
        f"{'Config':<24} {'AvgAnn':>8} {'MedAnn':>8} {'W3Ann':>8} {'AvgShp':>7} {'MedShp':>7} "
        f"{'AvgTO':>7} {'CostDr':>7} {'Pos%':>5} {'#Spl':>5}"
    )
    logger.info("-" * 110)

    summary = {}
    for cfg in CONFIGS:
        name = cfg["name"]
        splits = all_results[name]
        if not splits:
            continue
        annuals = [s["annual_return"] for s in splits]
        sharpes = [s["sharpe"] for s in splits]
        turnovers = [s["avg_turnover"] for s in splits]
        costs = [s["cost_to_return"] for s in splits]
        dds = [s["max_drawdown"] for s in splits]

        # Worst-3 average
        sorted_ann = sorted(annuals)
        worst3_ann = np.mean(sorted_ann[:3])
        sorted_shp = sorted(sharpes)
        worst3_shp = np.mean(sorted_shp[:3])

        pos_pct = sum(1 for a in annuals if a > 0) / len(annuals)

        summary[name] = {
            "avg_annual": round(float(np.mean(annuals)) * 100, 2),
            "med_annual": round(float(np.median(annuals)) * 100, 2),
            "worst3_annual": round(float(worst3_ann) * 100, 2),
            "avg_sharpe": round(float(np.mean(sharpes)), 3),
            "med_sharpe": round(float(np.median(sharpes)), 3),
            "worst3_sharpe": round(float(worst3_shp), 3),
            "avg_turnover": round(float(np.mean(turnovers)) * 100, 1),
            "avg_cost_drag": round(float(np.mean(costs)) * 100, 1),
            "avg_maxdd": round(float(np.mean(dds)) * 100, 1),
            "positive_pct": round(pos_pct * 100, 0),
            "n_splits": len(splits),
            "per_split_annual": [round(a * 100, 1) for a in annuals],
            "per_split_sharpe": [round(s, 3) for s in sharpes],
        }

        logger.info(
            f"{name:<24} {np.mean(annuals)*100:>+7.1f}% {np.median(annuals)*100:>+7.1f}% "
            f"{worst3_ann*100:>+7.1f}% {np.mean(sharpes):>+7.3f} {np.median(sharpes):>+7.3f} "
            f"{np.mean(turnovers)*100:>6.1f}% {np.mean(costs)*100:>6.1f}% "
            f"{pos_pct*100:>4.0f}% {len(splits):>5}"
        )

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    output = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "n_splits": N_SPLITS,
        "summary": summary,
    }
    out_path = OUTPUT_DIR / "opt100to10_24split_gate.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                   default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o)))
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
