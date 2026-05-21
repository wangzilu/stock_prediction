"""24-split gate with tradable mask: train + evaluate on filtered universe only.

Compares: baseline (no filter) vs filtered (ST/IPO/suspended/一字板/liquidity removed)
Both use opt_top100_to10 execution.

Usage:
    python scripts/phase4k_filtered_24split.py
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

EXCLUDE_PREFIXES = ("__", "_", "hsi_", "hstech_", "nasdaq_")
EXCLUDE_EXACT = {"holder_num"}
LABEL_COL = "__label_5d"
PNL_COL = "__pnl_return_1d"


def get_feature_cols(all_cols):
    return [c for c in all_cols
            if not any(c.startswith(p) for p in EXCLUDE_PREFIXES) and c not in EXCLUDE_EXACT]


def run_backtest(predictions, pnl_returns, full_index):
    from backtest.cost_model import CostModel
    from backtest.portfolio_backtest import PortfolioBacktest
    from backtest.optimizer_v2 import TurnoverConstrainedOptimizer

    cost = CostModel()
    optimizer = TurnoverConstrainedOptimizer(
        top_k=100, max_turnover=0.10, max_single_weight=0.05,
        weight_method="alpha_proportional", cost_model=cost,
    )
    bt = PortfolioBacktest(
        top_k=100, cost_model=cost, mode="optimizer_v2",
        optimizer=optimizer, min_listing_days=60, min_hold_days=2,
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


def evaluate_signal(pred, label, index):
    mask = np.isfinite(pred) & np.isfinite(label)
    if mask.sum() < 200:
        return None
    pred_s = pd.Series(pred[mask], index=index[mask])
    label_s = pd.Series(label[mask], index=index[mask])
    ric_list, spreads = [], []
    for dt, g in pred_s.groupby(level=0):
        gl = label_s.reindex(g.index).dropna()
        common = g.index.intersection(gl.index)
        if len(common) < 40:
            continue
        p, l = g.loc[common].values, gl.loc[common].values
        ric = stats.spearmanr(p, l).statistic
        if np.isfinite(ric):
            ric_list.append(ric)
        tmp = pd.DataFrame({"p": p, "l": l}).sort_values("p", ascending=False)
        spreads.append(tmp.head(20)["l"].mean() - tmp.tail(20)["l"].mean())
    if not ric_list:
        return None
    return {
        "rank_ic": float(np.mean(ric_list)),
        "rank_ic_pos": float(np.mean([r > 0 for r in ric_list])),
        "spread_top20": float(np.mean(spreads)) if spreads else 0,
    }


def main():
    t_start = time.time()

    # Load data
    logger.info("Loading feature cache...")
    cache = pd.read_parquet(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet")
    feature_cols = get_feature_cols(cache.columns)
    logger.info(f"Cache: {cache.shape}, {len(feature_cols)} features")

    # Load tradable mask
    logger.info("Loading tradable mask...")
    mask_df = pd.read_parquet(DATA_DIR / "tradable_mask.parquet")
    tradable = mask_df["tradable"].values
    logger.info(f"Tradable: {tradable.sum():,} / {len(tradable):,} ({tradable.mean()*100:.1f}%)")

    # Load winsorized labels
    label_win_df = pd.read_parquet(DATA_DIR / "label_5d_winsorized.parquet")
    label_win = label_win_df["label_5d_win"]

    trade_dates = sorted(cache.index.get_level_values(0).unique())
    dates_level = cache.index.get_level_values(0)

    CONFIGS = [
        {"name": "unfiltered", "use_mask": False, "use_winsorized_label": False},
        {"name": "filtered", "use_mask": True, "use_winsorized_label": False},
        {"name": "filtered+winlabel", "use_mask": True, "use_winsorized_label": True},
    ]

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

        for cfg in CONFIGS:
            name = cfg["name"]
            t1 = time.time()

            try:
                use_mask = cfg["use_mask"]
                use_win = cfg["use_winsorized_label"]

                # Get labels
                if use_win:
                    y_col = label_win
                else:
                    y_col = cache[LABEL_COL]

                # Training data
                if use_mask:
                    tr_sel = train_mask & tradable
                    va_sel = valid_mask & tradable
                else:
                    tr_sel = train_mask
                    va_sel = valid_mask

                X_tr = cache.loc[tr_sel, feature_cols].values.astype(np.float32)
                y_tr = y_col.loc[tr_sel].values.astype(np.float32)
                X_va = cache.loc[va_sel, feature_cols].values.astype(np.float32)
                y_va = y_col.loc[va_sel].values.astype(np.float32)

                # Test data: always use FULL test set for fair comparison
                # (we filter at training, but evaluate on same universe)
                X_te = cache.loc[test_mask, feature_cols].values.astype(np.float32)
                y_te = cache.loc[test_mask, LABEL_COL].values.astype(np.float32)  # always raw label for eval
                test_idx = cache.index[test_mask]

                # NaN filter
                m_tr = np.isfinite(y_tr); X_tr, y_tr = X_tr[m_tr], y_tr[m_tr]
                m_va = np.isfinite(y_va); X_va, y_va = X_va[m_va], y_va[m_va]
                m_te = np.isfinite(y_te); X_te_c, y_te_c = X_te[m_te], y_te[m_te]
                test_idx_c = test_idx[m_te]

                if len(X_tr) < 100:
                    continue

                # Train
                dt = xgb.DMatrix(X_tr, label=y_tr)
                dv = xgb.DMatrix(X_va, label=y_va)
                model = xgb.train(XGB_PARAMS, dt, num_boost_round=400,
                                  evals=[(dv, "valid")], early_stopping_rounds=50, verbose_eval=0)
                pred = model.predict(xgb.DMatrix(X_te_c))

                # Signal metrics
                signal = evaluate_signal(pred, y_te_c, test_idx_c)

                # Portfolio metrics
                predictions = pd.Series(pred, index=test_idx_c)
                predictions = predictions[np.isfinite(predictions)]
                pnl = cache.loc[test_mask, PNL_COL].loc[test_idx_c].replace([np.inf, -np.inf], np.nan).dropna()
                portfolio = run_backtest(predictions, pnl, cache.index)

                elapsed = time.time() - t1
                metrics = {
                    "split": split_idx + 1,
                    "train_samples": len(X_tr),
                    **(signal or {}),
                    **portfolio,
                }
                all_results[name].append(metrics)

                ric = metrics.get("rank_ic", 0)
                shp = metrics.get("sharpe", 0)
                logger.info(
                    f"  {name:<22} RIC={ric:+.4f} Shp={shp:+.3f} "
                    f"Ann={metrics.get('annual_return',0)*100:+.1f}% "
                    f"Train={len(X_tr):,} ({elapsed:.1f}s)"
                )

            except Exception as e:
                logger.error(f"  {name}: FAILED: {e}")
                import traceback; traceback.print_exc()

    # Summary
    logger.info(f"\n{'='*110}")
    logger.info(f"FILTERED vs UNFILTERED: 24-split gate (opt_top100_to10)")
    logger.info(f"{'='*110}")
    logger.info(
        f"{'Config':<24} {'AvgRIC':>8} {'MedRIC':>8} {'W3RIC':>8} {'AvgShp':>7} {'MedShp':>7} "
        f"{'AvgTO':>7} {'CostDr':>7} {'Pos%':>5} {'#Spl':>5}"
    )
    logger.info("-" * 110)

    summary = {}
    for cfg in CONFIGS:
        name = cfg["name"]
        splits = all_results[name]
        if not splits:
            continue
        rics = [s.get("rank_ic", 0) for s in splits]
        sharpes = [s.get("sharpe", 0) for s in splits]
        turnovers = [s.get("avg_turnover", 0) for s in splits]
        costs = [s.get("cost_to_return", 0) for s in splits]
        annuals = [s.get("annual_return", 0) for s in splits]
        sorted_ric = sorted(rics)
        worst3 = np.mean(sorted_ric[:3])
        pos_pct = sum(1 for a in annuals if a > 0) / len(annuals)

        summary[name] = {
            "avg_rank_ic": round(float(np.mean(rics)), 6),
            "med_rank_ic": round(float(np.median(rics)), 6),
            "worst3_rank_ic": round(float(worst3), 6),
            "avg_sharpe": round(float(np.mean(sharpes)), 3),
            "med_sharpe": round(float(np.median(sharpes)), 3),
            "avg_turnover": round(float(np.mean(turnovers)) * 100, 1),
            "avg_cost_drag": round(float(np.mean(costs)) * 100, 1),
            "avg_annual": round(float(np.mean(annuals)) * 100, 2),
            "med_annual": round(float(np.median(annuals)) * 100, 2),
            "positive_pct": round(pos_pct * 100, 0),
            "n_splits": len(splits),
        }

        logger.info(
            f"{name:<24} {np.mean(rics):>+8.4f} {np.median(rics):>+8.4f} {worst3:>+8.4f} "
            f"{np.mean(sharpes):>+7.3f} {np.median(sharpes):>+7.3f} "
            f"{np.mean(turnovers)*100:>6.1f}% {np.mean(costs)*100:>6.1f}% "
            f"{pos_pct*100:>4.0f}% {len(splits):>5}"
        )

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    output = {"evaluated_at": datetime.now().isoformat(timespec="seconds"),
              "n_splits": N_SPLITS, "summary": summary,
              "per_split": {k: v for k, v in all_results.items()}}
    out_path = OUTPUT_DIR / "filtered_vs_unfiltered_24split.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                   default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o)))
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
