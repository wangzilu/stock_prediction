"""Track B upgraded rolling backtest: open-to-open + IPO filter + frozen valuation.

Compares 3 configs across 12 splits using feature cache (no Qlib dataset rebuild).
Upgrades vs previous run (rolling_backtest_configs.json):
  - Open-to-open execution (not close-to-close)
  - IPO filter: min_listing_days=60
  - Suspended stock frozen valuation (already in PortfolioBacktest)

Usage:
    python scripts/rerun_track_b_upgraded.py
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

from config.qlib_runtime import init_qlib
from backtest.cost_model import CostModel
from backtest.portfolio_backtest import PortfolioBacktest
from models.feature_pipeline import load_daily_returns
from utils.json_utils import json_default

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
SEED = 42

# --- Backtest configs ---
CONFIGS = [
    {
        "name": "daily_rebal",
        "bt_kwargs": {
            "mode": "fixed",
            "rebalance_freq": 1,
            "min_listing_days": 60,
        },
    },
    {
        "name": "buffered_partial",
        "bt_kwargs": {
            "mode": "buffered_partial",
            "rebalance_freq": 1,
            "min_listing_days": 60,
            "buffer": 5,
            "trade_rate": 0.35,
            "min_hold_days": 2,
            "max_daily_turnover": 0.15,
        },
    },
    {
        "name": "buffered+stop8%",
        "bt_kwargs": {
            "mode": "buffered_partial",
            "rebalance_freq": 1,
            "min_listing_days": 60,
            "buffer": 5,
            "trade_rate": 0.35,
            "min_hold_days": 2,
            "max_daily_turnover": 0.15,
            "drawdown_stop": 0.08,
        },
    },
]

# --- Rolling params ---
N_SPLITS = 12
TEST_DAYS = 40
TRAIN_DAYS = 750
VALID_DAYS = 60


def train_xgb(X_train, y_train, X_valid, y_valid, nthread=12, max_rounds=400):
    import xgboost as xgb
    params = {
        "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
        "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
        "objective": "reg:squarederror", "nthread": nthread, "verbosity": 0, "seed": SEED,
    }
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_valid, label=y_valid)
    model = xgb.train(params, dt, num_boost_round=max_rounds,
                      evals=[(dv, "valid")], early_stopping_rounds=30, verbose_eval=0)
    return model


def get_base_feature_cols(all_cols):
    """Exclude meta (__), underscore-only, and cross-market prefixed columns."""
    base = []
    cross_prefixes = ("hsi_", "hstech_", "nasdaq_")
    for c in all_cols:
        if c.startswith("__"):
            continue
        if c.startswith("_"):
            continue
        if any(c.startswith(p) for p in cross_prefixes):
            continue
        base.append(c)
    return base


def main():
    import xgboost as xgb

    init_qlib(QLIB_DATA)

    # 1. Load feature cache
    cache_path = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
    logger.info(f"Loading cache: {cache_path}")
    t0 = time.time()
    cache = pd.read_parquet(str(cache_path))
    logger.info(f"  Loaded: {cache.shape}, {time.time()-t0:.1f}s")

    # Separate features and labels
    label_col = "__label_5d"
    all_cols = list(cache.columns)
    feature_cols = get_base_feature_cols(all_cols)
    logger.info(f"  Base feature cols: {len(feature_cols)}")

    if label_col not in cache.columns:
        logger.error(f"Label column {label_col} not found in cache")
        sys.exit(1)

    # Get trading dates
    trade_dates = sorted(cache.index.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1
    dates_level = cache.index.get_level_values(0)
    logger.info(f"  Trading dates: {len(trade_dates)}")

    # 2. Load old results for comparison
    old_results_path = DATA_DIR / "rolling_backtest_configs.json"
    old_results = None
    if old_results_path.exists():
        with open(str(old_results_path)) as f:
            old_results = json.load(f)
        logger.info(f"  Loaded old results: {old_results_path}")

    cost = CostModel()
    all_split_results = []
    t_total = time.time()

    for split_idx in range(N_SPLITS):
        test_end_idx = today_idx - split_idx * TEST_DAYS
        test_start_idx = test_end_idx - TEST_DAYS
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - VALID_DAYS
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - TRAIN_DAYS

        if train_start_idx < 0:
            logger.warning(f"  Split {split_idx+1}: not enough data, stopping")
            break

        test_end = trade_dates[test_end_idx]
        test_start = trade_dates[test_start_idx]
        valid_start = trade_dates[valid_start_idx]
        valid_end = trade_dates[valid_end_idx]
        train_start = trade_dates[train_start_idx]
        train_end = trade_dates[train_end_idx]

        logger.info(f"\nSplit {split_idx+1}/{N_SPLITS}: "
                    f"test {str(test_start)[:10]}~{str(test_end)[:10]}")

        try:
            # Slice data
            train_mask = (dates_level >= train_start) & (dates_level <= train_end)
            valid_mask = (dates_level >= valid_start) & (dates_level <= valid_end)
            test_mask = (dates_level >= test_start) & (dates_level <= test_end)

            X_train = cache.loc[train_mask, feature_cols].values.astype(np.float32)
            y_train = cache.loc[train_mask, label_col].values.astype(np.float32)
            X_valid = cache.loc[valid_mask, feature_cols].values.astype(np.float32)
            y_valid = cache.loc[valid_mask, label_col].values.astype(np.float32)
            X_test = cache.loc[test_mask, feature_cols].values.astype(np.float32)
            y_test = cache.loc[test_mask, label_col].values.astype(np.float32)
            test_idx = cache.index[test_mask]

            mask_tr = np.isfinite(y_train)
            mask_va = np.isfinite(y_valid)
            mask_te = np.isfinite(y_test)

            # Train XGB
            t1 = time.time()
            model = train_xgb(X_train[mask_tr], y_train[mask_tr],
                              X_valid[mask_va], y_valid[mask_va])
            train_time = time.time() - t1

            # Predict on test
            pred = model.predict(xgb.DMatrix(X_test[mask_te]))
            predictions = pd.Series(pred, index=test_idx[mask_te], name="score")
            predictions = predictions[np.isfinite(predictions)]

            # Load daily returns with OPEN execution
            daily_returns = load_daily_returns(test_idx, execution_price="open")
            if isinstance(daily_returns, pd.DataFrame):
                daily_returns = daily_returns.rename(columns={"pnl_return_1d": "return"})

            # Run each backtest config
            split_result = {
                "split": split_idx + 1,
                "test": f"{str(test_start)[:10]}~{str(test_end)[:10]}",
                "train_time_s": round(train_time, 1),
            }

            for cfg in CONFIGS:
                bt = PortfolioBacktest(
                    top_k=20,
                    cost_model=cost,
                    **cfg["bt_kwargs"],
                )
                r = bt.run(
                    predictions=predictions.to_frame("score"),
                    returns=daily_returns,
                    return_horizon_days=1,
                    full_history_index=cache.index,
                )

                split_result[cfg["name"]] = {
                    "annual": round(r.annual_return * 100, 2),
                    "sharpe": round(r.sharpe_ratio, 3),
                    "maxdd": round(r.max_drawdown * 100, 2),
                    "turnover": round(r.avg_turnover * 100, 1),
                    "cost_ratio": round(r.cost_to_return_ratio * 100, 1),
                    "ipo_filtered": r.ipo_filtered_count,
                    "suspended_days": r.suspended_days,
                    "n_days": r.n_days,
                }
                logger.info(
                    f"  {cfg['name']:<22} annual={r.annual_return*100:+.1f}% "
                    f"sharpe={r.sharpe_ratio:+.3f} dd={r.max_drawdown*100:.1f}% "
                    f"ipo_filt={r.ipo_filtered_count} susp={r.suspended_days}"
                )

            all_split_results.append(split_result)

        except Exception as e:
            logger.error(f"  Split {split_idx+1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - t_total
    n = len(all_split_results)

    if n == 0:
        logger.error("No valid splits completed")
        sys.exit(1)

    # --- Summary ---
    logger.info(f"\n{'='*90}")
    logger.info(f"TRACK B UPGRADED: {n} splits x {TEST_DAYS} days  |  open-to-open + IPO60 + frozen valuation")
    logger.info(f"{'='*90}")

    summary = {}
    for cfg in CONFIGS:
        name = cfg["name"]
        annuals = [r[name]["annual"] for r in all_split_results if name in r]
        sharpes = [r[name]["sharpe"] for r in all_split_results if name in r]
        maxdds = [r[name]["maxdd"] for r in all_split_results if name in r]
        pos_pct = sum(1 for a in annuals if a > 0) / len(annuals) if annuals else 0

        summary[name] = {
            "avg_annual": round(float(np.mean(annuals)), 2),
            "avg_sharpe": round(float(np.mean(sharpes)), 3),
            "avg_maxdd": round(float(np.mean(maxdds)), 2),
            "positive_splits": f"{sum(1 for a in annuals if a > 0)}/{len(annuals)}",
            "per_split_annual": [round(a, 1) for a in annuals],
            "per_split_sharpe": [round(s, 3) for s in sharpes],
        }

        logger.info(f"\n  {name}:")
        logger.info(f"    avg annual:  {np.mean(annuals):+.1f}%")
        logger.info(f"    avg sharpe:  {np.mean(sharpes):+.3f}")
        logger.info(f"    avg maxdd:   {np.mean(maxdds):.1f}%")
        logger.info(f"    annual>0:    {pos_pct:.0%} ({sum(1 for a in annuals if a > 0)}/{len(annuals)})")
        logger.info(f"    per-split:   {['%+.1f%%' % a for a in annuals]}")

    # --- Comparison with old close-to-close results ---
    if old_results and "splits" in old_results:
        logger.info(f"\n{'='*90}")
        logger.info("COMPARISON: old (close-to-close, no IPO) vs new (open-to-open, IPO60)")
        logger.info(f"{'='*90}")

        # Map old config names to new names
        old_name_map = {
            "daily": "daily_rebal",
            "buffered_partial": "buffered_partial",
            "buffered+stop8%": "buffered+stop8%",
        }

        comparison = {}
        for old_name, new_name in old_name_map.items():
            old_annuals = [s[old_name]["annual"] for s in old_results["splits"]
                          if old_name in s]
            old_sharpes = [s[old_name]["sharpe"] for s in old_results["splits"]
                          if old_name in s]

            new_annuals = summary.get(new_name, {}).get("per_split_annual", [])
            new_sharpes = summary.get(new_name, {}).get("per_split_sharpe", [])

            old_avg_ann = float(np.mean(old_annuals)) if old_annuals else 0
            new_avg_ann = float(np.mean(new_annuals)) if new_annuals else 0
            old_avg_sh = float(np.mean(old_sharpes)) if old_sharpes else 0
            new_avg_sh = float(np.mean(new_sharpes)) if new_sharpes else 0

            comparison[new_name] = {
                "old_avg_annual": round(old_avg_ann, 2),
                "new_avg_annual": round(new_avg_ann, 2),
                "delta_annual": round(new_avg_ann - old_avg_ann, 2),
                "old_avg_sharpe": round(old_avg_sh, 3),
                "new_avg_sharpe": round(new_avg_sh, 3),
                "delta_sharpe": round(new_avg_sh - old_avg_sh, 3),
            }

            logger.info(f"\n  {new_name}:")
            logger.info(f"    OLD avg annual: {old_avg_ann:+.1f}%  sharpe: {old_avg_sh:+.3f}")
            logger.info(f"    NEW avg annual: {new_avg_ann:+.1f}%  sharpe: {new_avg_sh:+.3f}")
            logger.info(f"    DELTA annual:   {new_avg_ann - old_avg_ann:+.1f}%  "
                        f"sharpe: {new_avg_sh - old_avg_sh:+.3f}")

    # --- Save results ---
    out_path = DATA_DIR / "phase4" / "track_b_upgraded.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "description": "Track B upgraded: open-to-open execution + IPO filter (60d) + frozen valuation",
        "execution_price": "open",
        "min_listing_days": 60,
        "n_splits": n,
        "test_days": TEST_DAYS,
        "train_days": TRAIN_DAYS,
        "valid_days": VALID_DAYS,
        "total_time_s": round(total_time, 1),
        "summary": summary,
        "comparison": comparison if old_results else None,
        "splits": all_split_results,
    }

    with open(str(out_path), "w") as f:
        json.dump(output, f, indent=2, default=json_default)

    logger.info(f"\nSaved: {out_path}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info("Done!")


if __name__ == "__main__":
    main()
