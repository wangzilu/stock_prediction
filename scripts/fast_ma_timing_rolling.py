"""Fast rolling validation of MA timing strategy using feature cache.

Tests XGB selection + 5MA/20MA entry/exit across 12 splits.

Prerequisite: python scripts/build_feature_cache.py --all

Usage:
    python scripts/fast_ma_timing_rolling.py
"""
import argparse
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

from backtest.cost_model import CostModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
SEED = 42


def train_xgb(X_train, y_train, X_valid, y_valid):
    import xgboost as xgb
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_valid, label=y_valid)
    params = {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
              "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
              "objective": "reg:squarederror", "nthread": 4, "verbosity": 0, "seed": SEED}
    model = xgb.train(params, dt, num_boost_round=500,
                      evals=[(dv, "valid")], early_stopping_rounds=50, verbose_eval=0)
    return model


def run_ma_timing(scores, daily_ret, ma_data, top_k=50, max_holdings=20,
                  ma_entry_pct=0.02, ma_exit_above=0.08, take_profit=0.20,
                  cost=None):
    """Run MA timing on one split's test data."""
    cost = cost or CostModel()
    dates = sorted(scores.index.get_level_values(0).unique())

    holdings = {}  # {stock: entry_price}
    daily_pnl = []
    total_trades = 0

    for i, date in enumerate(dates):
        if date not in scores.index.get_level_values(0):
            continue

        day_scores = scores.loc[date]
        day_ma = ma_data.loc[date] if date in ma_data.index.get_level_values(0) else pd.DataFrame()
        if isinstance(day_ma, pd.Series):
            day_ma = day_ma.to_frame().T

        # Get next day return
        if i + 1 < len(dates):
            next_date = dates[i + 1]
            if next_date in daily_ret.index.get_level_values(0):
                day_ret = daily_ret.loc[next_date]
                if isinstance(day_ret, pd.DataFrame):
                    day_ret = day_ret.iloc[:, 0]
            else:
                day_ret = pd.Series(dtype=float)
        else:
            day_ret = pd.Series(dtype=float)

        # SELL decisions
        sells = []
        for stock, entry_price in list(holdings.items()):
            if stock not in day_ma.index:
                continue
            try:
                close = float(day_ma.loc[stock, "_close"])
                ma5 = float(day_ma.loc[stock, "_ma5"])
                ma20 = float(day_ma.loc[stock, "_ma20"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isnan(close) or np.isnan(ma20):
                continue

            gain = (close - entry_price) / entry_price
            if close < ma20:  # stop loss
                sells.append(stock)
            elif gain >= take_profit:  # take profit
                sells.append(stock)
            elif not np.isnan(ma5) and ma5 > 0 and (close - ma5) / ma5 > ma_exit_above:
                sells.append(stock)  # overbought

        for s in sells:
            del holdings[s]
        total_trades += len(sells)

        # BUY decisions
        buys = 0
        if len(holdings) < max_holdings and len(day_scores) >= top_k:
            candidates = day_scores.nlargest(top_k).index
            for stock in candidates:
                if stock in holdings or len(holdings) + buys >= max_holdings:
                    break
                if stock not in day_ma.index:
                    continue
                try:
                    close = float(day_ma.loc[stock, "_close"])
                    ma5 = float(day_ma.loc[stock, "_ma5"])
                    ma20 = float(day_ma.loc[stock, "_ma20"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isnan(close) or np.isnan(ma5) or np.isnan(ma20):
                    continue
                if close < ma20:
                    continue
                if abs(close - ma5) / ma5 <= ma_entry_pct:
                    holdings[stock] = close
                    buys += 1

        total_trades += buys

        # PnL
        if holdings:
            port_rets = day_ret.reindex(list(holdings.keys())).dropna()
            raw_ret = port_rets.mean() if len(port_rets) > 0 else 0.0
        else:
            raw_ret = 0.0

        n_trades = len(sells) + buys
        cost_rate = cost.round_trip_rate() * n_trades / (2 * max(len(holdings), 1)) if holdings else 0
        daily_pnl.append(raw_ret - cost_rate)

    pnl = np.array(daily_pnl)
    if len(pnl) == 0:
        return {"annual": 0, "sharpe": 0, "maxdd": 0, "win_rate": 0, "avg_hold": 0}

    n = len(pnl)
    total_ret = float(np.prod(1 + pnl) - 1)
    annual = float((1 + total_ret) ** (250 / n) - 1)
    vol = float(np.std(pnl) * np.sqrt(250))
    sharpe = annual / (vol + 1e-8)
    cum = np.cumprod(1 + pnl)
    maxdd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)))

    return {
        "annual": round(annual * 100, 2),
        "sharpe": round(sharpe, 3),
        "maxdd": round(maxdd * 100, 2),
        "win_rate": round(float(np.mean(pnl > 0)) * 100, 1),
        "avg_hold": round(float(np.mean([len(holdings)])), 1),  # approximate
        "total_trades": total_trades,
    }


def main():
    import xgboost as xgb
    from config.qlib_runtime import init_qlib
    init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="feature_cache_174_holder_regime_ma.parquet")
    parser.add_argument("--n-splits", type=int, default=12)
    parser.add_argument("--test-days", type=int, default=40)
    parser.add_argument("--train-days", type=int, default=750)
    parser.add_argument("--valid-days", type=int, default=60)
    args = parser.parse_args()

    cache_path = DATA_DIR / args.cache
    if not cache_path.exists():
        logger.error(f"Cache not found: {cache_path}")
        sys.exit(1)

    logger.info(f"Loading cache: {cache_path}")
    t0 = time.time()
    cache = pd.read_parquet(str(cache_path))
    logger.info(f"  Loaded: {cache.shape}, {time.time()-t0:.1f}s")

    feature_cols = [c for c in cache.columns if not c.startswith("__") and not c.startswith("_")]
    ma_cols = ["_close", "_ma5", "_ma20"]
    has_ma = all(c in cache.columns for c in ma_cols)
    if not has_ma:
        logger.error("Cache missing MA columns. Rebuild with --include-ma")
        sys.exit(1)

    trade_dates = sorted(cache.index.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1

    cost = CostModel()
    all_results = []
    t_total = time.time()

    for split_idx in range(args.n_splits):
        test_end_idx = today_idx - split_idx * args.test_days
        test_start_idx = test_end_idx - args.test_days
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - args.valid_days
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - args.train_days

        if train_start_idx < 0:
            break

        test_end = trade_dates[test_end_idx]
        test_start = trade_dates[test_start_idx]
        valid_end = trade_dates[valid_end_idx]
        valid_start = trade_dates[valid_start_idx]
        train_end = trade_dates[train_end_idx]
        train_start = trade_dates[train_start_idx]

        logger.info(f"\nSplit {split_idx+1}/{args.n_splits}: "
                    f"test {str(test_start)[:10]}~{str(test_end)[:10]}")

        dates_level = cache.index.get_level_values(0)
        train_mask = (dates_level >= train_start) & (dates_level <= train_end)
        valid_mask = (dates_level >= valid_start) & (dates_level <= valid_end)
        test_mask = (dates_level >= test_start) & (dates_level <= test_end)

        # Train XGB
        y_train = cache.loc[train_mask, "__label_5d"].values.astype(np.float32)
        y_valid = cache.loc[valid_mask, "__label_5d"].values.astype(np.float32)
        X_train = cache.loc[train_mask, feature_cols].values.astype(np.float32)
        X_valid = cache.loc[valid_mask, feature_cols].values.astype(np.float32)

        mask_tr = np.isfinite(y_train)
        mask_va = np.isfinite(y_valid)

        t1 = time.time()
        model = train_xgb(X_train[mask_tr], y_train[mask_tr],
                          X_valid[mask_va], y_valid[mask_va])

        # Predict on test
        X_test = cache.loc[test_mask, feature_cols].values.astype(np.float32)
        pred = model.predict(xgb.DMatrix(X_test))
        test_idx = cache.index[test_mask]
        scores = pd.Series(pred, index=test_idx)
        scores = scores[np.isfinite(scores)]

        # Daily returns
        daily_ret = cache.loc[test_mask, "__pnl_return_1d"]
        daily_ret = daily_ret[np.isfinite(daily_ret)]

        # MA data
        ma_data = cache.loc[test_mask, ma_cols]

        # Run MA timing
        r = run_ma_timing(scores, daily_ret, ma_data, cost=cost)
        r["time_s"] = round(time.time() - t1, 1)
        all_results.append({"split": split_idx + 1,
                            "test": f"{str(test_start)[:10]}~{str(test_end)[:10]}",
                            **r})

        logger.info(f"  annual={r['annual']:+.1f}% sharpe={r['sharpe']:+.3f} "
                    f"dd={r['maxdd']:.1f}% win={r['win_rate']:.0f}% [{r['time_s']:.0f}s]")

    total_time = time.time() - t_total
    n = len(all_results)

    annuals = [r["annual"] for r in all_results]
    sharpes = [r["sharpe"] for r in all_results]

    logger.info(f"\n{'='*60}")
    logger.info(f"MA TIMING ROLLING ({n} splits, {total_time:.0f}s)")
    logger.info(f"{'='*60}")
    logger.info(f"  avg annual:  {np.mean(annuals):+.1f}%")
    logger.info(f"  avg sharpe:  {np.mean(sharpes):+.3f}")
    logger.info(f"  annual>0:    {sum(1 for a in annuals if a > 0)}/{n} "
                f"({sum(1 for a in annuals if a > 0)/n:.0%})")
    logger.info(f"  per-split:   {['%+.1f%%' % a for a in annuals]}")

    out_path = DATA_DIR / "phase4" / "rolling_ma_timing.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump({"evaluated_at": datetime.now().isoformat(timespec="seconds"),
                    "n_splits": n, "total_time_s": round(total_time, 1),
                    "avg_annual": round(float(np.mean(annuals)), 2),
                    "avg_sharpe": round(float(np.mean(sharpes)), 3),
                    "annual_pos_pct": round(sum(1 for a in annuals if a > 0) / n, 4),
                    "splits": all_results}, f, indent=2)
    logger.info(f"Saved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
