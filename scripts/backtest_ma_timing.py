"""Phase 4H: MA timing backtest — XGB selection + MA entry/exit rules.

Strategy (from lzz):
- Entry: XGB Top候选 + price near 5MA (贴着5日线买)
- Stop loss: price breaks below 20MA (跌破20日线止损)
- Take profit: +20% gain or price too far above 5MA (远离5日线止盈)

Combines model stock selection with technical timing.

Usage:
    python scripts/backtest_ma_timing.py
    python scripts/backtest_ma_timing.py --top-k 50 --ma-entry-pct 0.02
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"


def load_ma_data(index: pd.MultiIndex) -> pd.DataFrame:
    """Load MA5, MA20, close price for timing signals."""
    from qlib.data import D

    insts = sorted(set(str(c) for c in index.get_level_values(1)))
    dates = sorted(index.get_level_values(0).unique())

    exprs = [
        "$close",
        "Mean($close, 5)",    # MA5
        "Mean($close, 20)",   # MA20
    ]
    col_names = ["close", "ma5", "ma20"]

    df = D.features(insts, exprs,
                    start_time=str(min(dates))[:10],
                    end_time=str(max(dates))[:10])
    if df is None or df.empty:
        return pd.DataFrame()

    df.columns = col_names
    df = df.swaplevel().sort_index()
    return df.replace([np.inf, -np.inf], np.nan)


def run_ma_timing_backtest(
    predictions: pd.Series,
    daily_returns: pd.DataFrame,
    ma_data: pd.DataFrame,
    top_k: int = 50,
    max_holdings: int = 20,
    ma_entry_pct: float = 0.02,    # buy within 2% of MA5
    ma_exit_above: float = 0.08,   # take profit: 8%+ above MA5
    take_profit: float = 0.20,     # +20% absolute gain
    cost_model: CostModel = None,
) -> dict:
    """Run MA-timing backtest.

    Logic:
    1. XGB selects top_k candidates each day
    2. Only BUY candidates that are within ma_entry_pct of MA5 (near support)
    3. SELL if: price breaks below MA20 (stop loss) OR
                price is ma_exit_above above MA5 (overbought) OR
                cumulative gain > take_profit
    4. Max holdings capped at max_holdings
    """
    cost = cost_model or CostModel()
    dates = sorted(predictions.index.get_level_values(0).unique())

    # Track holdings: {stock: {"entry_price": float, "entry_date": date}}
    holdings = {}
    daily_pnl = []
    daily_n_holdings = []
    daily_n_buys = []
    daily_n_sells = []
    total_trades = 0

    for i, date in enumerate(dates):
        if date not in predictions.index.get_level_values(0):
            continue

        # Get today's scores and MA data
        day_pred = predictions.loc[date] if date in predictions.index.get_level_values(0) else pd.Series()
        if isinstance(day_pred, pd.DataFrame):
            scores = day_pred.iloc[:, 0]
        else:
            scores = day_pred
        scores = scores.dropna()

        day_ma = ma_data.loc[date] if date in ma_data.index.get_level_values(0) else pd.DataFrame()
        if isinstance(day_ma, pd.Series):
            day_ma = day_ma.to_frame().T

        # Get daily return for next day (T+1)
        if i + 1 < len(dates):
            next_date = dates[i + 1]
            if next_date in daily_returns.index.get_level_values(0):
                day_ret = daily_returns.loc[next_date]
                if isinstance(day_ret, pd.DataFrame):
                    day_ret = day_ret.iloc[:, 0]
            else:
                day_ret = pd.Series(dtype=float)
        else:
            day_ret = pd.Series(dtype=float)

        # === SELL decisions ===
        sells = []
        for stock, info in list(holdings.items()):
            if stock not in day_ma.index:
                continue

            try:
                close = float(day_ma.loc[stock, "close"])
                ma5 = float(day_ma.loc[stock, "ma5"])
                ma20 = float(day_ma.loc[stock, "ma20"])
            except (KeyError, TypeError):
                continue

            if np.isnan(close) or np.isnan(ma20):
                continue

            entry_price = info["entry_price"]
            gain = (close - entry_price) / entry_price

            # Stop loss: price below MA20
            if close < ma20:
                sells.append(stock)
                continue

            # Take profit: +20% gain
            if gain >= take_profit:
                sells.append(stock)
                continue

            # Overbought: too far above MA5
            if not np.isnan(ma5) and ma5 > 0:
                dist_from_ma5 = (close - ma5) / ma5
                if dist_from_ma5 > ma_exit_above:
                    sells.append(stock)
                    continue

        for stock in sells:
            del holdings[stock]
        total_trades += len(sells)

        # === BUY decisions ===
        buys = []
        if len(holdings) < max_holdings and len(scores) >= top_k:
            candidates = scores.nlargest(top_k).index

            for stock in candidates:
                if stock in holdings:
                    continue
                if len(holdings) + len(buys) >= max_holdings:
                    break
                if stock not in day_ma.index:
                    continue

                try:
                    close = float(day_ma.loc[stock, "close"])
                    ma5 = float(day_ma.loc[stock, "ma5"])
                    ma20 = float(day_ma.loc[stock, "ma20"])
                except (KeyError, TypeError):
                    continue

                if np.isnan(close) or np.isnan(ma5) or np.isnan(ma20):
                    continue

                # Must be above MA20 (uptrend)
                if close < ma20:
                    continue

                # Must be near MA5 (within entry_pct)
                dist_from_ma5 = abs(close - ma5) / ma5
                if dist_from_ma5 <= ma_entry_pct:
                    buys.append(stock)
                    holdings[stock] = {"entry_price": close, "entry_date": date}

        total_trades += len(buys)

        # === Compute daily PnL ===
        if holdings:
            held_stocks = list(holdings.keys())
            port_rets = day_ret.reindex(held_stocks).dropna()
            if len(port_rets) > 0:
                raw_ret = port_rets.mean()
            else:
                raw_ret = 0.0
        else:
            raw_ret = 0.0

        # Cost: per trade
        n_trades = len(sells) + len(buys)
        cost_rate = cost.round_trip_rate() * n_trades / (2 * max(len(holdings), 1)) if holdings else 0
        net_ret = raw_ret - cost_rate

        daily_pnl.append(net_ret)
        daily_n_holdings.append(len(holdings))
        daily_n_buys.append(len(buys))
        daily_n_sells.append(len(sells))

    # Metrics
    pnl = np.array(daily_pnl)
    n_days = len(pnl)
    if n_days == 0:
        return {"error": "no trading days"}

    annual_factor = 250 / n_days
    total_ret = float(np.prod(1 + pnl) - 1)
    annual_ret = float((1 + total_ret) ** annual_factor - 1)
    annual_vol = float(np.std(pnl) * np.sqrt(250))
    sharpe = annual_ret / (annual_vol + 1e-8)
    cum = np.cumprod(1 + pnl)
    running_max = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum - running_max) / running_max))
    win_rate = float(np.mean(pnl > 0))

    return {
        "n_days": n_days,
        "total_return": round(total_ret * 100, 2),
        "annual_return": round(annual_ret * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "avg_holdings": round(float(np.mean(daily_n_holdings)), 1),
        "total_trades": total_trades,
        "avg_daily_buys": round(float(np.mean(daily_n_buys)), 1),
        "avg_daily_sells": round(float(np.mean(daily_n_sells)), 1),
    }


def main():
    import xgboost as xgb
    from qlib.utils import init_instance_by_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=50,
                        help="XGB candidate pool size")
    parser.add_argument("--max-holdings", type=int, default=20)
    parser.add_argument("--ma-entry-pct", type=float, default=0.02,
                        help="Max distance from MA5 to buy (2%%)")
    parser.add_argument("--ma-exit-above", type=float, default=0.08,
                        help="Take profit when 8%%+ above MA5")
    parser.add_argument("--take-profit", type=float, default=0.20,
                        help="Absolute take profit (20%%)")
    parser.add_argument("--test-months", type=int, default=6)
    args = parser.parse_args()

    init_qlib(QLIB_DATA)
    merger = FeatureMerger(DATA_DIR)

    today = datetime.now()
    test_end = today.strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=30 * args.test_months)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=30 * args.test_months + 1)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=30 * args.test_months + 61)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=30 * args.test_months + 62)).strftime("%Y-%m-%d")
    train_start = (today - timedelta(days=365 * 3 + 30 * args.test_months + 62)).strftime("%Y-%m-%d")

    logger.info(f"=== Phase 4H: MA Timing Backtest ===")
    logger.info(f"Test: {test_start}~{test_end}")
    logger.info(f"Params: top_k={args.top_k}, entry={args.ma_entry_pct:.0%}, "
                f"exit_above={args.ma_exit_above:.0%}, tp={args.take_profit:.0%}")

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

    # Train model
    logger.info("Preparing features...")
    X_train_df, y_train_s = prepare_features_174(dataset, "train", merger)
    X_valid_df, y_valid_s = prepare_features_174(dataset, "valid", merger)
    X_test_df, y_test_s = prepare_features_174(dataset, "test", merger)

    y_train = y_train_s.values.astype(np.float32)
    mask_train = np.isfinite(y_train)
    y_valid = y_valid_s.values.astype(np.float32)
    mask_valid = np.isfinite(y_valid)

    logger.info("Training XGB...")
    model = train_xgb(
        X_train_df.values.astype(np.float32)[mask_train], y_train[mask_train],
        X_valid_df.values.astype(np.float32)[mask_valid], y_valid[mask_valid])

    # Predict
    pred_raw = model.predict(xgb.DMatrix(X_test_df.values.astype(np.float32)))
    predictions = pd.Series(pred_raw, index=X_test_df.index, name="score")
    predictions = predictions[np.isfinite(predictions)]

    # Load daily returns and MA data
    logger.info("Loading daily returns and MA data...")
    daily_returns = load_daily_returns(X_test_df.index)
    if isinstance(daily_returns, pd.DataFrame):
        daily_returns = daily_returns.rename(columns={"pnl_return_1d": "return"})
    ma_data = load_ma_data(X_test_df.index)
    logger.info(f"MA data: {ma_data.shape}")

    # Run multiple configs
    configs = [
        {"name": "default", "top_k": args.top_k, "ma_entry_pct": 0.02,
         "ma_exit_above": 0.08, "take_profit": 0.20},
        {"name": "tight_entry", "top_k": args.top_k, "ma_entry_pct": 0.01,
         "ma_exit_above": 0.08, "take_profit": 0.20},
        {"name": "wide_entry", "top_k": args.top_k, "ma_entry_pct": 0.03,
         "ma_exit_above": 0.08, "take_profit": 0.20},
        {"name": "quick_profit", "top_k": args.top_k, "ma_entry_pct": 0.02,
         "ma_exit_above": 0.05, "take_profit": 0.15},
        {"name": "patient", "top_k": args.top_k, "ma_entry_pct": 0.02,
         "ma_exit_above": 0.10, "take_profit": 0.30},
    ]

    cost = CostModel()
    results = {}

    for cfg in configs:
        logger.info(f"\nConfig: {cfg['name']}")
        r = run_ma_timing_backtest(
            predictions, daily_returns, ma_data,
            top_k=cfg["top_k"],
            max_holdings=args.max_holdings,
            ma_entry_pct=cfg["ma_entry_pct"],
            ma_exit_above=cfg["ma_exit_above"],
            take_profit=cfg["take_profit"],
            cost_model=cost,
        )
        results[cfg["name"]] = r
        logger.info(f"  annual={r.get('annual_return', 0):+.1f}% "
                    f"sharpe={r.get('sharpe', 0):.3f} "
                    f"dd={r.get('max_drawdown', 0):.1f}% "
                    f"win={r.get('win_rate', 0):.0f}% "
                    f"avg_hold={r.get('avg_holdings', 0):.1f}")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("MA TIMING BACKTEST RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"{'Config':<15} {'Annual':>8} {'Sharpe':>8} {'MaxDD':>8} "
                f"{'WinRate':>8} {'AvgHold':>8} {'Trades':>7}")
    logger.info("-" * 65)
    for name, r in results.items():
        logger.info(f"{name:<15} {r.get('annual_return',0):+.1f}%   "
                    f"{r.get('sharpe',0):+.3f}   {r.get('max_drawdown',0):.1f}%   "
                    f"{r.get('win_rate',0):.0f}%     {r.get('avg_holdings',0):.1f}     "
                    f"{r.get('total_trades',0)}")

    # Save
    out_path = DATA_DIR / "phase4" / "ma_timing_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump({"evaluated_at": datetime.now().isoformat(timespec="seconds"),
                    "results": results}, f, indent=2)
    logger.info(f"\nSaved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
