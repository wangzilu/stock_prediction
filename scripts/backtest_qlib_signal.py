"""Backtest LGB signal with TopK portfolio simulation and transaction costs.

Since TopkDropoutStrategy is blocked by cvxpy/numpy conflict,
this uses a local TopK implementation with Qlib data.

Usage:
    python scripts/backtest_qlib_signal.py
    python scripts/backtest_qlib_signal.py --json
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

from config.settings import PREDICTION_HORIZON_DAYS
from config.qlib_runtime import init_qlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
MODEL_PATH = str(DATA_DIR / "lgb_model.pkl")
BACKTEST_PATH = DATA_DIR / "lgb_backtest_latest.json"

# A-share transaction costs
OPEN_COST = 0.0005   # buy commission
CLOSE_COST = 0.0015  # sell commission + stamp tax
LIMIT_THRESHOLD = 0.095  # 9.5% limit up/down


def backtest(
    model_path: str = MODEL_PATH,
    qlib_data: str = QLIB_DATA,
    universe: str = "all",
    test_days: int = 60,
    topk: int = 20,
    max_drop: int = 5,
) -> dict:
    if not os.path.exists(model_path):
        return {"ok": False, "error": f"Model file not found: {model_path}"}

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    from qlib.utils import init_instance_by_config
    from qlib.contrib.evaluate import risk_analysis

    init_qlib(qlib_data)

    today = datetime.now()
    test_start = (today - timedelta(days=test_days)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")

    logger.info(f"Backtest: {test_start} ~ {test_end}, TopK={topk}, MaxDrop={max_drop}")

    # Use 5-day forward return label (matches training), rebalance every 5 days
    rebalance_period = PREDICTION_HORIZON_DAYS
    handler_config = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": test_start,
            "end_time": test_end,
            "instruments": universe,
            "label": [f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"],
        },
    }
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {"test": (test_start, test_end)},
        },
    }

    logger.info("Loading dataset...")
    dataset = init_instance_by_config(dataset_config)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    pred = model.predict(dataset=dataset)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")

    label = dataset.prepare("test", col_set="label")
    if isinstance(label, pd.DataFrame):
        label = label.iloc[:, 0]

    # Build daily cross-section
    common = pred.index.intersection(label.index)
    df = pd.DataFrame({
        "pred": pred.loc[common, "score"] if "score" in pred.columns else pred.loc[common].iloc[:, 0],
        "label": label.loc[common],
    })
    df = df.dropna()
    df = df[np.isfinite(df["pred"]) & np.isfinite(df["label"])]

    dates = sorted(df.index.get_level_values(0).unique())
    logger.info(f"Trading dates: {len(dates)}")

    if len(dates) < 5:
        return {"ok": False, "error": f"Too few trading dates: {len(dates)}"}

    # TopK with dropout simulation — rebalance every N days (non-overlapping)
    holdings = set()
    period_returns = []
    period_turnover = []
    period_holdings_count = []

    rebalance_dates = dates[::rebalance_period]  # every 5 trading days
    logger.info(f"Rebalance dates: {len(rebalance_dates)} (every {rebalance_period} days)")

    for i, date in enumerate(rebalance_dates):
        day_data = df.loc[date].copy()
        if len(day_data) < topk * 2:
            continue

        # Rank by prediction
        day_data = day_data.sort_values("pred", ascending=False)
        candidates = set(day_data.head(topk).index)

        if not holdings:
            new_holdings = candidates
            buys = new_holdings
            sells = set()
        else:
            keep_zone = set(day_data.head(topk + max_drop).index)
            keep = holdings & keep_zone
            n_fill = topk - len(keep)
            fill = set()
            for code in day_data.index:
                if code not in keep and len(fill) < n_fill:
                    fill.add(code)
            new_holdings = keep | fill
            buys = new_holdings - holdings
            sells = holdings - new_holdings

        # Period return = equal weight average of holdings' forward returns
        holding_returns = []
        for code in new_holdings:
            if code in day_data.index:
                ret = float(day_data.loc[code, "label"])
                holding_returns.append(ret)

        if holding_returns:
            port_return = np.mean(holding_returns)
        else:
            port_return = 0.0

        # Transaction costs
        turnover = (len(buys) + len(sells)) / max(len(new_holdings), 1)
        cost = len(buys) / max(len(new_holdings), 1) * OPEN_COST + \
               len(sells) / max(len(new_holdings), 1) * CLOSE_COST
        net_return = port_return - cost

        period_returns.append(net_return)
        period_turnover.append(turnover)
        period_holdings_count.append(len(new_holdings))
        holdings = new_holdings

    daily_returns = period_returns

    if not daily_returns:
        return {"ok": False, "error": "No valid trading days"}

    ret_series = pd.Series(daily_returns)

    # Metrics — each return covers rebalance_period trading days
    periods_per_year = 252 / rebalance_period
    total_return = float((1 + ret_series).prod() - 1)
    ann_return = float(ret_series.mean() * periods_per_year)
    ann_vol = float(ret_series.std() * np.sqrt(periods_per_year))
    sharpe = ann_return / (ann_vol + 1e-8)
    cumulative = (1 + ret_series).cumprod()
    max_dd = float((cumulative / cumulative.cummax() - 1).min())
    win_rate = float((ret_series > 0).mean())
    avg_turnover = float(np.mean(period_turnover))

    # Universe benchmark
    universe_returns = []
    for date in dates:
        day_data = df.loc[date]
        if len(day_data) > 0:
            universe_returns.append(float(day_data["label"].mean()))
    bench_return = float(np.mean(universe_returns) * 252) if universe_returns else 0.0
    excess_return = ann_return - bench_return

    result = {
        "ok": True,
        "backtest_at": datetime.now().isoformat(timespec="seconds"),
        "test_start": test_start,
        "test_end": test_end,
        "n_trading_days": len(daily_returns),
        "topk": topk,
        "max_drop": max_drop,
        "costs": {"open": OPEN_COST, "close": CLOSE_COST},
        "metrics": {
            "total_return_pct": round(total_return * 100, 2),
            "annualized_return_pct": round(ann_return * 100, 2),
            "annualized_volatility_pct": round(ann_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "win_rate": round(win_rate, 4),
            "avg_period_turnover": round(avg_turnover, 4),
            "excess_return_pct": round(excess_return * 100, 2),
            "avg_holdings": round(np.mean(period_holdings_count), 1),
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Backtest LGB signal")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--universe", default="all")
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max-drop", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = backtest(
        model_path=args.model_path,
        universe=args.universe,
        test_days=args.test_days,
        topk=args.topk,
        max_drop=args.max_drop,
    )

    # Save
    BACKTEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = BACKTEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    os.replace(tmp, BACKTEST_PATH)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        m = result.get("metrics", {})
        print(f"Backtest: {result.get('n_trading_days', 0)} days, Top{args.topk}")
        print(f"Total Return: {m.get('total_return_pct', 0):+.2f}%")
        print(f"Ann. Return:  {m.get('annualized_return_pct', 0):+.2f}%")
        print(f"Sharpe:       {m.get('sharpe_ratio', 0):.3f}")
        print(f"Max Drawdown: {m.get('max_drawdown_pct', 0):.2f}%")
        print(f"Win Rate:     {m.get('win_rate', 0):.1%}")
        print(f"Avg Turnover: {m.get('avg_period_turnover', 0):.1%}")
        print(f"Excess Return:{m.get('excess_return_pct', 0):+.2f}%")

    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
