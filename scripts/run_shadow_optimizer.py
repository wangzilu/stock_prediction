"""Shadow paper trading with opt_top100_to10 — runs alongside champion.

Uses the same predictions as champion (lgb_latest_predictions.json),
but applies optimizer_v2 (top100, max_turnover=10%, alpha_proportional)
instead of buffered_partial.

Daily output: shadow portfolio state, trades, comparison with champion.

Usage:
    python scripts/run_shadow_optimizer.py              # run for today
    python scripts/run_shadow_optimizer.py --status     # show current status
    python scripts/run_shadow_optimizer.py --report     # show PnL history
    python scripts/run_shadow_optimizer.py --reset      # reset to initial state
    python scripts/run_shadow_optimizer.py --compare    # compare with champion
"""
import argparse
import json
import logging
import os
import sys
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
SHADOW_DIR = DATA_DIR / "paper_shadow"
SHADOW_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = SHADOW_DIR / "oms_state.json"
TRADES_FILE = SHADOW_DIR / "trades.jsonl"
COMPARE_FILE = SHADOW_DIR / "daily_compare.jsonl"

# Optimizer config (the 24-split gate winner)
TOP_K = 100
MAX_TURNOVER = 0.10
MAX_SINGLE_WEIGHT = 0.05
WEIGHT_METHOD = "alpha_proportional"
MIN_HOLD_DAYS = 2

COMMISSION_RATE = 0.0003
STAMP_TAX_RATE = 0.0005
SLIPPAGE_RATE = 0.001


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "cash": 1_000_000,
        "positions": {},  # {code: {"shares": N, "weight": W, "avg_price": P, "entry_date": D, "holding_days": H}}
        "total_value": 1_000_000,
        "trade_count": 0,
        "start_date": None,
        "last_update": None,
        "daily_pnl_history": [],
        "prev_weights": {},  # {code: weight} for optimizer
    }


def save_state(state: dict):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2,
                              default=lambda o: str(o) if isinstance(o, datetime) else o))
    os.replace(str(tmp), str(STATE_FILE))


def load_predictions() -> dict:
    cache_path = DATA_DIR / "lgb_latest_predictions.json"
    if not cache_path.exists():
        return {}
    payload = json.loads(cache_path.read_text())
    return payload.get("predictions", {})


def load_prices(codes: list, date: str) -> dict:
    """Load closing prices from Qlib."""
    prices = {}
    try:
        from config.qlib_runtime import init_qlib
        from qlib.data import D
        try:
            D.calendar(freq="day", start_time="2020-01-01", end_time="2020-01-02")
        except Exception:
            init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

        qlib_codes = [c.lower() for c in codes]
        df = D.features(qlib_codes, ["$close"], start_time=date, end_time=date)
        if df is not None and not df.empty:
            for idx, row in df.iterrows():
                inst = str(idx[0]).upper() if isinstance(idx, tuple) else str(idx).upper()
                price = float(row.iloc[0])
                if np.isfinite(price) and price > 0:
                    prices[inst] = price
    except Exception as e:
        logger.warning(f"Failed to load prices: {e}")
    return prices


def compute_target_weights(predictions: dict, prev_weights: dict,
                           holding_days: dict) -> dict:
    """Use optimizer_v2 to compute target weights."""
    from backtest.optimizer_v2 import TurnoverConstrainedOptimizer
    from backtest.constraints import PortfolioConstraints

    scores = pd.Series(predictions).sort_values(ascending=False)

    optimizer = TurnoverConstrainedOptimizer(
        top_k=TOP_K,
        max_turnover=MAX_TURNOVER,
        max_single_weight=MAX_SINGLE_WEIGHT,
        weight_method=WEIGHT_METHOD,
    )

    constraints = PortfolioConstraints(min_hold_days=MIN_HOLD_DAYS)

    target = optimizer.optimize(
        alpha_scores=scores,
        prev_weights=prev_weights,
        constraints=constraints,
        holding_days=holding_days,
    )
    return target


def run_daily(date: str = None):
    """Execute one day of shadow paper trading."""
    date = date or datetime.now().strftime("%Y-%m-%d")
    state = load_state()
    logger.info(f"=== Shadow Optimizer: {date} ===")

    if state["start_date"] is None:
        state["start_date"] = date

    # Dedup guard
    if state["last_update"] and state["last_update"][:10] == date:
        logger.info("  Already ran today, skipping")
        return state

    # Load predictions
    predictions = load_predictions()
    if not predictions:
        logger.warning("  No predictions, holding current")
        _record_daily_pnl(state, date, {})
        save_state(state)
        return state

    logger.info(f"  Predictions: {len(predictions)}")

    # Get holding days
    holding_days = {}
    for code, pos in state["positions"].items():
        holding_days[code] = pos.get("holding_days", 0)

    # Compute target weights
    prev_weights = state.get("prev_weights", {})
    target_weights = compute_target_weights(predictions, prev_weights, holding_days)
    logger.info(f"  Target: {len(target_weights)} stocks")

    # Load prices for all relevant stocks
    all_codes = list(set(list(state["positions"].keys()) + list(target_weights.keys())))
    prices = load_prices(all_codes, date)
    logger.info(f"  Prices loaded: {len(prices)}/{len(all_codes)}")

    # Execute: sell stocks not in target, buy stocks in target
    cash = state["cash"]
    positions = state["positions"]
    trades = []

    # Current portfolio value
    total_value = cash
    for code, pos in positions.items():
        price = prices.get(code, pos["avg_price"])
        total_value += pos["shares"] * price

    # Determine sells (stocks in positions but not in target, or weight decreased)
    for code in list(positions.keys()):
        if code not in target_weights:
            # Full sell
            if code in prices:
                pos = positions[code]
                amount = pos["shares"] * prices[code]
                cost = amount * (COMMISSION_RATE + STAMP_TAX_RATE + SLIPPAGE_RATE)
                cash += amount - cost
                trades.append({"code": code, "side": "sell", "shares": pos["shares"],
                               "price": prices[code], "cost": round(cost, 2)})
                del positions[code]
                state["trade_count"] += 1

    # Determine buys
    for code, target_w in sorted(target_weights.items(), key=lambda x: -x[1]):
        if code in positions:
            continue  # already holding, skip rebalance for simplicity in v1
        if code not in prices or prices[code] <= 0:
            continue

        target_value = total_value * target_w
        buy_price = prices[code]
        shares = int(target_value / buy_price / 100) * 100
        if shares <= 0:
            continue

        amount = shares * buy_price
        cost = amount * (COMMISSION_RATE + SLIPPAGE_RATE)
        if amount + cost > cash:
            # Reduce shares to fit
            shares = int(cash * 0.95 / buy_price / 100) * 100
            if shares <= 0:
                continue
            amount = shares * buy_price
            cost = amount * (COMMISSION_RATE + SLIPPAGE_RATE)

        cash -= amount + cost
        positions[code] = {
            "shares": shares,
            "weight": round(target_w, 6),
            "avg_price": round(buy_price, 4),
            "entry_date": date,
            "holding_days": 0,
        }
        trades.append({"code": code, "side": "buy", "shares": shares,
                       "price": buy_price, "cost": round(cost, 2)})
        state["trade_count"] += 1

    # Update holding days
    for code in positions:
        positions[code]["holding_days"] = positions[code].get("holding_days", 0) + 1

    state["cash"] = round(cash, 2)
    state["positions"] = positions
    state["prev_weights"] = target_weights
    state["last_update"] = datetime.now().isoformat(timespec="seconds")

    # Compute PnL
    _record_daily_pnl(state, date, prices)
    save_state(state)

    # Log trades
    if trades:
        with open(TRADES_FILE, "a") as f:
            for t in trades:
                t["date"] = date
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    logger.info(f"  Trades: {len(trades)} (sells={sum(1 for t in trades if t['side']=='sell')}, "
                f"buys={sum(1 for t in trades if t['side']=='buy')})")
    logger.info(f"  Positions: {len(positions)}")
    logger.info(f"  Value: {state['total_value']:,.0f}")

    return state


def _record_daily_pnl(state, date, prices):
    """Compute and record daily PnL."""
    total_value = state["cash"]
    for code, pos in state["positions"].items():
        price = prices.get(code, pos["avg_price"])
        total_value += pos["shares"] * price

    prev_value = state["total_value"]
    daily_return = (total_value / prev_value - 1) if prev_value > 0 else 0.0

    state["total_value"] = round(total_value, 2)
    state["daily_pnl_history"].append({
        "date": date,
        "cash": state["cash"],
        "n_positions": len(state["positions"]),
        "total_value": round(total_value, 2),
        "trade_count": state["trade_count"],
        "daily_return": round(daily_return, 6),
    })


def show_status(state):
    logger.info("=== Shadow Optimizer Status ===")
    logger.info(f"  Value: {state['total_value']:,.2f}")
    logger.info(f"  Cash: {state['cash']:,.2f}")
    logger.info(f"  Positions: {len(state['positions'])}")
    logger.info(f"  Trades: {state['trade_count']}")
    logger.info(f"  Days: {len(state['daily_pnl_history'])}")
    total_ret = state["total_value"] / 1_000_000 - 1
    logger.info(f"  Total return: {total_ret*100:+.2f}%")


def show_report(state):
    history = state.get("daily_pnl_history", [])
    if not history:
        logger.info("No history yet")
        return
    logger.info(f"=== Shadow Report ({len(history)} days) ===")
    logger.info(f"{'Date':<12} {'Value':>12} {'Return':>10} {'Pos':>5}")
    logger.info("-" * 45)
    for h in history[-20:]:
        logger.info(f"{h['date']:<12} {h['total_value']:>12,.0f} "
                    f"{h['daily_return']:>+10.4f} {h['n_positions']:>5}")


def compare_with_champion():
    """Compare shadow vs champion paper trading."""
    shadow_state = load_state()
    champion_path = DATA_DIR / "paper" / "oms_state.json"
    if not champion_path.exists():
        logger.warning("Champion state not found")
        return

    champion_state = json.loads(champion_path.read_text())

    sh_history = shadow_state.get("daily_pnl_history", [])
    ch_history = champion_state.get("daily_pnl_history", [])

    logger.info("=== Shadow vs Champion ===")
    logger.info(f"  Shadow:   value={shadow_state['total_value']:,.0f}, "
                f"positions={len(shadow_state['positions'])}, days={len(sh_history)}")
    logger.info(f"  Champion: value={champion_state['total_value']:,.0f}, "
                f"positions={len(champion_state['positions'])}, days={len(ch_history)}")

    sh_ret = shadow_state["total_value"] / 1_000_000 - 1
    ch_ret = champion_state["total_value"] / 1_000_000 - 1
    logger.info(f"  Shadow return:   {sh_ret*100:+.2f}%")
    logger.info(f"  Champion return: {ch_ret*100:+.2f}%")
    logger.info(f"  Excess:          {(sh_ret-ch_ret)*100:+.2f}%")

    # Log comparison
    record = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "shadow_value": shadow_state["total_value"],
        "champion_value": champion_state["total_value"],
        "shadow_return": round(sh_ret, 6),
        "champion_return": round(ch_ret, 6),
        "excess": round(sh_ret - ch_ret, 6),
        "shadow_positions": len(shadow_state["positions"]),
        "champion_positions": len(champion_state["positions"]),
    }
    with open(COMPARE_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Shadow paper trading with optimizer_v2")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true", default=True)
    group.add_argument("--status", action="store_true")
    group.add_argument("--report", action="store_true")
    group.add_argument("--reset", action="store_true")
    group.add_argument("--compare", action="store_true")
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()

    if args.reset:
        for f in [STATE_FILE, TRADES_FILE, COMPARE_FILE]:
            if f.exists():
                os.remove(str(f))
        logger.info("Shadow optimizer reset")
        return

    if args.status:
        show_status(load_state())
        return

    if args.report:
        show_report(load_state())
        return

    if args.compare:
        compare_with_champion()
        return

    # Run daily
    state = run_daily(args.date)

    # Auto-compare after run
    compare_with_champion()

    # Push notification
    try:
        from push.wechat import WeChatPusher
        sh_ret = state["total_value"] / 1_000_000 - 1
        msg = (f"Shadow Opt100to10 {datetime.now().strftime('%Y-%m-%d')}\n"
               f"Value: {state['total_value']:,.0f}\n"
               f"Return: {sh_ret*100:+.2f}%\n"
               f"Positions: {len(state['positions'])}\n"
               f"Day {len(state['daily_pnl_history'])}/20")
        WeChatPusher().send(msg, title="Shadow Trading")
    except Exception as e:
        logger.warning(f"Push failed: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
