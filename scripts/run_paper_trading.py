"""Run daily paper trading — call after market close.

Loads latest predictions, generates orders, simulates fills, updates PnL.

Usage:
    python scripts/run_paper_trading.py                # run for today
    python scripts/run_paper_trading.py --status       # show current status
    python scripts/run_paper_trading.py --report       # show PnL history
    python scripts/run_paper_trading.py --reset        # reset to initial state
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from paper.oms import PaperOMS

DATA_DIR = PROJECT_ROOT / "data" / "storage"
PAPER_DIR = DATA_DIR / "paper"


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true", default=True, help="Run daily paper trading")
    group.add_argument("--status", action="store_true", help="Show current status")
    group.add_argument("--report", action="store_true", help="Show PnL report")
    group.add_argument("--reset", action="store_true", help="Reset paper trading")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--capital", type=float, default=1_000_000)
    args = parser.parse_args()

    oms = PaperOMS(initial_capital=args.capital, mode="pending")

    if args.status:
        s = oms.status()
        logger.info("=== Paper Trading Status ===")
        for k, v in s.items():
            logger.info(f"  {k}: {v}")
        return

    if args.report:
        history = oms.state.get("daily_pnl_history", [])
        if not history:
            logger.info("No trading history yet")
            return

        logger.info(f"=== Paper Trading Report ({len(history)} days) ===")
        logger.info(f"{'Date':<12} {'Value':>12} {'Return':>10} {'Positions':>10}")
        logger.info("-" * 50)
        for h in history[-20:]:  # last 20 days
            logger.info(f"{h['date']:<12} {h['total_value']:>12,.2f} "
                        f"{h['daily_return']:>+10.4f} {h['n_positions']:>10}")

        import numpy as np
        returns = [h["daily_return"] for h in history]
        total_ret = oms.state["total_value"] / oms.initial_capital - 1
        logger.info(f"\n  Total return: {total_ret*100:+.2f}%")
        logger.info(f"  Win rate: {np.mean([r > 0 for r in returns])*100:.0f}%")
        logger.info(f"  Avg daily return: {np.mean(returns)*100:+.3f}%")
        logger.info(f"  Total trades: {oms.state['trade_count']}")
        return

    if args.reset:
        state_path = PAPER_DIR / "oms_state.json"
        trades_path = PAPER_DIR / "trades.jsonl"
        if state_path.exists():
            os.remove(str(state_path))
        if trades_path.exists():
            os.remove(str(trades_path))
        logger.info("Paper trading reset")
        return

    # Run daily
    date = args.date or datetime.now().strftime("%Y-%m-%d")
    logger.info(f"=== Paper Trading: {date} ===")

    # --- Freshness check: warn if predictions are stale ---
    from scheduler.data_health import is_fresh
    pred_fresh = (
        is_fresh("lgb_smoke_predict") or is_fresh("lgb_after_close_smoke")
    )
    if not pred_fresh:
        logger.warning("Using stale predictions — no fresh prediction health found for today")

    pnl = oms.run_daily(date)

    # Handle pending mode: run_daily may return {"status": "pending"} when
    # T+1 open prices are not yet available (e.g., running same-day after close)
    if isinstance(pnl, dict) and pnl.get("status") == "pending":
        logger.info(f"\n  Orders generated, awaiting T+1 open for reconciliation.")
        logger.info(f"  Run again tomorrow after 10:00 to reconcile.")
        return

    logger.info(f"\nDaily summary:")
    logger.info(f"  Value: {pnl.get('total_value', 0):,.2f}")
    logger.info(f"  Return: {pnl.get('daily_return', 0):+.4f}")
    logger.info(f"  Positions: {pnl.get('n_positions', 0)}")

    # Push notification
    try:
        from push.wechat import WeChatPusher
        s = oms.status()
        stale_tag = " [STALE PREDICTIONS]" if not pred_fresh else ""
        msg = (f"📋 Paper Trading {date}{stale_tag}\n"
               f"Value: {s['total_value']:,.0f}\n"
               f"Return: {s['total_return']*100:+.2f}%\n"
               f"Positions: {s['n_positions']}\n"
               f"Day {s['n_days']}/20")
        WeChatPusher().send(msg, title="Paper Trading")
    except Exception as e:
        logger.warning(f"Push failed: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
