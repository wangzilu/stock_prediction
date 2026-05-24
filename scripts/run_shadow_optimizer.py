"""Shadow paper trading with opt_top100_to10 — uses unified PaperOMS.

Same predictions as champion, different execution strategy (optimizer_v2).
Runs daily at 18:40 (champion at 18:42).

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
SHADOW_DIR = str(DATA_DIR / "paper_shadow")
CHAMPION_DIR = str(DATA_DIR / "paper")


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

    from paper.oms import PaperOMS

    # Shadow uses optimizer_v2 with opt_top100_to10 config
    oms = PaperOMS(
        initial_capital=1_000_000,
        execution_mode="optimizer_v2",
        top_k=100,
        max_turnover=0.10,
        max_single_weight=0.05,
        weight_method="alpha_proportional",
        min_hold_days=2,
        state_dir=SHADOW_DIR,
        mode="pending",
    )

    if args.reset:
        state_path = Path(SHADOW_DIR) / "oms_state.json"
        trades_path = Path(SHADOW_DIR) / "trades.jsonl"
        for f in [state_path, trades_path]:
            if f.exists():
                os.remove(str(f))
        logger.info("Shadow optimizer reset")
        return

    if args.status:
        s = oms.status()
        logger.info("=== Shadow Optimizer Status ===")
        for k, v in s.items():
            logger.info(f"  {k}: {v}")
        return

    if args.report:
        history = oms.state.get("daily_pnl_history", [])
        if not history:
            logger.info("No trading history yet")
            return
        logger.info(f"=== Shadow Report ({len(history)} days) ===")
        logger.info(f"{'Date':<12} {'Value':>12} {'Return':>10} {'Pos':>5}")
        logger.info("-" * 45)
        for h in history[-20:]:
            logger.info(f"{h['date']:<12} {h['total_value']:>12,.0f} "
                        f"{h['daily_return']:>+10.4f} {h['n_positions']:>5}")
        return

    if args.compare:
        _compare_with_champion(oms)
        return

    # Run daily
    date = args.date or datetime.now().strftime("%Y-%m-%d")
    logger.info(f"=== Shadow Optimizer: {date} ===")
    pnl = oms.run_daily(date)

    logger.info(f"\nDaily summary:")
    logger.info(f"  Value: {pnl['total_value']:,.2f}")
    logger.info(f"  Return: {pnl['daily_return']:+.4f}")
    logger.info(f"  Positions: {pnl['n_positions']}")

    # Auto-compare
    _compare_with_champion(oms)

    # Push notification
    try:
        from push.wechat import WeChatPusher
        s = oms.status()
        msg = (f"Shadow Opt100to10 {date}\n"
               f"Value: {s['total_value']:,.0f}\n"
               f"Return: {s['total_return']*100:+.2f}%\n"
               f"Positions: {s['n_positions']}\n"
               f"Day {s['n_days']}/20")
        WeChatPusher().send(msg, title="Shadow Trading")
    except Exception as e:
        logger.warning(f"Push failed: {e}")

    logger.info("Done!")


def _compare_with_champion(shadow_oms):
    """Compare shadow vs champion."""
    champion_path = Path(CHAMPION_DIR) / "oms_state.json"
    if not champion_path.exists():
        logger.warning("Champion state not found")
        return

    champion_state = json.loads(champion_path.read_text())
    shadow_state = shadow_oms.state

    sh_ret = shadow_state.get("total_value", 1e6) / 1e6 - 1
    ch_ret = champion_state.get("total_value", 1e6) / 1e6 - 1

    logger.info("=== Shadow vs Champion ===")
    logger.info(f"  Shadow:   value={shadow_state['total_value']:,.0f}, "
                f"positions={len(shadow_state.get('positions', {}))}, "
                f"return={sh_ret:+.2%}")
    logger.info(f"  Champion: value={champion_state['total_value']:,.0f}, "
                f"positions={len(champion_state.get('positions', {}))}, "
                f"return={ch_ret:+.2%}")
    logger.info(f"  Excess:   {(sh_ret-ch_ret):+.2%}")

    # Log comparison
    compare_path = Path(SHADOW_DIR) / "daily_compare.jsonl"
    record = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "shadow_value": shadow_state.get("total_value"),
        "champion_value": champion_state.get("total_value"),
        "shadow_return": round(sh_ret, 6),
        "champion_return": round(ch_ret, 6),
        "excess": round(sh_ret - ch_ret, 6),
    }
    with open(compare_path, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
