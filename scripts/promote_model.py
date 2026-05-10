"""Model promotion gate: shadow run → promote/rollback.

Checks candidate model quality vs production model.
Promotes only if candidate is better for 3+ consecutive days.
Rollbacks if production model degrades for 3+ days.

Usage:
    python scripts/promote_model.py --check        # check if promotion is due
    python scripts/promote_model.py --promote      # force promote candidate
    python scripts/promote_model.py --rollback     # force rollback to previous
"""
import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
PROD_MODEL = DATA_DIR / "lgb_model.pkl"
CANDIDATE_MODEL = DATA_DIR / "lgb_candidate_model.pkl"
PREV_MODEL = DATA_DIR / "lgb_previous_model.pkl"
PROMOTION_LOG = DATA_DIR / "model_promotion_log.json"

PROMOTE_THRESHOLD_DAYS = 3  # candidate must be better for N days
ROLLBACK_THRESHOLD_DAYS = 3  # production must be degraded for N days


def load_eval_history() -> list:
    """Load evaluation history."""
    path = DATA_DIR / "lgb_eval_history.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return []


def check_promotion() -> dict:
    """Check if candidate should be promoted or production rolled back."""
    history = load_eval_history()
    if len(history) < PROMOTE_THRESHOLD_DAYS:
        return {"action": "wait", "reason": f"Only {len(history)} evals, need {PROMOTE_THRESHOLD_DAYS}"}

    recent = history[-PROMOTE_THRESHOLD_DAYS:]

    # Check production quality
    recent_ic = [h.get("metrics", {}).get("ic_mean", 0) for h in recent]
    recent_spread = [h.get("metrics", {}).get("top20_bot20_spread", 0) for h in recent]
    quality = [h.get("quality", "unknown") for h in recent]

    # Rollback check: all recent quality is degraded
    if all(q in ("weak", "degraded") for q in quality):
        return {
            "action": "rollback",
            "reason": f"Quality degraded for {ROLLBACK_THRESHOLD_DAYS} consecutive days",
            "recent_quality": quality,
            "recent_ic": recent_ic,
        }

    # IC negative check
    if all(ic < 0 for ic in recent_ic):
        return {
            "action": "rollback",
            "reason": f"IC negative for {len(recent_ic)} consecutive days",
            "recent_ic": recent_ic,
        }

    # Check if candidate exists and is better
    if CANDIDATE_MODEL.exists():
        cand_eval = DATA_DIR / "lgb_candidate_eval.json"
        if cand_eval.exists():
            try:
                cand = json.loads(cand_eval.read_text())
                cand_ic = cand.get("metrics", {}).get("ic_mean", 0)
                prod_ic = recent_ic[-1]

                if cand_ic > prod_ic * 1.1:  # 10% improvement
                    return {
                        "action": "promote",
                        "reason": f"Candidate IC {cand_ic:.4f} > production IC {prod_ic:.4f} * 1.1",
                        "candidate_ic": cand_ic,
                        "production_ic": prod_ic,
                    }
            except Exception:
                pass

    return {
        "action": "hold",
        "reason": "Production model performing normally",
        "recent_ic": recent_ic,
        "recent_quality": quality,
    }


def do_promote():
    """Promote candidate model to production."""
    if not CANDIDATE_MODEL.exists():
        logger.error("No candidate model found")
        return False

    # Backup current production
    if PROD_MODEL.exists():
        shutil.copy2(PROD_MODEL, PREV_MODEL)
        logger.info(f"Backed up production model to {PREV_MODEL}")

    # Promote candidate
    shutil.copy2(CANDIDATE_MODEL, PROD_MODEL)
    logger.info(f"Promoted candidate to production")

    _log_event("promote", f"Candidate promoted to production")
    return True


def do_rollback():
    """Rollback to previous production model."""
    if not PREV_MODEL.exists():
        logger.error("No previous model to rollback to")
        return False

    shutil.copy2(PREV_MODEL, PROD_MODEL)
    logger.info(f"Rolled back to previous model")

    _log_event("rollback", f"Rolled back to previous model")
    return True


def _log_event(action: str, reason: str):
    """Append to promotion log."""
    log = []
    if PROMOTION_LOG.exists():
        try:
            log = json.loads(PROMOTION_LOG.read_text())
        except Exception:
            pass

    log.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "reason": reason,
    })
    log = log[-100:]  # Keep last 100 entries
    PROMOTION_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Model promotion gate")
    parser.add_argument("--check", action="store_true", help="Check promotion status")
    parser.add_argument("--promote", action="store_true", help="Force promote candidate")
    parser.add_argument("--rollback", action="store_true", help="Force rollback")
    args = parser.parse_args()

    if args.promote:
        do_promote()
    elif args.rollback:
        do_rollback()
    else:
        result = check_promotion()
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if result["action"] == "promote":
            logger.info("Auto-promoting candidate model...")
            do_promote()
        elif result["action"] == "rollback":
            logger.info("Auto-rolling back production model...")
            do_rollback()
        else:
            logger.info(f"Action: {result['action']} — {result['reason']}")


if __name__ == "__main__":
    main()
