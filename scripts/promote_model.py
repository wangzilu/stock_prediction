"""Legacy promotion script — DISABLED 2026-06-04 (cx round 6 P0-1).

This script bypassed every promotion safeguard the codebase has
since added:
  * No PIT audit (look-ahead candidates pass freely).
  * No feature_contract check (would have NOT caught the 6-3 22:00
    158-vs-242 mismatch).
  * No 24-split / hold-out / cost-adjusted PnL check.
  * Does not sync the contract / dataset artifact when swapping
    lgb_candidate_model.pkl into lgb_model.pkl.
  * Single ``cand_ic > prod_ic * 1.1`` threshold, IC-only.

A single ``python scripts/promote_model.py --promote`` could swap a
candidate that fails any of the above into production.

To promote a candidate now, use the unified flow:
  1. ``tracker/promotion_gate.PromotionGate.check(...)``
  2. After PASS, run ``scripts/train_lgb.py`` against the chosen
     feature set — that path enforces PRODUCTION_SUPPLEMENTARY_GROUPS,
     writes the feature contract artifact, and atomic-saves the
     model only after prediction-health passes.

Running this script aborts immediately with a pointer to the new
flow. The implementation is kept for git-history reference but
gated behind the abort. Delete the file once the team is comfortable.
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


def _abort_legacy_entry() -> int:
    logger.error(
        "scripts/promote_model.py is DISABLED (cx round 6 P0-1 — 2026-06-04). "
        "It bypassed PIT audit, feature_contract check, 24-split, and "
        "cost-adjusted gate. Use tracker/promotion_gate.PromotionGate.check() "
        "then scripts/train_lgb.py instead. To re-enable for an emergency "
        "rollback ONLY, set environment variable "
        "LEGACY_PROMOTE_OVERRIDE=acknowledge_unsafe."
    )
    return 2

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
    # cx round 6 P0-1 + round 8 P2-6: hard-block every code path by
    # default. The legacy --check / --promote / --rollback subcommands
    # are kept in this module's source for git-blame archaeology but
    # cannot run unless an explicit override is set in the environment.
    # And even with the override, ONLY --rollback is permitted —
    # promote MUST go through the unified PromotionGate +
    # scripts/train_lgb.py path so the contract + PIT + cost-adjusted
    # gates run.
    if os.environ.get("LEGACY_PROMOTE_OVERRIDE") != "acknowledge_unsafe":
        return _abort_legacy_entry()

    parser = argparse.ArgumentParser(description="Model promotion gate (LEGACY rollback only)")
    parser.add_argument("--check", action="store_true", help="(DISABLED) Check promotion status")
    parser.add_argument("--promote", action="store_true", help="(DISABLED) Force promote candidate")
    parser.add_argument("--rollback", action="store_true", help="Force rollback (the only path the override permits)")
    args = parser.parse_args()

    if args.promote or args.check:
        logger.error(
            "LEGACY_PROMOTE_OVERRIDE permits --rollback ONLY (round 8 P2-6). "
            "Forward promotion must use tracker.promotion_gate.PromotionGate "
            "+ scripts/train_lgb.py so the contract / PIT / cost-adjusted "
            "gates run. Refusing --promote / --check."
        )
        return 2

    if args.rollback:
        logger.warning(
            "LEGACY_PROMOTE_OVERRIDE=acknowledge_unsafe + --rollback: "
            "executing emergency rollback. This bypasses the contract "
            "artifact — the resulting lgb_model.pkl may NOT match the "
            "current production_feature_contract.json, and inference "
            "will refuse to serve until train_lgb.py is re-run to "
            "regenerate the contract."
        )
        do_rollback()
        return 0

    logger.error("LEGACY_PROMOTE_OVERRIDE set but no action specified. "
                 "Pass --rollback to use the rollback path.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
