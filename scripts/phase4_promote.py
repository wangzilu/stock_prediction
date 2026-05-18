"""Phase 4D: Model promotion gate — champion / shadow / research_only.

Checks Track A/B/C artifacts and manages model lifecycle.

Usage:
    python scripts/phase4_promote.py --check --model xgb_205
    python scripts/phase4_promote.py --promote shadow --model xgb_205
    python scripts/phase4_promote.py --status
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
REGISTRY_PATH = DATA_DIR / "phase4" / "model_registry.json"

ROLLING_GATE = {"avg_rank_ic": 0.04, "avg_spread": 0.012,
                "rank_ic_pos_pct": 0.65, "spread_pos_pct": 0.65}
EXPOSURE_GATE = {"max_stock_weight": 0.08, "max_industry_weight": 0.40}


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"models": {}, "champion": None, "shadow": None, "updated_at": None}


def save_registry(reg: dict):
    reg["updated_at"] = datetime.now().isoformat(timespec="seconds")
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=2, ensure_ascii=False))
    os.replace(tmp, REGISTRY_PATH)


def find_latest(pattern: str) -> Path | None:
    candidates = list(DATA_DIR.glob(pattern)) + list((DATA_DIR / "phase4").glob(pattern))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def check_rolling(model_id: str) -> tuple[dict, bool]:
    path = find_latest("fast_rolling_gate*.json") or find_latest("phase4_rolling_gate*.json")
    if not path:
        return {"error": "No rolling results"}, False

    data = json.loads(path.read_text())
    splits = data.get("splits", [])

    # Try "all" key first (non-ablation mode), then "base+extra"
    key = "all" if splits and "all" in splits[0] else "base+extra" if splits and "base+extra" in splits[0] else None
    if key:
        rics = [s[key]["rank_ic_mean"] for s in splits]
        sprs = [s[key]["top20_spread"] for s in splits]
    else:
        return {"error": "Cannot parse rolling splits"}, False

    n = len(rics)
    agg = {
        "avg_rank_ic": sum(rics) / n,
        "avg_spread": sum(sprs) / n,
        "rank_ic_pos_pct": sum(1 for r in rics if r > 0) / n,
        "spread_pos_pct": sum(1 for s in sprs if s > 0) / n,
    }

    checks = {k: {"value": round(agg[k], 4), "threshold": v, "pass": agg[k] >= v}
              for k, v in ROLLING_GATE.items()}
    return {"source": path.name, "n_splits": n, "checks": checks}, all(c["pass"] for c in checks.values())


def check_exposure(model_id: str) -> tuple[dict, bool]:
    path = DATA_DIR / "phase4" / "exposure_report.json"
    if not path.exists():
        return {"error": "No exposure report"}, False

    data = json.loads(path.read_text())
    sc = data.get("stock_concentration", {})
    ie = data.get("industry_exposure", {})

    checks = {
        "stock_weight": {"value": sc.get("max_max_weight", 1),
                         "pass": sc.get("max_max_weight", 1) <= EXPOSURE_GATE["max_stock_weight"]},
        "industry_weight": {"value": ie.get("max_single_industry_weight", 1),
                            "pass": ie.get("max_single_industry_weight", 1) <= EXPOSURE_GATE["max_industry_weight"]},
    }
    return {"source": "exposure_report.json", "checks": checks}, all(c["pass"] for c in checks.values())


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--promote", type=str, choices=["shadow", "champion", "reject"])
    group.add_argument("--status", action="store_true")
    parser.add_argument("--model", type=str, default="xgb_205")
    args = parser.parse_args()

    reg = load_registry()

    if args.status:
        logger.info(f"Champion: {reg.get('champion', 'none')}")
        logger.info(f"Shadow:   {reg.get('shadow', 'none')}")
        for mid, info in reg.get("models", {}).items():
            logger.info(f"  {mid}: {info.get('status', '?')}")
        return

    if args.check:
        logger.info(f"=== PROMOTION GATE: {args.model} ===\n")

        r_report, r_pass = check_rolling(args.model)
        e_report, e_pass = check_exposure(args.model)

        logger.info("Track A (Rolling):")
        if "checks" in r_report:
            for k, v in r_report["checks"].items():
                logger.info(f"  {k}: {v['value']:.4f} (>= {v['threshold']}) {'✅' if v['pass'] else '❌'}")
        else:
            logger.info(f"  {r_report.get('error')}")

        logger.info("\nTrack C (Exposure):")
        if "checks" in e_report:
            for k, v in e_report["checks"].items():
                logger.info(f"  {k}: {v['value']:.4f} {'✅' if v['pass'] else '❌'}")
        else:
            logger.info(f"  {e_report.get('error')}")

        all_pass = r_pass and e_pass
        logger.info(f"\nOverall: {'✅ ELIGIBLE' if all_pass else '❌ NOT ELIGIBLE'}")

        reg.setdefault("models", {})[args.model] = {
            "status": "eligible" if all_pass else "research_only",
            "checked_at": datetime.now().isoformat(timespec="seconds"),
            "rolling_pass": r_pass, "exposure_pass": e_pass,
        }
        save_registry(reg)

    elif args.promote:
        if args.promote == "shadow":
            reg["shadow"] = args.model
            reg.setdefault("models", {})[args.model] = {
                "status": "shadow",
                "shadow_since": datetime.now().isoformat(timespec="seconds"),
            }
            logger.info(f"✅ {args.model} → SHADOW")

        elif args.promote == "champion":
            old = reg.get("champion")
            reg["champion"] = args.model
            reg.setdefault("models", {})[args.model] = {
                "status": "champion",
                "promoted_at": datetime.now().isoformat(timespec="seconds"),
            }
            if old and old in reg.get("models", {}):
                reg["models"][old]["status"] = "shadow"
            logger.info(f"✅ {args.model} → CHAMPION (prev: {old})")

        elif args.promote == "reject":
            reg.setdefault("models", {})[args.model] = {"status": "rejected"}
            logger.info(f"❌ {args.model} → REJECTED")

        save_registry(reg)


if __name__ == "__main__":
    main()
