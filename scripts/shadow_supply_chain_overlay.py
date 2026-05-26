"""Shadow comparison: supply chain overlay vs pure XGB Top20.

Daily script that compares Top20 stock picks with and without
the global supply chain overlay. Shadow-only — does NOT change
actual recommendations.

Pipeline:
  1. Load today's XGB predictions from lgb_latest_predictions.json
  2. Load global_chain_factors.parquet for today
  3. Compute overlay: final_score = zscore(xgb) + 0.2 * zscore(chain_alpha)
  4. Compare Top20 with vs without overlay
  5. Save comparison to data/storage/shadow_chain_overlay/YYYY-MM-DD.json
  6. Print which stocks moved in/out, chain alpha of affected stocks

Usage:
    python scripts/shadow_supply_chain_overlay.py
    python scripts/shadow_supply_chain_overlay.py --date 2026-05-26
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
PREDICTIONS_PATH = DATA_DIR / "lgb_latest_predictions.json"
CHAIN_FACTORS_PATH = DATA_DIR / "global_chain_factors.parquet"
OUTPUT_DIR = DATA_DIR / "shadow_chain_overlay"

OVERLAY_WEIGHT = 0.2
TOP_N = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_zscore(s: pd.Series) -> pd.Series:
    """Compute zscore, handling NaN and constant series."""
    finite = s[np.isfinite(s)]
    if len(finite) < 2 or finite.std() == 0:
        return pd.Series(0.0, index=s.index)
    z = (s - finite.mean()) / finite.std()
    return z.fillna(0.0)


def load_xgb_predictions() -> pd.Series:
    """Load XGB predictions from lgb_latest_predictions.json."""
    if not PREDICTIONS_PATH.exists():
        logger.error("lgb_latest_predictions.json not found at %s", PREDICTIONS_PATH)
        return pd.Series(dtype=float)
    payload = json.loads(PREDICTIONS_PATH.read_text())
    preds = payload.get("predictions", {})
    s = pd.Series(preds, dtype=float)
    s.index.name = "instrument"
    logger.info("Loaded XGB predictions: %d stocks", len(s))
    return s


def load_chain_factors(target_date: str) -> pd.Series:
    """Load global_chain_alpha for target_date from parquet."""
    if not CHAIN_FACTORS_PATH.exists():
        logger.warning("global_chain_factors.parquet not found — no overlay available")
        return pd.Series(dtype=float)

    df = pd.read_parquet(CHAIN_FACTORS_PATH)
    if df.empty:
        return pd.Series(dtype=float)

    dt = pd.Timestamp(target_date)
    dates = df.index.get_level_values("datetime")

    if dt in dates:
        chain_today = df.xs(dt, level="datetime")
    else:
        # Fall back to the latest available date
        latest = dates.max()
        logger.warning("No chain factors for %s, using latest: %s", target_date, latest)
        chain_today = df.xs(latest, level="datetime")

    alpha = chain_today["global_chain_alpha"]
    alpha.index = alpha.index.str.upper()
    logger.info("Loaded chain factors: %d stocks with alpha", len(alpha))
    return alpha


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare_top_n(xgb_preds: pd.Series, chain_alpha: pd.Series, target_date: str) -> dict:
    """Compare Top-N with and without supply chain overlay."""
    # --- Baseline: pure XGB Top-N ---
    baseline_top = set(xgb_preds.nlargest(TOP_N).index)

    # --- Overlay: zscore(xgb) + weight * zscore(chain_alpha) ---
    z_xgb = safe_zscore(xgb_preds)
    final_score = z_xgb.copy()

    common = xgb_preds.index.intersection(chain_alpha.index)
    if len(common) > 0:
        z_chain = safe_zscore(chain_alpha.reindex(xgb_preds.index).fillna(0.0))
        final_score = z_xgb + OVERLAY_WEIGHT * z_chain
    else:
        logger.warning("No common stocks between XGB and chain — overlay is a no-op")

    overlay_top = set(final_score.nlargest(TOP_N).index)

    # --- Diff ---
    added = overlay_top - baseline_top       # moved IN by overlay
    removed = baseline_top - overlay_top     # moved OUT by overlay
    kept = baseline_top & overlay_top

    # Build detail records for added/removed stocks
    def stock_detail(code: str) -> dict:
        d = {
            "code": code,
            "xgb_pred": round(float(xgb_preds.get(code, 0)), 6),
            "xgb_zscore": round(float(z_xgb.get(code, 0)), 4),
            "final_score": round(float(final_score.get(code, 0)), 4),
            "chain_alpha": round(float(chain_alpha.get(code, 0)), 6) if code in chain_alpha.index else None,
            "xgb_rank": int(xgb_preds.rank(ascending=False).get(code, 0)),
            "overlay_rank": int(final_score.rank(ascending=False).get(code, 0)),
        }
        return d

    added_details = sorted([stock_detail(c) for c in added], key=lambda x: x["overlay_rank"])
    removed_details = sorted([stock_detail(c) for c in removed], key=lambda x: x["xgb_rank"])

    result = {
        "date": target_date,
        "overlay_weight": OVERLAY_WEIGHT,
        "top_n": TOP_N,
        "n_xgb_stocks": len(xgb_preds),
        "n_chain_stocks": len(chain_alpha),
        "n_common": len(common),
        "baseline_top": sorted(baseline_top),
        "overlay_top": sorted(overlay_top),
        "overlap_count": len(kept),
        "added_by_overlay": added_details,
        "removed_by_overlay": removed_details,
        "turnover": len(added),
    }
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_result(result: dict, target_date: str) -> Path:
    """Save comparison JSON to shadow_chain_overlay/YYYY-MM-DD.json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{target_date}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Saved comparison to %s", out_path)
    return out_path


def print_summary(result: dict):
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print(f"Shadow Supply Chain Overlay — {result['date']}")
    print("=" * 70)
    print(f"XGB stocks: {result['n_xgb_stocks']}  |  "
          f"Chain stocks: {result['n_chain_stocks']}  |  "
          f"Common: {result['n_common']}")
    print(f"Top-{result['top_n']} overlap: {result['overlap_count']}/{result['top_n']}  |  "
          f"Turnover: {result['turnover']} stocks")

    if result["added_by_overlay"]:
        print(f"\n  ADDED by overlay ({len(result['added_by_overlay'])}):")
        for s in result["added_by_overlay"]:
            chain_str = f"chain_alpha={s['chain_alpha']:+.4f}" if s["chain_alpha"] is not None else "chain_alpha=N/A"
            print(f"    {s['code']:12s}  xgb_rank={s['xgb_rank']:>5d} -> overlay_rank={s['overlay_rank']:>5d}  "
                  f"{chain_str}")

    if result["removed_by_overlay"]:
        print(f"\n  REMOVED by overlay ({len(result['removed_by_overlay'])}):")
        for s in result["removed_by_overlay"]:
            chain_str = f"chain_alpha={s['chain_alpha']:+.4f}" if s["chain_alpha"] is not None else "chain_alpha=N/A"
            print(f"    {s['code']:12s}  xgb_rank={s['xgb_rank']:>5d} -> overlay_rank={s['overlay_rank']:>5d}  "
                  f"{chain_str}")

    if result["turnover"] == 0:
        print("\n  No changes — overlay had no effect on Top-20.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=str, default=None,
                        help="Target date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")

    # 1. Load predictions
    xgb_preds = load_xgb_predictions()
    if xgb_preds.empty:
        logger.error("No XGB predictions — cannot compare")
        sys.exit(1)

    # 2. Load chain factors
    chain_alpha = load_chain_factors(target_date)
    if chain_alpha.empty:
        logger.warning("No chain factors — saving baseline-only comparison")
        chain_alpha = pd.Series(dtype=float)

    # 3-4. Compare
    result = compare_top_n(xgb_preds, chain_alpha, target_date)

    # 5. Save
    save_result(result, target_date)

    # 6. Print
    print_summary(result)


if __name__ == "__main__":
    main()
