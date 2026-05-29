"""Shadow comparison: supply chain overlay vs pure XGB Top20.

Daily script that compares Top20 stock picks with and without
the global supply chain overlay.  Now computes 3 separate shadows:

  Shadow A (positive):     only events with direction > 0
  Shadow B (negative/risk): only events with direction < 0
  Shadow C (propagation):  full propagation including industry-level

For each shadow, compares Top20 with vs without that overlay and
saves results in the daily JSON.

Pipeline:
  1. Load today's XGB predictions from lgb_latest_predictions.json
  2. Load global_chain_factors.parquet for today
  3. For each shadow variant, compute overlay and compare Top20
  4. Save comparison to data/storage/shadow_chain_overlay/YYYY-MM-DD.json
  5. Print which stocks moved in/out per shadow

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

WEIGHT_GRID = [0.0, 0.02, 0.05, 0.10, 0.20]
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


def load_chain_factors(target_date: str) -> pd.DataFrame:
    """Load global_chain_factors for target_date from parquet.

    Returns a DataFrame with columns:
        global_chain_alpha, global_chain_pos_score, global_chain_neg_score, ...
    Indexed by instrument.
    """
    if not CHAIN_FACTORS_PATH.exists():
        logger.warning("global_chain_factors.parquet not found -- no overlay available")
        return pd.DataFrame()

    df = pd.read_parquet(CHAIN_FACTORS_PATH)
    if df.empty:
        return pd.DataFrame()

    dt = pd.Timestamp(target_date)
    dates = df.index.get_level_values("datetime")

    if dt in dates:
        chain_today = df.xs(dt, level="datetime")
    else:
        # Fall back to the latest available date
        latest = dates.max()
        logger.warning("No chain factors for %s, using latest: %s", target_date, latest)
        chain_today = df.xs(latest, level="datetime")

    chain_today.index = chain_today.index.str.upper()
    logger.info("Loaded chain factors: %d stocks", len(chain_today))
    return chain_today


# ---------------------------------------------------------------------------
# Single shadow comparison
# ---------------------------------------------------------------------------

def _compare_single_shadow(
    xgb_preds: pd.Series,
    shadow_alpha: pd.Series,
    shadow_name: str,
    overlay_weight: float,
) -> dict:
    """Compare Top-N with and without a single shadow overlay.

    Args:
        xgb_preds: XGB prediction scores (higher = better).
        shadow_alpha: per-stock alpha for this shadow variant.
        shadow_name: label for this shadow (e.g. "shadow_a_positive").
        overlay_weight: weight to apply to the shadow alpha z-score.

    Returns:
        Dict with top20_changed list, n_affected count, and overlay_top.
    """
    baseline_top = set(xgb_preds.nlargest(TOP_N).index)

    z_xgb = safe_zscore(xgb_preds)
    final_score = z_xgb.copy()

    common = xgb_preds.index.intersection(shadow_alpha.index)
    if len(common) > 0:
        z_shadow = safe_zscore(shadow_alpha.reindex(xgb_preds.index).fillna(0.0))
        final_score = z_xgb + overlay_weight * z_shadow
    else:
        logger.warning("No common stocks for %s -- overlay is a no-op", shadow_name)

    overlay_top = set(final_score.nlargest(TOP_N).index)

    added = overlay_top - baseline_top
    removed = baseline_top - overlay_top

    def stock_detail(code: str) -> dict:
        return {
            "code": code,
            "xgb_pred": round(float(xgb_preds.get(code, 0)), 6),
            "xgb_zscore": round(float(z_xgb.get(code, 0)), 4),
            "final_score": round(float(final_score.get(code, 0)), 4),
            "shadow_alpha": round(float(shadow_alpha.get(code, 0)), 6) if code in shadow_alpha.index else None,
            "xgb_rank": int(xgb_preds.rank(ascending=False).get(code, 0)),
            "overlay_rank": int(final_score.rank(ascending=False).get(code, 0)),
        }

    top20_changed = []
    for c in sorted(added):
        d = stock_detail(c)
        d["change"] = "added"
        top20_changed.append(d)
    for c in sorted(removed):
        d = stock_detail(c)
        d["change"] = "removed"
        top20_changed.append(d)

    return {
        "top20_changed": top20_changed,
        "n_affected": len(added) + len(removed),
        "overlay_top": sorted(overlay_top),
    }


# ---------------------------------------------------------------------------
# Core: compute 3 shadows
# ---------------------------------------------------------------------------

def compute_three_shadows(
    xgb_preds: pd.Series,
    chain_df: pd.DataFrame,
    target_date: str,
) -> dict:
    """Compute three shadow overlays across a weight grid.

    For each weight in WEIGHT_GRID, computes all 3 shadows:
      Shadow A (positive): only positive events (global_chain_pos_score)
      Shadow B (negative): only negative events (global_chain_neg_score)
      Shadow C (propagation): full alpha (global_chain_alpha, includes industry)

    Returns:
        Result dict ready for JSON serialization, with per-weight variants.
    """
    baseline_top = sorted(xgb_preds.nlargest(TOP_N).index)

    # --- Build shadow alpha series ---
    if chain_df.empty:
        alpha_pos = pd.Series(dtype=float)
        alpha_neg = pd.Series(dtype=float)
        alpha_full = pd.Series(dtype=float)
    else:
        alpha_pos = chain_df.get("global_chain_pos_score", pd.Series(dtype=float))
        # neg_score is stored as negative values; keep as-is so that stocks
        # with large negative exposure get penalised in the overlay
        alpha_neg = chain_df.get("global_chain_neg_score", pd.Series(dtype=float))
        alpha_full = chain_df.get("global_chain_alpha", pd.Series(dtype=float))

    shadow_variants = ("shadow_a_positive", "shadow_b_negative", "shadow_c_propagation")
    alpha_map = {
        "shadow_a_positive": alpha_pos,
        "shadow_b_negative": alpha_neg,
        "shadow_c_propagation": alpha_full,
    }

    # Compute each shadow x each weight
    weight_results = {}
    for w in WEIGHT_GRID:
        w_key = f"w_{w:.2f}"
        w_entry = {}
        for sname in shadow_variants:
            res = _compare_single_shadow(xgb_preds, alpha_map[sname], sname,
                                         overlay_weight=w)
            w_entry[sname] = {
                "top20_changed": res["top20_changed"],
                "n_affected": res["n_affected"],
                "overlay_top20": res["overlay_top"],
            }
        weight_results[w_key] = w_entry

    result = {
        "date": target_date,
        "xgb_top20": baseline_top,
        "weight_grid": WEIGHT_GRID,
        "variants": weight_results,
        # Metadata
        "top_n": TOP_N,
        "n_xgb_stocks": len(xgb_preds),
        "n_chain_stocks": len(chain_df),
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
    """Print human-readable summary for all weight x shadow variants."""
    print("\n" + "=" * 70)
    print(f"Shadow Supply Chain Overlay (weight grid) -- {result['date']}")
    print("=" * 70)
    print(f"XGB stocks: {result['n_xgb_stocks']}  |  "
          f"Chain stocks: {result['n_chain_stocks']}")
    print(f"Weight grid: {result['weight_grid']}")
    print(f"XGB Top-{result['top_n']}: {', '.join(result['xgb_top20'][:5])} ...")

    shadow_labels = [
        ("shadow_a_positive", "Shadow A (positive events)"),
        ("shadow_b_negative", "Shadow B (negative/risk events)"),
        ("shadow_c_propagation", "Shadow C (full propagation)"),
    ]

    for w_key, w_entry in result.get("variants", {}).items():
        print(f"\n--- Weight: {w_key} ---")
        for key, label in shadow_labels:
            shadow = w_entry.get(key, {})
            n_aff = shadow.get("n_affected", 0)
            print(f"  {label}:  n_affected={n_aff}")
            if shadow.get("top20_changed"):
                for s in shadow["top20_changed"]:
                    alpha_str = (f"alpha={s['shadow_alpha']:+.4f}"
                                 if s["shadow_alpha"] is not None else "alpha=N/A")
                    print(f"    {s['change']:>7s}  {s['code']:12s}  "
                          f"xgb_rank={s['xgb_rank']:>5d} -> overlay_rank={s['overlay_rank']:>5d}  "
                          f"{alpha_str}")
            else:
                print("    No changes -- overlay had no effect on Top-20.")

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
        logger.error("No XGB predictions -- cannot compare")
        sys.exit(1)

    # 2. Load chain factors (full DataFrame now, not just alpha series)
    chain_df = load_chain_factors(target_date)
    if chain_df.empty:
        logger.warning("No chain factors -- saving baseline-only comparison")
        chain_df = pd.DataFrame()

    # 3. Compute 3 shadows
    result = compute_three_shadows(xgb_preds, chain_df, target_date)

    # 4. Save
    save_result(result, target_date)

    # 5. Print
    print_summary(result)


if __name__ == "__main__":
    main()
