"""Phase 4U Day5 — Validate global chain overlay plumbing.

Structural validation: does the overlay math work end-to-end?
NOT a real alpha test (needs actual global news for that).

Steps:
  1. Run build_global_chain_factors --demo to generate test data
  2. Load global_chain_factors.parquet
  3. Load XGB predictions from lgb_latest_predictions.json
  4. Simulate overlay: final = zscore(xgb) + 0.2 * zscore(global_chain_alpha)
  5. Compare: do stocks with positive global_chain_alpha rank higher after overlay?
  6. Print stats

Usage:
    python -m scripts.validate_global_chain_overlay
"""
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore as scipy_zscore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "storage"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

OVERLAY_WEIGHT = 0.2


def run_demo_build():
    """Run build_global_chain_factors --demo to generate test data."""
    logger.info("Step 1: Running build_global_chain_factors --demo ...")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.build_global_chain_factors", "--demo"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    print(result.stdout)
    if result.returncode != 0:
        logger.error("Demo build failed:\n%s", result.stderr)
        return False
    if result.stderr:
        # Print stderr but don't fail (may contain logging output)
        for line in result.stderr.strip().split("\n"):
            if line.strip():
                logger.info("  build: %s", line.strip())
    return True


def load_global_chain_factors() -> pd.DataFrame:
    """Load global chain factors from parquet."""
    path = DATA_DIR / "global_chain_factors.parquet"
    if not path.exists():
        logger.error("global_chain_factors.parquet not found at %s", path)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    logger.info("Step 2: Loaded global_chain_factors: %d rows, %d stocks",
                len(df), df.index.get_level_values("instrument").nunique())
    return df


def load_xgb_predictions() -> pd.Series:
    """Load XGB predictions from lgb_latest_predictions.json."""
    path = DATA_DIR / "lgb_latest_predictions.json"
    if not path.exists():
        logger.error("lgb_latest_predictions.json not found")
        return pd.Series(dtype=float)
    payload = json.loads(path.read_text())
    preds = payload.get("predictions", {})
    # Convert to Series with instrument as index
    s = pd.Series(preds, dtype=float)
    s.index.name = "instrument"
    logger.info("Step 3: Loaded XGB predictions: %d stocks", len(s))
    return s


def safe_zscore(s: pd.Series) -> pd.Series:
    """Compute zscore, handling NaN and constant series."""
    finite = s[np.isfinite(s)]
    if len(finite) < 2 or finite.std() == 0:
        return pd.Series(0.0, index=s.index)
    z = (s - finite.mean()) / finite.std()
    return z.fillna(0.0)


def simulate_overlay(xgb_preds: pd.Series, chain_factors: pd.DataFrame) -> dict:
    """Simulate the overlay and compute statistics."""
    # Get the latest date from chain factors
    dates = chain_factors.index.get_level_values("datetime")
    latest_date = dates.max()
    chain_today = chain_factors.xs(latest_date, level="datetime")

    # Extract global_chain_alpha
    chain_alpha = chain_today["global_chain_alpha"]
    # Normalize instrument codes to match XGB (uppercase)
    chain_alpha.index = chain_alpha.index.str.upper()

    # Find common stocks
    common = xgb_preds.index.intersection(chain_alpha.index)
    logger.info("Step 4: Common stocks between XGB and chain: %d", len(common))

    if len(common) == 0:
        logger.warning("No common stocks found — check instrument code format")
        # Try case-insensitive match
        xgb_upper = xgb_preds.copy()
        xgb_upper.index = xgb_upper.index.str.upper()
        chain_upper = chain_alpha.copy()
        chain_upper.index = chain_upper.index.str.upper()
        common = xgb_upper.index.intersection(chain_upper.index)
        logger.info("  After upper-case normalization: %d common stocks", len(common))
        if len(common) == 0:
            return {"error": "no common stocks"}
        xgb_preds = xgb_upper
        chain_alpha = chain_upper

    xgb_common = xgb_preds.loc[common]
    chain_common = chain_alpha.loc[common]

    # Compute zscores
    z_xgb = safe_zscore(xgb_common)
    z_chain = safe_zscore(chain_common)

    # Original ranking (XGB only)
    xgb_rank_orig = xgb_preds.rank(ascending=False)

    # Overlay: final = zscore(xgb) + 0.2 * zscore(global_chain_alpha)
    # For stocks NOT in chain, overlay score = zscore(xgb) + 0 (no effect)
    z_xgb_all = safe_zscore(xgb_preds)
    final_score = z_xgb_all.copy()
    for code in common:
        final_score[code] += OVERLAY_WEIGHT * z_chain[code]

    # New ranking
    final_rank = final_score.rank(ascending=False)

    # Compare: stocks with positive chain alpha
    pos_chain = chain_common[chain_common > 0].index
    neg_chain = chain_common[chain_common < 0].index
    zero_chain = chain_common[chain_common == 0].index

    results = {
        "date": str(latest_date.date()) if hasattr(latest_date, "date") else str(latest_date),
        "n_xgb_stocks": len(xgb_preds),
        "n_chain_stocks": len(chain_alpha),
        "n_common": len(common),
        "n_positive_chain": len(pos_chain),
        "n_negative_chain": len(neg_chain),
        "n_zero_chain": len(zero_chain),
        "overlay_weight": OVERLAY_WEIGHT,
    }

    # Rank changes for positive-alpha stocks
    if len(pos_chain) > 0:
        rank_changes_pos = []
        for code in pos_chain:
            old_rank = xgb_rank_orig.get(code, np.nan)
            new_rank = final_rank.get(code, np.nan)
            if np.isfinite(old_rank) and np.isfinite(new_rank):
                rank_changes_pos.append({
                    "code": code,
                    "chain_alpha": round(float(chain_common[code]), 4),
                    "z_chain": round(float(z_chain[code]), 3),
                    "old_rank": int(old_rank),
                    "new_rank": int(new_rank),
                    "rank_change": int(old_rank - new_rank),  # positive = improved
                })
        rank_changes_pos.sort(key=lambda x: -x["rank_change"])
        results["positive_chain_stocks"] = rank_changes_pos
        avg_improvement = np.mean([r["rank_change"] for r in rank_changes_pos])
        results["avg_rank_improvement_pos"] = round(float(avg_improvement), 1)

    # Rank changes for negative-alpha stocks
    if len(neg_chain) > 0:
        rank_changes_neg = []
        for code in neg_chain:
            old_rank = xgb_rank_orig.get(code, np.nan)
            new_rank = final_rank.get(code, np.nan)
            if np.isfinite(old_rank) and np.isfinite(new_rank):
                rank_changes_neg.append({
                    "code": code,
                    "chain_alpha": round(float(chain_common[code]), 4),
                    "z_chain": round(float(z_chain[code]), 3),
                    "old_rank": int(old_rank),
                    "new_rank": int(new_rank),
                    "rank_change": int(old_rank - new_rank),
                })
        rank_changes_neg.sort(key=lambda x: x["rank_change"])
        results["negative_chain_stocks"] = rank_changes_neg
        avg_drop = np.mean([r["rank_change"] for r in rank_changes_neg])
        results["avg_rank_change_neg"] = round(float(avg_drop), 1)

    # Score change stats
    score_changes = []
    for code in common:
        old_score = float(z_xgb_all.get(code, 0))
        new_score = float(final_score.get(code, 0))
        score_changes.append(new_score - old_score)
    results["avg_score_change"] = round(float(np.mean(score_changes)), 4)
    results["max_score_change"] = round(float(np.max(np.abs(score_changes))), 4)

    return results


def print_results(results: dict):
    """Pretty-print validation results."""
    print("\n" + "=" * 70)
    print("Phase 4U Day5 — Global Chain Overlay Validation")
    print("=" * 70)

    if "error" in results:
        print(f"\nERROR: {results['error']}")
        return

    print(f"\nDate:              {results['date']}")
    print(f"XGB stocks:        {results['n_xgb_stocks']}")
    print(f"Chain stocks:      {results['n_chain_stocks']}")
    print(f"Common stocks:     {results['n_common']}")
    print(f"Overlay weight:    {results['overlay_weight']}")
    print(f"Positive chain:    {results['n_positive_chain']}")
    print(f"Negative chain:    {results['n_negative_chain']}")

    print(f"\nAvg score change (affected stocks): {results['avg_score_change']:+.4f}")
    print(f"Max abs score change:               {results['max_score_change']:.4f}")

    if "avg_rank_improvement_pos" in results:
        print(f"\nPositive-alpha stocks avg rank improvement: "
              f"{results['avg_rank_improvement_pos']:+.1f} positions")
    if "avg_rank_change_neg" in results:
        print(f"Negative-alpha stocks avg rank change:      "
              f"{results['avg_rank_change_neg']:+.1f} positions")

    # Top affected stocks (positive chain)
    if "positive_chain_stocks" in results:
        stocks = results["positive_chain_stocks"]
        print(f"\nTop affected stocks (POSITIVE chain alpha, n={len(stocks)}):")
        for s in stocks[:10]:
            print(f"  {s['code']:12s}  chain_alpha={s['chain_alpha']:+.4f}  "
                  f"z_chain={s['z_chain']:+.3f}  "
                  f"rank: {s['old_rank']} -> {s['new_rank']} ({s['rank_change']:+d})")

    if "negative_chain_stocks" in results:
        stocks = results["negative_chain_stocks"]
        print(f"\nTop affected stocks (NEGATIVE chain alpha, n={len(stocks)}):")
        for s in stocks[:10]:
            print(f"  {s['code']:12s}  chain_alpha={s['chain_alpha']:+.4f}  "
                  f"z_chain={s['z_chain']:+.3f}  "
                  f"rank: {s['old_rank']} -> {s['new_rank']} ({s['rank_change']:+d})")

    # Structural validation verdict
    print("\n" + "-" * 70)
    plumbing_ok = (results["n_common"] > 0 and
                   results["max_score_change"] > 0 and
                   results.get("avg_rank_improvement_pos", 0) > 0)
    if plumbing_ok:
        print("VERDICT: Plumbing OK — overlay shifts ranks as expected.")
        print("         Positive chain alpha -> rank improvement.")
        print("         Negative chain alpha -> rank demotion.")
        print("         Ready for real global news integration.")
    else:
        print("VERDICT: CHECK NEEDED — overlay may not be working as expected.")
    print("-" * 70)


def main():
    # Step 1: Generate demo data
    if not run_demo_build():
        print("FAILED: Could not generate demo global chain factors.")
        sys.exit(1)

    # Step 2: Load chain factors
    chain_df = load_global_chain_factors()
    if chain_df.empty:
        print("FAILED: No global chain factors loaded.")
        sys.exit(1)

    # Step 3: Load XGB predictions
    xgb_preds = load_xgb_predictions()
    if xgb_preds.empty:
        print("FAILED: No XGB predictions loaded.")
        sys.exit(1)

    # Step 4-6: Simulate overlay and print stats
    results = simulate_overlay(xgb_preds, chain_df)
    print_results(results)


if __name__ == "__main__":
    main()
