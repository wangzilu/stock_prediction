#!/usr/bin/env python3
"""Build feature_cache_205+holder_decrease and validate marginal improvement.

Steps:
  1. Load existing 174 champion cache
  2. Load st_holder_number.parquet, compute holder_decrease
  3. PIT-safe asof merge onto the cache grid
  4. Save as feature_cache_205_plus_holder_decrease.parquet
  5. Fast 6-split XGB comparison: 175 vs 174
  6. If 175 wins >= 4/6, run full 24-split
  7. Save ExperimentArtifact

Usage:
    python scripts/build_feature_cache_175.py
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STORAGE = PROJECT_ROOT / "data" / "storage"
CACHE_174_PATH = STORAGE / "feature_cache_174_holder_regime_ma.parquet"
HOLDER_PATH = STORAGE / "st_holder_number.parquet"
CACHE_PLUS_PATH = STORAGE / "feature_cache_205_plus_holder_decrease.parquet"
LABEL_COL = "__label_5d"
SEED = 42

# XGB params for fair marginal comparison (NOT exact champion params).
# Champion uses max_depth=8, reg_alpha/reg_lambda. Same params for both
# sides ensures delta is due to the new feature, not hyperparams.
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "tree_method": "hist",
    "random_state": 42,
}


# ---------------------------------------------------------------------------
# Step 1-4: Build the 175 cache
# ---------------------------------------------------------------------------

def build_175_cache() -> pd.DataFrame:
    """Load 174 cache, compute ext_holder_decrease, merge, save."""
    logger.info("=== Step 1: Loading 174 champion cache ===")
    cache = pd.read_parquet(str(CACHE_174_PATH))
    logger.info(f"  Shape: {cache.shape}")
    logger.info(f"  Date range: {cache.index.get_level_values(0).min().date()} ~ "
                f"{cache.index.get_level_values(0).max().date()}")

    logger.info("\n=== Step 2: Loading st_holder_number.parquet ===")
    df = pd.read_parquet(str(HOLDER_PATH))
    logger.info(f"  Raw shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")

    df["holder_num"] = pd.to_numeric(df.get("holder_num"), errors="coerce")
    df["ann_date"] = pd.to_datetime(df["ann_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["ann_date", "holder_num"])
    df["qlib_code"] = df["qlib_code"].str.upper()

    # Sort and dedup
    df = df.sort_values(["qlib_code", "ann_date"]).drop_duplicates(
        ["qlib_code", "ann_date"], keep="last"
    )

    logger.info("\n=== Step 3: Computing ext_holder_decrease ===")
    # pct_change per stock, then negate (decrease = bullish signal)
    df["holder_pct_chg"] = df.groupby("qlib_code")["holder_num"].pct_change()
    df["ext_holder_decrease"] = -df["holder_pct_chg"]
    df = df.dropna(subset=["ext_holder_decrease"])
    logger.info(f"  Non-null ext_holder_decrease: {len(df):,}")

    # PIT safety: ann_date + 1 BDay (same as feature_merger._load_st_holder_number)
    df["pit_date"] = df["ann_date"] + pd.tseries.offsets.BDay(1)

    logger.info("\n=== Step 4: Asof-merge onto cache grid ===")
    # We need to forward-fill this quarterly signal onto every trading day.
    # Strategy: unstack by stock, reindex to cache dates, ffill.

    # Prepare sparse factor DataFrame
    factor_sparse = df[["qlib_code", "pit_date", "ext_holder_decrease"]].copy()
    factor_sparse = factor_sparse.drop_duplicates(["qlib_code", "pit_date"], keep="last")

    # Get cache dates and instruments
    cache_dates = sorted(cache.index.get_level_values(0).unique())
    cache_instruments = sorted(cache.index.get_level_values(1).unique())
    inst_upper = {str(inst).upper(): inst for inst in cache_instruments}

    # Map qlib_code to cache instrument names
    factor_sparse["instrument"] = factor_sparse["qlib_code"].map(inst_upper)
    factor_sparse = factor_sparse.dropna(subset=["instrument"])
    logger.info(f"  Matched instruments: {factor_sparse['instrument'].nunique()} / "
                f"{factor_sparse['qlib_code'].nunique()}")

    # Pivot to (date x instrument) for easy ffill
    pivot = factor_sparse.pivot_table(
        index="pit_date", columns="instrument",
        values="ext_holder_decrease", aggfunc="last"
    )

    # Combine with cache dates, sort, ffill
    all_dates_idx = pd.DatetimeIndex(cache_dates)
    combined_idx = pivot.index.union(all_dates_idx).sort_values()
    pivot_filled = pivot.reindex(combined_idx).ffill()

    # Slice to cache dates only
    pivot_filled = pivot_filled.reindex(all_dates_idx)

    # Stack back to MultiIndex
    try:
        factor_series = pivot_filled.stack(future_stack=True)
    except TypeError:
        factor_series = pivot_filled.stack(dropna=False)
    factor_series.index.names = ["datetime", "instrument"]
    factor_series.name = "ext_holder_decrease"

    # Join onto cache
    cache_175 = cache.copy()
    cache_175 = cache_175.join(factor_series, how="left")

    n_nonnull = cache_175["ext_holder_decrease"].notna().sum()
    n_total = len(cache_175)
    logger.info(f"  ext_holder_decrease coverage: {n_nonnull:,} / {n_total:,} "
                f"({n_nonnull/n_total:.1%})")

    # Save
    cache_175.to_parquet(str(CACHE_PLUS_PATH))
    size_mb = CACHE_PLUS_PATH.stat().st_size / 1024 / 1024
    logger.info(f"\n  Saved: {CACHE_PLUS_PATH}")
    logger.info(f"  Shape: {cache_175.shape}")
    logger.info(f"  Size: {size_mb:.1f} MB")

    feature_cols = [c for c in cache_175.columns if not c.startswith("__")]
    meta_cols = [c for c in cache_175.columns if c.startswith("__")]
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Meta: {meta_cols}")

    return cache_175


# ---------------------------------------------------------------------------
# XGB training and evaluation
# ---------------------------------------------------------------------------

def train_xgb_sklearn(X_train, y_train, X_valid, y_valid):
    """Train XGB with the specified params using sklearn API."""
    from xgboost import XGBRegressor

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    return model


def evaluate_split(pred, label, index):
    """Compute RankIC and Top20 spread for a single split."""
    from scipy.stats import spearmanr

    mask = np.isfinite(pred) & np.isfinite(label)
    ps = pd.Series(pred[mask], index=index[mask])
    ls = pd.Series(label[mask], index=index[mask])

    ric_vals = []
    spreads = []
    for date in ps.index.get_level_values(0).unique():
        p = ps.loc[date].values
        l_vals = ls.loc[date].values
        n = len(p)
        if n < 40:
            continue
        corr, _ = spearmanr(p, l_vals)
        if np.isfinite(corr):
            ric_vals.append(corr)
        k = min(20, n // 2)
        top_idx = np.argpartition(p, -k)[-k:]
        bot_idx = np.argpartition(p, k)[:k]
        spreads.append(l_vals[top_idx].mean() - l_vals[bot_idx].mean())

    ric = np.array(ric_vals) if ric_vals else np.array([0.0])
    return {
        "rank_ic_mean": round(float(np.nanmean(ric)), 6),
        "rank_ic_pos": round(float(np.nanmean(ric > 0)), 4) if len(ric) > 0 else 0,
        "top20_spread": round(float(np.mean(spreads)), 6) if spreads else 0,
        "spread_pos": round(float(np.mean([s > 0 for s in spreads])), 4) if spreads else 0,
    }


def run_rolling_comparison(cache: pd.DataFrame, n_splits: int, test_days: int,
                           valid_days: int, train_days: int, label: str):
    """Run XGB174 vs XGB175 rolling comparison.

    Returns list of per-split results.
    """
    from xgboost import XGBRegressor

    # Identify feature columns
    # Exclude: __ meta columns AND _ prefixed auxiliary columns (_close, _ma5, _ma20)
    # These are not XGB input features per config/feature_path.py
    EXCLUDE_PREFIXES = ("__", "_")
    all_feature_cols = [c for c in cache.columns
                        if not any(c.startswith(p) for p in EXCLUDE_PREFIXES)]
    cols_base = [c for c in all_feature_cols if c != "ext_holder_decrease"]
    cols_plus = all_feature_cols  # base + ext_holder_decrease

    # NOTE: These are NOT the exact champion hyperparams (champion uses
    # max_depth=8, reg_alpha/reg_lambda). Same params for both sides
    # ensures fair marginal comparison, but absolute IC values differ
    # from champion. See config/feature_path.py for true champion config.
    logger.info(f"  Base features (excl _close/_ma5/_ma20): {len(cols_base)}")
    logger.info(f"  Plus features (+ext_holder_decrease): {len(cols_plus)}")

    # Get trading dates
    trade_dates = sorted(cache.index.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1
    dates_level = cache.index.get_level_values(0)

    split_results = []

    for split_idx in range(n_splits):
        test_end_idx = today_idx - split_idx * test_days
        test_start_idx = test_end_idx - test_days
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - valid_days
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - train_days

        if train_start_idx < 0:
            logger.warning(f"  Split {split_idx}: not enough data, stopping")
            break

        test_end = trade_dates[test_end_idx]
        test_start = trade_dates[test_start_idx]
        valid_start = trade_dates[valid_start_idx]
        valid_end = trade_dates[valid_end_idx]
        train_start = trade_dates[train_start_idx]
        train_end = trade_dates[train_end_idx]

        train_mask = (dates_level >= train_start) & (dates_level <= train_end)
        valid_mask = (dates_level >= valid_start) & (dates_level <= valid_end)
        test_mask = (dates_level >= test_start) & (dates_level <= test_end)

        y_tr = cache.loc[train_mask, label].values.astype(np.float32)
        y_va = cache.loc[valid_mask, label].values.astype(np.float32)
        y_te = cache.loc[test_mask, label].values.astype(np.float32)
        test_idx = cache.index[test_mask]

        m_tr = np.isfinite(y_tr)
        m_va = np.isfinite(y_va)
        m_te = np.isfinite(y_te)

        result = {
            "split": split_idx + 1,
            "test": f"{str(test_start)[:10]}~{str(test_end)[:10]}",
        }

        for name, cols in [("base", cols_base), ("plus_holder", cols_plus)]:
            t1 = time.time()
            X_tr = cache.loc[train_mask, cols].values.astype(np.float32)
            X_va = cache.loc[valid_mask, cols].values.astype(np.float32)
            X_te = cache.loc[test_mask, cols].values.astype(np.float32)

            model = train_xgb_sklearn(X_tr[m_tr], y_tr[m_tr], X_va[m_va], y_va[m_va])
            pred = model.predict(X_te[m_te])

            metrics = evaluate_split(pred, y_te[m_te], test_idx[m_te])
            elapsed = time.time() - t1
            result[name] = {**metrics, "n_feat": len(cols), "time_s": round(elapsed, 1)}

        # Delta
        result["delta_rank_ic"] = round(
            result["xgb175"]["rank_ic_mean"] - result["xgb174"]["rank_ic_mean"], 6)
        result["delta_spread"] = round(
            result["xgb175"]["top20_spread"] - result["xgb174"]["top20_spread"], 6)

        split_results.append(result)
        logger.info(
            f"  Split {result['split']:2d}  [{result['test']}]  "
            f"174={result['xgb174']['rank_ic_mean']:+.4f}  "
            f"175={result['xgb175']['rank_ic_mean']:+.4f}  "
            f"delta={result['delta_rank_ic']:+.4f}"
        )

    return split_results


def print_summary(results: list, tag: str):
    """Print a summary table of results."""
    n = len(results)
    ric_174 = [r["xgb174"]["rank_ic_mean"] for r in results]
    ric_175 = [r["xgb175"]["rank_ic_mean"] for r in results]
    deltas = [r["delta_rank_ic"] for r in results]

    spr_174 = [r["xgb174"]["top20_spread"] for r in results]
    spr_175 = [r["xgb175"]["top20_spread"] for r in results]
    delta_spr = [r["delta_spread"] for r in results]

    wins_ric = sum(1 for d in deltas if d > 0)
    wins_spr = sum(1 for d in delta_spr if d > 0)

    print(f"\n{'='*80}")
    print(f"  {tag} — XGB174 vs XGB175 Comparison ({n} splits)")
    print(f"{'='*80}")

    # Per-split table
    print(f"\n{'Split':>6} {'Test Window':<25} {'RankIC-174':>11} {'RankIC-175':>11} {'Delta':>8}")
    print("-" * 65)
    for r in results:
        print(f"{r['split']:>6} {r['test']:<25} "
              f"{r['xgb174']['rank_ic_mean']:>+11.4f} "
              f"{r['xgb175']['rank_ic_mean']:>+11.4f} "
              f"{r['delta_rank_ic']:>+8.4f}")

    print(f"\n--- Aggregate ---")
    print(f"  XGB174: avg RankIC = {np.mean(ric_174):+.4f}, avg Spread = {np.mean(spr_174)*100:+.3f}%")
    print(f"  XGB175: avg RankIC = {np.mean(ric_175):+.4f}, avg Spread = {np.mean(spr_175)*100:+.3f}%")
    print(f"  Delta:  avg RankIC = {np.mean(deltas):+.4f}, avg Spread = {np.mean(delta_spr)*100:+.3f}%")
    print(f"  175 wins (RankIC): {wins_ric}/{n}")
    print(f"  175 wins (Spread): {wins_spr}/{n}")

    return wins_ric, wins_spr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(force_rebuild: bool = False):
    from tracker.artifact_contract import ExperimentArtifact

    t_start = time.time()

    # ------- Build cache -------
    if CACHE_PLUS_PATH.exists() and not force_rebuild:
        logger.info(f"Cache already exists: {CACHE_PLUS_PATH}")
        logger.info("Loading existing cache... (use --force to rebuild)")
        cache = pd.read_parquet(str(CACHE_PLUS_PATH))
        logger.info(f"  Shape: {cache.shape}")
    else:
        if force_rebuild and CACHE_PLUS_PATH.exists():
            logger.info("Force rebuild requested — regenerating cache")
        cache = build_175_cache()

    # Verify ext_holder_decrease exists
    assert "ext_holder_decrease" in cache.columns, "ext_holder_decrease not in cache!"
    feature_cols = [c for c in cache.columns if not c.startswith("__")]
    logger.info(f"\nFeature columns ({len(feature_cols)}): "
                f"{[c for c in feature_cols if 'holder' in c.lower() or 'decrease' in c.lower()]}")

    # ------- Fast 6-split gate -------
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: Fast 6-split validation (test_days=60, train_days=480)")
    logger.info("=" * 80)

    results_6 = run_rolling_comparison(
        cache, n_splits=6, test_days=60, valid_days=60, train_days=480,
        label=LABEL_COL,
    )

    wins_6, wins_spr_6 = print_summary(results_6, "6-Split Fast Gate")

    if wins_6 < 4:
        print(f"\n  >>> 6-split: 175 wins {wins_6}/{len(results_6)} (need >= 4).")
        print("  Proceeding to full 24-split anyway for definitive result.")

    results_6_ref = results_6  # save for artifact

    # ------- Full 24-split -------
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Full 24-split validation")
    logger.info("=" * 80)

    results_24 = run_rolling_comparison(
        cache, n_splits=24, test_days=20, valid_days=60, train_days=750,
        label=LABEL_COL,
    )

    wins_24, wins_spr_24 = print_summary(results_24, "24-Split Full Gate")

    # Gate: 175 wins >= 13/24 on RankIC
    gate_pass = wins_24 >= 13
    verdict = "PASS" if gate_pass else "FAIL"

    ric_174_all = [r["xgb174"]["rank_ic_mean"] for r in results_24]
    ric_175_all = [r["xgb175"]["rank_ic_mean"] for r in results_24]

    print(f"\n  >>> 24-split VERDICT: {verdict} — 175 wins {wins_24}/24 <<<")
    if gate_pass:
        print("  CONCLUSION: ext_holder_decrease ADDS marginal value. Promoting to champion.")
    else:
        print("  CONCLUSION: ext_holder_decrease insufficient. 174 remains champion.")

    # Save artifact
    exp_id = f"xgb175_24split_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    art = ExperimentArtifact.create(
        experiment_id=exp_id,
        model_name="xgb_175",
        feature_set="FS-175",
        description="24-split rolling gate: XGB175 (174 + ext_holder_decrease) vs XGB174",
        gate_verdict=verdict,
        n_splits=24,
        xgb_params=XGB_PARAMS,
    )
    art.save_metrics({
        "gate_verdict": verdict,
        "n_splits": len(results_24),
        "wins_rank_ic": wins_24,
        "wins_spread": wins_spr_24,
        "xgb174_avg_rank_ic": round(float(np.mean(ric_174_all)), 6),
        "xgb175_avg_rank_ic": round(float(np.mean(ric_175_all)), 6),
        "delta_avg_rank_ic": round(float(np.mean(ric_175_all) - np.mean(ric_174_all)), 6),
        "rank_ic_mean": round(float(np.mean(ric_175_all)), 6),
        "rank_ic_std": round(float(np.std(ric_175_all)), 6),
        "per_split_6": results_6_ref,
        "per_split_24": results_24,
    })
    art.validate()
    logger.info(f"Artifact saved: {art.artifact_dir}")

    total_time = time.time() - t_start
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # Return verdict for config update
    return verdict, wins_24, results_24


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force rebuild cache even if exists")
    args = parser.parse_args()

    result = main(force_rebuild=args.force)

    # Step 4: Update config/feature_path.py guidance
    if result is not None:
        verdict, wins, _ = result
        if verdict == "PASS":
            print("\n" + "=" * 80)
            print("RESULT: Delta positive — BUT still research_only.")
            print("Before promotion, must also:")
            print("  1. Fix feature builder (prepare_features_174 doesn't produce ext_holder_decrease)")
            print("  2. Verify with true champion params (max_depth=8, reg_alpha/reg_lambda)")
            print("  3. Run full backtest with optimizer_v2 (not just IC)")
            print("  4. Shadow 20 days paper trading")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("NO ACTION: base features remain champion. +holder_decrease is research_only.")
            print("=" * 80)
