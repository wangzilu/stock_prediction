"""Phase 4: Full 24-split rolling validation — raw vs rank-preprocessed.

Compares two feature variants:
1. base_174_raw: all 174 features as-is
2. base_174_custom_flow_ranked: same 174 features, but per-date rank(pct=True)
   applied to the 16 custom+flow columns only

Usage:
    python scripts/phase4_rank_preprocess_24split.py
"""
import json
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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
SEED = 42

# The 16 columns to rank-preprocess
RANK_COLS = [
    "pe", "pb", "turn_raw", "amount_raw",
    "pe_mom20", "pb_mom20", "turn_anom20", "turn_anom60",
    "amount_anom20", "turn_vol20", "ep", "bp", "price_pos20",
    "flow_net_mf_latest", "flow_net_mf_5d", "flow_net_mf_20d_avg",
]

LABEL_COL = "__label_5d"

# Columns to exclude from features
EXCLUDE_PREFIXES = ("__", "_", "hsi_", "hstech_", "nasdaq_")
EXCLUDE_EXACT = {"holder_num"}

XGB_PARAMS = {
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8789,
    "colsample_bytree": 0.8879,
    "reg_alpha": 205.6999,
    "reg_lambda": 580.9768,
    "objective": "reg:squarederror",
    "nthread": 12,
    "verbosity": 0,
    "seed": SEED,
}


def get_feature_cols(columns):
    """Extract base 174 feature columns (excluding label, meta, cross-market, holder)."""
    return [
        c for c in columns
        if not any(c.startswith(p) for p in EXCLUDE_PREFIXES)
        and c not in EXCLUDE_EXACT
    ]


def rank_transform(cache, rank_cols):
    """Per-date rank(pct=True) on specified columns."""
    logger.info(f"Applying per-date rank(pct=True) to {len(rank_cols)} columns...")
    t0 = time.time()
    result = cache.copy()
    for col in rank_cols:
        if col not in result.columns:
            logger.warning(f"  Column {col} not found, skipping")
            continue
        result[col] = result[col].groupby(level=0).rank(pct=True)
    logger.info(f"  Rank transform done in {time.time() - t0:.1f}s")
    return result


def train_xgb(X_train, y_train, X_valid, y_valid):
    import xgboost as xgb
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_valid, label=y_valid)
    return xgb.train(
        XGB_PARAMS, dt, num_boost_round=400,
        evals=[(dv, "valid")], early_stopping_rounds=50, verbose_eval=0,
    )


def evaluate_split(pred, label, index):
    """Compute RankIC, Spread (top20-bottom20), IC for one test split."""
    from scipy.stats import spearmanr, pearsonr

    mask = np.isfinite(pred) & np.isfinite(label)
    ps = pd.Series(pred[mask], index=index[mask])
    ls = pd.Series(label[mask], index=index[mask])

    rics, spreads, ics = [], [], []
    for date in ps.index.get_level_values(0).unique():
        p = ps.loc[date].values
        l = ls.loc[date].values
        if len(p) < 40:
            continue
        # RankIC (Spearman)
        ric, _ = spearmanr(p, l)
        if np.isfinite(ric):
            rics.append(ric)
        # IC (Pearson)
        ic, _ = pearsonr(p, l)
        if np.isfinite(ic):
            ics.append(ic)
        # Spread: top20 - bottom20
        k = min(20, len(p) // 2)
        top_idx = np.argpartition(p, -k)[-k:]
        bot_idx = np.argpartition(p, k)[:k]
        spreads.append(l[top_idx].mean() - l[bot_idx].mean())

    return {
        "rank_ic": round(float(np.nanmean(rics)), 6) if rics else 0.0,
        "spread": round(float(np.mean(spreads)), 6) if spreads else 0.0,
        "ic": round(float(np.nanmean(ics)), 6) if ics else 0.0,
    }


def main():
    import xgboost as xgb

    n_splits = 24
    test_days = 20
    valid_days = 60
    train_days = 750

    # Load cache
    cache_path = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
    logger.info(f"Loading cache: {cache_path}")
    cache = pd.read_parquet(str(cache_path))
    logger.info(f"  Shape: {cache.shape}")

    # Feature columns
    feature_cols = get_feature_cols(cache.columns)
    rank_cols_present = [c for c in RANK_COLS if c in feature_cols]
    logger.info(f"  Feature cols: {len(feature_cols)}")
    logger.info(f"  Rank cols present: {len(rank_cols_present)} -> {rank_cols_present}")

    # Build two variants
    cache_ranked = rank_transform(cache, rank_cols_present)

    variants = {
        "base_174_raw": cache,
        "base_174_custom_flow_ranked": cache_ranked,
    }

    # Rolling splits
    trade_dates = sorted(cache.index.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1
    dates_level = cache.index.get_level_values(0)

    # Per-variant per-split storage
    split_results = {name: [] for name in variants}
    all_splits_info = []

    t_total = time.time()

    for split_idx in range(n_splits):
        test_end_idx = today_idx - split_idx * test_days
        test_start_idx = test_end_idx - test_days
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - valid_days
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - train_days
        if train_start_idx < 0:
            logger.warning(f"Split {split_idx + 1}: not enough data, stopping at {split_idx} splits")
            break

        tm = (dates_level >= trade_dates[train_start_idx]) & (dates_level <= trade_dates[train_end_idx])
        vm = (dates_level >= trade_dates[valid_start_idx]) & (dates_level <= trade_dates[valid_end_idx])
        em = (dates_level >= trade_dates[test_start_idx]) & (dates_level <= trade_dates[test_end_idx])

        y_tr = cache[LABEL_COL].values[tm].astype(np.float32)
        y_va = cache[LABEL_COL].values[vm].astype(np.float32)
        y_te = cache[LABEL_COL].values[em].astype(np.float32)
        mtr = np.isfinite(y_tr)
        mva = np.isfinite(y_va)
        mte = np.isfinite(y_te)
        test_idx = cache.index[em]

        split_info = {
            "split": split_idx + 1,
            "test_start": str(trade_dates[test_start_idx])[:10],
            "test_end": str(trade_dates[test_end_idx])[:10],
        }

        logger.info(f"\nSplit {split_idx + 1}/{n_splits} "
                     f"[test: {split_info['test_start']} ~ {split_info['test_end']}]:")

        for name, data_source in variants.items():
            t1 = time.time()
            X_tr = data_source.loc[tm, feature_cols].values.astype(np.float32)
            X_va = data_source.loc[vm, feature_cols].values.astype(np.float32)
            X_te = data_source.loc[em, feature_cols].values.astype(np.float32)

            model = train_xgb(X_tr[mtr], y_tr[mtr], X_va[mva], y_va[mva])
            pred = model.predict(xgb.DMatrix(X_te[mte]))
            metrics = evaluate_split(pred, y_te[mte], test_idx[mte])
            elapsed = time.time() - t1

            metrics["time_s"] = round(elapsed, 1)
            split_results[name].append(metrics)
            split_info[name] = metrics

            logger.info(f"  {name}: RankIC={metrics['rank_ic']:+.4f}  "
                         f"Spread={metrics['spread']*100:+.3f}%  "
                         f"IC={metrics['ic']:+.4f}  [{elapsed:.0f}s]")

        all_splits_info.append(split_info)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    total_time = time.time() - t_total
    n = len(all_splits_info)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"RANK PREPROCESS 24-SPLIT VALIDATION ({n} splits, {total_time:.0f}s)")
    logger.info(f"{'=' * 70}")

    summary = {}
    for name in variants:
        rics = [r["rank_ic"] for r in split_results[name]]
        sprs = [r["spread"] for r in split_results[name]]
        ic_vals = [r["ic"] for r in split_results[name]]

        # Worst-3 average
        worst3_ric = sorted(rics)[:3]
        worst3_spr = sorted(sprs)[:3]

        s = {
            "avg_rank_ic": round(float(np.mean(rics)), 6),
            "median_rank_ic": round(float(np.median(rics)), 6),
            "worst3_avg_rank_ic": round(float(np.mean(worst3_ric)), 6),
            "pct_positive_rank_ic": round(sum(1 for r in rics if r > 0) / n, 4),
            "avg_spread": round(float(np.mean(sprs)), 6),
            "median_spread": round(float(np.median(sprs)), 6),
            "worst3_avg_spread": round(float(np.mean(worst3_spr)), 6),
            "pct_positive_spread": round(sum(1 for s in sprs if s > 0) / n, 4),
            "avg_ic": round(float(np.mean(ic_vals)), 6),
            "n_features": len(feature_cols),
        }
        summary[name] = s

        logger.info(f"\n  {name} ({len(feature_cols)} features):")
        logger.info(f"    avg RankIC:       {s['avg_rank_ic']:+.4f}")
        logger.info(f"    median RankIC:    {s['median_rank_ic']:+.4f}")
        logger.info(f"    worst-3 RankIC:   {s['worst3_avg_rank_ic']:+.4f}")
        logger.info(f"    %positive RankIC: {s['pct_positive_rank_ic']:.1%}")
        logger.info(f"    avg Spread:       {s['avg_spread']*100:+.3f}%")
        logger.info(f"    median Spread:    {s['median_spread']*100:+.3f}%")
        logger.info(f"    worst-3 Spread:   {s['worst3_avg_spread']*100:+.3f}%")
        logger.info(f"    %positive Spread: {s['pct_positive_spread']:.1%}")
        logger.info(f"    avg IC:           {s['avg_ic']:+.4f}")

    # Deltas
    raw = summary["base_174_raw"]
    ranked = summary["base_174_custom_flow_ranked"]
    logger.info(f"\n  Delta (ranked - raw):")
    logger.info(f"    dRankIC:       {ranked['avg_rank_ic'] - raw['avg_rank_ic']:+.6f}")
    logger.info(f"    dSpread:       {(ranked['avg_spread'] - raw['avg_spread'])*100:+.4f}%")
    logger.info(f"    dIC:           {ranked['avg_ic'] - raw['avg_ic']:+.6f}")
    logger.info(f"    dWorst3 RankIC:{ranked['worst3_avg_rank_ic'] - raw['worst3_avg_rank_ic']:+.6f}")

    # Save
    out_path = DATA_DIR / "phase4" / "rank_preprocess_24split.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "n_splits": n,
        "total_time_s": round(total_time, 1),
        "summary": summary,
        "delta_ranked_minus_raw": {
            "avg_rank_ic": round(ranked["avg_rank_ic"] - raw["avg_rank_ic"], 6),
            "avg_spread": round(ranked["avg_spread"] - raw["avg_spread"], 6),
            "avg_ic": round(ranked["avg_ic"] - raw["avg_ic"], 6),
            "worst3_avg_rank_ic": round(ranked["worst3_avg_rank_ic"] - raw["worst3_avg_rank_ic"], 6),
        },
        "splits": all_splits_info,
    }

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(str(out_path), "w") as f:
        json.dump(output, f, indent=2, default=_default)
    logger.info(f"\nSaved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
