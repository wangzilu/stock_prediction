"""Market-cap stratified analysis: does small-cap alpha > large-cap alpha?

Splits universe into 3 tiers by daily market cap proxy (close * volume_20d_avg),
then runs XGB + institutional metrics for each tier separately.

Usage:
    python scripts/phase4k_mcap_stratified.py
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
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import xgboost as xgb

from config.qlib_runtime import init_qlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
OUTPUT_DIR = DATA_DIR / "phase4k"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 12
TEST_DAYS = 20
VALID_DAYS = 60
TRAIN_DAYS = 750

XGB_PARAMS = {
    "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
    "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
    "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": 42,
}

EXCLUDE_PREFIXES = ("__", "_", "hsi_", "hstech_", "nasdaq_")
EXCLUDE_EXACT = {"holder_num"}
LABEL_COL = "__label_5d"
PNL_COL = "__pnl_return_1d"

# Market cap tiers (by cross-sectional percentile each day)
# Bottom 30% = small, Middle 40% = mid, Top 30% = large
TIERS = {
    "small_cap": (0.0, 0.30),     # smallest 30%
    "mid_cap": (0.30, 0.70),      # middle 40%
    "large_cap": (0.70, 1.0),     # largest 30%
    "all": (0.0, 1.0),            # full universe (baseline)
}


def get_feature_cols(all_cols):
    result = []
    for c in all_cols:
        if any(c.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if c in EXCLUDE_EXACT:
            continue
        result.append(c)
    return result


def load_mcap_proxy(cache_index: pd.MultiIndex) -> pd.Series:
    """Load market cap proxy: use Qlib $close * Mean($volume, 20) as liquidity-weighted size.

    Actually simpler: just use $close as a rough proxy for price-level (correlated with mcap).
    Better: load $close * $factor (cumulative adjustment factor) for size.
    Best for our purposes: load daily turnover amount = $amount (already in cache via amount_raw).
    """
    from qlib.data import D

    init_qlib(QLIB_DATA)

    dates = sorted(cache_index.get_level_values(0).unique())
    insts = sorted(set(str(c) for c in cache_index.get_level_values(1)))

    logger.info(f"Loading market cap proxy for {len(insts)} instruments...")
    # Use $close * Ref($factor, 0) * some_share_count — but we don't have shares
    # Simpler: use Mean($amount, 20) as ADV proxy (highly correlated with market cap)
    df = D.features(insts, ["Mean($amount, 20)"],
                    start_time=str(min(dates))[:10],
                    end_time=str(max(dates))[:10])
    if df is None or df.empty:
        logger.warning("Failed to load mcap proxy")
        return None

    df.columns = ["adv20"]
    df = df.swaplevel().sort_index()
    mcap = df["adv20"].reindex(cache_index)
    logger.info(f"  ADV20 loaded: {mcap.notna().sum()} valid / {len(mcap)} total")
    return mcap


def assign_tiers(mcap: pd.Series) -> pd.Series:
    """Assign each stock-day to a market cap tier based on daily cross-sectional percentile."""
    result = pd.Series("unknown", index=mcap.index, dtype="object")

    for dt, group in mcap.groupby(level=0):
        valid = group.dropna()
        if len(valid) < 100:
            continue
        pct = valid.rank(pct=True)
        for tier_name, (lo, hi) in TIERS.items():
            if tier_name == "all":
                continue
            mask = (pct >= lo) & (pct < hi)
            tier_stocks = pct[mask].index
            result.loc[tier_stocks] = tier_name

    return result


def compute_split_metrics(pred: np.ndarray, label: np.ndarray,
                          pnl: np.ndarray, index: pd.MultiIndex) -> dict:
    """Compute RankIC, spreads, and portfolio metrics for one split."""
    mask = np.isfinite(pred) & np.isfinite(label)
    if mask.sum() < 100:
        return None

    p, l = pred[mask], label[mask]
    idx = index[mask]

    # RankIC per date
    pred_s = pd.Series(p, index=idx)
    label_s = pd.Series(l, index=idx)

    ric_list = []
    spreads = {20: [], 50: [], 100: []}

    for dt, g_pred in pred_s.groupby(level=0):
        g_label = label_s.reindex(g_pred.index).dropna()
        common = g_pred.index.intersection(g_label.index)
        if len(common) < 40:
            continue
        pv = g_pred.loc[common].values
        lv = g_label.loc[common].values

        ric = stats.spearmanr(pv, lv).statistic
        if np.isfinite(ric):
            ric_list.append(ric)

        # Spreads
        tmp = pd.DataFrame({"p": pv, "l": lv}).sort_values("p", ascending=False)
        for k in spreads:
            if len(tmp) >= k * 2:
                spreads[k].append(tmp.head(k)["l"].mean() - tmp.tail(k)["l"].mean())

    if not ric_list:
        return None

    result = {
        "rank_ic": float(np.mean(ric_list)),
        "rank_ic_pos": float(np.mean([r > 0 for r in ric_list])),
        "n_stocks_avg": int(mask.sum() / max(len(set(idx.get_level_values(0))), 1)),
    }
    for k, sp in spreads.items():
        if sp:
            result[f"spread_top{k}"] = float(np.mean(sp))
            result[f"spread_top{k}_pos"] = float(np.mean([s > 0 for s in sp]))

    return result


def main():
    t_start = time.time()

    # Load cache
    logger.info("Loading feature cache...")
    cache = pd.read_parquet(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet")
    feature_cols = get_feature_cols(cache.columns)
    logger.info(f"Cache: {cache.shape}, {len(feature_cols)} features")

    # Load market cap proxy
    mcap = load_mcap_proxy(cache.index)
    if mcap is None:
        logger.error("Cannot load market cap data")
        return

    # Assign tiers
    logger.info("Assigning market cap tiers...")
    tiers = assign_tiers(mcap)
    tier_counts = tiers.value_counts()
    logger.info(f"Tier distribution:\n{tier_counts}")

    trade_dates = sorted(cache.index.get_level_values(0).unique())
    dates_level = cache.index.get_level_values(0)

    # Results per tier
    all_results = {tier: [] for tier in TIERS}

    for split_idx in range(N_SPLITS):
        test_end_idx = len(trade_dates) - 1 - split_idx * TEST_DAYS
        test_start_idx = test_end_idx - TEST_DAYS
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - VALID_DAYS
        train_end_idx = valid_start_idx - 1
        train_start_idx = max(0, train_end_idx - TRAIN_DAYS)

        if train_start_idx >= train_end_idx:
            break

        test_start = trade_dates[test_start_idx]
        test_end = trade_dates[test_end_idx]
        train_start = trade_dates[train_start_idx]
        train_end = trade_dates[train_end_idx]
        valid_start = trade_dates[valid_start_idx]

        logger.info(f"\nSplit {split_idx+1}/{N_SPLITS}: test {str(test_start)[:10]}~{str(test_end)[:10]}")

        # Train on FULL universe (not stratified — same model for all tiers)
        train_mask = (dates_level >= train_start) & (dates_level <= train_end)
        valid_mask = (dates_level >= valid_start) & (dates_level <= trade_dates[valid_end_idx])
        test_mask = (dates_level >= test_start) & (dates_level <= test_end)

        X_tr = cache.loc[train_mask, feature_cols].values.astype(np.float32)
        y_tr = cache.loc[train_mask, LABEL_COL].values.astype(np.float32)
        X_va = cache.loc[valid_mask, feature_cols].values.astype(np.float32)
        y_va = cache.loc[valid_mask, LABEL_COL].values.astype(np.float32)
        X_te = cache.loc[test_mask, feature_cols].values.astype(np.float32)
        y_te = cache.loc[test_mask, LABEL_COL].values.astype(np.float32)
        test_idx = cache.index[test_mask]
        test_tiers = tiers.loc[test_mask]

        # Filter NaN labels
        m_tr = np.isfinite(y_tr); X_tr, y_tr = X_tr[m_tr], y_tr[m_tr]
        m_va = np.isfinite(y_va); X_va, y_va = X_va[m_va], y_va[m_va]
        m_te = np.isfinite(y_te)

        # Train XGB on full universe
        dt = xgb.DMatrix(X_tr, label=y_tr)
        dv = xgb.DMatrix(X_va, label=y_va)
        model = xgb.train(XGB_PARAMS, dt, num_boost_round=400,
                          evals=[(dv, "valid")], early_stopping_rounds=50,
                          verbose_eval=0)
        pred_all = model.predict(xgb.DMatrix(X_te))

        # Get PnL returns
        pnl_all = cache.loc[test_mask, PNL_COL].values.astype(np.float32)

        # Evaluate each tier
        for tier_name in TIERS:
            if tier_name == "all":
                tier_mask = m_te  # all valid labels
            else:
                tier_mask = m_te & (test_tiers.values == tier_name)

            if tier_mask.sum() < 200:
                logger.info(f"  {tier_name}: only {tier_mask.sum()} samples, skip")
                continue

            metrics = compute_split_metrics(
                pred_all[tier_mask], y_te[tier_mask],
                pnl_all[tier_mask], test_idx[tier_mask],
            )
            if metrics:
                metrics["split"] = split_idx + 1
                all_results[tier_name].append(metrics)
                sp20 = metrics.get("spread_top20", 0)
                logger.info(
                    f"  {tier_name:<12} RankIC={metrics['rank_ic']:+.4f} "
                    f"Top20={sp20*100:+.3f}% "
                    f"n_stocks={metrics['n_stocks_avg']}"
                )

    # Summary
    logger.info(f"\n{'='*100}")
    logger.info(f"MARKET CAP STRATIFIED ANALYSIS: {N_SPLITS} splits")
    logger.info(f"{'='*100}")
    logger.info(
        f"{'Tier':<14} {'AvgRIC':>8} {'MedRIC':>8} {'RICIR':>7} "
        f"{'Spr20':>8} {'Spr50':>8} {'Spr100':>8} "
        f"{'RIC>0':>6} {'#Stocks':>8} {'#Split':>6}"
    )
    logger.info("-" * 100)

    summary = {}
    for tier_name in ["all", "small_cap", "mid_cap", "large_cap"]:
        splits = all_results[tier_name]
        if not splits:
            continue
        rics = [s["rank_ic"] for s in splits]
        sp20 = [s.get("spread_top20", 0) for s in splits]
        sp50 = [s.get("spread_top50", 0) for s in splits]
        sp100 = [s.get("spread_top100", 0) for s in splits]
        ric_pos = [s.get("rank_ic_pos", 0) for s in splits]
        n_stocks = [s.get("n_stocks_avg", 0) for s in splits]

        ricir = float(np.mean(rics) / (np.std(rics) + 1e-8))

        summary[tier_name] = {
            "avg_rank_ic": round(float(np.mean(rics)), 6),
            "med_rank_ic": round(float(np.median(rics)), 6),
            "rank_icir": round(ricir, 4),
            "avg_spread_top20": round(float(np.mean(sp20)), 6),
            "avg_spread_top50": round(float(np.mean(sp50)), 6),
            "avg_spread_top100": round(float(np.mean(sp100)), 6),
            "avg_ric_pos": round(float(np.mean(ric_pos)), 4),
            "avg_n_stocks": round(float(np.mean(n_stocks))),
            "n_splits": len(splits),
            "per_split": splits,
        }

        logger.info(
            f"{tier_name:<14} {np.mean(rics):>+8.4f} {np.median(rics):>+8.4f} {ricir:>+7.3f} "
            f"{np.mean(sp20)*100:>+7.3f}% {np.mean(sp50)*100:>+7.3f}% {np.mean(sp100)*100:>+7.3f}% "
            f"{np.mean(ric_pos)*100:>5.0f}% {np.mean(n_stocks):>8.0f} {len(splits):>6}"
        )

    # Compute small vs large ratio
    if "small_cap" in summary and "large_cap" in summary:
        s = summary["small_cap"]
        l = summary["large_cap"]
        logger.info(f"\n--- Small vs Large ---")
        logger.info(f"  RankIC:  small {s['avg_rank_ic']:+.4f} vs large {l['avg_rank_ic']:+.4f} "
                    f"(ratio: {s['avg_rank_ic']/(l['avg_rank_ic']+1e-8):.2f}x)")
        logger.info(f"  Top20:   small {s['avg_spread_top20']*100:+.3f}% vs large {l['avg_spread_top20']*100:+.3f}%")
        if l['avg_spread_top20'] != 0:
            logger.info(f"  Spread ratio: {s['avg_spread_top20']/(l['avg_spread_top20']+1e-8):.2f}x")

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s")

    # Save
    output = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "n_splits": N_SPLITS,
        "tier_definition": {k: {"lo_pct": v[0], "hi_pct": v[1]} for k, v in TIERS.items()},
        "mcap_proxy": "Mean($amount, 20) — 20-day average daily turnover amount",
        "summary": summary,
    }
    out_path = OUTPUT_DIR / "mcap_stratified.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                   default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o)))
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
