"""Phase 4K: Institutional preprocessing pipeline comparison.

Tests the full institutional factor preprocessing pipeline on the 16
non-Alpha158 features and compares against the raw baseline.

Pipeline: raw -> MAD winsorize (5sigma) -> Z-Score -> market_cap+industry
neutralize (OLS residual) -> re-Z-Score

4-way comparison:
1. FS-174: current champion (raw non-Alpha158 features)
2. FS-174-mad-zscore: MAD winsorize + Z-Score only (no neutralization)
3. FS-174-neutralized: MAD + Z-Score + mcap+industry OLS neutralization + re-Z-Score
4. FS-174-rankgauss: rank -> norm.ppf (RankGauss transform)

Usage:
    python scripts/phase4k_preprocessing_compare.py
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
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
SEED = 42

# Non-Alpha158 custom features
CUSTOM_FEATURES = [
    "pe", "pb", "turn_raw", "amount_raw",
    "pe_mom20", "pb_mom20", "turn_anom20", "turn_anom60",
    "amount_anom20", "turn_vol20", "ep", "bp", "price_pos20",
]

# Flow features
FLOW_FEATURES = ["flow_net_mf_latest", "flow_net_mf_5d", "flow_net_mf_20d_avg"]

ALL_NON_ALPHA158 = CUSTOM_FEATURES + FLOW_FEATURES


def train_xgb(X_train, y_train, X_valid, y_valid):
    import xgboost as xgb
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_valid, label=y_valid)
    params = {
        "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
        "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
        "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": SEED,
    }
    return xgb.train(params, dt, num_boost_round=400,
                     evals=[(dv, "valid")], early_stopping_rounds=30, verbose_eval=0)


def evaluate(pred, label, index):
    from scipy.stats import spearmanr
    mask = np.isfinite(pred) & np.isfinite(label)
    ps = pd.Series(pred[mask], index=index[mask])
    ls = pd.Series(label[mask], index=index[mask])
    rics, sprs = [], []
    for date in ps.index.get_level_values(0).unique():
        p = ps.loc[date].values
        l = ls.loc[date].values
        if len(p) < 40:
            continue
        c, _ = spearmanr(p, l)
        if np.isfinite(c):
            rics.append(c)
        k = min(20, len(p) // 2)
        top = np.argpartition(p, -k)[-k:]
        bot = np.argpartition(p, k)[:k]
        sprs.append(l[top].mean() - l[bot].mean())
    return {
        "rank_ic_mean": round(float(np.nanmean(rics)), 6) if rics else 0,
        "top20_spread": round(float(np.mean(sprs)), 6) if sprs else 0,
    }


# ---------------------------------------------------------------------------
# Preprocessing transforms (applied per cross-section / date)
# ---------------------------------------------------------------------------

def mad_zscore_columns(df, cols, date_level=0):
    """MAD winsorize (5 sigma_MAD) + Z-Score per date cross-section."""
    result = df.copy()
    for col in cols:
        if col not in result.columns:
            continue
        s = result[col]
        median = s.groupby(level=date_level).transform("median")
        mad = (s - median).abs().groupby(level=date_level).transform("median")
        sigma_mad = 1.4826 * mad
        # Winsorize at 5 sigma
        lower = median - 5 * sigma_mad
        upper = median + 5 * sigma_mad
        clipped = s.clip(lower=lower, upper=upper)
        # Z-Score
        mean = clipped.groupby(level=date_level).transform("mean")
        std = clipped.groupby(level=date_level).transform("std") + 1e-8
        result[col] = (clipped - mean) / std
    return result


def neutralize_columns(df, cols, industry_dummies, log_mcap, date_level=0):
    """OLS neutralization: regress each factor on log(mcap) + industry dummies.

    Returns residuals re-Z-Scored per date.
    """
    result = df.copy()
    dates = df.index.get_level_values(date_level)
    unique_dates = dates.unique()

    for col in cols:
        if col not in result.columns:
            continue
        vals = result[col].copy()
        resid = pd.Series(np.nan, index=df.index, dtype=np.float32)

        for date in unique_dates:
            mask_date = dates == date
            y = vals[mask_date]
            x_mcap = log_mcap[mask_date]
            valid = y.notna() & x_mcap.notna()
            if valid.sum() < 30:
                continue

            yv = y[valid].values.astype(np.float64)
            xv = x_mcap[valid].values.reshape(-1, 1).astype(np.float64)
            xm = np.column_stack([np.ones(len(xv)), xv])

            if industry_dummies is not None:
                try:
                    ind_sub = industry_dummies.loc[y[valid].index]
                    ind_arr = ind_sub.values
                    nonzero = ind_arr.sum(axis=0) > 0
                    if nonzero.sum() > 1:
                        xm = np.column_stack([xm, ind_arr[:, nonzero][:, 1:]])
                except Exception:
                    pass

            try:
                beta, _, _, _ = np.linalg.lstsq(xm, yv, rcond=None)
                r = yv - xm @ beta
                resid.loc[y[valid].index] = r.astype(np.float32)
            except np.linalg.LinAlgError:
                continue

        # Re-Z-Score the residuals per date
        r_mean = resid.groupby(level=date_level).transform("mean")
        r_std = resid.groupby(level=date_level).transform("std") + 1e-8
        result[col] = (resid - r_mean) / r_std
    return result


def rankgauss_columns(df, cols, date_level=0):
    """RankGauss: per-date rank -> norm.ppf(rank / (n+1))."""
    from scipy.stats import norm
    result = df.copy()
    for col in cols:
        if col not in result.columns:
            continue
        # Rank per date (1-based, NaN stays NaN)
        ranked = result[col].groupby(level=date_level).rank(method="average")
        count = result[col].groupby(level=date_level).transform("count")
        # norm.ppf(rank / (n+1))
        gauss = norm.ppf(ranked / (count + 1))
        result[col] = gauss
    return result


# ---------------------------------------------------------------------------
# Market cap + industry loading
# ---------------------------------------------------------------------------

def load_mcap_series(index):
    """Load market cap series aligned to the multi-index (date, instrument)."""
    # Try st_daily_basic
    path = DATA_DIR / "st_daily_basic.parquet"
    if path.exists():
        df = pd.read_parquet(str(path))
        mcap_col = None
        for c in ["st_circ_mv", "st_total_mv", "circ_mv", "total_mv"]:
            if c in df.columns:
                mcap_col = c
                break
        if mcap_col and "qlib_code" in df.columns and "date" in df.columns:
            df[mcap_col] = pd.to_numeric(df[mcap_col], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            ts = df[["date", "qlib_code", mcap_col]].dropna(subset=[mcap_col])
            # Deduplicate and keep last
            ts = ts.drop_duplicates(subset=["date", "qlib_code"], keep="last")
            ts = ts.set_index(["date", "qlib_code"])[mcap_col]
            ts.index = ts.index.set_names(index.names)

            # Use merge instead of reindex to handle non-unique target index
            target = pd.DataFrame({"_idx": range(len(index))}, index=index)
            target = target.join(ts.rename("_mcap"), how="left")
            aligned = target["_mcap"]
            aligned.index = index
            # Forward fill within instrument for small gaps
            inst_level = 1
            aligned = aligned.groupby(level=inst_level).ffill(limit=5)
            coverage = aligned.notna().mean()
            logger.info(f"  MCap loaded from {mcap_col}: coverage={coverage:.1%}")
            if coverage > 0.2:
                return np.log1p(aligned.clip(lower=1))

    # Fallback: use amount_raw as proxy for market cap
    logger.warning("  MCap not available, using amount_raw as size proxy")
    return None


def load_industry_dummies(index):
    """Load industry mapping and return one-hot dummies aligned to index."""
    path = DATA_DIR / "industry_mapping.parquet"
    if not path.exists():
        logger.warning("  industry_mapping.parquet not found")
        return None
    try:
        df = pd.read_parquet(str(path))
        if df.empty or "qlib_code" not in df.columns or "industry" not in df.columns:
            return None
        ind_map = df.drop_duplicates("qlib_code").set_index("qlib_code")["industry"]
        inst_level = 1 if index.nlevels > 1 else 0
        instruments = index.get_level_values(inst_level).astype(str).str.lower()
        matched = instruments.map(ind_map).fillna("unknown")
        dummies = pd.get_dummies(matched, prefix="ind", dtype=np.float32)
        dummies.index = index
        coverage = (matched != "unknown").mean()
        logger.info(f"  Industry dummies: {dummies.shape[1]} industries, coverage={coverage:.1%}")
        return dummies
    except Exception as e:
        logger.warning(f"  Industry dummies load failed: {e}")
        return None


def main():
    import xgboost as xgb
    from config.qlib_runtime import init_qlib
    init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

    n_splits = 12
    test_days = 20
    train_days = 750
    valid_days = 60

    # Load cache
    cache_path = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
    logger.info(f"Loading cache: {cache_path}")
    cache = pd.read_parquet(str(cache_path))
    logger.info(f"  Shape: {cache.shape}")

    # Identify column groups
    all_feature_cols = [
        c for c in cache.columns
        if not c.startswith("__") and not c.startswith("_")
        and not c.startswith("hsi_") and not c.startswith("hstech_")
        and not c.startswith("nasdaq_")
    ]
    label_col = "__label_5d"

    # Non-Alpha158 columns present in cache
    non_a158_in_cache = [c for c in ALL_NON_ALPHA158 if c in all_feature_cols]
    logger.info(f"  All features: {len(all_feature_cols)}")
    logger.info(f"  Non-Alpha158 in cache: {len(non_a158_in_cache)} -> {non_a158_in_cache}")

    # ---------------------------------------------------------------------------
    # Load neutralization inputs
    # ---------------------------------------------------------------------------
    logger.info("Loading mcap and industry for neutralization...")
    log_mcap = load_mcap_series(cache.index)
    ind_dummies = load_industry_dummies(cache.index)

    can_neutralize = log_mcap is not None and ind_dummies is not None
    if not can_neutralize and log_mcap is None:
        # If no mcap at all, try using amount_raw from the cache as a proxy
        if "amount_raw" in cache.columns:
            logger.info("  Falling back to amount_raw as mcap proxy")
            log_mcap = np.log1p(cache["amount_raw"].clip(lower=0))
            can_neutralize = ind_dummies is not None

    # ---------------------------------------------------------------------------
    # Pre-compute transformed versions
    # ---------------------------------------------------------------------------
    logger.info("Pre-computing MAD+ZScore version...")
    t0 = time.time()
    cache_mz = mad_zscore_columns(cache, non_a158_in_cache)
    logger.info(f"  MAD+ZScore done in {time.time()-t0:.1f}s")

    if can_neutralize:
        logger.info("Pre-computing neutralized version (MAD+ZScore -> neutralize -> re-ZScore)...")
        t0 = time.time()
        cache_neu = neutralize_columns(cache_mz, non_a158_in_cache, ind_dummies, log_mcap)
        logger.info(f"  Neutralized done in {time.time()-t0:.1f}s")
    else:
        logger.warning("  Cannot neutralize (missing mcap or industry). "
                        "Skipping FS-174-neutralized variant.")
        cache_neu = None

    logger.info("Pre-computing RankGauss version...")
    t0 = time.time()
    cache_rg = rankgauss_columns(cache, non_a158_in_cache)
    logger.info(f"  RankGauss done in {time.time()-t0:.1f}s")

    # Build feature set variants
    feature_sets = {
        "FS-174": (cache, all_feature_cols),
        "FS-174-mad-zscore": (cache_mz, all_feature_cols),
        "FS-174-rankgauss": (cache_rg, all_feature_cols),
    }
    if cache_neu is not None:
        feature_sets["FS-174-neutralized"] = (cache_neu, all_feature_cols)

    fs_names = list(feature_sets.keys())
    logger.info(f"\nFeature sets: {fs_names}")
    for name, (_, cols) in feature_sets.items():
        logger.info(f"  {name}: {len(cols)} features")

    # ---------------------------------------------------------------------------
    # Rolling comparison
    # ---------------------------------------------------------------------------
    trade_dates = sorted(cache.index.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1
    dates_level = cache.index.get_level_values(0)

    all_results = []
    t_total = time.time()

    for split_idx in range(n_splits):
        test_end_idx = today_idx - split_idx * test_days
        test_start_idx = test_end_idx - test_days
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - valid_days
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - train_days
        if train_start_idx < 0:
            break

        tm = (dates_level >= trade_dates[train_start_idx]) & (dates_level <= trade_dates[train_end_idx])
        vm = (dates_level >= trade_dates[valid_start_idx]) & (dates_level <= trade_dates[valid_end_idx])
        em = (dates_level >= trade_dates[test_start_idx]) & (dates_level <= trade_dates[test_end_idx])

        y_tr = cache.loc[tm, label_col].values.astype(np.float32)
        y_va = cache.loc[vm, label_col].values.astype(np.float32)
        y_te = cache.loc[em, label_col].values.astype(np.float32)
        mtr = np.isfinite(y_tr)
        mva = np.isfinite(y_va)
        mte = np.isfinite(y_te)
        test_idx = cache.index[em]

        split_result = {"split": split_idx + 1}
        logger.info(f"\nSplit {split_idx+1}/{n_splits}:")

        for fs_name in fs_names:
            data_source, cols = feature_sets[fs_name]
            t1 = time.time()
            X_tr = data_source.loc[tm, cols].values.astype(np.float32)
            X_va = data_source.loc[vm, cols].values.astype(np.float32)
            X_te = data_source.loc[em, cols].values.astype(np.float32)

            model = train_xgb(X_tr[mtr], y_tr[mtr], X_va[mva], y_va[mva])
            pred = model.predict(xgb.DMatrix(X_te[mte]))
            metrics = evaluate(pred, y_te[mte], test_idx[mte])
            elapsed = time.time() - t1

            split_result[fs_name] = {**metrics, "n_feat": len(cols), "time_s": round(elapsed, 1)}
            logger.info(f"  {fs_name}({len(cols)}): RankIC={metrics['rank_ic_mean']:+.4f} "
                        f"Spread={metrics['top20_spread']*100:+.3f}% [{elapsed:.0f}s]")

        all_results.append(split_result)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    total_time = time.time() - t_total
    n = len(all_results)

    logger.info(f"\n{'='*70}")
    logger.info(f"PHASE 4K: PREPROCESSING COMPARISON ({n} splits, {total_time:.0f}s)")
    logger.info(f"{'='*70}")

    avg_rics = {}
    avg_sprs = {}
    for fs_name in fs_names:
        rics = [r[fs_name]["rank_ic_mean"] for r in all_results]
        sprs = [r[fs_name]["top20_spread"] for r in all_results]
        avg_rics[fs_name] = float(np.mean(rics))
        avg_sprs[fs_name] = float(np.mean(sprs))
        logger.info(f"\n  {fs_name} ({all_results[0][fs_name]['n_feat']} features):")
        logger.info(f"    avg RankIC: {avg_rics[fs_name]:+.4f}")
        logger.info(f"    avg Spread: {avg_sprs[fs_name]*100:+.3f}%")
        logger.info(f"    RankIC>0:   {sum(1 for r in rics if r > 0)}/{n}")
        logger.info(f"    Spread>0:   {sum(1 for s in sprs if s > 0)}/{n}")

    best = max(avg_rics, key=avg_rics.get)
    logger.info(f"\n  Best: {best} (avg RankIC {avg_rics[best]:+.4f})")

    # Deltas vs baseline
    baseline_ic = avg_rics["FS-174"]
    baseline_sp = avg_sprs["FS-174"]
    logger.info(f"\n  Deltas vs FS-174 baseline:")
    for fs_name in fs_names:
        if fs_name == "FS-174":
            continue
        dic = avg_rics[fs_name] - baseline_ic
        dsp = avg_sprs[fs_name] - baseline_sp
        logger.info(f"    {fs_name}: dRankIC={dic:+.4f}  dSpread={dsp*100:+.3f}%")

    # Save
    from utils.json_utils import json_default
    out_path = DATA_DIR / "phase4" / "phase4k_preprocessing_compare.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump({
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "n_splits": n,
            "total_time_s": round(total_time, 1),
            "avg_rank_ic": {k: round(float(v), 6) for k, v in avg_rics.items()},
            "avg_spread": {k: round(float(v), 6) for k, v in avg_sprs.items()},
            "best": best,
            "splits": all_results,
        }, f, indent=2, default=json_default)
    logger.info(f"\nSaved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
