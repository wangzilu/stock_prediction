"""Phase 2 Forecast Split Test: binary signal vs actual content.

st_forecast.parquet has 18,444 records but 99% are >180 days old.
The factor passed 75% ablation gate, but we suspect the signal comes from
the binary "has_forecast vs no_forecast" indicator, not the content
(p_change_min/max, net_profit_min/max).

This script tests three variants via 8-split rolling ablation:
  A) base (174+ cols from cache)
  B) base + has_forecast_binary (1 col: 1 if stock has ANY forecast, 0 otherwise)
  C) base + content_only (4 cols: p_change_min/max, net_profit_min/max; NaN if no forecast)

Usage:
    python scripts/phase2_forecast_split_test.py
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

from config.qlib_runtime import init_qlib
from models.feature_merger import FeatureMerger
from utils.json_utils import json_default

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
SEED = 42

N_SPLITS = 8
TEST_DAYS = 20
TRAIN_DAYS = 750
VALID_DAYS = 60


def train_xgb(X_train, y_train, X_valid, y_valid):
    import xgboost as xgb
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_valid, label=y_valid)
    params = {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
              "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
              "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": SEED}
    return xgb.train(params, dt, num_boost_round=400,
                     evals=[(dv, "valid")], early_stopping_rounds=30, verbose_eval=0)


def evaluate(pred, label, index):
    mask = np.isfinite(pred) & np.isfinite(label)
    ps = pd.Series(pred[mask], index=index[mask])
    ls = pd.Series(label[mask], index=index[mask])

    ric_vals = []
    spreads = []
    for date in ps.index.get_level_values(0).unique():
        p_day = ps.loc[date]
        l_day = ls.loc[date]
        if len(p_day) < 40:
            continue
        ric_vals.append(float(p_day.corr(l_day, method="spearman")))
        s = pd.DataFrame({"p": p_day, "l": l_day}).sort_values("p", ascending=False)
        spreads.append(s.head(20)["l"].mean() - s.tail(20)["l"].mean())

    ric = np.array(ric_vals)
    return {
        "rank_ic_mean": round(float(np.nanmean(ric)), 6) if len(ric) > 0 else 0,
        "rank_ic_pos": round(float(np.nanmean(ric > 0)), 4) if len(ric) > 0 else 0,
        "top20_spread": round(float(np.mean(spreads)), 6) if spreads else 0,
        "spread_pos": round(float(np.mean([s > 0 for s in spreads])), 4) if spreads else 0,
    }


def build_forecast_variants(cache_index: pd.MultiIndex, merger: FeatureMerger):
    """Build two forecast feature variants:
    1. has_forecast_binary: 1 if stock has ANY forecast record (via asof merge), 0 otherwise
    2. content_only: p_change_min, p_change_max, net_profit_min, net_profit_max (NaN if no forecast)
    """
    path = DATA_DIR / "st_forecast.parquet"
    df = pd.read_parquet(str(path))
    logger.info(f"Loaded st_forecast: {df.shape[0]} rows")

    df["ann_date"] = pd.to_datetime(df["ann_date"], format="%Y%m%d", errors="coerce")
    content_cols = ["p_change_min", "p_change_max", "net_profit_min", "net_profit_max"]
    content_cols = [c for c in content_cols if c in df.columns]
    for c in content_cols:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["ann_date"])

    # --- Variant A: has_forecast_binary ---
    # Add a constant column=1 to the forecast df, do asof merge, then fill NaN->0
    df_bin = df[["qlib_code", "ann_date"]].copy()
    df_bin["has_forecast"] = 1.0
    binary_merged = merger._asof_merge_timeseries(
        df_bin, cache_index, "ann_date", ["has_forecast"]
    )
    if binary_merged is not None:
        binary_merged["has_forecast"] = binary_merged["has_forecast"].fillna(0.0)
    else:
        # Fallback: all zeros
        binary_merged = pd.DataFrame(
            {"has_forecast": np.zeros(len(cache_index))},
            index=cache_index
        )

    coverage_bin = (binary_merged["has_forecast"] == 1.0).mean()
    logger.info(f"  has_forecast_binary: coverage={coverage_bin:.1%}")

    # --- Variant B: content_only ---
    df_content = df[["qlib_code", "ann_date"] + content_cols].copy()
    content_merged = merger._asof_merge_timeseries(
        df_content, cache_index, "ann_date", content_cols
    )
    if content_merged is None:
        content_merged = pd.DataFrame(
            {c: np.full(len(cache_index), np.nan) for c in content_cols},
            index=cache_index
        )

    coverage_content = content_merged.notna().any(axis=1).mean()
    logger.info(f"  content_only: {len(content_cols)} cols, coverage={coverage_content:.1%}")

    return binary_merged, content_merged, content_cols


def main():
    import xgboost as xgb

    init_qlib(QLIB_DATA)

    # Load base cache
    cache_path = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
    logger.info(f"Loading base cache: {cache_path}")
    cache = pd.read_parquet(str(cache_path))
    logger.info(f"  Shape: {cache.shape}")

    base_cols = [c for c in cache.columns if not c.startswith("__") and not c.startswith("_")]
    label_col = "__label_5d"
    logger.info(f"  Base feature cols: {len(base_cols)}")

    trade_dates = sorted(cache.index.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1

    # Build forecast variants
    merger = FeatureMerger(DATA_DIR)
    logger.info("\nBuilding forecast feature variants...")
    t0 = time.time()
    binary_df, content_df, content_cols = build_forecast_variants(cache.index, merger)
    logger.info(f"  Built variants in {time.time()-t0:.1f}s")

    # Pre-compute split specs
    dl = cache.index.get_level_values(0)
    split_specs = []
    for split_idx in range(N_SPLITS):
        test_end_idx = today_idx - split_idx * TEST_DAYS
        test_start_idx = test_end_idx - TEST_DAYS
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - VALID_DAYS
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - TRAIN_DAYS
        if train_start_idx < 0:
            break
        split_specs.append({
            "idx": split_idx,
            "tm": (dl >= trade_dates[train_start_idx]) & (dl <= trade_dates[train_end_idx]),
            "vm": (dl >= trade_dates[valid_start_idx]) & (dl <= trade_dates[valid_end_idx]),
            "em": (dl >= trade_dates[test_start_idx]) & (dl <= trade_dates[test_end_idx]),
        })

    logger.info(f"\nRunning {len(split_specs)}-split rolling ablation...")

    # Results storage
    variants = {
        "base": {"splits": [], "n_cols": len(base_cols)},
        "base+binary": {"splits": [], "n_cols": len(base_cols) + 1},
        "base+content": {"splits": [], "n_cols": len(base_cols) + len(content_cols)},
    }

    t_total = time.time()

    for spec in split_specs:
        split_idx = spec["idx"]
        tm, vm, em = spec["tm"], spec["vm"], spec["em"]

        y_tr = cache.loc[tm, label_col].values.astype(np.float32)
        y_va = cache.loc[vm, label_col].values.astype(np.float32)
        y_te = cache.loc[em, label_col].values.astype(np.float32)
        mtr = np.isfinite(y_tr); mva = np.isfinite(y_va); mte = np.isfinite(y_te)
        test_idx = cache.index[em]

        # Base features
        X_tr_base = cache.loc[tm, base_cols].values.astype(np.float32)
        X_va_base = cache.loc[vm, base_cols].values.astype(np.float32)
        X_te_base = cache.loc[em, base_cols].values.astype(np.float32)

        # --- A) Base model ---
        t1 = time.time()
        m_base = train_xgb(X_tr_base[mtr], y_tr[mtr], X_va_base[mva], y_va[mva])
        p_base = m_base.predict(xgb.DMatrix(X_te_base[mte]))
        e_base = evaluate(p_base, y_te[mte], test_idx[mte])
        logger.info(f"\nSplit {split_idx+1}/{len(split_specs)}: "
                    f"base RankIC={e_base['rank_ic_mean']:+.4f} [{time.time()-t1:.0f}s]")

        # --- B) Base + has_forecast_binary ---
        bin_tr = binary_df.loc[tm].values.astype(np.float32)
        bin_va = binary_df.loc[vm].values.astype(np.float32)
        bin_te = binary_df.loc[em].values.astype(np.float32)

        X_tr_bin = np.hstack([X_tr_base, bin_tr])
        X_va_bin = np.hstack([X_va_base, bin_va])
        X_te_bin = np.hstack([X_te_base, bin_te])

        t2 = time.time()
        m_bin = train_xgb(X_tr_bin[mtr], y_tr[mtr], X_va_bin[mva], y_va[mva])
        p_bin = m_bin.predict(xgb.DMatrix(X_te_bin[mte]))
        e_bin = evaluate(p_bin, y_te[mte], test_idx[mte])
        logger.info(f"  +binary:  RankIC={e_bin['rank_ic_mean']:+.4f} "
                    f"delta={e_bin['rank_ic_mean']-e_base['rank_ic_mean']:+.4f} [{time.time()-t2:.0f}s]")

        # --- C) Base + content_only ---
        cnt_tr = content_df.loc[tm].values.astype(np.float32)
        cnt_va = content_df.loc[vm].values.astype(np.float32)
        cnt_te = content_df.loc[em].values.astype(np.float32)

        X_tr_cnt = np.hstack([X_tr_base, cnt_tr])
        X_va_cnt = np.hstack([X_va_base, cnt_va])
        X_te_cnt = np.hstack([X_te_base, cnt_te])

        t3 = time.time()
        m_cnt = train_xgb(X_tr_cnt[mtr], y_tr[mtr], X_va_cnt[mva], y_va[mva])
        p_cnt = m_cnt.predict(xgb.DMatrix(X_te_cnt[mte]))
        e_cnt = evaluate(p_cnt, y_te[mte], test_idx[mte])
        logger.info(f"  +content: RankIC={e_cnt['rank_ic_mean']:+.4f} "
                    f"delta={e_cnt['rank_ic_mean']-e_base['rank_ic_mean']:+.4f} [{time.time()-t3:.0f}s]")

        # Store split results
        split_result_base = {
            "split": split_idx + 1,
            **e_base,
        }
        split_result_bin = {
            "split": split_idx + 1,
            **e_bin,
            "delta_ric_vs_base": round(e_bin["rank_ic_mean"] - e_base["rank_ic_mean"], 6),
            "delta_spr_vs_base": round(e_bin["top20_spread"] - e_base["top20_spread"], 6),
        }
        split_result_cnt = {
            "split": split_idx + 1,
            **e_cnt,
            "delta_ric_vs_base": round(e_cnt["rank_ic_mean"] - e_base["rank_ic_mean"], 6),
            "delta_spr_vs_base": round(e_cnt["top20_spread"] - e_base["top20_spread"], 6),
        }
        variants["base"]["splits"].append(split_result_base)
        variants["base+binary"]["splits"].append(split_result_bin)
        variants["base+content"]["splits"].append(split_result_cnt)

    total_time = time.time() - t_total

    # --- Compute summaries ---
    for vname, vdata in variants.items():
        splits = vdata["splits"]
        n = len(splits)
        if n == 0:
            continue

        avg_ric = np.mean([s["rank_ic_mean"] for s in splits])
        vdata["avg_rank_ic"] = round(float(avg_ric), 6)
        vdata["avg_spread"] = round(float(np.mean([s["top20_spread"] for s in splits])), 6)

        if "delta_ric_vs_base" in splits[0]:
            d_rics = [s["delta_ric_vs_base"] for s in splits]
            d_sprs = [s["delta_spr_vs_base"] for s in splits]
            ric_helps = sum(1 for d in d_rics if d > 0)
            spr_helps = sum(1 for d in d_sprs if d > 0)
            vdata["avg_delta_ric"] = round(float(np.mean(d_rics)), 6)
            vdata["avg_delta_spr"] = round(float(np.mean(d_sprs)), 6)
            vdata["ric_helps_pct"] = round(ric_helps / n, 4)
            vdata["spr_helps_pct"] = round(spr_helps / n, 4)
            vdata["pass_75pct"] = (ric_helps / n >= 0.75) or (spr_helps / n >= 0.75)

    # --- Determine verdict ---
    bin_delta = variants["base+binary"].get("avg_delta_ric", 0)
    cnt_delta = variants["base+content"].get("avg_delta_ric", 0)
    bin_pass = variants["base+binary"].get("pass_75pct", False)
    cnt_pass = variants["base+content"].get("pass_75pct", False)

    if bin_pass and not cnt_pass:
        verdict = "BINARY_SIGNAL_ONLY"
        explanation = ("The binary has_forecast signal passes the 75% gate, "
                       "but the actual content does not. The value comes from "
                       "knowing a forecast EXISTS, not its content.")
    elif cnt_pass and not bin_pass:
        verdict = "CONTENT_VALUE"
        explanation = ("The content columns pass the 75% gate, but the binary "
                       "signal alone does not. The actual forecast values matter.")
    elif bin_pass and cnt_pass:
        if cnt_delta > bin_delta * 1.2:
            verdict = "CONTENT_STRONGER"
            explanation = ("Both pass, but content provides >20% more lift. "
                           "The actual values add meaningful information beyond the binary signal.")
        elif bin_delta > cnt_delta * 1.2:
            verdict = "BINARY_DOMINATES"
            explanation = ("Both pass, but binary provides >20% more lift. "
                           "The content adds noise; the binary signal is the real driver.")
        else:
            verdict = "BOTH_CONTRIBUTE"
            explanation = ("Both variants pass and provide similar lift. "
                           "The binary signal and content both contribute.")
    else:
        verdict = "NEITHER_PASSES"
        explanation = ("Neither variant passes the 75% gate in this test. "
                       "The original ablation result may be unstable.")

    # --- Print summary ---
    logger.info(f"\n{'='*70}")
    logger.info(f"FORECAST SPLIT TEST RESULTS ({total_time:.0f}s, {len(split_specs)} splits)")
    logger.info(f"{'='*70}")
    logger.info(f"{'Variant':<20} {'Cols':>5} {'Avg RankIC':>12} {'Avg Spread':>12} "
                f"{'Avg dRIC':>10} {'RIC helps':>10} {'Gate':>6}")
    logger.info("-" * 80)
    for vname, vdata in variants.items():
        avg_ric = vdata.get("avg_rank_ic", 0)
        avg_spr = vdata.get("avg_spread", 0)
        d_ric = vdata.get("avg_delta_ric", 0)
        rh = vdata.get("ric_helps_pct", None)
        gate = vdata.get("pass_75pct", None)
        rh_str = f"{rh:.0%}" if rh is not None else "N/A"
        gate_str = ("PASS" if gate else "FAIL") if gate is not None else "base"
        logger.info(f"{vname:<20} {vdata['n_cols']:>5} {avg_ric:+.4f}       "
                    f"{avg_spr*100:+.3f}%      {d_ric:+.4f}     "
                    f"{rh_str:>6}     {gate_str:>6}")

    logger.info(f"\nVERDICT: {verdict}")
    logger.info(f"  {explanation}")

    # --- Save results ---
    output = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "total_time_s": round(total_time, 1),
        "n_splits": len(split_specs),
        "forecast_stats": {
            "total_records": int(pd.read_parquet(str(DATA_DIR / "st_forecast.parquet")).shape[0]),
            "binary_coverage": float(
                (binary_df["has_forecast"] == 1.0).mean()
            ),
            "content_cols": content_cols,
        },
        "variants": variants,
        "verdict": verdict,
        "explanation": explanation,
    }

    out_path = DATA_DIR / "phase4" / "phase2_forecast_split.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump(output, f, indent=2, default=json_default)
    logger.info(f"\nSaved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
