"""Phase 2: Factor ablation — test each new factor group against 174 baseline.

Uses feature cache + fast rolling for quick results (~8 min per factor group).

Factors tested:
- moneyflow: 个股资金流 (主力/大单/超大单净流入)
- cyq: 筹码分布 (获利比例/平均成本/集中度)
- pledge: 股权质押 (质押比例/质押数量)
- forecast: 业绩预告 (预计增幅)

Usage:
    python scripts/phase2_factor_ablation.py
    python scripts/phase2_factor_ablation.py --factor moneyflow
"""
import argparse
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
SEED = 42


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


def load_factor_group(name: str, index: pd.MultiIndex, merger: FeatureMerger) -> pd.DataFrame | None:
    """Load a factor group and align to training index via asof merge."""

    if name == "moneyflow":
        path = DATA_DIR / "st_moneyflow.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        factor_cols = [c for c in df.columns if c.startswith("st_")]
        for c in factor_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return merger._asof_merge_timeseries(
            df[["qlib_code", "date"] + factor_cols], index, "date", factor_cols)

    elif name == "cyq":
        path = DATA_DIR / "st_cyq_perf.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        factor_cols = [c for c in df.columns if c.startswith("cyq_")]
        for c in factor_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return merger._asof_merge_timeseries(
            df[["qlib_code", "date"] + factor_cols], index, "date", factor_cols)

    elif name == "pledge":
        path = DATA_DIR / "st_pledge_stat.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df["end_date"] = pd.to_datetime(df["end_date"], format="%Y%m%d", errors="coerce")
        df["qlib_code"] = df["qlib_code"].str.upper()
        factor_cols = ["pledge_count", "pledge_ratio"]
        for c in factor_cols:
            df[c] = pd.to_numeric(df.get(c), errors="coerce")
        df = df.dropna(subset=["end_date"])
        return merger._asof_merge_timeseries(
            df[["qlib_code", "end_date"] + factor_cols], index, "end_date", factor_cols)

    elif name == "forecast":
        path = DATA_DIR / "st_forecast.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df["ann_date"] = pd.to_datetime(df["ann_date"], format="%Y%m%d", errors="coerce")
        factor_cols = ["p_change_min", "p_change_max", "net_profit_min", "net_profit_max"]
        factor_cols = [c for c in factor_cols if c in df.columns]
        for c in factor_cols:
            df[c] = pd.to_numeric(df.get(c), errors="coerce")
        df = df.dropna(subset=["ann_date"])
        return merger._asof_merge_timeseries(
            df[["qlib_code", "ann_date"] + factor_cols], index, "ann_date", factor_cols)

    return None


def main():
    import xgboost as xgb

    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=str, default="all",
                        choices=["all", "moneyflow", "cyq", "pledge", "forecast"])
    parser.add_argument("--n-splits", type=int, default=12)
    parser.add_argument("--test-days", type=int, default=20)
    parser.add_argument("--train-days", type=int, default=750)
    parser.add_argument("--valid-days", type=int, default=60)
    args = parser.parse_args()

    init_qlib(QLIB_DATA)

    # Load base cache
    cache_path = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
    logger.info(f"Loading base cache: {cache_path}")
    cache = pd.read_parquet(str(cache_path))
    logger.info(f"  Shape: {cache.shape}")

    base_cols = [c for c in cache.columns if not c.startswith("__") and not c.startswith("_")]
    label_col = "__label_5d"

    trade_dates = sorted(cache.index.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1

    # Determine which factors to test
    factor_groups = ["moneyflow", "cyq", "pledge", "forecast"] if args.factor == "all" else [args.factor]

    # Load factor data and merge to cache index
    merger = FeatureMerger(DATA_DIR)
    factor_data = {}

    for fg in factor_groups:
        logger.info(f"\nLoading factor: {fg}")
        t0 = time.time()
        fdf = load_factor_group(fg, cache.index, merger)
        if fdf is not None and not fdf.empty:
            factor_data[fg] = fdf
            n_cols = fdf.shape[1]
            coverage = fdf.notna().any(axis=1).mean()
            logger.info(f"  {fg}: {n_cols} cols, coverage={coverage:.1%}, {time.time()-t0:.1f}s")
        else:
            logger.warning(f"  {fg}: no data!")

    if not factor_data:
        logger.error("No factor data loaded!")
        sys.exit(1)

    # Rolling ablation
    all_results = {}
    t_total = time.time()

    for fg, fdf in factor_data.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"ABLATION: base vs base+{fg}")
        logger.info(f"{'='*60}")

        fg_cols = list(fdf.columns)
        results = []

        for split_idx in range(args.n_splits):
            test_end_idx = today_idx - split_idx * args.test_days
            test_start_idx = test_end_idx - args.test_days
            valid_end_idx = test_start_idx - 1
            valid_start_idx = valid_end_idx - args.valid_days
            train_end_idx = valid_start_idx - 1
            train_start_idx = train_end_idx - args.train_days

            if train_start_idx < 0:
                break

            test_start = trade_dates[test_start_idx]
            test_end = trade_dates[test_end_idx]
            train_start = trade_dates[train_start_idx]
            train_end = trade_dates[train_end_idx]
            valid_start = trade_dates[valid_start_idx]
            valid_end = trade_dates[valid_end_idx]

            dl = cache.index.get_level_values(0)
            tm = (dl >= train_start) & (dl <= train_end)
            vm = (dl >= valid_start) & (dl <= valid_end)
            em = (dl >= test_start) & (dl <= test_end)

            y_tr = cache.loc[tm, label_col].values.astype(np.float32)
            y_va = cache.loc[vm, label_col].values.astype(np.float32)
            y_te = cache.loc[em, label_col].values.astype(np.float32)
            mtr = np.isfinite(y_tr); mva = np.isfinite(y_va); mte = np.isfinite(y_te)
            test_idx = cache.index[em]

            # Base
            X_tr = cache.loc[tm, base_cols].values.astype(np.float32)
            X_va = cache.loc[vm, base_cols].values.astype(np.float32)
            X_te = cache.loc[em, base_cols].values.astype(np.float32)

            t1 = time.time()
            m_base = train_xgb(X_tr[mtr], y_tr[mtr], X_va[mva], y_va[mva])
            p_base = m_base.predict(xgb.DMatrix(X_te[mte]))
            e_base = evaluate(p_base, y_te[mte], test_idx[mte])

            # Base + factor
            fg_tr = fdf.loc[tm].values.astype(np.float32)
            fg_va = fdf.loc[vm].values.astype(np.float32)
            fg_te = fdf.loc[em].values.astype(np.float32)

            X_tr2 = np.hstack([X_tr, fg_tr])
            X_va2 = np.hstack([X_va, fg_va])
            X_te2 = np.hstack([X_te, fg_te])

            m_plus = train_xgb(X_tr2[mtr], y_tr[mtr], X_va2[mva], y_va[mva])
            p_plus = m_plus.predict(xgb.DMatrix(X_te2[mte]))
            e_plus = evaluate(p_plus, y_te[mte], test_idx[mte])
            elapsed = time.time() - t1

            delta_ric = e_plus["rank_ic_mean"] - e_base["rank_ic_mean"]
            delta_spr = e_plus["top20_spread"] - e_base["top20_spread"]

            results.append({
                "split": split_idx + 1,
                "base_ric": e_base["rank_ic_mean"],
                "plus_ric": e_plus["rank_ic_mean"],
                "delta_ric": round(delta_ric, 6),
                "base_spr": e_base["top20_spread"],
                "plus_spr": e_plus["top20_spread"],
                "delta_spr": round(delta_spr, 6),
            })

            logger.info(f"  Split {split_idx+1}: base RankIC={e_base['rank_ic_mean']:+.4f} "
                        f"+{fg} RankIC={e_plus['rank_ic_mean']:+.4f} "
                        f"Δ={delta_ric:+.4f} [{elapsed:.0f}s]")

        # Summary
        n = len(results)
        d_rics = [r["delta_ric"] for r in results]
        d_sprs = [r["delta_spr"] for r in results]
        ric_helps = sum(1 for d in d_rics if d > 0)
        spr_helps = sum(1 for d in d_sprs if d > 0)

        summary = {
            "factor": fg,
            "n_cols": len(fg_cols),
            "n_splits": n,
            "avg_delta_ric": round(float(np.mean(d_rics)), 6),
            "avg_delta_spr": round(float(np.mean(d_sprs)), 6),
            "ric_helps_pct": round(ric_helps / n, 4),
            "spr_helps_pct": round(spr_helps / n, 4),
            "pass_70pct": ric_helps / n >= 0.7 or spr_helps / n >= 0.7,
            "splits": results,
        }
        all_results[fg] = summary

        logger.info(f"\n  {fg} summary ({n} splits):")
        logger.info(f"    avg Δ RankIC: {np.mean(d_rics):+.4f}")
        logger.info(f"    avg Δ Spread: {np.mean(d_sprs)*100:+.3f}%")
        logger.info(f"    Δ RankIC>0: {ric_helps}/{n} ({ric_helps/n:.0%})")
        logger.info(f"    Δ Spread>0: {spr_helps}/{n} ({spr_helps/n:.0%})")
        logger.info(f"    Phase 2 gate (≥70%): {'✅ PASS' if summary['pass_70pct'] else '❌ FAIL'}")

    # Final summary
    total_time = time.time() - t_total
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 2 FACTOR ABLATION SUMMARY ({total_time:.0f}s)")
    logger.info(f"{'='*60}")
    logger.info(f"{'Factor':<15} {'Cols':>5} {'Δ RankIC':>10} {'Δ Spread':>10} "
                f"{'RIC helps':>10} {'SPR helps':>10} {'Gate':>6}")
    logger.info("-" * 70)
    for fg, s in all_results.items():
        logger.info(f"{fg:<15} {s['n_cols']:>5} {s['avg_delta_ric']:+.4f}    "
                    f"{s['avg_delta_spr']*100:+.3f}%    "
                    f"{s['ric_helps_pct']:.0%}         {s['spr_helps_pct']:.0%}         "
                    f"{'✅' if s['pass_70pct'] else '❌'}")

    # Save
    out_path = DATA_DIR / "phase4" / "phase2_factor_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump({"evaluated_at": datetime.now().isoformat(timespec="seconds"),
                    "total_time_s": round(total_time, 1),
                    "results": all_results}, f, indent=2)
    logger.info(f"\nSaved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
