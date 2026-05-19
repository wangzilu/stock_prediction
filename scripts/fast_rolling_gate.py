"""Fast rolling gate using pre-computed feature cache.

100x faster than phase4_rolling_gate.py: reads parquet instead of
recomputing Alpha158 + asof_merge for every split.

Prerequisite:
    python scripts/build_feature_cache.py --all

Usage:
    python scripts/fast_rolling_gate.py
    python scripts/fast_rolling_gate.py --n-splits 24 --cache feature_cache_174_regime.parquet
    python scripts/fast_rolling_gate.py --feature-cols "hsi_*,hstech_*,nasdaq_*" --ablation
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
SEED = 42


def get_trading_dates(cache: pd.DataFrame) -> list:
    """Extract sorted trading dates from cache index."""
    return sorted(cache.index.get_level_values(0).unique())


def train_xgb(X_train, y_train, X_valid, y_valid, nthread=12, max_rounds=300):
    import xgboost as xgb
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_valid, label=y_valid)
    params = {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
              "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
              "objective": "reg:squarederror", "nthread": nthread, "verbosity": 0, "seed": SEED}
    model = xgb.train(params, dt, num_boost_round=max_rounds,
                      evals=[(dv, "valid")], early_stopping_rounds=30, verbose_eval=0)
    return model


def evaluate(pred, label, index):
    from qlib.contrib.eva.alpha import calc_ic
    mask = np.isfinite(pred) & np.isfinite(label)
    ps = pd.Series(pred[mask], index=index[mask])
    ls = pd.Series(label[mask], index=index[mask])
    ic, ric = calc_ic(ps, ls)
    spreads = []
    for _, g in pd.DataFrame({"pred": ps, "label": ls}).groupby(level=0):
        if len(g) < 40:
            continue
        s = g.sort_values("pred", ascending=False)
        spreads.append(s.head(20)["label"].mean() - s.tail(20)["label"].mean())
    return {
        "ic_mean": round(float(ic.mean()), 6),
        "icir": round(float(ic.mean()) / (float(ic.std()) + 1e-8), 4),
        "rank_ic_mean": round(float(ric.mean()), 6),
        "rank_ic_pos": round(float((ric > 0).mean()), 4),
        "top20_spread": round(float(np.mean(spreads)) if spreads else 0, 6),
        "spread_pos": round(float(np.mean([s > 0 for s in spreads])) if spreads else 0, 4),
    }


def _run_one_split(args_tuple):
    """Worker function for parallel splits. No Qlib dependency."""
    (split_idx, X_train, y_train, X_valid, y_valid, X_test, y_test,
     test_idx_values, test_idx_tuples, feature_sets_cols, n_splits) = args_tuple

    import xgboost as xgb
    results = {"split": split_idx + 1}

    for fs_name, col_indices in feature_sets_cols.items():
        t1 = time.time()
        Xtr = X_train[:, col_indices]
        Xva = X_valid[:, col_indices]
        Xte = X_test[:, col_indices]

        model = train_xgb(Xtr, y_train, Xva, y_valid)
        pred = model.predict(xgb.DMatrix(Xte))

        # Inline evaluate (no Qlib dependency)
        mask = np.isfinite(pred) & np.isfinite(y_test)
        ps = pd.Series(pred[mask], index=pd.MultiIndex.from_tuples(
            [test_idx_tuples[i] for i in range(len(mask)) if mask[i]]))
        ls = pd.Series(y_test[mask], index=ps.index)

        # Rank IC
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

        ric_arr = np.array(ric_vals)
        metrics = {
            "rank_ic_mean": round(float(np.nanmean(ric_arr)), 6) if len(ric_arr) > 0 else 0,
            "rank_ic_pos": round(float(np.nanmean(ric_arr > 0)), 4) if len(ric_arr) > 0 else 0,
            "top20_spread": round(float(np.mean(spreads)), 6) if spreads else 0,
            "spread_pos": round(float(np.mean([s > 0 for s in spreads])), 4) if spreads else 0,
        }
        results[fs_name] = {"n_feat": len(col_indices), **metrics, "time_s": round(time.time() - t1, 1)}

    return results


def main():
    import xgboost as xgb
    from config.qlib_runtime import init_qlib
    init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="feature_cache_174_holder_regime_ma.parquet",
                        help="Cache file name in data/storage/")
    parser.add_argument("--n-splits", type=int, default=24)
    parser.add_argument("--test-days", type=int, default=20)
    parser.add_argument("--valid-days", type=int, default=60)
    parser.add_argument("--train-days", type=int, default=750,  # ~3 years
                        help="Training window in trading days")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation: base vs base+extra columns")
    parser.add_argument("--extra-cols", type=str, default="hsi_*,hstech_*,nasdaq_*",
                        help="Extra column patterns for ablation (comma-separated globs)")
    parser.add_argument("--parallel", type=int, default=0,
                        help="Number of parallel workers (0=sequential)")
    args = parser.parse_args()

    # Load cache
    cache_path = DATA_DIR / args.cache
    if not cache_path.exists():
        logger.error(f"Cache not found: {cache_path}")
        logger.error("Run: python scripts/build_feature_cache.py --all")
        sys.exit(1)

    logger.info(f"Loading cache: {cache_path}")
    t0 = time.time()
    cache = pd.read_parquet(str(cache_path))
    logger.info(f"  Loaded: {cache.shape}, {time.time()-t0:.1f}s")

    # Separate features, labels, and metadata columns
    meta_cols = [c for c in cache.columns if c.startswith("__")]
    feature_cols = [c for c in cache.columns if not c.startswith("__")]
    label_col = "__label_5d"

    if label_col not in cache.columns:
        logger.error(f"Label column {label_col} not found")
        sys.exit(1)

    # For ablation: split feature cols into base and extra
    if args.ablation:
        import fnmatch
        extra_patterns = [p.strip() for p in args.extra_cols.split(",")]
        extra_cols = []
        for col in feature_cols:
            if any(fnmatch.fnmatch(col, pat) for pat in extra_patterns):
                extra_cols.append(col)
        base_cols = [c for c in feature_cols if c not in extra_cols]
        logger.info(f"  Ablation: {len(base_cols)} base + {len(extra_cols)} extra")
        logger.info(f"  Extra cols: {extra_cols[:10]}...")
        feature_sets = {"base": base_cols, "base+extra": feature_cols}
    else:
        feature_sets = {"all": feature_cols}

    # Get trading dates
    trade_dates = get_trading_dates(cache)
    today_idx = len(trade_dates) - 1
    logger.info(f"  Trading dates: {len(trade_dates)}")

    # Pre-compute column indices for each feature set (for parallel mode)
    all_cols = list(cache.columns)
    feature_sets_col_indices = {}
    for fs_name, cols in feature_sets.items():
        feature_sets_col_indices[fs_name] = [all_cols.index(c) for c in cols]

    # Pre-slice all splits
    dates_level = cache.index.get_level_values(0)
    split_specs = []

    for split_idx in range(args.n_splits):
        test_end_idx = today_idx - split_idx * args.test_days
        test_start_idx = test_end_idx - args.test_days
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - args.valid_days
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - args.train_days

        if train_start_idx < 0:
            break

        test_end = trade_dates[test_end_idx]
        test_start = trade_dates[test_start_idx]
        valid_start = trade_dates[valid_start_idx]
        valid_end = trade_dates[valid_end_idx]
        train_start = trade_dates[train_start_idx]
        train_end = trade_dates[train_end_idx]

        split_specs.append({
            "split_idx": split_idx,
            "test": f"{str(test_start)[:10]}~{str(test_end)[:10]}",
            "train_start": train_start, "train_end": train_end,
            "valid_start": valid_start, "valid_end": valid_end,
            "test_start": test_start, "test_end": test_end,
        })

    logger.info(f"  {len(split_specs)} splits prepared, parallel={args.parallel}")

    # Rolling
    all_results = []
    t_total = time.time()

    if args.parallel > 1:
        # === Parallel mode ===
        # Pre-slice ALL splits in main thread (pandas GIL bottleneck),
        # then dispatch only numpy arrays to threads (XGB releases GIL).
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import xgboost as xgb

        logger.info(f"  Pre-slicing {len(split_specs)} splits in main thread...")
        t_pre = time.time()

        prepped_splits = []
        for spec in split_specs:
            tm = (dates_level >= spec["train_start"]) & (dates_level <= spec["train_end"])
            vm = (dates_level >= spec["valid_start"]) & (dates_level <= spec["valid_end"])
            em = (dates_level >= spec["test_start"]) & (dates_level <= spec["test_end"])

            split_data = {"split_idx": spec["split_idx"], "test": spec["test"], "fs": {}}

            for fs_name, cols in feature_sets.items():
                y_tr = cache.loc[tm, label_col].values.astype(np.float32)
                y_va = cache.loc[vm, label_col].values.astype(np.float32)
                y_te = cache.loc[em, label_col].values.astype(np.float32)
                X_tr = cache.loc[tm, cols].values.astype(np.float32)
                X_va = cache.loc[vm, cols].values.astype(np.float32)
                X_te = cache.loc[em, cols].values.astype(np.float32)
                test_idx = cache.index[em]

                mtr = np.isfinite(y_tr); mva = np.isfinite(y_va); mte = np.isfinite(y_te)

                split_data["fs"][fs_name] = {
                    "X_tr": X_tr[mtr], "y_tr": y_tr[mtr],
                    "X_va": X_va[mva], "y_va": y_va[mva],
                    "X_te": X_te[mte], "y_te": y_te[mte],
                    "test_idx": test_idx[mte], "n_feat": len(cols),
                }

            prepped_splits.append(split_data)

        logger.info(f"  Pre-sliced in {time.time()-t_pre:.1f}s")

        # Use more cores for parallel: nthread=3 * 4 workers = 12 of 14 cores
        parallel_xgb_params = {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
                               "colsample_bytree": 0.8879, "reg_alpha": 205.6999,
                               "reg_lambda": 580.9768, "objective": "reg:squarederror",
                               "nthread": 3, "verbosity": 0, "seed": SEED}

        def _train_xgb_single(X_tr, y_tr, X_va, y_va):
            dt = xgb.DMatrix(X_tr, label=y_tr)
            dv = xgb.DMatrix(X_va, label=y_va)
            return xgb.train(parallel_xgb_params, dt, num_boost_round=500,
                             evals=[(dv, "valid")], early_stopping_rounds=50, verbose_eval=0)

        def train_and_eval(split_data):
            """Thread worker: only XGB train + predict + evaluate. No pandas slicing."""
            result = {"split": split_data["split_idx"] + 1, "test": split_data["test"]}
            for fs_name, d in split_data["fs"].items():
                t1 = time.time()
                model = _train_xgb_single(d["X_tr"], d["y_tr"], d["X_va"], d["y_va"])
                pred = model.predict(xgb.DMatrix(d["X_te"]))
                metrics = evaluate(pred, d["y_te"], d["test_idx"])
                result[fs_name] = {"n_feat": d["n_feat"], **metrics,
                                   "time_s": round(time.time() - t1, 1)}
            if args.ablation and "base" in result and "base+extra" in result:
                result["delta_rank_ic"] = round(
                    result["base+extra"]["rank_ic_mean"] - result["base"]["rank_ic_mean"], 6)
                result["delta_spread"] = round(
                    result["base+extra"]["top20_spread"] - result["base"]["top20_spread"], 6)
            return result

        logger.info(f"  Launching {args.parallel} threads for {len(prepped_splits)} splits...")
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(train_and_eval, sd): sd for sd in prepped_splits}
            results = []
            for future in as_completed(futures):
                try:
                    r = future.result(timeout=300)
                    results.append(r)
                    # Quick log
                    fs0 = list(feature_sets.keys())[0]
                    m = r.get(fs0, {})
                    logger.info(f"  Split {r['split']} done ({len(results)}/{len(prepped_splits)}) "
                                f"RankIC={m.get('rank_ic_mean',0):+.4f} [{m.get('time_s',0):.0f}s]")
                except Exception as e:
                    sd = futures[future]
                    logger.error(f"  Split {sd['split_idx']+1} FAILED: {e}")

        for r in sorted(results, key=lambda x: x["split"]):
            all_results.append(r)

    else:
        # === Sequential mode (original) ===
        for spec in split_specs:
            split_idx = spec["split_idx"]
            logger.info(f"\nSplit {split_idx+1}/{args.n_splits}: test {spec['test']}")

            train_mask = (dates_level >= spec["train_start"]) & (dates_level <= spec["train_end"])
            valid_mask = (dates_level >= spec["valid_start"]) & (dates_level <= spec["valid_end"])
            test_mask = (dates_level >= spec["test_start"]) & (dates_level <= spec["test_end"])

            split_result = {"split": split_idx + 1, "test": spec["test"]}

            for fs_name, cols in feature_sets.items():
                y_train = cache.loc[train_mask, label_col].values.astype(np.float32)
                y_valid = cache.loc[valid_mask, label_col].values.astype(np.float32)
                y_test = cache.loc[test_mask, label_col].values.astype(np.float32)

                X_train = cache.loc[train_mask, cols].values.astype(np.float32)
                X_valid = cache.loc[valid_mask, cols].values.astype(np.float32)
                X_test = cache.loc[test_mask, cols].values.astype(np.float32)

                test_idx = cache.index[test_mask]

                mask_tr = np.isfinite(y_train)
                mask_va = np.isfinite(y_valid)
                mask_te = np.isfinite(y_test)

                t1 = time.time()
                model = train_xgb(X_train[mask_tr], y_train[mask_tr],
                                   X_valid[mask_va], y_valid[mask_va])
                pred = model.predict(xgb.DMatrix(X_test[mask_te]))
                metrics = evaluate(pred, y_test[mask_te], test_idx[mask_te])
                elapsed = time.time() - t1

                split_result[fs_name] = {"n_feat": len(cols), **metrics, "time_s": round(elapsed, 1)}
                logger.info(f"  {fs_name}({len(cols)}): RankIC={metrics['rank_ic_mean']:+.4f} "
                            f"Spread={metrics['top20_spread']*100:+.3f}% [{elapsed:.1f}s]")

            if args.ablation:
                delta_ric = split_result["base+extra"]["rank_ic_mean"] - split_result["base"]["rank_ic_mean"]
                delta_spr = split_result["base+extra"]["top20_spread"] - split_result["base"]["top20_spread"]
                split_result["delta_rank_ic"] = round(delta_ric, 6)
                split_result["delta_spread"] = round(delta_spr, 6)
                logger.info(f"  Δ RankIC={delta_ric:+.4f} Δ Spread={delta_spr*100:+.3f}%")

            all_results.append(split_result)

    # Summary
    total_time = time.time() - t_total
    n = len(all_results)

    logger.info(f"\n{'='*70}")
    logger.info(f"FAST ROLLING GATE ({n} splits, {total_time:.1f}s total)")
    logger.info(f"{'='*70}")

    for fs_name in feature_sets:
        rics = [r[fs_name]["rank_ic_mean"] for r in all_results]
        sprs = [r[fs_name]["top20_spread"] for r in all_results]
        logger.info(f"\n  {fs_name}:")
        logger.info(f"    avg RankIC: {np.mean(rics):+.4f}")
        logger.info(f"    avg Spread: {np.mean(sprs)*100:+.3f}%")
        logger.info(f"    RankIC>0:   {sum(1 for r in rics if r > 0)}/{n}")
        logger.info(f"    Spread>0:   {sum(1 for s in sprs if s > 0)}/{n}")

    if args.ablation:
        delta_rics = [r["delta_rank_ic"] for r in all_results]
        delta_sprs = [r["delta_spread"] for r in all_results]
        logger.info(f"\n  Ablation delta:")
        logger.info(f"    Δ RankIC>0: {sum(1 for d in delta_rics if d > 0)}/{n}")
        logger.info(f"    Δ Spread>0: {sum(1 for d in delta_sprs if d > 0)}/{n}")

    # Save
    out_path = DATA_DIR / "phase4" / f"fast_rolling_{'ablation' if args.ablation else 'gate'}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump({"evaluated_at": datetime.now().isoformat(timespec="seconds"),
                    "cache": args.cache, "n_splits": n,
                    "total_time_s": round(total_time, 1),
                    "splits": all_results}, f, indent=2)
    logger.info(f"\nSaved: {out_path}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info("Done!")


if __name__ == "__main__":
    main()
