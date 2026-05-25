"""Phase 4S: Validate regime-weighted training vs uniform training.

Compares XGBoost trained with uniform sample weights against
regime-weighted sample weights across 6 rolling splits and 3 temperature
settings (0.2, 0.5, 1.0).

Usage: python scripts/phase4s_regime_weighted_validation.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "storage")
FEATURE_CACHE = os.path.join(DATA_DIR, "feature_cache_174_holder_regime_ma.parquet")

LABEL_COL = "__label_5d"

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

TEMPERATURES = [0.2, 0.5, 1.0]


def compute_daily_rank_ic(pred: pd.Series, label: pd.Series) -> list[float]:
    """Compute per-day Spearman RankIC between prediction and label."""
    common = pred.index.intersection(label.index)
    pred = pred.loc[common]
    label = label.loc[common]
    mask = pred.notna() & label.notna()
    pred = pred[mask]
    label = label[mask]

    rank_ics = []
    dates = pred.index.get_level_values("datetime").unique()
    for dt in dates:
        p = pred.xs(dt, level="datetime")
        l = label.xs(dt, level="datetime")
        ci = p.index.intersection(l.index)
        p, l = p.loc[ci], l.loc[ci]
        finite = np.isfinite(p.values) & np.isfinite(l.values)
        p_f, l_f = p[finite], l[finite]
        if len(p_f) < 30:
            continue
        ric = spearmanr(p_f.values, l_f.values).statistic
        if np.isfinite(ric):
            rank_ics.append(ric)
    return rank_ics


def main():
    from config.rolling_splits import get_standard_splits
    from backtest.regime_sampler import compute_sample_weights_for_index
    import xgboost as xgb

    print("=" * 70)
    print("Phase 4S: Regime-Weighted Training Validation")
    print("=" * 70)

    # ── Load data ───────────────────────────────────────────────────────
    print(f"\nLoading feature cache: {FEATURE_CACHE}")
    df = pd.read_parquet(FEATURE_CACHE)
    date_idx = df.index.get_level_values("datetime")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {date_idx.min()} to {date_idx.max()}")

    feat_cols = [c for c in df.columns if not c.startswith("__")]
    print(f"  Features: {len(feat_cols)}, Label: {LABEL_COL}")

    # ── Generate 6-split configuration ──────────────────────────────────
    splits = get_standard_splits(preset="6split")
    print(f"\nUsing 6-split fast config ({len(splits)} splits generated)")

    # ── Results storage ─────────────────────────────────────────────────
    # Key: (split_id, method) -> list of daily RankICs
    all_results = []

    total_tasks = len(splits) * (1 + len(TEMPERATURES))  # uniform + 3 temps
    task_num = 0
    t_global = time.time()

    for sp in splits:
        sid = sp["split_id"]
        train_start, train_end = sp["train_start"], sp["train_end"]
        test_start, test_end = sp["test_start"], sp["test_end"]

        print(f"\n{'─' * 60}")
        print(f"Split {sid}: train [{train_start} .. {train_end}], "
              f"test [{test_start} .. {test_end}]")
        print(f"{'─' * 60}")

        # Slice data
        dates = df.index.get_level_values("datetime")
        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= test_start) & (dates <= test_end)

        train_df = df.loc[train_mask].dropna(subset=[LABEL_COL])
        test_df = df.loc[test_mask].dropna(subset=[LABEL_COL])

        if train_df.shape[0] < 100 or test_df.shape[0] < 100:
            print(f"  SKIP: insufficient data (train={train_df.shape[0]}, test={test_df.shape[0]})")
            continue

        X_train = train_df[feat_cols].replace([np.inf, -np.inf], np.nan)
        y_train = train_df[LABEL_COL]
        X_test = test_df[feat_cols].replace([np.inf, -np.inf], np.nan)
        y_test = test_df[LABEL_COL]

        print(f"  Train: {X_train.shape[0]:,} rows, Test: {X_test.shape[0]:,} rows")

        # ── (a) Uniform baseline ────────────────────────────────────────
        task_num += 1
        print(f"\n  [{task_num}/{total_tasks}] Training UNIFORM baseline ...")
        t0 = time.time()
        model_u = xgb.XGBRegressor(**XGB_PARAMS)
        model_u.fit(X_train.values, y_train.values)
        pred_u = pd.Series(model_u.predict(X_test.values), index=X_test.index)
        rics_u = compute_daily_rank_ic(pred_u, y_test)
        elapsed = time.time() - t0
        mean_ric_u = np.mean(rics_u) if rics_u else float("nan")
        print(f"    RankIC = {mean_ric_u:.6f}  ({len(rics_u)} days, {elapsed:.1f}s)")

        all_results.append({
            "split": sid,
            "method": "uniform",
            "temperature": None,
            "rank_ic_mean": mean_ric_u,
            "rank_ic_std": np.std(rics_u) if rics_u else float("nan"),
            "n_days": len(rics_u),
            "time_sec": round(elapsed, 1),
        })

        # ── (b) Regime-weighted for each temperature ────────────────────
        target_date = test_start  # first date of test period

        for temp in TEMPERATURES:
            task_num += 1
            print(f"\n  [{task_num}/{total_tasks}] Training REGIME-WEIGHTED "
                  f"(temp={temp}, target={target_date}) ...")
            t0 = time.time()

            # Compute regime weights
            try:
                sample_weights = compute_sample_weights_for_index(
                    index=train_df.index,
                    target_date=target_date,
                    temperature=temp,
                )
            except Exception as e:
                print(f"    FAILED to compute regime weights: {e}")
                all_results.append({
                    "split": sid,
                    "method": f"regime_t{temp}",
                    "temperature": temp,
                    "rank_ic_mean": float("nan"),
                    "rank_ic_std": float("nan"),
                    "n_days": 0,
                    "time_sec": 0,
                })
                continue

            # Normalize weights so they sum to len(sample_weights)
            # This keeps the effective sample size comparable to uniform
            w = sample_weights * len(sample_weights) / sample_weights.sum()

            model_r = xgb.XGBRegressor(**XGB_PARAMS)
            model_r.fit(X_train.values, y_train.values, sample_weight=w)
            pred_r = pd.Series(model_r.predict(X_test.values), index=X_test.index)
            rics_r = compute_daily_rank_ic(pred_r, y_test)
            elapsed = time.time() - t0
            mean_ric_r = np.mean(rics_r) if rics_r else float("nan")
            print(f"    RankIC = {mean_ric_r:.6f}  ({len(rics_r)} days, {elapsed:.1f}s)")

            # Weight concentration diagnostic
            w_sorted = np.sort(sample_weights)[::-1]
            top10pct = w_sorted[: max(1, len(w_sorted) // 10)].sum()
            print(f"    Weight concentration: top 10% of dates hold "
                  f"{top10pct:.1%} of total weight")

            all_results.append({
                "split": sid,
                "method": f"regime_t{temp}",
                "temperature": temp,
                "rank_ic_mean": mean_ric_r,
                "rank_ic_std": np.std(rics_r) if rics_r else float("nan"),
                "n_days": len(rics_r),
                "time_sec": round(elapsed, 1),
            })

    # ── Aggregate results ───────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 70}")

    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        print("No results collected. Exiting.")
        return

    # Per-split comparison table
    print("\n── Per-Split RankIC ──")
    pivot = results_df.pivot(index="split", columns="method", values="rank_ic_mean")
    # Reorder columns: uniform first, then regime temps
    col_order = ["uniform"] + [f"regime_t{t}" for t in TEMPERATURES]
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]
    print(pivot.to_string(float_format=lambda x: f"{x:.6f}"))

    # Summary statistics
    print("\n── Summary ──")
    for method in col_order:
        vals = results_df[results_df["method"] == method]["rank_ic_mean"].dropna()
        print(f"  {method:16s}  mean={vals.mean():.6f}  std={vals.std():.6f}  "
              f"n_splits={len(vals)}")

    # Win rates vs uniform
    uniform_vals = results_df[results_df["method"] == "uniform"].set_index("split")["rank_ic_mean"]

    print("\n── Regime-Weighted vs Uniform ──")
    for temp in TEMPERATURES:
        method = f"regime_t{temp}"
        regime_vals = results_df[results_df["method"] == method].set_index("split")["rank_ic_mean"]
        common_splits = uniform_vals.index.intersection(regime_vals.index)
        if len(common_splits) == 0:
            continue
        u = uniform_vals.loc[common_splits]
        r = regime_vals.loc[common_splits]
        diff = r - u
        wins = (diff > 0).sum()
        total = len(diff)
        mean_diff = diff.mean()
        print(f"  temp={temp:.1f}:  wins={wins}/{total}  "
              f"mean_diff={mean_diff:+.6f}  "
              f"({'BETTER' if mean_diff > 0 else 'WORSE'})")

        # Simple t-test on paired differences
        if total >= 3:
            from scipy.stats import ttest_rel
            t_stat, p_val = ttest_rel(r.values, u.values)
            sig = "YES" if p_val < 0.05 else "NO"
            print(f"           paired t-test: t={t_stat:.3f}, p={p_val:.4f}  "
                  f"significant(p<0.05): {sig}")

    # Overall timing
    total_time = time.time() - t_global
    print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
    print("Done!")


if __name__ == "__main__":
    main()
