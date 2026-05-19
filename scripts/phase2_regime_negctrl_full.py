"""Phase 2: Full regime negative control suite (3 tests per CX).

1. date_shuffle: randomize date→value mapping (already validated: 75% pass)
2. circular_shift: shift regime by 60/120/250 days (preserves autocorrelation)
3. future_shift_guard: use FUTURE 1/5/20d regime (catches time-point leakage)

Real regime must beat ALL three controls to be confirmed as genuine signal.

Usage:
    python scripts/phase2_regime_negctrl_full.py
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


def train_xgb(X_train, y_train, X_valid, y_valid):
    import xgboost as xgb
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_valid, label=y_valid)
    params = {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
              "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
              "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": SEED}
    return xgb.train(params, dt, num_boost_round=400,
                     evals=[(dv, "valid")], early_stopping_rounds=30, verbose_eval=0)


def evaluate_ric(pred, label, index):
    from scipy.stats import spearmanr
    mask = np.isfinite(pred) & np.isfinite(label)
    ps = pd.Series(pred[mask], index=index[mask])
    ls = pd.Series(label[mask], index=index[mask])
    rics = []
    for date in ps.index.get_level_values(0).unique():
        p = ps.loc[date].values; l = ls.loc[date].values
        if len(p) < 40:
            continue
        corr, _ = spearmanr(p, l)
        if np.isfinite(corr):
            rics.append(corr)
    return round(float(np.nanmean(rics)), 6) if rics else 0.0


def make_date_shuffled(regime_df, dates):
    """Shuffle date→value mapping."""
    rng = np.random.RandomState(SEED)
    date_vals = regime_df.groupby(level=0).first()
    shuffled_dates = list(dates)
    rng.shuffle(shuffled_dates)
    remap = dict(zip(dates, shuffled_dates))
    result = regime_df.copy()
    mapped = result.index.get_level_values(0).map(remap)
    for col in result.columns:
        result[col] = date_vals[col].reindex(mapped).values
    return result


def make_circular_shift(regime_df, dates, shift_days):
    """Shift regime by N trading days (preserves autocorrelation)."""
    date_vals = regime_df.groupby(level=0).first()
    shifted_dates = list(dates)
    n = len(shifted_dates)
    remap = {shifted_dates[i]: shifted_dates[(i + shift_days) % n] for i in range(n)}
    result = regime_df.copy()
    mapped = result.index.get_level_values(0).map(remap)
    for col in result.columns:
        result[col] = date_vals[col].reindex(mapped).values
    return result


def make_future_shift(regime_df, dates, shift_days):
    """Use FUTURE regime (shift backward = look ahead). Should NOT improve model."""
    date_vals = regime_df.groupby(level=0).first()
    n = len(dates)
    # shift_days forward in time = look at future
    remap = {dates[i]: dates[min(i + shift_days, n - 1)] for i in range(n)}
    result = regime_df.copy()
    mapped = result.index.get_level_values(0).map(remap)
    for col in result.columns:
        result[col] = date_vals[col].reindex(mapped).values
    return result


def main():
    import xgboost as xgb
    from config.qlib_runtime import init_qlib
    init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

    n_splits = 8
    test_days = 20
    train_days = 750
    valid_days = 60

    cache_path = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
    logger.info(f"Loading cache: {cache_path}")
    cache = pd.read_parquet(str(cache_path))

    regime_cols = [c for c in cache.columns if c.startswith(("hsi_", "hstech_", "nasdaq_"))]
    base_cols = [c for c in cache.columns if not c.startswith("__") and not c.startswith("_")
                 and c not in regime_cols]
    label_col = "__label_5d"

    trade_dates = sorted(cache.index.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1
    dates_level = cache.index.get_level_values(0)

    logger.info(f"  Base: {len(base_cols)} cols, Regime: {len(regime_cols)} cols")

    # Build all control versions
    regime_df = cache[regime_cols]
    logger.info("Building control versions...")

    controls = {
        "real": regime_df,
        "date_shuffle": make_date_shuffled(regime_df, trade_dates),
        "circular_60d": make_circular_shift(regime_df, trade_dates, 60),
        "circular_120d": make_circular_shift(regime_df, trade_dates, 120),
        "circular_250d": make_circular_shift(regime_df, trade_dates, 250),
        "future_1d": make_future_shift(regime_df, trade_dates, 1),
        "future_5d": make_future_shift(regime_df, trade_dates, 5),
        "future_20d": make_future_shift(regime_df, trade_dates, 20),
    }
    logger.info(f"  {len(controls)} versions built")

    # Rolling comparison
    all_results = {name: [] for name in controls}
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
        mtr = np.isfinite(y_tr); mva = np.isfinite(y_va); mte = np.isfinite(y_te)
        test_idx = cache.index[em]

        X_base_tr = cache.loc[tm, base_cols].values.astype(np.float32)
        X_base_va = cache.loc[vm, base_cols].values.astype(np.float32)
        X_base_te = cache.loc[em, base_cols].values.astype(np.float32)

        logger.info(f"\nSplit {split_idx+1}/{n_splits}:")

        for name, ctrl_df in controls.items():
            r_tr = ctrl_df.loc[tm].values.astype(np.float32)
            r_va = ctrl_df.loc[vm].values.astype(np.float32)
            r_te = ctrl_df.loc[em].values.astype(np.float32)

            X_tr = np.hstack([X_base_tr, r_tr])
            X_va = np.hstack([X_base_va, r_va])
            X_te = np.hstack([X_base_te, r_te])

            model = train_xgb(X_tr[mtr], y_tr[mtr], X_va[mva], y_va[mva])
            pred = model.predict(xgb.DMatrix(X_te[mte]))
            ric = evaluate_ric(pred, y_te[mte], test_idx[mte])
            all_results[name].append(ric)

            marker = "✅" if name == "real" else "  "
            logger.info(f"  {marker} {name:<18} RankIC={ric:+.4f}")

    # Summary
    total_time = time.time() - t_total
    n = len(all_results["real"])

    logger.info(f"\n{'='*70}")
    logger.info(f"REGIME NEGATIVE CONTROL SUITE ({n} splits, {total_time:.0f}s)")
    logger.info(f"{'='*70}")

    real_rics = all_results["real"]
    real_avg = np.mean(real_rics)

    for name, rics in all_results.items():
        avg = np.mean(rics)
        beats = sum(1 for r, c in zip(real_rics, rics) if r > c)
        marker = "★" if name == "real" else ("✅" if beats / n >= 0.7 else "⚠️")
        logger.info(f"  {marker} {name:<18} avg={avg:+.4f}  real>ctrl: {beats}/{n} ({beats/n:.0%})")

    # Gate: real must beat ALL non-real controls in ≥70% splits
    gate_results = {}
    for name, rics in all_results.items():
        if name == "real":
            continue
        beats = sum(1 for r, c in zip(real_rics, rics) if r > c)
        gate_results[name] = {"avg": round(float(np.mean(rics)), 6),
                              "real_beats": beats,
                              "real_beats_pct": round(beats / n, 4),
                              "pass": beats / n >= 0.7}

    all_pass = all(g["pass"] for g in gate_results.values())
    logger.info(f"\n  Overall gate: {'✅ ALL PASS' if all_pass else '❌ SOME FAILED'}")

    # Check future_shift specifically — if future regime is STRONGER, we have leakage
    for name in ["future_1d", "future_5d", "future_20d"]:
        if name in all_results:
            future_avg = np.mean(all_results[name])
            if future_avg > real_avg * 1.1:
                logger.warning(f"  ⚠️ {name} avg={future_avg:+.4f} > real={real_avg:+.4f} — possible time leakage!")

    # Save
    from utils.json_utils import json_default
    out_path = DATA_DIR / "phase4" / "phase2_regime_negctrl_full.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump({"evaluated_at": datetime.now().isoformat(timespec="seconds"),
                    "n_splits": n, "total_time_s": round(total_time, 1),
                    "real_avg_ric": round(float(real_avg), 6),
                    "gate_results": gate_results,
                    "all_pass": all_pass,
                    "all_results": {k: [round(float(v), 6) for v in vs]
                                    for k, vs in all_results.items()}},
                   f, indent=2, default=json_default)
    logger.info(f"\nSaved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
