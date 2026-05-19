"""Phase 2: Negative control — shuffled factor should NOT improve model.

For each factor group, shuffle instrument-factor mapping within each date
(preserving per-date distribution and coverage, only breaking the
stock-specific signal). If shuffled factor still improves the model,
the improvement is spurious (noise or pipeline leak).

Usage:
    python scripts/phase2_negative_control.py
    python scripts/phase2_negative_control.py --factor regime
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
        p_day = ps.loc[date]; l_day = ls.loc[date]
        if len(p_day) < 40:
            continue
        ric_vals.append(float(p_day.corr(l_day, method="spearman")))
        s = pd.DataFrame({"p": p_day, "l": l_day}).sort_values("p", ascending=False)
        spreads.append(s.head(20)["l"].mean() - s.tail(20)["l"].mean())
    ric = np.array(ric_vals)
    return {
        "rank_ic_mean": round(float(np.nanmean(ric)), 6) if len(ric) > 0 else 0,
        "top20_spread": round(float(np.mean(spreads)), 6) if spreads else 0,
    }


def shuffle_within_date(df: pd.DataFrame) -> pd.DataFrame:
    """Shuffle factor values within each date, breaking stock-specific signal.

    Preserves: per-date distribution, coverage, NaN pattern per date.
    Breaks: which stock gets which value.
    """
    rng = np.random.RandomState(SEED)
    result = df.copy()
    for date in result.index.get_level_values(0).unique():
        mask = result.index.get_level_values(0) == date
        sub = result.loc[mask]
        for col in result.columns:
            vals = sub[col].values.copy()
            # Only shuffle non-NaN values
            finite_mask = np.isfinite(vals) if vals.dtype.kind == 'f' else ~pd.isna(vals)
            finite_vals = vals[finite_mask]
            rng.shuffle(finite_vals)
            vals[finite_mask] = finite_vals
            result.loc[mask, col] = vals
    return result


def main():
    import xgboost as xgb
    from config.qlib_runtime import init_qlib
    init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", default="regime",
                        choices=["regime", "moneyflow", "cyq", "all"],
                        help="Factor to test negative control")
    parser.add_argument("--n-splits", type=int, default=12)
    parser.add_argument("--test-days", type=int, default=20)
    parser.add_argument("--train-days", type=int, default=750)
    parser.add_argument("--valid-days", type=int, default=60)
    args = parser.parse_args()

    cache_path = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
    logger.info(f"Loading cache: {cache_path}")
    cache = pd.read_parquet(str(cache_path))
    logger.info(f"  Shape: {cache.shape}")

    base_cols = [c for c in cache.columns
                 if not c.startswith("__") and not c.startswith("_")
                 and not c.startswith("hsi_") and not c.startswith("hstech_")
                 and not c.startswith("nasdaq_")]
    label_col = "__label_5d"

    # Define factor groups
    factor_defs = {
        "regime": [c for c in cache.columns if c.startswith(("hsi_", "hstech_", "nasdaq_"))],
    }
    # For moneyflow/cyq, we'd need to load from the ablation script's asof merge results
    # For now, only test regime (which is already in cache)

    factors_to_test = list(factor_defs.keys()) if args.factor == "all" else [args.factor]

    trade_dates = sorted(cache.index.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1

    all_results = {}

    for fg in factors_to_test:
        if fg not in factor_defs or not factor_defs[fg]:
            logger.warning(f"  {fg}: no columns in cache, skip")
            continue

        fg_cols = factor_defs[fg]
        logger.info(f"\n{'='*60}")
        logger.info(f"NEGATIVE CONTROL: {fg} ({len(fg_cols)} cols)")
        logger.info(f"{'='*60}")

        # Shuffle the factor columns
        logger.info("  Shuffling within each date...")
        t_shuf = time.time()
        shuffled = shuffle_within_date(cache[fg_cols])
        logger.info(f"  Shuffled in {time.time()-t_shuf:.1f}s")

        results_real = []
        results_shuffled = []

        dates_level = cache.index.get_level_values(0)

        for split_idx in range(args.n_splits):
            test_end_idx = today_idx - split_idx * args.test_days
            test_start_idx = test_end_idx - args.test_days
            valid_end_idx = test_start_idx - 1
            valid_start_idx = valid_end_idx - args.valid_days
            train_end_idx = valid_start_idx - 1
            train_start_idx = train_end_idx - args.train_days
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

            # base + real factor
            fg_tr = cache.loc[tm, fg_cols].values.astype(np.float32)
            fg_va = cache.loc[vm, fg_cols].values.astype(np.float32)
            fg_te = cache.loc[em, fg_cols].values.astype(np.float32)

            m_real = train_xgb(np.hstack([X_base_tr, fg_tr])[mtr], y_tr[mtr],
                               np.hstack([X_base_va, fg_va])[mva], y_va[mva])
            p_real = m_real.predict(xgb.DMatrix(np.hstack([X_base_te, fg_te])[mte]))
            e_real = evaluate(p_real, y_te[mte], test_idx[mte])

            # base + shuffled factor
            sfg_tr = shuffled.loc[tm].values.astype(np.float32)
            sfg_va = shuffled.loc[vm].values.astype(np.float32)
            sfg_te = shuffled.loc[em].values.astype(np.float32)

            m_shuf = train_xgb(np.hstack([X_base_tr, sfg_tr])[mtr], y_tr[mtr],
                               np.hstack([X_base_va, sfg_va])[mva], y_va[mva])
            p_shuf = m_shuf.predict(xgb.DMatrix(np.hstack([X_base_te, sfg_te])[mte]))
            e_shuf = evaluate(p_shuf, y_te[mte], test_idx[mte])

            results_real.append(e_real)
            results_shuffled.append(e_shuf)

            logger.info(f"  Split {split_idx+1}: "
                        f"real RankIC={e_real['rank_ic_mean']:+.4f} "
                        f"shuffled RankIC={e_shuf['rank_ic_mean']:+.4f} "
                        f"{'✅ real>shuf' if e_real['rank_ic_mean'] > e_shuf['rank_ic_mean'] else '⚠️ shuf≥real'}")

        # Summary
        n = len(results_real)
        real_rics = [r["rank_ic_mean"] for r in results_real]
        shuf_rics = [r["rank_ic_mean"] for r in results_shuffled]
        real_beats = sum(1 for r, s in zip(real_rics, shuf_rics) if r > s)

        logger.info(f"\n  {fg} negative control ({n} splits):")
        logger.info(f"    avg real RankIC:     {np.mean(real_rics):+.4f}")
        logger.info(f"    avg shuffled RankIC: {np.mean(shuf_rics):+.4f}")
        logger.info(f"    real > shuffled:     {real_beats}/{n} ({real_beats/n:.0%})")
        logger.info(f"    PASS (real > shuffled ≥70%): "
                    f"{'✅' if real_beats/n >= 0.7 else '❌'}")

        all_results[fg] = {
            "n_splits": n,
            "avg_real_ric": round(float(np.mean(real_rics)), 6),
            "avg_shuf_ric": round(float(np.mean(shuf_rics)), 6),
            "real_beats_pct": round(real_beats / n, 4),
            "pass": real_beats / n >= 0.7,
        }

    # Save
    out_path = DATA_DIR / "phase4" / "phase2_negative_control.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump({"evaluated_at": datetime.now().isoformat(timespec="seconds"),
                    "results": all_results}, f, indent=2)
    logger.info(f"\nSaved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
