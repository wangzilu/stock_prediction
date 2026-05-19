"""Ablation: 174-dim base vs base + sector spillover features.

Sector spillover = per-stock overseas proxy features (not broad broadcast).
Each stock gets its industry-specific proxy: NASDAQ→电子, HSTECH→计算机, HSI→fallback.

Usage:
    python scripts/ablation_sector_spillover.py
"""
import json
import logging
import os
import sys
import time
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


def train_xgb(X_tr, y_tr, X_va, y_va):
    import xgboost as xgb
    dt = xgb.DMatrix(X_tr, label=y_tr)
    dv = xgb.DMatrix(X_va, label=y_va)
    params = {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
              "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
              "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": SEED}
    return xgb.train(params, dt, num_boost_round=400,
                     evals=[(dv, "valid")], early_stopping_rounds=30, verbose_eval=0)


def evaluate(pred, label, index):
    from scipy.stats import spearmanr
    mask = np.isfinite(pred) & np.isfinite(label)
    ps = pd.Series(pred[mask], index=index[mask])
    ls = pd.Series(label[mask], index=index[mask])
    rics, sprs = [], []
    for date in ps.index.get_level_values(0).unique():
        p = ps.loc[date].values; l = ls.loc[date].values
        if len(p) < 40: continue
        c, _ = spearmanr(p, l)
        if np.isfinite(c): rics.append(c)
        k = min(20, len(p) // 2)
        top = np.argpartition(p, -k)[-k:]
        bot = np.argpartition(p, k)[:k]
        sprs.append(l[top].mean() - l[bot].mean())
    return {"ric": round(float(np.nanmean(rics)), 6) if rics else 0,
            "spr": round(float(np.mean(sprs)), 6) if sprs else 0}


def main():
    import xgboost as xgb
    from config.qlib_runtime import init_qlib
    init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

    # Load base cache
    cache = pd.read_parquet(str(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"))
    base_cols = [c for c in cache.columns if not c.startswith("__") and not c.startswith("_")
                 and not c.startswith("hsi_") and not c.startswith("hstech_")
                 and not c.startswith("nasdaq_")]
    label_col = "__label_5d"
    logger.info(f"Base cache: {cache.shape}, {len(base_cols)} base cols")

    # Load sector spillover
    spill_path = DATA_DIR / "sector_spillover_features.parquet"
    if not spill_path.exists():
        logger.error("Sector spillover not found. Run: python scripts/build_sector_spillover.py")
        sys.exit(1)

    spill = pd.read_parquet(str(spill_path))
    spill_cols = [c for c in spill.columns if c.startswith("spill_") and c != "spill_proxy"]
    logger.info(f"Spillover: {spill.shape}, {len(spill_cols)} factor cols")

    # Align indexes
    common = cache.index.intersection(spill.index)
    logger.info(f"Common index: {len(common)} rows")
    cache = cache.loc[common]
    spill = spill.loc[common]

    trade_dates = sorted(cache.index.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1
    dl = cache.index.get_level_values(0)

    # Rolling ablation
    n_splits = 12
    results = []
    t_total = time.time()

    for split_idx in range(n_splits):
        test_end_idx = today_idx - split_idx * 20
        test_start_idx = test_end_idx - 20
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - 60
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - 750
        if train_start_idx < 0: break

        tm = (dl >= trade_dates[train_start_idx]) & (dl <= trade_dates[train_end_idx])
        vm = (dl >= trade_dates[valid_start_idx]) & (dl <= trade_dates[valid_end_idx])
        em = (dl >= trade_dates[test_start_idx]) & (dl <= trade_dates[test_end_idx])

        y_tr = cache.loc[tm, label_col].values.astype(np.float32)
        y_va = cache.loc[vm, label_col].values.astype(np.float32)
        y_te = cache.loc[em, label_col].values.astype(np.float32)
        mtr = np.isfinite(y_tr); mva = np.isfinite(y_va); mte = np.isfinite(y_te)
        test_idx = cache.index[em]

        X_tr = cache.loc[tm, base_cols].values.astype(np.float32)
        X_va = cache.loc[vm, base_cols].values.astype(np.float32)
        X_te = cache.loc[em, base_cols].values.astype(np.float32)

        # Base
        m_base = train_xgb(X_tr[mtr], y_tr[mtr], X_va[mva], y_va[mva])
        p_base = m_base.predict(xgb.DMatrix(X_te[mte]))
        e_base = evaluate(p_base, y_te[mte], test_idx[mte])

        # Base + spillover
        s_tr = spill.loc[tm, spill_cols].values.astype(np.float32)
        s_va = spill.loc[vm, spill_cols].values.astype(np.float32)
        s_te = spill.loc[em, spill_cols].values.astype(np.float32)

        m_plus = train_xgb(np.hstack([X_tr, s_tr])[mtr], y_tr[mtr],
                           np.hstack([X_va, s_va])[mva], y_va[mva])
        p_plus = m_plus.predict(xgb.DMatrix(np.hstack([X_te, s_te])[mte]))
        e_plus = evaluate(p_plus, y_te[mte], test_idx[mte])

        delta_ric = e_plus["ric"] - e_base["ric"]
        delta_spr = e_plus["spr"] - e_base["spr"]
        results.append({"split": split_idx + 1, "delta_ric": delta_ric, "delta_spr": delta_spr,
                        "base_ric": e_base["ric"], "plus_ric": e_plus["ric"]})

        logger.info(f"Split {split_idx+1}: base={e_base['ric']:+.4f} +spill={e_plus['ric']:+.4f} "
                    f"Δ={delta_ric:+.4f}")

    # Summary
    n = len(results)
    d_rics = [r["delta_ric"] for r in results]
    d_sprs = [r["delta_spr"] for r in results]
    pos_ric = sum(1 for d in d_rics if d > 0)
    pos_spr = sum(1 for d in d_sprs if d > 0)

    logger.info(f"\n{'='*60}")
    logger.info(f"SECTOR SPILLOVER ABLATION ({n} splits, {time.time()-t_total:.0f}s)")
    logger.info(f"{'='*60}")
    logger.info(f"  avg Δ RankIC: {np.mean(d_rics):+.4f}")
    logger.info(f"  avg Δ Spread: {np.mean(d_sprs)*100:+.3f}%")
    logger.info(f"  Δ RankIC>0: {pos_ric}/{n} ({pos_ric/n:.0%}) {'✅' if pos_ric/n >= 0.7 else '❌'}")
    logger.info(f"  Δ Spread>0: {pos_spr}/{n} ({pos_spr/n:.0%}) {'✅' if pos_spr/n >= 0.7 else '❌'}")

    from utils.json_utils import json_default
    out_path = DATA_DIR / "phase4" / "sector_spillover_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump({"n_splits": n, "results": results,
                   "avg_delta_ric": round(float(np.mean(d_rics)), 6),
                   "ric_pos_pct": round(pos_ric / n, 4),
                   "spr_pos_pct": round(pos_spr / n, 4)}, f, indent=2, default=json_default)
    logger.info(f"Saved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
