"""8-split rolling ablation for derived moneyflow+cyq factors against 174-base."""
import os, sys, time, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
from config.qlib_runtime import init_qlib
from models.feature_merger import FeatureMerger
from utils.json_utils import json_default
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

# ---------- load base cache ----------
cache = pd.read_parquet(str(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"))
base_cols = [c for c in cache.columns if not c.startswith('__') and not c.startswith('_')
             and not c.startswith('hsi_') and not c.startswith('hstech_') and not c.startswith('nasdaq_')]
label_col = '__label_5d'
merger = FeatureMerger(DATA_DIR)

trade_dates = sorted(cache.index.get_level_values(0).unique())
today_idx = len(trade_dates) - 1
dl = cache.index.get_level_values(0)
SEED = 42

# ---------- load derived factors ----------
logger.info("Loading derived moneyflow+cyq factors...")
df = pd.read_parquet(str(DATA_DIR / "derived_moneyflow_cyq.parquet"))
df['date'] = pd.to_datetime(df['date'])
df['qlib_code'] = df['qlib_code'].str.upper()

factor_cols = [c for c in df.columns if c not in ('qlib_code', 'date')]
logger.info(f"  factor columns: {factor_cols}")

merged = merger._asof_merge_timeseries(df, cache.index, 'date', factor_cols)
if merged is None:
    logger.error("asof_merge returned None — aborting")
    sys.exit(1)
logger.info(f"  merged shape: {merged.shape}, coverage={merged.notna().any(axis=1).mean():.1%}")

# ---------- helpers ----------
def train_xgb_fn(X_tr, y_tr, X_va, y_va):
    dt = xgb.DMatrix(X_tr, label=y_tr)
    dv = xgb.DMatrix(X_va, label=y_va)
    params = {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
              "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
              "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": SEED}
    return xgb.train(params, dt, num_boost_round=400,
                     evals=[(dv, "valid")], early_stopping_rounds=30, verbose_eval=0)

def evaluate_fn(pred, label, index):
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
        k = min(20, len(p)//2)
        top = np.argpartition(p, -k)[-k:]
        bot = np.argpartition(p, k)[:k]
        sprs.append(l[top].mean() - l[bot].mean())
    return {"ric": round(float(np.nanmean(rics)), 6) if rics else 0,
            "spr": round(float(np.mean(sprs)), 6) if sprs else 0}

# ---------- 8-split rolling ablation ----------
n_splits = 8
results = {"derived_mf_cyq": []}
base_rics = []
plus_rics = []

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

    # Base model
    m_base = train_xgb_fn(X_tr[mtr], y_tr[mtr], X_va[mva], y_va[mva])
    p_base = m_base.predict(xgb.DMatrix(X_te[mte]))
    e_base = evaluate_fn(p_base, y_te[mte], test_idx[mte])

    # Base + derived
    fg_tr = merged.loc[tm].values.astype(np.float32)
    fg_va = merged.loc[vm].values.astype(np.float32)
    fg_te = merged.loc[em].values.astype(np.float32)

    m_plus = train_xgb_fn(np.hstack([X_tr, fg_tr])[mtr], y_tr[mtr],
                           np.hstack([X_va, fg_va])[mva], y_va[mva])
    p_plus = m_plus.predict(xgb.DMatrix(np.hstack([X_te, fg_te])[mte]))
    e_plus = evaluate_fn(p_plus, y_te[mte], test_idx[mte])

    delta = e_plus['ric'] - e_base['ric']
    results["derived_mf_cyq"].append(delta)
    base_rics.append(e_base['ric'])
    plus_rics.append(e_plus['ric'])

    logger.info(f"Split {split_idx+1}/{n_splits}: "
                f"base RankIC={e_base['ric']:+.4f}, "
                f"+derived RankIC={e_plus['ric']:+.4f}, "
                f"delta={delta:+.4f}, spr_base={e_base['spr']:+.4f}, spr_plus={e_plus['spr']:+.4f}")

# ---------- summary ----------
deltas = results["derived_mf_cyq"]
n = len(deltas)
pos = sum(1 for d in deltas if d > 0)

logger.info(f"\n{'='*60}")
logger.info("DERIVED FACTOR ABLATION RESULTS")
logger.info(f"{'='*60}")
logger.info(f"  Splits: {n}")
logger.info(f"  Avg base RankIC:    {np.mean(base_rics):+.4f}")
logger.info(f"  Avg +derived RankIC: {np.mean(plus_rics):+.4f}")
logger.info(f"  Avg delta:          {np.mean(deltas):+.4f}")
logger.info(f"  Positive splits:    {pos}/{n} ({pos/n:.0%})")
logger.info(f"  Verdict:            {'PASS' if pos/n >= 0.7 else 'FAIL'}")
logger.info(f"  Per-split deltas:   {[round(d,4) for d in deltas]}")

# ---------- save ----------
out_path = DATA_DIR / "phase4" / "derived_factor_ablation.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "factor_group": "derived_mf_cyq",
    "factor_cols": factor_cols,
    "n_splits": n,
    "avg_base_ric": round(float(np.mean(base_rics)), 6),
    "avg_plus_ric": round(float(np.mean(plus_rics)), 6),
    "avg_delta": round(float(np.mean(deltas)), 6),
    "positive_splits": pos,
    "pass": pos / n >= 0.7,
    "deltas": [round(float(d), 6) for d in deltas],
    "base_rics": [round(float(r), 6) for r in base_rics],
    "plus_rics": [round(float(r), 6) for r in plus_rics],
}
with open(str(out_path), 'w') as f:
    json.dump(payload, f, indent=2, default=json_default)
logger.info(f"Saved to {out_path}")
