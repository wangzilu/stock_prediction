"""Quick ablation for new ST data: block_trade, top_inst."""
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

cache = pd.read_parquet(str(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"))
base_cols = [c for c in cache.columns if not c.startswith('__') and not c.startswith('_')
             and not c.startswith('hsi_') and not c.startswith('hstech_') and not c.startswith('nasdaq_')]
label_col = '__label_5d'
merger = FeatureMerger(DATA_DIR)

trade_dates = sorted(cache.index.get_level_values(0).unique())
today_idx = len(trade_dates) - 1
dl = cache.index.get_level_values(0)
SEED = 42

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

# Load factors
factors = {}

logger.info("Loading block_trade...")
bt = pd.read_parquet(str(DATA_DIR / 'st_block_trade.parquet'))
bt['trade_date'] = pd.to_datetime(bt['trade_date'], format='%Y%m%d', errors='coerce')
bt['qlib_code'] = bt['qlib_code'].str.upper()
bt['bt_vol'] = pd.to_numeric(bt['vol'], errors='coerce')
bt['bt_amount'] = pd.to_numeric(bt['amount'], errors='coerce')
bt_agg = bt.groupby(['qlib_code', 'trade_date']).agg(
    bt_count=('bt_vol', 'count'),
    bt_total_vol=('bt_vol', 'sum'),
    bt_total_amount=('bt_amount', 'sum'),
).reset_index()
bt_agg['date'] = bt_agg['trade_date']
bt_cols = ['bt_count', 'bt_total_vol', 'bt_total_amount']
bt_merged = merger._asof_merge_timeseries(bt_agg[['qlib_code', 'date'] + bt_cols],
                                           cache.index, 'date', bt_cols)
if bt_merged is not None:
    factors['block_trade'] = bt_merged
    logger.info(f"  block_trade: {bt_merged.shape[1]} cols, coverage={bt_merged.notna().any(axis=1).mean():.1%}")

logger.info("Loading top_inst...")
ti = pd.read_parquet(str(DATA_DIR / 'st_top_inst.parquet'))
ti['trade_date'] = pd.to_datetime(ti['trade_date'], format='%Y%m%d', errors='coerce')
ti['qlib_code'] = ti['qlib_code'].str.upper()
ti['ti_net_buy'] = pd.to_numeric(ti['net_buy'], errors='coerce')
ti['ti_buy'] = pd.to_numeric(ti['buy'], errors='coerce')
ti_agg = ti.groupby(['qlib_code', 'trade_date']).agg(
    ti_count=('ti_net_buy', 'count'),
    ti_net_buy_sum=('ti_net_buy', 'sum'),
    ti_buy_sum=('ti_buy', 'sum'),
).reset_index()
ti_agg['date'] = ti_agg['trade_date']
ti_cols = ['ti_count', 'ti_net_buy_sum', 'ti_buy_sum']
ti_merged = merger._asof_merge_timeseries(ti_agg[['qlib_code', 'date'] + ti_cols],
                                           cache.index, 'date', ti_cols)
if ti_merged is not None:
    factors['top_inst'] = ti_merged
    logger.info(f"  top_inst: {ti_merged.shape[1]} cols, coverage={ti_merged.notna().any(axis=1).mean():.1%}")

# Rolling ablation
n_splits = 8
results = {fg: [] for fg in factors}

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

    m_base = train_xgb_fn(X_tr[mtr], y_tr[mtr], X_va[mva], y_va[mva])
    p_base = m_base.predict(xgb.DMatrix(X_te[mte]))
    e_base = evaluate_fn(p_base, y_te[mte], test_idx[mte])

    logger.info(f"\nSplit {split_idx+1}: base RankIC={e_base['ric']:+.4f}")

    for fg, fdf in factors.items():
        fg_tr = fdf.loc[tm].values.astype(np.float32)
        fg_va = fdf.loc[vm].values.astype(np.float32)
        fg_te = fdf.loc[em].values.astype(np.float32)

        m_plus = train_xgb_fn(np.hstack([X_tr, fg_tr])[mtr], y_tr[mtr],
                              np.hstack([X_va, fg_va])[mva], y_va[mva])
        p_plus = m_plus.predict(xgb.DMatrix(np.hstack([X_te, fg_te])[mte]))
        e_plus = evaluate_fn(p_plus, y_te[mte], test_idx[mte])

        delta = e_plus['ric'] - e_base['ric']
        results[fg].append(delta)
        logger.info(f"  +{fg}: RankIC={e_plus['ric']:+.4f} Δ={delta:+.4f}")

logger.info(f"\n{'='*50}")
logger.info("NEW FACTOR ABLATION")
logger.info(f"{'='*50}")
for fg, deltas in results.items():
    n = len(deltas)
    pos = sum(1 for d in deltas if d > 0)
    logger.info(f"  {fg}: avg Δ={np.mean(deltas):+.4f}, >0: {pos}/{n} ({pos/n:.0%}) "
                f"{'✅ PASS' if pos/n >= 0.7 else '❌ FAIL'}")

out_path = DATA_DIR / "phase4" / "new_factor_ablation.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(str(out_path), 'w') as f:
    json.dump({"results": {k: [round(float(v), 6) for v in vs] for k, vs in results.items()}},
              f, indent=2, default=json_default)
logger.info("Done!")
