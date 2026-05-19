"""Ablation test for event kernel factors: forecast vs top_inst vs all.

12-split rolling ablation:
  - base (174 features)
  - base + forecast events (4 factors)
  - base + toplist events (4 factors)
  - base + all events (8 factors)
"""
import os, sys, json, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"

# ── Load data ──────────────────────────────────────────────────────
logger.info("Loading feature cache...")
cache = pd.read_parquet(str(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"))
base_cols = [c for c in cache.columns if not c.startswith('__') and not c.startswith('_')
             and not c.startswith('hsi_') and not c.startswith('hstech_')
             and not c.startswith('nasdaq_')]
label_col = '__label_5d'
logger.info(f"Base features: {len(base_cols)}")

logger.info("Loading event factors...")
event_path = DATA_DIR / "event_factors_v2.parquet"
if not event_path.exists():
    logger.error(f"Event factors not found at {event_path}. Run build_event_factors.py first.")
    sys.exit(1)

events = pd.read_parquet(str(event_path))
logger.info(f"Event factors: {events.shape}")

fc_cols = [c for c in events.columns if c.startswith('fc_')]
ti_cols = [c for c in events.columns if c.startswith('ti_')]
all_event_cols = fc_cols + ti_cols
logger.info(f"  Forecast cols ({len(fc_cols)}): {fc_cols}")
logger.info(f"  Top_inst cols ({len(ti_cols)}): {ti_cols}")

# Verify index alignment
assert cache.index.equals(events.index), "Index mismatch between cache and events"

# ── Setup ──────────────────────────────────────────────────────────
trade_dates = sorted(cache.index.get_level_values(0).unique())
today_idx = len(trade_dates) - 1
dl = cache.index.get_level_values(0)
SEED = 42
N_SPLITS = 12

def train_xgb_fn(X_tr, y_tr, X_va, y_va):
    dt = xgb.DMatrix(X_tr, label=y_tr)
    dv = xgb.DMatrix(X_va, label=y_va)
    params = {
        "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
        "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
        "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": SEED,
    }
    return xgb.train(params, dt, num_boost_round=400,
                     evals=[(dv, "valid")], early_stopping_rounds=30, verbose_eval=0)

def evaluate_fn(pred, label, index):
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
        "ric": round(float(np.nanmean(rics)), 6) if rics else 0,
        "spr": round(float(np.mean(sprs)), 6) if sprs else 0,
    }

# ── Factor groups for ablation ─────────────────────────────────────
factor_groups = {
    'forecast_events': fc_cols,
    'toplist_events': ti_cols,
    'all_events': all_event_cols,
}

# ── Rolling ablation ──────────────────────────────────────────────
results = {fg: [] for fg in factor_groups}
base_results = []

logger.info(f"\n{'='*60}")
logger.info(f"EVENT FACTOR ABLATION ({N_SPLITS} splits)")
logger.info(f"{'='*60}")

for split_idx in range(N_SPLITS):
    test_end_idx = today_idx - split_idx * 20
    test_start_idx = test_end_idx - 20
    valid_end_idx = test_start_idx - 1
    valid_start_idx = valid_end_idx - 60
    train_end_idx = valid_start_idx - 1
    train_start_idx = train_end_idx - 750
    if train_start_idx < 0:
        logger.info(f"Split {split_idx + 1}: not enough data, stopping.")
        break

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
    base_results.append(e_base)

    logger.info(f"\nSplit {split_idx + 1}/{N_SPLITS}: "
                f"test=[{trade_dates[test_start_idx].date()}..{trade_dates[test_end_idx].date()}] "
                f"base RankIC={e_base['ric']:+.4f} SPR={e_base['spr']:+.4f}")

    for fg, fg_cols in factor_groups.items():
        # Get event factor columns for this split
        ev_tr = events.loc[tm, fg_cols].values.astype(np.float32)
        ev_va = events.loc[vm, fg_cols].values.astype(np.float32)
        ev_te = events.loc[em, fg_cols].values.astype(np.float32)

        m_plus = train_xgb_fn(
            np.hstack([X_tr, ev_tr])[mtr], y_tr[mtr],
            np.hstack([X_va, ev_va])[mva], y_va[mva],
        )
        p_plus = m_plus.predict(xgb.DMatrix(np.hstack([X_te, ev_te])[mte]))
        e_plus = evaluate_fn(p_plus, y_te[mte], test_idx[mte])

        delta_ric = e_plus['ric'] - e_base['ric']
        delta_spr = e_plus['spr'] - e_base['spr']
        results[fg].append({
            'delta_ric': delta_ric,
            'delta_spr': delta_spr,
            'ric': e_plus['ric'],
            'spr': e_plus['spr'],
        })
        logger.info(f"  +{fg}: RankIC={e_plus['ric']:+.4f} (d={delta_ric:+.4f}) "
                    f"SPR={e_plus['spr']:+.4f} (d={delta_spr:+.4f})")

# ── Summary ────────────────────────────────────────────────────────
logger.info(f"\n{'='*60}")
logger.info("EVENT FACTOR ABLATION RESULTS")
logger.info(f"{'='*60}")
logger.info(f"Base avg RankIC: {np.mean([r['ric'] for r in base_results]):+.4f}")
logger.info(f"Base avg SPR:    {np.mean([r['spr'] for r in base_results]):+.4f}")
logger.info("")

summary = {}
for fg, splits in results.items():
    n = len(splits)
    drics = [s['delta_ric'] for s in splits]
    dsprs = [s['delta_spr'] for s in splits]
    pos_ric = sum(1 for d in drics if d > 0)
    pos_spr = sum(1 for d in dsprs if d > 0)
    avg_dric = np.mean(drics)
    avg_dspr = np.mean(dsprs)
    pass_ric = pos_ric / n >= 0.6
    pass_spr = pos_spr / n >= 0.6
    verdict = "PASS" if (pass_ric or pass_spr) else "FAIL"

    summary[fg] = {
        'avg_delta_ric': round(avg_dric, 6),
        'avg_delta_spr': round(avg_dspr, 6),
        'ric_positive': f"{pos_ric}/{n}",
        'spr_positive': f"{pos_spr}/{n}",
        'verdict': verdict,
    }

    logger.info(f"  {fg}:")
    logger.info(f"    avg dRankIC = {avg_dric:+.5f}  (>0: {pos_ric}/{n} = {pos_ric/n:.0%})"
                f"  {'OK' if pass_ric else '--'}")
    logger.info(f"    avg dSPR    = {avg_dspr:+.5f}  (>0: {pos_spr}/{n} = {pos_spr/n:.0%})"
                f"  {'OK' if pass_spr else '--'}")
    logger.info(f"    verdict: {verdict}")

# ── Save ───────────────────────────────────────────────────────────
out_dir = DATA_DIR / "phase4"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "event_factor_ablation_v2.json"

from utils.json_utils import json_default
with open(str(out_path), 'w') as f:
    json.dump({
        'summary': summary,
        'base_results': base_results,
        'detail': {k: v for k, v in results.items()},
    }, f, indent=2, default=json_default)

logger.info(f"\nResults saved to {out_path}")
logger.info("Done!")
