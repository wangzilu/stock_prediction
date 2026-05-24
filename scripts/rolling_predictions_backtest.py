"""Rolling predictions backtest — per-split model + event overlay test.

For each historical split:
  1. Train XGB on train window
  2. Predict on test window
  3. Apply gated LLM event overlay
  4. Compare: XGB alone vs XGB + overlay

This gives proper out-of-sample overlay evaluation (not same-model-all-dates).

Usage:
    python scripts/rolling_predictions_backtest.py
"""
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
OUTPUT_DIR = DATA_DIR / "phase4n"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 12
TEST_DAYS = 20
VALID_DAYS = 60
TRAIN_DAYS = 750

EXCLUDE_PREFIXES = ("__", "_", "hsi_", "hstech_", "nasdaq_")
EXCLUDE_EXACT = {"holder_num"}
LABEL_COL = "__label_5d"

XGB_PARAMS = {
    "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
    "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
    "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": 42,
}

# Gating config (CX approved)
NOISE_TYPES = {"other", "routine_announcement", "reorganize"}
UNSTABLE_TYPES = {"earnings_negative", "industry_trend_positive",
                  "product_launch", "analyst_upgrade"}
UNSTABLE_WEIGHT = 0.2


def get_feature_cols(all_cols):
    return [c for c in all_cols
            if not any(c.startswith(p) for p in EXCLUDE_PREFIXES) and c not in EXCLUDE_EXACT]


def load_gated_events(date: str) -> dict:
    """Load gated LLM events for a date."""
    path = DATA_DIR / "llm_events" / f"{date}.jsonl"
    if not path.exists():
        return {}
    impacts = defaultdict(list)
    for line in open(path):
        e = json.loads(line)
        etype = e.get("event_type", "other")
        impact = e.get("impact_1d", 0)
        code = e.get("stock_code", "")
        if not code or etype in NOISE_TYPES:
            continue
        if etype in UNSTABLE_TYPES:
            impact *= UNSTABLE_WEIGHT
        impacts[code].append(impact)
    return {c: np.mean(v) for c, v in impacts.items()}


def evaluate(pred, label, index):
    mask = np.isfinite(pred) & np.isfinite(label)
    if mask.sum() < 200:
        return None
    ps = pd.Series(pred[mask], index=index[mask])
    ls = pd.Series(label[mask], index=index[mask])
    rics = []
    for dt, g in ps.groupby(level=0):
        gl = ls.reindex(g.index).dropna()
        c = g.index.intersection(gl.index)
        if len(c) >= 40:
            r = stats.spearmanr(g.loc[c].values, gl.loc[c].values).statistic
            if np.isfinite(r):
                rics.append(r)
    if not rics:
        return None
    return {"rank_ic": float(np.mean(rics)), "n_days": len(rics)}


def main():
    t_start = time.time()

    logger.info("Loading feature cache...")
    cache = pd.read_parquet(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet")
    feature_cols = get_feature_cols(cache.columns)
    trade_dates = sorted(cache.index.get_level_values(0).unique())
    dates_level = cache.index.get_level_values(0)
    logger.info(f"Cache: {cache.shape}, {len(feature_cols)} features, {len(trade_dates)} dates")

    results_xgb = []
    results_overlay = []

    for split_idx in range(N_SPLITS):
        test_end_idx = len(trade_dates) - 1 - split_idx * TEST_DAYS
        test_start_idx = test_end_idx - TEST_DAYS
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - VALID_DAYS
        train_end_idx = valid_start_idx - 1
        train_start_idx = max(0, train_end_idx - TRAIN_DAYS)
        if train_start_idx >= train_end_idx:
            break

        test_start = trade_dates[test_start_idx]
        test_end = trade_dates[test_end_idx]
        logger.info(f"\nSplit {split_idx+1}/{N_SPLITS}: test {str(test_start)[:10]}~{str(test_end)[:10]}")

        # Data splits
        train_mask = (dates_level >= trade_dates[train_start_idx]) & (dates_level <= trade_dates[train_end_idx])
        valid_mask = (dates_level >= trade_dates[valid_start_idx]) & (dates_level <= trade_dates[valid_end_idx])
        test_mask = (dates_level >= test_start) & (dates_level <= test_end)

        X_tr = cache.loc[train_mask, feature_cols].values.astype(np.float32)
        y_tr = cache.loc[train_mask, LABEL_COL].values.astype(np.float32)
        X_va = cache.loc[valid_mask, feature_cols].values.astype(np.float32)
        y_va = cache.loc[valid_mask, LABEL_COL].values.astype(np.float32)
        X_te = cache.loc[test_mask, feature_cols].values.astype(np.float32)
        y_te = cache.loc[test_mask, LABEL_COL].values.astype(np.float32)
        test_idx = cache.index[test_mask]

        m_tr = np.isfinite(y_tr); X_tr, y_tr = X_tr[m_tr], y_tr[m_tr]
        m_va = np.isfinite(y_va); X_va, y_va = X_va[m_va], y_va[m_va]
        m_te = np.isfinite(y_te); X_te, y_te = X_te[m_te], y_te[m_te]
        test_idx = test_idx[m_te]

        # Train XGB
        t1 = time.time()
        dt = xgb.DMatrix(X_tr, label=y_tr)
        dv = xgb.DMatrix(X_va, label=y_va)
        model = xgb.train(XGB_PARAMS, dt, num_boost_round=400,
                          evals=[(dv, "valid")], early_stopping_rounds=50, verbose_eval=0)
        xgb_pred = model.predict(xgb.DMatrix(X_te))
        train_time = time.time() - t1

        # Evaluate XGB alone
        xgb_metrics = evaluate(xgb_pred, y_te, test_idx)
        if xgb_metrics:
            xgb_metrics["split"] = split_idx + 1
            results_xgb.append(xgb_metrics)

        # Apply gated overlay per test date
        overlay_pred = xgb_pred.copy()
        test_dates_unique = sorted(test_idx.get_level_values(0).unique())
        n_overlaid = 0

        for test_date in test_dates_unique:
            date_str = str(test_date)[:10]
            events = load_gated_events(date_str)
            if not events:
                continue

            # Get indices for this date
            date_mask = test_idx.get_level_values(0) == test_date
            date_indices = np.where(date_mask)[0]

            # Z-score event impacts
            evt_vals = [v for v in events.values() if v != 0]
            if not evt_vals:
                continue
            evt_mean, evt_std = np.mean(evt_vals), np.std(evt_vals) + 1e-8

            for i in date_indices:
                inst = str(test_idx[i][1])  # instrument
                code6 = inst[2:]  # sh600519 → 600519
                impact = events.get(code6, 0)
                if impact != 0:
                    evt_z = (impact - evt_mean) / evt_std
                    # Blend: overlay with alpha=1.0
                    overlay_pred[i] += evt_z * np.std(xgb_pred)
                    n_overlaid += 1

        overlay_metrics = evaluate(overlay_pred, y_te, test_idx)
        if overlay_metrics:
            overlay_metrics["split"] = split_idx + 1
            overlay_metrics["n_overlaid"] = n_overlaid
            results_overlay.append(overlay_metrics)

        xgb_ric = xgb_metrics["rank_ic"] if xgb_metrics else 0
        ovl_ric = overlay_metrics["rank_ic"] if overlay_metrics else 0
        delta = ovl_ric - xgb_ric
        logger.info(f"  XGB RIC={xgb_ric:+.4f} | Overlay RIC={ovl_ric:+.4f} | Δ={delta:+.4f} | "
                    f"overlaid={n_overlaid} ({train_time:.0f}s)")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"ROLLING OVERLAY BACKTEST: {N_SPLITS} splits")
    logger.info(f"{'='*70}")

    if results_xgb:
        xgb_rics = [r["rank_ic"] for r in results_xgb]
        logger.info(f"XGB alone:     AvgRIC={np.mean(xgb_rics):+.4f} MedRIC={np.median(xgb_rics):+.4f} "
                    f"RICIR={np.mean(xgb_rics)/(np.std(xgb_rics)+1e-8):+.3f}")

    if results_overlay:
        ovl_rics = [r["rank_ic"] for r in results_overlay]
        logger.info(f"XGB+overlay:   AvgRIC={np.mean(ovl_rics):+.4f} MedRIC={np.median(ovl_rics):+.4f} "
                    f"RICIR={np.mean(ovl_rics)/(np.std(ovl_rics)+1e-8):+.3f}")

    if results_xgb and results_overlay:
        deltas = [o["rank_ic"] - x["rank_ic"]
                  for o, x in zip(results_overlay, results_xgb)]
        logger.info(f"Delta:         Avg={np.mean(deltas):+.4f} Med={np.median(deltas):+.4f} "
                    f"positive={sum(d>0 for d in deltas)}/{len(deltas)}")

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save
    output = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "n_splits": N_SPLITS,
        "xgb_results": results_xgb,
        "overlay_results": results_overlay,
    }
    out_path = OUTPUT_DIR / "rolling_overlay_backtest.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False,
                                   default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o)))
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
