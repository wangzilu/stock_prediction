"""Phase 4M: Deep model comparison — ALSTM, Transformer vs XGB on 174 features.

Tests whether deep models can beat XGB's RankIC +0.051 on the same features.
Uses 12 rolling splits for speed (24 for final gate).

Usage:
    python scripts/phase4m_deep_model_gate.py
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
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
OUTPUT_DIR = DATA_DIR / "phase4m"
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


def get_feature_cols(all_cols):
    return [c for c in all_cols
            if not any(c.startswith(p) for p in EXCLUDE_PREFIXES) and c not in EXCLUDE_EXACT]


def train_xgb(X_tr, y_tr, X_va, y_va):
    import xgboost as xgb
    dt = xgb.DMatrix(X_tr, label=y_tr)
    dv = xgb.DMatrix(X_va, label=y_va)
    model = xgb.train(XGB_PARAMS, dt, num_boost_round=400,
                      evals=[(dv, "valid")], early_stopping_rounds=50, verbose_eval=0)
    return model.predict(xgb.DMatrix(X_va)), model


def train_deep(net_class, X_tr, y_tr, X_va, y_va, net_kwargs=None):
    """Train a deep model and return test predictions."""
    from models.deep_models import DeepModel

    model = DeepModel(
        net_class=net_class,
        net_kwargs=net_kwargs or {},
        lr=1e-3,
        batch_size=4096,
        n_epochs=30,
        early_stop=5,
    )
    model.fit(X_tr, y_tr, X_va, y_va)
    return model


def evaluate_split(pred, label, index):
    """Compute RankIC and spread for one split."""
    mask = np.isfinite(pred) & np.isfinite(label)
    if mask.sum() < 200:
        return None

    pred_s = pd.Series(pred[mask], index=index[mask])
    label_s = pd.Series(label[mask], index=index[mask])

    ric_list = []
    spreads = []

    for dt, g_pred in pred_s.groupby(level=0):
        g_label = label_s.reindex(g_pred.index).dropna()
        common = g_pred.index.intersection(g_label.index)
        if len(common) < 40:
            continue
        p = g_pred.loc[common].values
        l = g_label.loc[common].values
        ric = stats.spearmanr(p, l).statistic
        if np.isfinite(ric):
            ric_list.append(ric)
        tmp = pd.DataFrame({"p": p, "l": l}).sort_values("p", ascending=False)
        spreads.append(tmp.head(20)["l"].mean() - tmp.tail(20)["l"].mean())

    if not ric_list:
        return None

    return {
        "rank_ic": float(np.mean(ric_list)),
        "rank_ic_pos": float(np.mean([r > 0 for r in ric_list])),
        "spread_top20": float(np.mean(spreads)) if spreads else 0,
    }


def main():
    t_start = time.time()

    logger.info("Loading feature cache...")
    cache = pd.read_parquet(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet")
    feature_cols = get_feature_cols(cache.columns)
    d_feat = len(feature_cols)
    logger.info(f"Cache: {cache.shape}, {d_feat} features")

    trade_dates = sorted(cache.index.get_level_values(0).unique())
    dates_level = cache.index.get_level_values(0)

    # Models to test
    from models.deep_models import ALSTMNet, TransformerNet

    MODEL_CONFIGS = [
        {"name": "xgb_174", "type": "xgb"},
        {"name": "alstm_174", "type": "deep", "net_class": ALSTMNet,
         "net_kwargs": {"d_feat": d_feat, "hidden_size": 128, "num_layers": 2}},
        {"name": "transformer_174", "type": "deep", "net_class": TransformerNet,
         "net_kwargs": {"d_feat": d_feat, "d_model": 128, "nhead": 4, "num_layers": 2}},
    ]

    all_results = {cfg["name"]: [] for cfg in MODEL_CONFIGS}

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

        # Filter NaN labels
        m_tr = np.isfinite(y_tr); X_tr, y_tr = X_tr[m_tr], y_tr[m_tr]
        m_va = np.isfinite(y_va); X_va, y_va = X_va[m_va], y_va[m_va]
        m_te = np.isfinite(y_te); X_te_clean, y_te_clean = X_te[m_te], y_te[m_te]
        test_idx_clean = test_idx[m_te]

        # Replace NaN features with 0 for deep models
        X_tr_nn = np.nan_to_num(X_tr, nan=0.0)
        X_va_nn = np.nan_to_num(X_va, nan=0.0)
        X_te_nn = np.nan_to_num(X_te_clean, nan=0.0)

        for cfg in MODEL_CONFIGS:
            name = cfg["name"]
            t1 = time.time()

            try:
                if cfg["type"] == "xgb":
                    import xgboost as xgb
                    dt = xgb.DMatrix(X_tr, label=y_tr)
                    dv = xgb.DMatrix(X_va, label=y_va)
                    model = xgb.train(XGB_PARAMS, dt, num_boost_round=400,
                                      evals=[(dv, "valid")], early_stopping_rounds=50,
                                      verbose_eval=0)
                    pred = model.predict(xgb.DMatrix(X_te_clean))

                elif cfg["type"] == "deep":
                    dm = train_deep(
                        cfg["net_class"], X_tr_nn, y_tr, X_va_nn, y_va,
                        net_kwargs=cfg.get("net_kwargs"),
                    )
                    pred = dm.predict(X_te_nn)

                elapsed = time.time() - t1
                metrics = evaluate_split(pred, y_te_clean, test_idx_clean)

                if metrics:
                    metrics["split"] = split_idx + 1
                    metrics["train_time"] = round(elapsed, 1)
                    all_results[name].append(metrics)
                    logger.info(
                        f"  {name:<20} RankIC={metrics['rank_ic']:+.4f} "
                        f"Spread={metrics['spread_top20']*100:+.3f}% "
                        f"({elapsed:.1f}s)"
                    )
                else:
                    logger.warning(f"  {name}: insufficient data for evaluation")

            except Exception as e:
                logger.error(f"  {name}: FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    logger.info(f"\n{'='*90}")
    logger.info(f"PHASE 4M DEEP MODEL COMPARISON: {N_SPLITS} splits")
    logger.info(f"{'='*90}")
    logger.info(
        f"{'Model':<22} {'AvgRIC':>8} {'MedRIC':>8} {'RICIR':>7} "
        f"{'Spr20':>8} {'RIC>0':>6} {'AvgTime':>8} {'#Spl':>5}"
    )
    logger.info("-" * 90)

    summary = {}
    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        splits = all_results[name]
        if not splits:
            continue
        rics = [s["rank_ic"] for s in splits]
        spreads = [s["spread_top20"] for s in splits]
        ric_pos = [s.get("rank_ic_pos", 0) for s in splits]
        times = [s.get("train_time", 0) for s in splits]
        ricir = float(np.mean(rics) / (np.std(rics) + 1e-8))

        summary[name] = {
            "avg_rank_ic": round(float(np.mean(rics)), 6),
            "med_rank_ic": round(float(np.median(rics)), 6),
            "rank_icir": round(ricir, 4),
            "avg_spread_top20": round(float(np.mean(spreads)), 6),
            "avg_ric_pos": round(float(np.mean(ric_pos)), 4),
            "avg_train_time": round(float(np.mean(times)), 1),
            "n_splits": len(splits),
            "per_split": splits,
        }

        logger.info(
            f"{name:<22} {np.mean(rics):>+8.4f} {np.median(rics):>+8.4f} {ricir:>+7.3f} "
            f"{np.mean(spreads)*100:>+7.3f}% {np.mean(ric_pos)*100:>5.0f}% "
            f"{np.mean(times):>7.1f}s {len(splits):>5}"
        )

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    output = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "n_splits": N_SPLITS,
        "summary": summary,
    }
    out_path = OUTPUT_DIR / "deep_model_comparison.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                   default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o)))
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
