"""Step 2: XGB on Alpha174 + Transformer embeddings (238 dims).

Only loads Alpha174 cache + embedding parquet (from Step 1).
Compares: xgb_174 baseline vs xgb_174+embed64.

Run AFTER step1 has produced transformer_embeddings.parquet.

Usage:
    python scripts/phase4m_step2_xgb_with_embedding.py
"""
import gc
import json
import logging
import os
import sys
import time
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
ALPHA174_PATH = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
EMBED_PATH = DATA_DIR / "transformer_embeddings.parquet"
OUTPUT_DIR = DATA_DIR / "phase4m"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 6
TEST_DAYS = 20
VALID_DAYS = 60
TRAIN_DAYS = 250
LABEL_COL = "__label_5d"

EXCLUDE_PREFIXES = ("__", "_", "hsi_", "hstech_", "nasdaq_")
EXCLUDE_EXACT = {"holder_num"}

XGB_PARAMS = {
    "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
    "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
    "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": 42,
}


def evaluate_split(pred, label, index):
    mask = np.isfinite(pred) & np.isfinite(label)
    if mask.sum() < 200:
        return None
    pred_s = pd.Series(pred[mask], index=index[mask])
    label_s = pd.Series(label[mask], index=index[mask])
    ric_list, spreads = [], []
    for dt, g in pred_s.groupby(level=0):
        gl = label_s.reindex(g.index).dropna()
        common = g.index.intersection(gl.index)
        if len(common) < 40:
            continue
        p, l = g.loc[common].values, gl.loc[common].values
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

    if not EMBED_PATH.exists():
        logger.error(f"Embedding file not found: {EMBED_PATH}")
        logger.error("Run step1 first: python scripts/phase4m_step1_train_transformer.py")
        return

    # Load 174 cache (only recent for memory)
    logger.info("Loading Alpha174 cache (recent only)...")
    cache = pd.read_parquet(ALPHA174_PATH)
    all_dates = cache.index.get_level_values(0)
    unique_dates = sorted(all_dates.unique())
    needed = N_SPLITS * TEST_DAYS + VALID_DAYS + TRAIN_DAYS + 20
    cutoff = unique_dates[max(0, len(unique_dates) - needed)]
    cache = cache.loc[all_dates >= cutoff].copy()
    gc.collect()

    feat_174 = [c for c in cache.columns
                if not any(c.startswith(p) for p in EXCLUDE_PREFIXES) and c not in EXCLUDE_EXACT]
    trade_dates = sorted(cache.index.get_level_values(0).unique())
    dates_level = cache.index.get_level_values(0)
    logger.info(f"  174 cache: {cache.shape}, {len(feat_174)} features")

    # Load embeddings
    logger.info("Loading Transformer embeddings...")
    embed_df = pd.read_parquet(EMBED_PATH)
    embed_cols = list(embed_df.columns)
    logger.info(f"  Embeddings: {embed_df.shape}")

    # Align: only keep rows in both
    common_idx = cache.index.intersection(embed_df.index)
    cache = cache.loc[common_idx]
    embed_df = embed_df.loc[common_idx]
    dates_level = cache.index.get_level_values(0)
    trade_dates = sorted(dates_level.unique())
    logger.info(f"  Aligned: {len(common_idx):,} rows")

    CONFIGS = [
        {"name": "xgb_174", "use_embed": False},
        {"name": "xgb_174+embed64", "use_embed": True},
    ]
    all_results = {c["name"]: [] for c in CONFIGS}

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

        # Base 174 features
        X_tr_174 = cache.loc[train_mask, feat_174].values.astype(np.float32)
        y_tr = cache.loc[train_mask, LABEL_COL].values.astype(np.float32)
        X_va_174 = cache.loc[valid_mask, feat_174].values.astype(np.float32)
        y_va = cache.loc[valid_mask, LABEL_COL].values.astype(np.float32)
        X_te_174 = cache.loc[test_mask, feat_174].values.astype(np.float32)
        y_te = cache.loc[test_mask, LABEL_COL].values.astype(np.float32)
        test_idx = cache.index[test_mask]

        m_tr = np.isfinite(y_tr); X_tr_174, y_tr = X_tr_174[m_tr], y_tr[m_tr]
        m_va = np.isfinite(y_va); X_va_174, y_va = X_va_174[m_va], y_va[m_va]
        m_te = np.isfinite(y_te); X_te_174, y_te = X_te_174[m_te], y_te[m_te]
        test_idx = test_idx[m_te]

        # Embeddings
        E_tr = embed_df.loc[train_mask, embed_cols].values.astype(np.float32)[m_tr]
        E_va = embed_df.loc[valid_mask, embed_cols].values.astype(np.float32)[m_va]
        E_te = embed_df.loc[test_mask, embed_cols].values.astype(np.float32)[m_te]

        for cfg in CONFIGS:
            name = cfg["name"]
            t1 = time.time()

            if cfg["use_embed"]:
                X_tr = np.concatenate([X_tr_174, np.nan_to_num(E_tr, nan=0.0)], axis=1)
                X_va = np.concatenate([X_va_174, np.nan_to_num(E_va, nan=0.0)], axis=1)
                X_te = np.concatenate([X_te_174, np.nan_to_num(E_te, nan=0.0)], axis=1)
            else:
                X_tr, X_va, X_te = X_tr_174, X_va_174, X_te_174

            dt = xgb.DMatrix(X_tr, label=y_tr)
            dv = xgb.DMatrix(X_va, label=y_va)
            model = xgb.train(XGB_PARAMS, dt, num_boost_round=400,
                              evals=[(dv, "valid")], early_stopping_rounds=50, verbose_eval=0)
            pred = model.predict(xgb.DMatrix(X_te))

            metrics = evaluate_split(pred, y_te, test_idx)
            if metrics:
                metrics["split"] = split_idx + 1
                metrics["n_features"] = X_tr.shape[1]
                all_results[name].append(metrics)
                logger.info(f"  {name:<22} RIC={metrics['rank_ic']:+.4f} "
                            f"Spr={metrics['spread_top20']*100:+.3f}% "
                            f"({X_tr.shape[1]} feats, {time.time()-t1:.1f}s)")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"TRANSFORMER EMBEDDING + XGB: {N_SPLITS} splits")
    logger.info(f"{'='*80}")
    logger.info(f"{'Model':<22} {'AvgRIC':>8} {'MedRIC':>8} {'RICIR':>7} {'Spread':>8} {'#Feat':>6} {'#Spl':>5}")
    logger.info("-" * 70)

    summary = {}
    for cfg in CONFIGS:
        name = cfg["name"]
        splits = all_results[name]
        if not splits:
            continue
        rics = [s["rank_ic"] for s in splits]
        spreads = [s.get("spread_top20", 0) for s in splits]
        ricir = float(np.mean(rics) / (np.std(rics) + 1e-8))
        n_feat = splits[0].get("n_features", 0)
        summary[name] = {
            "avg_rank_ic": round(float(np.mean(rics)), 6),
            "med_rank_ic": round(float(np.median(rics)), 6),
            "rank_icir": round(ricir, 4),
            "avg_spread": round(float(np.mean(spreads)), 6),
            "n_features": n_feat,
            "n_splits": len(splits),
            "per_split": splits,
        }
        logger.info(f"{name:<22} {np.mean(rics):>+8.4f} {np.median(rics):>+8.4f} {ricir:>+7.3f} "
                    f"{np.mean(spreads)*100:>+7.3f}% {n_feat:>6} {len(splits):>5}")

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    output = {"evaluated_at": datetime.now().isoformat(timespec="seconds") if 'datetime' in dir() else "",
              "n_splits": N_SPLITS, "summary": summary}
    out_path = OUTPUT_DIR / "transformer_embedding_xgb.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                   default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o)))
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
