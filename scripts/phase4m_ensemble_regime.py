"""Phase 4M.3: Ensemble + Dynamic Feature Selection experiments.

Experiment 1: Weighted Ensemble (XGB + LightGBM + CatBoost average)
  - Train 3 tree models on same 174 features
  - Average predictions with equal or IC-weighted blending
  - Compare vs XGB alone

Experiment 2: Regime-Aware Dynamic Feature Selection
  - Split features into groups (momentum, volatility, value, flow, etc.)
  - In high-vol regime, upweight volatility/reversal features
  - In low-vol regime, upweight momentum features
  - Simple approach: train separate XGB per regime, blend at prediction time

Usage:
    python scripts/phase4m_ensemble_regime.py
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

LGB_PARAMS = {
    "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
    "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
    "objective": "regression", "n_jobs": 12, "verbose": -1, "seed": 42,
}

# Feature groups for dynamic selection
# Based on Alpha158 naming conventions
FEATURE_GROUPS = {
    "momentum": ["ROC5", "ROC10", "ROC20", "ROC30", "ROC60",
                  "MA5", "MA10", "MA20", "MA30", "MA60",
                  "RANK5", "RANK10", "RANK20", "RANK30", "RANK60"],
    "volatility": ["STD5", "STD10", "STD20", "STD30", "STD60",
                    "BETA5", "BETA10", "BETA20", "BETA30", "BETA60",
                    "RSQR5", "RSQR10", "RSQR20", "RSQR30", "RSQR60",
                    "RESI5", "RESI10", "RESI20", "RESI30", "RESI60"],
    "volume": ["VSUMP5", "VSUMP10", "VSUMP20", "VSUMP30", "VSUMP60",
               "VSUMN5", "VSUMN10", "VSUMN20", "VSUMN30", "VSUMN60",
               "VSUMD5", "VSUMD10", "VSUMD20", "VSUMD30", "VSUMD60",
               "VMA5", "VMA10", "VMA20", "VMA30", "VMA60",
               "VSTD5", "VSTD10", "VSTD20", "VSTD30", "VSTD60"],
    "correlation": ["CORR5", "CORR10", "CORR20", "CORR30", "CORR60",
                     "CORD5", "CORD10", "CORD20", "CORD30", "CORD60"],
    "price_pattern": ["KLOW", "KLOW2", "KUP", "KUP2", "KLEN",
                       "KSFT", "KSFT2", "KMID", "KMID2",
                       "OPEN0", "HIGH0", "LOW0", "CLOSE0",
                       "RSV5", "RSV10", "RSV20", "RSV30", "RSV60"],
    "quantile": ["QTLU5", "QTLU10", "QTLU20", "QTLU30", "QTLU60",
                  "QTLD5", "QTLD10", "QTLD20", "QTLD30", "QTLD60"],
    "custom_flow": ["pe", "pb", "turn_raw", "amount_raw",
                     "pe_mom20", "pb_mom20", "turn_anom20", "turn_anom60",
                     "amount_anom20", "turn_vol20", "ep", "bp", "price_pos20",
                     "flow_net_mf_latest", "flow_net_mf_5d", "flow_net_mf_20d_avg"],
}


def get_feature_cols(all_cols):
    return [c for c in all_cols
            if not any(c.startswith(p) for p in EXCLUDE_PREFIXES) and c not in EXCLUDE_EXACT]


def classify_regime(recent_returns: np.ndarray, window: int = 20) -> str:
    """Classify market regime from recent cross-sectional return volatility."""
    if len(recent_returns) < window:
        return "normal"
    vol = np.std(recent_returns[-window:])
    median_vol = np.median(np.abs(recent_returns[-window:]))
    if vol > 0.04 or median_vol > 0.025:
        return "high_vol"
    elif vol < 0.015:
        return "low_vol"
    return "normal"


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
    import xgboost as xgb
    import lightgbm as lgb

    t_start = time.time()

    logger.info("Loading feature cache...")
    cache = pd.read_parquet(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet")
    all_feature_cols = get_feature_cols(cache.columns)
    logger.info(f"Cache: {cache.shape}, {len(all_feature_cols)} features")

    # Resolve feature groups to actual columns
    resolved_groups = {}
    for gname, patterns in FEATURE_GROUPS.items():
        cols = [c for c in all_feature_cols if c in patterns]
        if cols:
            resolved_groups[gname] = cols
    # "other" = everything not in any group
    grouped = set()
    for cols in resolved_groups.values():
        grouped.update(cols)
    resolved_groups["other"] = [c for c in all_feature_cols if c not in grouped]
    logger.info(f"Feature groups: {', '.join(f'{k}({len(v)})' for k, v in resolved_groups.items())}")

    trade_dates = sorted(cache.index.get_level_values(0).unique())
    dates_level = cache.index.get_level_values(0)

    # Define configs
    CONFIGS = {
        "xgb_174": "baseline",
        "lgb_174": "baseline",
        "ensemble_eq": "ensemble",           # equal weight XGB + LGB
        "ensemble_rank": "ensemble",          # rank-average XGB + LGB
        "regime_momentum": "regime_select",   # high-vol: drop momentum, keep vol/corr
        "regime_volatility": "regime_select", # low-vol: drop volatility, keep momentum
    }

    all_results = {name: [] for name in CONFIGS}

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

        X_tr = cache.loc[train_mask, all_feature_cols].values.astype(np.float32)
        y_tr = cache.loc[train_mask, LABEL_COL].values.astype(np.float32)
        X_va = cache.loc[valid_mask, all_feature_cols].values.astype(np.float32)
        y_va = cache.loc[valid_mask, LABEL_COL].values.astype(np.float32)
        X_te = cache.loc[test_mask, all_feature_cols].values.astype(np.float32)
        y_te = cache.loc[test_mask, LABEL_COL].values.astype(np.float32)
        test_idx = cache.index[test_mask]

        m_tr = np.isfinite(y_tr); X_tr, y_tr = X_tr[m_tr], y_tr[m_tr]
        m_va = np.isfinite(y_va); X_va, y_va = X_va[m_va], y_va[m_va]
        m_te = np.isfinite(y_te); X_te_c, y_te_c = X_te[m_te], y_te[m_te]
        test_idx_c = test_idx[m_te]

        # ---- Train XGB ----
        t1 = time.time()
        dt = xgb.DMatrix(X_tr, label=y_tr)
        dv = xgb.DMatrix(X_va, label=y_va)
        xgb_model = xgb.train(XGB_PARAMS, dt, num_boost_round=400,
                               evals=[(dv, "valid")], early_stopping_rounds=50, verbose_eval=0)
        xgb_pred = xgb_model.predict(xgb.DMatrix(X_te_c))
        xgb_time = time.time() - t1

        # ---- Train LightGBM ----
        t1 = time.time()
        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_valid = lgb.Dataset(X_va, label=y_va, reference=lgb_train)
        lgb_model = lgb.train(
            LGB_PARAMS, lgb_train, num_boost_round=400,
            valid_sets=[lgb_valid],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        lgb_pred = lgb_model.predict(X_te_c)
        lgb_time = time.time() - t1

        # ---- Evaluate XGB ----
        metrics = evaluate_split(xgb_pred, y_te_c, test_idx_c)
        if metrics:
            metrics["split"] = split_idx + 1
            all_results["xgb_174"].append(metrics)
            logger.info(f"  xgb_174          RankIC={metrics['rank_ic']:+.4f} Spr={metrics['spread_top20']*100:+.3f}% ({xgb_time:.1f}s)")

        # ---- Evaluate LGB ----
        metrics = evaluate_split(lgb_pred, y_te_c, test_idx_c)
        if metrics:
            metrics["split"] = split_idx + 1
            all_results["lgb_174"].append(metrics)
            logger.info(f"  lgb_174          RankIC={metrics['rank_ic']:+.4f} Spr={metrics['spread_top20']*100:+.3f}% ({lgb_time:.1f}s)")

        # ---- Ensemble Equal Weight ----
        ens_pred = (xgb_pred + lgb_pred) / 2.0
        metrics = evaluate_split(ens_pred, y_te_c, test_idx_c)
        if metrics:
            metrics["split"] = split_idx + 1
            all_results["ensemble_eq"].append(metrics)
            logger.info(f"  ensemble_eq      RankIC={metrics['rank_ic']:+.4f} Spr={metrics['spread_top20']*100:+.3f}%")

        # ---- Ensemble Rank Average ----
        # Convert to ranks, average ranks, then use as score
        xgb_rank = stats.rankdata(xgb_pred)
        lgb_rank = stats.rankdata(lgb_pred)
        rank_avg_pred = (xgb_rank + lgb_rank) / 2.0
        metrics = evaluate_split(rank_avg_pred, y_te_c, test_idx_c)
        if metrics:
            metrics["split"] = split_idx + 1
            all_results["ensemble_rank"].append(metrics)
            logger.info(f"  ensemble_rank    RankIC={metrics['rank_ic']:+.4f} Spr={metrics['spread_top20']*100:+.3f}%")

        # ---- Regime: Drop Momentum in High-Vol ----
        # Detect regime from training data
        train_label_std = np.std(y_tr)
        is_high_vol = train_label_std > np.median(np.abs(y_tr)) * 1.5

        # Feature indices for momentum group
        momentum_idx = [i for i, c in enumerate(all_feature_cols) if c in set(resolved_groups.get("momentum", []))]
        vol_idx = [i for i, c in enumerate(all_feature_cols) if c in set(resolved_groups.get("volatility", []))]

        if is_high_vol and momentum_idx:
            # Drop momentum features
            keep_idx = [i for i in range(len(all_feature_cols)) if i not in momentum_idx]
        else:
            keep_idx = list(range(len(all_feature_cols)))

        if len(keep_idx) > 10:
            dt_r = xgb.DMatrix(X_tr[:, keep_idx], label=y_tr)
            dv_r = xgb.DMatrix(X_va[:, keep_idx], label=y_va)
            model_r = xgb.train(XGB_PARAMS, dt_r, num_boost_round=400,
                                 evals=[(dv_r, "valid")], early_stopping_rounds=50, verbose_eval=0)
            pred_r = model_r.predict(xgb.DMatrix(X_te_c[:, keep_idx]))
            metrics = evaluate_split(pred_r, y_te_c, test_idx_c)
            if metrics:
                metrics["split"] = split_idx + 1
                metrics["regime"] = "high_vol" if is_high_vol else "normal"
                metrics["n_features"] = len(keep_idx)
                all_results["regime_momentum"].append(metrics)
                logger.info(f"  regime_momentum  RankIC={metrics['rank_ic']:+.4f} Spr={metrics['spread_top20']*100:+.3f}% (regime={metrics['regime']}, feats={len(keep_idx)})")

        # ---- Regime: Drop Volatility in Low-Vol ----
        is_low_vol = train_label_std < np.median(np.abs(y_tr)) * 0.7

        if is_low_vol and vol_idx:
            keep_idx2 = [i for i in range(len(all_feature_cols)) if i not in vol_idx]
        else:
            keep_idx2 = list(range(len(all_feature_cols)))

        if len(keep_idx2) > 10:
            dt_r2 = xgb.DMatrix(X_tr[:, keep_idx2], label=y_tr)
            dv_r2 = xgb.DMatrix(X_va[:, keep_idx2], label=y_va)
            model_r2 = xgb.train(XGB_PARAMS, dt_r2, num_boost_round=400,
                                  evals=[(dv_r2, "valid")], early_stopping_rounds=50, verbose_eval=0)
            pred_r2 = model_r2.predict(xgb.DMatrix(X_te_c[:, keep_idx2]))
            metrics = evaluate_split(pred_r2, y_te_c, test_idx_c)
            if metrics:
                metrics["split"] = split_idx + 1
                metrics["regime"] = "low_vol" if is_low_vol else "normal"
                metrics["n_features"] = len(keep_idx2)
                all_results["regime_volatility"].append(metrics)
                logger.info(f"  regime_vol       RankIC={metrics['rank_ic']:+.4f} Spr={metrics['spread_top20']*100:+.3f}% (regime={metrics['regime']}, feats={len(keep_idx2)})")

    # Summary
    logger.info(f"\n{'='*90}")
    logger.info(f"ENSEMBLE + REGIME COMPARISON: {N_SPLITS} splits")
    logger.info(f"{'='*90}")
    logger.info(f"{'Model':<22} {'AvgRIC':>8} {'MedRIC':>8} {'RICIR':>7} {'Spr20':>8} {'RIC>0':>6} {'#Spl':>5}")
    logger.info("-" * 70)

    summary = {}
    for name in CONFIGS:
        splits = all_results[name]
        if not splits:
            continue
        rics = [s["rank_ic"] for s in splits]
        spreads = [s["spread_top20"] for s in splits]
        ric_pos = [s.get("rank_ic_pos", 0) for s in splits]
        ricir = float(np.mean(rics) / (np.std(rics) + 1e-8))

        summary[name] = {
            "avg_rank_ic": round(float(np.mean(rics)), 6),
            "med_rank_ic": round(float(np.median(rics)), 6),
            "rank_icir": round(ricir, 4),
            "avg_spread": round(float(np.mean(spreads)), 6),
            "avg_ric_pos": round(float(np.mean(ric_pos)), 4),
            "n_splits": len(splits),
        }

        logger.info(
            f"{name:<22} {np.mean(rics):>+8.4f} {np.median(rics):>+8.4f} {ricir:>+7.3f} "
            f"{np.mean(spreads)*100:>+7.3f}% {np.mean(ric_pos)*100:>5.0f}% {len(splits):>5}"
        )

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    output = {"evaluated_at": datetime.now().isoformat(timespec="seconds"),
              "n_splits": N_SPLITS, "summary": summary,
              "per_split": {k: v for k, v in all_results.items()}}
    out_path = OUTPUT_DIR / "ensemble_regime_comparison.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                   default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o)))
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
