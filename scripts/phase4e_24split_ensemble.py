"""Phase 4E: 24-split rolling ensemble — train XGB, LGB, CatBoost across all
24 rolling splits and produce per-split OOF predictions for ensemble fusion.

Resumable: saves per-split predictions to disk so interrupted runs continue
from the last completed split.

Output:
    data/storage/phase4e_24split/{split_id}_{model}.pkl   — per-split preds
    data/storage/experiments/{exp_id}/pred.pkl             — full OOF preds
    data/storage/phase4e_24split/summary.json              — metrics

Usage: python scripts/phase4e_24split_ensemble.py
"""
import json
import os
import pickle
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATA_DIR = PROJECT_ROOT / "data" / "storage"
FEATURE_CACHE = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
CHECKPOINT_DIR = DATA_DIR / "phase4e_24split"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "__label_5d"


# ── Metrics ─────────────────────────────────────────────────────────────

def compute_metrics(pred: pd.Series, label: pd.Series) -> dict:
    """Compute IC, RankIC, ICIR, spread metrics per-day then aggregate."""
    common = pred.index.intersection(label.index)
    pred = pred.loc[common]
    label = label.loc[common]

    mask = pred.notna() & label.notna()
    pred = pred[mask]
    label = label[mask]

    if len(pred) < 100:
        return {"error": "too few samples"}

    ics, rank_ics = [], []
    spreads_top20, spreads_top50, spreads_top100 = [], [], []

    dates = pred.index.get_level_values("datetime").unique()
    for dt in dates:
        p = pred.xs(dt, level="datetime")
        l = label.xs(dt, level="datetime")
        common_inst = p.index.intersection(l.index)
        p = p.loc[common_inst]
        l = l.loc[common_inst]

        finite = np.isfinite(p.values) & np.isfinite(l.values)
        p_f, l_f = p[finite], l[finite]
        if len(p_f) < 30:
            continue

        ic = pearsonr(p_f.values, l_f.values)[0]
        ric = spearmanr(p_f.values, l_f.values).statistic
        if np.isfinite(ic):
            ics.append(ic)
        if np.isfinite(ric):
            rank_ics.append(ric)

        n = len(p_f)
        ranked = p_f.sort_values(ascending=False)
        for topk, store in [(20, spreads_top20), (50, spreads_top50), (100, spreads_top100)]:
            if n >= topk * 2:
                top = l_f.loc[ranked.index[:topk]].mean()
                bot = l_f.loc[ranked.index[-topk:]].mean()
                store.append(top - bot)

    if not rank_ics:
        return {"error": "no valid days"}

    metrics = {
        "ic_mean": round(float(np.mean(ics)), 6),
        "ic_std": round(float(np.std(ics)), 6),
        "icir": round(float(np.mean(ics) / (np.std(ics) + 1e-9)), 4),
        "rank_ic_mean": round(float(np.mean(rank_ics)), 6),
        "rank_ic_std": round(float(np.std(rank_ics)), 6),
        "rank_icir": round(float(np.mean(rank_ics) / (np.std(rank_ics) + 1e-9)), 4),
        "rank_ic_pos_ratio": round(sum(1 for r in rank_ics if r > 0) / len(rank_ics), 4),
        "n_days": len(rank_ics),
        "n_predictions": len(pred),
    }
    if spreads_top20:
        metrics["spread_top20"] = round(float(np.mean(spreads_top20)) * 10000, 2)
    if spreads_top50:
        metrics["spread_top50"] = round(float(np.mean(spreads_top50)) * 10000, 2)
    if spreads_top100:
        metrics["spread_top100"] = round(float(np.mean(spreads_top100)) * 10000, 2)
    return metrics


# ── Model Trainers ──────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_valid, y_valid, feat_cols):
    """Train XGBoost regressor with early stopping."""
    import xgboost as xgb

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.88,
        "colsample_bytree": 0.88,
        "reg_alpha": 205.7,
        "reg_lambda": 580.98,
        "n_estimators": 500,
        "n_jobs": 4,
        "verbosity": 0,
        "tree_method": "hist",
    }

    X_tr = np.nan_to_num(X_train.replace([np.inf, -np.inf], np.nan).values, nan=0.0)
    X_va = np.nan_to_num(X_valid.replace([np.inf, -np.inf], np.nan).values, nan=0.0)

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_tr, y_train.values,
        eval_set=[(X_va, y_valid.values)],
        verbose=False,
    )
    return model, params


def train_lightgbm(X_train, y_train, X_valid, y_valid, feat_cols):
    """Train LightGBM regressor with early stopping."""
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "mse",
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 210,
        "subsample": 0.8789,
        "colsample_bytree": 0.8879,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "n_estimators": 500,
        "n_jobs": 4,
        "verbose": -1,
    }

    X_tr = X_train.replace([np.inf, -np.inf], np.nan).values
    X_va = X_valid.replace([np.inf, -np.inf], np.nan).values

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_train.values,
        eval_set=[(X_va, y_valid.values)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return model, params


def train_catboost(X_train, y_train, X_valid, y_valid, feat_cols):
    """Train CatBoost regressor with early stopping. Handles NaN natively."""
    from catboost import CatBoostRegressor

    params = {
        "iterations": 500,
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 580.0,
        "subsample": 0.88,
        "rsm": 0.88,
        "loss_function": "RMSE",
        "verbose": 0,
        "thread_count": 4,
        "allow_writing_files": False,
        "early_stopping_rounds": 30,
    }

    X_tr = X_train.replace([np.inf, -np.inf], np.nan)
    X_va = X_valid.replace([np.inf, -np.inf], np.nan)

    model = CatBoostRegressor(**params)
    model.fit(X_tr, y_train, eval_set=(X_va, y_valid))
    return model, params


def predict_model(model, X_test, model_name, feat_cols):
    """Generate predictions from a trained model."""
    if model_name == "xgb":
        X = np.nan_to_num(X_test.replace([np.inf, -np.inf], np.nan).values, nan=0.0)
    elif model_name == "catboost":
        X = X_test.replace([np.inf, -np.inf], np.nan).values
    else:  # lgb
        X = X_test.replace([np.inf, -np.inf], np.nan).values
    return model.predict(X)


# ── Checkpoint helpers ──────────────────────────────────────────────────

def checkpoint_path(split_id: int, model_name: str) -> Path:
    return CHECKPOINT_DIR / f"split{split_id:02d}_{model_name}.pkl"


def save_checkpoint(split_id: int, model_name: str, pred: pd.Series):
    path = checkpoint_path(split_id, model_name)
    with open(path, "wb") as f:
        pickle.dump(pred, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(split_id: int, model_name: str) -> pd.Series | None:
    path = checkpoint_path(split_id, model_name)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def is_split_complete(split_id: int, model_names: list[str]) -> bool:
    return all(checkpoint_path(split_id, m).exists() for m in model_names)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    from config.rolling_splits import get_standard_splits
    from models.ensemble_fusion import fuse_rank_mean, fuse_robust_z_mean
    from tracker.artifact_contract import ExperimentArtifact

    MODEL_SPECS = {
        "xgb": {"fn": train_xgboost, "desc": "XGBoost regression"},
        "lgb": {"fn": train_lightgbm, "desc": "LightGBM regression"},
        "catboost": {"fn": train_catboost, "desc": "CatBoost regression"},
    }
    MODEL_NAMES = list(MODEL_SPECS.keys())

    print("=" * 70)
    print("Phase 4E: 24-Split Rolling Ensemble Training")
    print(f"Models: {MODEL_NAMES}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print("=" * 70)

    # ── 1. Load feature cache ───────────────────────────────────────────
    print(f"\n[1/5] Loading feature cache: {FEATURE_CACHE}")
    df = pd.read_parquet(FEATURE_CACHE)
    print(f"  Shape: {df.shape}")
    dates_all = df.index.get_level_values("datetime")
    print(f"  Date range: {dates_all.min()} ~ {dates_all.max()}")

    feat_cols = [c for c in df.columns if not c.startswith("__")]
    label_cols = [c for c in df.columns if c.startswith("__label")]
    label_col = LABEL_COL if LABEL_COL in df.columns else label_cols[0]
    print(f"  Features: {len(feat_cols)}, Label: {label_col}")

    # ── 2. Get 24-split config ──────────────────────────────────────────
    print(f"\n[2/5] Generating 24-split configuration...")
    splits = get_standard_splits("24split")
    print(f"  {len(splits)} splits generated")
    print(f"  First: split_id={splits[0]['split_id']} "
          f"train={splits[0]['train_start']}~{splits[0]['train_end']} "
          f"test={splits[0]['test_start']}~{splits[0]['test_end']}")
    print(f"  Last:  split_id={splits[-1]['split_id']} "
          f"train={splits[-1]['train_start']}~{splits[-1]['train_end']} "
          f"test={splits[-1]['test_start']}~{splits[-1]['test_end']}")

    # Check resumability
    completed = sum(1 for s in splits if is_split_complete(s["split_id"], MODEL_NAMES))
    print(f"  Already completed: {completed}/{len(splits)} splits")

    # ── 3. Train all splits ─────────────────────────────────────────────
    print(f"\n[3/5] Training across {len(splits)} splits...")
    total_t0 = time.time()

    split_metrics = {}  # {split_id: {model: metrics}}

    for si, split in enumerate(splits):
        sid = split["split_id"]
        print(f"\n{'─' * 60}")
        print(f"Split {sid} ({si+1}/{len(splits)}): "
              f"train={split['train_start']}~{split['train_end']} | "
              f"valid={split['valid_start']}~{split['valid_end']} | "
              f"test={split['test_start']}~{split['test_end']}")

        # Check if already done
        if is_split_complete(sid, MODEL_NAMES):
            print(f"  [SKIP] All models already checkpointed")
            # Load cached metrics
            split_metrics[sid] = {}
            for mname in MODEL_NAMES:
                pred = load_checkpoint(sid, mname)
                if pred is not None:
                    dates_idx = df.index.get_level_values("datetime")
                    test_mask = (dates_idx >= split["test_start"]) & (dates_idx <= split["test_end"])
                    test_labels = df.loc[test_mask, label_col].dropna()
                    m = compute_metrics(pred, test_labels)
                    split_metrics[sid][mname] = m
                    print(f"    {mname}: RankIC={m.get('rank_ic_mean', 'N/A')}")
            continue

        # Split data
        dates_idx = df.index.get_level_values("datetime")

        train_mask = (dates_idx >= split["train_start"]) & (dates_idx <= split["train_end"])
        valid_mask = (dates_idx >= split["valid_start"]) & (dates_idx <= split["valid_end"])
        test_mask = (dates_idx >= split["test_start"]) & (dates_idx <= split["test_end"])

        train_df = df.loc[train_mask].dropna(subset=[label_col])
        valid_df = df.loc[valid_mask].dropna(subset=[label_col])
        test_df = df.loc[test_mask].dropna(subset=[label_col])

        print(f"  Rows — train: {len(train_df):,}, valid: {len(valid_df):,}, test: {len(test_df):,}")

        if len(train_df) < 1000 or len(test_df) < 100:
            print(f"  [SKIP] Insufficient data")
            continue

        X_train = train_df[feat_cols]
        y_train = train_df[label_col]
        X_valid = valid_df[feat_cols]
        y_valid = valid_df[label_col]
        X_test = test_df[feat_cols]
        y_test = test_df[label_col]

        split_metrics[sid] = {}

        for mname, spec in MODEL_SPECS.items():
            # Check per-model checkpoint
            cached = load_checkpoint(sid, mname)
            if cached is not None:
                print(f"  {mname}: [CACHED]")
                m = compute_metrics(cached, y_test)
                split_metrics[sid][mname] = m
                print(f"    RankIC={m.get('rank_ic_mean', 'N/A')}")
                continue

            t0 = time.time()
            try:
                model, params = spec["fn"](X_train, y_train, X_valid, y_valid, feat_cols)
                pred_vals = predict_model(model, X_test, mname, feat_cols)
                pred = pd.Series(pred_vals, index=X_test.index, name="score")

                # Save checkpoint
                save_checkpoint(sid, mname, pred)

                # Compute metrics
                m = compute_metrics(pred, y_test)
                split_metrics[sid][mname] = m
                elapsed = time.time() - t0
                print(f"  {mname}: RankIC={m.get('rank_ic_mean', 'N/A')}, "
                      f"ICIR={m.get('rank_icir', 'N/A')}, "
                      f"time={elapsed:.1f}s")

                # Free memory
                del model
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  {mname}: FAILED after {elapsed:.1f}s — {e}")
                traceback.print_exc()
                split_metrics[sid][mname] = {"error": str(e)}

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 60}")
    print(f"Training complete: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    # ── 4. Concatenate OOF predictions & evaluate ───────────────────────
    print(f"\n[4/5] Concatenating OOF predictions & computing ensemble...")

    oof_preds = {m: [] for m in MODEL_NAMES}
    oof_labels = []

    for split in splits:
        sid = split["split_id"]
        dates_idx = df.index.get_level_values("datetime")
        test_mask = (dates_idx >= split["test_start"]) & (dates_idx <= split["test_end"])
        test_labels = df.loc[test_mask, label_col].dropna()
        oof_labels.append(test_labels)

        for mname in MODEL_NAMES:
            pred = load_checkpoint(sid, mname)
            if pred is not None:
                oof_preds[mname].append(pred)

    # Concatenate (test windows are non-overlapping, so simple concat works)
    oof_series = {}
    for mname in MODEL_NAMES:
        if oof_preds[mname]:
            oof_series[mname] = pd.concat(oof_preds[mname])
            print(f"  {mname}: {len(oof_series[mname]):,} OOF predictions")
        else:
            print(f"  {mname}: NO predictions")

    full_labels = pd.concat(oof_labels)
    # Remove duplicate index entries (overlapping test windows, if any)
    full_labels = full_labels[~full_labels.index.duplicated(keep="first")]
    for mname in oof_series:
        oof_series[mname] = oof_series[mname][~oof_series[mname].index.duplicated(keep="first")]

    # Per-model full-OOF metrics
    print(f"\n  Full OOF metrics:")
    full_metrics = {}
    for mname in MODEL_NAMES:
        if mname in oof_series:
            m = compute_metrics(oof_series[mname], full_labels)
            full_metrics[mname] = m
            print(f"    {mname}: RankIC={m.get('rank_ic_mean', 'N/A')}, "
                  f"ICIR={m.get('rank_icir', 'N/A')}, "
                  f"Spread20={m.get('spread_top20', 'N/A')}bps")

    # Ensemble fusion on OOF predictions
    print(f"\n  Ensemble fusions:")
    ensemble_methods = {}
    if len(oof_series) >= 2:
        for fusion_name, fusion_fn in [("rank_mean", fuse_rank_mean),
                                        ("robust_z_mean", fuse_robust_z_mean)]:
            fused = fusion_fn(oof_series, min_model_count=2)
            fused_clean = fused.dropna()
            m = compute_metrics(fused_clean, full_labels)
            ensemble_methods[fusion_name] = {"pred": fused_clean, "metrics": m}
            full_metrics[f"ensemble_{fusion_name}"] = m
            print(f"    {fusion_name}: RankIC={m.get('rank_ic_mean', 'N/A')}, "
                  f"ICIR={m.get('rank_icir', 'N/A')}, "
                  f"Spread20={m.get('spread_top20', 'N/A')}bps")
    else:
        print("    [SKIP] Need >= 2 models for ensemble")

    # Per-split ensemble vs best single comparison
    print(f"\n  Per-split ensemble wins:")
    ensemble_wins = {fn: 0 for fn in ["rank_mean", "robust_z_mean"]}
    n_comparable = 0

    for split in splits:
        sid = split["split_id"]
        if sid not in split_metrics:
            continue
        sm = split_metrics[sid]
        single_rics = [sm[m].get("rank_ic_mean", -999)
                       for m in MODEL_NAMES if m in sm and "error" not in sm[m]]
        if not single_rics:
            continue
        best_single = max(single_rics)

        # Compute ensemble metrics for this split
        split_preds = {}
        for mname in MODEL_NAMES:
            p = load_checkpoint(sid, mname)
            if p is not None:
                split_preds[mname] = p

        if len(split_preds) < 2:
            continue

        dates_idx = df.index.get_level_values("datetime")
        test_mask = (dates_idx >= split["test_start"]) & (dates_idx <= split["test_end"])
        test_labels = df.loc[test_mask, label_col].dropna()

        n_comparable += 1
        for fn_name, fn in [("rank_mean", fuse_rank_mean), ("robust_z_mean", fuse_robust_z_mean)]:
            fused = fn(split_preds, min_model_count=2).dropna()
            em = compute_metrics(fused, test_labels)
            ens_ric = em.get("rank_ic_mean", -999)
            if ens_ric > best_single:
                ensemble_wins[fn_name] += 1

    for fn_name, wins in ensemble_wins.items():
        print(f"    {fn_name}: {wins}/{n_comparable} splits "
              f"({wins/n_comparable*100:.0f}%)" if n_comparable > 0 else "")

    # ── 5. Save results ─────────────────────────────────────────────────
    print(f"\n[5/5] Saving results...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save per-model OOF pred as ExperimentArtifact
    for mname in MODEL_NAMES:
        if mname not in oof_series:
            continue
        exp_id = f"{mname}_174_24split_{timestamp}"
        art = ExperimentArtifact.create(
            experiment_id=exp_id,
            model_name=f"{mname}_174",
            feature_set="FS-174",
            description=f"{mname} 24-split rolling OOF predictions",
            n_features=len(feat_cols),
            n_splits=len(splits),
        )
        pred_s = oof_series[mname]
        aligned_labels = full_labels.reindex(pred_s.index).dropna()
        pred_s = pred_s.reindex(aligned_labels.index)
        art.save_predictions(
            pred_s.to_frame("score"),
            aligned_labels.to_frame("label"),
        )
        art.save_metrics(full_metrics.get(mname, {}))
        print(f"  Saved artifact: {exp_id}")

    # Save ensemble OOF pred
    for fn_name, ens_data in ensemble_methods.items():
        exp_id = f"ensemble_{fn_name}_24split_{timestamp}"
        art = ExperimentArtifact.create(
            experiment_id=exp_id,
            model_name=f"ensemble_{fn_name}",
            feature_set="FS-174 multi-model",
            description=f"Ensemble {fn_name} 24-split rolling OOF",
            models_used=MODEL_NAMES,
            fusion_method=fn_name,
            n_splits=len(splits),
        )
        pred_s = ens_data["pred"]
        aligned_labels = full_labels.reindex(pred_s.index).dropna()
        pred_s = pred_s.reindex(aligned_labels.index)
        art.save_predictions(
            pred_s.to_frame("score"),
            aligned_labels.to_frame("label"),
        )
        art.save_metrics(ens_data["metrics"])
        print(f"  Saved artifact: {exp_id}")

    # Summary JSON
    summary = {
        "timestamp": timestamp,
        "n_splits": len(splits),
        "n_features": len(feat_cols),
        "label_col": label_col,
        "models": MODEL_NAMES,
        "full_oof_metrics": full_metrics,
        "per_split_metrics": {str(k): v for k, v in split_metrics.items()},
        "ensemble_wins": ensemble_wins,
        "n_comparable_splits": n_comparable,
        "total_time_sec": round(time.time() - total_t0, 1),
    }
    summary_path = CHECKPOINT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved: {summary_path}")

    # ── Final comparison table ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FULL 24-SPLIT OOF COMPARISON")
    print(f"{'=' * 70}")

    rows = []
    for name, m in full_metrics.items():
        if "error" in m:
            rows.append({"Model": name, "Error": m["error"]})
        else:
            rows.append({
                "Model": name,
                "RankIC": m.get("rank_ic_mean"),
                "ICIR": m.get("rank_icir"),
                "IC": m.get("ic_mean"),
                "Spread20": m.get("spread_top20"),
                "Spread100": m.get("spread_top100"),
                "PosRatio": m.get("rank_ic_pos_ratio"),
                "Days": m.get("n_days"),
            })

    comparison_df = pd.DataFrame(rows)
    if "RankIC" in comparison_df.columns:
        comparison_df = comparison_df.sort_values("RankIC", ascending=False, na_position="last")
    print(comparison_df.to_string(index=False))
    print(f"\nTotal training time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    print("Done!")


if __name__ == "__main__":
    main()
