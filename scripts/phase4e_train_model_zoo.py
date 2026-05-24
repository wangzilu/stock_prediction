"""Phase 4E: Train model zoo (LGB, CatBoost, LGBMRanker) on FS-174 features.

Produces ExperimentArtifact outputs for ensemble fusion.
Uses a single recent train/test split to keep runtime reasonable.

Usage: python scripts/phase4e_train_model_zoo.py
"""
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "storage")
FEATURE_CACHE = os.path.join(DATA_DIR, "feature_cache_174_holder_regime_ma.parquet")

# Split config
TRAIN_START = "2023-01-01"
TRAIN_END = "2025-06-30"
TEST_START = "2025-07-01"
TEST_END = "2026-05-19"

LABEL_COL = "__label_5d"


# ── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(pred: pd.Series, label: pd.Series) -> dict:
    """Compute IC, RankIC, ICIR, spread metrics per-day then aggregate."""
    common = pred.index.intersection(label.index)
    pred = pred.loc[common]
    label = label.loc[common]

    # Drop NaN
    mask = pred.notna() & label.notna()
    pred = pred[mask]
    label = label[mask]

    if len(pred) < 100:
        return {"error": "too few samples"}

    # Per-day metrics
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

        # Spread: mean return of top N vs bottom N
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
        "rank_ic_pos_ratio": round(float(np.mean([1 for r in rank_ics if r > 0])), 4) if rank_ics else 0,
        "n_days": len(rank_ics),
        "n_predictions": len(pred),
    }
    # Fix pos ratio
    metrics["rank_ic_pos_ratio"] = round(sum(1 for r in rank_ics if r > 0) / len(rank_ics), 4)

    if spreads_top20:
        metrics["spread_top20"] = round(float(np.mean(spreads_top20)) * 10000, 2)  # bps
    if spreads_top50:
        metrics["spread_top50"] = round(float(np.mean(spreads_top50)) * 10000, 2)
    if spreads_top100:
        metrics["spread_top100"] = round(float(np.mean(spreads_top100)) * 10000, 2)

    return metrics


# ── Data Loading ─────────────────────────────────────────────────────

def load_data():
    """Load feature cache, split into train/test."""
    print(f"Loading feature cache: {FEATURE_CACHE}")
    df = pd.read_parquet(FEATURE_CACHE)
    print(f"  Shape: {df.shape}, date range: {df.index.get_level_values(0).min()} to {df.index.get_level_values(0).max()}")

    # Identify feature vs label columns
    all_cols = list(df.columns)
    feat_cols = [c for c in all_cols if not c.startswith("__")]
    print(f"  Features: {len(feat_cols)}, Label: {LABEL_COL}")

    # Split by date
    dates = df.index.get_level_values("datetime")
    train_mask = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    test_mask = (dates >= TEST_START) & (dates <= TEST_END)

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    print(f"  Train: {train_df.shape[0]:,} rows ({TRAIN_START} to {TRAIN_END})")
    print(f"  Test:  {test_df.shape[0]:,} rows ({TEST_START} to {TEST_END})")

    # Drop rows with NaN labels
    train_df = train_df.dropna(subset=[LABEL_COL])
    test_df = test_df.dropna(subset=[LABEL_COL])
    print(f"  After label dropna - Train: {train_df.shape[0]:,}, Test: {test_df.shape[0]:,}")

    X_train = train_df[feat_cols]
    y_train = train_df[LABEL_COL]
    X_test = test_df[feat_cols]
    y_test = test_df[LABEL_COL]

    return X_train, y_train, X_test, y_test, feat_cols


# ── Model Trainers ───────────────────────────────────────────────────

def train_lightgbm(X_train, y_train, X_test, y_test, feat_cols):
    """Train LightGBM regressor."""
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

    # LGB needs nan_to_num for inf but handles NaN natively
    X_tr = X_train.replace([np.inf, -np.inf], np.nan).values
    X_te = X_test.replace([np.inf, -np.inf], np.nan).values

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_train.values,
        eval_set=[(X_te, y_test.values)],
        callbacks=[lgb.log_evaluation(100)],
    )

    pred = pd.Series(model.predict(X_te), index=X_test.index, name="score")
    return pred, params


def train_catboost(X_train, y_train, X_test, y_test, feat_cols):
    """Train CatBoost regressor. CatBoost handles NaN natively."""
    from catboost import CatBoostRegressor

    params = {
        "iterations": 500,
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 580.0,
        "subsample": 0.88,
        "rsm": 0.88,  # colsample equivalent
        "loss_function": "RMSE",
        "verbose": 100,
        "thread_count": 4,
        "allow_writing_files": False,
    }

    # CatBoost handles NaN but not inf
    X_tr = X_train.replace([np.inf, -np.inf], np.nan)
    X_te = X_test.replace([np.inf, -np.inf], np.nan)

    model = CatBoostRegressor(**params)
    model.fit(
        X_tr, y_train,
        eval_set=(X_te, y_test),
        use_best_model=False,
    )

    pred = pd.Series(model.predict(X_te.values), index=X_test.index, name="score")
    return pred, params


def train_lgbm_ranker(X_train, y_train, X_test, y_test, feat_cols):
    """Train LGBMRanker (learning-to-rank).

    LGBMRanker requires integer labels. We discretize continuous returns
    into quantile bins (0..N_BINS-1) per date so the ranker sees
    within-day ordinal relevance grades.
    """
    import lightgbm as lgb

    N_BINS = 10  # quantile bins for label discretization

    def _discretize_per_date(y: pd.Series) -> np.ndarray:
        """Convert continuous labels to per-date quantile bins."""
        result = np.zeros(len(y), dtype=np.int32)
        dates = y.index.get_level_values("datetime")
        for dt in dates.unique():
            mask = dates == dt
            vals = y.values[mask]
            # qcut with duplicates='drop'; fallback to rank-based
            try:
                bins = pd.qcut(vals, N_BINS, labels=False, duplicates="drop")
            except Exception:
                bins = pd.Series(vals).rank(method="first", pct=True).values
                bins = (bins * (N_BINS - 1)).astype(int)
            result[mask] = bins
        return result

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
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
        "label_gain": list(range(N_BINS)),
    }

    # LGBMRanker needs group sizes (number of instruments per date)
    X_tr = X_train.replace([np.inf, -np.inf], np.nan)
    X_te = X_test.replace([np.inf, -np.inf], np.nan)

    y_tr_int = _discretize_per_date(y_train)
    y_te_int = _discretize_per_date(y_test)

    train_groups = X_tr.groupby(level="datetime").size().values
    test_groups = X_te.groupby(level="datetime").size().values

    model = lgb.LGBMRanker(**params)
    model.fit(
        X_tr.values, y_tr_int,
        group=train_groups,
        eval_set=[(X_te.values, y_te_int)],
        eval_group=[test_groups],
        callbacks=[lgb.log_evaluation(100)],
    )

    pred = pd.Series(model.predict(X_te.values), index=X_test.index, name="score")
    return pred, params


# ── Main ─────────────────────────────────────────────────────────────

def main():
    from tracker.artifact_contract import ExperimentArtifact

    print("=" * 70)
    print("Phase 4E: Model Zoo Training (FS-174)")
    print("=" * 70)

    X_train, y_train, X_test, y_test, feat_cols = load_data()

    models = {
        "lgb_174_single": {
            "fn": train_lightgbm,
            "model_name": "lgb_174",
            "description": "LightGBM regression on FS-174, single split",
        },
        "catboost_174_single": {
            "fn": train_catboost,
            "model_name": "catboost_174",
            "description": "CatBoost regression on FS-174, single split",
        },
        "lgbranker_174_single": {
            "fn": train_lgbm_ranker,
            "model_name": "lgbranker_174",
            "description": "LGBMRanker (LTR) on FS-174, single split",
        },
    }

    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for exp_key, spec in models.items():
        exp_id = f"{exp_key}_{timestamp}"
        print(f"\n{'─' * 60}")
        print(f"Training: {spec['model_name']} ({exp_id})")
        print(f"{'─' * 60}")

        t0 = time.time()
        try:
            pred, params = spec["fn"](X_train, y_train, X_test, y_test, feat_cols)
            elapsed = time.time() - t0
            print(f"Training took {elapsed:.1f}s")

            # Compute metrics
            metrics = compute_metrics(pred, y_test)
            metrics["train_time_sec"] = round(elapsed, 1)
            metrics["train_start"] = TRAIN_START
            metrics["train_end"] = TRAIN_END
            metrics["test_start"] = TEST_START
            metrics["test_end"] = TEST_END

            print(f"  RankIC: {metrics.get('rank_ic_mean', 'N/A')}")
            print(f"  ICIR:   {metrics.get('rank_icir', 'N/A')}")
            print(f"  Spread(top20): {metrics.get('spread_top20', 'N/A')} bps")

            # Save as ExperimentArtifact
            art = ExperimentArtifact.create(
                experiment_id=exp_id,
                model_name=spec["model_name"],
                feature_set="FS-174",
                description=spec["description"],
                hyperparams=params,
                train_start=TRAIN_START,
                train_end=TRAIN_END,
                test_start=TEST_START,
                test_end=TEST_END,
                n_features=len(feat_cols),
            )
            art.save_predictions(pred.to_frame("score"), y_test.to_frame("label"))
            art.save_metrics(metrics)

            validation = art.validate()
            print(f"  Artifact saved: {validation['artifact_dir']}")
            print(f"  Complete: {validation['complete']}")

            results[spec["model_name"]] = metrics

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED after {elapsed:.1f}s: {e}")
            traceback.print_exc()
            results[spec["model_name"]] = {"error": str(e)}

    # ── Try to load existing XGB174 predictions ──────────────────────
    print(f"\n{'─' * 60}")
    print("Attempting to load existing XGB174 champion predictions...")
    print(f"{'─' * 60}")
    try:
        import pickle
        model_path = os.path.join(DATA_DIR, "lgb_model.pkl")
        dataset_path = os.path.join(DATA_DIR, "lgb_dataset.pkl")

        if os.path.exists(model_path) and os.path.exists(dataset_path):
            with open(model_path, "rb") as f:
                xgb_model = pickle.load(f)
            with open(dataset_path, "rb") as f:
                xgb_dataset = pickle.load(f)

            # The saved model is a Qlib wrapper. Extract the underlying
            # xgboost/lightgbm booster and predict on raw numpy arrays.
            X_te_clean = X_test.replace([np.inf, -np.inf], np.nan)
            inner = getattr(xgb_model, "model", None)
            if inner is None:
                raise ValueError("Cannot extract inner model from Qlib wrapper")
            # xgboost.Booster or XGBRegressor
            if hasattr(inner, "predict"):
                import xgboost as xgb
                if isinstance(inner, xgb.Booster):
                    dmat = xgb.DMatrix(X_te_clean.values, feature_names=feat_cols)
                    xgb_pred_vals = inner.predict(dmat)
                else:
                    xgb_pred_vals = inner.predict(X_te_clean.values)
            else:
                raise ValueError(f"Unknown inner model type: {type(inner)}")
            if hasattr(xgb_pred_vals, 'values'):
                xgb_pred_vals = xgb_pred_vals.values
            xgb_pred = pd.Series(
                xgb_pred_vals.flatten(), index=X_test.index, name="score"
            )

            xgb_metrics = compute_metrics(xgb_pred, y_test)
            print(f"  XGB174 RankIC: {xgb_metrics.get('rank_ic_mean', 'N/A')}")

            # Save as artifact
            xgb_exp_id = f"xgb_174_single_{timestamp}"
            art = ExperimentArtifact.create(
                experiment_id=xgb_exp_id,
                model_name="xgb_174",
                feature_set="FS-174",
                description="XGB174 champion re-predicted on single split test set",
                train_start=TRAIN_START,
                train_end=TRAIN_END,
                test_start=TEST_START,
                test_end=TEST_END,
                n_features=len(feat_cols),
            )
            art.save_predictions(xgb_pred.to_frame("score"), y_test.to_frame("label"))
            art.save_metrics(xgb_metrics)
            results["xgb_174"] = xgb_metrics
            print(f"  XGB174 artifact saved: {art.artifact_dir}")
        else:
            print(f"  XGB model not found at {model_path}")
    except Exception as e:
        print(f"  XGB174 load failed: {e}")
        traceback.print_exc()

    # ── Comparison Table ─────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("MODEL ZOO COMPARISON (FS-174, single split)")
    print(f"{'=' * 70}")

    rows = []
    for name, m in results.items():
        if "error" in m:
            rows.append({"Model": name, "Error": m["error"]})
        else:
            rows.append({
                "Model": name,
                "RankIC": m.get("rank_ic_mean"),
                "ICIR": m.get("rank_icir"),
                "IC": m.get("ic_mean"),
                "Spread20(bps)": m.get("spread_top20"),
                "Spread100(bps)": m.get("spread_top100"),
                "PosRatio": m.get("rank_ic_pos_ratio"),
                "Days": m.get("n_days"),
                "Time(s)": m.get("train_time_sec"),
            })

    comparison_df = pd.DataFrame(rows)
    if "RankIC" in comparison_df.columns:
        comparison_df = comparison_df.sort_values("RankIC", ascending=False, na_position="last")
    print(comparison_df.to_string(index=False))

    print(f"\nArtifacts saved to: {os.path.join(DATA_DIR, 'experiments')}/")
    print("Done!")


if __name__ == "__main__":
    main()
