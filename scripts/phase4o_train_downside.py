"""Phase 4O: Train downside risk (crash probability) models.

Trains LightGBM binary classifiers to predict:
  - crash_1d: next-day drop > 5%
  - crash_5d: 5-day cumulative drop > 10%

Uses the same 205-feature cache as XGB174, joined with crash labels.

Usage: python scripts/phase4o_train_downside.py
"""
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "storage")
FEATURE_CACHE = os.path.join(DATA_DIR, "feature_cache_174_holder_regime_ma.parquet")
CRASH_LABELS = os.path.join(DATA_DIR, "crash_labels.parquet")

# Split config
TRAIN_START = "2023-01-01"
TRAIN_END = "2025-06-30"
TEST_START = "2025-07-01"
TEST_END = "2026-05-19"

CRASH_TARGETS = ["crash_1d", "crash_5d"]


def load_data():
    """Load feature cache and crash labels, join on (datetime, instrument)."""
    print(f"Loading feature cache: {FEATURE_CACHE}")
    df = pd.read_parquet(FEATURE_CACHE)
    print(f"  Feature cache shape: {df.shape}")
    print(f"  Date range: {df.index.get_level_values(0).min()} ~ "
          f"{df.index.get_level_values(0).max()}")

    # Identify feature columns (exclude labels/internals)
    feat_cols = [c for c in df.columns if not c.startswith("__") and not c.startswith("_")]
    print(f"  Feature columns: {len(feat_cols)}")

    print(f"\nLoading crash labels: {CRASH_LABELS}")
    cl = pd.read_parquet(CRASH_LABELS)
    print(f"  Crash labels shape: {cl.shape}, columns: {cl.columns.tolist()}")

    # Join features with crash labels
    merged = df[feat_cols].join(cl[CRASH_TARGETS], how="inner")
    print(f"  After join: {merged.shape}")

    # Split by date
    dates = merged.index.get_level_values("datetime")
    train_mask = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    test_mask = (dates >= TEST_START) & (dates <= TEST_END)

    train_df = merged.loc[train_mask].copy()
    test_df = merged.loc[test_mask].copy()

    print(f"  Train: {train_df.shape[0]:,} rows ({TRAIN_START} ~ {TRAIN_END})")
    print(f"  Test:  {test_df.shape[0]:,} rows ({TEST_START} ~ {TEST_END})")

    return train_df, test_df, feat_cols


def train_crash_model(train_df, test_df, feat_cols, target_col):
    """Train a LightGBM binary classifier for a single crash target."""
    import lightgbm as lgb
    from sklearn.metrics import (
        precision_score, recall_score, roc_auc_score,
        classification_report, confusion_matrix,
    )

    print(f"\n{'=' * 60}")
    print(f"Training crash model: {target_col}")
    print(f"{'=' * 60}")

    # Drop rows with NaN target
    tr = train_df.dropna(subset=[target_col])
    te = test_df.dropna(subset=[target_col])

    X_train = tr[feat_cols].replace([np.inf, -np.inf], np.nan).values
    y_train = tr[target_col].astype(int).values
    X_test = te[feat_cols].replace([np.inf, -np.inf], np.nan).values
    y_test = te[target_col].astype(int).values

    # Base rate
    train_pos_rate = y_train.mean()
    test_pos_rate = y_test.mean()
    print(f"  Train base rate: {train_pos_rate:.4f} ({y_train.sum():,}/{len(y_train):,})")
    print(f"  Test  base rate: {test_pos_rate:.4f} ({y_test.sum():,}/{len(y_test):,})")

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 6,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "n_jobs": -1,
        "is_unbalance": True,  # crash events are rare
    }

    model = lgb.LGBMClassifier(n_estimators=500, **params)
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(100)],
    )
    elapsed = time.time() - t0
    print(f"  Training took {elapsed:.1f}s")

    # Predict probabilities
    proba = model.predict_proba(X_test)[:, 1]
    pred_binary = (proba > 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, proba)
    precision = precision_score(y_test, pred_binary, zero_division=0)
    recall = recall_score(y_test, pred_binary, zero_division=0)
    pred_rate = pred_binary.mean()

    print(f"\n  --- Results for {target_col} ---")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Base rate (test):    {test_pos_rate:.4f}")
    print(f"  Model pred rate:     {pred_rate:.4f}")

    # Among stocks flagged as high crash_prob (>0.5), what fraction actually crashes?
    flagged_mask = proba > 0.5
    n_flagged = flagged_mask.sum()
    if n_flagged > 0:
        actual_crash_among_flagged = y_test[flagged_mask].mean()
        print(f"  Flagged (prob>0.5): {n_flagged:,} stocks")
        print(f"  Actual crash among flagged: {actual_crash_among_flagged:.4f} "
              f"({y_test[flagged_mask].sum()}/{n_flagged})")
    else:
        actual_crash_among_flagged = 0.0
        print(f"  Flagged (prob>0.5): 0 stocks (model never triggers at 0.5)")

    # Also check at lower thresholds
    for threshold in [0.3, 0.2, 0.1]:
        flagged_t = proba > threshold
        n_t = flagged_t.sum()
        if n_t > 0:
            crash_rate_t = y_test[flagged_t].mean()
            print(f"  Flagged (prob>{threshold}): {n_t:,} stocks, "
                  f"actual crash rate: {crash_rate_t:.4f}")

    # Confusion matrix
    print(f"\n  Confusion matrix (threshold=0.5):")
    cm = confusion_matrix(y_test, pred_binary)
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    # Feature importance (top 15)
    fi = pd.Series(model.feature_importances_, index=feat_cols)
    fi = fi.sort_values(ascending=False).head(15)
    print(f"\n  Top 15 features:")
    for fname, imp in fi.items():
        print(f"    {fname:30s} {imp}")

    # Build crash_prob series with original index
    crash_prob = pd.Series(proba, index=te.index, name=f"{target_col}_prob")

    metrics = {
        "target": target_col,
        "auc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "base_rate_train": round(train_pos_rate, 4),
        "base_rate_test": round(test_pos_rate, 4),
        "model_pred_rate": round(pred_rate, 4),
        "n_flagged_0.5": int(n_flagged),
        "actual_crash_among_flagged": round(float(actual_crash_among_flagged), 4),
        "train_time_sec": round(elapsed, 1),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }

    return crash_prob, metrics, model


def analyze_xgb_overlap(test_df, feat_cols, crash_probs):
    """Among top-100 XGB picks per day, how many does the crash model flag?"""
    print(f"\n{'=' * 60}")
    print("XGB Top-100 vs Crash Risk Overlap Analysis")
    print(f"{'=' * 60}")

    # Load XGB model to generate predictions on test set
    import pickle
    model_path = os.path.join(DATA_DIR, "lgb_model.pkl")
    if not os.path.exists(model_path):
        print("  XGB model not found, skipping overlap analysis")
        return

    try:
        with open(model_path, "rb") as f:
            xgb_wrapper = pickle.load(f)

        inner = getattr(xgb_wrapper, "model", None)
        if inner is None:
            print("  Cannot extract inner model, skipping")
            return

        X_test = test_df[feat_cols].replace([np.inf, -np.inf], np.nan)

        if hasattr(inner, "predict"):
            import xgboost as xgb
            if isinstance(inner, xgb.Booster):
                dmat = xgb.DMatrix(X_test.values, feature_names=feat_cols)
                xgb_pred = inner.predict(dmat)
            else:
                xgb_pred = inner.predict(X_test.values)
        else:
            print(f"  Unknown model type: {type(inner)}")
            return

        xgb_scores = pd.Series(
            xgb_pred.flatten(), index=test_df.index, name="xgb_score"
        )

        # Per day: get top-100 XGB picks, check crash flags
        # Use groupby for efficiency instead of per-day .xs()
        for target_col, crash_prob in crash_probs.items():
            print(f"\n  --- {target_col} overlap with XGB top-100 ---")

            combined = pd.DataFrame({
                "xgb": xgb_scores,
                "crash_prob": crash_prob,
            }).dropna()

            daily_stats = []
            for dt, grp in combined.groupby(level="datetime"):
                if len(grp) < 100:
                    continue
                top100_idx = grp["xgb"].nlargest(100).index
                crash_probs_top100 = grp.loc[top100_idx, "crash_prob"]

                daily_stats.append({
                    "n_flagged_50": (crash_probs_top100 > 0.5).sum(),
                    "n_flagged_30": (crash_probs_top100 > 0.3).sum(),
                    "n_flagged_20": (crash_probs_top100 > 0.2).sum(),
                    "avg_crash_prob": crash_probs_top100.mean(),
                })

            if daily_stats:
                stats_df = pd.DataFrame(daily_stats)
                print(f"    Avg flagged (>0.5) per day in top-100: "
                      f"{stats_df['n_flagged_50'].mean():.1f}")
                print(f"    Avg flagged (>0.3) per day in top-100: "
                      f"{stats_df['n_flagged_30'].mean():.1f}")
                print(f"    Avg flagged (>0.2) per day in top-100: "
                      f"{stats_df['n_flagged_20'].mean():.1f}")
                print(f"    Avg crash prob in top-100: "
                      f"{stats_df['avg_crash_prob'].mean():.4f}")
                print(f"    Max flagged (>0.5) any day: "
                      f"{stats_df['n_flagged_50'].max()}")
                print(f"    Days analyzed: {len(daily_stats)}")

    except Exception as e:
        print(f"  XGB overlap analysis failed: {e}")
        traceback.print_exc()


def main():
    from tracker.artifact_contract import ExperimentArtifact

    print("=" * 70)
    print("Phase 4O: Downside Risk Model Training")
    print("=" * 70)

    train_df, test_df, feat_cols = load_data()

    crash_probs = {}
    all_metrics = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for target_col in CRASH_TARGETS:
        try:
            crash_prob, metrics, model = train_crash_model(
                train_df, test_df, feat_cols, target_col
            )
            crash_probs[target_col] = crash_prob
            all_metrics[target_col] = metrics

            # Save as ExperimentArtifact
            exp_id = f"crash_{target_col}_{timestamp}"
            art = ExperimentArtifact.create(
                experiment_id=exp_id,
                model_name=f"crash_{target_col}",
                feature_set="FS-205",
                description=f"LightGBM binary classifier for {target_col}",
                train_start=TRAIN_START,
                train_end=TRAIN_END,
                test_start=TEST_START,
                test_end=TEST_END,
                n_features=len(feat_cols),
            )
            art.save_predictions(
                crash_prob.to_frame("crash_prob"),
                test_df[target_col].to_frame("label"),
            )
            art.save_metrics(metrics)

            # Save model.pkl so predict_crash_daily.py can find it
            art._write_pickle("model.pkl", model)

            validation = art.validate()
            print(f"\n  Artifact saved: {validation['artifact_dir']}")
            print(f"  Complete: {validation['complete']}")

        except Exception as e:
            print(f"\n  FAILED for {target_col}: {e}")
            traceback.print_exc()
            all_metrics[target_col] = {"error": str(e)}

    # XGB overlap analysis
    if crash_probs:
        analyze_xgb_overlap(test_df, feat_cols, crash_probs)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for target_col, m in all_metrics.items():
        if "error" in m:
            print(f"  {target_col}: ERROR - {m['error']}")
        else:
            print(f"  {target_col}:")
            print(f"    AUC={m['auc']:.4f}  Precision={m['precision']:.4f}  "
                  f"Recall={m['recall']:.4f}")
            print(f"    Base rate={m['base_rate_test']:.4f}  "
                  f"Model pred rate={m['model_pred_rate']:.4f}  "
                  f"Flagged(>0.5)={m['n_flagged_0.5']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
