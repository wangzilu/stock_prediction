"""Phase 4L: DoubleEnsemble training — iterative hard-sample + feature reweighting.

Trains multiple LGBM rounds where each round upweights hard samples
(high loss from previous round) and selects important features.
Final prediction averages rank-normalized predictions from all rounds.

Usage: python scripts/phase4l_double_ensemble.py
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

# Split config — same as model zoo
TRAIN_START = "2023-01-01"
TRAIN_END = "2025-06-30"
TEST_START = "2025-07-01"
TEST_END = "2026-05-19"

LABEL_COL = "__label_5d"

# DoubleEnsemble config
N_ROUNDS = 5
FEATURE_SUBSAMPLE = 0.80  # keep 80% features per round
SEED_BASE = 42


# ── Metrics (same as phase4e) ────────────────────────────────────────

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


# ── Data Loading ─────────────────────────────────────────────────────

def load_data():
    """Load feature cache, split into train/test."""
    print(f"Loading feature cache: {FEATURE_CACHE}")
    df = pd.read_parquet(FEATURE_CACHE)
    print(f"  Shape: {df.shape}, date range: {df.index.get_level_values(0).min()} to {df.index.get_level_values(0).max()}")

    all_cols = list(df.columns)
    feat_cols = [c for c in all_cols if not c.startswith("__")]
    print(f"  Features: {len(feat_cols)}, Label: {LABEL_COL}")

    dates = df.index.get_level_values("datetime")
    train_mask = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    test_mask = (dates >= TEST_START) & (dates <= TEST_END)

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    print(f"  Train: {train_df.shape[0]:,} rows ({TRAIN_START} to {TRAIN_END})")
    print(f"  Test:  {test_df.shape[0]:,} rows ({TEST_START} to {TEST_END})")

    train_df = train_df.dropna(subset=[LABEL_COL])
    test_df = test_df.dropna(subset=[LABEL_COL])
    print(f"  After label dropna - Train: {train_df.shape[0]:,}, Test: {test_df.shape[0]:,}")

    X_train = train_df[feat_cols]
    y_train = train_df[LABEL_COL]
    X_test = test_df[feat_cols]
    y_test = test_df[LABEL_COL]

    return X_train, y_train, X_test, y_test, feat_cols


# ── DoubleEnsemble Training ─────────────────────────────────────────

def train_double_ensemble(X_train, y_train, X_test, y_test, feat_cols):
    """Train DoubleEnsemble: N rounds of LGBM with hard-sample reweighting
    and feature subsampling.

    Algorithm:
    1. Round 0: train base LGBM on uniform weights, all features
    2. For each subsequent round:
       a. Compute residuals from previous round on training set
       b. Upweight hard samples (high |residual|)
       c. Select top features by importance from previous round + random subset
       d. Train new LGBM with sample weights and feature subset
    3. Average rank-normalized test predictions across all rounds
    """
    import lightgbm as lgb

    base_params = {
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
    X_te = X_test.replace([np.inf, -np.inf], np.nan).values
    y_tr = y_train.values
    y_te = y_test.values

    all_feat_indices = np.arange(len(feat_cols))
    n_feat_keep = int(len(feat_cols) * FEATURE_SUBSAMPLE)

    round_preds_test = []
    round_preds_train = []
    sample_weights = np.ones(len(y_tr), dtype=np.float64)
    round_metrics = []

    for rnd in range(N_ROUNDS):
        rng = np.random.RandomState(SEED_BASE + rnd)
        print(f"\n  Round {rnd + 1}/{N_ROUNDS} (seed={SEED_BASE + rnd})")

        # Feature selection: top importance from prev round + random
        if rnd == 0:
            feat_idx = all_feat_indices.copy()
            feat_names_round = list(feat_cols)
        else:
            # Get feature importance from previous model
            importances = prev_model.feature_importances_
            # Pad importances to full feature set if needed (prev round used subset)
            full_importances = np.zeros(len(feat_cols))
            full_importances[prev_feat_idx] = importances

            # Select: top 60% by importance + random 20% from remainder
            n_top = int(n_feat_keep * 0.75)
            n_random = n_feat_keep - n_top

            top_idx = np.argsort(full_importances)[::-1][:n_top]
            remaining = np.setdiff1d(all_feat_indices, top_idx)
            rand_idx = rng.choice(remaining, size=min(n_random, len(remaining)), replace=False)
            feat_idx = np.sort(np.concatenate([top_idx, rand_idx]))
            feat_names_round = [feat_cols[i] for i in feat_idx]

        print(f"    Features: {len(feat_idx)}/{len(feat_cols)}")

        # Normalize sample weights
        w = sample_weights / sample_weights.sum() * len(sample_weights)

        # Train
        params = base_params.copy()
        params["random_state"] = SEED_BASE + rnd

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr[:, feat_idx], y_tr,
            sample_weight=w,
            eval_set=[(X_te[:, feat_idx], y_te)],
            callbacks=[lgb.log_evaluation(200)],
        )

        # Predict
        pred_train = model.predict(X_tr[:, feat_idx])
        pred_test = model.predict(X_te[:, feat_idx])

        # Rank-normalize predictions per day for ensemble averaging
        pred_test_series = pd.Series(pred_test, index=X_test.index, name="score")
        pred_test_ranked = pred_test_series.groupby(level="datetime").rank(pct=True)
        round_preds_test.append(pred_test_ranked)
        round_preds_train.append(pred_train)

        # Compute per-round metrics
        rnd_metrics = compute_metrics(pred_test_series, y_test)
        print(f"    Round RankIC: {rnd_metrics.get('rank_ic_mean', 'N/A')}")
        round_metrics.append(rnd_metrics)

        # Update sample weights: upweight hard samples (high |residual|)
        residuals = np.abs(y_tr - pred_train)
        # Convert residuals to weights: higher residual = higher weight
        # Use percentile-based soft reweighting
        residual_rank = pd.Series(residuals).rank(pct=True).values
        # Hard samples get up to 3x weight, easy samples get 0.5x
        sample_weights = 0.5 + 2.5 * residual_rank

        # Store for next round
        prev_model = model
        prev_feat_idx = feat_idx

    # Ensemble: average rank-normalized predictions
    print(f"\n  Combining {N_ROUNDS} round predictions...")
    ensemble_pred = pd.concat(round_preds_test, axis=1).mean(axis=1)
    ensemble_pred.name = "score"

    hyperparams = {
        **base_params,
        "n_rounds": N_ROUNDS,
        "feature_subsample": FEATURE_SUBSAMPLE,
        "seed_base": SEED_BASE,
        "method": "hard_sample_reweight + feature_importance_selection",
    }

    return ensemble_pred, hyperparams, round_metrics


# ── Main ─────────────────────────────────────────────────────────────

def main():
    from tracker.artifact_contract import ExperimentArtifact

    print("=" * 70)
    print("Phase 4L: DoubleEnsemble Training (FS-174)")
    print("=" * 70)

    X_train, y_train, X_test, y_test, feat_cols = load_data()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"double_ensemble_174_{timestamp}"

    print(f"\nExperiment: {exp_id}")
    t0 = time.time()

    try:
        pred, hyperparams, round_metrics = train_double_ensemble(
            X_train, y_train, X_test, y_test, feat_cols
        )
        elapsed = time.time() - t0
        print(f"\nTotal training took {elapsed:.1f}s")

        # Compute ensemble metrics
        metrics = compute_metrics(pred, y_test)
        metrics["train_time_sec"] = round(elapsed, 1)
        metrics["train_start"] = TRAIN_START
        metrics["train_end"] = TRAIN_END
        metrics["test_start"] = TEST_START
        metrics["test_end"] = TEST_END
        metrics["n_rounds"] = N_ROUNDS

        # Add per-round RankICs
        for i, rm in enumerate(round_metrics):
            metrics[f"round_{i}_rank_ic"] = rm.get("rank_ic_mean")

        print(f"\n{'=' * 60}")
        print("ENSEMBLE RESULTS")
        print(f"{'=' * 60}")
        print(f"  RankIC:     {metrics.get('rank_ic_mean', 'N/A')}")
        print(f"  RankICIR:   {metrics.get('rank_icir', 'N/A')}")
        print(f"  Spread(20): {metrics.get('spread_top20', 'N/A')} bps")
        print(f"  Spread(100):{metrics.get('spread_top100', 'N/A')} bps")
        print(f"  PosRatio:   {metrics.get('rank_ic_pos_ratio', 'N/A')}")

        print("\n  Per-round RankIC:")
        for i, rm in enumerate(round_metrics):
            print(f"    Round {i}: {rm.get('rank_ic_mean', 'N/A')}")

        # Save as ExperimentArtifact
        art = ExperimentArtifact.create(
            experiment_id=exp_id,
            model_name="double_ensemble_174",
            feature_set="FS-174",
            description=f"DoubleEnsemble: {N_ROUNDS}-round LGBM with hard-sample reweighting + feature selection",
            hyperparams=hyperparams,
            train_start=TRAIN_START,
            train_end=TRAIN_END,
            test_start=TEST_START,
            test_end=TEST_END,
            n_features=len(feat_cols),
        )
        art.save_predictions(pred.to_frame("score"), y_test.to_frame("label"))
        art.save_metrics(metrics)

        validation = art.validate()
        print(f"\n  Artifact saved: {validation['artifact_dir']}")
        print(f"  Complete: {validation['complete']}")

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\nFAILED after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return

    print("\nDone!")


if __name__ == "__main__":
    main()
