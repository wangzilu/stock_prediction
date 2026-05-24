"""Phase 4M: Alpha360 artifact contract + gate evaluation.

Runs Alpha360 (360-dim) feature set through the same artifact contract
and promotion gate system as the 174-dim champion, producing an
apples-to-apples comparison.

Steps:
  1. Load Alpha360 cache and 174-dim cache (for returns)
  2. Train XGBoost and LightGBM on Alpha360 across 24 rolling splits
  3. Compute signal-layer metrics (IC, RankIC, ICIR, spreads)
  4. Compute per-feature IC for Alpha360 (top features report)
  5. Register ExperimentArtifact entries
  6. Run PromotionGate checks
  7. Print comparison table vs 174-dim champion

Usage:
    python scripts/phase4m_alpha360_gate.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import xgboost as xgb

from tracker.artifact_contract import ExperimentArtifact, compare_experiments
from tracker.promotion_gate import PromotionGate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"

# Rolling config (same as champion)
N_SPLITS = 24
TEST_DAYS = 20
VALID_DAYS = 60
TRAIN_DAYS = 750

# XGB params (same as champion)
XGB_PARAMS = {
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8789,
    "colsample_bytree": 0.8879,
    "reg_alpha": 205.6999,
    "reg_lambda": 580.9768,
    "objective": "reg:squarederror",
    "nthread": 12,
    "verbosity": 0,
    "seed": 42,
}
NUM_BOOST_ROUND = 400
EARLY_STOPPING_ROUNDS = 50

LABEL_COL = "__label_5d"
PNL_COL = "__pnl_return_1d"

# Cost model (same as champion)
COMMISSION = 0.0003 * 2
STAMP_TAX = 0.0005
SLIPPAGE = 0.001 * 2
TOTAL_COST_PER_TRADE = COMMISSION + STAMP_TAX + SLIPPAGE

SPREAD_TIERS = [20, 50, 100]


def load_alpha360_cache():
    """Load Alpha360 feature cache."""
    path = DATA_DIR / "feature_cache_alpha360.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Alpha360 cache not found: {path}")
    logger.info("Loading Alpha360 cache...")
    cache = pd.read_parquet(str(path))
    feat_cols = [c for c in cache.columns if not c.startswith("__")]
    logger.info(f"  Shape: {cache.shape}, features: {len(feat_cols)}")
    return cache, feat_cols


def load_174_cache():
    """Load 174-dim champion cache (for reference)."""
    path = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
    if not path.exists():
        raise FileNotFoundError(f"174-dim cache not found: {path}")
    logger.info("Loading 174-dim cache...")
    cache = pd.read_parquet(str(path))
    feat_cols = [c for c in cache.columns
                 if not c.startswith("__") and not c.startswith("_")
                 and not c.startswith("hsi_") and not c.startswith("hstech_")
                 and not c.startswith("nasdaq_")]
    logger.info(f"  Shape: {cache.shape}, features: {len(feat_cols)}")
    return cache, feat_cols


def compute_feature_ic(cache, feat_cols, n_sample_dates=60):
    """Compute individual IC for each Alpha360 feature against 1d returns.

    Uses a random subsample of dates for speed.
    """
    logger.info(f"Computing per-feature IC ({len(feat_cols)} features)...")
    from scipy.stats import spearmanr

    dates = sorted(cache.index.get_level_values(0).unique())
    # Sample dates evenly across the range
    if len(dates) > n_sample_dates:
        step = len(dates) // n_sample_dates
        sample_dates = dates[::step][:n_sample_dates]
    else:
        sample_dates = dates

    ics = {}
    for col in feat_cols:
        col_ics = []
        for d in sample_dates:
            try:
                g = cache.loc[d]
                x = g[col].values
                y = g[PNL_COL].values
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 30:
                    continue
                corr, _ = spearmanr(x[mask], y[mask])
                if np.isfinite(corr):
                    col_ics.append(corr)
            except Exception:
                continue
        if col_ics:
            ics[col] = {
                "ic_mean": float(np.mean(col_ics)),
                "ic_std": float(np.std(col_ics)),
                "ic_abs_mean": float(np.mean(np.abs(col_ics))),
                "n_dates": len(col_ics),
            }

    # Sort by |IC|
    sorted_ics = sorted(ics.items(), key=lambda kv: abs(kv[1]["ic_mean"]), reverse=True)
    return sorted_ics


def run_rolling_splits(cache, feat_cols, model_type="xgb"):
    """Run 24-split rolling evaluation. Returns per-split metrics and predictions."""
    trade_dates = sorted(cache.index.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1
    dl = cache.index.get_level_values(0)

    all_split_metrics = []
    all_pred = []
    all_label = []
    all_pnl = []
    t_total = time.time()

    for split_idx in range(N_SPLITS):
        test_end_idx = today_idx - split_idx * TEST_DAYS
        test_start_idx = test_end_idx - TEST_DAYS
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - VALID_DAYS
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - TRAIN_DAYS

        if train_start_idx < 0 or test_end_idx >= len(trade_dates):
            logger.warning(f"Split {split_idx+1}: out of bounds, stopping")
            break

        test_end = trade_dates[test_end_idx]
        test_start = trade_dates[test_start_idx]
        train_end = trade_dates[train_end_idx]
        train_start = trade_dates[train_start_idx]
        valid_end = trade_dates[valid_end_idx]
        valid_start = trade_dates[valid_start_idx]

        logger.info(f"Split {split_idx+1}/{N_SPLITS}: "
                    f"test {str(test_start)[:10]}~{str(test_end)[:10]}")

        try:
            t0 = time.time()

            train_mask = (dl >= train_start) & (dl <= train_end)
            valid_mask = (dl >= valid_start) & (dl <= valid_end)
            test_mask = (dl >= test_start) & (dl <= test_end)

            train_df = cache.loc[train_mask]
            valid_df = cache.loc[valid_mask]
            test_df = cache.loc[test_mask]

            def prep(df):
                X = df[feat_cols].values.astype(np.float32)
                y = df[LABEL_COL].values.astype(np.float32)
                mask = np.isfinite(y)
                return X[mask], y[mask], df.index[mask]

            X_tr, y_tr, idx_tr = prep(train_df)
            X_va, y_va, idx_va = prep(valid_df)
            X_te, y_te, idx_te = prep(test_df)

            if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
                logger.warning(f"  Split {split_idx+1}: empty segment, skipping")
                continue

            if model_type == "xgb":
                dt = xgb.DMatrix(X_tr, label=y_tr)
                dv = xgb.DMatrix(X_va, label=y_va)
                model = xgb.train(
                    XGB_PARAMS, dt,
                    num_boost_round=NUM_BOOST_ROUND,
                    evals=[(dv, "valid")],
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose_eval=0,
                )
                pred = model.predict(xgb.DMatrix(X_te))
            elif model_type == "lgb":
                import lightgbm as lgb
                lgb_params = {
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "subsample": 0.88,
                    "colsample_bytree": 0.88,
                    "reg_alpha": 200.0,
                    "reg_lambda": 580.0,
                    "objective": "regression",
                    "n_jobs": 12,
                    "verbosity": -1,
                    "seed": 42,
                }
                dtrain = lgb.Dataset(X_tr, label=y_tr)
                dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
                model = lgb.train(
                    lgb_params, dtrain,
                    num_boost_round=NUM_BOOST_ROUND,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                               lgb.log_evaluation(period=0)],
                )
                pred = model.predict(X_te)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            # Compute IC / RankIC
            from scipy.stats import spearmanr, pearsonr
            mask_valid = np.isfinite(pred) & np.isfinite(y_te)
            ps = pd.Series(pred[mask_valid], index=idx_te[mask_valid])
            ls = pd.Series(y_te[mask_valid], index=idx_te[mask_valid])

            # Daily IC/RankIC
            daily_ics = []
            daily_rics = []
            for d in sorted(ps.index.get_level_values(0).unique()):
                p_d = ps.loc[d]
                l_d = ls.loc[d]
                if len(p_d) < 20:
                    continue
                ic_val, _ = pearsonr(p_d.values, l_d.values)
                ric_val, _ = spearmanr(p_d.values, l_d.values)
                if np.isfinite(ic_val):
                    daily_ics.append(ic_val)
                if np.isfinite(ric_val):
                    daily_rics.append(ric_val)

            ic_mean = float(np.mean(daily_ics)) if daily_ics else 0.0
            ic_std = float(np.std(daily_ics)) if daily_ics else 1.0
            icir = ic_mean / (ic_std + 1e-8)
            ric_mean = float(np.mean(daily_rics)) if daily_rics else 0.0
            ric_std = float(np.std(daily_rics)) if daily_rics else 1.0
            ricir = ric_mean / (ric_std + 1e-8)

            # Spread metrics
            pnl_data = test_df.loc[idx_te[mask_valid], PNL_COL] if PNL_COL in cache.columns else None
            spreads = {}
            if pnl_data is not None:
                df_sp = pd.DataFrame({"pred": ps, "pnl": pnl_data}).dropna()
                for k in SPREAD_TIERS:
                    tier_spreads = []
                    for date, g in df_sp.groupby(level=0):
                        if len(g) < k * 2:
                            continue
                        s = g.sort_values("pred", ascending=False)
                        tier_spreads.append(s.head(k)["pnl"].mean() - s.tail(k)["pnl"].mean())
                    if tier_spreads:
                        spreads[f"spread_top{k}"] = float(np.mean(tier_spreads))

            elapsed = time.time() - t0

            split_result = {
                "split": split_idx + 1,
                "test_start": str(test_start)[:10],
                "test_end": str(test_end)[:10],
                "n_train": len(X_tr),
                "n_test": len(X_te),
                "ic_mean": round(ic_mean, 6),
                "ic_std": round(ic_std, 6),
                "icir": round(icir, 4),
                "rank_ic_mean": round(ric_mean, 6),
                "rank_ic_std": round(ric_std, 6),
                "rank_icir": round(ricir, 4),
                "time_s": round(elapsed, 1),
                **{k: round(v, 6) for k, v in spreads.items()},
            }
            all_split_metrics.append(split_result)
            all_pred.append(ps)
            all_label.append(ls)
            if pnl_data is not None:
                all_pnl.append(pnl_data)

            logger.info(f"  IC={ic_mean:+.4f} ICIR={icir:+.3f} "
                        f"RankIC={ric_mean:+.4f} RICIR={ricir:+.3f} [{elapsed:.0f}s]")

        except Exception as e:
            logger.error(f"  Split {split_idx+1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - t_total
    return all_split_metrics, all_pred, all_label, all_pnl, total_time


def aggregate_metrics(split_metrics):
    """Aggregate per-split metrics into summary."""
    n = len(split_metrics)
    if n == 0:
        return {}

    ric_vals = [s["rank_ic_mean"] for s in split_metrics]
    ic_vals = [s["ic_mean"] for s in split_metrics]

    return {
        "rank_ic_mean": round(float(np.mean(ric_vals)), 6),
        "rank_ic_std": round(float(np.std(ric_vals)), 6),
        "rank_icir": round(float(np.mean(ric_vals) / (np.std(ric_vals) + 1e-8)), 4),
        "ic_mean": round(float(np.mean(ic_vals)), 6),
        "ic_std": round(float(np.std(ic_vals)), 6),
        "icir": round(float(np.mean(ic_vals) / (np.std(ic_vals) + 1e-8)), 4),
        "rank_ic_pos_ratio": round(float(np.mean([r > 0 for r in ric_vals])), 4),
        "n_splits": n,
        "n_days": sum(s.get("n_test", 0) for s in split_metrics),
        # Average spreads across splits (where available)
        **{k: round(float(np.mean([s[k] for s in split_metrics if k in s])), 6)
           for k in ["spread_top20", "spread_top50", "spread_top100"]
           if any(k in s for s in split_metrics)},
    }


def register_artifact(experiment_id, model_name, feature_set, description,
                       metrics, split_metrics, n_features):
    """Register an experiment in the artifact contract system."""
    art = ExperimentArtifact.create(
        experiment_id=experiment_id,
        model_name=model_name,
        feature_set=feature_set,
        description=description,
        n_features=n_features,
        n_splits=len(split_metrics),
        train_days=TRAIN_DAYS,
        xgb_params=XGB_PARAMS if "xgb" in model_name else None,
        cost_model={
            "commission": COMMISSION,
            "stamp_tax": STAMP_TAX,
            "slippage": SLIPPAGE,
            "total_per_trade": TOTAL_COST_PER_TRADE,
        },
    )
    art.save_metrics(metrics)
    logger.info(f"  Registered artifact: {experiment_id}")
    return art


def main():
    logger.info("=" * 80)
    logger.info("PHASE 4M: Alpha360 Artifact Contract + Gate Evaluation")
    logger.info("=" * 80)

    # ----------------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------------
    alpha_cache, alpha_feat_cols = load_alpha360_cache()
    logger.info(f"Alpha360: {len(alpha_feat_cols)} features, "
                f"{alpha_cache.shape[0]:,} rows")

    # ----------------------------------------------------------------
    # 2. Per-feature IC analysis (quick, sampled)
    # ----------------------------------------------------------------
    feature_ics = compute_feature_ic(alpha_cache, alpha_feat_cols, n_sample_dates=60)

    logger.info(f"\n{'='*80}")
    logger.info("TOP 30 ALPHA360 FEATURES BY |IC|")
    logger.info(f"{'='*80}")
    logger.info(f"{'Feature':<20} {'IC_mean':>10} {'|IC|':>10} {'IC_std':>10}")
    logger.info("-" * 55)
    for col, vals in feature_ics[:30]:
        logger.info(f"{col:<20} {vals['ic_mean']:+.5f}   {vals['ic_abs_mean']:.5f}   "
                    f"{vals['ic_std']:.5f}")

    # ----------------------------------------------------------------
    # 3. Run rolling XGBoost on Alpha360
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*80}")
    logger.info("RUNNING ALPHA360 + XGBoost (24-split rolling)")
    logger.info(f"{'='*80}")

    xgb_splits, xgb_pred, xgb_label, xgb_pnl, xgb_time = run_rolling_splits(
        alpha_cache, alpha_feat_cols, model_type="xgb"
    )
    xgb_agg = aggregate_metrics(xgb_splits)

    # ----------------------------------------------------------------
    # 4. Run rolling LightGBM on Alpha360
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*80}")
    logger.info("RUNNING ALPHA360 + LightGBM (24-split rolling)")
    logger.info(f"{'='*80}")

    try:
        lgb_splits, lgb_pred, lgb_label, lgb_pnl, lgb_time = run_rolling_splits(
            alpha_cache, alpha_feat_cols, model_type="lgb"
        )
        lgb_agg = aggregate_metrics(lgb_splits)
        lgb_available = True
    except Exception as e:
        logger.warning(f"LightGBM run failed: {e}")
        lgb_splits, lgb_agg = [], {}
        lgb_available = False

    # ----------------------------------------------------------------
    # 5. Register artifacts
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*80}")
    logger.info("REGISTERING ARTIFACTS")
    logger.info(f"{'='*80}")

    xgb_art = register_artifact(
        experiment_id="alpha360_xgb",
        model_name="xgb_alpha360",
        feature_set="Alpha360",
        description="Alpha360 360-dim features + XGBoost, 24-split rolling",
        metrics=xgb_agg,
        split_metrics=xgb_splits,
        n_features=len(alpha_feat_cols),
    )

    if lgb_available:
        lgb_art = register_artifact(
            experiment_id="alpha360_lgb",
            model_name="lgb_alpha360",
            feature_set="Alpha360",
            description="Alpha360 360-dim features + LightGBM, 24-split rolling",
            metrics=lgb_agg,
            split_metrics=lgb_splits,
            n_features=len(alpha_feat_cols),
        )

    # ----------------------------------------------------------------
    # 6. Promotion gate checks
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*80}")
    logger.info("PROMOTION GATE CHECKS")
    logger.info(f"{'='*80}")

    gate = PromotionGate()

    xgb_gate = gate.check("alpha360_xgb", champion_id="xgb_174_champion")
    logger.info(f"\nalpha360_xgb gate: {'PASS' if xgb_gate['pass'] else 'FAIL'} "
                f"-> {xgb_gate['recommendation']}")
    if xgb_gate["failures"]:
        for f in xgb_gate["failures"]:
            logger.info(f"  FAIL: {f}")
    if xgb_gate["warnings"]:
        for w in xgb_gate["warnings"]:
            logger.info(f"  WARN: {w}")

    if lgb_available:
        lgb_gate = gate.check("alpha360_lgb", champion_id="xgb_174_champion")
        logger.info(f"\nalpha360_lgb gate: {'PASS' if lgb_gate['pass'] else 'FAIL'} "
                    f"-> {lgb_gate['recommendation']}")
        if lgb_gate["failures"]:
            for f in lgb_gate["failures"]:
                logger.info(f"  FAIL: {f}")
        if lgb_gate["warnings"]:
            for w in lgb_gate["warnings"]:
                logger.info(f"  WARN: {w}")

    # ----------------------------------------------------------------
    # 7. Comparison table
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON TABLE: Alpha360 vs 174-dim Champion")
    logger.info(f"{'='*80}")

    compare_ids = ["xgb_174_champion", "alpha360_xgb"]
    if lgb_available:
        compare_ids.append("alpha360_lgb")

    # Also include any other existing experiments
    all_ids = ExperimentArtifact.list_all()
    extra_ids = [eid for eid in all_ids if eid not in compare_ids]
    compare_ids.extend(extra_ids)

    comp_df = compare_experiments(compare_ids)

    # Print formatted table
    display_cols = ["experiment_id", "feature_set", "rank_ic", "rank_icir",
                    "spread_top20", "sharpe", "annual_return", "max_drawdown"]
    available_cols = [c for c in display_cols if c in comp_df.columns]
    logger.info("\n" + comp_df[available_cols].to_string(index=False))

    # ----------------------------------------------------------------
    # 8. Head-to-head summary
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*80}")
    logger.info("HEAD-TO-HEAD: Alpha360 XGB vs FS-174 Champion")
    logger.info(f"{'='*80}")

    # Load champion metrics for comparison
    champ_art = ExperimentArtifact.load("xgb_174_champion")
    champ_m = champ_art.load_metrics()

    rows = [
        ("Metric", "FS-174 Champion", "Alpha360 XGB",
         "Alpha360 LGB" if lgb_available else ""),
        ("RankIC mean", f"{champ_m.get('rank_ic_mean', 0):.4f}",
         f"{xgb_agg.get('rank_ic_mean', 0):.4f}",
         f"{lgb_agg.get('rank_ic_mean', 0):.4f}" if lgb_available else ""),
        ("RankIC std", f"{champ_m.get('rank_ic_std', 0):.4f}",
         f"{xgb_agg.get('rank_ic_std', 0):.4f}",
         f"{lgb_agg.get('rank_ic_std', 0):.4f}" if lgb_available else ""),
        ("RankICIR", f"{champ_m.get('rank_icir', 0):.3f}",
         f"{xgb_agg.get('rank_icir', 0):.3f}",
         f"{lgb_agg.get('rank_icir', 0):.3f}" if lgb_available else ""),
        ("IC mean", f"{champ_m.get('ic_mean', 0):.4f}",
         f"{xgb_agg.get('ic_mean', 0):.4f}",
         f"{lgb_agg.get('ic_mean', 0):.4f}" if lgb_available else ""),
        ("ICIR", f"{champ_m.get('icir', 0):.3f}",
         f"{xgb_agg.get('icir', 0):.3f}",
         f"{lgb_agg.get('icir', 0):.3f}" if lgb_available else ""),
        ("RankIC pos ratio", f"{champ_m.get('rank_ic_pos_ratio', 0):.2f}",
         f"{xgb_agg.get('rank_ic_pos_ratio', 0):.2f}",
         f"{lgb_agg.get('rank_ic_pos_ratio', 0):.2f}" if lgb_available else ""),
        ("Spread top20", f"{champ_m.get('spread_top20', 0):.4f}",
         f"{xgb_agg.get('spread_top20', 0):.4f}",
         f"{lgb_agg.get('spread_top20', 0):.4f}" if lgb_available else ""),
        ("Spread top100", f"{champ_m.get('spread_top100', 0):.4f}",
         f"{xgb_agg.get('spread_top100', 0):.4f}",
         f"{lgb_agg.get('spread_top100', 0):.4f}" if lgb_available else ""),
        ("N splits", f"{champ_m.get('n_splits', 'N/A')}",
         f"{xgb_agg.get('n_splits', 0)}",
         f"{lgb_agg.get('n_splits', 0)}" if lgb_available else ""),
        ("N features", "174", f"{len(alpha_feat_cols)}",
         f"{len(alpha_feat_cols)}" if lgb_available else ""),
    ]

    header = rows[0]
    if lgb_available:
        logger.info(f"{'':>20} {'FS-174 Champion':>18} {'Alpha360 XGB':>15} {'Alpha360 LGB':>15}")
        logger.info("-" * 70)
        for row in rows[1:]:
            logger.info(f"{row[0]:>20} {row[1]:>18} {row[2]:>15} {row[3]:>15}")
    else:
        logger.info(f"{'':>20} {'FS-174 Champion':>18} {'Alpha360 XGB':>15}")
        logger.info("-" * 55)
        for row in rows[1:]:
            logger.info(f"{row[0]:>20} {row[1]:>18} {row[2]:>15}")

    # Delta analysis
    delta_ric = xgb_agg.get("rank_ic_mean", 0) - champ_m.get("rank_ic_mean", 0)
    logger.info(f"\nDelta RankIC (Alpha360 XGB - Champion): {delta_ric:+.4f}")
    if delta_ric > 0:
        logger.info("  -> Alpha360 XGB has HIGHER RankIC")
    else:
        logger.info("  -> FS-174 Champion retains advantage")

    # ----------------------------------------------------------------
    # 9. Save results
    # ----------------------------------------------------------------
    out_dir = DATA_DIR / "phase4"
    out_dir.mkdir(exist_ok=True)

    result = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "alpha360_xgb": {
            "aggregate": xgb_agg,
            "per_split": xgb_splits,
            "time_s": round(xgb_time, 1),
            "gate_pass": xgb_gate["pass"],
            "gate_recommendation": xgb_gate["recommendation"],
            "gate_failures": xgb_gate["failures"],
        },
        "feature_ic_top30": [
            {"feature": col, **vals} for col, vals in feature_ics[:30]
        ],
    }

    if lgb_available:
        result["alpha360_lgb"] = {
            "aggregate": lgb_agg,
            "per_split": lgb_splits,
            "time_s": round(lgb_time, 1),
            "gate_pass": lgb_gate["pass"],
            "gate_recommendation": lgb_gate["recommendation"],
            "gate_failures": lgb_gate["failures"],
        }

    out_path = out_dir / "phase4m_alpha360_gate.json"
    with open(str(out_path), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nSaved results: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
