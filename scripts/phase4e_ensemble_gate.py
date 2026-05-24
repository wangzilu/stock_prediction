"""Phase 4E: Ensemble fusion evaluation — compare fused signal vs best single model.

Single-split exploratory check (NOT the final 24-split gate).
Loads pred.pkl from all available model artifacts, fuses them, and evaluates.

Usage:
    python scripts/phase4e_ensemble_gate.py
"""
import logging
import os
import pickle
import sys
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
EXPERIMENTS_DIR = DATA_DIR / "experiments"


def load_all_predictions() -> dict[str, pd.Series]:
    """Load pred.pkl from all experiments, skip broken ones."""
    preds = {}
    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        pred_path = exp_dir / "pred.pkl"
        if not pred_path.exists():
            continue
        try:
            with open(pred_path, "rb") as f:
                pred = pickle.load(f)
            # Convert DataFrame to Series if needed
            if isinstance(pred, pd.DataFrame):
                pred = pred.iloc[:, 0]
            if isinstance(pred, pd.Series) and len(pred) > 1000:
                if pred.nunique() <= 1:
                    logger.info(f"  Skip {exp_dir.name}: constant predictions")
                    continue
                preds[exp_dir.name] = pred
                logger.info(f"  Loaded {exp_dir.name}: {len(pred)} predictions")
        except Exception as e:
            logger.warning(f"  Failed {exp_dir.name}: {e}")
    return preds


def daily_rank_ic(pred: pd.Series, ret: pd.Series) -> list[float]:
    """Compute daily cross-sectional Spearman RankIC."""
    common = pred.index.intersection(ret.index)
    p = pred.reindex(common).dropna()
    r = ret.reindex(p.index).dropna()
    p = p.reindex(r.index)

    rics = []
    for date in p.index.get_level_values(0).unique():
        try:
            p_day = p.loc[date]
            r_day = r.loc[date]
            if len(p_day) < 50:
                continue
            ric = stats.spearmanr(p_day, r_day).statistic
            if np.isfinite(ric):
                rics.append(ric)
        except Exception:
            pass
    return rics


def daily_spread(pred: pd.Series, ret: pd.Series, top_k: int = 20) -> list[float]:
    """Compute daily top_k vs bottom_k spread."""
    common = pred.index.intersection(ret.index)
    p = pred.reindex(common).dropna()
    r = ret.reindex(p.index).dropna()
    p = p.reindex(r.index)

    spreads = []
    for date in p.index.get_level_values(0).unique():
        try:
            p_day = p.loc[date]
            r_day = r.loc[date]
            if len(p_day) < top_k * 4:
                continue
            ranked = p_day.sort_values(ascending=False)
            top = r_day.reindex(ranked.index[:top_k]).mean()
            bot = r_day.reindex(ranked.index[-top_k:]).mean()
            spreads.append(top - bot)
        except Exception:
            pass
    return spreads


def main():
    from models.ensemble_fusion import (
        rank_normalize, fuse_rank_mean, fuse_robust_z_mean,
        compute_model_disagreement, compute_model_consensus,
    )
    from tracker.artifact_contract import ExperimentArtifact

    logger.info("=== Phase 4E: Ensemble Fusion Gate ===\n")

    # 1. Load predictions
    logger.info("Loading model predictions...")
    all_preds = load_all_predictions()
    if len(all_preds) < 2:
        logger.error(f"Need >= 2 models, got {len(all_preds)}")
        return

    logger.info(f"\n{len(all_preds)} models loaded")

    # 2. Load returns
    logger.info("Loading returns...")
    cache = pd.read_parquet(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet",
                            columns=["__pnl_return_1d"])
    returns = cache["__pnl_return_1d"]

    # 3. Evaluate individual models
    logger.info("\nEvaluating individual models...")
    results = []
    for name, pred in all_preds.items():
        rics = daily_rank_ic(pred, returns)
        sp20 = daily_spread(pred, returns, 20)
        sp100 = daily_spread(pred, returns, 100)
        if rics:
            results.append({
                "model": name,
                "type": "single",
                "rank_ic": np.mean(rics),
                "rank_icir": np.mean(rics) / (np.std(rics) + 1e-8),
                "spread_top20": np.mean(sp20) * 1e4 if sp20 else 0,
                "spread_top100": np.mean(sp100) * 1e4 if sp100 else 0,
                "n_days": len(rics),
            })

    # 4. Fusion
    logger.info("\nComputing ensemble fusions...")

    # Use only latest pred.pkl per model type to avoid duplicates
    # Pick the most recent experiment per base name
    deduped = {}
    for name, pred in all_preds.items():
        base = name.split("_174_")[0] if "_174_" in name else name
        deduped[base] = pred
    logger.info(f"  Deduped to {len(deduped)} models: {list(deduped.keys())}")

    fusions = {
        "rank_mean": fuse_rank_mean(deduped),
        "robust_z_mean": fuse_robust_z_mean(deduped),
    }

    for fname, fused in fusions.items():
        fused_clean = fused.dropna()
        rics = daily_rank_ic(fused_clean, returns)
        sp20 = daily_spread(fused_clean, returns, 20)
        sp100 = daily_spread(fused_clean, returns, 100)
        if rics:
            results.append({
                "model": f"ensemble_{fname}",
                "type": "ensemble",
                "rank_ic": np.mean(rics),
                "rank_icir": np.mean(rics) / (np.std(rics) + 1e-8),
                "spread_top20": np.mean(sp20) * 1e4 if sp20 else 0,
                "spread_top100": np.mean(sp100) * 1e4 if sp100 else 0,
                "n_days": len(rics),
            })

    # 5. Disagreement stats
    disagreement = compute_model_disagreement(deduped)
    consensus = compute_model_consensus(deduped)
    logger.info(f"  Disagreement: mean={disagreement.mean():.3f}, std={disagreement.std():.3f}")
    logger.info(f"  Consensus (top 10%): mean={consensus.mean():.2f}")

    # 6. Print comparison
    df = pd.DataFrame(results).sort_values("rank_ic", ascending=False)
    print("\n" + "=" * 90)
    print("Phase 4E Ensemble Gate — Single Split (exploratory, NOT final gate)")
    print("=" * 90)
    print(df.to_string(index=False, float_format="%.4f"))

    # 7. Gate check
    best_single = df[df["type"] == "single"]["rank_ic"].max()
    best_ensemble = df[df["type"] == "ensemble"]["rank_ic"].max()
    best_single_spread = df[df["type"] == "single"]["spread_top20"].max()
    best_ensemble_spread = df[df["type"] == "ensemble"]["spread_top20"].max()

    ic_improvement = (best_ensemble / best_single - 1) if best_single > 0 else 0
    spread_improvement = (best_ensemble_spread / best_single_spread - 1) if best_single_spread > 0 else 0

    print(f"\nBest single RankIC: {best_single:.4f}")
    print(f"Best ensemble RankIC: {best_ensemble:.4f} ({ic_improvement:+.1%})")
    print(f"Best single spread20: {best_single_spread:.1f} bps")
    print(f"Best ensemble spread20: {best_ensemble_spread:.1f} bps ({spread_improvement:+.1%})")

    gate_pass = ic_improvement >= 0.05 and spread_improvement >= 0.10
    print(f"\nGate (IC >= +5%, Spread >= +10%): {'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        print("  NOTE: Single-split only. Full 24-split may differ.")
    print("=" * 90)

    # 8. Save artifact
    exp_id = f"ensemble_gate_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    art = ExperimentArtifact.create(
        experiment_id=exp_id,
        model_name="ensemble_fusion",
        feature_set="FS-174 multi-model",
        description="Ensemble fusion gate check (single split, exploratory)",
        models_used=list(deduped.keys()),
        fusion_methods=list(fusions.keys()),
    )
    best_row = df[df["type"] == "ensemble"].iloc[0].to_dict() if len(df[df["type"] == "ensemble"]) > 0 else {}
    art.save_metrics({
        "rank_ic_mean": best_row.get("rank_ic", 0),
        "rank_icir": best_row.get("rank_icir", 0),
        "spread_top20": best_row.get("spread_top20", 0),
        "spread_top100": best_row.get("spread_top100", 0),
        "ic_improvement_vs_best_single": ic_improvement,
        "spread_improvement_vs_best_single": spread_improvement,
        "gate_pass": gate_pass,
        "n_models": len(deduped),
        "disagreement_mean": float(disagreement.mean()),
    })
    logger.info(f"\nArtifact saved: {exp_id}")


if __name__ == "__main__":
    main()
