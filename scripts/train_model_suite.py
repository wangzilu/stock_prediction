"""Train and compare multiple models: LGB, XGB, CatBoost, DoubleEnsemble, ALSTM, Transformer.

All models share the same Alpha158 dataset and evaluation pipeline.
Results saved to data/storage/model_suite_results.json.

Usage:
    python scripts/train_model_suite.py                    # tree models only
    python scripts/train_model_suite.py --include-deep     # tree + deep models (MPS)
    python scripts/train_model_suite.py --models lgb xgb   # specific models
"""
import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config.settings import PREDICTION_HORIZON_DAYS, LGB_MIN_PREDICTIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
RESULTS_PATH = DATA_DIR / "model_suite_results.json"

LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"

# Model configs
MODEL_CONFIGS = {
    "lgb": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.05,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 4,
        },
    },
    "xgb": {
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.xgboost",
        "kwargs": {
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8789,
            "colsample_bytree": 0.8879,
            "reg_alpha": 205.6999,
            "reg_lambda": 580.9768,
            "n_jobs": 4,
        },
    },
    "catboost": {
        "class": "CatBoostModel",
        "module_path": "qlib.contrib.model.catboost_model",
        "kwargs": {
            "loss_function": "RMSE",
            "iterations": 500,
            "depth": 8,
            "learning_rate": 0.05,
        },
    },
    "double_ensemble": {
        "class": "DEnsembleModel",
        "module_path": "qlib.contrib.model.double_ensemble",
        "kwargs": {
            "base_model": "gbm",
            "loss": "mse",
            "num_models": 6,
            "enable_sr": True,
            "enable_fs": True,
            "alpha1": 1.0,
            "alpha2": 1.0,
            "bins_sr": 10,
            "bins_fs": 5,
        },
    },
    "alstm": {
        "class": "ALSTM",
        "module_path": "qlib.contrib.model.pytorch_alstm",
        "kwargs": {
            "d_feat": 158,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "n_epochs": 50,
            "lr": 1e-3,
            "batch_size": 2048,
            "early_stop": 10,
            "optimizer": "adam",
            "GPU": None,  # Qlib ALSTM doesn't support MPS string; use CPU
        },
    },
    "transformer": {
        "class": "DNNModelPytorch",
        "module_path": "qlib.contrib.model.pytorch_nn",
        "kwargs": {
            "lr": 1e-3,
            "max_steps": 2000,
            "batch_size": 2048,
            "early_stop_rounds": 10,
            "eval_steps": 50,
            "optimizer": "adam",
            "loss": "mse",
            "GPU": None,  # CPU; Qlib pytorch_nn doesn't support MPS
            "pt_model_uri": "qlib.contrib.model.pytorch_transformer_ts.TransformerModel",
            "pt_model_kwargs": {
                "d_feat": 158,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.1,
            },
        },
    },
}

TREE_MODELS = ["lgb", "xgb", "catboost", "double_ensemble"]
DEEP_MODELS = ["alstm", "transformer"]


def build_dataset():
    """Build shared Alpha158 dataset."""
    import qlib
    from qlib.constant import REG_CN
    from qlib.utils import init_instance_by_config

    qlib.init(provider_uri=QLIB_DATA, region=REG_CN)

    today = datetime.now()
    train_start = (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=89)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")

    logger.info(f"Train: {train_start}~{train_end}, Valid: {valid_start}~{valid_end}, Test: {test_start}~{test_end}")

    handler_config = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": train_start,
            "end_time": test_end,
            "instruments": "all",
            "label": [LABEL_EXPR],
        },
    }
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (test_start, test_end),
            },
        },
    }

    logger.info("Loading Alpha158 dataset (all A-shares)...")
    dataset = init_instance_by_config(dataset_config)
    return dataset, {"train_start": train_start, "test_end": test_end}


def evaluate_model(model, dataset, model_name):
    """Evaluate a model's predictions with IC/RankIC/TopK analysis."""
    from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return

    logger.info(f"Evaluating {model_name}...")
    pred = model.predict(dataset=dataset)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")

    label = dataset.prepare("test", col_set="label")
    if isinstance(label, pd.DataFrame):
        label = label.iloc[:, 0]

    common = pred.index.intersection(label.index)
    pred_s = pred.loc[common, "score"] if "score" in pred.columns else pred.loc[common].iloc[:, 0]
    label_s = label.loc[common]

    mask = pred_s.notna() & label_s.notna() & np.isfinite(pred_s) & np.isfinite(label_s)
    pred_clean = pred_s[mask]
    label_clean = label_s[mask]

    if len(pred_clean) < 100:
        return {"model": model_name, "error": f"Too few samples: {len(pred_clean)}"}

    ic, rank_ic = calc_ic(pred_clean, label_clean)

    # TopK analysis
    df = pd.DataFrame({"pred": pred_clean, "label": label_clean})
    spreads = []
    for date, g in df.groupby(level=0):
        if len(g) < 40:
            continue
        s = g.sort_values("pred", ascending=False)
        top = s.head(20)["label"].mean()
        bot = s.tail(20)["label"].mean()
        spreads.append(top - bot)

    return {
        "model": model_name,
        "n_samples": len(pred_clean),
        "n_dates": len(ic),
        "ic_mean": round(float(ic.mean()), 6),
        "ic_std": round(float(ic.std()), 6),
        "icir": round(float(ic.mean() / (ic.std() + 1e-8)), 4),
        "rank_ic_mean": round(float(rank_ic.mean()), 6),
        "rank_ic_pos_ratio": round(float((rank_ic > 0).mean()), 4),
        "top20_bot20_spread": round(float(np.mean(spreads)), 6) if spreads else 0.0,
        "spread_pos_ratio": round(float(np.mean([s > 0 for s in spreads])), 4) if spreads else 0.0,
    }


def train_deep_model(model_name, dataset):
    """Train a PyTorch deep model with MPS support (bypasses Qlib's deep model)."""
    from models.deep_models import DeepModel, ALSTMNet, TransformerNet

    net_map = {
        "alstm": (ALSTMNet, {"d_feat": 158, "hidden_size": 64, "num_layers": 2, "dropout": 0.1}),
        "transformer": (TransformerNet, {"d_feat": 158, "d_model": 64, "nhead": 4, "num_layers": 2, "dropout": 0.1}),
    }

    if model_name not in net_map:
        return {"model": model_name, "error": f"Unknown deep model: {model_name}"}, None

    net_class, net_kwargs = net_map[model_name]

    logger.info(f"Training {model_name} (PyTorch native, MPS)...")
    t0 = time.time()

    try:
        # Extract numpy arrays from Qlib dataset
        X_train = dataset.prepare("train", col_set="feature").values
        y_train = dataset.prepare("train", col_set="label").values.ravel()
        X_valid = dataset.prepare("valid", col_set="feature").values
        y_valid = dataset.prepare("valid", col_set="label").values.ravel()

        dm = DeepModel(
            net_class=net_class,
            net_kwargs=net_kwargs,
            n_epochs=50,
            lr=1e-3,
            batch_size=2048,
            early_stop=10,
        )
        dm.fit(X_train, y_train, X_valid, y_valid)

        train_time = time.time() - t0
        logger.info(f"{model_name} trained in {train_time:.0f}s on {dm.device}")

        # Predict on test for evaluation
        X_test = dataset.prepare("test", col_set="feature").values
        test_preds = dm.predict(X_test)

        # Build prediction Series with proper index
        test_label = dataset.prepare("test", col_set="label")
        if isinstance(test_label, pd.DataFrame):
            test_label = test_label.iloc[:, 0]

        pred_series = pd.Series(test_preds, index=test_label.index, name="score")

        # Evaluate
        result = _evaluate_predictions(pred_series, test_label, model_name)
        result["train_time_s"] = round(train_time, 1)
        result["device"] = str(dm.device)

        # Save
        model_path = DATA_DIR / f"{model_name}_model.pt"
        import torch
        torch.save({"model_state": dm.model.state_dict(), "net_kwargs": net_kwargs}, model_path)
        result["model_path"] = str(model_path)

        return result, pred_series
    except Exception as e:
        logger.error(f"{model_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return {"model": model_name, "error": str(e), "train_time_s": round(time.time() - t0, 1)}, None


def _evaluate_predictions(pred, label, model_name):
    """Evaluate prediction Series against label Series."""
    from qlib.contrib.eva.alpha import calc_ic

    common = pred.index.intersection(label.index)
    pred_s = pred.loc[common]
    label_s = label.loc[common]

    mask = pred_s.notna() & label_s.notna() & np.isfinite(pred_s) & np.isfinite(label_s)
    pred_clean = pred_s[mask]
    label_clean = label_s[mask]

    if len(pred_clean) < 100:
        return {"model": model_name, "error": f"Too few samples: {len(pred_clean)}"}

    ic, rank_ic = calc_ic(pred_clean, label_clean)

    df = pd.DataFrame({"pred": pred_clean, "label": label_clean})
    spreads = []
    for date, g in df.groupby(level=0):
        if len(g) < 40:
            continue
        s = g.sort_values("pred", ascending=False)
        spreads.append(s.head(20)["label"].mean() - s.tail(20)["label"].mean())

    return {
        "model": model_name,
        "n_samples": len(pred_clean),
        "n_dates": len(ic),
        "ic_mean": round(float(ic.mean()), 6),
        "ic_std": round(float(ic.std()), 6),
        "icir": round(float(ic.mean() / (ic.std() + 1e-8)), 4),
        "rank_ic_mean": round(float(rank_ic.mean()), 6),
        "rank_ic_pos_ratio": round(float((rank_ic > 0).mean()), 4),
        "top20_bot20_spread": round(float(np.mean(spreads)), 6) if spreads else 0.0,
        "spread_pos_ratio": round(float(np.mean([s > 0 for s in spreads])), 4) if spreads else 0.0,
    }


def train_and_evaluate(model_name, dataset):
    """Train one model and evaluate it."""
    # Deep models use PyTorch native (MPS support)
    if model_name in DEEP_MODELS:
        result, pred = train_deep_model(model_name, dataset)
        return result, pred  # pred is a Series, not a model

    from qlib.utils import init_instance_by_config

    config = MODEL_CONFIGS[model_name]
    logger.info(f"Training {model_name}...")
    t0 = time.time()

    try:
        model = init_instance_by_config(config)
        model.fit(dataset)
        train_time = time.time() - t0
        logger.info(f"{model_name} trained in {train_time:.0f}s")

        result = evaluate_model(model, dataset, model_name)
        result["train_time_s"] = round(train_time, 1)

        # Save model
        model_path = DATA_DIR / f"{model_name}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        result["model_path"] = str(model_path)

        return result, model
    except Exception as e:
        logger.error(f"{model_name} failed: {e}")
        return {"model": model_name, "error": str(e), "train_time_s": round(time.time() - t0, 1)}, None


def rank_ensemble(models_preds, dataset):
    """Create rank-weighted ensemble of multiple models."""
    from qlib.contrib.eva.alpha import calc_ic

    label = dataset.prepare("test", col_set="label")
    if isinstance(label, pd.DataFrame):
        label = label.iloc[:, 0]

    # Collect all predictions and rank-normalize
    all_ranks = []
    weights = {"lgb": 0.40, "xgb": 0.25, "catboost": 0.25, "double_ensemble": 0.10}

    for name, pred in models_preds.items():
        if isinstance(pred, pd.Series):
            pred = pred.to_frame("score")
        col = "score" if "score" in pred.columns else pred.columns[0]
        ranked = pred[col].groupby(level=0).rank(pct=True)
        w = weights.get(name, 1.0 / len(models_preds))
        all_ranks.append(ranked * w)

    if not all_ranks:
        return {"model": "ensemble", "error": "No predictions to ensemble"}

    ensemble = sum(all_ranks)
    ensemble.name = "score"

    common = ensemble.index.intersection(label.index)
    pred_clean = ensemble.loc[common]
    label_clean = label.loc[common]

    mask = pred_clean.notna() & label_clean.notna() & np.isfinite(pred_clean) & np.isfinite(label_clean)
    pred_clean = pred_clean[mask]
    label_clean = label_clean[mask]

    if len(pred_clean) < 100:
        return {"model": "ensemble", "error": f"Too few samples: {len(pred_clean)}"}

    ic, rank_ic = calc_ic(pred_clean, label_clean)

    df = pd.DataFrame({"pred": pred_clean, "label": label_clean})
    spreads = []
    for date, g in df.groupby(level=0):
        if len(g) < 40:
            continue
        s = g.sort_values("pred", ascending=False)
        spreads.append(s.head(20)["label"].mean() - s.tail(20)["label"].mean())

    return {
        "model": "rank_ensemble",
        "components": list(models_preds.keys()),
        "n_samples": len(pred_clean),
        "n_dates": len(ic),
        "ic_mean": round(float(ic.mean()), 6),
        "icir": round(float(ic.mean() / (ic.std() + 1e-8)), 4),
        "rank_ic_mean": round(float(rank_ic.mean()), 6),
        "rank_ic_pos_ratio": round(float((rank_ic > 0).mean()), 4),
        "top20_bot20_spread": round(float(np.mean(spreads)), 6) if spreads else 0.0,
        "spread_pos_ratio": round(float(np.mean([s > 0 for s in spreads])), 4) if spreads else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and compare model suite")
    parser.add_argument("--include-deep", action="store_true", help="Include ALSTM/Transformer (needs MPS)")
    parser.add_argument("--models", nargs="+", default=None, help="Specific models to train")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.models:
        model_names = args.models
    elif args.include_deep:
        model_names = TREE_MODELS + DEEP_MODELS
    else:
        model_names = TREE_MODELS

    logger.info(f"Models to train: {model_names}")

    dataset, date_info = build_dataset()

    results = []
    models_preds = {}

    for name in model_names:
        if name not in MODEL_CONFIGS:
            logger.warning(f"Unknown model: {name}, skipping")
            continue
        result, model = train_and_evaluate(name, dataset)
        results.append(result)

        # Collect predictions for ensemble
        if "error" not in result:
            if name in DEEP_MODELS:
                # Deep models return predictions directly as Series
                if model is not None:
                    models_preds[name] = model  # already a Series
            elif model is not None:
                pred = model.predict(dataset=dataset)
                models_preds[name] = pred

        # Print result
        if "error" not in result:
            logger.info(
                f"  {name}: IC={result['ic_mean']:.4f} ICIR={result['icir']:.3f} "
                f"RankIC={result['rank_ic_mean']:.4f} Spread={result['top20_bot20_spread']*100:.3f}% "
                f"({result['train_time_s']:.0f}s)"
            )

    # Rank ensemble (tree models only)
    tree_preds = {k: v for k, v in models_preds.items() if k in TREE_MODELS}
    if len(tree_preds) >= 2:
        logger.info("Building rank ensemble...")
        ens_result = rank_ensemble(tree_preds, dataset)
        results.append(ens_result)
        if "error" not in ens_result:
            logger.info(
                f"  ensemble: IC={ens_result['ic_mean']:.4f} ICIR={ens_result['icir']:.3f} "
                f"RankIC={ens_result['rank_ic_mean']:.4f} Spread={ens_result['top20_bot20_spread']*100:.3f}%"
            )

    # Save results
    output = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "date_info": date_info,
        "label_expression": LABEL_EXPR,
        "models": results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    os.replace(tmp, RESULTS_PATH)

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        # Summary table
        print("\n" + "=" * 80)
        print(f"{'Model':<20} {'IC':>8} {'ICIR':>8} {'RankIC':>8} {'Spread%':>10} {'Time':>8}")
        print("-" * 80)
        for r in results:
            if "error" in r:
                print(f"{r['model']:<20} ERROR: {r['error'][:50]}")
            else:
                print(
                    f"{r['model']:<20} {r['ic_mean']:>8.4f} {r.get('icir',0):>8.3f} "
                    f"{r['rank_ic_mean']:>8.4f} {r['top20_bot20_spread']*100:>9.3f}% "
                    f"{r.get('train_time_s',0):>7.0f}s"
                )
        print("=" * 80)


if __name__ == "__main__":
    main()
