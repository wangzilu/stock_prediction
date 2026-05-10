"""Compare Qlib data processors: default vs RobustZScoreNorm vs CSRankNorm.

Tests if different feature normalization improves XGB IC.
All comparisons in same script, same seed, same split.

Usage:
    python scripts/experiment_processors.py
    python scripts/experiment_processors.py --universe csi300
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config.settings import PREDICTION_HORIZON_DAYS
from config.qlib_runtime import init_qlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
RESULTS_PATH = DATA_DIR / "processor_experiment_results.json"
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"
SEED = 42


def train_and_eval(dataset, model_config, segment="test"):
    """Train XGB and evaluate IC/Spread on a dataset."""
    from qlib.utils import init_instance_by_config
    from qlib.contrib.eva.alpha import calc_ic

    model = init_instance_by_config(model_config)
    model.fit(dataset)

    pred = model.predict(dataset=dataset)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")

    label = dataset.prepare(segment, col_set="label")
    if isinstance(label, pd.DataFrame):
        label = label.iloc[:, 0]

    common = pred.index.intersection(label.index)
    p = pred.loc[common, "score"] if "score" in pred.columns else pred.loc[common].iloc[:, 0]
    l = label.loc[common]
    mask = p.notna() & l.notna() & np.isfinite(p) & np.isfinite(l)
    p, l = p[mask], l[mask]

    if len(p) < 100:
        return {"error": f"Too few samples: {len(p)}"}

    ic, ric = calc_ic(p, l)
    df = pd.DataFrame({"pred": p, "label": l})
    spreads = [g.sort_values("pred", ascending=False).head(20)["label"].mean() -
               g.sort_values("pred", ascending=False).tail(20)["label"].mean()
               for _, g in df.groupby(level=0) if len(g) >= 40]

    return {
        "ic_mean": round(float(ic.mean()), 6),
        "icir": round(float(ic.mean() / (ic.std() + 1e-8)), 4),
        "rank_ic_mean": round(float(ric.mean()), 6),
        "top20_spread": round(float(np.mean(spreads)) if spreads else 0, 6),
        "n_samples": len(p),
    }


def main():
    parser = argparse.ArgumentParser(description="Processor comparison experiment")
    parser.add_argument("--universe", default="csi300")
    args = parser.parse_args()

    init_qlib(QLIB_DATA)
    from qlib.utils import init_instance_by_config

    today = datetime.now()
    train_start = (today - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=89)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")

    model_config = {
        "class": "XGBModel", "module_path": "qlib.contrib.model.xgboost",
        "kwargs": {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05,
                   "subsample": 0.8789, "colsample_bytree": 0.8879,
                   "reg_alpha": 205.7, "reg_lambda": 580.98, "n_jobs": 4, "seed": SEED},
    }

    # Define processor variants to test
    processor_configs = {
        "default (Alpha158)": {
            # Alpha158 default: DropnaLabel + CSZScoreNorm on label only
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ],
            "infer_processors": [],
        },
        "feature_CSZScoreNorm": {
            # Add CSZScoreNorm on features too
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ],
            "infer_processors": [
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}},
            ],
        },
        "feature_RobustZScoreNorm": {
            # RobustZScoreNorm on features (median-based, resistant to outliers)
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature"}},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ],
            "infer_processors": [
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature"}},
            ],
        },
        "feature_CSRankNorm": {
            # CSRankNorm: cross-sectional rank normalization
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "feature"}},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ],
            "infer_processors": [
                {"class": "CSRankNorm", "kwargs": {"fields_group": "feature"}},
            ],
        },
    }

    results = []
    for proc_name, proc_config in processor_configs.items():
        logger.info(f"Testing: {proc_name}...")
        t0 = time.time()
        try:
            handler_config = {
                "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": train_start, "end_time": test_end,
                    "instruments": args.universe, "label": [LABEL_EXPR],
                    **proc_config,
                },
            }
            dataset_config = {
                "class": "DatasetH", "module_path": "qlib.data.dataset",
                "kwargs": {"handler": handler_config, "segments": {
                    "train": (train_start, train_end),
                    "valid": (valid_start, valid_end),
                    "test": (test_start, test_end),
                }},
            }
            dataset = init_instance_by_config(dataset_config)
            metrics = train_and_eval(dataset, model_config)
            metrics["processor"] = proc_name
            metrics["time_s"] = round(time.time() - t0, 1)
            results.append(metrics)

            if "error" not in metrics:
                logger.info(f"  IC={metrics['ic_mean']:+.4f} RankIC={metrics['rank_ic_mean']:+.4f} Spread={metrics['top20_spread']*100:+.3f}%")
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append({"processor": proc_name, "error": str(e)})

    # Save and print
    output = {"experiment_at": datetime.now().isoformat(timespec="seconds"), "results": results}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    os.replace(tmp, RESULTS_PATH)

    print(f"\n{'='*70}")
    print(f"{'Processor':<30} {'IC':>8} {'RankIC':>8} {'Spread%':>9}")
    print(f"{'-'*70}")
    for r in results:
        if "error" in r:
            print(f"{r['processor']:<30} ERROR: {r['error'][:35]}")
        else:
            print(f"{r['processor']:<30} {r['ic_mean']:>+8.4f} {r['rank_ic_mean']:>+8.4f} {r['top20_spread']*100:>+8.3f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
