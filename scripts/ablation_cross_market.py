"""Ablation: 174-dim vs 174+cross_market_regime (174+27 regime features).

Tests whether HSI/HSTECH/NASDAQ regime signals improve alpha.

Usage:
    python scripts/ablation_cross_market.py
"""
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
from models.feature_merger import FeatureMerger
from models.feature_pipeline import (
    prepare_features_174, prepare_segment_numpy, train_xgb, evaluate_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"


def get_trading_dates():
    from qlib.data import D
    cal = D.calendar(start_time="2020-01-01", end_time=datetime.now().strftime("%Y-%m-%d"))
    return sorted(cal)


def main():
    import xgboost as xgb
    from qlib.utils import init_instance_by_config

    init_qlib(QLIB_DATA)
    merger = FeatureMerger(DATA_DIR)

    trade_dates = get_trading_dates()
    today_idx = len(trade_dates) - 1

    n_splits = 12
    test_days = 20
    train_years = 3
    valid_days = 60

    all_results = []

    for split_idx in range(n_splits):
        test_end_idx = today_idx - split_idx * test_days
        test_start_idx = test_end_idx - test_days
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - valid_days
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - (train_years * 250)

        if train_start_idx < 0:
            break

        test_end = str(trade_dates[test_end_idx])[:10]
        test_start = str(trade_dates[test_start_idx])[:10]
        valid_end = str(trade_dates[valid_end_idx])[:10]
        valid_start = str(trade_dates[valid_start_idx])[:10]
        train_end = str(trade_dates[train_end_idx])[:10]
        train_start = str(trade_dates[train_start_idx])[:10]

        logger.info(f"\nSplit {split_idx+1}/{n_splits}: test {test_start}~{test_end}")

        try:
            dataset = init_instance_by_config({
                "class": "DatasetH", "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
                        "kwargs": {"start_time": train_start, "end_time": test_end,
                                   "instruments": "all", "label": [LABEL_EXPR]},
                    },
                    "segments": {
                        "train": (train_start, train_end),
                        "valid": (valid_start, valid_end),
                        "test": (test_start, test_end),
                    },
                },
            })

            # --- 174 base (no cross-market) ---
            # Temporarily disable cross-market loading
            orig_loader = merger._load_cross_market_regime
            merger._load_cross_market_regime = lambda idx: None

            segs_base = {}
            for seg in ["train", "valid", "test"]:
                X, y = prepare_features_174(dataset, seg, merger)
                Xn = X.values.astype(np.float32)
                yn = y.values.astype(np.float32)
                mask = np.isfinite(yn)
                segs_base[seg] = (Xn[mask], yn[mask], X.index[mask])

            # Restore loader
            merger._load_cross_market_regime = orig_loader

            t0 = time.time()
            model_base = train_xgb(segs_base["train"][0], segs_base["train"][1],
                                   segs_base["valid"][0], segs_base["valid"][1])
            pred_base = model_base.predict(xgb.DMatrix(segs_base["test"][0]))
            m_base = evaluate_predictions(pred_base, segs_base["test"][1], segs_base["test"][2])
            n_base = segs_base["train"][0].shape[1]

            # --- 174 + cross-market regime ---
            segs_regime = {}
            for seg in ["train", "valid", "test"]:
                X, y = prepare_features_174(dataset, seg, merger)
                # Manually add cross-market regime features
                cross_mkt = merger._load_cross_market_regime(X.index)
                if cross_mkt is not None and not cross_mkt.empty:
                    X = X.join(cross_mkt, how="left")
                Xn = X.values.astype(np.float32)
                yn = y.values.astype(np.float32)
                mask = np.isfinite(yn)
                segs_regime[seg] = (Xn[mask], yn[mask], X.index[mask])

            model_regime = train_xgb(segs_regime["train"][0], segs_regime["train"][1],
                                     segs_regime["valid"][0], segs_regime["valid"][1])
            pred_regime = model_regime.predict(xgb.DMatrix(segs_regime["test"][0]))
            m_regime = evaluate_predictions(pred_regime, segs_regime["test"][1], segs_regime["test"][2])
            n_regime = segs_regime["train"][0].shape[1]

            result = {
                "split": split_idx + 1,
                "test": f"{test_start}~{test_end}",
                "base": {"n_feat": n_base, **m_base},
                "regime": {"n_feat": n_regime, **m_regime},
                "delta_rank_ic": round(m_regime["rank_ic_mean"] - m_base["rank_ic_mean"], 6),
                "delta_spread": round(m_regime["top20_spread"] - m_base["top20_spread"], 6),
            }
            all_results.append(result)

            logger.info(f"  base({n_base}):   RankIC={m_base['rank_ic_mean']:+.4f} "
                        f"Spread={m_base['top20_spread']*100:+.3f}%")
            logger.info(f"  +regime({n_regime}): RankIC={m_regime['rank_ic_mean']:+.4f} "
                        f"Spread={m_regime['top20_spread']*100:+.3f}%")
            logger.info(f"  Δ RankIC={result['delta_rank_ic']:+.4f} "
                        f"Δ Spread={result['delta_spread']*100:+.3f}%")

        except Exception as e:
            logger.error(f"  Split {split_idx+1} failed: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        logger.error("No results")
        sys.exit(1)

    # Summary
    n = len(all_results)
    base_rics = [r["base"]["rank_ic_mean"] for r in all_results]
    regime_rics = [r["regime"]["rank_ic_mean"] for r in all_results]
    base_sprs = [r["base"]["top20_spread"] for r in all_results]
    regime_sprs = [r["regime"]["top20_spread"] for r in all_results]
    delta_rics = [r["delta_rank_ic"] for r in all_results]
    delta_sprs = [r["delta_spread"] for r in all_results]

    logger.info(f"\n{'='*70}")
    logger.info(f"CROSS-MARKET REGIME ABLATION ({n} splits)")
    logger.info(f"{'='*70}")
    logger.info(f"  base avg RankIC:   {np.mean(base_rics):+.4f}  Spread: {np.mean(base_sprs)*100:+.3f}%")
    logger.info(f"  +regime avg RankIC: {np.mean(regime_rics):+.4f}  Spread: {np.mean(regime_sprs)*100:+.3f}%")
    logger.info(f"  Δ RankIC>0: {sum(1 for d in delta_rics if d > 0)}/{n} "
                f"({sum(1 for d in delta_rics if d > 0)/n:.0%})")
    logger.info(f"  Δ Spread>0: {sum(1 for d in delta_sprs if d > 0)}/{n} "
                f"({sum(1 for d in delta_sprs if d > 0)/n:.0%})")

    # Save
    out_path = DATA_DIR / "phase4" / "ablation_cross_market.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump({"evaluated_at": datetime.now().isoformat(timespec="seconds"),
                    "n_splits": n, "splits": all_results}, f, indent=2)
    logger.info(f"Saved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
