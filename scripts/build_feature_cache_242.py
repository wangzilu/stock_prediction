"""Pre-compute the production xgb_242 feature matrix and save as parquet.

The 174-family cache (``feature_cache_174_holder_regime_ma.parquet``)
that ``phase4e_24split_ensemble.py`` reads is a 205-column selection
that targets the legacy 174/175 lineage. xgb_242 needs a different
shape: ALL 158 Alpha158 columns + the 84 columns from
``PRODUCTION_SUPPLEMENTARY_GROUPS`` (fundamental, capital_flow,
macro_zero_baseline, shareholder, valuation, northbound, quality,
st_daily_basic, st_moneyflow, st_holder_number, cross_market_regime).

Without this cache there is no way to ask the 24-split runner the most
basic question the production model owes us:

    "Does xgb_242 — currently serving recommendations — actually
    beat 174 / 175 on the same 24-split exam?"

Output: ``data/storage/feature_cache_242_production.parquet`` with
roughly (~6M rows × 244 cols) including label + 1-day return.

Wall time: ~5 min (Alpha158 ~3 min + supp ~1 min + save ~30s) after
the 2026-06-05 _load_supplementary perf fix.

Usage:
    python scripts/build_feature_cache_242.py                # default window
    python scripts/build_feature_cache_242.py --end 2026-05-19  # match xgb175 5-26 baseline
"""
from __future__ import annotations

import argparse
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
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config.settings import PREDICTION_HORIZON_DAYS, DATA_DIR
from config.qlib_runtime import init_qlib
from config.production_features import (
    PRODUCTION_SUPPLEMENTARY_GROUPS, PRODUCTION_MODEL_PROFILE,
    PROFILE_EXPECTED_COUNTS,
)
from models.feature_merger import FeatureMerger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"
DAILY_RET_EXPR = "Ref($close, -1) / $close - 1"

DEFAULT_OUT_PATH = DATA_DIR / "feature_cache_242_production.parquet"


def main():
    from qlib.utils import init_instance_by_config
    from qlib.data import D
    from qlib.data.dataset.handler import DataHandlerLP as DK

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"),
                        help="End date (default: today). Use 2026-05-19 to match "
                             "the xgb175 5-26 baseline window.")
    parser.add_argument("--instruments", default="all",
                        help="Qlib instrument universe (default: all)")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH),
                        help="Output parquet path (default: "
                             "data/storage/feature_cache_242_production.parquet)")
    args = parser.parse_args()

    # Refuse silently overwriting the legacy 174-cache.
    out_path = Path(args.out).expanduser().resolve()
    legacy = (DATA_DIR / "feature_cache_174_holder_regime_ma.parquet").resolve()
    if out_path == legacy:
        raise SystemExit(
            f"Refusing to overwrite the 174-family cache at {legacy}. "
            f"Pass --out with a different filename (default: "
            f"feature_cache_242_production.parquet)."
        )

    logger.info("=" * 70)
    logger.info("Building the production xgb_242 feature cache")
    logger.info("=" * 70)
    logger.info("Period:        %s ~ %s", args.start, args.end)
    logger.info("Instruments:   %s", args.instruments)
    logger.info("Profile:       %s", PRODUCTION_MODEL_PROFILE)
    logger.info("Supp groups:   %s", PRODUCTION_SUPPLEMENTARY_GROUPS)
    logger.info("Expected dim:  %s", PROFILE_EXPECTED_COUNTS.get(PRODUCTION_MODEL_PROFILE))
    logger.info("Out path:      %s", out_path)

    init_qlib(QLIB_DATA)
    merger = FeatureMerger(DATA_DIR)

    t0 = time.time()

    # ── 1. Build the Alpha158 dataset with label ────────────────────
    logger.info("[1/4] Loading Alpha158 over %s ~ %s ...", args.start, args.end)
    t_alpha = time.time()
    dataset = init_instance_by_config({
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": args.start,
                    "end_time": args.end,
                    "instruments": args.instruments,
                    "label": [LABEL_EXPR],
                },
            },
            "segments": {"full": (args.start, args.end)},
        },
    })
    logger.info("    Alpha158 loaded in %.1fs", time.time() - t_alpha)

    # ── 2. Inject the 84 production supplementary cols ──────────────
    logger.info("[2/4] Injecting supplementary loaders: %s",
                list(PRODUCTION_SUPPLEMENTARY_GROUPS))
    t_supp = time.time()
    n_supp = merger.inject_supplementary_into_handler(
        dataset.handler,
        preprocess=False,
        groups=PRODUCTION_SUPPLEMENTARY_GROUPS,
    )
    logger.info("    Injected %d supp cols in %.1fs", n_supp, time.time() - t_supp)

    # ── 3. Materialise the full feature frame ────────────────────────
    logger.info("[3/4] Preparing the materialised feature frame ...")
    t_prep = time.time()
    X = dataset.prepare("full", col_set="feature", data_key=DK.DK_I)
    y = dataset.prepare("full", col_set="label", data_key=DK.DK_L)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    logger.info("    Feature frame: %s in %.1fs", X.shape, time.time() - t_prep)

    actual_dim = int(X.shape[1])
    expected = PROFILE_EXPECTED_COUNTS.get(PRODUCTION_MODEL_PROFILE, {}).get("total")
    if expected is not None and actual_dim != int(expected):
        # Don't kill the cache, but loudly note that the production
        # contract drift will surface before training.
        logger.warning(
            "[supp-dim] WARN: actual %d cols != PROFILE_EXPECTED_COUNTS total %d. "
            "The cache is still saved; train_lgb's contract gate will refuse "
            "to use it if the count is wrong at train time.",
            actual_dim, expected,
        )

    # Flatten the MultiIndex columns ((feature, name) → name) so the
    # parquet stores plain column names — easier for the runner to
    # consume without Qlib at read time.
    if isinstance(X.columns, pd.MultiIndex):
        X.columns = [
            col[1] if isinstance(col, tuple) and len(col) > 1 else str(col)
            for col in X.columns
        ]

    cache = X.copy()
    cache["__label_5d"] = y.values

    # ── 4. Attach daily return + save ────────────────────────────────
    logger.info("[4/4] Adding 1-day return + saving parquet ...")
    insts = sorted(set(str(c) for c in X.index.get_level_values(1)))
    dates = sorted(X.index.get_level_values(0).unique())
    t_ret = time.time()
    ret = D.features(
        insts, [DAILY_RET_EXPR],
        start_time=str(min(dates))[:10],
        end_time=str(max(dates))[:10],
    )
    if ret is not None and not ret.empty:
        ret.columns = ["__pnl_return_1d"]
        ret = ret.swaplevel().sort_index()
        ret = ret.replace([np.inf, -np.inf], np.nan)
        cache = cache.join(ret, how="left")
        logger.info(
            "    1-day return added in %.1fs (%d non-null)",
            time.time() - t_ret, int(ret.notna().sum().iloc[0]),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_save = time.time()
    cache.to_parquet(str(out_path))
    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info("    Saved in %.1fs (%.1f MB)", time.time() - t_save, size_mb)

    logger.info("")
    logger.info("=== xgb_242 cache built ===")
    logger.info("  Path:        %s", out_path)
    logger.info("  Shape:       %s", cache.shape)
    logger.info(
        "  Features:    %d",
        len([c for c in cache.columns if not c.startswith("__")]),
    )
    logger.info("  Labels:      %s", [c for c in cache.columns if c.startswith("__")])
    logger.info("  Date range:  %s ~ %s",
                str(min(dates))[:10], str(max(dates))[:10])
    logger.info("  Stocks:      %d", len(insts))
    logger.info("  Wall time:   %.1fs", time.time() - t0)
    logger.info("Done!")


if __name__ == "__main__":
    main()
