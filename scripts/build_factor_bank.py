"""Build Factor Bank — pre-computed rolling variants for AlphaForge AF-2.

Pre-computes time-series rolling features from the champion cache and saves
to a separate parquet. AlphaForge then searches COMBINATIONS of these
pre-computed columns (fast) instead of recomputing rolling windows (slow).

Bank structure:
  Base features (35 from Alpha158)
  × Rolling variants: mean/std/max/min/delta/sum/pctchange
  × Windows: 5, 10, 20
  = ~700 pre-computed columns

Usage:
    python scripts/build_factor_bank.py
"""
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
CACHE_PATH = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
BANK_PATH = DATA_DIR / "factor_bank.parquet"

# Base features to compute rolling variants for
BASE_FEATURES = [
    "KMID", "KLEN", "OPEN0", "HIGH0", "LOW0",
    "ROC5", "ROC10", "ROC20",
    "MA5", "MA10", "MA20",
    "STD5", "STD10", "STD20",
    "RSV5", "RSV10", "RSV20",
    "CORR5", "CORR10", "CORR20",
    "CNTP5", "CNTP10", "CNTP20",
    "VMA5", "VMA10", "VMA20",
    "VSTD5", "VSTD10", "VSTD20",
]

# Rolling operations to apply (fast vectorized ones only)
ROLLING_OPS = {
    "mean": lambda g, w: g.rolling(w, min_periods=max(1, w // 2)).mean(),
    "std": lambda g, w: g.rolling(w, min_periods=max(1, w // 2)).std(),
    "max": lambda g, w: g.rolling(w, min_periods=max(1, w // 2)).max(),
    "min": lambda g, w: g.rolling(w, min_periods=max(1, w // 2)).min(),
    "sum": lambda g, w: g.rolling(w, min_periods=max(1, w // 2)).sum(),
}

WINDOWS = [5, 10, 20]


def build_factor_bank():
    t0 = time.time()

    # Load base features + returns
    logger.info("Loading base cache...")
    cols = BASE_FEATURES + ["__pnl_return_1d", "__label_5d"]
    cache = pd.read_parquet(CACHE_PATH, columns=cols)
    logger.info(f"Cache: {cache.shape}")

    # Start with base features
    bank = cache.copy()

    # Add delta (diff) — fast, doesn't need rolling
    logger.info("Computing deltas...")
    for feat in BASE_FEATURES:
        for w in WINDOWS:
            col_name = f"{feat}_delta{w}"
            bank[col_name] = cache[feat].groupby(level=1).diff(w)

    # Add pct_change — fast
    logger.info("Computing pct_changes...")
    for feat in ["KLEN", "ROC5", "ROC20", "STD5", "STD20", "VMA20", "VSTD20"]:
        for w in [5, 10, 20]:
            col_name = f"{feat}_pctchg{w}"
            bank[col_name] = cache[feat].groupby(level=1).pct_change(w)

    # Add rolling operations
    for op_name, op_fn in ROLLING_OPS.items():
        logger.info(f"Computing rolling {op_name}...")
        # Only apply to a subset of features to keep bank manageable
        features_for_rolling = [
            "KLEN", "ROC5", "ROC20", "STD5", "STD20",
            "RSV5", "RSV20", "CORR5", "CORR20",
            "VMA20", "VSTD20", "CNTP20",
        ]
        for feat in features_for_rolling:
            for w in WINDOWS:
                col_name = f"{feat}_ts{op_name}{w}"
                bank[col_name] = cache[feat].groupby(level=1).transform(
                    lambda g: op_fn(g, w)
                )

    # Save
    n_new = len(bank.columns) - len(cols)
    logger.info(f"\nFactor Bank: {bank.shape} ({n_new} new columns)")
    logger.info(f"Saving to {BANK_PATH}...")
    bank.to_parquet(BANK_PATH)

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Summary
    groups = {"base": 0, "delta": 0, "pctchg": 0}
    for op in ROLLING_OPS:
        groups[f"ts{op}"] = 0
    groups["meta"] = 0

    for col in bank.columns:
        if col.startswith("__"):
            groups["meta"] += 1
        elif "_delta" in col:
            groups["delta"] += 1
        elif "_pctchg" in col:
            groups["pctchg"] += 1
        elif "_ts" in col:
            for op in ROLLING_OPS:
                if f"_ts{op}" in col:
                    groups[f"ts{op}"] += 1
                    break
        else:
            groups["base"] += 1

    logger.info("\nColumn groups:")
    for g, n in sorted(groups.items(), key=lambda x: -x[1]):
        logger.info(f"  {g:12s}: {n}")

    return bank


if __name__ == "__main__":
    build_factor_bank()
