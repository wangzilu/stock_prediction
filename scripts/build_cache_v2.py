"""Cache V2: Partitioned storage with lazy loading.

Splits monolithic 3.7G parquet into:
- date-partitioned blocks (yearly)
- feature-group vertical splits (base174, regime, holder, ma, labels)
- All float32

Loading a single split (750d train + 60d valid + 20d test) reads only
the relevant partitions and columns, not the full 3.7G.

Usage:
    python scripts/build_cache_v2.py              # build from existing cache
    python scripts/build_cache_v2.py --rebuild    # rebuild from Qlib
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
CACHE_V2_DIR = DATA_DIR / "cache_v2"

# Feature group definitions
FEATURE_GROUPS = {
    "base158": lambda cols: [c for c in cols if not c.startswith(("flow_", "pe", "pb", "turn_",
                             "amount_", "ep", "bp", "price_", "hsi_", "hstech_", "nasdaq_",
                             "holder_num")) and not c.startswith("__") and not c.startswith("_")],
    "custom": lambda cols: [c for c in cols if c in ("pe", "pb", "turn_raw", "amount_raw",
                            "pe_mom20", "pb_mom20", "turn_anom20", "turn_anom60",
                            "amount_anom20", "turn_vol20", "ep", "bp", "price_pos20")],
    "flow": lambda cols: [c for c in cols if c.startswith("flow_")],
    "holder": lambda cols: [c for c in cols if c == "holder_num"],
    "regime": lambda cols: [c for c in cols if c.startswith(("hsi_", "hstech_", "nasdaq_"))],
    "ma": lambda cols: [c for c in cols if c.startswith("_") and not c.startswith("__")],
    "labels": lambda cols: [c for c in cols if c.startswith("__")],
}


def build_from_existing_cache(source_path: Path):
    """Convert monolithic cache to partitioned V2."""
    logger.info(f"Loading source: {source_path}")
    t0 = time.time()
    df = pd.read_parquet(str(source_path))
    logger.info(f"  Shape: {df.shape}, {time.time()-t0:.1f}s")

    all_cols = list(df.columns)

    # Create output directory
    CACHE_V2_DIR.mkdir(parents=True, exist_ok=True)

    # Save row index (date, instrument mapping)
    logger.info("Saving row index...")
    dates = df.index.get_level_values(0)
    insts = df.index.get_level_values(1)

    # Convert dates to integer day offset for fast slicing
    unique_dates = sorted(dates.unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    date_indices = np.array([date_to_idx[d] for d in dates], dtype=np.int32)

    # Save index metadata
    index_meta = {
        "n_rows": len(df),
        "n_dates": len(unique_dates),
        "n_instruments": len(insts.unique()),
        "date_range": [str(unique_dates[0])[:10], str(unique_dates[-1])[:10]],
        "dates": [str(d)[:10] for d in unique_dates],
        "feature_groups": {},
    }

    # Save date_indices for fast slicing
    np.save(str(CACHE_V2_DIR / "date_indices.npy"), date_indices)

    # Save instrument codes
    inst_codes = insts.astype(str).values
    np.save(str(CACHE_V2_DIR / "instruments.npy"), inst_codes)

    # Save each feature group as separate parquet
    for group_name, col_selector in FEATURE_GROUPS.items():
        cols = col_selector(all_cols)
        if not cols:
            logger.info(f"  {group_name}: no columns, skip")
            continue

        group_data = df[cols].values.astype(np.float32)
        group_path = CACHE_V2_DIR / f"{group_name}.npy"
        np.save(str(group_path), group_data)

        size_mb = group_path.stat().st_size / 1024 / 1024
        index_meta["feature_groups"][group_name] = {
            "columns": cols,
            "n_cols": len(cols),
            "size_mb": round(size_mb, 1),
        }
        logger.info(f"  {group_name}: {len(cols)} cols, {size_mb:.1f} MB")

    # Save metadata
    with open(str(CACHE_V2_DIR / "meta.json"), "w") as f:
        json.dump(index_meta, f, indent=2, ensure_ascii=False)

    total_size = sum(f.stat().st_size for f in CACHE_V2_DIR.iterdir()) / 1024 / 1024
    logger.info(f"\nCache V2 saved: {CACHE_V2_DIR}")
    logger.info(f"  Total size: {total_size:.1f} MB")
    logger.info(f"  Source was: {source_path.stat().st_size / 1024 / 1024:.1f} MB")


class CacheV2Reader:
    """Fast reader for Cache V2 — loads only what's needed."""

    def __init__(self, cache_dir: Path = CACHE_V2_DIR):
        self.cache_dir = Path(cache_dir)
        with open(str(self.cache_dir / "meta.json")) as f:
            self.meta = json.load(f)
        self.date_indices = np.load(str(self.cache_dir / "date_indices.npy"))
        self.dates = self.meta["dates"]
        self._arrays = {}  # lazy loaded

    def _load_group(self, group: str) -> np.ndarray:
        if group not in self._arrays:
            path = self.cache_dir / f"{group}.npy"
            self._arrays[group] = np.load(str(path), mmap_mode="r")  # memory-mapped!
        return self._arrays[group]

    def get_date_range_mask(self, start_date: str, end_date: str) -> np.ndarray:
        """Get boolean mask for rows in date range (fast, uses integer index)."""
        start_idx = next((i for i, d in enumerate(self.dates) if d >= start_date), 0)
        end_idx = next((i for i, d in enumerate(self.dates) if d > end_date), len(self.dates))
        return (self.date_indices >= start_idx) & (self.date_indices < end_idx)

    def load_split(self, train_start: str, train_end: str,
                   valid_start: str, valid_end: str,
                   test_start: str, test_end: str,
                   groups: list[str] = None,
                   ) -> dict:
        """Load one split's data. Only reads needed groups.

        Returns dict with keys: X_train, y_train, X_valid, y_valid,
                                X_test, y_test, test_instruments
        """
        if groups is None:
            groups = ["base158", "custom", "flow"]  # default: 174 base

        # Get masks
        tm = self.get_date_range_mask(train_start, train_end)
        vm = self.get_date_range_mask(valid_start, valid_end)
        em = self.get_date_range_mask(test_start, test_end)

        # Load and concat feature groups
        arrays_tr, arrays_va, arrays_te = [], [], []
        for g in groups:
            arr = self._load_group(g)
            arrays_tr.append(arr[tm])
            arrays_va.append(arr[vm])
            arrays_te.append(arr[em])

        X_tr = np.hstack(arrays_tr) if len(arrays_tr) > 1 else arrays_tr[0].copy()
        X_va = np.hstack(arrays_va) if len(arrays_va) > 1 else arrays_va[0].copy()
        X_te = np.hstack(arrays_te) if len(arrays_te) > 1 else arrays_te[0].copy()

        # Load labels
        labels = self._load_group("labels")
        y_tr = labels[tm, 0].astype(np.float32)  # column 0 = __label_5d
        y_va = labels[vm, 0].astype(np.float32)
        y_te = labels[em, 0].astype(np.float32)

        # Load instruments for test set
        insts = np.load(str(self.cache_dir / "instruments.npy"), allow_pickle=True)
        test_insts = insts[em]

        # NaN filter
        mtr = np.isfinite(y_tr)
        mva = np.isfinite(y_va)
        mte = np.isfinite(y_te)

        return {
            "X_train": X_tr[mtr], "y_train": y_tr[mtr],
            "X_valid": X_va[mva], "y_valid": y_va[mva],
            "X_test": X_te[mte], "y_test": y_te[mte],
            "test_dates": self.date_indices[em][mte],
            "test_instruments": test_insts[mte],
            "n_features": X_tr.shape[1],
        }

    def get_feature_cols(self, groups: list[str]) -> list[str]:
        """Get column names for given groups."""
        cols = []
        for g in groups:
            if g in self.meta["feature_groups"]:
                cols.extend(self.meta["feature_groups"][g]["columns"])
        return cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="feature_cache_174_holder_regime_ma.parquet")
    parser.add_argument("--test", action="store_true", help="Test loading speed after build")
    args = parser.parse_args()

    source_path = DATA_DIR / args.source
    if not source_path.exists():
        logger.error(f"Source not found: {source_path}")
        sys.exit(1)

    # Build
    build_from_existing_cache(source_path)

    # Test
    if args.test:
        logger.info("\n=== Speed Test ===")
        reader = CacheV2Reader()

        t0 = time.time()
        split = reader.load_split(
            "2023-01-01", "2025-12-31",  # ~3yr train
            "2026-01-01", "2026-03-01",  # valid
            "2026-03-01", "2026-05-01",  # test
            groups=["base158", "custom", "flow"],
        )
        t1 = time.time()

        logger.info(f"  Load time: {t1-t0:.2f}s")
        logger.info(f"  X_train: {split['X_train'].shape}")
        logger.info(f"  X_test: {split['X_test'].shape}")
        logger.info(f"  Memory: ~{split['X_train'].nbytes / 1024 / 1024:.0f} MB train")

        # Compare with full parquet read
        t2 = time.time()
        pd.read_parquet(str(source_path))
        t3 = time.time()
        logger.info(f"  Full parquet read: {t3-t2:.2f}s")
        logger.info(f"  Speedup: {(t3-t2)/(t1-t0):.1f}x")

    logger.info("Done!")


if __name__ == "__main__":
    main()
