"""Cache V2 Mixed: partitioned parquet + group npy sidecar.

Strategy per CX:
- Parquet as canonical store (compressed, column-selective)
- Small groups as .npy sidecar with mmap (regime/holder/flow/labels/ma)
- base158 stays in parquet (too big for raw npy, compression helps)
- Row index for fast date-range slicing

Usage:
    python scripts/build_cache_v2_mixed.py
    python scripts/build_cache_v2_mixed.py --test
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
CACHE_DIR = DATA_DIR / "cache_v2m"

# Groups: small ones go to npy (fast mmap), large ones stay parquet (compressed)
NPY_GROUPS = {
    "regime": lambda cols: [c for c in cols if c.startswith(("hsi_", "hstech_", "nasdaq_"))],
    "holder": lambda cols: [c for c in cols if c == "holder_num"],
    "flow": lambda cols: [c for c in cols if c.startswith("flow_")],
    "ma": lambda cols: [c for c in cols if c.startswith("_") and not c.startswith("__")],
    "labels": lambda cols: [c for c in cols if c.startswith("__")],
}

# Everything else → parquet (base158 + custom)
def get_parquet_cols(all_cols):
    npy_cols = set()
    for selector in NPY_GROUPS.values():
        npy_cols.update(selector(all_cols))
    return [c for c in all_cols if c not in npy_cols]


def build(source_path: Path):
    logger.info(f"Loading source: {source_path}")
    t0 = time.time()
    df = pd.read_parquet(str(source_path))
    logger.info(f"  Shape: {df.shape}, {time.time()-t0:.1f}s")

    all_cols = list(df.columns)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Row index
    dates = df.index.get_level_values(0)
    unique_dates = sorted(dates.unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    date_indices = np.array([date_to_idx[d] for d in dates], dtype=np.int32)
    np.save(str(CACHE_DIR / "date_indices.npy"), date_indices)

    # Instruments
    inst_codes = df.index.get_level_values(1).astype(str).values
    np.save(str(CACHE_DIR / "instruments.npy"), inst_codes)

    # Save multi-index tuples for reconstruction
    index_dates = dates.values
    np.save(str(CACHE_DIR / "index_dates.npy"), index_dates)

    meta = {
        "n_rows": len(df),
        "n_dates": len(unique_dates),
        "dates": [str(d)[:10] for d in unique_dates],
        "groups": {},
        "parquet_cols": [],
    }

    # NPY groups (small, fast mmap)
    for group_name, selector in NPY_GROUPS.items():
        cols = selector(all_cols)
        if not cols:
            continue
        arr = df[cols].values.astype(np.float32)
        path = CACHE_DIR / f"{group_name}.npy"
        np.save(str(path), arr)
        size_mb = path.stat().st_size / 1024 / 1024
        meta["groups"][group_name] = {"type": "npy", "columns": cols, "size_mb": round(size_mb, 1)}
        logger.info(f"  npy  {group_name}: {len(cols)} cols, {size_mb:.1f} MB")

    # Parquet group (large, compressed)
    pq_cols = get_parquet_cols(all_cols)
    if pq_cols:
        pq_path = CACHE_DIR / "base_features.parquet"
        # Save without index (row order matches npy files)
        pq_df = df[pq_cols].reset_index(drop=True)
        pq_df.to_parquet(str(pq_path), engine="pyarrow", compression="snappy")
        size_mb = pq_path.stat().st_size / 1024 / 1024
        meta["groups"]["base_features"] = {"type": "parquet", "columns": pq_cols, "size_mb": round(size_mb, 1)}
        meta["parquet_cols"] = pq_cols
        logger.info(f"  pq   base_features: {len(pq_cols)} cols, {size_mb:.1f} MB (compressed)")

    # Save metadata
    with open(str(CACHE_DIR / "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    total_size = sum(f.stat().st_size for f in CACHE_DIR.iterdir()) / 1024 / 1024
    logger.info(f"\nCache V2 Mixed: {CACHE_DIR}")
    logger.info(f"  Total: {total_size:.1f} MB (source: {source_path.stat().st_size/1024/1024:.1f} MB)")


class CacheV2MixedReader:
    """Read from mixed cache: parquet for base, npy for small groups."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.dir = Path(cache_dir)
        with open(str(self.dir / "meta.json")) as f:
            self.meta = json.load(f)
        self.date_indices = np.load(str(self.dir / "date_indices.npy"))
        self.dates = self.meta["dates"]
        self._npy_cache = {}
        self._pq_cache = None

    def _load_npy(self, group: str) -> np.ndarray:
        if group not in self._npy_cache:
            self._npy_cache[group] = np.load(str(self.dir / f"{group}.npy"), mmap_mode="r")
        return self._npy_cache[group]

    def _load_parquet_rows(self, row_mask: np.ndarray, columns: list = None) -> np.ndarray:
        """Load parquet rows. Uses column selection for speed."""
        if self._pq_cache is None:
            cols = columns or self.meta.get("parquet_cols")
            self._pq_cache = pd.read_parquet(
                str(self.dir / "base_features.parquet"),
                columns=cols,
            ).values.astype(np.float32)
        return self._pq_cache[row_mask]

    def get_mask(self, start_date: str, end_date: str) -> np.ndarray:
        start_idx = next((i for i, d in enumerate(self.dates) if d >= start_date), 0)
        end_idx = next((i for i, d in enumerate(self.dates) if d > end_date), len(self.dates))
        return (self.date_indices >= start_idx) & (self.date_indices < end_idx)

    def load_split(self, train_start, train_end, valid_start, valid_end,
                   test_start, test_end, npy_groups=None):
        """Load one split. Base features from parquet, extras from npy.

        Args:
            npy_groups: list of npy group names to include (e.g. ["regime", "flow"])
                        None = base only (no extras)
        """
        tm = self.get_mask(train_start, train_end)
        vm = self.get_mask(valid_start, valid_end)
        em = self.get_mask(test_start, test_end)

        # Base features (parquet)
        X_tr = self._load_parquet_rows(tm)
        X_va = self._load_parquet_rows(vm)
        X_te = self._load_parquet_rows(em)

        # Add npy groups
        if npy_groups:
            for g in npy_groups:
                if g in self.meta["groups"] and self.meta["groups"][g]["type"] == "npy":
                    arr = self._load_npy(g)
                    X_tr = np.hstack([X_tr, arr[tm]])
                    X_va = np.hstack([X_va, arr[vm]])
                    X_te = np.hstack([X_te, arr[em]])

        # Labels
        labels = self._load_npy("labels")
        y_tr = labels[tm, 0].copy()  # __label_5d
        y_va = labels[vm, 0].copy()
        y_te = labels[em, 0].copy()

        # NaN filter
        mtr = np.isfinite(y_tr)
        mva = np.isfinite(y_va)
        mte = np.isfinite(y_te)

        # Test instruments
        insts = np.load(str(self.dir / "instruments.npy"), allow_pickle=True)
        index_dates = np.load(str(self.dir / "index_dates.npy"), allow_pickle=True)
        test_tuples = list(zip(index_dates[em][mte], insts[em][mte]))
        test_idx = pd.MultiIndex.from_tuples(test_tuples)

        return {
            "X_train": X_tr[mtr], "y_train": y_tr[mtr],
            "X_valid": X_va[mva], "y_valid": y_va[mva],
            "X_test": X_te[mte], "y_test": y_te[mte],
            "test_idx": test_idx,
            "n_features": X_tr.shape[1],
        }

    def get_all_cols(self, npy_groups=None):
        """Get column names for current configuration."""
        cols = list(self.meta.get("parquet_cols", []))
        if npy_groups:
            for g in npy_groups:
                if g in self.meta["groups"]:
                    cols.extend(self.meta["groups"][g]["columns"])
        return cols


def test_speed(cache_dir, source_path):
    """Compare load speed: V2 mixed vs full parquet."""
    logger.info("\n=== Speed Test ===")

    reader = CacheV2MixedReader(cache_dir)

    # Test 1: Base only (parquet)
    t0 = time.time()
    split = reader.load_split("2023-01-01", "2025-12-31",
                              "2026-01-01", "2026-03-01",
                              "2026-03-01", "2026-05-01")
    t1 = time.time()
    logger.info(f"  Base only: {t1-t0:.2f}s, {split['X_train'].shape}")

    # Test 2: Base + regime (parquet + npy)
    reader2 = CacheV2MixedReader(cache_dir)  # fresh reader
    t2 = time.time()
    split2 = reader2.load_split("2023-01-01", "2025-12-31",
                                "2026-01-01", "2026-03-01",
                                "2026-03-01", "2026-05-01",
                                npy_groups=["regime", "flow", "holder"])
    t3 = time.time()
    logger.info(f"  Base+extras: {t3-t2:.2f}s, {split2['X_train'].shape}")

    # Test 3: Full parquet read (baseline)
    t4 = time.time()
    pd.read_parquet(str(source_path))
    t5 = time.time()
    logger.info(f"  Full parquet: {t5-t4:.2f}s")

    logger.info(f"\n  Speedup vs full parquet:")
    logger.info(f"    Base only: {(t5-t4)/(t1-t0):.1f}x")
    logger.info(f"    Base+extras: {(t5-t4)/(t3-t2):.1f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="feature_cache_174_holder_regime_ma.parquet")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    source_path = DATA_DIR / args.source
    if not source_path.exists():
        logger.error(f"Source not found: {source_path}")
        sys.exit(1)

    build(source_path)

    if args.test:
        test_speed(CACHE_DIR, source_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
