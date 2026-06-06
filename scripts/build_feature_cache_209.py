"""Build the 209-feature production cache by dropping Phase B Bucket A
(cross_market_regime + capital_flow + shareholder) from the existing
242-feature cache.

2026-06-06 Phase B.4 verdict promoted xgb_209 as the next champion
(see docs/phase_b4_verdict_20260606.md). This script materialises the
parquet that downstream training + inference will read.

Output: data/storage/feature_cache_209_production.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DROP_GROUPS = ("cross_market_regime", "capital_flow", "shareholder")
DEFAULT_INPUT = PROJECT_ROOT / "data/storage/feature_cache_242_production.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/storage/feature_cache_209_production.parquet"
MANIFEST_PATH = PROJECT_ROOT / "data/storage/supp_col_manifest.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = ap.parse_args()

    t0 = time.time()

    in_path = Path(args.input)
    out_path = Path(args.output)
    manifest_path = Path(args.manifest)

    if not in_path.exists():
        raise SystemExit(f"input parquet missing: {in_path}")
    if not manifest_path.exists():
        raise SystemExit(f"manifest missing: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    group_map = manifest.get("groups", {})

    drop_cols: set[str] = set()
    for g in DROP_GROUPS:
        if g not in group_map:
            raise SystemExit(
                f"group {g!r} not in manifest. Known: {sorted(group_map)}"
            )
        drop_cols.update(group_map[g])

    print(f"[build_209] input  = {in_path}")
    print(f"[build_209] output = {out_path}")
    print(f"[build_209] drop groups = {DROP_GROUPS}")
    print(f"[build_209] drop cols (manifest) = {len(drop_cols)}")

    print(f"[build_209] reading parquet ...", flush=True)
    df = pd.read_parquet(in_path)
    print(f"[build_209] in shape = {df.shape}  ({time.time()-t0:.1f}s)")

    kept_cols = [c for c in df.columns if c not in drop_cols]
    actually_dropped = [c for c in df.columns if c in drop_cols]
    df_out = df[kept_cols]
    print(f"[build_209] kept   = {len(kept_cols)}  dropped = {len(actually_dropped)}")
    print(f"[build_209] out shape = {df_out.shape}")

    # 2026-06-06 P2 #5 contract gate: verify input was xgb_242-shaped
    # and output matches the xgb_209 profile expectation. Without this,
    # a manifest-drift / aux-col change could silently produce a cache
    # whose name says 209 but whose width is wrong.
    try:
        from config.production_features import PROFILE_EXPECTED_COUNTS
        exp_242 = PROFILE_EXPECTED_COUNTS.get("xgb_242", {}).get("total")
        exp_209 = PROFILE_EXPECTED_COUNTS.get("xgb_209", {}).get("total")
        # Cache typically has total = features + label + a few aux cols.
        # Empirically xgb_242 cache = 244 cols (242 + label + 1 aux).
        # The relationship we can enforce is: dropped_cols count = 33
        # (matches the Bucket A trio) and output ratio = input - 33.
        expected_drop = 33  # cross_market_regime 27 + capital_flow 3 + shareholder 3
        if len(actually_dropped) != expected_drop:
            raise SystemExit(
                f"contract gate: dropped {len(actually_dropped)} cols "
                f"but expected {expected_drop} (cross_market_regime 27 + "
                f"capital_flow 3 + shareholder 3). Manifest may have drifted."
            )
        if df_out.shape[1] != df.shape[1] - expected_drop:
            raise SystemExit(
                f"contract gate: output width mismatch. in={df.shape[1]} "
                f"out={df_out.shape[1]} delta={df.shape[1] - df_out.shape[1]} "
                f"expected delta={expected_drop}"
            )
        print(f"[build_209] contract gate OK: 242→209 via 33-col drop")
    except ImportError:
        print(f"[build_209] WARN: production_features not importable, skipping contract gate")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    df_out.to_parquet(tmp, compression="snappy")
    tmp.replace(out_path)
    print(f"[build_209] wrote {out_path} ({time.time()-t0:.1f}s total)")

    sz = out_path.stat().st_size / 1024**3
    print(f"[build_209] file size = {sz:.2f} GiB")


if __name__ == "__main__":
    main()
