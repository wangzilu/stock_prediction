"""Build the xgb_209_pbc CANDIDATE feature cache.

2026-06-07 ablation joiner: takes
``data/storage/feature_cache_209_production.parquet`` and joins the 4
PBC liquidity factor cols from
``data/storage/pbc_liquidity_factors.parquet`` (keyed by
``(datetime, "MARKET")``) by broadcasting the MARKET row across every
stock on the same date. Result: ``feature_cache_209_pbc.parquet`` with
211 base cols + 4 PBC cols = 215 total (213 trained features
+ label + aux), matching ``PROFILE_EXPECTED_COUNTS["xgb_209_pbc"]``
``supplementary == 55`` (51 base + 4 PBC).

Broadcast logic mirrors ``FeatureMerger._load_pbc_liquidity``:
collapse the MARKET-keyed parquet to a date-indexed frame, then
reindex by the base cache's datetime level so every stock on date D
gets the same PBC values.

Manifest update: registers group ``pbc_liquidity`` so the 24-split
runner's ``--drop-group pbc_liquidity`` can ablate cleanly. The
``--allow-schema-drift`` flag overrides the contract gate (P1 #1
fix from build_feature_cache_209_llm.py).

Usage::

    python scripts/build_feature_cache_209_pbc.py
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

# Cross-worktree paths: when invoked from a `.claude/worktrees/agent-*`
# checkout, write outputs to the canonical project tree so phase4e
# scripts find them at the expected location.
CANONICAL_ROOT = Path("/Users/wangzilu/MyProjects/stockPrediction")
DATA_ROOT = (CANONICAL_ROOT if (CANONICAL_ROOT / "data/storage").exists()
             else PROJECT_ROOT) / "data/storage"

DEFAULT_BASE = DATA_ROOT / "feature_cache_209_production.parquet"
DEFAULT_PBC = DATA_ROOT / "pbc_liquidity_factors.parquet"
DEFAULT_OUT = DATA_ROOT / "feature_cache_209_pbc.parquet"
MANIFEST_PATH = DATA_ROOT / "supp_col_manifest.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--pbc", default=str(DEFAULT_PBC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--allow-schema-drift", action="store_true",
        help="Override the PBC-col-count contract gate.",
    )
    args = ap.parse_args()

    base_path = Path(args.base)
    pbc_path = Path(args.pbc)
    out_path = Path(args.out)

    if not base_path.exists():
        raise SystemExit(f"base cache missing: {base_path}")
    if not pbc_path.exists():
        raise SystemExit(f"PBC parquet missing: {pbc_path}")

    t0 = time.time()
    print(f"[209_pbc] reading base cache: {base_path}", flush=True)
    base = pd.read_parquet(base_path)
    print(f"[209_pbc] base shape: {base.shape}  ({time.time()-t0:.1f}s)")

    print(f"[209_pbc] reading PBC factors: {pbc_path}", flush=True)
    pbc = pd.read_parquet(pbc_path)
    print(f"[209_pbc] pbc shape: {pbc.shape}")

    # Normalize: should be flat (datetime, instrument, cols...). Promote
    # to MultiIndex via reset_index if already indexed.
    if isinstance(pbc.index, pd.MultiIndex):
        pbc = pbc.reset_index()
    if "datetime" not in pbc.columns or "instrument" not in pbc.columns:
        raise SystemExit(
            f"PBC parquet missing keys; have cols={list(pbc.columns)}"
        )
    pbc["datetime"] = pd.to_datetime(pbc["datetime"])

    factor_cols = [
        c for c in pbc.columns
        if c not in ("datetime", "instrument")
        and pd.api.types.is_numeric_dtype(pbc[c])
    ]
    if not factor_cols:
        raise SystemExit("PBC parquet has no numeric factor cols")

    # Contract gate (P1 #1).
    try:
        from config.production_features import PROFILE_EXPECTED_COUNTS
        contract = PROFILE_EXPECTED_COUNTS.get("xgb_209_pbc", {})
        if contract:
            expected_supp = contract.get("supplementary", 0)
            base_supp = PROFILE_EXPECTED_COUNTS.get("xgb_209", {}).get("supplementary", 0)
            expected_pbc = expected_supp - base_supp
            if expected_pbc > 0 and len(factor_cols) != expected_pbc:
                if not args.allow_schema_drift:
                    raise SystemExit(
                        f"PBC schema drift: parquet has {len(factor_cols)} "
                        f"cols ({factor_cols}), profile expects "
                        f"{expected_pbc}. Pass --allow-schema-drift to override."
                    )
                print(f"[209_pbc] WARN: schema drift accepted "
                      f"({len(factor_cols)} vs expected {expected_pbc})")
    except ImportError:
        pass

    # Reduce to MARKET-only date-keyed frame.
    market_df = (
        pbc[pbc["instrument"] == "MARKET"]
        .drop_duplicates("datetime", keep="last")
        .set_index("datetime")[factor_cols]
        .sort_index()
    )
    print(f"[209_pbc] MARKET rows: {len(market_df)} dates")
    if market_df.empty:
        raise SystemExit("No MARKET rows in PBC parquet — cannot broadcast")

    # Broadcast: align on the base cache's datetime level.
    caller_dates = base.index.get_level_values("datetime")
    pbc_aligned = market_df.reindex(caller_dates).fillna(0.0)
    pbc_aligned.index = base.index
    non_zero_rows = int((pbc_aligned != 0).any(axis=1).sum())
    print(f"[209_pbc] PBC coverage: {non_zero_rows} / {len(pbc_aligned)} "
          f"rows = {100.0 * non_zero_rows / max(1, len(pbc_aligned)):.2f}%")

    out_df = pd.concat([base, pbc_aligned], axis=1)
    expected_cols = base.shape[1] + len(factor_cols)
    assert out_df.shape[1] == expected_cols, (
        f"col mismatch: base={base.shape[1]} + pbc={len(factor_cols)} = "
        f"{expected_cols}, got {out_df.shape[1]}"
    )
    print(f"[209_pbc] out shape: {out_df.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    out_df.to_parquet(tmp, compression="snappy")
    tmp.replace(out_path)
    sz = out_path.stat().st_size / 1024**3
    print(f"[209_pbc] wrote {out_path} ({sz:.2f} GiB, {time.time()-t0:.1f}s total)")

    # Manifest update so --drop-group pbc_liquidity works.
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
    else:
        manifest = {"groups": {}}
    manifest.setdefault("groups", {})
    if manifest["groups"].get("pbc_liquidity") != factor_cols:
        manifest["groups"]["pbc_liquidity"] = factor_cols
        manifest_tmp = MANIFEST_PATH.with_suffix(".tmp.json")
        manifest_tmp.write_text(json.dumps(manifest, indent=2))
        manifest_tmp.replace(MANIFEST_PATH)
        print(f"[209_pbc] manifest updated: pbc_liquidity = {factor_cols}")
    else:
        print(f"[209_pbc] manifest already up to date")


if __name__ == "__main__":
    main()
