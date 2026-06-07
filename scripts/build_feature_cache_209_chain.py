"""Build the xgb_209_chain or xgb_209_chain_llm candidate cache.

Joins ``feature_cache_209_production.parquet`` with either:
  - ``global_chain_factors.parquet`` (rule-based, --source rule)
  - ``global_chain_factors_llm.parquet`` (LLM, --source llm)

Both chain parquets are already keyed on (datetime, instrument) so the
left-join is a simple reindex + horizontal concat.

The contract gate (P1 #1 lesson from earlier): verify the chain
parquet's numeric column count matches PROFILE_EXPECTED_COUNTS so
schema drift can't ship a wrong-shape cache. Override via
``--allow-schema-drift`` when intentional.

Output:
  - ``feature_cache_209_chain.parquet`` (rule)
  - ``feature_cache_209_chain_llm.parquet`` (llm)

Also registers a manifest entry under group ``global_chain`` /
``global_chain_llm`` so the 24-split runner's ``--drop-group`` can
ablate the corresponding cols cleanly.
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

DEFAULT_BASE = PROJECT_ROOT / "data/storage/feature_cache_209_production.parquet"
CHAIN_RULE = PROJECT_ROOT / "data/storage/global_chain_factors.parquet"
CHAIN_LLM = PROJECT_ROOT / "data/storage/global_chain_factors_llm.parquet"
DEFAULT_OUT_RULE = PROJECT_ROOT / "data/storage/feature_cache_209_chain.parquet"
DEFAULT_OUT_LLM = PROJECT_ROOT / "data/storage/feature_cache_209_chain_llm.parquet"
MANIFEST_PATH = PROJECT_ROOT / "data/storage/supp_col_manifest.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["rule", "llm"], default="rule")
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--out", default=None,
                    help="Override output path (default chosen per --source)")
    ap.add_argument("--allow-schema-drift", action="store_true",
                    help="Skip the contract gate when the chain parquet's "
                         "numeric col count differs from PROFILE_EXPECTED_COUNTS.")
    args = ap.parse_args()

    chain_path = CHAIN_LLM if args.source == "llm" else CHAIN_RULE
    if args.out is None:
        out_path = DEFAULT_OUT_LLM if args.source == "llm" else DEFAULT_OUT_RULE
    else:
        out_path = Path(args.out)

    base_path = Path(args.base)
    profile = "xgb_209_chain_llm" if args.source == "llm" else "xgb_209_chain"
    group_key = "global_chain_llm" if args.source == "llm" else "global_chain"

    if not base_path.exists():
        raise SystemExit(f"base cache missing: {base_path}")
    if not chain_path.exists():
        raise SystemExit(
            f"chain parquet missing: {chain_path}\n"
            f"Run scripts/build_global_chain_factors.py --source {args.source} first."
        )

    t0 = time.time()
    print(f"[209_chain] source={args.source}  base={base_path}", flush=True)
    base = pd.read_parquet(base_path)
    print(f"[209_chain] base shape: {base.shape}  ({time.time()-t0:.1f}s)")

    print(f"[209_chain] reading chain: {chain_path}", flush=True)
    chain = pd.read_parquet(chain_path)
    print(f"[209_chain] chain shape: {chain.shape}  cols={list(chain.columns)}")

    # Drop non-numeric cols (matches FeatureMerger loader behaviour).
    factor_cols = [c for c in chain.columns
                   if pd.api.types.is_numeric_dtype(chain[c])]
    if not factor_cols:
        raise SystemExit("chain parquet has no numeric factor cols")
    print(f"[209_chain] numeric factor cols ({len(factor_cols)}): {factor_cols}")

    # Contract gate.
    try:
        from config.production_features import PROFILE_EXPECTED_COUNTS
        contract = PROFILE_EXPECTED_COUNTS.get(profile, {})
        if contract:
            expected_chain = (contract.get("supplementary", 0) -
                              PROFILE_EXPECTED_COUNTS["xgb_209"]["supplementary"])
            if expected_chain > 0 and len(factor_cols) != expected_chain:
                if not args.allow_schema_drift:
                    raise SystemExit(
                        f"chain schema drift: parquet has {len(factor_cols)} "
                        f"numeric cols {factor_cols}, profile expects "
                        f"{expected_chain}. Pass --allow-schema-drift to override."
                    )
                else:
                    print(f"[209_chain] WARN: schema drift accepted "
                          f"({len(factor_cols)} vs expected {expected_chain})")
    except ImportError:
        pass

    # Reindex and concat — drop duplicate keys keeping last.
    chain_clean = chain[factor_cols][~chain.index.duplicated(keep="last")]
    chain_aligned = chain_clean.reindex(base.index).fillna(0.0)
    print(f"[209_chain] chain non-zero rows after align: "
          f"{(chain_aligned != 0).any(axis=1).sum()} / {len(chain_aligned)}")

    out_df = pd.concat([base, chain_aligned], axis=1)
    expected_cols = base.shape[1] + len(factor_cols)
    assert out_df.shape[1] == expected_cols, (
        f"col mismatch: base={base.shape[1]} + chain={len(factor_cols)} "
        f"= {expected_cols}, got {out_df.shape[1]}"
    )
    print(f"[209_chain] out shape: {out_df.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    out_df.to_parquet(tmp, compression="snappy")
    tmp.replace(out_path)
    sz = out_path.stat().st_size / 1024**3
    print(f"[209_chain] wrote {out_path} ({sz:.2f} GiB, {time.time()-t0:.1f}s total)")

    # Manifest update so --drop-group can find the cols.
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
    else:
        manifest = {"groups": {}}
    manifest.setdefault("groups", {})
    if manifest["groups"].get(group_key) != factor_cols:
        manifest["groups"][group_key] = factor_cols
        tmp_m = MANIFEST_PATH.with_suffix(".tmp.json")
        tmp_m.write_text(json.dumps(manifest, indent=2))
        tmp_m.replace(MANIFEST_PATH)
        print(f"[209_chain] manifest updated: {group_key} = {factor_cols}")


if __name__ == "__main__":
    main()
