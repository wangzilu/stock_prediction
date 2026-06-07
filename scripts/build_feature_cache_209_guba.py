"""Build the xgb_209_guba CANDIDATE feature cache.

2026-06-07 ablation joiner: joins
``data/storage/feature_cache_209_production.parquet`` with the 3 guba
popularity cols from ``data/storage/guba_factors.parquet`` (keyed by
``(datetime, instrument)`` MultiIndex). Result:
``feature_cache_209_guba.parquet`` with 211 + 3 = 214 cols (212
trained features + label + aux), matching
``PROFILE_EXPECTED_COUNTS["xgb_209_guba"]`` ``supplementary == 54``
(51 base + 3 guba).

Per cx F.P1 #3: the collector is supposed to write UPPERCASE qlib
codes, but the base cache uses LOWERCASE. This joiner normalizes the
guba index to LOWERCASE before reindex so the join succeeds whichever
case the upstream producer used.

Coverage is intentionally sparse — guba popularity only started
2026-05-22 so only ~10 trading days carry signal. Missing rows
fall back to 0 via fillna.

Manifest update: registers group ``guba`` so the 24-split runner's
``--drop-group guba`` can ablate cleanly. ``--allow-schema-drift``
overrides the contract gate (P1 #1).

Usage::

    python scripts/build_feature_cache_209_guba.py
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

CANONICAL_ROOT = Path("/Users/wangzilu/MyProjects/stockPrediction")
DATA_ROOT = (CANONICAL_ROOT if (CANONICAL_ROOT / "data/storage").exists()
             else PROJECT_ROOT) / "data/storage"

DEFAULT_BASE = DATA_ROOT / "feature_cache_209_production.parquet"
DEFAULT_GUBA = DATA_ROOT / "guba_factors.parquet"
DEFAULT_OUT = DATA_ROOT / "feature_cache_209_guba.parquet"
MANIFEST_PATH = DATA_ROOT / "supp_col_manifest.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--guba", default=str(DEFAULT_GUBA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--allow-schema-drift", action="store_true",
        help="Override the guba-col-count contract gate.",
    )
    args = ap.parse_args()

    base_path = Path(args.base)
    guba_path = Path(args.guba)
    out_path = Path(args.out)

    if not base_path.exists():
        raise SystemExit(f"base cache missing: {base_path}")
    if not guba_path.exists():
        raise SystemExit(f"guba parquet missing: {guba_path}")

    t0 = time.time()
    print(f"[209_guba] reading base cache: {base_path}", flush=True)
    base = pd.read_parquet(base_path)
    print(f"[209_guba] base shape: {base.shape}  ({time.time()-t0:.1f}s)")

    print(f"[209_guba] reading guba factors: {guba_path}", flush=True)
    guba = pd.read_parquet(guba_path)
    print(f"[209_guba] guba shape: {guba.shape}")

    if not isinstance(guba.index, pd.MultiIndex):
        if {"datetime", "instrument"}.issubset(guba.columns):
            guba = guba.set_index(["datetime", "instrument"])
        else:
            raise SystemExit(
                f"guba parquet missing keys; cols={list(guba.columns)}"
            )

    factor_cols = [
        c for c in guba.columns
        if pd.api.types.is_numeric_dtype(guba[c])
    ]
    if not factor_cols:
        raise SystemExit("guba parquet has no numeric factor cols")

    # Contract gate (P1 #1).
    try:
        from config.production_features import PROFILE_EXPECTED_COUNTS
        contract = PROFILE_EXPECTED_COUNTS.get("xgb_209_guba", {})
        if contract:
            expected_supp = contract.get("supplementary", 0)
            base_supp = PROFILE_EXPECTED_COUNTS.get("xgb_209", {}).get("supplementary", 0)
            expected_guba = expected_supp - base_supp
            if expected_guba > 0 and len(factor_cols) != expected_guba:
                if not args.allow_schema_drift:
                    raise SystemExit(
                        f"guba schema drift: parquet has {len(factor_cols)} "
                        f"cols ({factor_cols}), profile expects "
                        f"{expected_guba}. Pass --allow-schema-drift to override."
                    )
                print(f"[209_guba] WARN: schema drift accepted "
                      f"({len(factor_cols)} vs expected {expected_guba})")
    except ImportError:
        pass

    # Normalize instrument case to match base cache (lowercase sh/sz).
    # The base cache uses lowercase; cx F.P1 #3 wants upstream to write
    # UPPERCASE, but as of 2026-06-07 the parquet is still lowercase.
    # Force lowercase here so the join works either way.
    try:
        guba.index = guba.index.set_levels(
            guba.index.levels[1].astype(str).str.lower(), level=1,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[209_guba] WARN: index case normalize skipped: {exc}")

    # Dedupe by index (keep last per date+stock).
    guba = guba[factor_cols][~guba.index.duplicated(keep="last")]
    print(f"[209_guba] guba deduped: {guba.shape}")

    # Pre-fillna coverage count (mirrors LLM joiner P2 #5 fix).
    guba_raw = guba.reindex(base.index)
    rows_with_real_guba = int(guba_raw.notna().any(axis=1).sum())
    print(f"[209_guba] guba coverage (pre-fillna): "
          f"{rows_with_real_guba} / {len(guba_raw)} rows = "
          f"{100.0 * rows_with_real_guba / max(1, len(guba_raw)):.3f}%")
    guba_aligned = guba_raw.fillna(0.0)

    out_df = pd.concat([base, guba_aligned], axis=1)
    expected_cols = base.shape[1] + len(factor_cols)
    assert out_df.shape[1] == expected_cols, (
        f"col mismatch: base={base.shape[1]} + guba={len(factor_cols)} = "
        f"{expected_cols}, got {out_df.shape[1]}"
    )
    print(f"[209_guba] out shape: {out_df.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    out_df.to_parquet(tmp, compression="snappy")
    tmp.replace(out_path)
    sz = out_path.stat().st_size / 1024**3
    print(f"[209_guba] wrote {out_path} ({sz:.2f} GiB, {time.time()-t0:.1f}s total)")

    # Manifest update.
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
    else:
        manifest = {"groups": {}}
    manifest.setdefault("groups", {})
    if manifest["groups"].get("guba") != factor_cols:
        manifest["groups"]["guba"] = factor_cols
        manifest_tmp = MANIFEST_PATH.with_suffix(".tmp.json")
        manifest_tmp.write_text(json.dumps(manifest, indent=2))
        manifest_tmp.replace(MANIFEST_PATH)
        print(f"[209_guba] manifest updated: guba = {factor_cols}")
    else:
        print(f"[209_guba] manifest already up to date")


if __name__ == "__main__":
    main()
