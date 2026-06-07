"""Build the xgb_209_llm CANDIDATE feature cache.

xgb_209_llm = xgb_209 (158 Alpha158 + 51 supp) + 5 LLM event factor cols.
This script joins:
  - data/storage/feature_cache_209_production.parquet  (rows × 211 cols)
  - data/storage/llm_event_factors.parquet              (5 LLM cols)

LLM rows that don't match a (datetime, instrument) key default to 0.0
(= no recent events), so the join is left-join with zero-fill. The
resulting parquet has 211 + 5 = 216 cols (including label + aux),
which corresponds to the 214 trained features expected by the
xgb_209_llm profile in config/production_features.py.

Also updates ``data/storage/supp_col_manifest.json`` to include an
``llm_event`` group entry, so the 24-split runner's --drop-group
flag can ablate it for the Phase B LOO comparison
``xgb_209_llm vs xgb_209``.

Output: ``data/storage/feature_cache_209_llm.parquet``
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
DEFAULT_LLM = PROJECT_ROOT / "data/storage/llm_event_factors.parquet"
DEFAULT_OUT = PROJECT_ROOT / "data/storage/feature_cache_209_llm.parquet"
MANIFEST_PATH = PROJECT_ROOT / "data/storage/supp_col_manifest.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--llm", default=str(DEFAULT_LLM))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--allow-schema-drift", action="store_true",
        help="Override the LLM-col-count contract gate (P1 #1 fix). "
             "Use only when you've updated PROFILE_EXPECTED_COUNTS to "
             "match the new LLM schema in the same change.",
    )
    args = ap.parse_args()

    base_path = Path(args.base)
    llm_path = Path(args.llm)
    out_path = Path(args.out)

    if not base_path.exists():
        raise SystemExit(f"base cache missing: {base_path}")
    if not llm_path.exists():
        raise SystemExit(f"LLM parquet missing: {llm_path}")

    t0 = time.time()
    print(f"[209_llm] reading base cache: {base_path}", flush=True)
    base = pd.read_parquet(base_path)
    print(f"[209_llm] base shape: {base.shape}  ({time.time()-t0:.1f}s)")

    print(f"[209_llm] reading LLM events: {llm_path}", flush=True)
    llm = pd.read_parquet(llm_path)
    print(f"[209_llm] llm shape: {llm.shape}")

    llm_cols = [c for c in llm.columns if c not in ("qlib_code", "signal_date")]
    if not llm_cols:
        raise SystemExit("LLM parquet has no factor columns")

    # 2026-06-06 P1 #1 contract gate: the candidate profile
    # xgb_209_llm in config/production_features.py asserts a specific
    # supplementary count. If the LLM parquet schema drifted (e.g.
    # L1 fact-count rebuild added 7 cols), the cache will silently
    # carry 12 LLM cols while the profile expects 5 — and a B-style
    # LOO would compare the wrong dimensions. Hard-fail unless the
    # ``--expected-llm-cols`` flag matches OR the user explicitly
    # opts into schema drift with ``--allow-schema-drift``.
    try:
        from config.production_features import PROFILE_EXPECTED_COUNTS
        contract = PROFILE_EXPECTED_COUNTS.get("xgb_209_llm", {})
        if contract:
            expected_supp = contract.get("supplementary", 0)
            base_supp = PROFILE_EXPECTED_COUNTS.get("xgb_209", {}).get("supplementary", 0)
            expected_llm = expected_supp - base_supp  # supp diff = LLM col count
            if expected_llm > 0 and len(llm_cols) != expected_llm:
                if not args.allow_schema_drift:
                    raise SystemExit(
                        f"LLM schema drift detected: parquet has "
                        f"{len(llm_cols)} cols ({llm_cols}), but "
                        f"xgb_209_llm profile expects {expected_llm} "
                        f"(supp={expected_supp} - base={base_supp}). "
                        f"Either update PROFILE_EXPECTED_COUNTS or "
                        f"re-run with --allow-schema-drift."
                    )
                else:
                    print(f"[209_llm] WARN: schema drift accepted "
                          f"({len(llm_cols)} vs expected {expected_llm})")
    except ImportError:
        pass

    # Align LLM frame to the (datetime, instrument) MultiIndex.
    llm["signal_date"] = pd.to_datetime(llm["signal_date"])
    llm = llm.rename(columns={
        "qlib_code": "instrument",
        "signal_date": "datetime",
    })
    llm = llm.set_index(["datetime", "instrument"])[llm_cols]
    # Drop duplicate (date, code) rows — keep most recent.
    llm = llm[~llm.index.duplicated(keep="last")]
    print(f"[209_llm] llm deduped: {llm.shape}")

    # 2026-06-07 (cx P2 #5 fix): pre-fix this fillna(0.0)'d FIRST and
    # then printed notna().any().sum() so the count was always == len(),
    # giving the false impression that LLM signal covered the whole
    # universe. Compute the TRUE coverage (rows where the reindex
    # actually found a matching LLM event) BEFORE filling.
    llm_raw = llm.reindex(base.index)
    rows_with_real_llm = int(llm_raw.notna().any(axis=1).sum())
    print(f"[209_llm] LLM coverage (real, pre-fillna): "
          f"{rows_with_real_llm} / {len(llm_raw)} rows = "
          f"{100.0 * rows_with_real_llm / max(1, len(llm_raw)):.3f}%")
    llm_aligned = llm_raw.fillna(0.0)

    # Concatenate horizontally.
    out_df = pd.concat([base, llm_aligned], axis=1)
    print(f"[209_llm] out shape: {out_df.shape}")
    expected_cols = base.shape[1] + len(llm_cols)
    assert out_df.shape[1] == expected_cols, (
        f"col count mismatch: base={base.shape[1]} + llm={len(llm_cols)} = "
        f"{expected_cols}, got {out_df.shape[1]}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    out_df.to_parquet(tmp, compression="snappy")
    tmp.replace(out_path)
    sz = out_path.stat().st_size / 1024**3
    print(f"[209_llm] wrote {out_path} ({sz:.2f} GiB, {time.time()-t0:.1f}s total)")

    # Update supp manifest with llm_event group so --drop-group works.
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
    else:
        manifest = {"groups": {}}
    manifest.setdefault("groups", {})
    if manifest["groups"].get("llm_event") != llm_cols:
        manifest["groups"]["llm_event"] = llm_cols
        manifest_tmp = MANIFEST_PATH.with_suffix(".tmp.json")
        manifest_tmp.write_text(json.dumps(manifest, indent=2))
        manifest_tmp.replace(MANIFEST_PATH)
        print(f"[209_llm] manifest updated: llm_event = {llm_cols}")
    else:
        print(f"[209_llm] manifest already up to date")


if __name__ == "__main__":
    main()
