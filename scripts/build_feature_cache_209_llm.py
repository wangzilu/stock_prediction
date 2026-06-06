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

    # Left-join on the base index; missing rows get 0.0 (no LLM signal).
    llm_aligned = llm.reindex(base.index).fillna(0.0)
    print(f"[209_llm] llm aligned non-null: "
          f"{llm_aligned.notna().any(axis=1).sum()} / {len(llm_aligned)}")

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
