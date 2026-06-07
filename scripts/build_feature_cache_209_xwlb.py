"""Build the xgb_209_xwlb CANDIDATE feature cache.

2026-06-07 ablation joiner: joins
``data/storage/feature_cache_209_production.parquet`` with the 4 XWLB
(新闻联播) theme factor cols from
``data/storage/xinwen_lianbo_theme_factors.parquet`` (keyed by
``(datetime, "THEME_<UPPER>")``).

The XWLB parquet is THEME-keyed; stock-level training samples never
match a ``THEME_<NAME>`` instrument directly. cx C.P1 #3 supplies
``factors.xwlb_theme_baskets.broadcast_theme_to_stocks`` (backed by
``config/xwlb_theme_baskets.yaml``) which replicates each
(date, THEME_X) row onto every (date, STOCK_K) row where STOCK_K is
in the theme's basket.

Result: ``feature_cache_209_xwlb.parquet`` with 211 + 4 = 215 cols
(213 trained + label + aux), matching
``PROFILE_EXPECTED_COUNTS["xgb_209_xwlb"]`` ``supplementary == 55``
(51 base + 4 XWLB).

Important: this script only runs successfully when the XWLB factor
parquet exists AND contains rows. As of 2026-06-07 the PE-4 cron has
not produced a historical scrape, so the file is absent / empty. The
script fails loud in that case — operator should rerun once XWLB
events accumulate.

Manifest update: registers group ``xinwen_lianbo`` so the 24-split
runner's ``--drop-group xinwen_lianbo`` can ablate cleanly.

Usage::

    python scripts/build_feature_cache_209_xwlb.py
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
DEFAULT_XWLB = DATA_ROOT / "xinwen_lianbo_theme_factors.parquet"
DEFAULT_OUT = DATA_ROOT / "feature_cache_209_xwlb.parquet"
MANIFEST_PATH = DATA_ROOT / "supp_col_manifest.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--xwlb", default=str(DEFAULT_XWLB))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--allow-schema-drift", action="store_true",
        help="Override the XWLB-col-count contract gate.",
    )
    args = ap.parse_args()

    base_path = Path(args.base)
    xwlb_path = Path(args.xwlb)
    out_path = Path(args.out)

    if not base_path.exists():
        raise SystemExit(f"base cache missing: {base_path}")
    if not xwlb_path.exists():
        raise SystemExit(
            f"XWLB factor parquet missing: {xwlb_path}\n"
            f"Run scripts/build_policy_factors.py --source xinwen_lianbo first."
        )

    t0 = time.time()
    print(f"[209_xwlb] reading base cache: {base_path}", flush=True)
    base = pd.read_parquet(base_path)
    print(f"[209_xwlb] base shape: {base.shape}  ({time.time()-t0:.1f}s)")

    print(f"[209_xwlb] reading XWLB factors: {xwlb_path}", flush=True)
    xwlb = pd.read_parquet(xwlb_path)
    print(f"[209_xwlb] xwlb shape: {xwlb.shape}")

    if xwlb.empty:
        raise SystemExit(
            "XWLB parquet is empty — no theme factors available. "
            "Skip the xwlb track and rerun once the PE-4 cron has "
            "accumulated historical XWLB events."
        )

    # Promote to flat columns if multi-indexed.
    if isinstance(xwlb.index, pd.MultiIndex):
        xwlb = xwlb.reset_index()
    if "datetime" not in xwlb.columns or "instrument" not in xwlb.columns:
        raise SystemExit(
            f"XWLB parquet missing keys; cols={list(xwlb.columns)}"
        )
    xwlb["datetime"] = pd.to_datetime(xwlb["datetime"], errors="coerce")
    xwlb = xwlb.dropna(subset=["datetime"])

    factor_cols = [
        c for c in xwlb.columns
        if c not in ("datetime", "instrument")
        and pd.api.types.is_numeric_dtype(xwlb[c])
    ]
    if not factor_cols:
        raise SystemExit("XWLB parquet has no numeric factor cols")

    # Contract gate.
    try:
        from config.production_features import PROFILE_EXPECTED_COUNTS
        contract = PROFILE_EXPECTED_COUNTS.get("xgb_209_xwlb", {})
        if contract:
            expected_supp = contract.get("supplementary", 0)
            base_supp = PROFILE_EXPECTED_COUNTS.get("xgb_209", {}).get("supplementary", 0)
            expected_xwlb = expected_supp - base_supp
            if expected_xwlb > 0 and len(factor_cols) != expected_xwlb:
                if not args.allow_schema_drift:
                    raise SystemExit(
                        f"XWLB schema drift: parquet has {len(factor_cols)} "
                        f"cols ({factor_cols}), profile expects "
                        f"{expected_xwlb}. Pass --allow-schema-drift to override."
                    )
                print(f"[209_xwlb] WARN: schema drift accepted "
                      f"({len(factor_cols)} vs expected {expected_xwlb})")
    except ImportError:
        pass

    # Broadcast THEME → STOCK via C.P1 #3 mapper.
    try:
        from factors.xwlb_theme_baskets import broadcast_theme_to_stocks
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"XWLB basket mapper unavailable ({exc}); cannot proceed."
        )
    broadcast = broadcast_theme_to_stocks(
        xwlb, factor_cols=factor_cols, prefix="xwlb_",
    )
    if broadcast is None or broadcast.empty:
        raise SystemExit(
            "XWLB broadcast produced an empty frame — basket YAML may "
            "not cover any themes present in the factor parquet."
        )
    print(f"[209_xwlb] broadcast frame shape: {broadcast.shape}")

    # 2026-06-08 case-bug prevention (post-B.8): centralised helpers.
    from factors.feature_cache_utils import (
        assert_join_coverage, normalize_instrument_index,
    )
    broadcast = normalize_instrument_index(broadcast, source_name="xwlb")

    # Reindex onto base index + coverage gate.
    xwlb_raw = broadcast.reindex(base.index)
    assert_join_coverage(
        source_df=broadcast, reindexed=xwlb_raw,
        factor_cols=factor_cols, source_name="xwlb",
    )
    rows_with_real = int(xwlb_raw.notna().any(axis=1).sum())
    print(f"[209_xwlb] XWLB coverage (pre-fillna): "
          f"{rows_with_real} / {len(xwlb_raw)} rows = "
          f"{100.0 * rows_with_real / max(1, len(xwlb_raw)):.3f}%")
    xwlb_aligned = xwlb_raw.fillna(0.0)

    out_df = pd.concat([base, xwlb_aligned], axis=1)
    expected_cols = base.shape[1] + len(factor_cols)
    assert out_df.shape[1] == expected_cols, (
        f"col mismatch: base={base.shape[1]} + xwlb={len(factor_cols)} = "
        f"{expected_cols}, got {out_df.shape[1]}"
    )
    print(f"[209_xwlb] out shape: {out_df.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    out_df.to_parquet(tmp, compression="snappy")
    tmp.replace(out_path)
    sz = out_path.stat().st_size / 1024**3
    print(f"[209_xwlb] wrote {out_path} ({sz:.2f} GiB, {time.time()-t0:.1f}s total)")

    # Manifest update — use prefixed col names (xwlb_*).
    broadcast_cols = list(xwlb_aligned.columns)
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
    else:
        manifest = {"groups": {}}
    manifest.setdefault("groups", {})
    if manifest["groups"].get("xinwen_lianbo") != broadcast_cols:
        manifest["groups"]["xinwen_lianbo"] = broadcast_cols
        manifest_tmp = MANIFEST_PATH.with_suffix(".tmp.json")
        manifest_tmp.write_text(json.dumps(manifest, indent=2))
        manifest_tmp.replace(MANIFEST_PATH)
        print(f"[209_xwlb] manifest updated: xinwen_lianbo = {broadcast_cols}")
    else:
        print(f"[209_xwlb] manifest already up to date")


if __name__ == "__main__":
    main()
