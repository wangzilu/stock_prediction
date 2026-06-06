"""Build the authoritative ``supp_col_manifest.json`` for LOO ablation.

Phase B (`plans/ashare-phases-2026-06.md`) needs to drop one
supplementary loader group at a time and re-run a 6-split. The
phase4e runner reads the production cache and sees raw column names;
without a column → group manifest there is no way to drop "all
capital_flow columns" precisely.

This script calls each ``FeatureMerger._load_<group>(...)`` against a
tiny synthetic Qlib index, captures the returned column names, and
writes::

    data/storage/supp_col_manifest.json
    {
      "generated_at": "...",
      "groups": {
        "capital_flow": ["flow_net_mf_latest", "flow_net_mf_5d", ...],
        "macro_zero_baseline": [...],
        ...
      }
    }

Usage::

    python scripts/introspect_supp_col_groups.py

The output JSON is read by ``phase4e_24split_ensemble.py --drop-group <X>``.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config.qlib_runtime import init_qlib
from config.settings import DATA_DIR, QLIB_PROVIDER_URI
from config.production_features import PRODUCTION_SUPPLEMENTARY_GROUPS
from models.feature_merger import FeatureMerger


def _tiny_index() -> pd.MultiIndex:
    """A small synthetic Qlib MultiIndex so each _load_<group> can run.

    Real Qlib instruments + a small date range so PIT aligners do not
    return empty. Single sample from the production universe is enough.
    """
    insts = ["SH600000", "SH600519", "SZ000001", "SZ000333"]
    dates = pd.date_range("2026-04-01", "2026-05-19", freq="B")
    return pd.MultiIndex.from_product(
        [dates, insts], names=["datetime", "instrument"]
    )


def main():
    init_qlib(QLIB_PROVIDER_URI)
    merger = FeatureMerger(DATA_DIR)
    idx = _tiny_index()

    # group_name → (loader_method_name, optional kwargs)
    LOADER_BY_GROUP = {
        "fundamental":          "_load_fundamental",
        "capital_flow":         "_load_capital_flow",
        "macro_zero_baseline":  "_load_macro",
        "shareholder":          "_load_shareholder",
        "valuation":            "_load_valuation",
        "northbound":           "_load_northbound",
        "quality":              "_load_quality",
        "st_daily_basic":       "_load_st_daily_basic",
        "st_moneyflow":         "_load_st_moneyflow",
        "st_holder_number":     "_load_st_holder_number",
        "cross_market_regime":  "_load_cross_market_regime",
    }

    # Sanity: every PRODUCTION_SUPPLEMENTARY_GROUPS entry must be covered.
    missing = set(PRODUCTION_SUPPLEMENTARY_GROUPS) - set(LOADER_BY_GROUP)
    if missing:
        raise SystemExit(
            f"PRODUCTION_SUPPLEMENTARY_GROUPS has entries with no loader "
            f"mapping in this script: {sorted(missing)}. Add them to "
            f"LOADER_BY_GROUP above."
        )

    manifest_groups: dict[str, list[str]] = {}
    total = 0
    for group, method_name in LOADER_BY_GROUP.items():
        method = getattr(merger, method_name, None)
        if method is None:
            print(f"  [skip] {group}: method {method_name} not found")
            manifest_groups[group] = []
            continue
        try:
            df = method(idx)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {group}: loader raised {type(e).__name__}: {e}")
            manifest_groups[group] = []
            continue
        if df is None:
            print(f"  [info] {group}: loader returned None (0 cols)")
            manifest_groups[group] = []
            continue
        cols = list(df.columns)
        manifest_groups[group] = cols
        total += len(cols)
        print(f"  {group}: {len(cols)} cols — {cols[:4]}{'...' if len(cols) > 4 else ''}")

    out_path = DATA_DIR / "supp_col_manifest.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "production_groups": list(PRODUCTION_SUPPLEMENTARY_GROUPS),
        "total_cols_across_groups": total,
        "groups": manifest_groups,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nManifest written: {out_path} (total {total} cols across "
          f"{len(manifest_groups)} groups)")


if __name__ == "__main__":
    main()
