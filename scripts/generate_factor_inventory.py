#!/usr/bin/env python3
"""Factor Inventory Scanner

Automatically catalogs every factor in the project and its status:
- Champion cache columns (what's actively used in training)
- FeatureMerger loaders (all available data sources)
- Candidate factors (gate pass/fail status)
- External parquets not yet wired into any loader

Output: data/storage/factor_inventory.json + summary table to stdout.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "storage"
CANDIDATE_DIR = DATA_DIR / "candidate_factors"


# ---------------------------------------------------------------------------
# 1. Champion cache columns
# ---------------------------------------------------------------------------
def scan_champion_cache() -> tuple[str, list[str]]:
    """Return (cache_filename, sorted list of columns)."""
    cache_name = "feature_cache_174_holder_regime_ma.parquet"
    path = DATA_DIR / cache_name
    if not path.exists():
        return cache_name, []
    df = pd.read_parquet(path, engine="pyarrow")
    return cache_name, sorted(df.columns.tolist())


# ---------------------------------------------------------------------------
# 2. Classify champion columns into groups
# ---------------------------------------------------------------------------
# Alpha158 base names (without window suffix)
ALPHA158_BASES = {
    "BETA", "CNTD", "CNTN", "CNTP", "CORD", "CORR",
    "HIGH", "IMAX", "IMIN", "IMXD", "KLEN", "KLOW", "KLOW2",
    "KMID", "KMID2", "KSFT", "KSFT2", "KUP", "KUP2", "LOW",
    "MA", "MAX", "MIN", "OPEN", "QTLD", "QTLU", "RANK",
    "RESI", "ROC", "RSQR", "RSV", "STD", "SUMD", "SUMN",
    "SUMP", "VMA", "VSTD", "VSUMD", "VSUMN", "VSUMP", "VWAP",
    "WVMA",
}

CUSTOM_EXPR_NAMES = {
    "pe", "pb", "turn_raw", "amount_raw",
    "pe_mom20", "pb_mom20", "turn_anom20", "turn_anom60",
    "amount_anom20", "turn_vol20", "ep", "bp", "price_pos20",
}

CAPITAL_FLOW_NAMES = {
    "flow_net_mf_latest", "flow_net_mf_5d", "flow_net_mf_20d_avg",
}

HOLDER_NAMES = {"holder_num"}

# Cross-market regime prefixes
CROSS_MARKET_PREFIXES = ("hsi_", "hstech_", "nasdaq_")

# Internal / non-feature columns
INTERNAL_COLS = {"__label_5d", "__pnl_return_1d", "_close", "_ma5", "_ma20"}


def classify_champion_column(col: str) -> tuple[str, str]:
    """Return (group, source) for a champion cache column."""
    if col in INTERNAL_COLS:
        return "internal", "cache_derived"
    if col in CUSTOM_EXPR_NAMES:
        return "custom_expr", "qlib_expressions"
    if col in CAPITAL_FLOW_NAMES:
        return "capital_flow", "fund_flow_history.parquet"
    if col in HOLDER_NAMES:
        return "holder", "st_holder_number.parquet"
    if any(col.startswith(p) for p in CROSS_MARKET_PREFIXES):
        return "cross_market_regime", "cross_market_indices.parquet"

    # Alpha158: strip trailing digits to get base name
    import re
    m = re.match(r"^([A-Z]+)\d*$", col)
    if m and m.group(1) in ALPHA158_BASES:
        return "alpha158", "qlib_alpha158"

    return "unknown", "unknown"


# ---------------------------------------------------------------------------
# 3. Catalog FeatureMerger loaders and their factors
# ---------------------------------------------------------------------------
LOADER_REGISTRY = [
    {
        "loader": "_load_fundamental",
        "parquet": "fundamental_features.parquet",  # not present currently
        "group": "fundamental",
        "factors_if_missing": ["fund_pe_ttm", "fund_pb", "fund_ep", "fund_bp",
                               "fund_log_mv", "fund_log_circ_mv", "fund_roe",
                               "fund_roa", "fund_gross_margin", "fund_net_margin",
                               "fund_debt_ratio", "fund_revenue_growth",
                               "fund_profit_growth"],
    },
    {
        "loader": "_load_capital_flow",
        "parquet": "fund_flow_history.parquet",
        "group": "capital_flow",
        "factor_cols": ["flow_net_mf_latest", "flow_net_mf_5d", "flow_net_mf_20d_avg"],
    },
    {
        "loader": "_load_macro",
        "parquet": "macro_features.parquet",
        "group": "macro",
        "prefix": "macro_",
    },
    {
        "loader": "_load_shareholder",
        "parquet": "shareholder_features.parquet",
        "group": "shareholder",
        "factor_cols_raw": ["holder_count", "holder_count_change", "pledge_ratio",
                            "total_share", "liquid_share", "liquid_ratio"],
        "prefix": "holder_",
    },
    {
        "loader": "_load_valuation",
        "parquet": "fundamental_valuation.parquet",
        "group": "valuation",
        "factor_cols_raw": ["pe_ttm", "pb_mrq", "ps_ttm", "ep", "bp", "sp", "pcf_ncf_ttm"],
        "prefix": "val_",
    },
    {
        "loader": "_load_northbound",
        "parquet": "northbound_history.parquet",
        "group": "northbound",
        "factor_cols": ["nb_hold_change_5d", "nb_hold_change_20d",
                        "nb_hold_ratio", "nb_ratio_change_5d"],
    },
    {
        "loader": "_load_quality",
        "parquet": "fundamental_quality.parquet",
        "group": "quality",
        "factor_cols_raw": ["roe", "net_margin", "gross_margin", "eps_ttm",
                            "asset_turnover", "equity_multiplier",
                            "yoy_net_profit", "yoy_revenue"],
        "prefix": "qual_",
    },
    {
        "loader": "_load_st_daily_basic",
        "parquet": "st_daily_basic.parquet",
        "group": "st_daily_basic",
    },
    {
        "loader": "_load_st_moneyflow",
        "parquet": "st_moneyflow.parquet",
        "group": "st_moneyflow",
    },
    {
        "loader": "_load_st_holder_number",
        "parquet": "st_holder_number.parquet",
        "group": "holder",
        "factor_cols": ["holder_num"],
    },
    {
        "loader": "_load_cross_market_regime",
        "parquet": "cross_market_indices.parquet",
        "group": "cross_market_regime",
    },
]

# Extra parquets that exist but have no loader in FeatureMerger
EXTRA_PARQUET_GROUPS = {
    "fundamental_factors_pit.parquet": "fundamental_pit",
    "derived_moneyflow_cyq.parquet": "moneyflow_derived",
    "event_factors_v2.parquet": "event",
    "guba_factors.parquet": "sentiment",
    "llm_event_factors.parquet": "llm_event",
    "moneyflow_v2.parquet": "moneyflow_v2",
    "regime_vectors_cache.parquet": "regime",
    "sector_spillover_features.parquet": "sector_spillover",
    "block_trade_v2.parquet": "block_trade",
    "feature_cache_alpha360.parquet": "alpha360_cache",
}

PIT_SAFE_LEVELS = {
    "qlib_alpha158": "safe (qlib internal)",
    "qlib_expressions": "safe (qlib internal)",
    "fund_flow_history.parquet": "safe (T+1 lag)",
    "st_daily_basic.parquet": "safe (T+1 lag)",
    "st_moneyflow.parquet": "safe (T+1 lag)",
    "st_holder_number.parquet": "safe (ann_date+1BDay)",
    "cross_market_indices.parquet": "safe (broadcast by date)",
    "fundamental_valuation.parquet": "safe (asof merge)",
    "fundamental_quality.parquet": "safe (PIT via stat_date)",
    "northbound_history.parquet": "safe (T+1 lag)",
    "shareholder_features.parquet": "safe (PIT via stat_date)",
    "macro_features.parquet": "caution (broadcast latest)",
    "fundamental_factors_pit.parquet": "safe (PIT aligned)",
    "derived_moneyflow_cyq.parquet": "unknown",
    "event_factors_v2.parquet": "unknown",
    "guba_factors.parquet": "unknown",
    "llm_event_factors.parquet": "unknown",
    "moneyflow_v2.parquet": "unknown",
    "regime_vectors_cache.parquet": "safe (market-level)",
    "sector_spillover_features.parquet": "unknown",
    "block_trade_v2.parquet": "unknown",
}


def get_parquet_factor_cols(parquet_name: str) -> list[str]:
    """Read a parquet and return its factor columns (excluding metadata cols)."""
    path = DATA_DIR / parquet_name
    if not path.exists():
        return []
    try:
        df = pd.read_parquet(path)
        meta_cols = {"qlib_code", "date", "trade_date", "ts_code", "code",
                     "ann_date", "end_date", "stat_date", "period",
                     "collected_at", "signal_date", "name", "type",
                     "type_name", "close"}
        return [c for c in df.columns if c not in meta_cols]
    except Exception:
        return []


def build_loader_factors(champion_cols: set) -> list[dict]:
    """Build factor entries from FeatureMerger loader registry."""
    factors = []
    for reg in LOADER_REGISTRY:
        parquet = reg["parquet"]
        loader = reg["loader"]
        group = reg["group"]
        pit = PIT_SAFE_LEVELS.get(parquet, "unknown")

        # Determine factor columns
        if "factor_cols" in reg:
            cols = reg["factor_cols"]
        elif "factor_cols_raw" in reg:
            prefix = reg.get("prefix", "")
            cols = [f"{prefix}{c}" for c in reg["factor_cols_raw"]]
        elif "factors_if_missing" in reg:
            cols = reg["factors_if_missing"]
        else:
            # Auto-detect from parquet
            raw_cols = get_parquet_factor_cols(parquet)
            prefix = reg.get("prefix", "")
            if prefix:
                cols = [f"{prefix}{c}" for c in raw_cols]
            else:
                cols = raw_cols

        parquet_exists = (DATA_DIR / parquet).exists()

        for col in cols:
            entry = {
                "name": col,
                "group": group,
                "in_champion": col in champion_cols,
                "source": parquet,
                "loader": loader,
                "parquet_exists": parquet_exists,
                "pit_safe": pit,
            }
            factors.append(entry)

    return factors


def build_extra_parquet_factors(champion_cols: set, already_sourced: set) -> list[dict]:
    """Build factor entries from parquets that have no FeatureMerger loader."""
    factors = []
    for parquet, group in EXTRA_PARQUET_GROUPS.items():
        cols = get_parquet_factor_cols(parquet)
        pit = PIT_SAFE_LEVELS.get(parquet, "unknown")
        for col in cols:
            if col in already_sourced:
                continue
            entry = {
                "name": col,
                "group": group,
                "in_champion": col in champion_cols,
                "source": parquet,
                "loader": None,
                "parquet_exists": True,
                "pit_safe": pit,
            }
            factors.append(entry)
    return factors


# ---------------------------------------------------------------------------
# 4. Candidate factors
# ---------------------------------------------------------------------------
def scan_candidates() -> list[dict]:
    """Scan candidate_factors/ for gate results."""
    results = []
    if not CANDIDATE_DIR.exists():
        return results
    for d in sorted(CANDIDATE_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        verdict_path = d / "verdict.json"
        verdict = "unknown"
        updated_at = None
        if verdict_path.exists():
            with open(verdict_path) as f:
                vdata = json.load(f)
                verdict = vdata.get("verdict", "unknown")
                updated_at = vdata.get("updated_at")

        tearsheet = None
        ts_path = d / "tearsheet.json"
        if ts_path.exists():
            try:
                with open(ts_path) as f:
                    tearsheet = json.load(f)
            except Exception:
                pass

        results.append({
            "name": name,
            "verdict": verdict,
            "updated_at": updated_at,
            "has_tearsheet": tearsheet is not None,
            "ic_mean": tearsheet.get("ic_mean") if tearsheet else None,
            "rank_ic_mean": tearsheet.get("rank_ic_mean") if tearsheet else None,
        })
    return results


# ---------------------------------------------------------------------------
# 5. Main assembly
# ---------------------------------------------------------------------------
def main():
    cache_name, champion_cols_list = scan_champion_cache()
    champion_cols = set(champion_cols_list)

    # Separate internal vs feature columns
    feature_cols = [c for c in champion_cols_list if c not in INTERNAL_COLS]
    internal_cols = [c for c in champion_cols_list if c in INTERNAL_COLS]

    # Build champion factor entries
    champion_factors = []
    for col in champion_cols_list:
        group, source = classify_champion_column(col)
        pit = PIT_SAFE_LEVELS.get(source, "n/a")
        champion_factors.append({
            "name": col,
            "group": group,
            "in_champion": True,
            "source": source,
            "pit_safe": pit,
        })

    # Build loader factors (available via FeatureMerger)
    loader_factors = build_loader_factors(champion_cols)
    already_sourced = {f["name"] for f in champion_factors} | {f["name"] for f in loader_factors}

    # Build extra parquet factors (no loader)
    extra_factors = build_extra_parquet_factors(champion_cols, already_sourced)

    # Merge: champion entries take priority
    all_factors_map = {}  # name -> entry
    # Start with loader + extra (lower priority)
    for f in loader_factors + extra_factors:
        all_factors_map[f["name"]] = f
    # Champion entries override
    for f in champion_factors:
        all_factors_map[f["name"]] = f

    all_factors = sorted(all_factors_map.values(), key=lambda x: (x["group"], x["name"]))

    # Candidates
    candidates = scan_candidates()

    # Summary stats
    in_champion_count = len(feature_cols)
    available_not_in_champion = sum(
        1 for f in all_factors
        if not f["in_champion"] and f["group"] != "internal"
    )
    candidate_pass = sum(1 for c in candidates if c["verdict"] == "pass")
    candidate_fail = sum(1 for c in candidates if c["verdict"] == "fail")
    candidate_pending = sum(1 for c in candidates if c["verdict"] == "unknown")

    # Group breakdown
    group_counts = {}
    for f in all_factors:
        g = f["group"]
        if g not in group_counts:
            group_counts[g] = {"in_champion": 0, "available": 0}
        if f["in_champion"]:
            group_counts[g]["in_champion"] += 1
        else:
            group_counts[g]["available"] += 1

    inventory = {
        "generated_at": datetime.now().isoformat(),
        "champion_cache": cache_name,
        "champion_feature_count": in_champion_count,
        "champion_total_columns": len(champion_cols_list),
        "summary": {
            "in_champion": in_champion_count,
            "internal_cols": len(internal_cols),
            "available_not_in_champion": available_not_in_champion,
            "candidate_total": len(candidates),
            "candidate_pass": candidate_pass,
            "candidate_fail": candidate_fail,
            "candidate_pending": candidate_pending,
        },
        "group_breakdown": {
            g: counts for g, counts in sorted(group_counts.items())
        },
        "factors": all_factors,
        "candidates": candidates,
    }

    # Write JSON
    out_path = DATA_DIR / "factor_inventory.json"
    with open(out_path, "w") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False, default=str)
    print(f"Written: {out_path}")

    # --------------- Pretty-print summary ---------------
    print("\n" + "=" * 80)
    print(f"FACTOR INVENTORY  |  Generated {inventory['generated_at']}")
    print(f"Champion cache: {cache_name}")
    print("=" * 80)

    s = inventory["summary"]
    print(f"\n  In champion (features):      {s['in_champion']}")
    print(f"  Internal columns:            {s['internal_cols']}")
    print(f"  Available (not in champion): {s['available_not_in_champion']}")
    print(f"  Candidate factors tested:    {s['candidate_total']}")
    print(f"    - pass: {s['candidate_pass']}  fail: {s['candidate_fail']}  pending: {s['candidate_pending']}")

    print(f"\n{'GROUP':<25} {'IN CHAMPION':>12} {'AVAILABLE':>10} {'TOTAL':>8}")
    print("-" * 58)
    for g, counts in sorted(group_counts.items()):
        total = counts["in_champion"] + counts["available"]
        print(f"  {g:<23} {counts['in_champion']:>10} {counts['available']:>10} {total:>8}")
    total_in = sum(c["in_champion"] for c in group_counts.values())
    total_avail = sum(c["available"] for c in group_counts.values())
    print("-" * 58)
    print(f"  {'TOTAL':<23} {total_in:>10} {total_avail:>10} {total_in + total_avail:>8}")

    # Candidate gate results
    if candidates:
        print(f"\n{'CANDIDATE FACTOR':<30} {'VERDICT':>8} {'IC':>10} {'RANK IC':>10}")
        print("-" * 62)
        for c in candidates:
            ic = f"{c['ic_mean']:.4f}" if c["ic_mean"] is not None else "n/a"
            ric = f"{c['rank_ic_mean']:.4f}" if c["rank_ic_mean"] is not None else "n/a"
            marker = " *" if c["verdict"] == "pass" else ""
            print(f"  {c['name']:<28} {c['verdict']:>8} {ic:>10} {ric:>10}{marker}")

    # Factors available but not in champion (opportunities)
    not_in_champ = [f for f in all_factors if not f["in_champion"] and f["group"] != "internal"]
    if not_in_champ:
        print(f"\n--- Available but NOT in champion ({len(not_in_champ)} factors) ---")
        print(f"{'FACTOR':<35} {'GROUP':<20} {'SOURCE':<35} {'PIT SAFE'}")
        print("-" * 110)
        for f in not_in_champ[:50]:
            print(f"  {f['name']:<33} {f['group']:<20} {f['source']:<35} {f.get('pit_safe', 'n/a')}")
        if len(not_in_champ) > 50:
            print(f"  ... and {len(not_in_champ) - 50} more")

    print()


if __name__ == "__main__":
    main()
