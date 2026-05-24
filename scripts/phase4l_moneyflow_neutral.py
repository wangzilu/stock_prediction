"""Phase 4L-2: Moneyflow large_order_ratio — industry & size neutralization.

Compare four variants of flipped large_order_ratio:
  a. Raw flipped (baseline, already passed gate)
  b. Industry-neutral (demean by SW L1 industry)
  c. Size-neutral (regress out log amount)
  d. Both: industry then size neutral

Uses the same base computation as phase4l_moneyflow_flip.py.
Industry labels from JQData SW classification; size proxy from amount_raw.

Usage: python scripts/phase4l_moneyflow_neutral.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from factors.processors import full_pipeline, industry_neutralize, size_neutralize
from tracker.alpha_factory import AlphaFactory

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MONEYFLOW_PATH = PROJECT_ROOT / "data" / "storage" / "st_moneyflow.parquet"
CACHE_PATH = PROJECT_ROOT / "data" / "storage" / "feature_cache_174_holder_regime_ma.parquet"
INDUSTRY_PATH = PROJECT_ROOT / "data" / "storage" / "jqdata" / "industry_sw.parquet"
RETURNS_COL = "__pnl_return_1d"


# ---------------------------------------------------------------------------
# Reuse load/compute helpers from phase4l_moneyflow_flip
# ---------------------------------------------------------------------------
from scripts.phase4l_moneyflow_flip import (
    load_moneyflow,
    detect_columns,
    compute_large_order_ratio,
    check_quintile_monotonicity,
)


def load_industry_labels(index: pd.MultiIndex) -> pd.Series:
    """Build (datetime, instrument) industry Series from JQData SW classification."""
    if not INDUSTRY_PATH.exists():
        logger.warning("Industry file not found: %s", INDUSTRY_PATH)
        return pd.Series(dtype=str)

    ind = pd.read_parquet(INDUSTRY_PATH)
    code_map = {}
    for _, row in ind.iterrows():
        jq_code = str(row.get("code", ""))
        sw_l1 = str(row.get("sw_l1_name", ""))
        if not sw_l1:
            continue
        if ".XSHE" in jq_code:
            qlib = f"sz{jq_code[:6]}"
        elif ".XSHG" in jq_code:
            qlib = f"sh{jq_code[:6]}"
        else:
            continue
        code_map[qlib] = sw_l1

    instruments = index.get_level_values("instrument")
    labels = instruments.map(code_map)
    result = pd.Series(labels.values, index=index, name="industry")
    n_mapped = result.notna().sum()
    logger.info("Industry labels: %d / %d mapped (%.1f%%)",
                n_mapped, len(result), 100 * n_mapped / max(len(result), 1))
    return result


def load_size_proxy(index: pd.MultiIndex) -> pd.Series:
    """Load log(amount_raw) as size proxy, aligned to given index."""
    cache = pd.read_parquet(CACHE_PATH, columns=["amount_raw"])
    amount = cache["amount_raw"].replace(0, np.nan)
    log_amount = np.log1p(amount.clip(lower=1))
    log_amount.name = "log_amount"

    common = index.intersection(log_amount.index)
    result = log_amount.loc[common]
    logger.info("Size proxy: %d / %d matched (%.1f%%)",
                len(result), len(index), 100 * len(result) / max(len(index), 1))
    return result


def main():
    # ------------------------------------------------------------------
    # 1. Load and compute flipped large_order_ratio
    # ------------------------------------------------------------------
    mf = load_moneyflow()
    col_map = detect_columns(mf)

    logger.info("Loading feature cache for forward returns: %s", CACHE_PATH)
    cache = pd.read_parquet(CACHE_PATH, columns=[RETURNS_COL])
    fwd_returns = cache[RETURNS_COL].copy()
    fwd_returns.name = "fwd_ret"
    logger.info("Forward returns: %d non-NaN values", fwd_returns.notna().sum())

    large_order_ratio = compute_large_order_ratio(mf, col_map)
    flipped = -large_order_ratio
    flipped.name = "large_order_ratio_flipped"
    logger.info("Flipped factor: %d non-NaN values", flipped.notna().sum())

    # ------------------------------------------------------------------
    # 2. Load industry and size
    # ------------------------------------------------------------------
    flipped_clean = flipped.dropna()
    industry = load_industry_labels(flipped_clean.index)
    log_amount = load_size_proxy(flipped_clean.index)

    # ------------------------------------------------------------------
    # 3. Compute four variants
    # ------------------------------------------------------------------
    variants = {}

    # a. Raw flipped — baseline (winsorize + zscore only)
    logger.info("\n--- Variant A: Raw flipped (winsorize + zscore) ---")
    variants["flipped_raw"] = full_pipeline(flipped_clean)

    # b. Industry-neutral
    logger.info("--- Variant B: Industry-neutral ---")
    ind_neutral = industry_neutralize(flipped_clean, industry)
    variants["flipped_ind_neutral"] = full_pipeline(ind_neutral.dropna())

    # c. Size-neutral
    logger.info("--- Variant C: Size-neutral ---")
    size_neutral = size_neutralize(flipped_clean, log_amount)
    variants["flipped_size_neutral"] = full_pipeline(size_neutral.dropna())

    # d. Both: industry then size
    logger.info("--- Variant D: Industry + Size neutral ---")
    both_neutral = industry_neutralize(flipped_clean, industry)
    both_neutral = size_neutralize(both_neutral.dropna(), log_amount)
    variants["flipped_both_neutral"] = full_pipeline(both_neutral.dropna())

    # ------------------------------------------------------------------
    # 4-6. Tearsheet + monotonicity for each variant
    # ------------------------------------------------------------------
    factory = AlphaFactory()
    results = []

    descriptions = {
        "flipped_raw": "Flipped (baseline), winsorize+zscore",
        "flipped_ind_neutral": "Flipped, industry-neutral (SW L1)",
        "flipped_size_neutral": "Flipped, size-neutral (log amount)",
        "flipped_both_neutral": "Flipped, industry + size neutral",
    }

    for name, factor in variants.items():
        desc = descriptions[name]
        logger.info("\n=== Evaluating: %s (%s) ===", name, desc)

        factory.register(
            name=name,
            description=desc,
            build_func=lambda s=factor: s,
        )

        try:
            tearsheet = factory.run_tearsheet(name, returns=fwd_returns)
            gate = factory.check_gate(name)
        except Exception as e:
            logger.error("Tearsheet failed for %s: %s", name, e)
            results.append({"name": name, "description": desc, "error": str(e)})
            continue

        mono = check_quintile_monotonicity(factor, fwd_returns)

        result = {
            "name": name,
            "description": desc,
            "rank_ic": tearsheet.get("rank_ic_mean", 0.0),
            "rank_icir": tearsheet.get("rank_icir", 0.0),
            "ic_mean": tearsheet.get("ic_mean", 0.0),
            "coverage": tearsheet.get("coverage", 0.0),
            "spread_q1_q5": tearsheet.get("spread_q1_q5"),
            "autocorr_1d": tearsheet.get("autocorr_1d"),
            "n_days": tearsheet.get("n_days", 0),
            "n_obs": tearsheet.get("n_obs", 0),
            "gate": gate.get("verdict", "fail"),
            "gate_failures": gate.get("failures", []),
            "monotonic": mono["is_monotonic"],
            "mono_rho": mono["spearman_rho"],
            "quintile_rets": mono["quintile_returns"],
        }
        results.append(result)

        logger.info(
            "  RankIC=%.4f  ICIR=%.4f  Gate=%s  Monotonic=%s  MonoRho=%.3f",
            result["rank_ic"], result["rank_icir"],
            result["gate"], result["monotonic"], result["mono_rho"],
        )

    # ------------------------------------------------------------------
    # 7. Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("Phase 4L-2: Moneyflow large_order_ratio — Industry/Size Neutralization")
    print("=" * 120)

    valid_results = [r for r in results if "error" not in r]

    if valid_results:
        summary_df = pd.DataFrame(valid_results)
        display_cols = [
            "name", "rank_ic", "rank_icir", "ic_mean",
            "coverage", "spread_q1_q5", "gate", "monotonic", "mono_rho",
        ]
        display_df = summary_df[[c for c in display_cols if c in summary_df.columns]]

        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 150)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(display_df.to_string(index=False))

    # Quintile returns
    print("\n--- Quintile Returns (Q1=lowest factor, Q5=highest factor) ---")
    for r in valid_results:
        print(f"\n  {r['name']}:")
        qr = r.get("quintile_rets", {})
        for q in range(1, 6):
            val = qr.get(q, np.nan)
            print(f"    Q{q}: {val:.6f}" if np.isfinite(val) else f"    Q{q}: NaN")
        print(f"    Monotonic: {r['monotonic']}  rho: {r.get('mono_rho', np.nan):.3f}")

    # Gate details
    print("\n--- Gate Check Details ---")
    for r in valid_results:
        status = "PASS" if r["gate"] == "pass" else "FAIL"
        failures = "; ".join(r["gate_failures"]) if r["gate_failures"] else "none"
        print(f"  {r['name']:30s} [{status}]  failures: {failures}")

    # Comparison summary
    print("\n--- Signal Preservation Analysis ---")
    baseline = next((r for r in valid_results if r["name"] == "flipped_raw"), None)
    if baseline:
        base_ic = abs(baseline["rank_ic"])
        for r in valid_results:
            if r["name"] == "flipped_raw":
                continue
            r_ic = abs(r["rank_ic"])
            pct_change = (r_ic / base_ic - 1) * 100 if base_ic > 0 else float("nan")
            verdict = "PRESERVED" if pct_change > -20 else "DESTROYED"
            print(
                f"  {r['name']:30s}  |RankIC|={r_ic:.4f}  "
                f"vs baseline {base_ic:.4f}  ({pct_change:+.1f}%)  -> {verdict}"
            )
            print(
                f"    Gate: {r['gate']}  Monotonic: {r['monotonic']}  "
                f"vs baseline gate={baseline['gate']} mono={baseline['monotonic']}"
            )
    else:
        print("  Baseline (flipped_raw) not available for comparison.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
