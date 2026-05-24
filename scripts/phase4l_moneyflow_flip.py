"""Phase 4L-1: Flip moneyflow large_order_ratio and run through Alpha Factory gate.

Hypothesis: large_order_ratio may have an inverted relationship with forward
returns (smart money sells into large buy orders). This script flips the sign
and checks if the flipped version passes the Alpha Factory gate.

Steps:
  1. Load st_moneyflow.parquet
  2. Compute large_order_ratio (same as phase4l_moneyflow_v2_gate.py)
  3. Flip sign: flipped = -large_order_ratio
  4. Process through full_pipeline (winsorize + zscore)
  5. Load forward returns (T+1) from feature cache
  6. Register with Alpha Factory, run tearsheet
  7. Check gate and quintile monotonicity

Usage: python scripts/phase4l_moneyflow_flip.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from factors.processors import full_pipeline
from tracker.alpha_factory import AlphaFactory, run_tearsheet_from_series

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MONEYFLOW_PATH = PROJECT_ROOT / "data" / "storage" / "st_moneyflow.parquet"
CACHE_PATH = PROJECT_ROOT / "data" / "storage" / "feature_cache_174_holder_regime_ma.parquet"
RETURNS_COL = "__pnl_return_1d"


def load_moneyflow():
    """Load moneyflow data and convert to (datetime, instrument) MultiIndex."""
    logger.info(f"Loading moneyflow from {MONEYFLOW_PATH}")
    mf = pd.read_parquet(MONEYFLOW_PATH)
    logger.info(f"  Shape: {mf.shape}, columns: {mf.columns.tolist()}")

    cols = mf.columns.tolist()

    if "qlib_code" in cols and "date" in cols:
        mf = mf.rename(columns={"qlib_code": "instrument", "date": "datetime"})
        mf["datetime"] = pd.to_datetime(mf["datetime"])
        mf = mf.set_index(["datetime", "instrument"]).sort_index()
        mf = mf[~mf.index.duplicated(keep="last")]
    elif "instrument" in mf.index.names and "datetime" in mf.index.names:
        pass
    else:
        raise ValueError(f"Cannot determine index columns from: {cols}")

    logger.info(f"  After indexing: {mf.shape}, "
                f"date range: {mf.index.get_level_values(0).min()} ~ "
                f"{mf.index.get_level_values(0).max()}")
    return mf


def detect_columns(mf):
    """Detect and map column names flexibly."""
    cols = set(mf.columns)
    mapping = {}

    # Buy amounts by size
    for size, label in [("sm", "small"), ("md", "medium"), ("lg", "large"), ("elg", "xlarge")]:
        for prefix in [f"st_buy_{size}_amount", f"buy_{size}_amount", f"{label}_buy_amount"]:
            if prefix in cols:
                mapping[f"buy_{size}_amount"] = prefix
                break
        for prefix in [f"st_sell_{size}_amount", f"sell_{size}_amount", f"{label}_sell_amount"]:
            if prefix in cols:
                mapping[f"sell_{size}_amount"] = prefix
                break

    logger.info(f"  Column mapping: {mapping}")
    return mapping


def compute_large_order_ratio(mf, col_map):
    """Compute large_order_ratio = (large_buy + xlarge_buy) / total_buy."""
    total_buy = pd.Series(0.0, index=mf.index)
    for size in ["sm", "md", "lg", "elg"]:
        buy_key = f"buy_{size}_amount"
        if buy_key in col_map:
            total_buy = total_buy + mf[col_map[buy_key]].fillna(0)

    large_buy = pd.Series(0.0, index=mf.index)
    if "buy_lg_amount" in col_map:
        large_buy = large_buy + mf[col_map["buy_lg_amount"]].fillna(0)
    if "buy_elg_amount" in col_map:
        large_buy = large_buy + mf[col_map["buy_elg_amount"]].fillna(0)

    large_order_ratio = large_buy / total_buy.replace(0, np.nan)
    large_order_ratio.name = "large_order_ratio"
    return large_order_ratio


def check_quintile_monotonicity(factor, returns, n_quantiles=5):
    """Check if quintile returns are monotonically increasing Q1 -> Q5.

    Returns:
        dict with quintile_returns, is_monotonic, and spearman_corr
    """
    date_level = factor.index.names[0] if factor.index.names[0] else 0
    common = factor.dropna().index.intersection(returns.dropna().index)
    f = factor.loc[common]
    r = returns.loc[common]

    dates = f.index.get_level_values(date_level).unique()

    # Collect quintile returns across all dates
    quintile_rets = {q: [] for q in range(1, n_quantiles + 1)}

    for dt in dates:
        try:
            f_day = f.xs(dt, level=date_level)
            r_day = r.xs(dt, level=date_level)
        except KeyError:
            continue

        valid = f_day.dropna().index.intersection(r_day.dropna().index)
        if len(valid) < n_quantiles * 2:
            continue

        fv = f_day.loc[valid]
        rv = r_day.loc[valid]

        try:
            quantiles = pd.qcut(fv, n_quantiles, labels=False, duplicates="drop") + 1
        except ValueError:
            continue

        for q in range(1, n_quantiles + 1):
            mask = quantiles == q
            if mask.sum() > 0:
                quintile_rets[q].append(float(rv[mask].mean()))

    # Average across dates
    avg_rets = {}
    for q in range(1, n_quantiles + 1):
        if quintile_rets[q]:
            avg_rets[q] = float(np.mean(quintile_rets[q]))
        else:
            avg_rets[q] = np.nan

    ret_values = [avg_rets[q] for q in range(1, n_quantiles + 1)]
    finite = [v for v in ret_values if np.isfinite(v)]

    # Check strict monotonicity
    is_monotonic = all(finite[i] <= finite[i + 1] for i in range(len(finite) - 1))

    # Spearman correlation of quintile rank vs return
    if len(finite) >= 3:
        rho, pval = stats.spearmanr(range(len(finite)), finite)
    else:
        rho, pval = np.nan, np.nan

    return {
        "quintile_returns": avg_rets,
        "is_monotonic": is_monotonic,
        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        "spearman_pval": float(pval) if np.isfinite(pval) else np.nan,
    }


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    mf = load_moneyflow()
    col_map = detect_columns(mf)

    logger.info(f"\nLoading feature cache for forward returns: {CACHE_PATH}")
    cache = pd.read_parquet(CACHE_PATH, columns=[RETURNS_COL])
    fwd_returns = cache[RETURNS_COL].copy()
    fwd_returns.name = "fwd_ret"
    logger.info(f"  Forward returns: {fwd_returns.notna().sum():,} non-NaN values")

    # ------------------------------------------------------------------
    # 2. Compute large_order_ratio
    # ------------------------------------------------------------------
    large_order_ratio = compute_large_order_ratio(mf, col_map)
    logger.info(f"\nlarge_order_ratio: {large_order_ratio.notna().sum():,} non-NaN values")

    # ------------------------------------------------------------------
    # 3. Flip the sign
    # ------------------------------------------------------------------
    flipped = -large_order_ratio
    flipped.name = "large_order_ratio_flipped"
    logger.info(f"Flipped: {flipped.notna().sum():,} non-NaN values")

    # ------------------------------------------------------------------
    # 4. Process through full_pipeline (winsorize + zscore)
    # ------------------------------------------------------------------
    variants = {}

    # 4a. Standard pipeline (winsorize + zscore)
    logger.info("\nProcessing flipped through full_pipeline (winsorize + zscore)...")
    processed = full_pipeline(flipped.dropna())
    variants["flipped_standard"] = processed
    logger.info(f"  Processed: {processed.notna().sum():,} non-NaN values")

    # 4b. Also process the ORIGINAL (non-flipped) for comparison
    logger.info("Processing original through full_pipeline for comparison...")
    processed_orig = full_pipeline(large_order_ratio.dropna())
    variants["original_standard"] = processed_orig

    # ------------------------------------------------------------------
    # 5-7. Register, tearsheet, gate, monotonicity
    # ------------------------------------------------------------------
    factory = AlphaFactory()
    results = []

    descriptions = {
        "flipped_standard": "FLIPPED (-1 * large_order_ratio), winsorize+zscore",
        "original_standard": "ORIGINAL large_order_ratio, winsorize+zscore (baseline)",
    }

    for name, processed_factor in variants.items():
        description = descriptions.get(name, name)
        logger.info(f"\n--- Evaluating: {name} ({description}) ---")

        def _make_builder(s):
            return lambda: s

        factory.register(
            name=name,
            description=description,
            build_func=_make_builder(processed_factor),
        )

        try:
            tearsheet = factory.run_tearsheet(name, returns=fwd_returns)
            gate = factory.check_gate(name)
        except Exception as e:
            logger.error(f"  Tearsheet failed for {name}: {e}")
            results.append({"name": name, "description": description, "error": str(e)})
            continue

        # Check quintile monotonicity
        mono = check_quintile_monotonicity(processed_factor, fwd_returns)

        result = {
            "name": name,
            "description": description,
            "rank_ic": tearsheet.get("rank_ic_mean", 0.0),
            "rank_icir": tearsheet.get("rank_icir", 0.0),
            "ic_mean": tearsheet.get("ic_mean", 0.0),
            "coverage": tearsheet.get("coverage", 0.0),
            "spread_q1_q5": tearsheet.get("spread_q1_q5"),
            "neg_ctrl_ic": tearsheet.get("negative_control_ic", 0.0),
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
            f"  RankIC={tearsheet.get('rank_ic_mean', 0):.4f}  "
            f"ICIR={tearsheet.get('rank_icir', 0):.4f}  "
            f"Gate={gate.get('verdict')}  "
            f"Monotonic={mono['is_monotonic']}  "
            f"MonoRho={mono['spearman_rho']:.3f}"
        )

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("Phase 4L-1: Moneyflow large_order_ratio FLIP Experiment")
    print("=" * 110)

    valid_results = [r for r in results if "error" not in r]

    if valid_results:
        summary_df = pd.DataFrame(valid_results)
        display_cols = [
            "name", "rank_ic", "rank_icir", "ic_mean",
            "coverage", "spread_q1_q5", "gate", "monotonic", "mono_rho",
        ]
        display_df = summary_df[[c for c in display_cols if c in summary_df.columns]]

        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 140)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(display_df.to_string(index=False))

    # Quintile returns detail
    print("\n--- Quintile Returns (Q1=lowest factor, Q5=highest factor) ---")
    for r in valid_results:
        print(f"\n  {r['name']}:")
        qr = r.get("quintile_rets", {})
        for q in range(1, 6):
            val = qr.get(q, np.nan)
            print(f"    Q{q}: {val:.6f}" if np.isfinite(val) else f"    Q{q}: NaN")
        print(f"    Monotonic (Q1->Q5 increasing): {r['monotonic']}")
        print(f"    Spearman rho: {r.get('mono_rho', np.nan):.3f}")

    # Gate details
    print("\n--- Gate Check Details ---")
    for r in valid_results:
        status = "PASS" if r["gate"] == "pass" else "FAIL"
        failures = "; ".join(r["gate_failures"]) if r["gate_failures"] else "none"
        print(f"  {r['name']:30s} [{status}]  failures: {failures}")

    # Conclusion
    print("\n--- Conclusion ---")
    flipped_r = next((r for r in valid_results if r["name"] == "flipped_standard"), None)
    orig_r = next((r for r in valid_results if r["name"] == "original_standard"), None)

    if flipped_r and orig_r:
        flip_ic = abs(flipped_r.get("rank_ic", 0))
        orig_ic = abs(orig_r.get("rank_ic", 0))
        better = "FLIPPED" if flip_ic > orig_ic else "ORIGINAL"
        print(f"  |RankIC| flipped={flip_ic:.4f}  original={orig_ic:.4f}  -> {better} is stronger")
        print(f"  Flipped gate: {flipped_r['gate']}  Original gate: {orig_r['gate']}")
        print(f"  Flipped monotonic: {flipped_r['monotonic']}  Original monotonic: {orig_r['monotonic']}")

        if flipped_r["gate"] == "pass" and flipped_r["monotonic"]:
            print("  => Flipped version PASSES gate with monotonic quintiles. USE FLIPPED.")
        elif flipped_r["gate"] == "pass":
            print("  => Flipped version PASSES gate but NOT monotonic. Review quintile returns.")
        else:
            print("  => Flipped version does NOT pass gate.")

    # Note about industry/size neutralization
    print("\n--- Note ---")
    print("  Industry-neutral and size-neutral versions skipped:")
    print("  No industry labels or market cap data available in feature cache.")
    print("  To enable, add SW industry labels and log_mcap to the cache.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
