"""Phase 4L: Moneyflow v2 derived factors through Alpha Factory gate.

Builds 4 derived factors from st_moneyflow.parquet:
  1. net_flow_zscore   — net flow ratio, 60d rolling zscore
  2. large_order_ratio — large+extra-large buy / total buy
  3. flow_momentum     — 5d MA of net flow minus 20d MA
  4. flow_persistence  — sign(net_flow) consistency over last 10 days

Each factor is processed through full_pipeline, then evaluated via
Alpha Factory tearsheet.

Usage: python scripts/phase4l_moneyflow_v2_gate.py
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

    # Detect column naming (Chinese vs English)
    cols = mf.columns.tolist()
    logger.info(f"  Columns: {cols}")

    # Set up MultiIndex (datetime, instrument)
    # The parquet has columns: qlib_code, date, and various flow cols
    if "qlib_code" in cols and "date" in cols:
        mf = mf.rename(columns={"qlib_code": "instrument", "date": "datetime"})
        mf["datetime"] = pd.to_datetime(mf["datetime"])
        mf = mf.set_index(["datetime", "instrument"]).sort_index()
        # Remove duplicate index entries (keep last)
        mf = mf[~mf.index.duplicated(keep="last")]
    elif "instrument" in mf.index.names and "datetime" in mf.index.names:
        pass  # already indexed
    else:
        raise ValueError(f"Cannot determine index columns from: {cols}")

    logger.info(f"  After indexing: {mf.shape}, "
                f"date range: {mf.index.get_level_values(0).min()} ~ "
                f"{mf.index.get_level_values(0).max()}")
    return mf


def detect_columns(mf):
    """Detect and map column names flexibly."""
    cols = set(mf.columns)

    # Expected column patterns (English from st_client)
    mapping = {}

    # Buy/sell amounts
    # Small
    for prefix in ["st_buy_sm_amount", "buy_sm_amount", "small_buy_amount"]:
        if prefix in cols:
            mapping["buy_sm_amount"] = prefix
            break
    for prefix in ["st_sell_sm_amount", "sell_sm_amount", "small_sell_amount"]:
        if prefix in cols:
            mapping["sell_sm_amount"] = prefix
            break

    # Medium
    for prefix in ["st_buy_md_amount", "buy_md_amount", "medium_buy_amount"]:
        if prefix in cols:
            mapping["buy_md_amount"] = prefix
            break
    for prefix in ["st_sell_md_amount", "sell_md_amount", "medium_sell_amount"]:
        if prefix in cols:
            mapping["sell_md_amount"] = prefix
            break

    # Large
    for prefix in ["st_buy_lg_amount", "buy_lg_amount", "large_buy_amount"]:
        if prefix in cols:
            mapping["buy_lg_amount"] = prefix
            break
    for prefix in ["st_sell_lg_amount", "sell_lg_amount", "large_sell_amount"]:
        if prefix in cols:
            mapping["sell_lg_amount"] = prefix
            break

    # Extra-large
    for prefix in ["st_buy_elg_amount", "buy_elg_amount", "xlarge_buy_amount"]:
        if prefix in cols:
            mapping["buy_elg_amount"] = prefix
            break
    for prefix in ["st_sell_elg_amount", "sell_elg_amount", "xlarge_sell_amount"]:
        if prefix in cols:
            mapping["sell_elg_amount"] = prefix
            break

    # Net money flow
    for prefix in ["st_net_mf_amount", "net_mf_amount"]:
        if prefix in cols:
            mapping["net_mf_amount"] = prefix
            break

    logger.info(f"  Column mapping: {mapping}")
    return mapping


def build_factors(mf, col_map):
    """Build 4 derived moneyflow factors."""
    factors = {}

    # --- 1. net_flow_zscore ---
    # (buy_amount - sell_amount) / total_amount, then 60d rolling zscore
    logger.info("Building net_flow_zscore...")

    total_buy = pd.Series(0.0, index=mf.index)
    total_sell = pd.Series(0.0, index=mf.index)
    for size in ["sm", "md", "lg", "elg"]:
        buy_key = f"buy_{size}_amount"
        sell_key = f"sell_{size}_amount"
        if buy_key in col_map:
            total_buy = total_buy + mf[col_map[buy_key]].fillna(0)
        if sell_key in col_map:
            total_sell = total_sell + mf[col_map[sell_key]].fillna(0)

    total_amount = total_buy + total_sell
    net_flow_ratio = (total_buy - total_sell) / total_amount.replace(0, np.nan)

    # 60d rolling zscore per instrument
    unstacked = net_flow_ratio.unstack(level="instrument")
    rolling_mean = unstacked.rolling(60, min_periods=20).mean()
    rolling_std = unstacked.rolling(60, min_periods=20).std()
    zscore_unstacked = (unstacked - rolling_mean) / rolling_std.replace(0, np.nan)
    try:
        net_flow_zscore = zscore_unstacked.stack(dropna=False)
    except (ValueError, TypeError):
        net_flow_zscore = zscore_unstacked.stack(future_stack=True)
    net_flow_zscore.index.names = ["datetime", "instrument"]
    net_flow_zscore.name = "net_flow_zscore"
    factors["net_flow_zscore"] = net_flow_zscore
    logger.info(f"  net_flow_zscore: {net_flow_zscore.notna().sum():,} non-NaN values")

    # --- 2. large_order_ratio ---
    # (large_buy + xlarge_buy) / total_buy
    logger.info("Building large_order_ratio...")

    large_buy = pd.Series(0.0, index=mf.index)
    if "buy_lg_amount" in col_map:
        large_buy = large_buy + mf[col_map["buy_lg_amount"]].fillna(0)
    if "buy_elg_amount" in col_map:
        large_buy = large_buy + mf[col_map["buy_elg_amount"]].fillna(0)

    large_order_ratio = large_buy / total_buy.replace(0, np.nan)
    large_order_ratio.name = "large_order_ratio"
    factors["large_order_ratio"] = large_order_ratio
    logger.info(f"  large_order_ratio: {large_order_ratio.notna().sum():,} non-NaN values")

    # --- 3. flow_momentum ---
    # 5d rolling mean of net_flow - 20d rolling mean
    logger.info("Building flow_momentum...")

    # Use net_mf_amount if available, otherwise use computed net flow
    if "net_mf_amount" in col_map:
        net_flow = mf[col_map["net_mf_amount"]]
    else:
        net_flow = total_buy - total_sell

    unstacked_nf = net_flow.unstack(level="instrument")
    ma5 = unstacked_nf.rolling(5, min_periods=3).mean()
    ma20 = unstacked_nf.rolling(20, min_periods=10).mean()
    flow_mom_unstacked = ma5 - ma20
    try:
        flow_momentum = flow_mom_unstacked.stack(dropna=False)
    except (ValueError, TypeError):
        flow_momentum = flow_mom_unstacked.stack(future_stack=True)
    flow_momentum.index.names = ["datetime", "instrument"]
    flow_momentum.name = "flow_momentum"
    factors["flow_momentum"] = flow_momentum
    logger.info(f"  flow_momentum: {flow_momentum.notna().sum():,} non-NaN values")

    # --- 4. flow_persistence ---
    # sign(net_flow) count over last 10 days / 10
    logger.info("Building flow_persistence...")

    net_flow_sign = np.sign(net_flow)
    unstacked_sign = net_flow_sign.unstack(level="instrument")
    # Count positive signs over 10d window (rolling sum of sign > 0)
    positive_count = (unstacked_sign > 0).astype(float).rolling(10, min_periods=5).sum()
    flow_persist_unstacked = positive_count / 10.0
    try:
        flow_persistence = flow_persist_unstacked.stack(dropna=False)
    except (ValueError, TypeError):
        flow_persistence = flow_persist_unstacked.stack(future_stack=True)
    flow_persistence.index.names = ["datetime", "instrument"]
    flow_persistence.name = "flow_persistence"
    factors["flow_persistence"] = flow_persistence
    logger.info(f"  flow_persistence: {flow_persistence.notna().sum():,} non-NaN values")

    return factors


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    mf = load_moneyflow()
    col_map = detect_columns(mf)

    # Load feature cache for forward returns
    logger.info(f"\nLoading feature cache for forward returns: {CACHE_PATH}")
    cache = pd.read_parquet(CACHE_PATH, columns=[RETURNS_COL])
    fwd_returns = cache[RETURNS_COL].copy()
    fwd_returns.name = "fwd_ret"
    logger.info(f"  Forward returns: {fwd_returns.notna().sum():,} non-NaN values")

    # ------------------------------------------------------------------
    # 2. Build factors
    # ------------------------------------------------------------------
    factors = build_factors(mf, col_map)

    # ------------------------------------------------------------------
    # 3. Validate each factor through Alpha Factory
    # ------------------------------------------------------------------
    factory = AlphaFactory()
    results = []

    factor_descriptions = {
        "net_flow_zscore": "Net flow ratio (buy-sell)/total, 60d rolling zscore",
        "large_order_ratio": "Large+XL buy amount / total buy amount",
        "flow_momentum": "5d MA of net flow minus 20d MA",
        "flow_persistence": "Fraction of positive net flow days in last 10d",
    }

    for factor_name, raw_factor in factors.items():
        description = factor_descriptions.get(factor_name, factor_name)
        logger.info(f"\n--- Processing: {factor_name} ({description}) ---")

        # Process through full_pipeline (fillna + winsorize + zscore)
        try:
            processed = full_pipeline(raw_factor.dropna())
        except Exception as e:
            logger.error(f"  Pipeline failed for {factor_name}: {e}")
            results.append({
                "factor_name": factor_name,
                "description": description,
                "error": str(e),
            })
            continue

        logger.info(f"  Processed: {processed.notna().sum():,} non-NaN values")

        # Register with Alpha Factory
        def _make_builder(s):
            return lambda: s

        factory.register(
            name=factor_name,
            description=description,
            build_func=_make_builder(processed),
        )

        # Run tearsheet
        try:
            tearsheet = factory.run_tearsheet(factor_name, returns=fwd_returns)
            gate = factory.check_gate(factor_name)
        except Exception as e:
            logger.error(f"  Tearsheet failed for {factor_name}: {e}")
            results.append({
                "factor_name": factor_name,
                "description": description,
                "error": str(e),
            })
            continue

        results.append({
            "factor_name": factor_name,
            "description": description,
            "rank_ic": tearsheet.get("rank_ic_mean", 0.0),
            "rank_icir": tearsheet.get("rank_icir", 0.0),
            "ic_mean": tearsheet.get("ic_mean", 0.0),
            "coverage": tearsheet.get("coverage", 0.0),
            "spread_q1_q5": tearsheet.get("spread_q1_q5"),
            "neg_ctrl_ic": tearsheet.get("negative_control_ic", 0.0),
            "autocorr_1d": tearsheet.get("autocorr_1d"),
            "autocorr_5d": tearsheet.get("autocorr_5d"),
            "n_days": tearsheet.get("n_days", 0),
            "n_obs": tearsheet.get("n_obs", 0),
            "gate": gate.get("verdict", "fail"),
            "gate_failures": gate.get("failures", []),
        })

        logger.info(
            f"  RankIC={tearsheet.get('rank_ic_mean', 0):.4f}  "
            f"ICIR={tearsheet.get('rank_icir', 0):.4f}  "
            f"Coverage={tearsheet.get('coverage', 0):.2f}  "
            f"Gate={gate.get('verdict')}"
        )

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("Moneyflow v2 Factor Validation Results (Phase 4L)")
    print("=" * 100)

    valid_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]

    if valid_results:
        summary_df = pd.DataFrame(valid_results)
        display_cols = [
            "factor_name", "rank_ic", "rank_icir", "ic_mean",
            "coverage", "spread_q1_q5", "neg_ctrl_ic",
            "autocorr_1d", "n_days", "gate",
        ]
        display_df = summary_df[[c for c in display_cols if c in summary_df.columns]]
        display_df = display_df.sort_values("rank_ic", ascending=False, key=abs)

        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 140)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(display_df.to_string(index=False))

    if error_results:
        print("\n--- Errors ---")
        for r in error_results:
            print(f"  {r['factor_name']}: {r['error']}")

    # Alpha Factory summary
    print("\n--- Alpha Factory Summary ---")
    af_summary = factory.summary_table()
    if not af_summary.empty:
        # Only show moneyflow factors
        mf_names = list(factors.keys())
        af_mf = af_summary[af_summary["name"].isin(mf_names)]
        if not af_mf.empty:
            show_cols = [c for c in [
                "name", "verdict", "rank_ic_mean", "rank_icir",
                "coverage", "spread_q1_q5", "negative_control_ic",
            ] if c in af_mf.columns]
            print(af_mf[show_cols].to_string(index=False))

    # Gate details
    print("\n--- Gate Check Details ---")
    for r in valid_results:
        status = "PASS" if r["gate"] == "pass" else "FAIL"
        failures = "; ".join(r["gate_failures"]) if r["gate_failures"] else "none"
        print(f"  {r['factor_name']:25s} [{status}]  failures: {failures}")

    n_pass = sum(1 for r in valid_results if r["gate"] == "pass")
    print(f"\nTotal: {n_pass}/{len(valid_results)} factors passed gate")

    return 0


if __name__ == "__main__":
    sys.exit(main())
