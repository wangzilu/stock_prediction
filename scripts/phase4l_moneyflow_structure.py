"""Phase 4L-3: Moneyflow structural factors.

Builds 5 structural factors from st_moneyflow.parquet that capture
flow patterns beyond simple net flow:

  1. flow_persistence_5d     — sign(net_flow) consistency over 5 days
  2. flow_reversal_1d_5d     — today's flow direction vs 5d average
  3. large_small_divergence  — normalized (large_buy - small_buy) divergence
  4. flow_price_divergence   — price vs flow direction disagreement over 5d
  5. flow_surprise           — today's net_flow z-score vs 20d rolling stats

Each factor: (datetime, instrument) MultiIndex -> full_pipeline -> tearsheet.

Usage: python scripts/phase4l_moneyflow_structure.py
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
    """Detect and map column names flexibly (handles Chinese/English)."""
    cols = set(mf.columns)
    mapping = {}

    searches = {
        "buy_sm_amount":  ["st_buy_sm_amount", "buy_sm_amount", "small_buy_amount"],
        "sell_sm_amount": ["st_sell_sm_amount", "sell_sm_amount", "small_sell_amount"],
        "buy_md_amount":  ["st_buy_md_amount", "buy_md_amount", "medium_buy_amount"],
        "sell_md_amount": ["st_sell_md_amount", "sell_md_amount", "medium_sell_amount"],
        "buy_lg_amount":  ["st_buy_lg_amount", "buy_lg_amount", "large_buy_amount"],
        "sell_lg_amount": ["st_sell_lg_amount", "sell_lg_amount", "large_sell_amount"],
        "buy_elg_amount": ["st_buy_elg_amount", "buy_elg_amount", "xlarge_buy_amount"],
        "sell_elg_amount":["st_sell_elg_amount", "sell_elg_amount", "xlarge_sell_amount"],
        "net_mf_amount":  ["st_net_mf_amount", "net_mf_amount"],
    }

    for key, candidates in searches.items():
        for c in candidates:
            if c in cols:
                mapping[key] = c
                break

    logger.info(f"  Column mapping: {mapping}")
    return mapping


def _safe_stack(unstacked_df):
    """Stack with compatibility across pandas versions."""
    try:
        return unstacked_df.stack(dropna=False)
    except (ValueError, TypeError):
        return unstacked_df.stack(future_stack=True)


def build_structural_factors(mf, col_map, fwd_returns):
    """Build 5 structural moneyflow factors."""
    factors = {}

    # Precompute common quantities
    total_buy = pd.Series(0.0, index=mf.index)
    total_sell = pd.Series(0.0, index=mf.index)
    for size in ["sm", "md", "lg", "elg"]:
        bk = f"buy_{size}_amount"
        sk = f"sell_{size}_amount"
        if bk in col_map:
            total_buy = total_buy + mf[col_map[bk]].fillna(0)
        if sk in col_map:
            total_sell = total_sell + mf[col_map[sk]].fillna(0)

    if "net_mf_amount" in col_map:
        net_flow = mf[col_map["net_mf_amount"]].copy()
    else:
        net_flow = total_buy - total_sell

    net_flow_unstacked = net_flow.unstack(level="instrument")

    # --- 1. flow_persistence_5d ---
    # sign(net_flow) count over last 5 days / 5
    logger.info("Building flow_persistence_5d...")
    sign_unstacked = np.sign(net_flow_unstacked)
    pos_count_5 = (sign_unstacked > 0).astype(float).rolling(5, min_periods=3).sum()
    fp5_unstacked = pos_count_5 / 5.0
    fp5 = _safe_stack(fp5_unstacked)
    fp5.index.names = ["datetime", "instrument"]
    fp5.name = "flow_persistence_5d"
    factors["flow_persistence_5d"] = fp5
    logger.info(f"  flow_persistence_5d: {fp5.notna().sum():,} non-NaN")

    # --- 2. flow_reversal_1d_5d ---
    # sign(today's net_flow) vs sign(5d average) -> -1 if opposite, +1 if same
    logger.info("Building flow_reversal_1d_5d...")
    sign_today = np.sign(net_flow_unstacked)
    ma5_sign = np.sign(net_flow_unstacked.rolling(5, min_periods=3).mean())
    # reversal = -1 * sign_today * ma5_sign  (negative means reversal)
    reversal_unstacked = -1.0 * sign_today * ma5_sign
    reversal = _safe_stack(reversal_unstacked)
    reversal.index.names = ["datetime", "instrument"]
    reversal.name = "flow_reversal_1d_5d"
    factors["flow_reversal_1d_5d"] = reversal
    logger.info(f"  flow_reversal_1d_5d: {reversal.notna().sum():,} non-NaN")

    # --- 3. large_small_divergence ---
    # (large_buy + elg_buy) - (small_buy) normalized by total buy
    logger.info("Building large_small_divergence...")
    large_buy = pd.Series(0.0, index=mf.index)
    small_buy = pd.Series(0.0, index=mf.index)
    if "buy_lg_amount" in col_map:
        large_buy = large_buy + mf[col_map["buy_lg_amount"]].fillna(0)
    if "buy_elg_amount" in col_map:
        large_buy = large_buy + mf[col_map["buy_elg_amount"]].fillna(0)
    if "buy_sm_amount" in col_map:
        small_buy = small_buy + mf[col_map["buy_sm_amount"]].fillna(0)

    divergence = (large_buy - small_buy) / total_buy.replace(0, np.nan)
    divergence.name = "large_small_divergence"
    factors["large_small_divergence"] = divergence
    logger.info(f"  large_small_divergence: {divergence.notna().sum():,} non-NaN")

    # --- 4. flow_price_divergence ---
    # Over last 5 days: count days where sign(return) != sign(net_flow) / 5
    logger.info("Building flow_price_divergence...")
    # Compute daily returns from fwd_returns (which is T+1 forward return)
    # We need realized returns, not forward returns. Use fwd_returns shifted by 1 as past return.
    # Actually: build from price data if available, or use lagged fwd_returns
    # fwd_returns at (t-1) = return from t-1 to t = today's realized return
    ret_unstacked = fwd_returns.unstack(level="instrument")
    past_ret = ret_unstacked.shift(1)  # return realized on day t

    # Align dates
    common_dates = net_flow_unstacked.index.intersection(past_ret.index)
    common_instruments = net_flow_unstacked.columns.intersection(past_ret.columns)

    nf_aligned = net_flow_unstacked.loc[common_dates, common_instruments]
    ret_aligned = past_ret.loc[common_dates, common_instruments]

    sign_nf = np.sign(nf_aligned)
    sign_ret = np.sign(ret_aligned)
    # Divergence: 1 when signs differ, 0 when same
    disagree = (sign_nf != sign_ret).astype(float)
    disagree[sign_nf.isna() | sign_ret.isna()] = np.nan
    diverge_5d = disagree.rolling(5, min_periods=3).mean()
    fpd = _safe_stack(diverge_5d)
    fpd.index.names = ["datetime", "instrument"]
    fpd.name = "flow_price_divergence"
    factors["flow_price_divergence"] = fpd
    logger.info(f"  flow_price_divergence: {fpd.notna().sum():,} non-NaN")

    # --- 5. flow_surprise ---
    # today's net_flow z-score vs 20d rolling mean/std
    logger.info("Building flow_surprise...")
    rolling_mean_20 = net_flow_unstacked.rolling(20, min_periods=10).mean()
    rolling_std_20 = net_flow_unstacked.rolling(20, min_periods=10).std()
    zscore_unstacked = (net_flow_unstacked - rolling_mean_20) / rolling_std_20.replace(0, np.nan)
    fsurp = _safe_stack(zscore_unstacked)
    fsurp.index.names = ["datetime", "instrument"]
    fsurp.name = "flow_surprise"
    factors["flow_surprise"] = fsurp
    logger.info(f"  flow_surprise: {fsurp.notna().sum():,} non-NaN")

    return factors


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
    # 2. Build factors
    # ------------------------------------------------------------------
    factors = build_structural_factors(mf, col_map, fwd_returns)

    # ------------------------------------------------------------------
    # 3. Validate each factor through Alpha Factory
    # ------------------------------------------------------------------
    factory = AlphaFactory()
    results = []

    factor_descriptions = {
        "flow_persistence_5d": "Fraction of positive net flow days in last 5d",
        "flow_reversal_1d_5d": "Today's flow direction vs 5d average (reversal=-1)",
        "large_small_divergence": "(large+xlarge buy - small buy) / total buy",
        "flow_price_divergence": "Fraction of days price and flow disagree in last 5d",
        "flow_surprise": "Today's net flow z-score vs 20d rolling stats",
    }

    for factor_name, raw_factor in factors.items():
        description = factor_descriptions.get(factor_name, factor_name)
        logger.info(f"\n--- Processing: {factor_name} ({description}) ---")

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

        # Run tearsheet directly
        try:
            tearsheet = run_tearsheet_from_series(processed, fwd_returns)
        except Exception as e:
            logger.error(f"  Tearsheet failed for {factor_name}: {e}")
            results.append({
                "factor_name": factor_name,
                "description": description,
                "error": str(e),
            })
            continue

        # Register with Alpha Factory
        def _make_builder(s):
            return lambda: s

        factory.register(
            name=factor_name,
            description=description,
            build_func=_make_builder(processed),
        )
        try:
            factory.run_tearsheet(factor_name, returns=fwd_returns)
            gate = factory.check_gate(factor_name)
        except Exception:
            gate = {"verdict": "unknown", "failures": []}

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
    print("Moneyflow Structural Factor Validation Results (Phase 4L-3)")
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

    # Gate details
    print("\n--- Gate Check Details ---")
    for r in valid_results:
        status = "PASS" if r["gate"] == "pass" else "FAIL"
        failures = "; ".join(r["gate_failures"]) if r["gate_failures"] else "none"
        print(f"  {r['factor_name']:30s} [{status}]  failures: {failures}")

    n_pass = sum(1 for r in valid_results if r["gate"] == "pass")
    print(f"\nTotal: {n_pass}/{len(valid_results)} factors passed gate")

    return 0


if __name__ == "__main__":
    sys.exit(main())
