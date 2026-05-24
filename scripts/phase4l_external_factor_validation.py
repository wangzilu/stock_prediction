#!/usr/bin/env python3
"""Phase 4L-ext -- External weak factor validation through processors + Alpha Factory.

Loads moneyflow, cyq_perf, and holder_number parquet files, derives candidate
factors, runs each through full_pipeline (winsorize + zscore), computes
standalone tearsheet and residual IC vs champion proxy, and gates via
Alpha Factory.

Usage:
    python scripts/phase4l_external_factor_validation.py
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

from factors.processors import full_pipeline, compute_residual_ic
from tracker.alpha_factory import AlphaFactory, run_tearsheet_from_series

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STORAGE = PROJECT_ROOT / "data" / "storage"
CACHE_PATH = STORAGE / "feature_cache_174_holder_regime_ma.parquet"
RETURNS_COL = "__pnl_return_1d"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cache_returns_and_champion():
    """Load forward returns and build a champion proxy from the feature cache."""
    logger.info(f"Loading feature cache from {CACHE_PATH}")
    df = pd.read_parquet(CACHE_PATH)
    logger.info(f"Cache shape: {df.shape}, date range: "
                f"{df.index.get_level_values(0).min()} ~ "
                f"{df.index.get_level_values(0).max()}")

    fwd_returns = df[RETURNS_COL].copy()
    fwd_returns.name = "fwd_ret"

    # Champion proxy: rank-average of core alpha158 features
    exclude_prefixes = ("__", "_", "hsi_", "hstech_", "nasdaq_")
    exclude_names = {
        "holder_num", "bp", "ep", "pb", "pe",
        "pb_mom20", "pe_mom20", "price_pos20",
        "flow_net_mf_latest", "flow_net_mf_5d",
        "flow_net_mf_20d_avg", "amount_anom20",
    }
    alpha_cols = [
        c for c in df.columns
        if not any(c.startswith(p) for p in exclude_prefixes)
        and c not in exclude_names
    ]
    logger.info(f"Champion proxy built from {len(alpha_cols)} features")
    rank_df = df[alpha_cols].rank(pct=True, axis=1)
    champion_pred = rank_df.mean(axis=1)
    champion_pred.name = "champion_pred"

    return fwd_returns, champion_pred


def _to_multiindex(df: pd.DataFrame, date_col: str, code_col: str = "qlib_code"):
    """Convert df with date and qlib_code columns to (datetime, instrument) MultiIndex."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    # Normalize instrument codes to lowercase (feature cache uses lowercase)
    out[code_col] = out[code_col].str.lower()
    out = out.set_index([date_col, code_col])
    out.index.names = ["datetime", "instrument"]
    return out


# ---------------------------------------------------------------------------
# External factor loaders
# ---------------------------------------------------------------------------

def load_moneyflow_factors() -> dict[str, pd.Series]:
    """Derive candidate factors from st_moneyflow.parquet."""
    path = STORAGE / "st_moneyflow.parquet"
    if not path.exists():
        logger.warning("st_moneyflow.parquet not found, skipping")
        return {}

    logger.info("Loading moneyflow data...")
    df = pd.read_parquet(path)
    logger.info(f"  Raw shape: {df.shape}")

    # PIT safety: shift date forward 1 business day
    df["date"] = pd.to_datetime(df["date"]) + pd.tseries.offsets.BDay(1)

    # Numeric conversion
    for c in df.columns:
        if c.startswith("st_"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    factors = {}

    # 1. Net money flow (already in data) -- normalize by total volume
    total_buy = df["st_buy_sm_amount"] + df["st_buy_md_amount"] + df["st_buy_lg_amount"] + df["st_buy_elg_amount"]
    total_sell = df["st_sell_sm_amount"] + df["st_sell_md_amount"] + df["st_sell_lg_amount"] + df["st_sell_elg_amount"]
    total_amount = total_buy + total_sell
    net_flow_norm = (total_buy - total_sell) / total_amount.replace(0, np.nan)
    df["ext_net_flow_norm"] = net_flow_norm

    # 2. Large + extra-large net flow ratio (institutional proxy)
    lg_net = (df["st_buy_lg_amount"] + df["st_buy_elg_amount"]
              - df["st_sell_lg_amount"] - df["st_sell_elg_amount"])
    df["ext_lg_net_ratio"] = lg_net / total_amount.replace(0, np.nan)

    # 3. Small-order net sell ratio (retail sentiment, inverted = retail selling is bullish)
    sm_net = df["st_buy_sm_amount"] - df["st_sell_sm_amount"]
    df["ext_sm_net_ratio"] = -sm_net / total_amount.replace(0, np.nan)

    # Convert to MultiIndex and extract series
    mi = _to_multiindex(df, "date")
    # Deduplicate: keep last entry for each (datetime, instrument)
    mi = mi[~mi.index.duplicated(keep="last")]

    for col in ["ext_net_flow_norm", "ext_lg_net_ratio", "ext_sm_net_ratio"]:
        if col in mi.columns:
            s = mi[col].copy()
            s.name = col
            factors[col] = s

    # 4. Rolling 5d and 20d averages of net flow
    for col_name, base_col in [("ext_net_flow_5d", "ext_net_flow_norm"),
                                ("ext_lg_net_5d", "ext_lg_net_ratio")]:
        unstacked = mi[base_col].unstack("instrument")
        rolled = unstacked.rolling(5, min_periods=3).mean()
        try:
            stacked = rolled.stack(future_stack=True)
        except TypeError:
            stacked = rolled.stack(dropna=False)
        stacked.index.names = ["datetime", "instrument"]
        stacked.name = col_name
        factors[col_name] = stacked

    logger.info(f"  MoneyFlow: {len(factors)} derived factors")
    return factors


def load_holder_number_factors() -> dict[str, pd.Series]:
    """Derive candidate factors from st_holder_number.parquet."""
    path = STORAGE / "st_holder_number.parquet"
    if not path.exists():
        logger.warning("st_holder_number.parquet not found, skipping")
        return {}

    logger.info("Loading holder number data...")
    df = pd.read_parquet(path)
    logger.info(f"  Raw shape: {df.shape}")

    df["holder_num"] = pd.to_numeric(df.get("holder_num"), errors="coerce")
    df["ann_date"] = pd.to_datetime(df["ann_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["ann_date", "holder_num"])
    df["qlib_code"] = df["qlib_code"].str.lower()

    # Sort and deduplicate
    df = df.sort_values(["qlib_code", "ann_date"]).drop_duplicates(
        ["qlib_code", "ann_date"], keep="last"
    )

    # Compute pct change in holder_num (fewer holders = more concentrated = bullish)
    df["holder_pct_chg"] = df.groupby("qlib_code")["holder_num"].pct_change()
    # Invert: decreasing holders is bullish signal
    df["ext_holder_decrease"] = -df["holder_pct_chg"]

    factors = {}

    # Convert to MultiIndex using ann_date as the date
    mi = df.set_index(["ann_date", "qlib_code"])
    mi.index.names = ["datetime", "instrument"]

    for col in ["ext_holder_decrease"]:
        if col in mi.columns:
            s = mi[col].dropna().copy()
            s.name = col
            factors[col] = s

    # Also add raw holder_num (log-transformed) -- might capture size effect
    s = np.log1p(mi["holder_num"]).copy()
    s.name = "ext_log_holder_num"
    factors["ext_log_holder_num"] = s.dropna()

    logger.info(f"  Holder: {len(factors)} derived factors")
    return factors


def load_cyq_factors() -> dict[str, pd.Series]:
    """Derive candidate factors from st_cyq_perf.parquet."""
    path = STORAGE / "st_cyq_perf.parquet"
    if not path.exists():
        logger.warning("st_cyq_perf.parquet not found, skipping")
        return {}

    logger.info("Loading CYQ (chip distribution) data...")
    df = pd.read_parquet(path)
    logger.info(f"  Raw shape: {df.shape}")

    # PIT safety: chip data published after close, use next bday
    df["date"] = pd.to_datetime(df["date"]) + pd.tseries.offsets.BDay(1)

    for c in df.columns:
        if c.startswith("cyq_"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    factors = {}

    # 1. Winner rate (% of holders currently in profit)
    df["ext_winner_rate"] = df["cyq_winner_rate"]

    # 2. Cost concentration: (cost_85 - cost_15) / weight_avg
    #    Tighter concentration = consensus = potentially bullish
    cost_spread = df["cyq_cost_85pct"] - df["cyq_cost_15pct"]
    df["ext_cost_concentration"] = -cost_spread / df["cyq_weight_avg"].replace(0, np.nan)

    # 3. Price vs weighted average cost (are holders in profit or loss?)
    #    Use cost_50pct as proxy for "current price vs median cost"
    #    We can't get actual close price here, so use winner_rate as proxy

    mi = _to_multiindex(df, "date")
    mi = mi[~mi.index.duplicated(keep="last")]

    for col in ["ext_winner_rate", "ext_cost_concentration"]:
        if col in mi.columns:
            s = mi[col].copy()
            s.name = col
            factors[col] = s

    # 4. Winner rate 5d change (momentum of chip sentiment)
    unstacked = mi["ext_winner_rate"].unstack("instrument")
    delta = unstacked.diff(5)
    try:
        stacked = delta.stack(future_stack=True)
    except TypeError:
        stacked = delta.stack(dropna=False)
    stacked.index.names = ["datetime", "instrument"]
    stacked.name = "ext_winner_rate_d5"
    factors["ext_winner_rate_d5"] = stacked

    logger.info(f"  CYQ: {len(factors)} derived factors")
    return factors


# ---------------------------------------------------------------------------
# Factor descriptions for reporting
# ---------------------------------------------------------------------------
FACTOR_DESCRIPTIONS = {
    "ext_net_flow_norm": "Net money flow / total amount (normalized)",
    "ext_lg_net_ratio": "Large+XL order net buy ratio (institutional proxy)",
    "ext_sm_net_ratio": "Inverted small-order net sell (retail sentiment)",
    "ext_net_flow_5d": "5-day avg net flow ratio",
    "ext_lg_net_5d": "5-day avg large-order net ratio",
    "ext_holder_decrease": "Holder number decrease (concentrated ownership)",
    "ext_log_holder_num": "Log holder count (size proxy)",
    "ext_winner_rate": "Chip winner rate (% holders in profit)",
    "ext_cost_concentration": "Chip cost concentration (neg spread/avg_cost)",
    "ext_winner_rate_d5": "Winner rate 5-day change",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    fwd_returns, champion_pred = _load_cache_returns_and_champion()

    # Collect all external factors
    all_factors: dict[str, pd.Series] = {}
    all_factors.update(load_moneyflow_factors())
    all_factors.update(load_holder_number_factors())
    all_factors.update(load_cyq_factors())

    if not all_factors:
        logger.error("No external factors loaded. Nothing to validate.")
        return

    logger.info(f"\n{'='*80}")
    logger.info(f"Total external factors to validate: {len(all_factors)}")
    logger.info(f"{'='*80}\n")

    factory = AlphaFactory()
    results = []

    for factor_name, raw_series in all_factors.items():
        desc = FACTOR_DESCRIPTIONS.get(factor_name, factor_name)
        logger.info(f"--- Processing: {factor_name} ({desc}) ---")

        # Basic stats
        n_raw = raw_series.notna().sum()
        n_total = len(raw_series)
        logger.info(f"  Raw: {n_raw:,} non-null / {n_total:,} total")

        if n_raw < 1000:
            logger.warning(f"  Too few observations ({n_raw}), skipping")
            results.append({
                "factor_name": factor_name,
                "description": desc,
                "rank_ic": np.nan,
                "rank_icir": np.nan,
                "residual_rank_ic": np.nan,
                "marginal_value": False,
                "coverage": 0.0,
                "spread_q1_q5": np.nan,
                "neg_ctrl_ic": np.nan,
                "n_days": 0,
                "gate": "skip",
                "gate_failures": ["too few observations"],
            })
            continue

        # Run through full_pipeline (fillna + winsorize + zscore)
        try:
            processed = full_pipeline(raw_series)
        except Exception as e:
            logger.error(f"  Pipeline failed: {e}")
            results.append({
                "factor_name": factor_name, "description": desc,
                "rank_ic": np.nan, "rank_icir": np.nan,
                "residual_rank_ic": np.nan, "marginal_value": False,
                "coverage": 0.0, "spread_q1_q5": np.nan,
                "neg_ctrl_ic": np.nan, "n_days": 0,
                "gate": "error", "gate_failures": [str(e)],
            })
            continue

        processed_nonnull = processed.notna().sum()
        logger.info(f"  After pipeline: {processed_nonnull:,} non-null")

        # Compute standalone tearsheet
        tearsheet = run_tearsheet_from_series(processed, fwd_returns)
        if "error" in tearsheet:
            logger.warning(f"  Tearsheet error: {tearsheet['error']}")
            results.append({
                "factor_name": factor_name, "description": desc,
                "rank_ic": np.nan, "rank_icir": np.nan,
                "residual_rank_ic": np.nan, "marginal_value": False,
                "coverage": 0.0, "spread_q1_q5": np.nan,
                "neg_ctrl_ic": np.nan, "n_days": 0,
                "gate": "error", "gate_failures": [tearsheet["error"]],
            })
            continue

        # Compute residual IC vs champion
        residual_result = compute_residual_ic(
            new_factor=processed,
            champion_pred=champion_pred,
            returns=fwd_returns,
        )

        # Register with Alpha Factory and gate-check
        def _make_builder(s):
            return lambda: s

        factory.register(
            name=factor_name,
            description=desc,
            build_func=_make_builder(processed),
        )
        factory.run_tearsheet(factor_name, returns=fwd_returns)
        gate = factory.check_gate(factor_name)

        row = {
            "factor_name": factor_name,
            "description": desc,
            "rank_ic": tearsheet.get("rank_ic_mean", 0.0),
            "rank_icir": tearsheet.get("rank_icir", 0.0),
            "residual_rank_ic": residual_result.get("residual_rank_ic", 0.0),
            "marginal_value": residual_result.get("marginal_value", False),
            "coverage": tearsheet.get("coverage", 0.0),
            "spread_q1_q5": tearsheet.get("spread_q1_q5"),
            "neg_ctrl_ic": tearsheet.get("negative_control_ic", 0.0),
            "autocorr_1d": tearsheet.get("autocorr_1d"),
            "n_days": tearsheet.get("n_days", 0),
            "gate": gate.get("verdict", "fail"),
            "gate_failures": gate.get("failures", []),
        }
        results.append(row)

        logger.info(
            f"  RankIC={row['rank_ic']:.4f}  ICIR={row['rank_icir']:.3f}  "
            f"ResidualRankIC={row['residual_rank_ic']:.4f}  "
            f"Marginal={row['marginal_value']}  Gate={row['gate']}"
        )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("External Factor Validation Results (Phase 4L-ext)")
    print("=" * 120)

    summary_df = pd.DataFrame(results)
    display_cols = [
        "factor_name", "rank_ic", "rank_icir", "residual_rank_ic",
        "marginal_value", "coverage", "spread_q1_q5", "neg_ctrl_ic",
        "autocorr_1d", "n_days", "gate",
    ]
    available_cols = [c for c in display_cols if c in summary_df.columns]
    display_df = summary_df[available_cols].copy()
    display_df = display_df.sort_values("rank_ic", ascending=False, key=lambda x: x.abs())

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 150)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(display_df.to_string(index=False))

    # Gate details
    print("\n--- Gate Check Details ---")
    for r in results:
        status = r["gate"].upper()
        failures = "; ".join(r["gate_failures"]) if r["gate_failures"] else "none"
        print(f"  {r['factor_name']:28s} [{status:5s}]  failures: {failures}")

    # Marginal value summary
    print("\n--- Marginal Value vs Champion ---")
    for r in results:
        if r.get("residual_rank_ic") and not np.isnan(r.get("residual_rank_ic", np.nan)):
            marker = "***" if r["marginal_value"] else "   "
            print(
                f"  {marker} {r['factor_name']:28s}  "
                f"ResidualRankIC={r['residual_rank_ic']:+.4f}"
            )

    # Verdict
    print("\n--- Overall Verdict ---")
    passing = [r for r in results if r["gate"] == "pass"]
    marginal = [r for r in results if r.get("marginal_value")]
    print(f"  Gate pass: {len(passing)}/{len(results)}")
    print(f"  Marginal value (residual RankIC > 0.005): {len(marginal)}/{len(results)}")

    if passing:
        print("\n  Factors that PASS gate:")
        for r in passing:
            print(f"    - {r['factor_name']} (RankIC={r['rank_ic']:.4f})")
    if marginal:
        print("\n  Factors with MARGINAL value beyond champion:")
        for r in marginal:
            print(f"    - {r['factor_name']} (ResidualRankIC={r['residual_rank_ic']:+.4f})")

    if not passing and not marginal:
        print("\n  CONCLUSION: No external factors show sufficient standalone or marginal value.")
        print("  These data sources have been mined out by the existing Alpha158 features.")


if __name__ == "__main__":
    main()
