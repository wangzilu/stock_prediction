#!/usr/bin/env python3
"""Phase 4L -- Individual factor validation through processors + Alpha Factory.

Picks 8 individual factors from the 174-dim feature cache, runs each through
the full_pipeline (winsorize + zscore), computes residual IC vs champion
(the full 174-dim XGB predictions), registers with Alpha Factory, and runs
tearsheet.  Prints a summary comparison table.

Usage:
    python scripts/phase4l_factor_validation.py
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
# Config
# ---------------------------------------------------------------------------

CACHE_PATH = PROJECT_ROOT / "data" / "storage" / "feature_cache_174_holder_regime_ma.parquet"
RETURNS_COL = "__pnl_return_1d"

# Factors to validate individually -- chosen to span different signal types
FACTORS_TO_TEST = {
    # Momentum
    "ROC5":        "5-day price momentum",
    "ROC20":       "20-day price momentum",
    "ROC60":       "60-day price momentum",
    # Volatility
    "STD20":       "20-day return volatility",
    # Liquidity / turnover
    "turn_raw":    "Raw turnover ratio",
    "amount_raw":  "Raw trading amount",
    "turn_anom20": "Turnover anomaly vs 20d avg",
    # Market microstructure
    "VWAP0":       "VWAP-to-close ratio",
}


def main():
    # ------------------------------------------------------------------
    # 1. Load cache
    # ------------------------------------------------------------------
    logger.info(f"Loading feature cache from {CACHE_PATH}")
    df = pd.read_parquet(CACHE_PATH)
    logger.info(f"Cache shape: {df.shape}, date range: "
                f"{df.index.get_level_values(0).min()} ~ "
                f"{df.index.get_level_values(0).max()}")

    fwd_returns = df[RETURNS_COL].copy()
    fwd_returns.name = "fwd_ret"

    # ------------------------------------------------------------------
    # 2. Build a champion prediction proxy (rank of mean of all 158 Alpha
    #    features -- this approximates what the full XGB sees).
    #    We use rank to simulate a model that uses all features jointly.
    # ------------------------------------------------------------------
    alpha158_cols = [c for c in df.columns
                     if not c.startswith("__") and not c.startswith("_")
                     and c not in FACTORS_TO_TEST
                     and c not in (
                         "holder_num", "bp", "ep", "pb", "pe",
                         "pb_mom20", "pe_mom20", "price_pos20",
                         "flow_net_mf_latest", "flow_net_mf_5d",
                         "flow_net_mf_20d_avg", "amount_anom20",
                     )
                     and not c.startswith("hsi_") and not c.startswith("hstech_")
                     and not c.startswith("nasdaq_")]
    logger.info(f"Champion proxy built from {len(alpha158_cols)} features "
                f"(rank-average ensemble)")

    # Simple rank-average ensemble as champion proxy
    rank_df = df[alpha158_cols].rank(pct=True, axis=1)
    champion_pred = rank_df.mean(axis=1)
    champion_pred.name = "champion_pred"

    # ------------------------------------------------------------------
    # 3. Validate each factor
    # ------------------------------------------------------------------
    factory = AlphaFactory()
    results = []

    for factor_name, description in FACTORS_TO_TEST.items():
        if factor_name not in df.columns:
            logger.warning(f"Factor {factor_name} not in cache, skipping")
            continue

        logger.info(f"--- Processing: {factor_name} ({description}) ---")

        raw_factor = df[factor_name].copy()
        raw_factor.name = factor_name

        # 3a. Run through full_pipeline (fillna + winsorize + zscore)
        processed = full_pipeline(raw_factor)

        # 3b. Compute residual IC vs champion
        residual_result = compute_residual_ic(
            new_factor=processed,
            champion_pred=champion_pred,
            returns=fwd_returns,
        )

        # 3c. Register with Alpha Factory and run tearsheet
        #     We use a closure to capture the processed factor
        def _make_builder(s):
            return lambda: s

        factory.register(
            name=factor_name,
            description=description,
            build_func=_make_builder(processed),
        )
        tearsheet = factory.run_tearsheet(factor_name, returns=fwd_returns)
        gate = factory.check_gate(factor_name)

        results.append({
            "factor_name": factor_name,
            "description": description,
            "rank_ic": tearsheet.get("rank_ic_mean", 0.0),
            "rank_icir": tearsheet.get("rank_icir", 0.0),
            "raw_ic": residual_result.get("raw_ic", 0.0),
            "residual_ic": residual_result.get("residual_ic", 0.0),
            "residual_rank_ic": residual_result.get("residual_rank_ic", 0.0),
            "marginal_value": residual_result.get("marginal_value", False),
            "coverage": tearsheet.get("coverage", 0.0),
            "spread_q1_q5": tearsheet.get("spread_q1_q5"),
            "neg_ctrl_ic": tearsheet.get("negative_control_ic", 0.0),
            "autocorr_1d": tearsheet.get("autocorr_1d"),
            "n_days": tearsheet.get("n_days", 0),
            "gate": gate.get("verdict", "fail"),
            "gate_failures": gate.get("failures", []),
        })

        logger.info(
            f"  RankIC={tearsheet.get('rank_ic_mean', 0):.4f}  "
            f"ResidualIC={residual_result.get('residual_rank_ic', 0):.4f}  "
            f"Gate={gate.get('verdict')}"
        )

    # ------------------------------------------------------------------
    # 4. Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("Factor Validation Results (Phase 4L)")
    print("=" * 100)

    summary_df = pd.DataFrame(results)
    # Format for display
    display_cols = [
        "factor_name", "rank_ic", "rank_icir", "residual_rank_ic",
        "marginal_value", "coverage", "spread_q1_q5", "neg_ctrl_ic",
        "autocorr_1d", "n_days", "gate",
    ]
    display_df = summary_df[display_cols].copy()
    display_df = display_df.sort_values("rank_ic", ascending=False, key=abs)

    # Pretty print
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(display_df.to_string(index=False))

    # Also print the Alpha Factory summary
    print("\n--- Alpha Factory Summary ---")
    af_summary = factory.summary_table()
    if not af_summary.empty:
        print(af_summary[[
            "name", "verdict", "rank_ic_mean", "rank_icir",
            "coverage", "spread_q1_q5", "negative_control_ic",
        ]].to_string(index=False))

    # Gate pass/fail details
    print("\n--- Gate Check Details ---")
    for r in results:
        status = "PASS" if r["gate"] == "pass" else "FAIL"
        failures = "; ".join(r["gate_failures"]) if r["gate_failures"] else "none"
        print(f"  {r['factor_name']:15s} [{status}]  failures: {failures}")

    # Marginal value summary
    print("\n--- Residual IC (marginal value beyond champion) ---")
    for r in sorted(results, key=lambda x: abs(x["residual_rank_ic"]), reverse=True):
        mv = "YES" if r["marginal_value"] else "no"
        print(f"  {r['factor_name']:15s}  residual_rank_ic={r['residual_rank_ic']:.4f}  "
              f"marginal_value={mv}")

    # Return non-zero if no factors pass
    n_pass = sum(1 for r in results if r["gate"] == "pass")
    print(f"\nTotal: {n_pass}/{len(results)} factors passed gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
