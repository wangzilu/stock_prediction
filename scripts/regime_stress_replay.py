"""Replay regime controller on historical stress periods.

Tests whether regime scores would have correctly identified:
- 2022.03-04: Shanghai lockdown bear
- 2024.01-02: Quant crash / microcap crisis
- 2024.09-10: Policy bull

Usage:
    python scripts/regime_stress_replay.py
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from signals.regime_controller import RegimeController


def main():
    rc = RegimeController()

    # Current regime
    logger.info("=== Current Regime ===")
    current = rc.compute()
    for k, v in current.items():
        if k != "suggested_adjustments":
            logger.info(f"  {k}: {v}")
    adj = current.get("suggested_adjustments", {})
    logger.info(f"  Adjustments: {adj.get('reason', '')}")
    logger.info(f"    max_position={adj.get('max_position')}, max_turnover={adj.get('max_turnover')}, "
                f"smallcap={adj.get('smallcap_exposure')}")

    # Historical replay with available data
    # Note: most regime inputs only have recent data, so we check what's available
    logger.info("\n=== Data Availability for Historical Replay ===")

    import pandas as pd

    data_ranges = {}
    for name, path in [
        ("Shibor", "st_shibor.parquet"),
        ("M2", "st_cn_m.parquet"),
        ("PMI", "st_cn_pmi.parquet"),
        ("CPI", "st_cn_cpi.parquet"),
        ("Margin", "st_margin_detail.parquet"),
        ("Limit-down", "st_limit_list_d.parquet"),
        ("US Treasury", "st_us_tycr.parquet"),
    ]:
        full_path = PROJECT_ROOT / "data" / "storage" / path
        if full_path.exists():
            try:
                df = pd.read_parquet(full_path)
                # Find date column
                date_col = None
                for c in ["date", "trade_date", "month", "MONTH"]:
                    if c in df.columns:
                        date_col = c
                        break
                if date_col:
                    dates = pd.to_datetime(df[date_col], format="%Y%m%d", errors="coerce")
                    dates = dates.dropna()
                    if not dates.empty:
                        data_ranges[name] = (str(dates.min())[:10], str(dates.max())[:10])
                        logger.info(f"  {name}: {data_ranges[name][0]} ~ {data_ranges[name][1]}")
                    else:
                        logger.info(f"  {name}: no valid dates")
                else:
                    logger.info(f"  {name}: no date column found")
            except Exception as e:
                logger.info(f"  {name}: error {e}")
        else:
            logger.info(f"  {name}: file not found")

    # Compute regime for each available historical month
    logger.info("\n=== Monthly Regime History ===")
    logger.info(f"{'Month':<10} {'liquidity':>10} {'credit':>8} {'leverage':>10} {'microcap':>9} {'external':>9} {'risk_on':>8} {'alert':>10}")
    logger.info("-" * 85)

    # Use shibor dates as timeline (most granular daily data)
    try:
        shibor = pd.read_parquet(PROJECT_ROOT / "data" / "storage" / "st_shibor.parquet")
        shibor["date"] = pd.to_datetime(shibor["date"], format="%Y%m%d", errors="coerce")
        months = sorted(shibor["date"].dt.to_period("M").unique())

        for month in months[-24:]:  # last 24 months
            date_str = f"{month.year}-{month.month:02d}-15"
            regime = rc.compute(date_str)

            logger.info(
                f"{str(month):<10} "
                f"{regime['liquidity_score']:>+10.3f} "
                f"{regime['credit_stress_score']:>+8.3f} "
                f"{regime['leverage_unwind_score']:>+10.3f} "
                f"{regime['microcap_crash_risk']:>+9.3f} "
                f"{regime['external_shock_score']:>+9.3f} "
                f"{regime['risk_on_score']:>+8.3f} "
                f"{regime['alert_level']:>10}"
            )
    except Exception as e:
        logger.error(f"Replay failed: {e}")

    logger.info("\n=== Interpretation ===")
    logger.info("Look for:")
    logger.info("  - 2022.03-04: should show negative scores (Shanghai lockdown)")
    logger.info("  - 2024.01-02: should show microcap_crash_risk negative (quant crash)")
    logger.info("  - 2024.09-10: should show policy_support positive (policy bull)")
    logger.info("  - Shibor spikes should show in credit_stress")
    logger.info("  - Note: many scores use latest data snapshot, not historical values")
    logger.info("         A proper replay would need per-date data loading")


if __name__ == "__main__":
    main()
