#!/usr/bin/env python3
"""Phase 4N — Sector Heat / Spillover Candidate Factors.

Builds sector-level momentum and breadth signals, then assigns them back
to individual stocks as candidate alpha factors.

Factors:
    sector_heat           — composite: 0.4*ret_5d + 0.3*vol_zscore + 0.3*up_ratio
    stock_relative_strength — stock return minus its sector's mean return

Industry classification: JQData 申万一级 where available, else stock-code-prefix
board classification (上海主板 / 深圳主板 / 创业板 / 科创板 / 北交所).

Uses last 200 trading days for tractable compute.

Usage:
    python scripts/phase4n_sector_heat.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from factors.processors import full_pipeline
from tracker.alpha_factory import AlphaFactory

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "storage"
CACHE_PATH = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
INDUSTRY_PATH = DATA_DIR / "jqdata" / "industry_sw.parquet"
RETURNS_COL = "__pnl_return_1d"
N_DAYS = 200  # last N trading days


# ── Helpers ─────────────────────────────────────────────────────────────────

def _board_from_code(instrument: str) -> str:
    """Map qlib-style instrument code to board name."""
    code = instrument.replace("sh", "").replace("sz", "").replace("bj", "")
    if code.startswith("688"):
        return "科创板"
    elif code.startswith("60"):
        return "上海主板"
    elif code.startswith("00"):
        return "深圳主板"
    elif code.startswith("30"):
        return "创业板"
    elif code.startswith("8") or code.startswith("4"):
        return "北交所"
    else:
        return "其他"


def load_industry_map() -> dict:
    """Load JQData 申万一级 industry mapping, keyed by qlib instrument code."""
    code_map = {}
    if INDUSTRY_PATH.exists():
        try:
            ind = pd.read_parquet(INDUSTRY_PATH)
            for _, row in ind.iterrows():
                jq_code = str(row.get("code", ""))
                sw_l1 = str(row.get("sw_l1_name", "")).strip()
                if not sw_l1:
                    continue
                if ".XSHE" in jq_code:
                    qlib = f"sz{jq_code[:6]}"
                elif ".XSHG" in jq_code:
                    qlib = f"sh{jq_code[:6]}"
                else:
                    continue
                code_map[qlib] = sw_l1
            logger.info(f"JQData SW L1 mapping: {len(code_map)} stocks")
        except Exception as e:
            logger.warning(f"Failed to load JQData industry: {e}")
    return code_map


def assign_industry(instruments: pd.Index, jq_map: dict) -> pd.Series:
    """Assign industry label to each instrument. JQData first, else board."""
    labels = {}
    jq_hit = 0
    for inst in instruments:
        inst_str = str(inst)
        if inst_str in jq_map:
            labels[inst] = jq_map[inst_str]
            jq_hit += 1
        else:
            labels[inst] = _board_from_code(inst_str)
    logger.info(f"Industry assignment: {jq_hit}/{len(instruments)} from JQData, "
                f"rest from board prefix")
    return pd.Series(labels)


def cross_sectional_zscore(s: pd.Series) -> pd.Series:
    """Z-score within each date (level 0 of MultiIndex)."""
    def _z(g):
        mu, sigma = g.mean(), g.std()
        if sigma is None or sigma < 1e-12:
            return pd.Series(0.0, index=g.index)
        return (g - mu) / sigma
    return s.groupby(level=0, group_keys=False).apply(_z)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Phase 4N — Sector Heat / Spillover Factors")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1] Loading feature cache...")
    cols = [RETURNS_COL, "amount_raw", "turn_raw"]
    cache = pd.read_parquet(CACHE_PATH, columns=cols)
    cache.index.names = ["datetime", "instrument"]

    # Keep last N_DAYS trading days
    all_dates = sorted(cache.index.get_level_values("datetime").unique())
    keep_dates = all_dates[-N_DAYS:]
    cache = cache.loc[cache.index.get_level_values("datetime").isin(keep_dates)]
    print(f"  Cache shape (last {N_DAYS}d): {cache.shape}")
    print(f"  Date range: {keep_dates[0].date()} ~ {keep_dates[-1].date()}")

    fwd_returns = cache[RETURNS_COL].copy()
    fwd_returns.name = "fwd_ret"

    # ------------------------------------------------------------------
    # 2. Industry classification
    # ------------------------------------------------------------------
    print("\n[2] Loading industry classification...")
    jq_map = load_industry_map()
    all_instruments = cache.index.get_level_values("instrument").unique()
    industry_labels = assign_industry(all_instruments, jq_map)
    n_sectors = industry_labels.nunique()
    print(f"  Total sectors: {n_sectors}")
    print(f"  Sector distribution (top 10):")
    for sec, cnt in industry_labels.value_counts().head(10).items():
        print(f"    {sec}: {cnt} stocks")

    # Build (datetime, instrument) -> industry Series aligned to cache index
    industry_mi = cache.index.get_level_values("instrument").map(
        lambda x: industry_labels.get(x, "其他")
    )
    industry_series = pd.Series(
        industry_mi.values,
        index=cache.index,
        name="industry",
    )

    # ------------------------------------------------------------------
    # 3. Compute sector-level metrics per date
    # ------------------------------------------------------------------
    print("\n[3] Computing sector-level metrics...")

    # Create a working DataFrame
    df = cache.copy()
    df["industry"] = industry_series.values

    # IMPORTANT: __pnl_return_1d at date T = close[T+1]/close[T]-1 (forward return).
    # To avoid lookahead, sector signals on date T must use returns from T-1
    # (which is the realized return close[T]/close[T-1], stored at date T-1).
    # We shift sector metrics by 1 day: compute on raw data, then .shift(1) before
    # mapping back to stocks.

    # 3a. sector_ret_1d: mean of __pnl_return_1d per sector per date (raw, not yet shifted)
    sector_ret_1d_raw = df.groupby(["datetime", "industry"])[RETURNS_COL].mean()
    sector_ret_1d_raw.name = "sector_ret_1d"
    print(f"  sector_ret_1d_raw: {sector_ret_1d_raw.shape[0]} (date, industry) pairs")

    # Unstack to (date x industry), shift by 1 to avoid lookahead, then compute rolling
    sector_ret_unstacked = sector_ret_1d_raw.unstack("industry")
    # Shift: on date T we see returns realized up to T-1
    sector_ret_lagged = sector_ret_unstacked.shift(1)

    # 3b. sector_ret_5d: rolling 5-day sum of LAGGED sector returns
    sector_ret_5d_unstacked = sector_ret_lagged.rolling(5, min_periods=1).sum()
    sector_ret_5d = sector_ret_5d_unstacked.stack()
    sector_ret_5d.index.names = ["datetime", "industry"]
    sector_ret_5d.name = "sector_ret_5d"

    # sector_ret_1d (lagged) for stock_relative_strength
    sector_ret_1d = sector_ret_lagged.stack()
    sector_ret_1d.index.names = ["datetime", "industry"]
    sector_ret_1d.name = "sector_ret_1d"

    # 3c. sector_volume_zscore: sector mean amount vs its own 60d rolling mean/std
    sector_amount = df.groupby(["datetime", "industry"])["amount_raw"].mean()
    sector_amount_unstacked = sector_amount.unstack("industry")
    # Shift amount too: on date T we see volume up to T-1
    sector_amount_lagged = sector_amount_unstacked.shift(1)
    roll_mean = sector_amount_lagged.rolling(60, min_periods=10).mean()
    roll_std = sector_amount_lagged.rolling(60, min_periods=10).std()
    sector_vol_z = ((sector_amount_lagged - roll_mean) / roll_std.replace(0, np.nan))
    sector_vol_z_stacked = sector_vol_z.stack()
    sector_vol_z_stacked.index.names = ["datetime", "industry"]
    sector_vol_z_stacked.name = "sector_volume_zscore"

    # 3d. sector_up_ratio: fraction of positive-return stocks per sector (lagged)
    df["is_up"] = (df[RETURNS_COL] > 0).astype(float)
    sector_up_ratio_raw = df.groupby(["datetime", "industry"])["is_up"].mean()
    sector_up_ratio = sector_up_ratio_raw.unstack("industry").shift(1).stack()
    sector_up_ratio.index.names = ["datetime", "industry"]
    sector_up_ratio.name = "sector_up_ratio"

    # Combine sector metrics into a single DataFrame
    sector_metrics = pd.DataFrame({
        "sector_ret_1d": sector_ret_1d,
        "sector_ret_5d": sector_ret_5d,
        "sector_volume_zscore": sector_vol_z_stacked,
        "sector_up_ratio": sector_up_ratio,
    })
    print(f"  Sector metrics shape: {sector_metrics.shape}")
    print(f"  NaN rates:")
    for col in sector_metrics.columns:
        nan_pct = sector_metrics[col].isna().mean()
        print(f"    {col}: {nan_pct:.1%}")

    # ------------------------------------------------------------------
    # 4. Map sector metrics back to individual stocks
    # ------------------------------------------------------------------
    print("\n[4] Mapping sector metrics to stocks...")

    # For each (datetime, instrument), look up (datetime, industry)
    dt_idx = cache.index.get_level_values("datetime")
    ind_idx = industry_series.values

    # Build a lookup key
    lookup_idx = pd.MultiIndex.from_arrays([dt_idx, ind_idx], names=["datetime", "industry"])

    # Align sector metrics to stock-level
    stock_sector_ret_5d = sector_metrics["sector_ret_5d"].reindex(lookup_idx).values
    stock_sector_vol_z = sector_metrics["sector_volume_zscore"].reindex(lookup_idx).values
    stock_sector_up_ratio = sector_metrics["sector_up_ratio"].reindex(lookup_idx).values
    stock_sector_ret_1d = sector_metrics["sector_ret_1d"].reindex(lookup_idx).values

    # 4a. sector_heat composite
    # Z-score each component cross-sectionally first
    s_ret5d = pd.Series(stock_sector_ret_5d, index=cache.index, name="s_ret5d")
    s_vol_z = pd.Series(stock_sector_vol_z, index=cache.index, name="s_vol_z")
    s_up_ratio = pd.Series(stock_sector_up_ratio, index=cache.index, name="s_up_ratio")

    z_ret5d = cross_sectional_zscore(s_ret5d)
    z_vol_z = cross_sectional_zscore(s_vol_z)
    z_up_ratio = cross_sectional_zscore(s_up_ratio)

    sector_heat = 0.4 * z_ret5d + 0.3 * z_vol_z + 0.3 * z_up_ratio
    sector_heat.name = "sector_heat"

    # 4b. stock_relative_strength = lagged(stock_return) - lagged(sector_return)
    #     stock's own T-1 return minus sector mean T-1 return (idiosyncratic momentum)
    #     We shift the stock return by 1 to get realized return visible at date T
    stock_return_lagged = (
        cache[RETURNS_COL]
        .unstack("instrument")
        .shift(1)
        .stack()
    )
    stock_return_lagged.index.names = ["datetime", "instrument"]
    sector_return_for_stock = pd.Series(stock_sector_ret_1d, index=cache.index)
    stock_relative_strength = stock_return_lagged - sector_return_for_stock
    stock_relative_strength.name = "stock_relative_strength"

    print(f"  sector_heat: non-NaN = {sector_heat.notna().sum()}")
    print(f"  stock_relative_strength: non-NaN = {stock_relative_strength.notna().sum()}")

    # ------------------------------------------------------------------
    # 5. Process through full_pipeline
    # ------------------------------------------------------------------
    print("\n[5] Running through full_pipeline...")
    factors = {}

    for name, raw in [("sector_heat", sector_heat),
                       ("stock_relative_strength", stock_relative_strength)]:
        processed = full_pipeline(raw)
        factors[name] = processed
        non_nan = processed.notna().sum()
        print(f"  {name}: {non_nan} non-NaN after pipeline")

    # Also register the raw components for diagnostics
    for name, raw in [("sector_ret_5d", s_ret5d),
                       ("sector_volume_zscore", s_vol_z),
                       ("sector_up_ratio", s_up_ratio)]:
        processed = full_pipeline(raw)
        factors[name] = processed
        non_nan = processed.notna().sum()
        print(f"  {name}: {non_nan} non-NaN after pipeline")

    # ------------------------------------------------------------------
    # 6. Register with Alpha Factory and run tearsheet
    # ------------------------------------------------------------------
    print("\n[6] Running Alpha Factory tearsheets...")
    factory = AlphaFactory()
    results = {}

    descriptions = {
        "sector_heat": "Composite sector momentum/breadth signal (0.4*ret5d + 0.3*vol_z + 0.3*up_ratio)",
        "stock_relative_strength": "Stock return minus sector mean return (idiosyncratic alpha)",
        "sector_ret_5d": "5-day cumulative sector return",
        "sector_volume_zscore": "Sector volume z-score vs 60d rolling stats",
        "sector_up_ratio": "Fraction of stocks with positive return in sector",
    }

    for factor_name, factor_series in factors.items():
        print(f"\n  --- {factor_name} ---")

        def make_build_func(s):
            return lambda: s
        build_func = make_build_func(factor_series)

        factory.register(
            name=factor_name,
            description=descriptions.get(factor_name, factor_name),
            build_func=build_func,
        )

        metrics = factory.run_tearsheet(factor_name, returns=fwd_returns)
        gate = factory.check_gate(factor_name)
        results[factor_name] = {
            **metrics,
            "gate_pass": gate["pass"],
            "gate_failures": gate.get("failures", []),
        }

        if "error" not in metrics:
            print(f"  RankIC:    {metrics['rank_ic_mean']:+.4f} (std={metrics['rank_ic_std']:.4f})")
            print(f"  RankICIR:  {metrics['rank_icir']:+.3f}")
            print(f"  IC pos%:   {metrics['rank_ic_pos_ratio']:.1%}")
            print(f"  Spread:    {metrics.get('spread_q1_q5', 'N/A')}")
            print(f"  Coverage:  {metrics['coverage']:.1%}")
            print(f"  Neg ctrl:  {metrics['negative_control_ic']:.4f}")
            print(f"  N days:    {metrics['n_days']}")
            print(f"  Gate:      {'PASS' if gate['pass'] else 'FAIL'}")
            if gate.get("failures"):
                for f in gate["failures"]:
                    print(f"    - {f}")
        else:
            print(f"  ERROR: {metrics['error']}")

    # ------------------------------------------------------------------
    # 7. Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTOR HEAT FACTOR COMPARISON TABLE")
    print("=" * 70)

    summary = factory.summary_table()
    factor_names = list(factors.keys())
    if not summary.empty:
        mask = summary["name"].isin(factor_names)
        if mask.any():
            tbl = summary[mask].copy()
            display_cols = [
                "name", "rank_ic_mean", "rank_icir", "rank_ic_pos_ratio",
                "spread_q1_q5", "coverage", "negative_control_ic",
                "autocorr_1d", "n_days", "verdict",
            ]
            display_cols = [c for c in display_cols if c in tbl.columns]
            pd.set_option("display.width", 200)
            pd.set_option("display.max_columns", 20)
            pd.set_option("display.float_format",
                          lambda x: f"{x:.4f}" if abs(x) < 10 else f"{x:.1f}")
            print(tbl[display_cols].to_string(index=False))

    # ------------------------------------------------------------------
    # 8. Interpretation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    valid = {k: v for k, v in results.items() if "error" not in v}
    if valid:
        best = max(valid, key=lambda k: abs(valid[k].get("rank_ic_mean", 0)))
        best_m = valid[best]
        print(f"\nBest factor by |RankIC|: {best}")
        print(f"  RankIC = {best_m['rank_ic_mean']:+.4f}, ICIR = {best_m['rank_icir']:+.3f}")

        passing = [k for k, v in valid.items() if v.get("gate_pass", False)]
        if passing:
            print(f"\nFactors passing gate: {', '.join(passing)}")
        else:
            print("\nNo factors passed the gate.")
            # Identify closest to passing
            closest = min(valid, key=lambda k: len(valid[k].get("gate_failures", [])))
            print(f"Closest to passing: {closest} "
                  f"(failures: {valid[closest].get('gate_failures', [])})")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
