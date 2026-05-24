"""Simple Barra-style risk model — compute style factor exposures.

Not a full CNE5 replication. Computes 5 core style factors from available data:
  1. size: log(market_cap) cross-sectional z-score
  2. beta: regression beta vs market over 60 days
  3. momentum: past 20-day return
  4. volatility: 20-day return standard deviation
  5. liquidity: log(ADV20) cross-sectional z-score

Plus industry factors from JQData 申万行业分类.

Used by optimizer_v2 for style/industry exposure constraints.

Usage:
    from backtest.barra_simple import compute_style_exposures
    exposures = compute_style_exposures(date="2026-05-22")
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "storage"


def compute_style_exposures(date: str = None) -> pd.DataFrame:
    """Compute cross-sectional style factor exposures for all stocks.

    Args:
        date: compute exposures as of this date (default: latest in cache)

    Returns:
        DataFrame indexed by instrument with columns: size, beta, momentum,
        volatility, liquidity, industry (申万一级)
    """
    cache = pd.read_parquet(
        DATA_DIR / "feature_cache_174_holder_regime_ma.parquet",
        columns=["amount_raw", "turn_raw", "ROC20", "STD20", "BETA60",
                 "__label_5d"],
    )

    dates = sorted(cache.index.get_level_values(0).unique())
    if date:
        target = pd.Timestamp(date)
        avail = [d for d in dates if d <= target]
        use_date = avail[-1] if avail else dates[-1]
    else:
        use_date = dates[-1]

    day = cache.loc[use_date]

    exposures = pd.DataFrame(index=day.index)

    # 1. Size: use amount_raw as market cap proxy (highly correlated)
    amount = day["amount_raw"].replace(0, np.nan)
    if amount.notna().sum() > 100:
        log_amount = np.log1p(amount.clip(lower=1))
        exposures["size"] = _zscore(log_amount)

    # 2. Beta: from Alpha158 BETA60
    if "BETA60" in day.columns:
        exposures["beta"] = _zscore(day["BETA60"])

    # 3. Momentum: ROC20 (20-day return)
    if "ROC20" in day.columns:
        exposures["momentum"] = _zscore(day["ROC20"])

    # 4. Volatility: STD20
    if "STD20" in day.columns:
        exposures["volatility"] = _zscore(day["STD20"])

    # 5. Liquidity: turnover
    if "turn_raw" in day.columns:
        turn = day["turn_raw"].replace(0, np.nan)
        if turn.notna().sum() > 100:
            exposures["liquidity"] = _zscore(np.log1p(turn.clip(lower=0.001)))

    # 6. Industry: from JQData 申万分类
    industry_path = DATA_DIR / "jqdata" / "industry_sw.parquet"
    if industry_path.exists():
        try:
            ind = pd.read_parquet(industry_path)
            # Map JQ code to qlib code
            code_map = {}
            for _, row in ind.iterrows():
                jq_code = str(row.get("code", ""))
                sw_l1 = str(row.get("sw_l1_name", ""))
                if ".XSHE" in jq_code:
                    qlib = f"sz{jq_code[:6]}"
                elif ".XSHG" in jq_code:
                    qlib = f"sh{jq_code[:6]}"
                else:
                    continue
                code_map[qlib] = sw_l1

            exposures["industry"] = pd.Series(
                {inst: code_map.get(str(inst), "未知") for inst in exposures.index}
            )
        except Exception:
            pass

    exposures = exposures.dropna(how="all")
    logger.info(f"Style exposures: {exposures.shape} for {use_date}")
    return exposures


def compute_portfolio_exposure(weights: dict, exposures: pd.DataFrame) -> dict:
    """Compute portfolio-level style exposure from stock weights.

    Args:
        weights: {instrument: weight}
        exposures: from compute_style_exposures()

    Returns:
        dict of {factor: weighted_exposure}
    """
    style_cols = [c for c in exposures.columns if c != "industry"]

    result = {}
    for col in style_cols:
        total = 0.0
        total_w = 0.0
        for inst, w in weights.items():
            if inst in exposures.index and col in exposures.columns:
                val = exposures.loc[inst, col]
                if np.isfinite(val):
                    total += w * val
                    total_w += w
        result[col] = round(total / max(total_w, 1e-8), 4)

    # Industry concentration
    if "industry" in exposures.columns:
        industry_weights = {}
        for inst, w in weights.items():
            if inst in exposures.index:
                ind = exposures.loc[inst, "industry"]
                industry_weights[ind] = industry_weights.get(ind, 0) + w
        result["top_industry"] = max(industry_weights.items(), key=lambda x: x[1]) if industry_weights else ("", 0)
        result["industry_hhi"] = round(sum(v**2 for v in industry_weights.values()), 4)
        result["n_industries"] = len(industry_weights)

    return result


def _zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score with winsorization at ±3."""
    s = series.copy()
    mu = s.mean()
    sigma = s.std()
    if sigma < 1e-10:
        return pd.Series(0, index=s.index)
    z = (s - mu) / sigma
    return z.clip(-3, 3)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    logging.basicConfig(level=logging.INFO)

    exp = compute_style_exposures()
    print(f"\nExposures shape: {exp.shape}")
    print(f"\nStyle stats:")
    for col in exp.columns:
        if col == "industry":
            print(f"  {col}: {exp[col].nunique()} unique industries")
        else:
            print(f"  {col}: mean={exp[col].mean():.3f} std={exp[col].std():.3f}")

    # Test portfolio exposure
    # Simulate top 20 equal weight
    top20 = list(exp.index[:20])
    weights = {inst: 0.05 for inst in top20}
    port_exp = compute_portfolio_exposure(weights, exp)
    print(f"\nPortfolio exposure (top 20 equal weight):")
    for k, v in port_exp.items():
        print(f"  {k}: {v}")
