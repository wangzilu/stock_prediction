"""4G.2 Global Sector Spillover: map overseas proxies to A-share industries.

Instead of broadcasting broad HSI/NASDAQ regime signals to ALL stocks identically,
this script assigns each A-share stock its sector-specific overseas proxy:

  NASDAQ  -> 电子 / 半导体 / 软件服务 / 元器件 / IT设备 / 互联网
  HSTECH  -> 计算机 / 传媒 / 通信 / 电气设备
  HSI     -> everything else (market-level fallback)

For each stock-date, the output features are the overseas proxy's lagged returns,
volatility, and momentum -- but the proxy is SPECIFIC to that stock's industry.

PIT safety:
  - NASDAQ dates already shifted +1 bday in cross_market_indices.parquet
  - HK indices close before A-share opens, so same-date is safe for HSI/HSTECH
  - All features are strictly backward-looking (lag1/lag2/lag5, rolling windows)

Output: data/storage/sector_spillover_features.parquet
  Index: (datetime, instrument) MultiIndex matching Qlib convention
  Columns: spill_ret1d, spill_ret2d, spill_ret5d, spill_vol5d, spill_vol20d,
           spill_mom5d, spill_mom20d, spill_dd20d, spill_proxy (categorical)

Usage:
    python scripts/build_sector_spillover.py
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"

# ---------------------------------------------------------------------------
# Sector mapping: overseas proxy -> list of A-share industries
# The keys must match the column prefixes in cross_market_indices.parquet
# (hsi_ret1d, hstech_ret1d, nasdaq_ret1d, etc.)
# ---------------------------------------------------------------------------
PROXY_INDUSTRY_MAP = {
    "nasdaq": [
        # Semiconductor / electronics / tech hardware
        "半导体", "元器件", "IT设备", "互联网", "通信设备",
        "电器仪表",
    ],
    "hstech": [
        # Software / media / telco / electrical equipment
        "软件服务", "电气设备", "影视音像", "出版业", "广告包装",
        "电信运营", "文教休闲", "家用电器", "家居用品",
    ],
    # HSI is the fallback -- assigned to everything not matched above
}

# Feature suffixes we extract from cross_market_indices.parquet per proxy
# These are the raw regime features computed by fetch_cross_market_indices.py
REGIME_SUFFIXES = [
    "ret1d", "ret5d", "ret20d",
    "vol5d", "vol20d",
    "mom5d", "mom20d",
    "up_ratio_10d", "dd20d",
]


def load_industry_mapping() -> pd.Series:
    """Load stock -> industry mapping.  Returns Series: qlib_code -> industry."""
    path = DATA_DIR / "industry_mapping.parquet"
    if not path.exists():
        raise FileNotFoundError(f"industry_mapping.parquet not found at {path}")
    df = pd.read_parquet(path)
    if "qlib_code" not in df.columns or "industry" not in df.columns:
        raise ValueError("industry_mapping.parquet must have 'qlib_code' and 'industry' columns")
    ind_map = df.drop_duplicates("qlib_code").set_index("qlib_code")["industry"]
    logger.info(f"Industry mapping: {len(ind_map)} stocks, "
                f"{ind_map.nunique()} unique industries")
    return ind_map


def load_cross_market_regime() -> dict[str, pd.DataFrame]:
    """Load cross_market_indices.parquet and split into per-proxy DataFrames.

    Returns dict: proxy_name -> DataFrame with columns [date, ret1d, ret5d, ...].
    """
    path = DATA_DIR / "cross_market_indices.parquet"
    if not path.exists():
        raise FileNotFoundError(f"cross_market_indices.parquet not found at {path}")
    raw = pd.read_parquet(path)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date")
    raw["date"] = raw["date"].dt.as_unit("ns")

    logger.info(f"Cross-market indices: {raw.shape}, "
                f"date range {raw['date'].min()} ~ {raw['date'].max()}")
    logger.info(f"  Columns: {list(raw.columns)}")

    proxies = {}
    for proxy in ["hsi", "hstech", "nasdaq"]:
        cols_found = [s for s in REGIME_SUFFIXES if f"{proxy}_{s}" in raw.columns]
        if not cols_found:
            logger.warning(f"  No columns for proxy '{proxy}', skipping")
            continue
        sub = raw[["date"] + [f"{proxy}_{s}" for s in cols_found]].copy()
        # Rename to generic names (strip proxy prefix)
        sub.columns = ["date"] + cols_found
        # Ensure numeric
        for c in cols_found:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        proxies[proxy] = sub.dropna(subset=["date"]).sort_values("date")
        logger.info(f"  Proxy '{proxy}': {len(sub)} rows, features={cols_found}")

    return proxies


def build_stock_proxy_assignment(ind_map: pd.Series) -> pd.Series:
    """Assign each stock to its best overseas proxy based on industry.

    Returns Series: qlib_code -> proxy_name (one of 'nasdaq', 'hstech', 'hsi').
    """
    # Build reverse map: industry -> proxy
    industry_to_proxy = {}
    for proxy, industries in PROXY_INDUSTRY_MAP.items():
        for ind in industries:
            industry_to_proxy[ind] = proxy

    # For each stock, look up its industry -> proxy; default to 'hsi'
    proxy_assignment = ind_map.map(
        lambda ind: industry_to_proxy.get(ind, "hsi")
    )
    proxy_assignment.name = "proxy"

    # Log distribution
    counts = proxy_assignment.value_counts()
    for proxy, cnt in counts.items():
        logger.info(f"  Proxy '{proxy}': {cnt} stocks")

    # Show which industries got matched vs fallback
    matched_industries = set(industry_to_proxy.keys()) & set(ind_map.values)
    all_industries = set(ind_map.values)
    fallback_industries = all_industries - matched_industries
    logger.info(f"  Matched industries: {len(matched_industries)}/{len(all_industries)}")
    if fallback_industries:
        logger.info(f"  Fallback (HSI) industries: {sorted(fallback_industries)[:20]}...")

    return proxy_assignment


def build_spillover_features(
    proxy_assignment: pd.Series,
    proxy_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build per-stock spillover features.

    For each stock, look up its assigned proxy and copy the proxy's regime
    features for each date. The result is a long DataFrame indexed by
    (date, qlib_code) with spillover features.

    This is more informative than broadcasting the same HSI/NASDAQ values to
    every stock, because now tech stocks see NASDAQ signals while commodity
    stocks see HSI signals.
    """
    # Collect all unique dates across all proxies
    all_dates = pd.Series(dtype="datetime64[ns]")
    for proxy, df in proxy_data.items():
        all_dates = pd.concat([all_dates, df["date"]])
    all_dates = all_dates.drop_duplicates().sort_values().reset_index(drop=True)

    logger.info(f"Building spillover: {len(proxy_assignment)} stocks x "
                f"{len(all_dates)} dates")

    # Determine common feature columns (intersection across all proxies)
    common_features = None
    for proxy, df in proxy_data.items():
        feat_cols = [c for c in df.columns if c != "date"]
        if common_features is None:
            common_features = set(feat_cols)
        else:
            common_features &= set(feat_cols)
    common_features = sorted(common_features)
    logger.info(f"  Common features across proxies: {common_features}")

    # Rename output columns with 'spill_' prefix
    out_col_map = {feat: f"spill_{feat}" for feat in common_features}
    out_cols = list(out_col_map.values())

    # Build date-indexed lookup per proxy (for fast vectorized merge)
    proxy_lookup = {}
    for proxy, df in proxy_data.items():
        lookup = df[["date"] + common_features].drop_duplicates("date").sort_values("date")
        lookup = lookup.set_index("date")
        proxy_lookup[proxy] = lookup

    # Group stocks by proxy for vectorized processing
    stocks_by_proxy = proxy_assignment.groupby(proxy_assignment).groups

    pieces = []
    for proxy, stock_codes in stocks_by_proxy.items():
        if proxy not in proxy_lookup:
            logger.warning(f"  Proxy '{proxy}' has no data, {len(stock_codes)} stocks get NaN")
            continue

        lookup = proxy_lookup[proxy]
        n_stocks = len(stock_codes)

        # For this proxy's dates, create (date, stock) cross product
        proxy_dates = lookup.index.values
        n_dates = len(proxy_dates)

        # Repeat dates for each stock, repeat stock codes for each date
        dates_repeated = np.repeat(proxy_dates, n_stocks)
        stocks_repeated = np.tile(stock_codes.values, n_dates)

        # Repeat feature values for each stock
        feat_values = lookup[common_features].values  # (n_dates, n_features)
        feat_repeated = np.repeat(feat_values, n_stocks, axis=0)  # (n_dates*n_stocks, n_features)

        chunk = pd.DataFrame(
            feat_repeated,
            columns=out_cols,
        )
        chunk["date"] = dates_repeated
        chunk["instrument"] = stocks_repeated
        chunk["spill_proxy"] = proxy

        pieces.append(chunk)
        logger.info(f"  Proxy '{proxy}': {n_stocks} stocks x {n_dates} dates "
                    f"= {len(chunk)} rows")

    if not pieces:
        raise RuntimeError("No spillover data built")

    result = pd.concat(pieces, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.as_unit("ns")
    result = result.sort_values(["date", "instrument"]).reset_index(drop=True)

    # Convert to (datetime, instrument) MultiIndex for Qlib compatibility
    result = result.set_index(["date", "instrument"])

    logger.info(f"Spillover features: {result.shape}")
    logger.info(f"  Proxy distribution:\n{result['spill_proxy'].value_counts()}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Build sector spillover features")
    parser.add_argument("--output", default=str(DATA_DIR / "sector_spillover_features.parquet"),
                        help="Output parquet path")
    args = parser.parse_args()

    # 1. Load industry mapping
    ind_map = load_industry_mapping()

    # 2. Load cross-market regime data
    proxy_data = load_cross_market_regime()
    if not proxy_data:
        logger.error("No cross-market proxy data available")
        sys.exit(1)

    # 3. Assign each stock to its overseas proxy
    proxy_assignment = build_stock_proxy_assignment(ind_map)

    # 4. Build per-stock spillover features
    spillover = build_spillover_features(proxy_assignment, proxy_data)

    # 5. Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spillover.to_parquet(str(out_path))

    logger.info(f"Saved: {out_path}")
    logger.info(f"  Shape: {spillover.shape}")
    logger.info(f"  Date range: {spillover.index.get_level_values(0).min()} ~ "
                f"{spillover.index.get_level_values(0).max()}")
    logger.info(f"  Unique stocks: {spillover.index.get_level_values(1).nunique()}")
    logger.info(f"  Columns: {list(spillover.columns)}")

    # Summary stats
    numeric_cols = [c for c in spillover.columns if c != "spill_proxy"]
    for col in numeric_cols:
        s = spillover[col]
        logger.info(f"  {col}: mean={s.mean():.6f}, std={s.std():.6f}, "
                    f"non-null={s.notna().sum()}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
