"""Fundamental factor collector: valuation, quality, growth.

Fetches PE/PB/PS/ROE/margins/growth via AKShare.
Updates weekly (fundamentals change slowly).
Stores to data/storage/fundamental_features.parquet.
"""
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "storage"
FUNDAMENTAL_PATH = DATA_DIR / "fundamental_features.parquet"


class FundamentalCollector:

    def fetch_valuation_batch(self, codes: list = None) -> pd.DataFrame:
        """Fetch PE/PB/PS/MV for all A-shares from spot data.

        Tries AKShare first, falls back to Tencent/cached spot data.
        """
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                raise RuntimeError("AKShare spot empty")

            result = df[["代码", "名称", "最新价", "市盈率-动态", "市净率", "总市值", "流通市值"]].copy()
            result.columns = ["code", "name", "price", "pe_ttm", "pb", "total_mv", "circ_mv"]
        except Exception as e:
            logger.warning(f"AKShare valuation failed: {e}, trying cached spot")
            # Fallback: use cached spot data from market collector
            try:
                from data.collectors.market import MarketCollector
                mc = MarketCollector()
                mc._load_spot_cache()
                df = mc._spot_cache
                if df is None or df.empty:
                    return pd.DataFrame()

                cols_map = {}
                for need, candidates in [
                    ("code", ["代码"]),
                    ("name", ["名称"]),
                    ("price", ["最新价"]),
                    ("pe_ttm", ["市盈率-动态", "市盈率"]),
                    ("pb", ["市净率"]),
                    ("total_mv", ["总市值"]),
                    ("circ_mv", ["流通市值"]),
                ]:
                    for c in candidates:
                        if c in df.columns:
                            cols_map[c] = need
                            break

                available_cols = [v for v in cols_map.values() if v in df.rename(columns=cols_map).columns]
                result = df.rename(columns=cols_map)[available_cols].copy()
                # Fill missing valuation columns with NaN
                for col in ["code", "name", "price", "pe_ttm", "pb", "total_mv", "circ_mv"]:
                    if col not in result.columns:
                        result[col] = np.nan
            except Exception as e2:
                logger.error(f"Cached spot also failed: {e2}")
                return pd.DataFrame()

        # Derived factors (runs on BOTH happy path and fallback path)
        result["qlib_code"] = result["code"].apply(
            lambda c: f"SH{c}" if str(c).startswith("6") else f"SZ{c}"
        )

        result["ep"] = 1.0 / result["pe_ttm"].replace(0, np.nan)  # Earnings yield
        result["bp"] = 1.0 / result["pb"].replace(0, np.nan)  # Book-to-price
        result["log_mv"] = np.log(result["total_mv"].replace(0, np.nan))
        result["log_circ_mv"] = np.log(result["circ_mv"].replace(0, np.nan))

        # Clean
        for col in ["pe_ttm", "pb", "ep", "bp", "log_mv", "log_circ_mv"]:
            result[col] = pd.to_numeric(result[col], errors="coerce")

        result["date"] = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Fetched valuation for {len(result)} stocks")
        return result

    def fetch_financial_quality(self, code: str) -> dict:
        """Fetch ROE/ROA/margins for one stock."""
        try:
            import akshare as ak
            df = ak.stock_financial_abstract_em(symbol=code)
            if df is None or df.empty:
                return {}

            # Take latest row
            latest = df.iloc[0] if len(df) > 0 else {}
            return {
                "roe": _safe_float(latest.get("净资产收益率")),
                "roa": _safe_float(latest.get("总资产报酬率")),
                "gross_margin": _safe_float(latest.get("销售毛利率")),
                "net_margin": _safe_float(latest.get("销售净利率")),
                "debt_ratio": _safe_float(latest.get("资产负债率")),
                "revenue_growth": _safe_float(latest.get("营业收入同比增长率")),
                "profit_growth": _safe_float(latest.get("净利润同比增长率")),
            }
        except Exception:
            return {}

    def fetch_all(self, top_n: int = 500) -> pd.DataFrame:
        """Fetch valuation for all stocks + quality for top N.

        Strategy: valuation is batch (fast), quality is per-stock (slow).
        Only fetch quality for most liquid stocks.
        """
        # Step 1: Batch valuation (< 5 seconds)
        val = self.fetch_valuation_batch()
        if val.empty:
            return val

        # Step 2: Quality for top N by market cap (rate limited)
        top_codes = val.nlargest(top_n, "total_mv")["code"].tolist()
        quality_data = []

        logger.info(f"Fetching quality factors for top {len(top_codes)} stocks...")
        for i, code in enumerate(top_codes):
            q = self.fetch_financial_quality(code)
            if q:
                q["code"] = code
                quality_data.append(q)
            if (i + 1) % 50 == 0:
                logger.info(f"  Quality: {i+1}/{len(top_codes)}")
            time.sleep(0.3)  # Rate limit

        if quality_data:
            quality_df = pd.DataFrame(quality_data)
            val = val.merge(quality_df, on="code", how="left")

        return val

    def save(self, df: pd.DataFrame, path: Path = FUNDAMENTAL_PATH):
        """Save fundamental features to parquet."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} fundamental records to {path}")

    def load(self, path: Path = FUNDAMENTAL_PATH) -> pd.DataFrame:
        """Load cached fundamental features."""
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()


def _safe_float(val) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan
