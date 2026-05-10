"""Capital flow factor collector: main force flow, northbound, margin.

Daily frequency factors from AKShare.
Stores to data/storage/capital_flow_features.parquet.
"""
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "storage"
FLOW_PATH = DATA_DIR / "capital_flow_features.parquet"


class CapitalFlowCollector:

    def fetch_individual_flow(self, code: str, market: str = "sz") -> pd.DataFrame:
        """Fetch main force / retail fund flow for one stock."""
        try:
            import akshare as ak
            df = ak.stock_individual_fund_flow(stock=code, market=market)
            if df is None or df.empty:
                return pd.DataFrame()
            return df
        except Exception:
            return pd.DataFrame()

    def fetch_individual_flow_latest(self, code: str, market: str = "sz") -> dict:
        """Get latest day's fund flow metrics for one stock."""
        df = self.fetch_individual_flow(code, market)
        if df.empty:
            return {}
        latest = df.iloc[-1]
        try:
            return {
                "main_net_inflow": _safe_float(latest.get("主力净流入-净额")),
                "main_net_pct": _safe_float(latest.get("主力净流入-净占比")),
                "super_large_net": _safe_float(latest.get("超大单净流入-净额")),
                "large_net": _safe_float(latest.get("大单净流入-净额")),
                "medium_net": _safe_float(latest.get("中单净流入-净额")),
                "small_net": _safe_float(latest.get("小单净流入-净额")),
            }
        except Exception:
            return {}

    def fetch_northbound_holdings(self) -> pd.DataFrame:
        """Fetch northbound (陆股通) individual stock holdings."""
        try:
            import akshare as ak
            df = ak.stock_hsgt_hold_stock_em(market="北向", indicator="今日排行")
            if df is None or df.empty:
                return pd.DataFrame()
            logger.info(f"Fetched northbound holdings: {len(df)} stocks")
            return df
        except Exception as e:
            logger.warning(f"Northbound holdings failed: {e}")
            return pd.DataFrame()

    def fetch_sector_flow(self) -> pd.DataFrame:
        """Fetch sector-level fund flow ranking."""
        try:
            import akshare as ak
            df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
            if df is None or df.empty:
                return pd.DataFrame()
            return df
        except Exception as e:
            logger.warning(f"Sector flow failed: {e}")
            return pd.DataFrame()

    def fetch_dragon_tiger(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch dragon-tiger list (龙虎榜) data."""
        try:
            import akshare as ak
            if not start_date:
                start_date = datetime.now().strftime("%Y%m%d")
            if not end_date:
                end_date = start_date
            df = ak.stock_lhb_detail_em(start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                return pd.DataFrame()
            logger.info(f"Fetched dragon-tiger: {len(df)} entries")
            return df
        except Exception as e:
            logger.warning(f"Dragon-tiger failed: {e}")
            return pd.DataFrame()

    def fetch_batch_flow(self, codes: list, market_map: dict = None) -> pd.DataFrame:
        """Fetch fund flow for multiple stocks.

        Args:
            codes: list of stock codes (e.g. ["000001", "600519"])
            market_map: dict mapping code -> market ("sh" or "sz")
        """
        results = []
        for i, code in enumerate(codes):
            market = "sh" if str(code).startswith("6") else "sz"
            if market_map:
                market = market_map.get(code, market)

            flow = self.fetch_individual_flow_latest(code, market)
            if flow:
                flow["code"] = code
                flow["qlib_code"] = f"SH{code}" if market == "sh" else f"SZ{code}"
                results.append(flow)

            if (i + 1) % 20 == 0:
                logger.info(f"Fund flow: {i+1}/{len(codes)}")
            time.sleep(0.5)  # Rate limit

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df["date"] = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Fetched fund flow for {len(df)} stocks")
        return df

    def save(self, df: pd.DataFrame, path: Path = FLOW_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def load(self, path: Path = FLOW_PATH) -> pd.DataFrame:
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()


def _safe_float(val) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan
