"""Limit-up (涨停板) data collector via AKShare.

Provides daily limit-up pool, consecutive board counts, and board premium
for monster stock (妖股) detection.
"""
import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class LimitUpCollector:
    """Collects limit-up data from AKShare for A-share market."""

    def fetch_today_pool(self) -> pd.DataFrame:
        """Fetch today's limit-up stock pool.

        Returns:
            DataFrame with columns: code, name, limit_up_time, seal_amount,
            open_count, consecutive_boards, change_pct
        """
        try:
            import akshare as ak
            df = ak.stock_zt_pool_em(date=datetime.now().strftime("%Y%m%d"))
            if df is None or df.empty:
                return pd.DataFrame()

            result = df.rename(columns={
                "代码": "code",
                "名称": "name",
                "涨停封单量": "seal_amount",
                "连板数": "consecutive_boards",
                "涨跌幅": "change_pct",
                "首次封板时间": "limit_up_time",
                "最后封板时间": "last_seal_time",
                "炸板次数": "open_count",
            })
            # Normalize code format
            for i, row in result.iterrows():
                code = str(row["code"]).zfill(6)
                prefix = "SH" if code.startswith("6") else "SZ"
                result.at[i, "qlib_code"] = f"{prefix}{code}"

            logger.info(f"Fetched {len(result)} limit-up stocks")
            return result
        except Exception as e:
            logger.warning(f"Failed to fetch limit-up pool: {e}")
            return pd.DataFrame()

    def fetch_previous_pool(self) -> pd.DataFrame:
        """Fetch previous day's limit-up stocks (for next-day premium calc).

        Returns:
            DataFrame with previous day's limit-up stocks.
        """
        try:
            import akshare as ak
            df = ak.stock_zt_pool_previous_em(date=datetime.now().strftime("%Y%m%d"))
            if df is None or df.empty:
                return pd.DataFrame()
            return df.rename(columns={
                "代码": "code",
                "名称": "name",
                "涨跌幅": "change_pct",
            })
        except Exception as e:
            logger.warning(f"Failed to fetch previous limit-up pool: {e}")
            return pd.DataFrame()

    def fetch_strong_stocks(self) -> pd.DataFrame:
        """Fetch strong stock pool (涨幅>5%).

        Returns:
            DataFrame with strong stocks.
        """
        try:
            import akshare as ak
            df = ak.stock_zt_pool_strong_em(date=datetime.now().strftime("%Y%m%d"))
            if df is None or df.empty:
                return pd.DataFrame()
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch strong pool: {e}")
            return pd.DataFrame()

    def get_consecutive_boards(self) -> dict:
        """Get current consecutive board counts.

        Returns:
            Dict mapping qlib_code -> consecutive_boards count.
        """
        pool = self.fetch_today_pool()
        if pool.empty or "qlib_code" not in pool.columns:
            return {}
        return {
            row["qlib_code"]: int(row.get("consecutive_boards", 1))
            for _, row in pool.iterrows()
            if pd.notna(row.get("consecutive_boards"))
        }

    def compute_board_premium(self) -> float:
        """Compute average next-day premium for yesterday's limit-up stocks.

        This is a market speculation temperature gauge.

        Returns:
            Average change_pct of yesterday's limit-up stocks today.
            Positive = speculative appetite, negative = risk aversion.
        """
        prev = self.fetch_previous_pool()
        if prev.empty or "change_pct" not in prev.columns:
            return 0.0
        try:
            return float(prev["change_pct"].mean())
        except Exception:
            return 0.0
