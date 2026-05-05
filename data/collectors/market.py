import pandas as pd
import akshare as ak
from datetime import datetime, timedelta


class MarketCollector:
    """Collects A-share market data via AKShare."""

    def fetch_daily(self, code: str, days: int = 60) -> pd.DataFrame:
        """Fetch daily OHLCV data for a stock.

        Args:
            code: AKShare format code, e.g. "sh600519"
            days: Number of trading days to fetch

        Returns:
            DataFrame with columns [open, high, low, close, volume], indexed by date.
            Empty DataFrame if fetch fails.
        """
        try:
            symbol = code[2:]  # "sh600519" -> "600519"
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )

            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df[["open", "high", "low", "close", "volume"]]
            return df.tail(days)

        except Exception:
            return pd.DataFrame()

    def fetch_realtime(self, code: str) -> dict:
        """Fetch realtime quote for a stock.

        Args:
            code: AKShare format code, e.g. "sh600519"

        Returns:
            Dict with keys: price, change_pct, volume, high, low.
            Empty dict if fetch fails.
        """
        try:
            symbol = code[2:]
            df = ak.stock_zh_a_spot_em()
            row = df[df["代码"] == symbol]

            if row.empty:
                return {}

            row = row.iloc[0]
            return {
                "price": float(row["最新价"]),
                "change_pct": float(row["涨跌幅"]),
                "volume": float(row["成交量"]),
                "high": float(row["最高"]),
                "low": float(row["最低"]),
            }

        except Exception:
            return {}

    def fetch_batch_daily(self, codes: list[str], days: int = 60) -> dict[str, pd.DataFrame]:
        """Fetch daily data for multiple stocks.

        Returns:
            Dict mapping code -> DataFrame
        """
        result = {}
        for code in codes:
            df = self.fetch_daily(code, days)
            if not df.empty:
                result[code] = df
        return result
