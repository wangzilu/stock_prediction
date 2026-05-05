import pandas as pd
import akshare as ak
from datetime import datetime, timedelta


class GoldCollector:
    """Collects gold price data via AKShare."""

    def fetch_daily(self, days: int = 60) -> pd.DataFrame:
        """Fetch daily international gold price (XAU/USD).

        Args:
            days: Number of trading days to fetch

        Returns:
            DataFrame with columns [open, high, low, close, volume], indexed by date.
            Empty DataFrame if fetch fails.
        """
        try:
            # Primary source: Shanghai Gold Exchange benchmark price
            # Columns: 交易时间, 晚盘价, 早盘价
            df = ak.spot_golden_benchmark_sge()

            if df is None or df.empty:
                return self._fetch_fallback(days)

            df.columns = [c.strip() for c in df.columns]

            # Map SGE columns: 交易时间=date, 晚盘价=close (evening fix), 早盘价=open (morning fix)
            rename_map = {}
            for col in df.columns:
                if "交易时间" in col or "date" in col.lower():
                    rename_map[col] = "date"
                elif "晚盘" in col:
                    rename_map[col] = "close"
                elif "早盘" in col:
                    rename_map[col] = "open"

            df = df.rename(columns=rename_map)

            if "date" not in df.columns:
                return self._fetch_fallback(days)

            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df.index.name = "date"

            # Derive OHLCV from available price columns
            if "close" not in df.columns and "open" in df.columns:
                df["close"] = df["open"]
            if "open" not in df.columns and "close" in df.columns:
                df["open"] = df["close"]

            # Fill high/low from open and close
            price_cols = [c for c in ["open", "close"] if c in df.columns]
            if price_cols:
                df["high"] = df[price_cols].max(axis=1)
                df["low"] = df[price_cols].min(axis=1)
            else:
                return self._fetch_fallback(days)

            df["volume"] = 0

            df = df[["open", "high", "low", "close", "volume"]]
            # Drop rows where all price columns are NaN
            df = df.dropna(subset=["open", "high", "low", "close"])
            return df.tail(days)

        except Exception:
            return self._fetch_fallback(days)

    def _fetch_fallback(self, days: int) -> pd.DataFrame:
        """Fallback: use gold futures (AU0) daily data from Sina."""
        try:
            df = ak.futures_zh_daily_sina(symbol="AU0")

            if df is None or df.empty:
                return pd.DataFrame()

            df.columns = [c.strip() for c in df.columns]

            # futures_zh_daily_sina returns standard English columns:
            # date, open, high, low, close, volume, hold, settle
            rename_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if "日期" in col or col_lower == "date":
                    rename_map[col] = "date"
                elif "开盘" in col or col_lower == "open":
                    rename_map[col] = "open"
                elif "最高" in col or col_lower == "high":
                    rename_map[col] = "high"
                elif "最低" in col or col_lower == "low":
                    rename_map[col] = "low"
                elif "收盘" in col or col_lower == "close":
                    rename_map[col] = "close"
                elif "成交量" in col or col_lower in ("volume", "vol"):
                    rename_map[col] = "volume"

            df = df.rename(columns=rename_map)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                df.index.name = "date"

            for col in ["open", "high", "low", "close"]:
                if col not in df.columns:
                    if "close" in df.columns:
                        df[col] = df["close"]
                    else:
                        return pd.DataFrame()

            if "volume" not in df.columns:
                df["volume"] = 0

            df = df[["open", "high", "low", "close", "volume"]]
            df = df.dropna(subset=["open", "high", "low", "close"])
            return df.tail(days)

        except Exception:
            return pd.DataFrame()

    def fetch_realtime(self) -> dict:
        """Fetch realtime gold price.

        Returns:
            Dict with keys: price, change_pct, high, low, volume.
            Empty dict if fetch fails.
        """
        try:
            df = self.fetch_daily(days=2)
            if df.empty or len(df) < 2:
                return {}

            latest = df.iloc[-1]
            prev = df.iloc[-2]
            change_pct = (
                (float(latest["close"]) - float(prev["close"]))
                / float(prev["close"])
                * 100
            )

            return {
                "price": float(latest["close"]),
                "change_pct": round(change_pct, 2),
                "volume": float(latest["volume"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
            }

        except Exception:
            return {}
