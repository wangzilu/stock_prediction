"""LEGACY market-context crypto collector.

This module is retained only for the LEGACY_MARKET_CONTEXT_ENABLED
evening-report BTC/ETH background context. It is NOT used by the new
crypto quant pipeline.

Per `plans/cc-crypto-implementation-spec-2026-05-30.md` §6.5 + §−0.5
Layer 1, new crypto code must use `data/collectors/crypto_market.py`
and `data/collectors/crypto_derivatives.py` (planned in Phase A) and
must NOT import this module.

`scripts/check_namespace_isolation.py` enforces that no file under
`crypto/`, `data/collectors/crypto_market.py`, or
`data/collectors/crypto_derivatives.py` imports from here.
"""

import pandas as pd
from datetime import datetime, timedelta


class CryptoCollector:
    """LEGACY collector for BTC/ETH evening-report context.

    Not used by the new crypto quant pipeline. See module docstring.
    """

    def __init__(self):
        try:
            import ccxt
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
            })
        except ImportError:
            # Fallback: use AKShare for crypto data
            self.exchange = None

    def fetch_daily(self, symbol: str = "BTC/USDT", days: int = 60) -> pd.DataFrame:
        """Fetch daily OHLCV data for a crypto pair.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT", "ETH/USDT"
            days: Number of days to fetch

        Returns:
            DataFrame with columns [open, high, low, close, volume], indexed by date.
            Empty DataFrame if fetch fails.
        """
        if self.exchange is None:
            return self._fetch_via_akshare(symbol, days)

        try:
            since = self.exchange.parse8601(
                (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
            )
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1d', since=since, limit=days)

            if not ohlcv:
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("date")
            df = df[["open", "high", "low", "close", "volume"]]
            return df

        except Exception:
            return pd.DataFrame()

    def fetch_realtime(self, symbol: str = "BTC/USDT") -> dict:
        """Fetch realtime ticker for a crypto pair.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT"

        Returns:
            Dict with keys: price, change_pct, volume, high, low.
            Empty dict if fetch fails.
        """
        if self.exchange is None:
            return self._fetch_realtime_via_akshare(symbol)

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                "price": float(ticker["last"]),
                "change_pct": float(ticker.get("percentage", 0) or 0),
                "volume": float(ticker.get("baseVolume", 0) or 0),
                "high": float(ticker.get("high", 0) or 0),
                "low": float(ticker.get("low", 0) or 0),
            }
        except Exception:
            return {}

    def fetch_hourly(self, symbol: str = "BTC/USDT", hours: int = 24) -> pd.DataFrame:
        """Fetch hourly OHLCV data for a crypto pair.

        Args:
            symbol: Trading pair
            hours: Number of hours to fetch

        Returns:
            DataFrame with OHLCV columns indexed by datetime.
        """
        if self.exchange is None:
            return pd.DataFrame()

        try:
            since = self.exchange.parse8601(
                (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT00:00:00Z")
            )
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1h', since=since, limit=hours)

            if not ohlcv:
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("date")
            df = df[["open", "high", "low", "close", "volume"]]
            return df

        except Exception:
            return pd.DataFrame()

    def _fetch_via_akshare(self, symbol: str, days: int) -> pd.DataFrame:
        """Fallback: fetch crypto data via AKShare."""
        try:
            import akshare as ak
            # AKShare provides BTC price history
            coin = symbol.split("/")[0].lower()  # "BTC/USDT" -> "btc"
            df = ak.crypto_hist_daily(symbol=coin)

            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            })
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            df = df[["open", "high", "low", "close", "volume"]]
            return df.tail(days)

        except Exception:
            return pd.DataFrame()

    def _fetch_realtime_via_akshare(self, symbol: str) -> dict:
        """Fallback: fetch realtime crypto data via AKShare."""
        try:
            df = self.fetch_daily(symbol, days=2)
            if df.empty:
                return {}
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            change_pct = (latest["close"] - prev["close"]) / prev["close"] * 100
            return {
                "price": float(latest["close"]),
                "change_pct": round(float(change_pct), 2),
                "volume": float(latest["volume"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
            }
        except Exception:
            return {}
