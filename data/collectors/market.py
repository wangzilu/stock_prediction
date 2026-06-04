import pandas as pd
import akshare as ak
import requests
import logging
import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

from config.settings import SPOT_CACHE_META_PATH, SPOT_CACHE_PATH, SPOT_CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)

MAX_RETRIES = 1  # Only try AKShare once, then fallback immediately
RETRY_DELAY = 3  # seconds
MARKET_OPEN_MINUTE = 9 * 60 + 25
MARKET_CLOSE_MINUTE = 15 * 60
AFTER_CLOSE_CACHE_MINUTE = 15 * 60 + 10


class MarketCollector:
    """Collects A-share market data via AKShare with Tencent fallback.

    Data source priority:
    1. AKShare (eastmoney) - primary, most complete
    2. Tencent Finance API - fallback, fast and stable from overseas
    """

    def __init__(
        self,
        spot_cache_path: Union[Path, str] = SPOT_CACHE_PATH,
        spot_cache_meta_path: Union[Path, str] = SPOT_CACHE_META_PATH,
        spot_cache_ttl_seconds: int = SPOT_CACHE_TTL_SECONDS,
    ):
        self._spot_cache = None
        self._spot_loaded = False  # Prevent retry loop
        self._akshare_down = False  # Skip AKShare entirely if it's down
        self._bs_logged_in = False  # baostock session state
        self._spot_cache_path = Path(spot_cache_path)
        self._spot_cache_meta_path = Path(spot_cache_meta_path)
        self._spot_cache_ttl_seconds = spot_cache_ttl_seconds

    # ========== Daily OHLCV ==========

    def fetch_daily(self, code: str, days: int = 60) -> pd.DataFrame:
        """Fetch daily OHLCV with smart fallback.

        Priority: AKShare → baostock → Tencent (realtime only)
        If AKShare is marked down, skip directly to baostock.
        """
        if not self._akshare_down:
            df = self._fetch_daily_akshare(code, days)
            if not df.empty:
                return df

        df = self._fetch_daily_baostock(code, days)
        if not df.empty:
            return df

        return self._fetch_daily_tencent(code, days)

    def _fetch_daily_akshare(self, code: str, days: int) -> pd.DataFrame:
        """Fetch daily via AKShare with retries. Skips if AKShare is known to be down."""
        if self._akshare_down:
            return pd.DataFrame()

        symbol = code[2:]
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

        for attempt in range(MAX_RETRIES):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol, period="daily",
                    start_date=start_date, end_date=end_date, adjust="qfq",
                )
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        "日期": "date", "开盘": "open", "最高": "high",
                        "最低": "low", "收盘": "close", "成交量": "volume",
                    })
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
                    return df.tail(days)
            except Exception as e:
                logger.warning(f"AKShare daily attempt {attempt+1}/{MAX_RETRIES} for {code}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

        # All retries failed — mark AKShare as down for this session
        self._akshare_down = True
        logger.info("AKShare marked as down — subsequent calls will skip to baostock")
        return pd.DataFrame()

    def _ensure_baostock_login(self):
        """Login to baostock once per session."""
        if not self._bs_logged_in:
            import baostock as bs
            bs.login()
            self._bs_logged_in = True

    def _fetch_daily_baostock(self, code: str, days: int) -> pd.DataFrame:
        """Fallback: fetch daily K-line from baostock (free, no registration)."""
        try:
            import baostock as bs
            self._ensure_baostock_login()

            symbol = code[2:]
            prefix = code[:2].lower()
            bs_code = f"{prefix}.{symbol}"

            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")

            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )

            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
            df = df.dropna()
            return df.tail(days)

        except Exception as e:
            logger.warning(f"baostock daily failed for {code}: {e}")
            return pd.DataFrame()

    def _fetch_daily_tencent(self, code: str, days: int) -> pd.DataFrame:
        """Fallback: construct daily data from Tencent realtime (last day only).

        Tencent daily K-line API is unreliable, so this is a minimal fallback
        that at least provides today's data point.
        """
        try:
            quote = self._fetch_realtime_tencent_single(code)
            if not quote or not quote.get("price"):
                return pd.DataFrame()

            today = datetime.now().strftime("%Y-%m-%d")
            df = pd.DataFrame([{
                "date": today,
                "open": quote["price"],  # approximate
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["price"],
                "volume": quote["volume"],
            }])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            return df

        except Exception as e:
            logger.warning(f"Tencent daily fallback failed for {code}: {e}")
            return pd.DataFrame()

    # ========== Realtime ==========

    @staticmethod
    def _minute_of_day(value: datetime) -> int:
        return value.hour * 60 + value.minute

    @classmethod
    def _is_trading_session(cls, value: datetime) -> bool:
        minute = cls._minute_of_day(value)
        return (
            value.weekday() < 5
            and MARKET_OPEN_MINUTE <= minute <= MARKET_CLOSE_MINUTE
        )

    @classmethod
    def _is_after_close(cls, value: datetime) -> bool:
        return value.weekday() < 5 and cls._minute_of_day(value) >= AFTER_CLOSE_CACHE_MINUTE

    @classmethod
    def _is_after_close_snapshot(cls, value: datetime) -> bool:
        return cls._minute_of_day(value) >= AFTER_CLOSE_CACHE_MINUTE

    def _read_spot_disk_cache(self) -> Optional[pd.DataFrame]:
        """Read a fresh-enough A-share spot snapshot from disk."""
        if not self._spot_cache_path.exists() or not self._spot_cache_meta_path.exists():
            return None

        try:
            meta = json.loads(self._spot_cache_meta_path.read_text(encoding="utf-8"))
            created_at = datetime.fromisoformat(meta.get("created_at", ""))
            now = datetime.now()
            age_seconds = (now - created_at).total_seconds()

            if self._is_trading_session(now):
                if created_at.date() != now.date() or age_seconds > self._spot_cache_ttl_seconds:
                    return None
            elif self._is_after_close(now):
                if created_at.date() != now.date() or not self._is_after_close_snapshot(created_at):
                    return None
            else:
                # Pre-open/off-hours: previous after-close cache is the latest confirmed snapshot.
                if age_seconds > 18 * 3600 or not self._is_after_close_snapshot(created_at):
                    return None

            df = pd.read_csv(self._spot_cache_path, dtype={"代码": str})
            if df.empty:
                return None
            logger.info(
                "Loaded A-share spot data from disk cache: %s stocks, source=%s, created_at=%s",
                len(df),
                meta.get("source", "unknown"),
                meta.get("created_at", ""),
            )
            return df
        except Exception as e:
            logger.warning(f"Failed to read A-share spot disk cache: {e}")
            return None

    def _write_spot_disk_cache(self, df: pd.DataFrame, source: str) -> None:
        """Persist a full A-share spot snapshot for later one-shot runs.

        cx code review 2026-06-04 P0: a partial source MUST NOT
        overwrite a healthy full cache. The 2026-06-03 22:00 incident
        was triggered by Tencent's 300-stock partial cache silently
        replacing AKShare's 5000-stock full cache. Block that here.

        Coverage contract:
          - Sources ending in '_partial' are considered partial.
          - A partial write refuses to replace an existing cache with
            row_count >= 4500 written within the same trading day.
          - Same-source overwrites are always allowed (e.g. AKShare
            refreshes itself).
        """
        if df is None or df.empty:
            return

        is_partial = source.endswith("_partial")
        if is_partial and self._spot_cache_path.exists() and \
                self._spot_cache_meta_path.exists():
            try:
                existing_meta = json.loads(
                    self._spot_cache_meta_path.read_text(encoding="utf-8")
                )
                existing_row_count = int(existing_meta.get("row_count", 0))
                existing_source = existing_meta.get("source", "")
                existing_created_at = datetime.fromisoformat(
                    existing_meta.get("created_at", "1970-01-01T00:00:00")
                )
                same_day = existing_created_at.date() == datetime.now().date()
                if existing_row_count >= 4500 and same_day and not existing_source.endswith("_partial"):
                    logger.error(
                        "REFUSING to overwrite healthy full-market spot cache "
                        "(source=%s, %d rows, %s) with PARTIAL source=%s "
                        "(%d rows). Keeping existing cache.",
                        existing_source, existing_row_count,
                        existing_created_at.isoformat(timespec="seconds"),
                        source, len(df),
                    )
                    return
            except Exception as e:  # noqa: BLE001
                logger.warning("Cache overwrite-protection check failed: %s", e)

        try:
            self._spot_cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_csv = self._spot_cache_path.with_name(f"{self._spot_cache_path.name}.tmp")
            tmp_meta = self._spot_cache_meta_path.with_name(f"{self._spot_cache_meta_path.name}.tmp")
            df.to_csv(tmp_csv, index=False, encoding="utf-8")
            meta = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source": source,
                "row_count": int(len(df)),
                "is_partial": is_partial,
            }
            tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_csv, self._spot_cache_path)
            os.replace(tmp_meta, self._spot_cache_meta_path)
        except Exception as e:
            logger.warning(f"Failed to write A-share spot disk cache: {e}")

    def _load_spot_cache(self):
        """Load full A-share spot data with retry + fallback."""
        if self._spot_loaded:
            return

        cached = self._read_spot_disk_cache()
        if cached is not None and not cached.empty:
            self._spot_cache = cached
            self._spot_loaded = True
            return

        # Try AKShare (eastmoney) with retries
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Loading A-share spot data via AKShare (attempt {attempt+1})...")
                self._spot_cache = ak.stock_zh_a_spot_em()
                if self._spot_cache is not None and not self._spot_cache.empty:
                    logger.info(f"Loaded {len(self._spot_cache)} stocks via AKShare")
                    self._write_spot_disk_cache(self._spot_cache, "akshare")
                    self._spot_loaded = True
                    return
            except Exception as e:
                logger.warning(f"AKShare spot attempt {attempt+1}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

        # Fallback Layer 2: ST_CLIENT / TuShare realtime_list (FULL MARKET).
        # cx code review 2026-06-04 P0: AKShare→Tencent skipped the
        # available ST_CLIENT realtime path, causing the 2026-06-03 22:00
        # incident where universe silently shrank to 300 stocks (Tencent
        # watchlist). ST_CLIENT returns the full 5000+ universe.
        logger.warning(
            "AKShare spot failed after %d retries — trying ST_CLIENT realtime_list fallback...",
            MAX_RETRIES,
        )
        st_df = self._load_spot_stclient()
        if st_df is not None and len(st_df) >= 4500:
            logger.info("ST_CLIENT spot fallback succeeded: %d stocks (full)", len(st_df))
            self._spot_cache = st_df
            self._write_spot_disk_cache(st_df, "stclient_realtime")
            self._spot_loaded = True
            return
        elif st_df is not None:
            logger.warning(
                "ST_CLIENT spot returned %d stocks (< 4500 full-market threshold); "
                "treating as partial and continuing fallback chain",
                len(st_df),
            )

        # Fallback Layer 3: Tencent batch (WATCHLIST partial).
        # Per cx P0: Tencent partial MUST NOT overwrite an existing
        # healthy full-market cache. The write below uses source =
        # "tencent_partial" and _write_spot_disk_cache refuses to
        # overwrite a > 4500-row existing cache.
        logger.error(
            "Both AKShare AND ST_CLIENT spot failed — falling back to Tencent "
            "(WATCHLIST-only, ~%d stocks). Downstream screeners must treat "
            "this as PARTIAL coverage.",
            50,
        )
        self._spot_cache = self._load_spot_tencent()
        self._spot_partial = True
        self._write_spot_disk_cache(self._spot_cache, "tencent_partial")
        self._spot_loaded = True  # Don't retry even if Tencent also failed

    def warm_spot_cache(self) -> dict:
        """Force-refresh the full-market spot cache and return cache metadata."""
        self._spot_cache = None
        self._spot_loaded = False
        self._akshare_down = False
        self._load_spot_cache()

        meta = {}
        if self._spot_cache_meta_path.exists():
            try:
                meta = json.loads(self._spot_cache_meta_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to read warmed spot cache metadata: {e}")
        return {
            "row_count": int(len(self._spot_cache)) if self._spot_cache is not None else 0,
            "source": meta.get("source", "memory"),
            "created_at": meta.get("created_at", ""),
            "cache_path": str(self._spot_cache_path),
        }

    def _load_spot_stclient(self) -> Optional[pd.DataFrame]:
        """Fallback 2: load FULL-MARKET realtime quotes via ST_CLIENT
        (TuShare-compatible). Tries before Tencent because it returns
        the full 5000+ universe, not a 50-stock watchlist.

        cx code review 2026-06-04: ST_CLIENT.realtime_list() and
        realtime_quote() have existed since project inception but
        were never wired into MarketCollector — historical tech debt.
        AKShare→Tencent skipped this layer entirely, causing the
        2026-06-03 22:00 incident where AKShare failed and the
        universe silently shrank from 5000+ to 300.
        """
        try:
            import os
            try:
                from ST_CLIENT import StockToday
            except ImportError as e:
                logger.warning("ST_CLIENT not importable for spot fallback: %s", e)
                return None

            token = None
            try:
                from config.settings import ST_TOKEN
                token = ST_TOKEN
            except Exception:
                pass
            if not token:
                token_file = Path(__file__).resolve().parents[2] / ".st_token"
                if token_file.exists():
                    token = token_file.read_text().strip()
            if not token:
                logger.warning("ST_CLIENT token unavailable; skipping realtime spot fallback")
                return None

            st = StockToday(token=token)
            try:
                raw = st.realtime_list()
            except Exception as e:  # noqa: BLE001
                logger.warning("ST_CLIENT realtime_list failed: %s", e)
                return None

            # Normalize shape — server may return list of dicts OR
            # {data: list, columns: [...]} envelope.
            rows = raw
            if isinstance(raw, dict):
                rows = raw.get("data") or raw.get("items") or []
            if not rows:
                return None
            df = pd.DataFrame(rows)
            if df.empty:
                return None

            # Normalize column names to MarketCollector's expected
            # Chinese keys (代码 / 名称 / 最新价 / 涨跌幅 / 成交量).
            col_map = {
                "ts_code": "代码", "code": "代码", "symbol": "代码",
                "name": "名称", "stock_name": "名称",
                "price": "最新价", "close": "最新价", "current": "最新价",
                "pct_change": "涨跌幅", "change_pct": "涨跌幅", "chg_pct": "涨跌幅",
                "volume": "成交量", "vol": "成交量",
                "high": "最高", "low": "最低",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            # Strip exchange suffix if ts_code form
            if "代码" in df.columns:
                df["代码"] = df["代码"].astype(str).str.replace(
                    r"\.(SH|SZ|BJ)$", "", regex=True,
                )

            # 2026-06-04 cx round 4 P1-6: validate required columns are
            # present AND non-empty / parseable BEFORE returning. Pre-fix
            # the function only checked len(df) >= 4500 — a ST_CLIENT
            # field rename or schema drift would land an unusable frame
            # that looked OK by row count and downstream sanitizer
            # would silently misjudge everything.
            required = ["代码", "名称", "最新价", "涨跌幅", "成交量"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                logger.warning(
                    "ST_CLIENT spot fallback missing columns %s — refusing "
                    "to return a partial frame", missing,
                )
                return None
            # Non-empty fraction on the must-have identifying cols
            id_nonempty = (
                df["代码"].astype(str).str.strip().ne("").mean()
                if len(df) else 0.0
            )
            name_nonempty = (
                df["名称"].astype(str).str.strip().ne("").mean()
                if len(df) else 0.0
            )
            # Numeric parseability on price/change/volume
            price_numeric = pd.to_numeric(df["最新价"], errors="coerce").notna().mean()
            chg_numeric = pd.to_numeric(df["涨跌幅"], errors="coerce").notna().mean()
            vol_numeric = pd.to_numeric(df["成交量"], errors="coerce").notna().mean()
            if id_nonempty < 0.98 or name_nonempty < 0.98:
                logger.warning(
                    "ST_CLIENT spot fallback identifying-col non-empty "
                    "rate too low (code=%.2f%%, name=%.2f%%) — refusing.",
                    id_nonempty * 100, name_nonempty * 100,
                )
                return None
            if price_numeric < 0.95 or chg_numeric < 0.95 or vol_numeric < 0.90:
                logger.warning(
                    "ST_CLIENT spot fallback numeric-col parseable rate "
                    "too low (price=%.2f%%, chg=%.2f%%, vol=%.2f%%) — refusing.",
                    price_numeric * 100, chg_numeric * 100, vol_numeric * 100,
                )
                return None

            logger.info(
                "Loaded %d stocks via ST_CLIENT realtime_list (full-market, "
                "validated cols)",
                len(df),
            )
            return df
        except Exception as e:  # noqa: BLE001
            logger.warning("ST_CLIENT spot fallback raised: %s", e)
            return None

    def _load_spot_tencent(self) -> pd.DataFrame:
        """Fallback: load realtime quotes from Tencent Finance API.

        Fetches in batches since Tencent API supports multi-stock queries.
        """
        try:
            from config.watchlist import WATCHLIST, MARKET_STOCK

            # Build Tencent symbol list from watchlist
            symbols = []
            for code, name, market in WATCHLIST:
                if market != MARKET_STOCK:
                    continue
                prefix = code[:2].lower()
                num = code[2:]
                symbols.append(f"{prefix}{num}")

            if not symbols:
                return pd.DataFrame()

            # Tencent real-time API accepts comma-separated symbols
            batch_size = 50
            all_records = []

            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                symbol_str = ",".join(batch)
                url = f"https://qt.gtimg.cn/q={symbol_str}"

                try:
                    resp = requests.get(url, timeout=10)
                    if resp.status_code != 200:
                        continue

                    # Parse Tencent response format
                    for line in resp.text.strip().split("\n"):
                        if "~" not in line:
                            continue
                        parts = line.split("~")
                        if len(parts) < 46:
                            continue
                        try:
                            all_records.append({
                                "代码": parts[2],
                                "名称": parts[1],
                                "最新价": float(parts[3]) if parts[3] else 0,
                                "涨跌幅": float(parts[32]) if parts[32] else 0,
                                "成交量": float(parts[6]) if parts[6] else 0,
                                "最高": float(parts[33]) if parts[33] else 0,
                                "最低": float(parts[34]) if parts[34] else 0,
                            })
                        except (ValueError, IndexError):
                            continue
                except Exception as e:
                    logger.warning(f"Tencent batch failed: {e}")

            if all_records:
                df = pd.DataFrame(all_records)
                logger.info(f"Loaded {len(df)} stocks via Tencent")
                return df

        except Exception as e:
            logger.warning(f"Tencent spot fallback failed: {e}")

        return pd.DataFrame()

    def invalidate_cache(self):
        """Clear all caches to force fresh data on next run."""
        self._spot_cache = None
        self._spot_loaded = False
        self._akshare_down = False

    def fetch_realtime(self, code: str) -> dict:
        """Fetch realtime quote with automatic fallback."""
        try:
            symbol = code[2:]
            self._load_spot_cache()

            if self._spot_cache is None or self._spot_cache.empty:
                # Last resort: single stock via Tencent
                return self._fetch_realtime_tencent_single(code)

            row = self._spot_cache[self._spot_cache["代码"] == symbol]
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

        except Exception as e:
            logger.warning(f"fetch_realtime failed for {code}: {e}")
            return self._fetch_realtime_tencent_single(code)

    def _fetch_realtime_tencent_single(self, code: str) -> dict:
        """Fallback: fetch single stock realtime from Tencent."""
        try:
            prefix = code[:2].lower()
            symbol = code[2:]
            tc_symbol = f"{prefix}{symbol}"

            resp = requests.get(f"https://qt.gtimg.cn/q={tc_symbol}", timeout=10)
            if resp.status_code != 200:
                return {}

            parts = resp.text.split("~")
            if len(parts) < 46:
                return {}

            return {
                "price": float(parts[3]) if parts[3] else 0,
                "change_pct": float(parts[32]) if parts[32] else 0,
                "volume": float(parts[6]) if parts[6] else 0,
                "high": float(parts[33]) if parts[33] else 0,
                "low": float(parts[34]) if parts[34] else 0,
            }

        except Exception as e:
            logger.warning(f"Tencent single realtime failed for {code}: {e}")
            return {}

    def fetch_batch_daily(self, codes: list, days: int = 60) -> dict:
        """Fetch daily data for multiple stocks."""
        result = {}
        for code in codes:
            df = self.fetch_daily(code, days)
            if not df.empty:
                result[code] = df
        return result
