"""Tradable Universe Filter — stock filtering for training and inference.

Per code-review P2 2026-05-31: docstring previously promised low-liquidity
filter (amount < 5M) + min_market_cap filter, but the implementation
never used them — `min_daily_amount` and `min_market_cap` were dead
parameters. Docstring now reflects what the code ACTUALLY does. The
liquidity / market-cap filters are deferred to a follow-up that plumbs
$amount and market cap data through (see feature_merger / spot_cache).

Training-time filter (get_tradable_mask) — actually implemented:
  1. ST / *ST / 退市整理 — from st_stock_list.json
  2. IPO < 60 trading days — cumulative day count
  3. BSE stocks — instrument prefix "bj"

Training-time filters NOT YET implemented (despite earlier comments):
  - Suspended (volume=0): applied separately in candidate_sanitizer at
    recommendation time, not in tradable_mask at training time
  - 一字板 (zero range): same as above
  - Low liquidity (amount < 5M): not implemented anywhere yet
  - Market cap filter: not implemented anywhere yet

Inference-time filter (filter_predictions):
  - ST stocks (from cached list)
  - BSE stocks
  NOTE: ADV, suspended, limit-up/down are NOT checked at inference time
  because real-time market data is not available when predictions are
  generated. CandidateSanitizer at run_daily_recommendation time covers
  the runtime guards.

Usage:
    from models.universe_filter import UniverseFilter

    uf = UniverseFilter()
    mask = uf.get_tradable_mask(feature_cache_index)  # boolean Series
    X_clean = X[mask]
"""
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "storage"


class UniverseFilter:
    """Filter stocks to tradable universe."""

    def __init__(
        self,
        min_listing_days: int = 60,
        exclude_st: bool = True,
        exclude_bse: bool = True,         # 北交所 (codes starting with 8/4 on BSE)
    ):
        # Per code-review P2 2026-05-31: removed dead parameters
        # `min_daily_amount`, `min_market_cap`, `exclude_suspended` —
        # they were accepted but never used in get_tradable_mask. Better
        # to delete than leave the false promise. If/when liquidity or
        # market-cap filters are added, re-add the params with actual
        # implementation in the same PR.
        self.min_listing_days = min_listing_days
        self.exclude_st = exclude_st
        self.exclude_bse = exclude_bse

        # Cache
        self._st_set = None
        self._listing_days = None

    def get_tradable_mask(self, index: pd.MultiIndex) -> pd.Series:
        """Return boolean mask for tradable stocks.

        Args:
            index: MultiIndex (datetime, instrument) from feature cache

        Returns:
            pd.Series of bool, True = tradable
        """
        mask = pd.Series(True, index=index)
        n_total = len(mask)

        # 1. Exclude ST stocks
        if self.exclude_st:
            st_set = self._load_st_set()
            if st_set:
                instruments = index.get_level_values(1).astype(str)
                is_st = instruments.isin(st_set)
                n_st = is_st.sum()
                mask &= ~is_st
                logger.info(f"  ST filter: removed {n_st} stock-days ({n_st/n_total*100:.1f}%)")

        # 2. Exclude IPO < N days
        if self.min_listing_days > 0:
            listing_mask = self._listing_days_mask(index)
            n_ipo = (~listing_mask).sum()
            mask &= listing_mask
            logger.info(f"  IPO filter (<{self.min_listing_days}d): removed {n_ipo} stock-days ({n_ipo/n_total*100:.1f}%)")

        # NOTE: low-liquidity filter is NOT implemented (was promised in
        # earlier docstring but never coded). CandidateSanitizer at
        # recommendation time covers volume==0 (suspended) check via the
        # spot quote, which is sufficient for production runtime.
        # Training-time low-liquidity filter is a separate follow-up.

        # 4. Exclude BSE stocks (北交所: codes like bj430xxx, bj83xxxx)
        if self.exclude_bse:
            instruments = index.get_level_values(1).astype(str)
            is_bse = instruments.str.startswith("bj")
            n_bse = is_bse.sum()
            if n_bse > 0:
                mask &= ~is_bse
                logger.info(f"  BSE filter: removed {n_bse} stock-days ({n_bse/n_total*100:.1f}%)")

        n_removed = n_total - mask.sum()
        logger.info(f"  Universe filter: {n_total} -> {mask.sum()} ({n_removed} removed, {n_removed/n_total*100:.1f}%)")
        return mask

    def filter_predictions(self, predictions: dict) -> dict:
        """Filter a prediction dict {code: score} to tradable universe.

        Codes are in uppercase format (SH600519, SZ000001).
        """
        st_set = self._load_st_set()
        filtered = {}
        n_removed = 0

        for code, score in predictions.items():
            code_lower = code.lower()

            # ST filter
            if self.exclude_st and code_lower in st_set:
                n_removed += 1
                continue

            # BSE filter
            if self.exclude_bse and code_lower.startswith("bj"):
                n_removed += 1
                continue

            filtered[code] = score

        if n_removed > 0:
            logger.info(f"  Prediction filter: {len(predictions)} -> {len(filtered)} ({n_removed} removed)")
        return filtered

    def _load_st_set(self) -> set:
        """Load set of ST stock codes (lowercase qlib format)."""
        if self._st_set is not None:
            return self._st_set

        st_set = set()

        # Method 1: ST_CLIENT bak_basic — check name for "ST"
        try:
            from ST_CLIENT import StockToday
            token_file = PROJECT_ROOT / ".st_token"
            if token_file.exists():
                token = token_file.read_text().strip()
                if token:
                    st_client = StockToday(token=token)
                    for days_back in range(1, 10):
                        date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
                        result = st_client.bak_basic(trade_date=date)
                        if isinstance(result, dict) and result.get("data"):
                            data = result["data"]
                            for item in data:
                                name = str(item.get("name", ""))
                                ts_code = str(item.get("ts_code", ""))
                                if "ST" in name.upper():
                                    # Convert ts_code (600519.SH) to qlib (sh600519)
                                    qlib_code = _ts_to_qlib(ts_code)
                                    if qlib_code:
                                        st_set.add(qlib_code)
                            logger.info(f"  ST set loaded: {len(st_set)} stocks (from bak_basic {date})")
                            break
        except Exception as e:
            logger.warning(f"  Failed to load ST set from ST_CLIENT: {e}")

        # Method 2: Fallback — check Qlib instrument names
        if not st_set:
            # Try loading from a cached file
            st_cache = DATA_DIR / "st_stock_list.json"
            if st_cache.exists():
                import json
                st_set = set(json.loads(st_cache.read_text()))
                logger.info(f"  ST set loaded from cache: {len(st_set)} stocks")

        self._st_set = st_set
        return st_set

    def _listing_days_mask(self, index: pd.MultiIndex) -> pd.Series:
        """Build per-stock listing day count mask."""
        # Count cumulative trading days per stock
        dates_by_stock = {}
        for dt, inst in index:
            inst_str = str(inst)
            if inst_str not in dates_by_stock:
                dates_by_stock[inst_str] = []
            dates_by_stock[inst_str].append(dt)

        mask_values = np.ones(len(index), dtype=bool)

        # For each stock, mark first N days as False
        idx_values = index.to_frame(index=False)
        for inst, dt_list in dates_by_stock.items():
            dt_sorted = sorted(set(dt_list))
            if len(dt_sorted) <= self.min_listing_days:
                # Entire history is too short — mark all as untradable
                inst_mask = idx_values.iloc[:, 1].astype(str) == inst
                mask_values[inst_mask.values] = False
            else:
                # Only mark first N days
                cutoff_date = dt_sorted[self.min_listing_days]
                inst_mask = (idx_values.iloc[:, 1].astype(str) == inst) & (idx_values.iloc[:, 0] < cutoff_date)
                mask_values[inst_mask.values] = False

        return pd.Series(mask_values, index=index)


def _ts_to_qlib(ts_code: str) -> str:
    """Convert TuShare format (600519.SH) to Qlib format (sh600519)."""
    if "." not in ts_code:
        return ""
    code, exchange = ts_code.split(".")
    return f"{exchange.lower()}{code}"


def save_st_list():
    """Utility: fetch and save current ST stock list for offline use."""
    import json

    uf = UniverseFilter()
    st_set = uf._load_st_set()
    if st_set:
        out_path = DATA_DIR / "st_stock_list.json"
        out_path.write_text(json.dumps(sorted(st_set), ensure_ascii=False, indent=2))
        logger.info(f"Saved {len(st_set)} ST stocks to {out_path}")
    return st_set
