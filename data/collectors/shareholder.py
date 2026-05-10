"""Shareholder and corporate event factor collector.

Quarterly: shareholder count, pledge ratio, restricted unlock
Event: insider trades, repurchase
Stores to data/storage/shareholder_features.parquet.
"""
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "storage"
SHAREHOLDER_PATH = DATA_DIR / "shareholder_features.parquet"


class ShareholderCollector:

    def fetch_holder_count(self, date: str = None) -> pd.DataFrame:
        """Fetch shareholder count for all stocks (quarterly disclosure).

        Args:
            date: format YYYYMMDD, defaults to latest quarter end.
        """
        try:
            import akshare as ak
            if not date:
                # Latest quarter end
                now = datetime.now()
                q = (now.month - 1) // 3
                quarter_ends = ["0331", "0630", "0930", "1231"]
                date = f"{now.year}{quarter_ends[q]}"

            df = ak.stock_hold_num_cninfo(date=date)
            if df is None or df.empty:
                return pd.DataFrame()
            logger.info(f"Shareholder count: {len(df)} stocks for {date}")
            return df
        except Exception as e:
            logger.warning(f"Shareholder count failed: {e}")
            return pd.DataFrame()

    def fetch_pledge_ratio(self) -> pd.DataFrame:
        """Fetch equity pledge ratio for all stocks."""
        try:
            import akshare as ak
            df = ak.stock_gpzy_pledge_ratio_em()
            if df is None or df.empty:
                return pd.DataFrame()
            logger.info(f"Pledge data: {len(df)} stocks")
            return df
        except Exception as e:
            logger.warning(f"Pledge data failed: {e}")
            return pd.DataFrame()

    def fetch_restricted_unlock(self) -> pd.DataFrame:
        """Fetch upcoming restricted share unlock schedule."""
        try:
            import akshare as ak
            df = ak.stock_restricted_release_queue_sina()
            if df is None or df.empty:
                return pd.DataFrame()
            logger.info(f"Restricted unlock: {len(df)} entries")
            return df
        except Exception as e:
            logger.warning(f"Restricted unlock failed: {e}")
            return pd.DataFrame()

    def fetch_repurchase(self) -> pd.DataFrame:
        """Fetch company repurchase data."""
        try:
            import akshare as ak
            df = ak.stock_repurchase_em()
            if df is None or df.empty:
                return pd.DataFrame()
            return df
        except Exception as e:
            logger.warning(f"Repurchase data failed: {e}")
            return pd.DataFrame()

    def build_features(self) -> pd.DataFrame:
        """Build shareholder features from all sources."""
        features = []

        # 1. Shareholder count
        holder_df = self.fetch_holder_count()
        if not holder_df.empty:
            for _, row in holder_df.iterrows():
                code = str(row.get("证券代码", "")).zfill(6)
                features.append({
                    "code": code,
                    "qlib_code": f"SH{code}" if code.startswith("6") else f"SZ{code}",
                    "holder_count": _safe_float(row.get("股东人数")),
                    "holder_count_change": _safe_float(row.get("股东人数增幅")),
                })

        if not features:
            return pd.DataFrame()

        result = pd.DataFrame(features)

        # 2. Merge pledge ratio
        pledge_df = self.fetch_pledge_ratio()
        if not pledge_df.empty and "股票代码" in pledge_df.columns:
            pledge_df = pledge_df.rename(columns={
                "股票代码": "code",
                "质押比例": "pledge_ratio",
                "质押股数": "pledged_shares",
            })
            pledge_df["code"] = pledge_df["code"].astype(str).str.zfill(6)
            result = result.merge(
                pledge_df[["code", "pledge_ratio"]].drop_duplicates("code"),
                on="code", how="left"
            )

        result["date"] = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Built shareholder features for {len(result)} stocks")
        return result

    def save(self, df: pd.DataFrame, path: Path = SHAREHOLDER_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def load(self, path: Path = SHAREHOLDER_PATH) -> pd.DataFrame:
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()


def _safe_float(val) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan
