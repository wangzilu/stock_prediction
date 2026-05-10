"""Macro / cross-asset factor collector.

All stocks share the same macro features (broadcast).
Daily update, < 1 minute total.
Stores to data/storage/macro_features.parquet.
"""
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "storage"
MACRO_PATH = DATA_DIR / "macro_features.parquet"


class MacroFactorCollector:

    def fetch_all(self) -> dict:
        """Fetch all macro indicators. Returns dict of factor_name -> value."""
        factors = {}

        # 1. Bond yields
        try:
            import akshare as ak
            df = ak.bond_china_yield(
                start_date=(datetime.now().replace(day=1)).strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d"),
            )
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                factors["bond_10y"] = _safe_float(latest.get("中国国债收益率10年"))
                factors["bond_1y"] = _safe_float(latest.get("中国国债收益率1年"))
                factors["term_spread"] = factors.get("bond_10y", 0) - factors.get("bond_1y", 0)
                logger.info(f"Bond yields: 10Y={factors.get('bond_10y')}, 1Y={factors.get('bond_1y')}")
        except Exception as e:
            logger.warning(f"Bond yield failed: {e}")

        # 2. FX (USD/CNY)
        try:
            import akshare as ak
            df = ak.fx_spot_quote()
            if df is not None and not df.empty:
                usd_row = df[df["货币对"].str.contains("USD/CNY", na=False)]
                if not usd_row.empty:
                    factors["usdcny"] = _safe_float(usd_row.iloc[0].get("买报价", usd_row.iloc[0].get("最新价")))
                    logger.info(f"USDCNY: {factors.get('usdcny')}")
        except Exception as e:
            logger.warning(f"FX failed: {e}")

        # 3. Commodities via futures
        for symbol, name in [("CU0", "copper"), ("I0", "iron_ore"), ("SC0", "crude_oil"), ("AU0", "gold_futures")]:
            try:
                import akshare as ak
                df = ak.futures_main_sina(symbol=symbol)
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    factors[f"{name}_close"] = _safe_float(latest.get("收盘价", latest.get("close")))
            except Exception:
                pass

        # 4. Macro indicators (monthly, forward-fill)
        for func_name, factor_name in [
            ("macro_china_pmi", "pmi"),
            ("macro_china_cpi", "cpi"),
        ]:
            try:
                import akshare as ak
                func = getattr(ak, func_name, None)
                if func:
                    df = func()
                    if df is not None and not df.empty:
                        factors[factor_name] = _safe_float(df.iloc[-1].iloc[-1])
            except Exception:
                pass

        factors["date"] = datetime.now().strftime("%Y-%m-%d")
        factors["collected_at"] = datetime.now().isoformat(timespec="seconds")
        logger.info(f"Macro factors: {len(factors)} collected")
        return factors

    def save(self, factors: dict, path: Path = MACRO_PATH):
        """Append today's macro factors to history."""
        path.parent.mkdir(parents=True, exist_ok=True)

        new_row = pd.DataFrame([factors])
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_row], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
        else:
            combined = new_row

        combined.to_parquet(path, index=False)

    def load(self, path: Path = MACRO_PATH) -> pd.DataFrame:
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()


def _safe_float(val) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan
