"""Fetch current ST stock list via multiple fallback methods.

Tries: ST_CLIENT stock_basic → ST_CLIENT bak_basic → AKShare → Manual input.
Saves to data/storage/st_stock_list.json

Usage:
    python scripts/fetch_st_list.py
"""
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
OUTPUT = DATA_DIR / "st_stock_list.json"


def ts_to_qlib(ts_code: str) -> str:
    if "." not in ts_code:
        return ""
    code, exch = ts_code.split(".")
    return f"{exch.lower()}{code}"


def method_st_client():
    """Try ST_CLIENT stock_basic and bak_basic."""
    token_file = PROJECT_ROOT / ".st_token"
    if not token_file.exists():
        return None
    token = token_file.read_text().strip()
    if not token:
        return None

    from ST_CLIENT import StockToday
    st = StockToday(token=token)

    # Method 1: stock_basic
    try:
        r = st.stock_basic(exchange='', list_status='L')
        data = r.get("data") if isinstance(r, dict) else r
        if isinstance(data, list) and data:
            st_list = [d for d in data if "ST" in str(d.get("name", "")).upper()]
            if st_list:
                codes = [ts_to_qlib(d["ts_code"]) for d in st_list if d.get("ts_code")]
                logger.info(f"  stock_basic: found {len(codes)} ST stocks")
                return [c for c in codes if c]
    except Exception as e:
        logger.warning(f"  stock_basic failed: {e}")

    # Method 2: bak_basic (try last 15 days)
    for days_back in range(1, 15):
        date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
        try:
            r = st.bak_basic(trade_date=date)
            data = r.get("data") if isinstance(r, dict) else r
            if isinstance(data, list) and data:
                st_list = [d for d in data if "ST" in str(d.get("name", "")).upper()]
                if st_list:
                    codes = [ts_to_qlib(d["ts_code"]) for d in st_list if d.get("ts_code")]
                    logger.info(f"  bak_basic({date}): found {len(codes)} ST stocks")
                    return [c for c in codes if c]
        except Exception:
            continue

    return None


def method_akshare():
    """Try AKShare to get ST stock list."""
    try:
        import akshare as ak
        # Get all A-share stock list
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            name_col = "名称" if "名称" in df.columns else None
            code_col = "代码" if "代码" in df.columns else None
            if name_col and code_col:
                st_df = df[df[name_col].str.contains("ST", case=False, na=False)]
                codes = []
                for _, row in st_df.iterrows():
                    code = str(row[code_col])
                    if code.startswith("6"):
                        codes.append(f"sh{code}")
                    elif code.startswith(("0", "3")):
                        codes.append(f"sz{code}")
                    elif code.startswith(("4", "8")):
                        codes.append(f"bj{code}")
                logger.info(f"  AKShare: found {len(codes)} ST stocks")
                return codes
    except Exception as e:
        logger.warning(f"  AKShare failed: {e}")
    return None


def method_eastmoney():
    """Get ST list from Eastmoney API directly."""
    try:
        import requests
        # Eastmoney market overview API — filter ST stocks
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": 1, "pz": 500, "po": 1,
            "np": 1, "fltt": 2, "invt": 2,
            "fs": "m:0+t:80,m:1+t:80",  # ST板块
            "fields": "f12,f14",  # code, name
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        items = data.get("data", {}).get("diff", [])
        if not items:
            # Try alternative: search all stocks for ST in name
            params["fs"] = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"  # all A-shares
            params["pz"] = 6000
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            data = resp.json()
            items = data.get("data", {}).get("diff", [])

        codes = []
        for item in items:
            name = str(item.get("f14", ""))
            code = str(item.get("f12", ""))
            if "ST" in name.upper():
                if code.startswith("6"):
                    codes.append(f"sh{code}")
                elif code.startswith(("0", "3")):
                    codes.append(f"sz{code}")
        if codes:
            logger.info(f"  Eastmoney API: found {len(codes)} ST stocks")
        return codes if codes else None
    except Exception as e:
        logger.warning(f"  Eastmoney failed: {e}")
    return None


def main():
    logger.info("Fetching ST stock list...")

    # Try methods in order
    for method_name, method_fn in [
        ("ST_CLIENT", method_st_client),
        ("Eastmoney", method_eastmoney),
        ("AKShare", method_akshare),
    ]:
        logger.info(f"Trying {method_name}...")
        result = method_fn()
        if result:
            # Save
            result = sorted(set(result))
            OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            logger.info(f"\n✅ Saved {len(result)} ST stocks to {OUTPUT}")
            logger.info(f"Sample: {result[:10]}")
            return

    logger.error("All methods failed. Please check your network/token.")
    logger.info("You can manually create data/storage/st_stock_list.json with format:")
    logger.info('  ["sh600000", "sz000001", ...]')


if __name__ == "__main__":
    main()
