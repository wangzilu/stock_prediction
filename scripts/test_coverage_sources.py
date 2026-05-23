"""Test alternative data sources for LLM event coverage.

Tests:
  1. 东财股吧 (Guba) — 个股讨论帖
  2. 互动易 (irm_qa) — 投资者问答 (via ST_CLIENT)
  3. 公告原文 (anns_d) — 交易所公告 (via ST_CLIENT)
  4. 雪球 — 个股讨论

Usage:
    python scripts/test_coverage_sources.py
"""
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}


def test_guba():
    """Test Eastmoney Guba (股吧) scraping."""
    logger.info("=== 1. 东财股吧 (Guba) ===")

    # Try multiple endpoints
    endpoints = [
        ("guba HTML scrape", f"https://guba.eastmoney.com/list,600519.html"),
        ("guba API v1", f"https://guba.eastmoney.com/interface/GetData?path=newtopic/api&type=8&code=600519&ps=5&p=1"),
        ("guba API v2", f"https://gubaapi.eastmoney.com/v1/Article/ArticleList?code=600519&ps=5&p=1"),
        ("guba mobile", f"https://gbapi.eastmoney.com/mapi/post/list?code=600519&ps=5&p=1"),
    ]

    for name, url in endpoints:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            logger.info(f"  {name}: status={resp.status_code}, len={len(resp.text)}")
            if resp.status_code == 200 and len(resp.text) > 100:
                # Check if it's HTML or JSON
                if resp.text.strip().startswith(("{", "[")):
                    data = resp.json()
                    logger.info(f"    JSON keys: {list(data.keys())[:5]}")
                else:
                    # Count post-like elements in HTML
                    n_titles = resp.text.count("post_title") + resp.text.count("articleh")
                    logger.info(f"    HTML with ~{n_titles} post references")
        except Exception as e:
            logger.info(f"  {name}: ERROR {e}")


def test_xueqiu():
    """Test Xueqiu (雪球) API."""
    logger.info("\n=== 2. 雪球 ===")

    endpoints = [
        ("xueqiu stock page", "https://xueqiu.com/S/SH600519"),
        ("xueqiu API timeline", "https://xueqiu.com/query/v1/symbol/search/status?u=100&count=10&comment=0&symbol=SH600519&hl=0&source=all&sort=time&q=&type=11"),
        ("xueqiu API comment", "https://xueqiu.com/statuses/stock_timeline.json?symbol_id=SH600519&count=5&source=all"),
    ]

    xueqiu_headers = {
        **HEADERS,
        "Cookie": "xq_a_token=test",  # may need real cookie
    }

    for name, url in endpoints:
        try:
            resp = requests.get(url, headers=xueqiu_headers, timeout=10, allow_redirects=False)
            logger.info(f"  {name}: status={resp.status_code}")
            if resp.status_code == 200:
                if len(resp.text) > 100:
                    try:
                        data = resp.json()
                        logger.info(f"    JSON keys: {list(data.keys())[:5]}")
                    except:
                        logger.info(f"    HTML len={len(resp.text)}")
            elif resp.status_code in (302, 403):
                logger.info(f"    Needs auth/cookie")
        except Exception as e:
            logger.info(f"  {name}: ERROR {e}")


def test_irm_qa():
    """Test 互动易 via ST_CLIENT."""
    logger.info("\n=== 3. 互动易 (ST_CLIENT) ===")

    try:
        from ST_CLIENT import StockToday
        token = Path(PROJECT_ROOT / ".st_token").read_text().strip()
        st = StockToday(token=token)

        # Test irm_qa_sh (上交所互动易)
        logger.info("  Testing irm_qa_sh...")
        try:
            r = st.irm_qa_sh(ts_code="600519.SH")
            if isinstance(r, dict):
                data = r.get("data")
                code = r.get("code")
                msg = r.get("msg", "")[:50]
                if code == 1:
                    logger.info(f"    需要升级: {msg}")
                elif data and isinstance(data, list):
                    logger.info(f"    ✅ {len(data)} records")
                    if data:
                        logger.info(f"    Sample keys: {list(data[0].keys())[:8]}")
                        logger.info(f"    Sample: {json.dumps(data[0], ensure_ascii=False)[:200]}")
                else:
                    logger.info(f"    Empty (code={code}, msg={msg})")
            elif isinstance(r, list):
                logger.info(f"    ✅ {len(r)} records")
        except Exception as e:
            logger.info(f"    ERROR: {e}")

        # Test irm_qa_sz (深交所互动易)
        logger.info("  Testing irm_qa_sz...")
        try:
            r = st.irm_qa_sz(ts_code="000001.SZ")
            if isinstance(r, dict):
                data = r.get("data")
                code = r.get("code")
                msg = r.get("msg", "")[:50]
                if code == 1:
                    logger.info(f"    需要升级: {msg}")
                elif data and isinstance(data, list):
                    logger.info(f"    ✅ {len(data)} records")
                    if data:
                        logger.info(f"    Sample: {json.dumps(data[0], ensure_ascii=False)[:200]}")
                else:
                    logger.info(f"    Empty (code={code}, msg={msg})")
            elif isinstance(r, list):
                logger.info(f"    ✅ {len(r)} records")
        except Exception as e:
            logger.info(f"    ERROR: {e}")

    except Exception as e:
        logger.info(f"  ST_CLIENT error: {e}")


def test_announcements():
    """Test 公告 via ST_CLIENT and direct API."""
    logger.info("\n=== 4. 公告原文 ===")

    # Method 1: ST_CLIENT anns_d
    try:
        from ST_CLIENT import StockToday
        token = Path(PROJECT_ROOT / ".st_token").read_text().strip()
        st = StockToday(token=token)

        logger.info("  Testing anns_d (ST_CLIENT)...")
        r = st.anns_d(ts_code="600519.SH")
        if isinstance(r, dict):
            code = r.get("code")
            msg = r.get("msg", "")[:50]
            data = r.get("data")
            if code == 1:
                logger.info(f"    需要龙虾套餐: {msg}")
            elif data:
                logger.info(f"    ✅ {len(data) if isinstance(data, list) else 'dict'}")
            else:
                logger.info(f"    Empty")
    except Exception as e:
        logger.info(f"  anns_d error: {e}")

    # Method 2: 东财公告 API (free)
    logger.info("  Testing 东财公告 API...")
    try:
        url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
        params = {
            "sr": -1,
            "page_size": 5,
            "page_index": 1,
            "ann_type": "A",
            "stock_list": "600519",
            "f_node": "0",
            "s_node": "0",
        }
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("data", {}).get("list", [])
            logger.info(f"    ✅ {len(items)} announcements")
            for item in items[:3]:
                title = item.get("title", "")
                date = item.get("notice_date", "")
                logger.info(f"      [{date}] {title[:50]}")
        else:
            logger.info(f"    status={resp.status_code}")
    except Exception as e:
        logger.info(f"    ERROR: {e}")

    # Method 3: 巨潮资讯 API (free)
    logger.info("  Testing 巨潮资讯 API...")
    try:
        url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
        data = {
            "pageNum": 1,
            "pageSize": 5,
            "tabName": "fulltext",
            "stock": "600519",
            "category": "",
            "seDate": "",
        }
        resp = requests.post(url, data=data, headers={**HEADERS, "Content-Type": "application/x-www-form-urlencoded"}, timeout=10)
        if resp.status_code == 200:
            result = resp.json()
            items = result.get("announcements", [])
            logger.info(f"    ✅ {len(items)} announcements")
            for item in items[:3]:
                title = item.get("announcementTitle", "")
                date = item.get("announcementTime", "")
                if isinstance(date, (int, float)):
                    from datetime import datetime
                    date = datetime.fromtimestamp(date / 1000).strftime("%Y-%m-%d")
                logger.info(f"      [{date}] {title[:50]}")
        else:
            logger.info(f"    status={resp.status_code}")
    except Exception as e:
        logger.info(f"    ERROR: {e}")


def main():
    logger.info("Testing alternative data sources for LLM event coverage...\n")
    test_guba()
    test_xueqiu()
    test_irm_qa()
    test_announcements()

    logger.info("\n=== SUMMARY ===")
    logger.info("Run this script and check which sources return data.")
    logger.info("Sources that work can be integrated into LLM event pipeline.")


if __name__ == "__main__":
    main()
