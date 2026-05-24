"""Batch fetch missing ST_CLIENT data sources.

Usage:
    python scripts/fetch_st_batch.py --list          # show what's available
    python scripts/fetch_st_batch.py --all            # fetch everything
    python scripts/fetch_st_batch.py --only balancesheet income cashflow
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"

# Data sources to fetch: (name, method, kwargs_fn, description)
# kwargs_fn takes (st, stock_list) and returns list of (kwargs, filename_suffix) to call

SOURCES = {
    # ===== 财务三表 (按报告期, 全A) =====
    "balancesheet": {
        "desc": "资产负债表",
        "method": "balancesheet",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "start_date": "20210101", "end_date": "20261231"},
    },
    "income": {
        "desc": "利润表",
        "method": "income",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "start_date": "20210101", "end_date": "20261231"},
    },
    "cashflow": {
        "desc": "现金流量表",
        "method": "cashflow",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "start_date": "20210101", "end_date": "20261231"},
    },
    # ===== 股东 =====
    "top10_holders": {
        "desc": "十大股东",
        "method": "top10_holders",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "start_date": "20210101", "end_date": "20261231"},
    },
    "top10_floatholders": {
        "desc": "十大流通股东",
        "method": "top10_floatholders",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "start_date": "20210101", "end_date": "20261231"},
    },
    "stk_holdertrade": {
        "desc": "股东增减持",
        "method": "stk_holdertrade",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "start_date": "20210101", "end_date": "20261231"},
    },
    "share_float": {
        "desc": "限售解禁",
        "method": "share_float",
        "per_stock": False,
        "kwargs": lambda: {"start_date": "20210101", "end_date": "20261231"},
    },
    # ===== 交易辅助 =====
    "suspend_d": {
        "desc": "停牌信息",
        "method": "suspend_d",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "start_date": "20210101", "end_date": "20261231"},
    },
    "adj_factor": {
        "desc": "复权因子",
        "method": "adj_factor",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "start_date": "20210101", "end_date": "20261231"},
    },
    "limit_list_d": {
        "desc": "涨跌停统计(日)",
        "method": "limit_list_d",
        "per_stock": False,
        "kwargs": lambda: {"start_date": "20240101", "end_date": "20261231"},
    },
    "stk_limit": {
        "desc": "涨跌停价格",
        "method": "stk_limit",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "start_date": "20240101", "end_date": "20261231"},
    },
    # ===== 指数 =====
    "index_weight": {
        "desc": "指数权重(中证500)",
        "method": "index_weight",
        "per_stock": False,
        "kwargs": lambda: {"index_code": "000905.SH"},
    },
    # ===== 北向资金 =====
    "hsgt_top10": {
        "desc": "北向十大成交股",
        "method": "hsgt_top10",
        "per_stock": False,
        "kwargs": lambda: {"start_date": "20210101", "end_date": "20261231"},
    },
    # ===== PIT关键 =====
    "disclosure_date": {
        "desc": "财报披露日期",
        "method": "disclosure_date",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "end_date": "20261231"},
    },
    # ===== 其他 =====
    "namechange": {
        "desc": "股票更名历史",
        "method": "namechange",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code},
    },
    "stk_holdernumber": {
        "desc": "股东人数",
        "method": "stk_holdernumber",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "start_date": "20210101", "end_date": "20261231"},
    },
    "fina_mainbz": {
        "desc": "主营业务构成",
        "method": "fina_mainbz",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "period": "20251231", "type": "P"},
    },
    # === 新增高价值数据 ===
    "kpl_concept": {
        "desc": "开盘啦概念板块列表",
        "method": "kpl_concept",
        "per_stock": False,
        "kwargs": lambda: {},
    },
    "kpl_concept_cons": {
        "desc": "开盘啦概念板块成分",
        "method": "kpl_concept_cons",
        "per_stock": False,
        "kwargs": lambda: {},
    },
    "limit_list_ths": {
        "desc": "同花顺涨跌停股票列表",
        "method": "limit_list_ths",
        "per_stock": False,
        "kwargs": lambda: {"start_date": "20240101", "end_date": "20261231"},
    },
    "limit_step": {
        "desc": "连板晋级数据",
        "method": "limit_step",
        "per_stock": False,
        "kwargs": lambda: {"start_date": "20240101", "end_date": "20261231"},
    },
    "stk_factor_pro": {
        "desc": "ST专业技术因子",
        "method": "stk_factor_pro",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "start_date": "20240101", "end_date": "20261231"},
    },
    "research_report": {
        "desc": "券商研报",
        "method": "research_report",
        "per_stock": False,
        "kwargs": lambda: {"start_date": "20240101", "end_date": "20261231"},
    },
    "moneyflow_ths": {
        "desc": "同花顺资金流",
        "method": "moneyflow_ths",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "start_date": "20240101", "end_date": "20261231"},
    },
    "index_member_all": {
        "desc": "指数成分历史",
        "method": "index_member_all",
        "per_stock": False,
        "kwargs": lambda: {"index_code": "000905.SH"},
    },
    # === 新增高价值数据 ===
    "etf_share_size": {
        "desc": "ETF份额变化(被动资金流)",
        "method": "etf_share_size",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "start_date": "20240101", "end_date": "20261231"},
    },
    "disclosure_date": {
        "desc": "财报披露日期(PIT关键)",
        "method": "disclosure_date",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "end_date": "20261231"},
    },
    "irm_qa_sh": {
        "desc": "互动易问答-上交所",
        "method": "irm_qa_sh",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code},
    },
    "irm_qa_sz": {
        "desc": "互动易问答-深交所",
        "method": "irm_qa_sz",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code},
    },
    "cyq_chips": {
        "desc": "筹码分布详细",
        "method": "cyq_chips",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "start_date": "20240101", "end_date": "20261231"},
    },
    "stk_auction": {
        "desc": "集合竞价数据",
        "method": "stk_auction",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "start_date": "20260101", "end_date": "20261231"},
    },
    "cn_pmi": {
        "desc": "PMI指数(宏观regime)",
        "method": "cn_pmi",
        "per_stock": False,
        "kwargs": lambda: {"start_date": "20210101", "end_date": "20261231"},
    },
    "cn_cpi": {
        "desc": "CPI指数(宏观regime)",
        "method": "cn_cpi",
        "per_stock": False,
        "kwargs": lambda: {"start_date": "20210101", "end_date": "20261231"},
    },
    "cn_m": {
        "desc": "货币供应M1/M2(宏观regime)",
        "method": "cn_m",
        "per_stock": False,
        "kwargs": lambda: {"start_date": "20210101", "end_date": "20261231"},
    },
    "shibor": {
        "desc": "银行间利率(流动性regime)",
        "method": "shibor",
        "per_stock": False,
        "kwargs": lambda: {"start_date": "20210101", "end_date": "20261231"},
    },
    "fund_portfolio": {
        "desc": "基金持仓(机构重仓/抱团)",
        "method": "fund_portfolio",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code},
    },
    "fund_share": {
        "desc": "基金份额变化(资金流入流出)",
        "method": "fund_share",
        "per_stock": True,
        "kwargs": lambda code: {"ts_code": code, "start_date": "20240101", "end_date": "20261231"},
    },
}


def get_stock_list():
    """Get list of A-share stock codes in ts_code format."""
    token_file = PROJECT_ROOT / ".st_token"
    token = token_file.read_text().strip()
    from ST_CLIENT import StockToday
    st = StockToday(token=token)

    # Try last 5 days for bak_basic
    for days_back in range(1, 10):
        date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
        try:
            result = st.bak_basic(trade_date=date)
            if isinstance(result, dict) and result.get("data"):
                codes = [item["ts_code"] for item in result["data"]]
                return codes
            elif isinstance(result, list) and result:
                codes = [item.get("ts_code", "") for item in result if item.get("ts_code")]
                if codes:
                    return codes
        except Exception:
            continue
    return []


def fetch_source(name: str, config: dict, stock_list: list = None):
    """Fetch a single data source."""
    token_file = PROJECT_ROOT / ".st_token"
    token = token_file.read_text().strip()
    from ST_CLIENT import StockToday
    st = StockToday(token=token)

    method = getattr(st, config["method"], None)
    if method is None:
        logger.error(f"  Method {config['method']} not found")
        return None

    output_path = DATA_DIR / f"st_{name}.parquet"

    if config.get("per_stock"):
        if not stock_list:
            logger.error("  Need stock list for per_stock source")
            return None

        all_records = []
        n = len(stock_list)
        for i, code in enumerate(stock_list):
            try:
                kwargs = config["kwargs"](code)
                result = method(**kwargs)
                if isinstance(result, dict):
                    data = result.get("data")
                    if isinstance(data, list) and data:
                        for item in data:
                            item["ts_code"] = code
                        all_records.extend(data)
                elif isinstance(result, list) and result:
                    for item in result:
                        item["ts_code"] = code
                    all_records.extend(result)
            except Exception as e:
                if "升级套餐" in str(e) or "龙虾" in str(e):
                    logger.warning(f"  {name}: 需要龙虾套餐，跳过")
                    return None
                pass

            if (i + 1) % 100 == 0:
                logger.info(f"  {name}: {i+1}/{n} stocks, {len(all_records)} records")

            # Rate limit
            time.sleep(0.15)

        if all_records:
            df = pd.DataFrame(all_records)
            # Fix mixed types: convert all object columns to str
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str)
            df.to_parquet(str(output_path), index=False)
            logger.info(f"  ✅ {name}: {len(df)} records -> {output_path.name}")
            return df
        else:
            logger.warning(f"  {name}: 0 records")
            return None

    else:
        # Non-per-stock: single call
        try:
            kwargs = config["kwargs"]()
            result = method(**kwargs)
            if isinstance(result, dict):
                if result.get("code") == 1:
                    logger.warning(f"  {name}: {result.get('msg', 'API error')}")
                    return None
                data = result.get("data")
                if isinstance(data, list) and data:
                    df = pd.DataFrame(data)
                    for col in df.columns:
                        if df[col].dtype == object:
                            df[col] = df[col].astype(str)
                    df.to_parquet(str(output_path), index=False)
                    logger.info(f"  ✅ {name}: {len(df)} records -> {output_path.name}")
                    return df
            elif isinstance(result, list) and result:
                df = pd.DataFrame(result)
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str)
                df.to_parquet(str(output_path), index=False)
                logger.info(f"  ✅ {name}: {len(df)} records -> {output_path.name}")
                return df
            logger.warning(f"  {name}: empty result")
            return None
        except Exception as e:
            logger.error(f"  {name}: FAILED: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Batch fetch ST_CLIENT data")
    parser.add_argument("--list", action="store_true", help="List available sources")
    parser.add_argument("--all", action="store_true", help="Fetch all sources")
    parser.add_argument("--only", nargs="*", help="Fetch specific sources")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'Source':<25} {'Description':<30} {'Per-Stock':>10}")
        print("-" * 70)
        for name, cfg in sorted(SOURCES.items()):
            existing = (DATA_DIR / f"st_{name}.parquet").exists()
            status = "✅" if existing else "❌"
            print(f"{status} {name:<23} {cfg['desc']:<30} {'是' if cfg.get('per_stock') else '否':>10}")
        return

    targets = list(SOURCES.keys()) if args.all else (args.only or [])
    if not targets:
        parser.print_help()
        return

    # Get stock list for per-stock sources
    stock_list = None
    if any(SOURCES[t].get("per_stock") for t in targets if t in SOURCES):
        logger.info("Getting stock list...")
        stock_list = get_stock_list()
        logger.info(f"  {len(stock_list)} stocks")

    t_start = time.time()
    success = 0
    for name in targets:
        if name not in SOURCES:
            logger.warning(f"Unknown source: {name}")
            continue
        logger.info(f"\nFetching: {name} ({SOURCES[name]['desc']})...")
        result = fetch_source(name, SOURCES[name], stock_list)
        if result is not None:
            success += 1

    elapsed = time.time() - t_start
    logger.info(f"\nDone: {success}/{len(targets)} sources fetched in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
