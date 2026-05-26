"""Supply Chain Mapper — map global events to A-share industries.

Two levels of mapping:
1. Company-level: specific edges (supply_chain_edges.yaml) → individual stocks
2. Industry-level: global event topics → A-share 申万行业 → all stocks in industry

Industry-level mapping greatly expands coverage from ~84 stocks to 1000+.

Usage:
    from factors.supply_chain_mapper import SupplyChainMapper

    mapper = SupplyChainMapper()
    # Get all affected stocks for an event
    affected = mapper.get_affected_stocks("Nvidia", "ai_server", direction=1)
    # Returns both company-level and industry-level hits
"""
import logging
import yaml
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "storage"
EDGES_PATH = PROJECT_ROOT / "data" / "config" / "supply_chain_edges.yaml"


# ============================================================
# Industry-level mapping: global topic → A-share 申万 L1 industry
# ============================================================

# When a global event affects a topic, ALL stocks in these industries get a signal
# weight is lower than company-level edges (0.2-0.4 vs 0.6-0.9)
TOPIC_TO_INDUSTRY = {
    "AI_server": {
        "电子I": 0.4,        # 半导体/光模块/PCB
        "通信I": 0.3,        # 光通信/数据中心
        "计算机I": 0.3,      # 服务器/AI应用
    },
    "semiconductor": {
        "电子I": 0.5,        # 芯片/设备/材料
    },
    "Apple_chain": {
        "电子I": 0.3,        # 消费电子零部件
    },
    "EV": {
        "汽车I": 0.4,        # 整车/零部件
        "电力设备I": 0.3,    # 电池/电机
    },
    "robot": {
        "机械设备I": 0.3,    # 减速器/电机/本体
        "电力设备I": 0.2,    # 伺服/控制器
    },
    "lithium": {
        "有色金属I": 0.4,    # 锂矿/钴
        "电力设备I": 0.3,    # 电池
        "基础化工I": 0.2,    # 电解液/正极
    },
    "solar": {
        "电力设备I": 0.4,    # 光伏组件/逆变器
        "基础化工I": 0.2,    # 硅料
    },
    "commodity": {
        "有色金属I": 0.3,    # 铜/稀土
        "石油石化I": 0.3,    # 油价
        "煤炭I": 0.2,        # 能源
        "钢铁I": 0.2,        # 工业金属
    },
    "strategic_material": {
        "有色金属I": 0.4,    # 稀土/钨/锗/镓
        "基础化工I": 0.2,    # 石墨/氟化工
    },
    "consumer_appliance": {
        "家用电器I": 0.5,    # 白电/小家电
    },
    "pharma": {
        "医药生物I": 0.4,    # 创新药/CRO
    },
    "defense": {
        "国防军工I": 0.5,    # 军工主机厂/零部件
    },
}


class SupplyChainMapper:
    """Maps global events to A-share stocks via company + industry level."""

    def __init__(self):
        self._company_edges = self._load_company_edges()
        self._industry_stocks = self._load_industry_stocks()

    def _load_company_edges(self) -> list[dict]:
        """Load company-level edges from YAML."""
        if not EDGES_PATH.exists():
            return []
        with open(EDGES_PATH) as f:
            return yaml.safe_load(f) or []

    def _load_industry_stocks(self) -> dict[str, list[str]]:
        """Load industry → stock mapping. Returns {industry_name: [qlib_codes]}."""
        result = {}

        # Try JQData SW classification
        jq_path = DATA_DIR / "jqdata" / "industry_sw.parquet"
        if jq_path.exists():
            ind = pd.read_parquet(jq_path)
            if "sw_l1_name" in ind.columns and "code" in ind.columns:
                for _, row in ind.iterrows():
                    industry = str(row.get("sw_l1_name", ""))
                    if not industry:
                        continue
                    jq_code = str(row.get("code", ""))
                    if ".XSHE" in jq_code:
                        qlib = f"sz{jq_code[:6]}"
                    elif ".XSHG" in jq_code:
                        qlib = f"sh{jq_code[:6]}"
                    else:
                        continue
                    result.setdefault(industry, []).append(qlib)

        logger.info(f"Industry mapping: {len(result)} industries, "
                    f"{sum(len(v) for v in result.values())} stocks")
        return result

    def get_affected_stocks(
        self,
        source_entity: str,
        topic: str,
        direction: int = 1,
        event_confidence: float = 0.7,
    ) -> list[dict]:
        """Get all affected A-share stocks for a global event.

        Returns list of {stock, weight, level, source} dicts.
        """
        affected = []
        seen_stocks = set()

        # Level 1: Company-level edges (high weight)
        for edge in self._company_edges:
            if edge.get("src_entity") == source_entity:
                stock = edge["dst_stock"].lower()
                w = (direction * edge.get("direction", 1) *
                     edge.get("weight", 0.5) * edge.get("confidence", 0.5) *
                     event_confidence)
                affected.append({
                    "stock": stock,
                    "weight": round(w, 4),
                    "level": "company",
                    "source": f"{source_entity}→{edge['dst_name']}",
                })
                seen_stocks.add(stock)

        # Level 2: Industry-level mapping (lower weight)
        industry_map = TOPIC_TO_INDUSTRY.get(topic, {})
        for industry_name, industry_weight in industry_map.items():
            stocks_in_industry = self._industry_stocks.get(industry_name, [])
            for stock in stocks_in_industry:
                stock_lower = stock.lower()
                if stock_lower in seen_stocks:
                    continue  # company-level already covered
                w = direction * industry_weight * event_confidence * 0.5  # dampened
                affected.append({
                    "stock": stock_lower,
                    "weight": round(w, 4),
                    "level": "industry",
                    "source": f"{topic}→{industry_name}",
                })
                seen_stocks.add(stock_lower)

        return affected

    def get_all_affected_stocks(
        self,
        events: list[dict],
    ) -> dict[str, float]:
        """Aggregate scores across multiple events for all affected stocks.

        Args:
            events: list of {source_entity, topic, direction, confidence}

        Returns:
            {stock_code: aggregated_score}
        """
        scores: dict[str, float] = {}

        for event in events:
            entity = event.get("source_entity", "")
            topic = event.get("topic", "")
            direction = event.get("direction", 0)
            confidence = event.get("confidence", 0.5)

            if not entity or direction == 0:
                continue

            affected = self.get_affected_stocks(
                entity, topic, direction, confidence
            )
            for item in affected:
                stock = item["stock"]
                scores[stock] = scores.get(stock, 0) + item["weight"]

        return scores

    def coverage_stats(self) -> dict:
        """Report coverage statistics."""
        company_stocks = set()
        for edge in self._company_edges:
            company_stocks.add(edge["dst_stock"].lower())

        industry_stocks = set()
        for industry in TOPIC_TO_INDUSTRY.values():
            for ind_name in industry:
                for stock in self._industry_stocks.get(ind_name, []):
                    industry_stocks.add(stock.lower())

        total = company_stocks | industry_stocks

        return {
            "company_level": len(company_stocks),
            "industry_level": len(industry_stocks - company_stocks),
            "total_unique": len(total),
            "industries_mapped": len(TOPIC_TO_INDUSTRY),
            "jqdata_industries": len(self._industry_stocks),
        }
