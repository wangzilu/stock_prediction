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
#
# 2026-06-07 (#174 step 1): topic keys are matched case-insensitively now —
# the rule extractor emits lowercase ``ai_server`` / ``apple_chain`` / etc.
# but the original dict used mixed-case ``AI_server`` / ``Apple_chain`` keys,
# so 4 of the 7 most common topics silently returned an empty industry map
# (verified by tracing 2026-06-04 events: ai_server, apple_chain, ev_battery,
# tesla_robot all NO MATCH while only semiconductor, commodity,
# strategic_material went through). That was the dominant cause of the
# <0.01 % chain-factor density flagged in docs/phase_b7_verdict_20260607.md.
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


# 2026-06-07 (#174 step 1): aliases for upstream topic strings that
# don't map 1:1 to TOPIC_TO_INDUSTRY keys. The lookup falls through
# this table BEFORE giving up. Keep lowercase keys here — the matcher
# normalises both sides to lower().
_TOPIC_ALIAS = {
    "ev_battery": "EV",       # rule extractor's split-out battery topic → EV
    "tesla_robot": "robot",   # specific entity → generic robot topic
    "apple": "Apple_chain",
    "ai": "AI_server",
}


def _resolve_topic(topic: str) -> dict:
    """Look up TOPIC_TO_INDUSTRY case-insensitively, with alias fallback.

    Returns ``{}`` when no mapping is found so callers can keep the
    ``industry_map.items()`` zero-row iteration without a None check.
    """
    if not topic:
        return {}
    # 1) exact (legacy) match
    direct = TOPIC_TO_INDUSTRY.get(topic)
    if direct:
        return direct
    # 2) case-insensitive match against canonical keys
    tl = topic.lower()
    for canonical, mapping in TOPIC_TO_INDUSTRY.items():
        if canonical.lower() == tl:
            return mapping
    # 3) alias table
    alias = _TOPIC_ALIAS.get(tl)
    if alias:
        return TOPIC_TO_INDUSTRY.get(alias, {})
    return {}


class SupplyChainMapper:
    """Maps global events to A-share stocks via company + industry level.

    2026-06-06 (cx review): the mapper used to read the YAML directly
    without respecting the SC-A3 tier filter, so a production caller
    that asked for A/B-only edges via ``load_edges("B")`` would still
    leak C/D edges into ``get_all_affected_stocks`` via the mapper.
    The constructor now accepts ``min_tier`` and ``explicit_tiers`` so
    the mapper honours the same contract as the top-level loader.
    Default ``min_tier="B"`` matches the production contract.
    """

    def __init__(
        self,
        *,
        min_tier: str = "B",
        explicit_tiers: frozenset[str] | None = None,
    ):
        self._min_tier = min_tier
        self._explicit_tiers = explicit_tiers
        self._company_edges = self._load_company_edges()
        self._industry_stocks = self._load_industry_stocks()

    def _load_company_edges(self) -> list[dict]:
        """Load tier-filtered company-level edges from YAML.

        Uses ``scripts.build_global_chain_factors.classify_edge_tier`` so
        the filter logic lives in exactly one place. Falls back to the
        unfiltered YAML when the classifier import fails — but in that
        case logs a loud warning so the contract violation is visible.
        """
        if not EDGES_PATH.exists():
            return []
        with open(EDGES_PATH) as f:
            raw = yaml.safe_load(f) or []

        try:
            from scripts.build_global_chain_factors import (
                classify_edge_tier,
            )
        except Exception as exc:  # noqa: BLE001
            # The classifier is the one source of truth. If we can't
            # import it, the safest behaviour is to drop EVERY edge so
            # we never silently leak unclassified rows into production.
            import logging
            logging.getLogger(__name__).error(
                "supply_chain_mapper: cannot import classify_edge_tier "
                "(%s); refusing to load edges to avoid a tier-filter "
                "bypass. Production overlay will see 0 edges this run.",
                exc,
            )
            return []

        if self._explicit_tiers is not None:
            keep = frozenset(self._explicit_tiers)
        else:
            TIER_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}
            threshold = TIER_ORDER.get(self._min_tier.upper(), 1)
            keep = frozenset(
                t for t, idx in TIER_ORDER.items() if idx <= threshold
            )

        kept = [e for e in raw if classify_edge_tier(e.get("source", "")) in keep]
        if len(kept) != len(raw):
            import logging
            logging.getLogger(__name__).info(
                "supply_chain_mapper: kept %d/%d edges (min_tier=%s, "
                "explicit_tiers=%s)",
                len(kept), len(raw), self._min_tier, self._explicit_tiers,
            )
        return kept

    def _load_industry_stocks(self) -> dict[str, list[str]]:
        """Load industry → stock mapping. Returns {industry_name: [qlib_codes]}.

        Priority: baostock (5500+ stocks) > JQData (698 stocks).
        Maps CSRC industry codes to TOPIC_TO_INDUSTRY keys via a mapping table.
        """
        result = {}

        # Map from CSRC (证监会) industry codes to our topic industry names
        CSRC_TO_SW = {
            "C39计算机、通信和其他电子设备制造业": "电子I",
            "I65软件和信息技术服务业": "计算机I",
            "I64互联网和相关服务": "计算机I",
            "C38电气机械和器材制造业": "电力设备I",
            "C36汽车制造业": "汽车I",
            "C35专用设备制造业": "机械设备I",
            "C34通用设备制造业": "机械设备I",
            "C32有色金属冶炼和压延加工业": "有色金属I",
            "C31黑色金属冶炼和压延加工业": "钢铁I",
            "C26化学原料和化学制品制造业": "基础化工I",
            "C25石油加工、炼焦和核燃料加工业": "石油石化I",
            "B07石油和天然气开采业": "石油石化I",
            "B09有色金属矿采选业": "有色金属I",
            "B06煤炭开采和洗选业": "煤炭I",
            "C27医药制造业": "医药生物I",
            "C37铁路、船舶、航空航天和其他运输设备制造业": "国防军工I",
            "C40仪器仪表制造业": "机械设备I",
            "C41其他制造业": "机械设备I",
            "C33金属制品业": "钢铁I",
            "I63电信、广播电视和卫星传输服务": "通信I",
            "R87文化艺术业": "传媒I",
            "C13农副食品加工业": "食品饮料I",
            "C14食品制造业": "食品饮料I",
            "C15酒、饮料和精制茶制造业": "食品饮料I",
            "C29橡胶和塑料制品业": "基础化工I",
            "C30非金属矿物制品业": "建筑材料I",
            "K70房地产业": "房地产I",
            "D44电力、热力生产和供应业": "公用事业I",
            "C17纺织业": "纺织服饰I",
            "C18纺织服装、服饰业": "纺织服饰I",
            "C28化学纤维制造业": "基础化工I",
        }

        # Try baostock classification first (5500+ stocks)
        bs_path = DATA_DIR / "baostock_industry.parquet"
        if bs_path.exists():
            df = pd.read_parquet(bs_path)
            if "qlib_code" in df.columns and "industry" in df.columns:
                for _, row in df.iterrows():
                    csrc_industry = str(row.get("industry", ""))
                    qlib = str(row.get("qlib_code", ""))
                    if not csrc_industry or not qlib:
                        continue
                    # Map CSRC to our industry names
                    sw_name = CSRC_TO_SW.get(csrc_industry)
                    if sw_name:
                        result.setdefault(sw_name, []).append(qlib)

        # Fallback: JQData SW classification
        if not result:
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
        # 2026-06-07 (#174 step 1): use _resolve_topic so case-mismatched
        # topics (ai_server vs AI_server, etc) still hit the industry map.
        industry_map = _resolve_topic(topic)
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
