"""Sector/concept heat tracker and dragon-tiger seat identification.

Computes:
1. sector_heat: limit-up count per concept/industry in last N days
2. hot_money_signal: known hot-money (游资) seat detection from dragon-tiger list
"""
import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

# Known hot-money (游资) trading seats
# Source: public dragon-tiger list data + market knowledge
KNOWN_HOT_MONEY_SEATS = {
    # 赵老哥系
    "银河证券绍兴": "赵老哥",
    "浙商证券绍兴": "赵老哥",
    # 养家系
    "华鑫证券上海分公司": "养家",
    # 章盟主
    "国泰君安上海江苏路": "章盟主",
    # 欢乐海岸
    "东方财富拉萨团结路": "欢乐海岸",
    "东方财富拉萨东环路": "欢乐海岸",
    # 金田路
    "中信证券深圳金田路": "金田路",
    # 古北路
    "中信证券上海古北路": "古北路",
    # 溧阳路
    "华泰证券上海武定路": "溧阳路",
    # 小鳄鱼
    "东方财富拉萨": "东财拉萨系",
    # 方新侠
    "光大证券佛山绿景路": "方新侠",
    # 作手新一
    "华泰证券深圳益田路": "作手新一",
}


class SectorHeatTracker:
    """Track sector/concept heat from limit-up pool data."""

    def __init__(self, limit_up_collector=None):
        self.limit_up = limit_up_collector

    def compute_sector_heat(self) -> dict:
        """Compute heat score per sector from today's limit-up pool.

        Returns:
            Dict mapping sector/concept → heat score (0-1)
        """
        if not self.limit_up:
            return {}

        pool = self.limit_up.fetch_today_pool()
        if pool.empty:
            return {}

        # Count limit-ups per concept/industry
        # Use stock code prefix as simple sector proxy
        from models.portfolio_policy import sector_from_code

        sector_counts = {}
        for _, row in pool.iterrows():
            code = row.get("qlib_code", "")
            if not code:
                continue
            sector = sector_from_code(code)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Normalize to 0-1 (10 limit-ups in one sector = max heat)
        total = sum(sector_counts.values())
        heat = {}
        for sector, count in sector_counts.items():
            heat[sector] = min(count / 10.0, 1.0)

        logger.info(f"Sector heat: {len(heat)} sectors, top: {sorted(heat.items(), key=lambda x: -x[1])[:3]}")
        return heat

    def get_stock_sector_heat(self, code: str) -> float:
        """Get heat score for a stock's sector."""
        from models.portfolio_policy import sector_from_code
        sector = sector_from_code(code)
        heat = self.compute_sector_heat()
        return heat.get(sector, 0.0)


class DragonTigerAnalyzer:
    """Analyze dragon-tiger list for hot-money seat signals."""

    def identify_hot_money(self, lhb_data: pd.DataFrame) -> list:
        """Identify known hot-money seats from dragon-tiger list data.

        Args:
            lhb_data: DataFrame from ak.stock_lhb_detail_em()

        Returns:
            List of dicts: {code, name, seat, hot_money_name, side(buy/sell), amount}
        """
        if lhb_data is None or lhb_data.empty:
            return []

        signals = []
        for _, row in lhb_data.iterrows():
            seat = str(row.get("营业部名称", row.get("买入营业部名称", "")))
            if not seat:
                continue

            # Check against known hot-money seats
            for seat_pattern, hm_name in KNOWN_HOT_MONEY_SEATS.items():
                if seat_pattern in seat:
                    code = str(row.get("代码", row.get("股票代码", ""))).zfill(6)
                    signals.append({
                        "code": code,
                        "name": str(row.get("名称", row.get("股票简称", ""))),
                        "seat": seat,
                        "hot_money": hm_name,
                        "buy_amount": float(row.get("买入额", row.get("买入金额", 0)) or 0),
                        "sell_amount": float(row.get("卖出额", row.get("卖出金额", 0)) or 0),
                    })

        if signals:
            logger.info(f"Hot money signals: {len(signals)} from {len(set(s['hot_money'] for s in signals))} traders")

        return signals

    def fetch_and_analyze(self, date: str = None) -> list:
        """Fetch dragon-tiger list and identify hot-money signals."""
        try:
            import akshare as ak
            if not date:
                date = datetime.now().strftime("%Y%m%d")
            lhb = ak.stock_lhb_detail_em(start_date=date, end_date=date)
            return self.identify_hot_money(lhb)
        except Exception as e:
            logger.warning(f"Dragon-tiger fetch failed: {e}")
            return []
