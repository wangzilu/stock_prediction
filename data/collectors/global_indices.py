"""Global stock market indices collector.

Uses Tencent Finance API (fast, reliable from overseas) for US and HK markets.
Other markets covered via RSS news analysis.
"""
import requests
import logging

logger = logging.getLogger(__name__)

# Tencent Finance API codes for global indices
INDICES = {
    "道琼斯": "us.DJI",
    "纳斯达克": "us.IXIC",
    "标普500": "us.INX",
    "恒生指数": "hkHSI",
    "恒生科技": "hkHSTECH",
    "沪深300": "sh000300",
    "上证指数": "sh000001",
    "创业板指": "sz399006",
}


class GlobalIndicesCollector:
    """Collects global stock market index data via Tencent Finance API."""

    def fetch_all(self) -> dict:
        """Fetch all global indices.

        Returns:
            Dict of {name: {price, change_pct}} for each index.
        """
        result = {}
        for name, code in INDICES.items():
            quote = self._fetch_index(code)
            if quote:
                result[name] = quote
        return result

    def _fetch_index(self, code: str) -> dict:
        """Fetch a single index from Tencent."""
        try:
            resp = requests.get(f"https://qt.gtimg.cn/q={code}", timeout=5)
            if resp.status_code != 200:
                return {}

            parts = resp.text.split("~")
            if len(parts) < 33:
                return {}

            price = parts[3]
            change_pct = parts[32]
            if not price or not change_pct:
                return {}

            return {
                "price": float(price),
                "change_pct": float(change_pct),
            }
        except Exception as e:
            logger.warning(f"Index fetch failed for {code}: {e}")
            return {}

    def format_for_report(self) -> str:
        """Fetch all indices and format as text for LLM context."""
        data = self.fetch_all()
        if not data:
            return "全球指数数据暂无"

        lines = []
        for name, quote in data.items():
            arrow = "↑" if quote["change_pct"] > 0 else "↓" if quote["change_pct"] < 0 else "→"
            lines.append(f"  {name}: {quote['price']:,.2f} ({quote['change_pct']:+.2f}%) {arrow}")
        return "\n".join(lines)
