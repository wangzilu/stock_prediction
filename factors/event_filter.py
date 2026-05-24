"""Rule-based pre-filter for news/announcements before sending to LLM.

Prioritizes high-value events by keyword matching and source quality,
reducing LLM cost and improving signal-to-noise ratio.

Usage:
    from factors.event_filter import filter_candidates, select_for_llm

    items = [{"stock_code": "600519", "title": "重大合同公告", "source": "exchange_announcement"}, ...]
    selected = select_for_llm(filter_candidates(items))
"""

HIGH_PRIORITY_KEYWORDS = [
    # 重大事件
    "重大合同", "中标", "订单", "重大项目",
    # 资本运作
    "回购", "增持", "减持", "股权激励", "定增", "配股", "可转债", "分红", "送股",
    # 业绩
    "业绩预告", "业绩快报", "年报", "亏损", "扭亏", "盈利预测",
    # 风险事件
    "处罚", "立案", "诉讼", "仲裁", "违规", "警示", "问询函",
    # 信用/债务
    "解禁", "质押", "债务违约", "信用评级",
    # 重组并购
    "重组", "并购", "收购", "资产注入", "借壳",
    # 政策/补贴
    "政府补贴", "税收优惠", "产业政策",
    # 经营
    "产能扩张", "停产", "召回", "退市", "ST", "摘帽",
    # 市场
    "市占率", "涨价", "降价",
]

SOURCE_PRIORITY = {
    "exchange_announcement": 1,
    "cninfo": 1,
    "official_disclosure": 1,
    "sse": 1,
    "szse": 1,
    "cailian": 2,
    "cls": 2,
    "eastmoney_announcement": 2,
    "eastmoney_news": 3,
    "sina_finance": 3,
    "tencent_finance": 3,
    "announcement": 2,
    "news": 3,
    "other_news": 4,
    "social": 5,
    "guba": 5,
    "xueqiu": 5,
}

# Items with score >= this are "must send to LLM"
MUST_SEND_THRESHOLD = 3


def _keyword_score(title: str) -> int:
    """Count how many high-priority keywords appear in the title."""
    if not title:
        return 0
    return sum(1 for kw in HIGH_PRIORITY_KEYWORDS if kw in title)


def _source_score(source: str) -> int:
    """Higher score for better sources (inverted priority: 1=best → 5=worst)."""
    priority = SOURCE_PRIORITY.get(source, SOURCE_PRIORITY.get("other_news", 4))
    return max(0, 6 - priority)  # priority 1 → score 5, priority 5 → score 1


def filter_candidates(news_items: list[dict]) -> list[dict]:
    """Score and sort news items by priority.

    Args:
        news_items: list of dicts with at least 'title' and 'source' fields.
            Optional: 'stock_code', 'publish_time', 'url'.

    Returns:
        Same list with 'priority_score' and 'must_send' fields added,
        sorted by priority_score descending.
    """
    for item in news_items:
        kw = _keyword_score(item.get("title", ""))
        src = _source_score(item.get("source", "other_news"))
        score = kw * 2 + src  # keyword hits weighted 2x
        item["priority_score"] = score
        item["must_send"] = score >= MUST_SEND_THRESHOLD

    return sorted(news_items, key=lambda x: -x["priority_score"])


def select_for_llm(
    news_items: list[dict],
    max_per_stock: int = 3,
    max_total: int = 500,
) -> list[dict]:
    """Select items to send to LLM, respecting per-stock and total limits.

    Args:
        news_items: output of filter_candidates() (sorted by priority_score).
        max_per_stock: max items per stock code.
        max_total: max total items to return.

    Returns:
        Selected items, sorted by priority_score descending.
    """
    selected = []
    stock_counts: dict[str, int] = {}
    seen_titles: set[str] = set()

    for item in news_items:
        if len(selected) >= max_total:
            break

        code = item.get("stock_code", "unknown")
        title = item.get("title", "")

        # Dedup by title prefix (first 20 chars)
        title_key = f"{code}_{title[:20]}"
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        # Per-stock limit (must_send items bypass if under 2x limit)
        count = stock_counts.get(code, 0)
        if count >= max_per_stock and not item.get("must_send"):
            continue
        if count >= max_per_stock * 2:
            continue

        selected.append(item)
        stock_counts[code] = count + 1

    return selected
