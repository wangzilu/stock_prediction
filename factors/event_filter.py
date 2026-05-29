"""Rule-based pre-filter for news/announcements before sending to LLM.

Provides two layers:

1. classify_l0() — L0 routing for the tiered pipeline. Each item is routed to
   one of: "direct" (rule emits a structured event, no LLM call),
   "drop" (irrelevant/noise, discarded), or "l1" (default cheap LLM extraction).
   Items hitting L2 hint patterns are marked but still flow through L1 first
   (L2 escalation lives in the extractor, not here).

2. filter_candidates() + select_for_llm() — the original priority ranker
   used to cap LLM volume at ~500 items/day, retained for the L1 path.

Cross-day dedup: classify_l0() optionally consumes a hash cache so that the
same announcement title isn't re-extracted on subsequent days.

Usage:
    from factors.event_filter import classify_l0, filter_candidates, select_for_llm

    routed = classify_l0(items, cache_path=DATA_DIR / "llm_event_cache" / "seen.jsonl")
    direct_events = routed["direct"]    # emit straight to EventStore
    l1_items      = routed["l1"]        # send to LLMEventExtractorV2
    # routed["drop"] / routed["dup"] are logged but discarded
"""
import hashlib
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

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


# -----------------------------------------------------------------------------
# L0 — direct classification, drop, L2 hint
# -----------------------------------------------------------------------------
# Each direct rule is (compiled_regex, builder). The builder receives the title
# and returns (event_type, direction, confidence, summary_suffix).
# Rules are evaluated in order — first match wins. Keep them HIGH confidence:
# wrong classification is silently shipped to EventStore, so prefer false
# negatives (fall through to L1) over false positives.
DIRECT_CLASSIFY_RULES: list[tuple] = [
    # ST摘帽 / 撤销退市风险警示 — unambiguously routine (positive direction by convention)
    (re.compile(r"(撤销退市风险警示|撤销.*风险警示|摘帽|撤销其他风险警示)"),
     ("routine_announcement", 1, 0.92, "撤销风险警示/摘帽")),

    # 询问函/问询函回复 — purely administrative, no impact
    (re.compile(r"(关于.*问询函的回复|问询函回复|关于.*年报问询函)"),
     ("routine_announcement", 0, 0.90, "问询函回复")),

    # 股东大会通知/召开/决议 — calendar item, no surprise
    (re.compile(r"(召开.{0,8}股东大会.{0,12}(通知|的公告)|股东大会决议公告)"),
     ("routine_announcement", 0, 0.88, "股东大会通知/决议")),

    # 独立董事/独董声明/任职 — governance routine
    (re.compile(r"(独立董事(关于|的)|独董(声明|任职)|独立董事公开声明)"),
     ("routine_announcement", 0, 0.88, "独立董事声明")),

    # 日常关联交易 — routine, by definition non-material
    (re.compile(r"(日常关联交易|常规关联交易).{0,20}(公告|说明)?$"),
     ("routine_announcement", 0, 0.88, "日常关联交易")),

    # 更正/补充公告 — corrections, by themselves no new info
    (re.compile(r"^.{0,40}(更正公告|补充公告|关于.*更正的公告)$"),
     ("routine_announcement", 0, 0.85, "更正/补充公告")),

    # 已实施回购进展 + 累计 — partially-completed buyback progress reports
    (re.compile(r"回购.*(进展|结果)公告|关于.*累计回购.*股份"),
     ("share_buyback", 1, 0.88, "回购进展")),

    # 高管股份解禁到期 — calendar event
    (re.compile(r"(限售股|限制性股票).*(解除限售|上市流通).{0,20}(公告|提示)"),
     ("share_unlock", 0, 0.85, "限售股解禁")),
]

# Drop patterns: items definitely not worth a slot. Conservative — only obvious noise.
DROP_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Empty / whitespace-only or absurdly short titles (will not yield extractable event)
    (re.compile(r"^.{0,7}$"), "title_too_short"),
    # Pure index/price ticker chatter from social sources with no event content
    (re.compile(r"^(今日|盘中|尾盘|早盘)[价涨跌幅\d\s\.\%]+$"), "pure_price_chatter"),
    # Tagged share-action signature lines that aren't really announcements
    (re.compile(r"^\s*$"), "empty"),
]

# L2 hint: items likely needing reasoning model (multi-entity / supply chain / long impact chain).
# These flow through L1 first; if L1 confidence is low or event_type ambiguous, the L2 path
# (to be implemented separately) can re-process. We just mark with `_l2_hint: True` here.
L2_HINT_PATTERNS: list[re.Pattern] = [
    re.compile(r"(产业链|供应链|上下游|卡脖子|关键材料|关键技术)"),
    re.compile(r"(出口管制|反制裁|进口替代|国产替代|断供)"),
    re.compile(r"(\d{6}[、,，]\s*){2,}"),  # 3+ comma-separated 6-digit codes in title
]


def _content_hash(item: dict) -> str:
    """Stable hash for cross-day dedup.

    Hashes on (stock_code, title[:60], source, publish_date_only). publish_date
    is the first 10 chars of publish_time, so two news items with identical
    title published on the same day still collide and dedup correctly.
    """
    code = (item.get("stock_code") or item.get("qlib_code", ""))[-6:]
    title = (item.get("title") or "")[:60].strip()
    source = item.get("source") or "unknown"
    pub_date = (item.get("publish_time") or "")[:10]
    key = f"{code}|{title}|{source}|{pub_date}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _load_seen_cache(cache_path: Path | None) -> set[str]:
    """Read existing hash cache into a set. Missing/corrupt cache returns empty set."""
    if cache_path is None or not cache_path.exists():
        return set()
    seen: set[str] = set()
    try:
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    h = obj.get("hash")
                    if h:
                        seen.add(h)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning("Failed to load L0 dedup cache %s: %s", cache_path, e)
    return seen


def _append_seen_cache(cache_path: Path, hashes: list[tuple[str, str]]) -> None:
    """Append new (hash, route) entries to the cache file."""
    if not hashes:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        for h, route in hashes:
            f.write(json.dumps({"hash": h, "route": route}, ensure_ascii=False) + "\n")


def _make_direct_event(item: dict, event_type: str, direction: int,
                       confidence: float, summary_suffix: str,
                       extract_date: str) -> dict:
    """Build the same dict shape that LLMEventExtractorV2 writes per event."""
    code = item.get("stock_code") or item.get("qlib_code", "")[-6:]
    title = item.get("title", "")
    source = item.get("source", "unknown")
    source_info = SOURCE_TIERS_LITE.get(source, {"tier": "media", "quality": 0.5})
    return {
        "stock_code": code,
        "stock_name": item.get("stock_name", ""),
        "qlib_code": item.get("qlib_code", ""),
        "publish_time": item.get("publish_time", ""),
        "title": title,
        "source": source,
        "source_tier": source_info["tier"],
        "source_quality": source_info["quality"],
        "extract_date": extract_date,
        "extractor_version": "l0_rule",
        "event_type": event_type,
        "direction": direction,
        "is_official_disclosure": source_info["tier"] == "official",
        "is_new_information": True,
        "is_repeated_news": False,
        "is_price_sensitive": event_type != "routine_announcement",
        "magnitude_description": "",
        "magnitude_value_wan": 0.0,
        "confidence": confidence,
        "summary": (summary_suffix + " — " + title)[:100],
    }


# Lightweight source tier map for direct events (mirrors LLMEventExtractorV2's table
# without forcing a circular import).
SOURCE_TIERS_LITE = {
    "交易所公告": {"tier": "official", "quality": 1.0},
    "巨潮资讯": {"tier": "official", "quality": 0.95},
    "上交所": {"tier": "official", "quality": 1.0},
    "深交所": {"tier": "official", "quality": 1.0},
    "exchange_announcement": {"tier": "official", "quality": 1.0},
    "cninfo": {"tier": "official", "quality": 0.95},
    "证券时报": {"tier": "media", "quality": 0.8},
    "财联社": {"tier": "media", "quality": 0.75},
    "eastmoney": {"tier": "media", "quality": 0.5},
    "雪球": {"tier": "social", "quality": 0.3},
    "股吧": {"tier": "social", "quality": 0.2},
}


def classify_l0(news_items: list[dict], *, extract_date: str,
                cache_path: Path | None = None) -> dict:
    """Route items into direct/drop/dup/l1.

    Args:
        news_items: list of news dicts (title, source, stock_code, publish_time, ...).
        extract_date: the date these items are being processed for (YYYY-MM-DD).
            Used as extract_date in any directly-emitted event.
        cache_path: optional path to a JSONL hash cache. If provided, items whose
            content hash already appears in the cache are routed to "dup". After
            classification, all NON-dropped item hashes are appended back.

    Returns:
        dict with keys:
            direct: list of structured event dicts (ready to write to extractor file)
            l1:     list of news items to send to L1 LLM extractor
            drop:   list of (item, drop_reason) tuples (for stats)
            dup:    list of items skipped via dedup cache
            stats:  counters per route + per direct-rule hit
    """
    seen_cache = _load_seen_cache(cache_path)

    direct, l1, drop, dup = [], [], [], []
    rule_hits: dict[str, int] = {}
    new_hashes: list[tuple[str, str]] = []

    for item in news_items:
        title = (item.get("title") or "").strip()

        # 1. dedup (must come first to avoid wasted work on already-handled items)
        h = _content_hash(item)
        if h in seen_cache:
            dup.append(item)
            continue

        # 2. drop
        dropped = False
        for pat, reason in DROP_PATTERNS:
            if pat.search(title):
                drop.append((item, reason))
                rule_hits[f"drop:{reason}"] = rule_hits.get(f"drop:{reason}", 0) + 1
                dropped = True
                break
        if dropped:
            # Don't add to seen cache — dropped items aren't "handled", we want a
            # fresh shot if the same title shows up with a more informative form.
            continue

        # 3. direct classify
        matched = False
        for pat, payload in DIRECT_CLASSIFY_RULES:
            if pat.search(title):
                event_type, direction, conf, summary_suffix = payload
                event = _make_direct_event(item, event_type, direction, conf,
                                           summary_suffix, extract_date)
                direct.append(event)
                rule_hits[f"direct:{event_type}"] = rule_hits.get(f"direct:{event_type}", 0) + 1
                new_hashes.append((h, "direct"))
                matched = True
                break
        if matched:
            continue

        # 4. fall through to L1 — also flag L2 hints for the extractor to use later
        for pat in L2_HINT_PATTERNS:
            if pat.search(title):
                item["_l2_hint"] = True
                rule_hits["l2_hint"] = rule_hits.get("l2_hint", 0) + 1
                break
        l1.append(item)
        new_hashes.append((h, "l1"))

    if cache_path is not None and new_hashes:
        _append_seen_cache(cache_path, new_hashes)

    stats = {
        "total_in": len(news_items),
        "direct": len(direct),
        "l1": len(l1),
        "drop": len(drop),
        "dup": len(dup),
        "rule_hits": rule_hits,
    }
    return {"direct": direct, "l1": l1, "drop": drop, "dup": dup, "stats": stats}


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
