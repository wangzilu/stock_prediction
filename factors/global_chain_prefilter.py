"""Global supply chain news pre-filter — rule-based recall layer.

Filters 1400+ daily global news items down to 100-200 high-value
candidates for LLM extraction. Two-stage funnel:

  1400+ raw news → rule prefilter (100-200) → LLM extract (30-80)

Matching rules:
  - Global core entities (Nvidia, Apple, Tesla, TSMC, etc.)
  - Industry keywords (AI server, HBM, optical module, etc.)
  - Event keywords (order, guidance, capex, shortage, etc.)
  - Source quality scoring (Reuters > blog)
  - Title dedup (keep highest source quality)

Usage:
    from factors.global_chain_prefilter import prefilter_news
    candidates = prefilter_news(raw_news_items)
"""
import hashlib
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# === Core entity patterns (must match at least one) ===
ENTITY_PATTERNS = [
    r"\bNvidia\b", r"\bAMD\b", r"\bIntel\b", r"\bTSMC\b", r"\bASML\b",
    r"\bApple\b", r"\bMicrosoft\b", r"\bGoogle\b|\bAlphabet\b", r"\bMeta\b",
    r"\bAmazon\b", r"\bTesla\b", r"\bOptimus\b",
    r"\bSK\s*Hynix\b", r"\bSamsung\b", r"\bMicron\b",
    r"\bCATL\b", r"\bBYD\b", r"\bQualcomm\b", r"\bMediaTek\b",
    r"\bOpenAI\b", r"\bAnthropic\b",
    r"\bPfizer\b", r"\bRoche\b", r"\bNovartis\b", r"\bAstraZeneca\b",
    r"\bModerna\b", r"\bBioNTech\b",
    r"\bLockheed\b", r"\bRaytheon\b", r"\bBAE\b",
    r"\bLG\b", r"\bWhirlpool\b",
]

# === Industry keywords ===
INDUSTRY_KEYWORDS = [
    r"\bAI\s+server\b", r"\bHBM\b", r"\boptical\s+module\b",
    r"\bGPU\b", r"\bsemiconductor\s+equipment\b", r"\bfoundry\b",
    r"\bEV\s+battery\b", r"\blithium\b", r"\bsolar\b", r"\brare\s+earth\b",
    r"\bexport\s+control\b", r"\bchip\s+ban\b", r"\bsanction\b",
    r"\bdata\s*center\b", r"\b5G\b", r"\bautonomous\b",
    r"\brobot\b|\bhumanoid\b", r"\breducer\b|\bactuator\b",
    r"\bPolysilicon\b", r"\bcopper\b", r"\bcobalt\b",
    r"\bCRO\b|\bCDMO\b", r"\bvaccine\b", r"\bmRNA\b",
    r"\bdefense\b|\bmilitary\b|\bfighter\b",
]

# === Event keywords (signals something happened) ===
EVENT_KEYWORDS = [
    r"\border[s]?\b", r"\bcontract\b", r"\bguidance\b", r"\bcapex\b",
    r"\bshortage\b", r"\binventory\b", r"\bprice\s+cut\b", r"\btariff\b",
    r"\brecall\b", r"\bban\b", r"\bsanction\b", r"\bshipment\b",
    r"\bdemand\b", r"\bsupply\b", r"\bproduction\b", r"\bexpansion\b",
    r"\bearnings\b", r"\brevenue\b", r"\bforecast\b", r"\bdowngrade\b",
    r"\bupgrade\b", r"\bpartnership\b", r"\bacquisition\b",
    r"\bIPO\b", r"\blayoff\b", r"\brestructur\b",
]

# === Source quality tiers ===
PREMIUM_DOMAINS = {
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
    "cnbc.com", "nytimes.com", "bbc.com",
    "prnewswire.com", "businesswire.com", "globenewswire.com",
    "sec.gov", "investor.com",
}
GOOD_DOMAINS = {
    "techcrunch.com", "theverge.com", "arstechnica.com",
    "semianalysis.com", "tomshardware.com", "anandtech.com",
    "electrek.co", "insideevs.com", "cleantechnica.com",
    "fiercepharma.com", "statnews.com",
}


def _score_source(domain: str) -> float:
    """Score source quality: 1.0=premium, 0.7=good, 0.4=other."""
    if not domain:
        return 0.3
    domain = domain.lower()
    for d in PREMIUM_DOMAINS:
        if d in domain:
            return 1.0
    for d in GOOD_DOMAINS:
        if d in domain:
            return 0.7
    return 0.4


def _match_count(text: str, patterns: list[str]) -> int:
    """Count how many patterns match in text."""
    n = 0
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            n += 1
    return n


def score_news_item(item: dict) -> float:
    """Score a news item for supply chain relevance.

    Returns 0-10 score. Higher = more likely supply chain relevant.
    """
    title = item.get("title", "")
    if not title:
        return 0.0

    # Entity match (0-3 points)
    entity_score = min(3, _match_count(title, ENTITY_PATTERNS))

    # Industry keyword match (0-3 points)
    industry_score = min(3, _match_count(title, INDUSTRY_KEYWORDS))

    # Event keyword match (0-2 points)
    event_score = min(2, _match_count(title, EVENT_KEYWORDS))

    # Source quality (0-2 points)
    domain = item.get("domain", "")
    source_score = _score_source(domain) * 2

    total = entity_score + industry_score + event_score + source_score
    return round(total, 2)


def prefilter_news(
    news_items: list[dict],
    max_candidates: int = 200,
    min_score: float = 2.0,
) -> list[dict]:
    """Filter raw news to supply chain candidates.

    Args:
        news_items: raw news from collector
        max_candidates: max items to return
        min_score: minimum relevance score

    Returns:
        Filtered and scored list, sorted by score descending.
    """
    # Score all items
    scored = []
    for item in news_items:
        score = score_news_item(item)
        if score >= min_score:
            item["chain_relevance_score"] = score
            scored.append(item)

    # Dedup by title similarity (keep highest source quality)
    seen_titles = {}
    deduped = []
    for item in sorted(scored, key=lambda x: -x["chain_relevance_score"]):
        title_key = hashlib.md5(item["title"][:50].lower().encode()).hexdigest()[:12]
        if title_key not in seen_titles:
            seen_titles[title_key] = True
            deduped.append(item)

    # Take top N
    result = deduped[:max_candidates]

    logger.info(
        f"Prefilter: {len(news_items)} raw → {len(scored)} scored → "
        f"{len(deduped)} deduped → {len(result)} candidates "
        f"(min_score={min_score})"
    )
    return result
