"""Global Supply Chain Event Extractor — Phase 4U Day 3.

Rule-based MVP extractor for global industry news.
Processes news titles to extract structured supply-chain events
(event_type, direction, confidence, source_entity, topic).

No LLM calls — pure keyword matching. LLM upgrade path available later.

Usage:
    from factors.global_supply_chain_extractor import extract_from_title, batch_extract

    events = batch_extract([
        {"title": "Nvidia Blackwell demand remains strong", "topic": "ai_server", "date": "2026-05-25"},
    ])
"""
import hashlib
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword rules: (pattern, event_type, direction, base_confidence)
# Patterns are matched case-insensitively against news titles.
# ---------------------------------------------------------------------------
_KEYWORD_RULES: list[tuple[str, str, int, float]] = [
    # Positive demand / growth signals
    (r"\b(?:order[s]?\s+(?:increased?|surge[ds]?|jump|soar|rose|grew|doubled|tripled))", "order_increase", +1, 0.8),
    (r"\b(?:strong|robust|record)\s+(?:demand|orders?|sales|revenue|bookings)", "strong_demand", +1, 0.7),
    (r"\b(?:demand)\s+(?:remains?\s+strong|surge[ds]?|soar[s]?|boom)", "strong_demand", +1, 0.7),
    (r"\braises?\s+(?:guidance|outlook|forecast|capex)", "guidance_raise", +1, 0.8),
    (r"\b(?:beats?|exceeded?|surpass|top[ps]?)\s+(?:expectations?|estimates?|forecast)", "earnings_beat", +1, 0.8),
    (r"\bcapex\s+(?:increase[ds]?|raise[ds]?|guidance|hike|boost)", "capex_increase", +1, 0.7),
    (r"\b(?:increase[ds]?|raise[ds]?|hike[ds]?|boost[s]?)\s+capex", "capex_increase", +1, 0.7),
    (r"\bexpansion|expand(?:ing|s)?\s+(?:capacity|production|fab)", "capacity_expansion", +1, 0.7),
    (r"\bnew\s+(?:factory|fab|plant|facility)", "capacity_expansion", +1, 0.7),
    (r"\b(?:revenue|sales|profit|earnings)\s+(?:grew|growth|surge[ds]?|jump|rose|increase)", "revenue_growth", +1, 0.7),
    (r"\brecord\s+(?:revenue|sales|profit|earnings|quarter)", "earnings_beat", +1, 0.7),
    (r"\bmarket\s+share\s+(?:gain|grew|increase|expand)", "market_share_gain", +1, 0.6),
    (r"\bbreakthrough", "tech_breakthrough", +1, 0.6),
    (r"\bpartnership|collaborat|joint\s+venture|strategic\s+(?:alliance|cooperation)", "strategic_cooperation", +1, 0.5),
    (r"\b(?:win[s]?|won|secure[ds]?|land[s]?)\s+(?:contract|deal|order)", "order_win", +1, 0.8),

    # Negative demand / weakness signals
    (r"\bcut[s]?\s+(?:order|production|output|forecast|guidance|outlook|capex|jobs?|workforce)", "order_cut", -1, 0.8),
    (r"\b(?:order|production)\s+cut", "order_cut", -1, 0.8),
    (r"\bshortage|supply\s+(?:crunch|squeeze|constraint|disruption|bottleneck)", "supply_shortage", -1, 0.7),
    (r"\bdelay(?:ed|s|ing)?(?:\s+(?:shipment|delivery|launch|production))?", "delay", -1, 0.6),
    (r"\bweak(?:er|ening|ness)?\s+(?:demand|sales|outlook|guidance|orders?)", "weak_demand", -1, 0.7),
    (r"\b(?:miss(?:ed|es)?|below|under)\s+(?:expectations?|estimates?|forecast)", "earnings_miss", -1, 0.8),
    (r"\blower[s]?\s+(?:guidance|outlook|forecast)", "guidance_cut", -1, 0.8),
    (r"\b(?:revenue|sales|profit|earnings)\s+(?:decline[ds]?|drop|fell|fall|plunge|slump)", "revenue_decline", -1, 0.7),
    (r"\b(?:layoff|lay\s+off|job\s+cuts?|workforce\s+reduction|restructur)", "restructuring", -1, 0.6),
    (r"\bmarket\s+share\s+(?:loss|lost|decline|shrink)", "market_share_loss", -1, 0.6),

    # Policy / regulation signals
    (r"\bban(?:ned|s)?\b", "policy_ban", -1, 0.8),
    (r"\btariff|sanction|export\s+(?:control|restriction|ban|curb)", "policy_restriction", -1, 0.8),
    (r"\b(?:restrict|block|blacklist|entity\s+list)", "policy_restriction", -1, 0.7),
    (r"\bsubsid(?:y|ies|ize)", "subsidy", +1, 0.6),
    (r"\bregulatory\s+approval", "regulatory_approval", +1, 0.6),

    # Commodity / price signals
    (r"\b(?:price[s]?)\s+(?:surge[ds]?|soar|spike|jump|rose|rise|increase|rally|rebound)", "price_increase", +1, 0.7),
    (r"\b(?:price[s]?)\s+(?:fall|fell|drop|plunge|decline|crash|slump|tumble|collapse|low)", "price_decline", -1, 0.7),
    (r"\b(?:multi-year|record|all-time)\s+(?:low|high)", "price_extreme", 0, 0.6),  # direction set by context
]

# Entity detection patterns: (regex, canonical_entity_name)
_ENTITY_PATTERNS: list[tuple[str, str]] = [
    (r"\bNvidia\b", "Nvidia"),
    (r"\bAMD\b", "AMD"),
    (r"\bIntel\b", "Intel"),
    (r"\bTSMC\b", "TSMC"),
    (r"\bASML\b", "ASML"),
    (r"\bApple\b", "Apple"),
    (r"\bMicrosoft\b", "Microsoft"),
    (r"\bGoogle\b|Alphabet", "Google"),
    (r"\bMeta\b", "Meta"),
    (r"\bAmazon\b", "Amazon"),
    (r"\bTesla\b", "Tesla"),
    (r"\bOptimus\b", "Tesla_Optimus"),
    (r"\bSK\s*Hynix\b", "SK_Hynix"),
    (r"\bSamsung\b", "Samsung"),
    (r"\bMicron\b", "Micron"),
    (r"\bCATL\b", "CATL"),
    (r"\bBYD\b", "BYD"),
    (r"\bQualcomm\b", "Qualcomm"),
    (r"\bTexas\s+Instruments\b|\bTI\s+analog\b", "Texas_Instruments"),
    (r"\bMediaTek\b", "MediaTek"),
    (r"\b(?:lithium)\b", "Lithium_Price"),
    (r"\b(?:polysilicon|poly-silicon)\b", "Polysilicon_Price"),
    (r"\b(?:copper)\s+(?:price|futures|supply)\b", "Copper_Price"),
    (r"\b(?:rare\s+earth)\b", "Rare_Earth_Price"),
    (r"\b(?:oil\s+price|crude\s+oil|brent|WTI)\b", "Oil_Price"),
    (r"\bUS\s+(?:export|sanctions?|tariff|ban)\b", "US_Export_Control"),
]


def _detect_entity(title: str) -> str:
    """Extract the most prominent source entity from a title."""
    for pattern, entity in _ENTITY_PATTERNS:
        if re.search(pattern, title, re.IGNORECASE):
            return entity
    return "unknown"


def _detect_price_extreme_direction(title: str) -> int:
    """For price_extreme events, determine direction from context."""
    low = bool(re.search(r"\blow\b", title, re.IGNORECASE))
    high = bool(re.search(r"\bhigh\b", title, re.IGNORECASE))
    if low and not high:
        return -1
    if high and not low:
        return +1
    return 0


def extract_from_title(title: str, topic: str = "") -> Optional[dict]:
    """Extract a structured event from a news title using keyword rules.

    Args:
        title: news headline (English)
        topic: supply-chain topic tag (e.g. "ai_server", "apple_chain")

    Returns:
        dict with keys: event_type, direction, source_entity, topic,
                        confidence, summary
        None if no keyword rule matched.
    """
    if not title or not title.strip():
        return None

    title_clean = title.strip()

    best_match = None
    best_confidence = 0.0

    for pattern, event_type, direction, base_conf in _KEYWORD_RULES:
        m = re.search(pattern, title_clean, re.IGNORECASE)
        if m:
            # Prefer longer matches / higher confidence
            conf = base_conf
            if conf > best_confidence:
                best_match = (event_type, direction, conf, m.group())
                best_confidence = conf

    if best_match is None:
        return None

    event_type, direction, confidence, matched_text = best_match

    # Fix direction for price_extreme
    if event_type == "price_extreme":
        direction = _detect_price_extreme_direction(title_clean)

    source_entity = _detect_entity(title_clean)

    # Build summary: truncated title
    summary = title_clean[:80]

    return {
        "event_type": event_type,
        "direction": direction,
        "source_entity": source_entity,
        "topic": topic or "unknown",
        "confidence": confidence,
        "summary": summary,
    }


def batch_extract(news_items: list[dict]) -> list[dict]:
    """Process a list of news items and extract structured events.

    Args:
        news_items: list of dicts, each with at least "title" and "topic" keys.
                    Optional: "date", "source_quality", "domain", etc.

    Returns:
        List of event dicts (deduplicated by title hash).
    """
    seen_hashes: set[str] = set()
    events: list[dict] = []

    for item in news_items:
        title = (item.get("title") or "").strip()
        if not title:
            continue

        # Deduplicate by title hash
        h = hashlib.sha256(title.lower().encode("utf-8")).hexdigest()[:16]
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        topic = item.get("topic", "")
        event = extract_from_title(title, topic)

        if event is None:
            logger.debug("No keyword match for: %s", title[:60])
            continue

        # Merge metadata from news item
        event["date"] = item.get("date", "")
        event["title_hash"] = h
        event["source_quality"] = item.get("source_quality", 0.5)
        event["domain"] = item.get("domain", "")

        # Adjust confidence by source quality
        sq = event["source_quality"]
        event["confidence"] = round(event["confidence"] * (0.5 + 0.5 * sq), 2)

        events.append(event)

    logger.info(
        "batch_extract: %d items -> %d events (deduped from %d)",
        len(news_items), len(events), len(seen_hashes),
    )
    return events
