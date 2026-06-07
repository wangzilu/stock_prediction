"""Build Global Chain Factors — Phase 4U Day 4.

Propagates global supply-chain events through the edge table
to produce per-stock factor scores.

Pipeline:
  1. Load supply_chain_edges.yaml
  2. Load global chain events (from extractor or news JSONL)
  3. For each event x matching edge: score = dir * conf * weight * edge_dir * decay
  4. Aggregate per stock per date
  5. Output: data/storage/global_chain_factors.parquet

Usage:
    python -m scripts.build_global_chain_factors [--date 2026-05-25] [--demo]
    python -m scripts.build_global_chain_factors --demo  # synthetic events for testing
"""
import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scheduler.data_health import HealthStatus, write_health

logger = logging.getLogger(__name__)

EDGES_PATH = PROJECT_ROOT / "data" / "config" / "supply_chain_edges.yaml"
NEWS_DIR = PROJECT_ROOT / "data" / "storage" / "global_industry_news"
EVENTS_DIR = PROJECT_ROOT / "data" / "storage" / "global_chain_events"
EVENTS_DIR_LLM = PROJECT_ROOT / "data" / "storage" / "global_chain_events_llm"
EVENTS_DIR_LLM_V2 = PROJECT_ROOT / "data" / "storage" / "global_chain_events_llm_v2"
OUTPUT_PATH = PROJECT_ROOT / "data" / "storage" / "global_chain_factors.parquet"
OUTPUT_PATH_LLM = PROJECT_ROOT / "data" / "storage" / "global_chain_factors_llm.parquet"

# Decay half-life in days (event impact halves every N days)
DECAY_HALF_LIFE = 3
DECAY_LAMBDA = math.log(2) / DECAY_HALF_LIFE

# How many days back to look for events
EVENT_LOOKBACK_DAYS = 10


# ---------------------------------------------------------------------------
# Edge table
# ---------------------------------------------------------------------------

# Phase D / SC-A3: tier classification per edge.
# Each edge in the YAML carries a ``source`` string; we infer an
# evidence tier from it so production overlays can opt into A/B only
# while shadow / research consumers can read C/D too.
#
#   A — official confirmations (年报 / 公告 / 政策 / regulatory disclosure).
#   B — public company-interaction signals (订单 / 合作 / 认证 /
#       公司互动 / 供应链公开信息 contracts).
#   C — research-report inferences (研报 / 行业研报 / 调研 / 卖方).
#   D — pure theme mapping (行业逻辑 / 行业常识 / 主题 / industry
#       narrative without per-edge evidence).
#
# When in doubt the inference falls back to ``"D"`` so production
# overlays do not accidentally consume an unclassified edge.
EDGE_TIER_RULES = (
    # (tier, ordered list of regex hints on the source string)
    ("A", ("公告", "年报", "政策", "公告/年报", "regulatory")),
    ("B", ("订单", "合作", "认证", "公司互动", "供应链公开信息",
           "合资", "合同")),
    ("C", ("研报", "行业研报", "调研", "卖方", "机构调研")),
    ("D", ("行业逻辑", "行业常识", "主题", "narrative", "公开信息")),
)


def classify_edge_tier(source: str) -> str:
    """Infer the evidence tier (``"A"`` / ``"B"`` / ``"C"`` / ``"D"``)
    from an edge's ``source`` field. Defaults to ``"D"`` when the
    source string is empty / unmatched so production overlays cannot
    silently consume an unclassified edge.
    """
    s = (source or "").strip()
    if not s:
        return "D"
    s_lower = s.lower()
    for tier, hints in EDGE_TIER_RULES:
        if any(h in s or h.lower() in s_lower for h in hints):
            return tier
    return "D"


# Tier inclusivity for the production overlay. The current default
# accepts A and B only; C / D are research / shadow-only. Phase D.3
# audit can flip this once the YAML is fully retagged.
PRODUCTION_EDGE_TIERS = frozenset({"A", "B"})


def load_edges(
    min_tier: str = "B",
    *,
    annotate: bool = True,
    explicit_tiers: frozenset[str] | None = None,
) -> list[dict]:
    """Load supply chain edges from YAML, tagged with an evidence tier.

    Parameters
    ----------
    min_tier:
        Minimum tier to include. ``"B"`` means production-grade
        (A + B); ``"D"`` means everything. Convenience wrapper around
        ``explicit_tiers`` so callers can write ``load_edges("B")``
        for the production set or ``load_edges("D")`` for research.
    annotate:
        When True (default), the returned edges carry an added
        ``tier`` key. Set False only if a caller depends on the
        original YAML key set.
    explicit_tiers:
        Override ``min_tier`` with an exact set, e.g.
        ``frozenset({"A"})`` to read official-only.

    Returns
    -------
    List of edge dicts. Filtered by tier when callers care.
    """
    with open(EDGES_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []

    # Resolve which tiers we keep.
    if explicit_tiers is not None:
        keep = frozenset(explicit_tiers)
    else:
        TIER_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}
        threshold = TIER_ORDER.get(min_tier.upper(), 1)
        keep = frozenset(
            t for t, idx in TIER_ORDER.items() if idx <= threshold
        )

    kept: list[dict] = []
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for edge in raw:
        tier = classify_edge_tier(edge.get("source", ""))
        counts[tier] += 1
        if tier not in keep:
            continue
        if annotate:
            edge = {**edge, "tier": tier}
        kept.append(edge)
    logger.info(
        "Loaded %d edges (tiers kept=%s, raw counts %s)",
        len(kept), sorted(keep), counts,
    )
    return kept


def _build_entity_edge_map(edges: list[dict]) -> dict[str, list[dict]]:
    """Index edges by (src_entity_lower, topic_lower) for fast lookup."""
    emap: dict[str, list[dict]] = defaultdict(list)
    for edge in edges:
        src = edge.get("src_entity", "").strip()
        # Index by entity name (case-insensitive)
        emap[src.lower()].append(edge)
    return emap


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------

def _load_pre_extracted_events(
    target_date: str,
    lookback_days: int = EVENT_LOOKBACK_DAYS,
    source: str = "rule",
) -> list[dict]:
    """Load pre-extracted events from global_chain_events/ or global_chain_events_llm/.

    2026-06-07: added ``source`` arg for B.7 ablation. ``"rule"`` reads
    from the rule-based extractor's output (production cron), ``"llm"``
    reads from the LLM extractor's output. Both share the downstream
    propagation logic so the LOO is comparing pipelines, not propagation.

    2026-06-07 (SC-A2): ``source="llm_v2"`` reads the v2 relations
    schema from ``global_chain_events_llm_v2/`` and rewrites each row
    into the v1-shaped event dict ``propagate_scores`` already
    understands. Direction is derived from ``relation_type`` (suppliers
    co-move with their customer's polarity sign of +1; competitors
    invert; theme_co_member dampens). This keeps the parquet schema
    identical so FeatureMerger does not need to change.
    """
    import json
    if source == "llm":
        events_dir = EVENTS_DIR_LLM
    elif source == "llm_v2":
        events_dir = EVENTS_DIR_LLM_V2
    elif source == "rule":
        events_dir = EVENTS_DIR
    else:
        raise ValueError(
            f"Unknown source {source!r}, must be 'rule', 'llm', or 'llm_v2'"
        )
    target = pd.Timestamp(target_date)
    events = []
    for i in range(lookback_days):
        d = (target - pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        path = events_dir / f"{d}.jsonl"
        if path.exists():
            for line in open(path):
                line = line.strip()
                if line:
                    e = json.loads(line)
                    e.setdefault("date", d)
                    if source == "llm_v2":
                        events.extend(_v2_event_to_v1_shape(e))
                    elif source == "llm":
                        events.append(_llm_v1_to_propagation_shape(e))
                    else:
                        events.append(e)
    if events:
        logger.info(f"Loaded {len(events)} pre-extracted events from {events_dir}")
    return events


# ---------------------------------------------------------------------------
# v2 schema adaptation
# ---------------------------------------------------------------------------

# Phase D / SC-A2: relation_type → direction sign + confidence-scale.
#
# The LLM emits only relations; this table is the downstream weighting
# step (news polarity × relation type) the spec demands be separated
# from extraction. Multiplied later by news polarity (currently
# assumed +1 because the prefilter accepts both polarities; a follow-on
# story will plumb sentiment into here).
#
# Values are deliberately small (≤ 0.7) so a v2 row carries the same
# magnitude order as a v1 row whose confidence∈[0,1].
_RELATION_SIGN = {
    "supplier":          (+1, 0.7),   # X is supplier to news subject → moves with subject
    "customer":          (+1, 0.6),   # X is customer of subject → moves with subject
    "joint_venture":     (+1, 0.6),
    "competitor":        (-1, 0.5),   # competitive readthrough — opposite sign
    "regulatory_target": (-1, 0.6),
    "theme_co_member":   (+1, 0.3),   # weakest tie — dampened
}

# Evidence-strength weighting. A=full weight; D=de-rated.
_EVIDENCE_WEIGHT = {"A": 1.0, "B": 0.8, "C": 0.5, "D": 0.3}


def _v2_event_to_v1_shape(v2_event: dict) -> list[dict]:
    """Translate one v2 ``{src_entity, relations[]}`` event into the
    list of v1-shaped event dicts ``propagate_scores`` consumes.

    One v2 row with N relations expands to N v1 rows, each pointing
    its ``source_entity`` at the relation's ``target_entity`` (so the
    propagation graph can match against ``supply_chain_edges.yaml``'s
    src_entity index in either direction).

    Direction = sign(relation_type) — derived here, NOT extracted by
    the LLM. Confidence = relation_sign_weight × evidence_weight.
    """
    src = (v2_event.get("src_entity") or "").strip()
    relations = v2_event.get("relations") or []
    if not src or not relations:
        return []

    factuality = (v2_event.get("factuality") or "").strip().lower()
    # Confirmed facts get full strength; speculation/rumor dampen.
    fact_scale = {"confirmed": 1.0, "speculation": 0.6, "rumor": 0.3}.get(
        factuality, 0.5
    )

    out: list[dict] = []
    for rel in relations:
        target = (rel.get("target_entity") or "").strip()
        rel_type = (rel.get("relation_type") or "").strip()
        if not target or rel_type not in _RELATION_SIGN:
            continue
        ev = (rel.get("evidence_strength") or "D").strip().upper()
        sign, base = _RELATION_SIGN[rel_type]
        ev_w = _EVIDENCE_WEIGHT.get(ev, 0.3)
        conf = base * ev_w * fact_scale

        # SC-A3 tier filter: caller (propagate_scores) already drops
        # to A/B by default via load_edges; passing evidence_strength
        # through as ``source`` lets classify_edge_tier downstream
        # see it should the propagation step ever filter on it.
        # We emit TWO v1-shaped events per relation: one keyed on
        # src_entity (Nvidia announces order with TSMC → look up
        # Nvidia in YAML) and one keyed on target_entity (look up
        # TSMC in YAML). The propagation step deduplicates by
        # downstream dst_stock so this is safe.
        for entity in (src, target):
            out.append({
                # propagate_scores indexes on source_entity
                "source_entity": entity,
                # legacy direction field — derived from relation, NOT
                # extracted by the LLM.
                "direction": sign,
                "confidence": round(conf, 4),
                "topic": v2_event.get("topic", ""),
                "date": v2_event.get("date", ""),
                "published_at": v2_event.get("published_at", ""),
                "news_title": v2_event.get("news_title", ""),
                "news_url": v2_event.get("news_url", ""),
                "summary": v2_event.get("summary", ""),
                # Tier — Phase D / SC-A3 reads this if it ever wires
                # event-side tier filtering.
                "evidence_strength": ev,
                "schema_version": "v2",
                "source": "llm_v2",
            })
    return out


# Evidence-level weight for v1-LLM events. The LLM extractor emits a
# categorical evidence_level instead of a numeric confidence; this is
# the multiplicative weight applied to chain_relevance_score (NOT a
# floor — cx batch C P2 #6 fix: the original max(floor, rel) shape let
# a major_media event with chain_relevance_score=1 (very weak supply-
# chain link) become confidence=0.7 because the media-level floor
# dominated. That would have biased B.7-LLM toward "famous outlets win
# over relevant outlets", the opposite of what the propagation model
# needs. Now: confidence = evidence_weight × relevance, both ∈ [0,1].
_LLM_V1_EVIDENCE_WEIGHT = {
    "major_media": 1.0,
    "trade_press": 0.85,
    "company_pr":  0.75,
    "rumor":       0.4,
}


def _llm_v1_to_propagation_shape(e: dict) -> dict:
    """Translate one v1-LLM event (``global_entity`` + ``chain_relevance_score``)
    into the shape ``propagate_scores`` consumes (``source_entity`` +
    numeric ``confidence``). Returns the event in-place with extra keys
    so unaffected downstream readers keep working.

    Without this, B.7-LLM ablation produced 0 propagation pairs because
    ``_match_event_to_edges`` looked up ``source_entity`` but the LLM
    pipeline only ever wrote ``global_entity``.
    """
    if "source_entity" not in e and "global_entity" in e:
        e["source_entity"] = e["global_entity"]
    if "confidence" not in e:
        rel = float(e.get("chain_relevance_score", 0.0) or 0.0) / 10.0
        rel = max(0.0, min(1.0, rel))
        ev_weight = _LLM_V1_EVIDENCE_WEIGHT.get(
            (e.get("evidence_level") or "").lower(), 0.6,
        )
        # Multiplicative fusion: a weak supply-chain link CAN'T be
        # rescued by a top-tier outlet. A high-relevance link from a
        # rumor still passes through dampened (0.4×). The default
        # 0.6 (unknown evidence_level) sits between trade_press and
        # rumor — conservative but non-zero.
        e["confidence"] = ev_weight * rel
    return e


def load_events_from_news(
    target_date: str,
    lookback_days: int = EVENT_LOOKBACK_DAYS,
) -> list[dict]:
    """Load and extract events from global industry news JSONL files.

    Looks back `lookback_days` from target_date.
    """
    from factors.global_supply_chain_extractor import batch_extract

    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    all_news = []

    for d in range(lookback_days):
        dt = target_dt - timedelta(days=d)
        date_str = dt.strftime("%Y-%m-%d")
        news_path = NEWS_DIR / f"{date_str}.jsonl"
        if not news_path.exists():
            continue
        with open(news_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item.setdefault("date", date_str)
                all_news.append(item)

    if not all_news:
        logger.warning("No news files found for %s (lookback %d days)", target_date, lookback_days)
        return []

    events = batch_extract(all_news)
    logger.info("Extracted %d events from %d news items", len(events), len(all_news))
    return events


def load_events_from_jsonl(events_path: Path) -> list[dict]:
    """Load pre-extracted events from a JSONL file."""
    events = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def generate_demo_events(target_date: str) -> list[dict]:
    """Generate synthetic events for testing the pipeline."""
    from factors.global_supply_chain_extractor import batch_extract

    demo_news = [
        {
            "title": "Nvidia Blackwell demand remains strong, orders increased significantly",
            "topic": "ai_server",
            "date": target_date,
            "source_quality": 0.8,
            "domain": "reuters.com",
        },
        {
            "title": "Apple cuts iPhone supplier orders amid weak demand outlook",
            "topic": "apple_chain",
            "date": target_date,
            "source_quality": 0.9,
            "domain": "bloomberg.com",
        },
        {
            "title": "TSMC raises capex guidance for 2026 amid AI chip boom",
            "topic": "semiconductor",
            "date": target_date,
            "source_quality": 0.9,
            "domain": "reuters.com",
        },
        {
            "title": "Lithium prices fall to multi-year lows on oversupply",
            "topic": "ev_battery",
            "date": target_date,
            "source_quality": 0.7,
            "domain": "cnbc.com",
        },
        {
            "title": "Tesla reports record deliveries, strong demand for Model Y",
            "topic": "ev",
            "date": target_date,
            "source_quality": 0.8,
            "domain": "reuters.com",
        },
        {
            "title": "US export control tightens restrictions on semiconductor equipment to China",
            "topic": "semiconductor",
            "date": target_date,
            "source_quality": 0.9,
            "domain": "ft.com",
        },
        {
            "title": "SK Hynix expands HBM production capacity with new factory in Korea",
            "topic": "semiconductor",
            "date": target_date,
            "source_quality": 0.8,
            "domain": "bloomberg.com",
        },
        {
            "title": "Meta increases AI infrastructure capex by 30 percent",
            "topic": "ai_server",
            "date": target_date,
            "source_quality": 0.8,
            "domain": "cnbc.com",
        },
    ]

    events = batch_extract(demo_news)
    logger.info("Generated %d demo events", len(events))
    return events


# ---------------------------------------------------------------------------
# Score propagation
# ---------------------------------------------------------------------------

def _compute_decay(event_date: str, target_date: str) -> float:
    """Compute exponential decay factor based on event age."""
    try:
        evt_dt = datetime.strptime(event_date, "%Y-%m-%d")
        tgt_dt = datetime.strptime(target_date, "%Y-%m-%d")
        age_days = (tgt_dt - evt_dt).days
    except (ValueError, TypeError):
        age_days = 0

    if age_days < 0:
        # Future event — full weight
        age_days = 0

    return math.exp(-DECAY_LAMBDA * age_days)


def _match_event_to_edges(
    event: dict,
    entity_edge_map: dict[str, list[dict]],
) -> list[dict]:
    """Find edges that match this event's source entity."""
    src_entity = event.get("source_entity", "").lower()
    if not src_entity or src_entity == "unknown":
        return []
    return entity_edge_map.get(src_entity, [])


def propagate_scores(
    events: list[dict],
    edges: list[dict],
    target_date: str,
) -> pd.DataFrame:
    """Propagate event scores through edge table to produce per-stock factors.

    For each event x matching edge:
        score = event_direction * event_confidence * edge_weight * edge_direction * decay

    Returns:
        DataFrame with (datetime, instrument) MultiIndex and factor columns.
    """
    entity_edge_map = _build_entity_edge_map(edges)

    # Accumulate scores per stock
    stock_scores: dict[str, dict] = defaultdict(lambda: {
        "alpha_sum": 0.0,
        "pos_score": 0.0,
        "neg_score": 0.0,
        "event_count": 0,
        "dst_name": "",
    })

    propagation_count = 0

    for event in events:
        event_dir = event.get("direction", 0)
        event_conf = event.get("confidence", 0.5)
        # 2026-06-06 PIT fix (P1 #5): prefer published_at (real PIT
        # timestamp from GDELT / Google RSS) over the legacy ``date``
        # field which the pre-fix collector clobbered to target_date.
        # Strip to first 10 chars (YYYY-MM-DD) since _compute_decay
        # parses with strptime "%Y-%m-%d".
        pub = (event.get("published_at") or "").strip()
        event_date = pub[:10] if pub else event.get("date", target_date)
        decay = _compute_decay(event_date, target_date)

        matched_edges = _match_event_to_edges(event, entity_edge_map)
        if not matched_edges:
            continue

        for edge in matched_edges:
            edge_weight = edge.get("weight", 0.5)
            edge_dir = edge.get("direction", 1)
            edge_conf = edge.get("confidence", 0.5)
            dst_stock = edge.get("dst_stock", "")
            dst_name = edge.get("dst_name", "")

            if not dst_stock:
                continue

            # Core formula
            score = event_dir * event_conf * edge_weight * edge_dir * edge_conf * decay

            entry = stock_scores[dst_stock]
            entry["alpha_sum"] += score
            entry["event_count"] += 1
            entry["dst_name"] = dst_name
            if score > 0:
                entry["pos_score"] += score
            elif score < 0:
                entry["neg_score"] += score  # negative value

            propagation_count += 1

    logger.info(
        "Propagated %d event-edge pairs across %d stocks",
        propagation_count, len(stock_scores),
    )

    if not stock_scores:
        logger.warning("No stock scores produced — returning empty DataFrame")
        return pd.DataFrame()

    # Build DataFrame
    records = []
    dt = pd.Timestamp(target_date)
    for stock_code, scores in stock_scores.items():
        # Convert stock code to qlib instrument format
        # sz300308 -> SZ300308, sh601138 -> SH601138
        instrument = stock_code.upper()

        records.append({
            "datetime": dt,
            "instrument": instrument,
            "global_chain_alpha": round(scores["alpha_sum"], 6),
            "global_chain_event_count": scores["event_count"],
            "global_chain_pos_score": round(scores["pos_score"], 6),
            "global_chain_neg_score": round(scores["neg_score"], 6),
            # Phase D / SC-A1: separate company-level alpha for the
            # consumer that wants the higher-confidence half of the
            # signal. Equal to global_chain_alpha at this stage because
            # the rows produced here ARE the company-level path; the
            # industry-level rows added later inject their own column
            # values.
            "company_level_alpha": round(scores["alpha_sum"], 6),
            "industry_level_alpha": 0.0,
            "level": "company",
        })

    df = pd.DataFrame(records)
    df = df.set_index(["datetime", "instrument"])
    df = df.sort_index()

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

SHRINK_MODES = ("hard_zero", "sqrt_dampen", "soft_clip")
DEFAULT_SHRINK_MODE = "sqrt_dampen"


def _apply_shrink(
    industry_only_pairs: list[tuple[str, float]],
    *,
    mode: str,
) -> list[tuple[str, float]]:
    """Apply the chosen industry-level shrink to ``industry_only_pairs``.

    Returns a list of ``(stock, shrunk_score)`` pairs. The three modes
    were introduced under #174 step 2 because the legacy zscore × 0.2
    + clip[±3] pipeline collapsed to a hard zero whenever the cross-
    stock variance was tiny, which killed the chain-factor density on
    low-event days (see docs/phase_b7_verdict_20260607.md).

    Modes
    -----
    hard_zero (legacy)
        zscore × 0.2 then clip to [-3, +3]; if cross-stock std < 1e-9
        the WHOLE column collapses to 0. Preserved for rollback.

    sqrt_dampen (new default)
        ``score / sqrt(n_industry_stocks)`` followed by tanh clamp
        into [-1, +1]. A 100-stock industry contributes 1/10 per stock
        instead of 0; a 1-stock industry contributes the raw magnitude
        bounded by tanh so a single outlier never blows up the scale.
        Bias-free: an industry where every event was negative stays
        negative across all its stocks.

    soft_clip
        ``tanh(score / 10) * 0.5``. Per-stock magnitude is bounded
        without any cross-stock normalisation; useful when callers want
        the raw event sign to dominate over the per-day distribution.
    """
    if mode not in SHRINK_MODES:
        raise ValueError(f"shrink_mode must be one of {SHRINK_MODES}, got {mode!r}")

    if not industry_only_pairs:
        return []

    import numpy as _np
    import math as _math

    if mode == "hard_zero":
        pool_scores = _np.array([s for _, s in industry_only_pairs], dtype=float)
        if pool_scores.std() > 1e-9:
            mu = float(pool_scores.mean())
            sigma = float(pool_scores.std())
            SHRINK = 0.2
            CLIP = 3.0
            out = []
            for stock, score in industry_only_pairs:
                z = (score - mu) / sigma
                shrunk = max(-CLIP, min(CLIP, z * SHRINK))
                out.append((stock, shrunk))
            return out
        # zero-variance branch — hard zero column.
        return [(stock, 0.0) for stock, _ in industry_only_pairs]

    if mode == "sqrt_dampen":
        # Group stocks by topic-affinity is approximated by the size of
        # the industry-only pool — every event already routes through
        # mapper.get_all_affected_stocks which collapses topic+industry
        # weight into a single per-stock score. n here is the total
        # cross-topic affected count, which is a fair proxy for the
        # "industry breadth" the spec asked us to dampen by. A 3000-
        # stock day downweights ~55× harder than a 30-stock day,
        # protecting against an entire-market sweep dominating the
        # cache.
        n = max(1, len(industry_only_pairs))
        denom = _math.sqrt(n)
        out = []
        for stock, score in industry_only_pairs:
            damped = score / denom
            # tanh clamp into [-1, +1] keeps the magnitude bounded
            # regardless of the raw event score scale.
            out.append((stock, _math.tanh(damped)))
        return out

    # soft_clip
    out = []
    for stock, score in industry_only_pairs:
        out.append((stock, _math.tanh(score / 10.0) * 0.5))
    return out


def build_factors(
    target_date: str,
    demo: bool = False,
    lookback_days: int = EVENT_LOOKBACK_DAYS,
    source: str = "rule",
    schema: str = "v1",
    shrink_mode: str = DEFAULT_SHRINK_MODE,
) -> pd.DataFrame:
    """End-to-end: load edges + events, propagate, output parquet.

    Args:
        target_date: YYYY-MM-DD
        demo: if True, use synthetic demo events
        lookback_days: how many days of news to look back
        source: "rule" (default, production cron path) or "llm"
            (reads extract_global_chain_llm.py output for B.7 ablation).
        schema: "v1" (default — legacy direction/confidence event shape)
            or "v2" (SC-A2 relations schema; reads global_chain_events_llm_v2/
            and adapts to v1-shaped propagation input). When schema="v2"
            the ``source`` arg is overridden to "llm_v2" since v2 only
            exists for the LLM path.
        shrink_mode: industry-level shrink policy — one of
            {hard_zero, sqrt_dampen, soft_clip}. Default sqrt_dampen
            (#174 step 2) preserves per-stock non-zero values even in
            low-variance topics. hard_zero is the legacy behaviour and
            stays available for fast rollback.

    Returns:
        DataFrame with factor values.
    """
    edges = load_edges()

    # SC-A2: v2 schema is always the LLM v2 path.
    if schema == "v2":
        source = "llm_v2"

    if demo:
        events = generate_demo_events(target_date)
    else:
        # Prefer pre-extracted events from global_chain_events/ (produced by
        # extract_global_supply_chain_events.py). Fall back to raw news + rule
        # extraction if pre-extracted events don't exist.
        # 2026-06-07: source="llm" routes to global_chain_events_llm/
        # produced by extract_global_chain_llm.py.
        events = _load_pre_extracted_events(target_date, lookback_days, source=source)
        if not events and source == "rule":
            # LLM source has no fallback — if extraction failed, the
            # B.7 ablation must reflect that absence honestly.
            events = load_events_from_news(target_date, lookback_days=lookback_days)

    if not events:
        logger.warning("No events found — cannot build factors")
        return pd.DataFrame()

    # Company-level propagation (high confidence)
    df = propagate_scores(events, edges, target_date)

    # Industry-level propagation (wider coverage, lower weight).
    # cx review 2026-06-06: explicitly pin the mapper to the production
    # A/B tier set so it cannot silently bypass the SC-A3 edge filter
    # the top-level load_edges call honours. Default would already be
    # ``min_tier="B"`` but stating it here keeps the contract local.
    try:
        from factors.supply_chain_mapper import SupplyChainMapper
        mapper = SupplyChainMapper(
            explicit_tiers=PRODUCTION_EDGE_TIERS,
        )
        industry_scores = mapper.get_all_affected_stocks(events)

        if industry_scores:
            dt = pd.Timestamp(target_date)
            # Normalize company-level instruments to lowercase
            if not df.empty:
                df.index = pd.MultiIndex.from_tuples(
                    [(d, inst.lower()) for d, inst in df.index],
                    names=df.index.names,
                )
            company_stocks = set()
            if not df.empty:
                company_stocks = set(df.index.get_level_values("instrument"))

            # Phase D / SC-A1: industry-level alpha used to be
            # zscore-by-date + shrunk × 0.2 + clipped [±3], with a
            # hard-zero collapse whenever cross-stock std < 1e-9.
            # docs/phase_b7_verdict_20260607.md showed that policy left
            # industry_level_alpha uniformly 0 in production caches
            # (< 0.01 % non-zero rows), so xgb trees never split on it.
            #
            # 2026-06-07 (#174 step 2): the shrink policy is now
            # selectable. Default ``sqrt_dampen`` preserves a per-stock
            # non-zero floor (score / sqrt(n) → tanh) so a low-
            # variance topic no longer zero-collapses the column.
            # ``hard_zero`` remains the rollback path.
            #
            # 2026-06-06 cx review (preserved): the industry pool MUST
            # be filtered against company-level stocks BEFORE any
            # cross-stock moment / dampening is computed, otherwise the
            # company-level rows would skew the per-day distribution.

            # Step A: build the (stock, score) list that will actually
            # become industry rows, dropping company-level overlap.
            industry_only_pairs = [
                (stock.lower(), score)
                for stock, score in industry_scores.items()
                if stock.lower() not in company_stocks
            ]

            shrunk_pairs = _apply_shrink(industry_only_pairs, mode=shrink_mode)

            industry_rows = []
            for stock, shrunk in shrunk_pairs:
                industry_rows.append({
                    "datetime": dt,
                    "instrument": stock,
                    # global_chain_alpha keeps the SHRUNK number for
                    # backward compat — consumers that were reading
                    # the raw value were the source of the production
                    # leak A.5-1 closed; they should NOT keep seeing
                    # the raw distribution.
                    "global_chain_alpha": round(shrunk, 4),
                    "global_chain_event_count": 1,
                    "global_chain_pos_score": round(max(0.0, shrunk), 4),
                    "global_chain_neg_score": round(min(0.0, shrunk), 4),
                    "company_level_alpha": 0.0,
                    "industry_level_alpha": round(shrunk, 4),
                    "level": "industry",
                })

            if industry_rows:
                import numpy as _np
                shrunk_arr = _np.array([s for _, s in shrunk_pairs], dtype=float)
                nz = int((shrunk_arr != 0).sum())
                logger.info(
                    f"Industry-level: +{len(industry_rows)} stocks "
                    f"(shrink_mode={shrink_mode}, non-zero={nz}, "
                    f"range=[{shrunk_arr.min():+.4f}, {shrunk_arr.max():+.4f}]). "
                    f"Total {len(df) + len(industry_rows)} stocks."
                )
                ind_df = pd.DataFrame(industry_rows).set_index(["datetime", "instrument"])
                df = pd.concat([df, ind_df])
    except ImportError:
        logger.warning("supply_chain_mapper not available — company-level only")

    if df.empty:
        logger.warning("No factor scores produced")
        return df

    # Save to parquet (append if exists). 2026-06-07: route to
    # OUTPUT_PATH_LLM when source="llm" (or "llm_v2") so neither LLM
    # ablation clobbers the production rule-based parquet. The v2
    # path SHARES the same parquet schema as v1 by design (FeatureMerger
    # loader does not change) so it lands on the same parquet file.
    out_path = OUTPUT_PATH_LLM if source in ("llm", "llm_v2") else OUTPUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        # Remove existing rows for this date to avoid duplicates
        if "datetime" in existing.index.names:
            dt = pd.Timestamp(target_date)
            mask = existing.index.get_level_values("datetime") != dt
            existing = existing[mask]
        df = pd.concat([existing, df])
        df = df.sort_index()

    df.to_parquet(out_path)
    logger.info("Saved factors to %s (%d rows)", out_path, len(df))

    return df


def _write_chain_factor_health(
    *,
    success: bool,
    target_date: str,
    n_items: int = 0,
    error_type: str = "",
    error_message: str = "",
    partial: bool = False,
) -> None:
    write_health("global_chain_factors", HealthStatus(
        success=success,
        n_items=n_items,
        latest_date=target_date if success else "",
        error_type=error_type,
        error_message=error_message[:200],
        network_profile="none",
        partial=partial,
    ))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build global supply chain factor scores"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Target date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use synthetic demo events for testing",
    )
    parser.add_argument(
        "--lookback", type=int, default=EVENT_LOOKBACK_DAYS,
        help=f"Days of news to look back (default: {EVENT_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--source", choices=["rule", "llm", "llm_v2"], default="rule",
        help="Event source: 'rule' (default, production cron), 'llm' "
             "(B.7 ablation — reads global_chain_events_llm/ from "
             "extract_global_chain_llm.py output), or 'llm_v2' "
             "(SC-A2 — reads global_chain_events_llm_v2/ with the "
             "relations schema). LLM sources write to "
             "global_chain_factors_llm.parquet so they don't "
             "clobber the production parquet.",
    )
    parser.add_argument(
        "--schema", choices=["v1", "v2"], default="v1",
        help="Event JSONL schema: 'v1' (default, legacy direction/"
             "confidence shape) or 'v2' (SC-A2 src_entity + relations). "
             "Setting --schema v2 implicitly sets --source llm_v2.",
    )
    parser.add_argument(
        "--shrink-mode", choices=list(SHRINK_MODES),
        default=DEFAULT_SHRINK_MODE,
        help=f"Industry-level shrink policy (#174 step 2). "
             f"Default {DEFAULT_SHRINK_MODE}. 'hard_zero' is the legacy "
             "zscore × 0.2 + clip[±3] with zero-variance collapse to 0; "
             "'sqrt_dampen' divides each per-stock score by sqrt(n) and "
             "tanh-clamps to [-1, +1]; 'soft_clip' uses tanh(score/10) × 0.5.",
    )
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")

    try:
        df = build_factors(target_date, demo=args.demo, lookback_days=args.lookback,
                           source=args.source, schema=args.schema,
                           shrink_mode=args.shrink_mode)
        if not df.empty:
            print(f"\n=== Global Chain Factors for {target_date} ===")
            print(f"Stocks affected: {len(df)}")
            print(f"\nTop positive alpha:")
            top_pos = df.nlargest(10, "global_chain_alpha")
            for idx, row in top_pos.iterrows():
                inst = idx[1] if isinstance(idx, tuple) else idx
                print(f"  {inst:12s}  alpha={row['global_chain_alpha']:+.4f}  "
                      f"events={int(row['global_chain_event_count'])}  "
                      f"pos={row['global_chain_pos_score']:+.4f}  "
                      f"neg={row['global_chain_neg_score']:+.4f}")
            print(f"\nTop negative alpha:")
            top_neg = df.nsmallest(5, "global_chain_alpha")
            for idx, row in top_neg.iterrows():
                inst = idx[1] if isinstance(idx, tuple) else idx
                print(f"  {inst:12s}  alpha={row['global_chain_alpha']:+.4f}  "
                      f"events={int(row['global_chain_event_count'])}  "
                      f"pos={row['global_chain_pos_score']:+.4f}  "
                      f"neg={row['global_chain_neg_score']:+.4f}")
        else:
            print("No factors produced.")
            _write_chain_factor_health(
                success=False,
                target_date=target_date,
                error_type="NoFactors",
                error_message="No events/factors produced; refusing to keep stale global_chain_factors parquet green",
            )
            sys.exit(1)
        _write_chain_factor_health(
            success=True,
            target_date=target_date,
            n_items=len(df),
        )
    except Exception as e:
        logger.error("Factor build failed: %s", e)
        _write_chain_factor_health(
            success=False,
            target_date=target_date,
            error_type=type(e).__name__,
            error_message=str(e),
        )
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
