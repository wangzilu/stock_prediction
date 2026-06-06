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
OUTPUT_PATH = PROJECT_ROOT / "data" / "storage" / "global_chain_factors.parquet"

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
) -> list[dict]:
    """Load pre-extracted events from global_chain_events/ directory.

    These are produced by extract_global_supply_chain_events.py and are
    preferred over re-extracting from raw news.
    """
    import json
    target = pd.Timestamp(target_date)
    events = []
    for i in range(lookback_days):
        d = (target - pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        path = EVENTS_DIR / f"{d}.jsonl"
        if path.exists():
            for line in open(path):
                line = line.strip()
                if line:
                    e = json.loads(line)
                    e.setdefault("date", d)
                    events.append(e)
    if events:
        logger.info(f"Loaded {len(events)} pre-extracted events from {EVENTS_DIR}")
    return events


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
        event_date = event.get("date", target_date)
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

def build_factors(
    target_date: str,
    demo: bool = False,
    lookback_days: int = EVENT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """End-to-end: load edges + events, propagate, output parquet.

    Args:
        target_date: YYYY-MM-DD
        demo: if True, use synthetic demo events
        lookback_days: how many days of news to look back

    Returns:
        DataFrame with factor values.
    """
    edges = load_edges()

    if demo:
        events = generate_demo_events(target_date)
    else:
        # Prefer pre-extracted events from global_chain_events/ (produced by
        # extract_global_supply_chain_events.py). Fall back to raw news + rule
        # extraction if pre-extracted events don't exist.
        events = _load_pre_extracted_events(target_date, lookback_days)
        if not events:
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

            # Phase D / SC-A1: industry-level alpha must be
            # zscore-by-date + shrunk + clipped before consumers see it.
            # Raw scores have mean ~ -14 and span [-47, +15] which is
            # not usable as either a feature or an overlay weight.
            #
            # 2026-06-06 cx review:
            #   1. Z-score must be computed AFTER filtering out the
            #      company-level stocks (those rows never reach the
            #      industry frame, so they would skew μ/σ — see review
            #      P2). Filter first, then compute moments.
            #   2. When the resulting σ is below the float threshold we
            #      set the WHOLE column to 0, not score * 0.2. Zero
            #      variance means "no information"; preserving the raw
            #      magnitude was what reintroduced the unscaled value
            #      flagged in P1.
            import numpy as _np
            SHRINK = 0.2
            CLIP = 3.0  # post-clip range [-3, 3], well within model input scale

            # Step A: build the (stock, score) list that will actually
            # become industry rows, dropping company-level overlap.
            industry_only_pairs = [
                (stock.lower(), score)
                for stock, score in industry_scores.items()
                if stock.lower() not in company_stocks
            ]

            if industry_only_pairs:
                pool_scores = _np.array(
                    [s for _, s in industry_only_pairs], dtype=float
                )
                if pool_scores.std() > 1e-9:
                    mu = float(pool_scores.mean())
                    sigma = float(pool_scores.std())
                    zero_variance = False
                else:
                    mu, sigma = 0.0, 1.0
                    zero_variance = True
            else:
                mu, sigma, zero_variance = 0.0, 1.0, True

            industry_rows = []
            for stock, score in industry_only_pairs:
                if zero_variance:
                    # No spread → no signal. Hard zero so the column
                    # never re-injects raw magnitude.
                    shrunk = 0.0
                else:
                    z = (score - mu) / sigma
                    shrunk = max(-CLIP, min(CLIP, z * SHRINK))
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
                ind_df = pd.DataFrame(industry_rows).set_index(["datetime", "instrument"])
                df = pd.concat([df, ind_df])
                logger.info(f"Industry-level: +{len(industry_rows)} stocks "
                            f"(zscore × {SHRINK} × clip[±{CLIP}], "
                            f"raw mean={mu:.1f} std={sigma:.1f}). "
                            f"Total {len(df)} stocks.")
    except ImportError:
        logger.warning("supply_chain_mapper not available — company-level only")

    if df.empty:
        logger.warning("No factor scores produced")
        return df

    # Save to parquet (append if exists)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        # Remove existing rows for this date to avoid duplicates
        if "datetime" in existing.index.names:
            dt = pd.Timestamp(target_date)
            mask = existing.index.get_level_values("datetime") != dt
            existing = existing[mask]
        df = pd.concat([existing, df])
        df = df.sort_index()

    df.to_parquet(OUTPUT_PATH)
    logger.info("Saved factors to %s (%d rows)", OUTPUT_PATH, len(df))

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
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")

    try:
        df = build_factors(target_date, demo=args.demo, lookback_days=args.lookback)
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
