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

def load_edges() -> list[dict]:
    """Load supply chain edges from YAML."""
    with open(EDGES_PATH, "r", encoding="utf-8") as f:
        edges = yaml.safe_load(f)
    if edges is None:
        edges = []
    logger.info("Loaded %d supply chain edges", len(edges))
    return edges


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

    # Industry-level propagation (wider coverage, lower weight)
    try:
        from factors.supply_chain_mapper import SupplyChainMapper
        mapper = SupplyChainMapper()
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

            industry_rows = []
            for stock, score in industry_scores.items():
                stock = stock.lower()
                if stock in company_stocks:
                    continue  # company-level already covered
                industry_rows.append({
                    "datetime": dt,
                    "instrument": stock,
                    "global_chain_alpha": round(score, 4),
                    "global_chain_event_count": 1,
                    "global_chain_pos_score": round(max(0, score), 4),
                    "global_chain_neg_score": round(min(0, score), 4),
                })

            if industry_rows:
                ind_df = pd.DataFrame(industry_rows).set_index(["datetime", "instrument"])
                df = pd.concat([df, ind_df])
                logger.info(f"Industry-level: +{len(industry_rows)} stocks "
                            f"(total {len(df)} stocks)")
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
    except Exception as e:
        logger.error("Factor build failed: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
