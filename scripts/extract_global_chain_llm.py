"""Extract global supply chain events using two-stage funnel.

Stage 1: Rule prefilter (1400+ → 100-200 candidates)
Stage 2: LLM extract (100-200 → 30-80 structured events)

Two output schemas coexist:

* **v1** (legacy)  → ``data/storage/global_chain_events_llm/<date>.jsonl``
  with the direction/confidence shape. This is the in-flight backfill
  path (PID 3026 on 2026-06-07). LEFT UNCHANGED.

* **v2** (SC-A2)   → ``data/storage/global_chain_events_llm_v2/<date>.jsonl``
  with the relations shape (``src_entity`` + ``relations[]``). The
  next backfill rolls this dir.

Falls back to pure rule extraction (v1 only) if LLM unavailable.

Usage:
    python scripts/extract_global_chain_llm.py [--date YYYY-MM-DD]
    python scripts/extract_global_chain_llm.py --schema v2 --date YYYY-MM-DD
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
NEWS_DIR = DATA_DIR / "global_industry_news"
LLM_EVENTS_DIR = DATA_DIR / "global_chain_events_llm"          # v1 (legacy)
LLM_EVENTS_DIR_V2 = DATA_DIR / "global_chain_events_llm_v2"    # v2 (SC-A2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--max-llm", type=int, default=80)
    parser.add_argument(
        "--schema", choices=["v1", "v2"], default="v1",
        help="LLM output schema. 'v1' = legacy direction/confidence "
             "(default, matches the in-flight backfill); 'v2' = SC-A2 "
             "relations / src_entity (new dir global_chain_events_llm_v2/).",
    )
    args = parser.parse_args()
    date = args.date
    schema = args.schema

    # Load raw news
    news_path = NEWS_DIR / f"{date}.jsonl"
    if not news_path.exists():
        logger.info(f"No global news for {date}, skipping")
        return

    raw_news = []
    with open(news_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_news.append(json.loads(line))

    logger.info(f"Loaded {len(raw_news)} raw news items for {date}")

    if not raw_news:
        return

    # Stage 1: Rule prefilter
    from factors.global_chain_prefilter import prefilter_news
    candidates = prefilter_news(raw_news, max_candidates=args.max_candidates)
    logger.info(f"Stage 1 prefilter: {len(raw_news)} → {len(candidates)} candidates")

    # Stage 2: LLM extraction (route by schema)
    if schema == "v2":
        from factors.global_chain_llm_extractor import extract_chain_events_llm_v2
        events = extract_chain_events_llm_v2(candidates, max_extract=args.max_llm)
        out_dir = LLM_EVENTS_DIR_V2
        health_key = "global_chain_llm_extract_v2"
    else:
        from factors.global_chain_llm_extractor import extract_chain_events_llm
        events = extract_chain_events_llm(candidates, max_extract=args.max_llm)
        out_dir = LLM_EVENTS_DIR
        health_key = "global_chain_llm_extract"
    logger.info(f"Stage 2 LLM ({schema}): {len(candidates)} → {len(events)} events")

    # 2026-06-06 P1 #2 + P2 #5 fixes:
    # - PIT: don't clobber e["date"]; preserve the publish date the
    #   downstream batch_extract already set. Only backfill when the
    #   event has no parseable publish info (consistent with the
    #   rule-based extractor at scripts/extract_global_supply_chain_events.py).
    # - health: empty extraction is a FAILURE state for downstream
    #   factors. Match the rule-based extractor's fail-loud semantics
    #   instead of unconditionally success=True with n_items=0.
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}.jsonl"
    with open(out_path, "w") as f:
        for e in events:
            if not e.get("date"):
                e["date"] = date  # only when truly unknown
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(events)} LLM events ({schema}) to {out_path}")

    # v1 also runs the rule-based comparison; v2 has no rule analogue
    # (rule extractor emits direction, v2 explicitly avoids direction)
    if schema == "v1":
        from factors.global_supply_chain_extractor import batch_extract
        rule_events = batch_extract(raw_news)
        logger.info(f"Rule-based comparison: {len(rule_events)} events")

    # Write health — empty extraction is a failure
    try:
        from scheduler.data_health import write_health, HealthStatus
        write_health(health_key, HealthStatus(
            success=len(events) > 0,
            n_items=len(events),
            latest_date=date,
            error_type="" if len(events) > 0 else "no_events",
            error_message="" if len(events) > 0 else "LLM extraction produced 0 events",
        ))
    except Exception:
        pass
    if len(events) == 0:
        logger.error(
            "LLM extraction (%s) produced 0 events for %s — exiting 1 so cron "
            "wrappers don't paint a green flag over an empty extraction "
            "(consistent with scripts/extract_global_supply_chain_events.py).",
            schema, date,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
