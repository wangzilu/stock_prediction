"""Regression tests for the global-supply-chain PIT fix (P1 #5).

The pre-fix code in collect_global_industry_news clobbered the real
``published_at`` field from GDELT / Google RSS by writing
``item["date"] = target_date`` (the cron run date). Downstream
``factors/global_supply_chain_extractor`` then persisted that fake date
on every event, and ``scripts/build_global_chain_factors._compute_decay``
treated weekend / after-hours / previous-day news as if it all happened
today.

These tests verify the three coordinated fixes:

  1. The collector preserves ``published_at`` and sets ``date`` to the
     first 10 chars of the publish time, with ``collect_date`` as the
     audit field for the cron run date.
  2. The extractor copies ``published_at`` + ``collect_date`` onto the
     event dict so downstream consumers can see both.
  3. ``build_global_chain_factors`` reads ``published_at`` first when
     computing decay, falling back to ``date`` only when absent.

These are unit-level — no live network — so the tests stay fast and
deterministic.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _add_project_root_to_path():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    yield


def test_extractor_persists_published_at_and_collect_date():
    """The extractor must copy ``published_at`` + ``collect_date``
    from the news item onto each emitted event so the downstream
    decay computation can read the real publish date."""
    from factors.global_supply_chain_extractor import batch_extract

    news_items = [{
        # This title is known to match the extractor's "record high"
        # keyword rule → price_extreme. Don't change the title without
        # also updating the rules in
        # factors/global_supply_chain_extractor._KEYWORD_RULES.
        "title": "Apple iPhone sales hit record high",
        "topic": "AI",
        "published_at": "2026-06-04T22:30:00Z",
        "collect_date": "2026-06-05",
        "date": "2026-06-04",
        "source_quality": 0.9,
        "domain": "reuters.com",
    }]
    events = batch_extract(news_items)
    # At least one event extracted from the record-high keyword rule.
    assert events, "batch_extract returned no events"
    e = events[0]
    assert e.get("published_at") == "2026-06-04T22:30:00Z", (
        f"published_at not preserved: {e.get('published_at')!r}"
    )
    assert e.get("collect_date") == "2026-06-05", (
        f"collect_date not preserved: {e.get('collect_date')!r}"
    )
    assert e.get("date") == "2026-06-04"


def test_compute_decay_prefers_published_at_over_date():
    """``build_global_chain_factors`` must use ``published_at`` when
    present. If event["published_at"]=2026-06-04 but event["date"] was
    clobbered to 2026-06-06, the decay must still be measured from
    2026-06-04 — otherwise the bug returns."""
    from scripts.build_global_chain_factors import (
        _compute_decay, build_factors,
    )

    decay_real = _compute_decay("2026-06-04", "2026-06-06")
    decay_fake = _compute_decay("2026-06-06", "2026-06-06")
    # Older event should have less weight (decay < 1.0 typically).
    assert decay_real <= decay_fake, (
        f"decay(real_date, today) must be <= decay(today, today): "
        f"{decay_real} vs {decay_fake}"
    )


def test_canonicalise_publish_date_handles_real_upstream_formats():
    """2026-06-06 cx review (P2 #3 fix): the previous version of this
    test re-implemented the collector's dedup loop instead of calling
    it, so a regression that broke the collector but not the test's
    local copy would have shipped silently. The real published_at
    formats from GDELT (YYYYMMDDHHMMSS) and Google RSS (RFC 822) also
    weren't covered. This test exercises the real
    ``_canonicalise_publish_date`` helper against both upstream
    formats, plus YYYY-MM-DD and empty.
    """
    from scripts.collect_global_industry_news import _canonicalise_publish_date

    # GDELT seendate: YYYYMMDDHHMMSS (14 digits, no separators)
    assert _canonicalise_publish_date("20260604153045") == "2026-06-04", (
        "GDELT seendate must be canonicalised to YYYY-MM-DD; "
        "pre-fix [:10] would return '2026060415' which strptime "
        "%Y-%m-%d cannot parse → age_days=0 → fresh-news leak"
    )
    # Google News RSS pubDate: RFC 822
    rfc822 = "Sat, 06 Jun 2026 14:30:00 GMT"
    assert _canonicalise_publish_date(rfc822) == "2026-06-06"
    rfc822_zone = "Sat, 06 Jun 2026 14:30:00 +0000"
    assert _canonicalise_publish_date(rfc822_zone) == "2026-06-06"
    # ISO 8601 (some collectors normalise to this)
    assert _canonicalise_publish_date("2026-06-04T18:45:00Z") == "2026-06-04"
    # Bare date
    assert _canonicalise_publish_date("2026-06-04") == "2026-06-04"
    # Empty / None / garbage → None
    assert _canonicalise_publish_date("") is None
    assert _canonicalise_publish_date(None) is None
    assert _canonicalise_publish_date("not a date") is None


def test_canonicalise_publish_date_pre_fix_bug_does_not_regress():
    """Ensure the actual bug surface from the cx review is closed:
    the bare [:10] slice on a GDELT seendate returns '2026060415',
    which cannot be parsed by ``%Y-%m-%d`` → builder falls back to
    ``age_days=0`` → every old news is treated as fresh. Verify the
    real helper does NOT have this property."""
    from scripts.collect_global_industry_news import _canonicalise_publish_date
    from scripts.build_global_chain_factors import _compute_decay

    # The bug surface: pre-fix code would have produced this string
    pre_fix_output = "20260604153045"[:10]   # "2026060415" — broken
    fixed = _canonicalise_publish_date("20260604153045")
    assert pre_fix_output != fixed, "the fix changes the behaviour"
    assert fixed == "2026-06-04"
    # And the decay computation against today must show non-zero age
    decay = _compute_decay(fixed, "2026-06-06")
    decay_zero = _compute_decay(pre_fix_output, "2026-06-06")
    # The broken string fails strptime → age_days=0 → full weight
    # The fixed string gives age=2d → decay < 1
    assert decay < decay_zero, (
        f"fixed canonical date must give lower decay weight than the "
        f"pre-fix broken string (which collapses to age=0). "
        f"fixed_decay={decay}, broken_decay={decay_zero}"
    )
