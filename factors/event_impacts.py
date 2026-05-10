"""Structured event impact table.

Converts raw events (announcements, insider trades, earnings, regulatory)
into standardized impact records with decay and confidence.

Schema per event:
  date, target_type (market/sector/stock), target_id,
  source_type, event_type, impact (-1~1), confidence (0~1),
  decay_days, hard_override (avoid/sell/reduce/none)
"""
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "storage"
EVENT_DIR = DATA_DIR / "event_impacts"


@dataclass
class EventImpact:
    """Single structured event impact record."""
    date: str                           # YYYY-MM-DD
    target_type: str                    # market / sector / stock
    target_id: str                      # e.g. "SH600519" or "科创板" or "全市场"
    source_type: str                    # announcement / insider / earnings / regulatory / news
    event_type: str                     # e.g. "减持", "业绩预增", "ST", "监管处罚"
    impact: float                       # -1 (very negative) to +1 (very positive)
    confidence: float                   # 0 to 1
    decay_days: int                     # how many days the impact lasts
    hard_override: str = "none"         # avoid / sell / reduce / none
    description: str = ""               # human-readable summary
    effective_date: str = ""            # when this event becomes usable (disclosure date + 1)


class EventImpactStore:
    """Store and query structured event impacts."""

    def __init__(self, event_dir: Path = EVENT_DIR):
        self.event_dir = event_dir
        self.event_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.event_dir / "events.json"
        self._events: list = self._load()

    def _load(self) -> list:
        if self.events_file.exists():
            try:
                return json.loads(self.events_file.read_text())
            except Exception:
                return []
        return []

    def _save(self):
        # Keep last 1000 events
        self._events = self._events[-1000:]
        tmp = self.events_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._events, ensure_ascii=False, indent=2))
        os.replace(tmp, self.events_file)

    def add(self, event: EventImpact):
        """Add a structured event."""
        if not event.effective_date:
            event.effective_date = event.date
        self._events.append(asdict(event))
        self._save()
        logger.info(f"Event added: {event.event_type} on {event.target_id} impact={event.impact:+.2f}")

    def add_batch(self, events: list):
        """Add multiple events."""
        for e in events:
            if not e.effective_date:
                e.effective_date = e.date
            self._events.append(asdict(e))
        self._save()
        logger.info(f"Added {len(events)} events")

    def get_active(self, as_of: str = None, target_id: str = None) -> list:
        """Get events still active (within decay window) as of a date.

        Args:
            as_of: YYYY-MM-DD, defaults to today
            target_id: filter by stock/sector/market code
        """
        if not as_of:
            as_of = datetime.now().strftime("%Y-%m-%d")

        active = []
        for e in self._events:
            # Check if within decay window
            event_date = e.get("effective_date", e.get("date", ""))
            if not event_date:
                continue
            if event_date > as_of:
                continue

            from datetime import datetime as dt, timedelta
            try:
                ed = dt.strptime(event_date, "%Y-%m-%d")
                expire = ed + timedelta(days=e.get("decay_days", 5))
                if dt.strptime(as_of, "%Y-%m-%d") > expire:
                    continue
            except ValueError:
                continue

            if target_id and e.get("target_id") != target_id:
                if e.get("target_type") != "market":  # market events apply to all
                    continue

            active.append(e)

        return active

    def get_hard_overrides(self, as_of: str = None) -> list:
        """Get stocks with hard override (avoid/sell/reduce) active."""
        active = self.get_active(as_of)
        return [e for e in active if e.get("hard_override", "none") != "none"]

    def get_stock_impact(self, code: str, as_of: str = None) -> float:
        """Get aggregate impact score for a stock."""
        active = self.get_active(as_of, target_id=code)
        if not active:
            return 0.0

        # Weighted average by confidence, with decay
        total_weight = 0
        total_impact = 0
        for e in active:
            weight = e.get("confidence", 0.5)
            total_impact += e.get("impact", 0) * weight
            total_weight += weight

        return total_impact / (total_weight + 1e-8)


# ── Pre-defined event templates ──────────────────────────────────────

def create_insider_sell_event(code: str, name: str, amount: float, date: str) -> EventImpact:
    """Create event for insider/major shareholder selling."""
    severity = min(abs(amount) / 1e8, 1.0)  # normalize by 亿
    return EventImpact(
        date=date, target_type="stock", target_id=code,
        source_type="insider", event_type="大股东减持",
        impact=-0.3 * severity, confidence=0.8, decay_days=10,
        hard_override="reduce" if severity > 0.5 else "none",
        description=f"{name} 大股东减持 {amount/1e8:.1f}亿",
        effective_date=date,
    )


def create_earnings_event(code: str, name: str, growth_pct: float, date: str) -> EventImpact:
    """Create event for earnings announcement."""
    if growth_pct > 50:
        impact, override = 0.5, "none"
    elif growth_pct > 0:
        impact, override = 0.2, "none"
    elif growth_pct > -30:
        impact, override = -0.2, "none"
    else:
        impact, override = -0.5, "reduce"

    return EventImpact(
        date=date, target_type="stock", target_id=code,
        source_type="earnings", event_type="业绩公告",
        impact=impact, confidence=0.9, decay_days=20,
        hard_override=override,
        description=f"{name} 业绩同比{growth_pct:+.0f}%",
        effective_date=date,
    )


def create_st_event(code: str, name: str, date: str) -> EventImpact:
    """Create event for ST/delisting risk."""
    return EventImpact(
        date=date, target_type="stock", target_id=code,
        source_type="regulatory", event_type="ST风险警示",
        impact=-0.8, confidence=1.0, decay_days=60,
        hard_override="avoid",
        description=f"{name} 被实施ST风险警示",
        effective_date=date,
    )


def create_policy_event(sector: str, direction: float, description: str, date: str) -> EventImpact:
    """Create event for policy/regulatory impact on a sector."""
    return EventImpact(
        date=date, target_type="sector", target_id=sector,
        source_type="news", event_type="政策事件",
        impact=direction, confidence=0.6, decay_days=10,
        hard_override="none",
        description=description,
        effective_date=date,
    )
