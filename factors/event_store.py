"""Phase 4T — Unified Event Store.

All LLM extractors, news collectors, and policy monitors feed into a single
event store with a unified schema, PIT-safe dating, and deduplication.

5 explicit time fields (added 2026-05-24):
    event_time      — when the event actually happened
    publish_time    — when it was published / disclosed
    available_time  — when the data became consumable (publish + parser lag)
    signal_date     — which trading day it can enter signals
    execution_date  — which trading day it can be traded (T+1 open)

PE-5 (task #144, 2026-06-07) — strict PIT 4-time contract:
    Every event row, on write, MUST have all four of
        publish_time / available_time / signal_date / execution_date
    populated and internally consistent. Writes that violate the contract
    raise ``PITContractError`` (suppressed only when ``strict_pit=False``
    is passed to the EventStore constructor, used by the migration
    helper and a handful of legacy tests).

    Derivation rules (used when a writer omits the derived fields):
      - available_time = publish_time + parser_lag (default: same as publish_time)
      - signal_date    = next business day AFTER available_time (pandas BDay)
      - execution_date = signal_date + 1 BDay  if available_time >= 15:00
                       = signal_date          otherwise

Usage:
    from factors.event_store import EventStore, migrate_legacy_events
    store = EventStore()
    store.add_event({...})
    df = store.query("2026-05-01", "2026-05-22")
    df = store.query_by_signal_date("2026-05-22")
"""

import hashlib
import json
import logging
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "date": str,           # available_date (PIT-safe), YYYY-MM-DD
    "stock_code": str,     # e.g. "600519" or "000001"
    "source": str,         # announcement / news / forum / policy / other
    "event_type": str,     # from EVENT_TYPES
    "direction": int,      # -1, 0, +1
    "confidence": float,   # 0.0 – 1.0
    "summary": str,        # max ~200 chars
}

OPTIONAL_FIELDS = {
    "magnitude": float,          # 0.0 – 1.0
    "affected_industries": list, # e.g. ["计算机", "通信"]
    "horizon_days": int,         # expected impact horizon
    "is_policy": bool,
    "is_regulatory": bool,
    "is_rumor": bool,
    "topic": str,                # e.g. "AI算力", "低空经济"
    "publish_time": str,         # original publish timestamp
}

METADATA_FIELDS = {
    "llm_model": str,
    "prompt_version": str,
    "extract_date": str,
    "source_quality": float,
}

# ---------------------------------------------------------------------------
# 5 explicit time fields (canonical temporal semantics)
# ---------------------------------------------------------------------------
TIME_FIELDS = {
    "event_time": "required",      # when the event actually happened
    "publish_time": "required",    # when it was published/disclosed
    "available_time": "optional",  # when system first sees it
    "signal_date": "required",     # which trading day it can enter signals
    "execution_date": "required",  # which trading day it can be traded
}

EVENT_SCHEMA = {
    "required": REQUIRED_FIELDS,
    "optional": OPTIONAL_FIELDS,
    "metadata": METADATA_FIELDS,
    # Explicit time fields — see TIME_FIELDS
    "event_time": "required",
    "publish_time": "required",
    "available_time": "optional",
    "signal_date": "required",
    "execution_date": "required",
}

# Recognized event types (superset of v1 and v2)
EVENT_TYPES = {
    # Earnings
    "earnings_beat", "earnings_miss", "earnings_inline",
    "earnings_positive", "earnings_negative",
    "revenue_growth", "revenue_decline",
    # Corporate actions
    "order_win", "major_contract",
    "product_launch", "tech_breakthrough",
    "market_share_gain", "market_share_loss",
    "share_buyback", "dividend_increase", "dividend",
    "insider_buy", "insider_sell",
    "share_placement", "share_unlock",
    # Analyst
    "analyst_upgrade", "analyst_downgrade",
    # Regulatory
    "regulatory_approval", "regulatory_penalty",
    "lawsuit_filed", "lawsuit_settled",
    # Management / strategy
    "management_change", "restructuring",
    "strategic_cooperation", "joint_venture",
    "government_subsidy", "tax_benefit",
    "debt_issue", "credit_rating_change",
    # Policy
    "policy_support", "policy_negative",
    # Routine / other
    "routine_announcement", "other",
}

# Mapping legacy direction-like fields
_DIRECTION_MAP = {
    "earnings_positive": 1,
    "earnings_negative": -1,
    "earnings_beat": 1,
    "earnings_miss": -1,
    "revenue_growth": 1,
    "revenue_decline": -1,
    "order_win": 1,
    "major_contract": 1,
    "product_launch": 1,
    "tech_breakthrough": 1,
    "share_buyback": 1,
    "dividend_increase": 1,
    "dividend": 1,
    "insider_buy": 1,
    "insider_sell": -1,
    "share_placement": -1,
    "share_unlock": -1,
    "analyst_upgrade": 1,
    "analyst_downgrade": -1,
    "regulatory_approval": 1,
    "regulatory_penalty": -1,
    "lawsuit_filed": -1,
    "government_subsidy": 1,
    "tax_benefit": 1,
    "policy_support": 1,
    "policy_negative": -1,
}

STORE_DIR = DATA_DIR / "events"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event_hash(event: dict) -> str:
    """Dedup key: hash of (stock_code, event_type, summary[:50], date)."""
    raw = f"{event.get('stock_code', '')}|{event.get('event_type', '')}|" \
          f"{event.get('summary', '')[:50]}|{event.get('date', '')}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _pit_available_date(publish_time_str: str, fallback_date: str) -> str:
    """Compute PIT-safe available_date from publish_time.

    Rule: if published after 15:00, the event is available next calendar day.
    Actual trading-day adjustment is left to downstream consumers.
    """
    if not publish_time_str:
        return fallback_date
    try:
        dt = datetime.strptime(publish_time_str[:19], "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        try:
            dt = datetime.strptime(publish_time_str[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            return fallback_date
    if dt.hour >= 15:
        dt += timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def _compute_signal_date(publish_time_str: str, fallback_date: str) -> str:
    """Compute the first trading day an event can enter signals.

    Rules:
    - If publish_time is after 15:00 on a trading day -> next business day
    - If publish_time is on a weekend/holiday -> next business day
    - Otherwise -> same business day

    Uses pandas BDay (business day) for calendar logic.
    """
    if not publish_time_str:
        # Fallback: treat fallback_date as the publish date at market open
        try:
            ts = pd.Timestamp(fallback_date)
        except (ValueError, TypeError):
            return fallback_date
        # If weekend, roll forward
        if ts.dayofweek >= 5:  # Saturday=5, Sunday=6
            ts = ts + BDay(1)
        return ts.strftime("%Y-%m-%d")

    try:
        dt = datetime.strptime(publish_time_str[:19], "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        try:
            dt = datetime.strptime(publish_time_str[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            return fallback_date
        # Date-only: treat as market open on that day
        ts = pd.Timestamp(dt)
        if ts.dayofweek >= 5:
            ts = ts + BDay(1)
        return ts.strftime("%Y-%m-%d")

    ts = pd.Timestamp(dt)
    # After market close (15:00) or on weekend -> next business day
    if ts.dayofweek >= 5 or dt.hour >= 15:
        ts = ts + BDay(1)
    elif ts.dayofweek < 5:
        # Weekday before 15:00 — same day, but ensure it's a business day
        # (BDay(0) normalises to the same day if already a business day)
        pass

    return ts.normalize().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# PE-5 (task #144) — strict PIT 4-time contract
# ---------------------------------------------------------------------------

class PITContractError(ValueError):
    """Raised when a write violates the PIT 4-time contract.

    See module docstring for the contract. Suppressed only when an
    EventStore was constructed with ``strict_pit=False``.
    """


# Market close in 24h local time. Anything published / available at or
# after this hour is treated as post-close for execution_date purposes.
MARKET_CLOSE_HOUR = 15


def _parse_dt(value: str) -> datetime | None:
    """Lenient datetime parser used by the PIT helpers.

    Accepts ``YYYY-MM-DD``, ``YYYY-MM-DD HH:MM:SS``, ``YYYY-MM-DDTHH:MM:SS``
    (optionally with trailing ``Z`` / fractional / offset). Returns None
    on any failure rather than raising — callers decide how to handle.
    """
    if not value or not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    # First try datetime patterns (need at least 19 chars).
    if len(s) >= 19:
        # Trim to a 19-char ``YYYY-MM-DD?HH:MM:SS`` window; this tolerates
        # trailing fractional seconds / timezone designators like Z or +08:00.
        head = s[:19]
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(head, fmt)
            except (ValueError, TypeError):
                continue
    # Fall back to date-only.
    if len(s) >= 10:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            return None
    return None


def _next_business_day(ts: pd.Timestamp) -> pd.Timestamp:
    """Strict next-business-day: always advance at least one BDay."""
    return (ts.normalize() + BDay(1)).normalize()


def _compute_pit_times(
    publish_time: str,
    *,
    parser_lag_seconds: int = 0,
    available_time_override: str | None = None,
) -> dict[str, str]:
    """Derive (available_time, signal_date, execution_date) from publish_time.

    Rules (see module docstring):
      - available_time = publish_time + parser_lag_seconds
                         (or override when caller supplies one)
      - signal_date    = next business day AFTER available_time
      - execution_date = signal_date + BDay(1) if available_time >= 15:00
                         else signal_date

    Returns a dict with keys ``available_time`` / ``signal_date`` /
    ``execution_date`` (all strings). Raises ``PITContractError`` if
    ``publish_time`` cannot be parsed and no override is supplied — the
    caller is expected to either provide a parseable ``publish_time`` or
    pre-compute the fields.
    """
    pub_dt = _parse_dt(publish_time)
    if pub_dt is None and not available_time_override:
        raise PITContractError(
            f"_compute_pit_times: cannot parse publish_time={publish_time!r}"
        )

    # available_time
    if available_time_override:
        avail_str = available_time_override
        avail_dt = _parse_dt(available_time_override)
        if avail_dt is None:
            # Override is a bare date — treat as midnight that day
            try:
                avail_dt = datetime.strptime(available_time_override[:10], "%Y-%m-%d")
            except (ValueError, TypeError) as e:
                raise PITContractError(
                    f"_compute_pit_times: unparseable available_time_override={available_time_override!r}"
                ) from e
    else:
        # Pad publish_time with parser_lag_seconds
        avail_dt = pub_dt + timedelta(seconds=max(0, int(parser_lag_seconds)))
        avail_str = avail_dt.strftime("%Y-%m-%d %H:%M:%S")

    # signal_date = strict next BDay after available_time (advance >= 1 BDay
    # so even an event available on a Monday morning maps to Tuesday).
    sig_ts = _next_business_day(pd.Timestamp(avail_dt))
    sig_str = sig_ts.strftime("%Y-%m-%d")

    # execution_date depends on whether available_time is post-close
    if avail_dt.hour >= MARKET_CLOSE_HOUR:
        exec_ts = (sig_ts + BDay(1)).normalize()
    else:
        exec_ts = sig_ts
    exec_str = exec_ts.strftime("%Y-%m-%d")

    return {
        "available_time": avail_str,
        "signal_date": sig_str,
        "execution_date": exec_str,
    }


_REQUIRED_PIT_FIELDS = ("publish_time", "available_time", "signal_date", "execution_date")


def _validate_pit_times(event: dict) -> None:
    """Assert the PIT 4-time contract on ``event``.

    Raises ``PITContractError`` if any required field is missing/empty
    or if the date ordering is violated. Returns silently on success.
    """
    missing = [
        f for f in _REQUIRED_PIT_FIELDS
        if not event.get(f) or (isinstance(event.get(f), str) and not event[f].strip())
    ]
    if missing:
        raise PITContractError(
            f"PIT contract: missing/empty field(s) {missing} "
            f"on event stock_code={event.get('stock_code', '?')!r} "
            f"event_type={event.get('event_type', '?')!r}"
        )

    # Consistency: signal_date >= available_time (compare on date portion),
    # execution_date >= signal_date.
    avail_dt = _parse_dt(str(event["available_time"]))
    if avail_dt is None:
        raise PITContractError(
            f"PIT contract: unparseable available_time={event['available_time']!r}"
        )

    try:
        sig_ts = pd.Timestamp(str(event["signal_date"]))
        exec_ts = pd.Timestamp(str(event["execution_date"]))
    except (ValueError, TypeError) as e:
        raise PITContractError(
            f"PIT contract: unparseable signal_date / execution_date "
            f"({event.get('signal_date')!r} / {event.get('execution_date')!r})"
        ) from e

    avail_date = pd.Timestamp(avail_dt.date())
    if sig_ts.normalize() < avail_date:
        raise PITContractError(
            f"PIT contract: signal_date {event['signal_date']!r} < "
            f"available_time {event['available_time']!r} (date portion)"
        )
    if exec_ts.normalize() < sig_ts.normalize():
        raise PITContractError(
            f"PIT contract: execution_date {event['execution_date']!r} < "
            f"signal_date {event['signal_date']!r}"
        )


def _validate_event(event: dict) -> tuple[dict, list[str]]:
    """Validate and normalise an event dict.

    Returns (normalised_event, warnings).  Never raises.
    """
    warnings: list[str] = []
    out: dict[str, Any] = {}

    # Required fields
    for field, typ in REQUIRED_FIELDS.items():
        val = event.get(field)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            warnings.append(f"missing required field '{field}'")
            # Apply sensible defaults so we can still store it
            if typ is str:
                out[field] = ""
            elif typ is int:
                out[field] = 0
            elif typ is float:
                out[field] = 0.0
        else:
            try:
                out[field] = typ(val)
            except (ValueError, TypeError):
                out[field] = val
                warnings.append(f"field '{field}' type mismatch")

    # Clamp confidence
    if "confidence" in out:
        out["confidence"] = max(0.0, min(1.0, float(out["confidence"])))

    # Clamp direction
    if "direction" in out:
        d = int(out["direction"])
        out["direction"] = max(-1, min(1, d))

    # Optional + metadata
    for fields_dict in (OPTIONAL_FIELDS, METADATA_FIELDS):
        for field, typ in fields_dict.items():
            val = event.get(field)
            if val is not None:
                out[field] = val  # keep as-is for flexibility

    # ---- 5 explicit time fields ----
    # event_time: when the event actually happened
    out["event_time"] = event.get("event_time") or event.get("publish_time") or out.get("date", "")

    # publish_time: when it was published/disclosed (already in OPTIONAL_FIELDS)
    if "publish_time" not in out:
        out["publish_time"] = event.get("publish_time", "")

    # PE-5 (task #144): derive available_time / signal_date / execution_date
    # via the canonical _compute_pit_times helper so that every write goes
    # through the same contract. Caller-supplied values still win — the
    # helper only fills gaps. If caller supplied none and publish_time is
    # unparseable, _compute_pit_times raises; we catch and let
    # _validate_pit_times produce the final, contract-shaped error at the
    # write-path entry point.
    pub = (out.get("publish_time") or "").strip()
    avail_override = event.get("available_time")
    try:
        derived = _compute_pit_times(
            pub, available_time_override=avail_override,
        )
    except PITContractError:
        derived = {}

    if event.get("available_time"):
        out["available_time"] = event["available_time"]
    elif derived.get("available_time"):
        out["available_time"] = derived["available_time"]
    else:
        # Last-resort fallback — leave empty so _validate_pit_times rejects.
        out["available_time"] = ""

    if event.get("signal_date"):
        out["signal_date"] = event["signal_date"]
    elif derived.get("signal_date"):
        out["signal_date"] = derived["signal_date"]
    else:
        out["signal_date"] = ""

    if event.get("execution_date"):
        out["execution_date"] = event["execution_date"]
    elif derived.get("execution_date"):
        out["execution_date"] = derived["execution_date"]
    else:
        out["execution_date"] = ""

    # Dedup hash
    out["_hash"] = _event_hash(out)

    return out, warnings


# ---------------------------------------------------------------------------
# EventStore
# ---------------------------------------------------------------------------

class EventStore:
    """Unified event store backed by daily JSONL files."""

    def __init__(
        self,
        store_dir: str | Path | None = None,
        *,
        strict_pit: bool = True,
    ):
        """Create / open an EventStore.

        Args:
            store_dir: directory of daily ``YYYY-MM-DD.jsonl`` files.
                Defaults to ``data/storage/events/``.
            strict_pit: enforce the PE-5 PIT 4-time contract on every
                write. Default True — set False only for the migration
                helper (``scripts/migrate_eventstore_pit_times.py``) or
                legacy regression tests that intentionally exercise
                non-PIT inputs. See module docstring.
        """
        self.store_dir = Path(store_dir) if store_dir else STORE_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.strict_pit = strict_pit
        # In-memory dedup set for current session (per-file dedup also on disk)
        self._seen_hashes: set[str] = set()

    # -- write --

    def _file_for_date(self, date_str: str) -> Path:
        return self.store_dir / f"{date_str}.jsonl"

    def _load_hashes_for_date(self, date_str: str) -> set[str]:
        """Load existing hashes from a daily file to avoid duplicates."""
        fp = self._file_for_date(date_str)
        hashes: set[str] = set()
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        h = obj.get("_hash") or _event_hash(obj)
                        hashes.add(h)
                    except json.JSONDecodeError:
                        continue
        return hashes

    def add_event(self, event: dict) -> bool:
        """Validate and append one event. Returns True if stored (not dup).

        PE-5 (task #144): when ``self.strict_pit`` is True (default),
        the PIT 4-time contract is enforced — a write missing any of
        ``publish_time`` / ``available_time`` / ``signal_date`` /
        ``execution_date`` (or with inconsistent date ordering) raises
        ``PITContractError``. The migration helper passes
        ``strict_pit=False`` because its input is intentionally
        pre-contract JSONL.
        """
        ev, warns = _validate_event(event)
        for w in warns:
            logger.warning("EventStore validation: %s  event=%s", w, event.get("stock_code", "?"))

        # PE-5 PIT 4-time contract — raise on violation so the bad row is
        # never persisted. Callers that intentionally want to write
        # pre-contract rows (migration / legacy backfill) must use
        # ``EventStore(strict_pit=False)``.
        if self.strict_pit:
            _validate_pit_times(ev)

        date_str = ev.get("date", "")
        if not date_str:
            logger.warning("EventStore: skipping event with no date")
            return False

        h = ev["_hash"]
        # Check in-memory cache first
        if h in self._seen_hashes:
            return False
        # Lazy-load on-disk hashes for this date
        cache_key = f"_disk_{date_str}"
        if not hasattr(self, cache_key):
            setattr(self, cache_key, self._load_hashes_for_date(date_str))
        disk_hashes: set = getattr(self, cache_key)
        if h in disk_hashes:
            self._seen_hashes.add(h)
            return False

        # Write
        fp = self._file_for_date(date_str)
        with open(fp, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        self._seen_hashes.add(h)
        disk_hashes.add(h)
        return True

    def add_events(self, events: list[dict]) -> int:
        """Batch add. Returns count of actually stored (non-dup) events."""
        count = 0
        for ev in events:
            if self.add_event(ev):
                count += 1
        return count

    # -- read / query --

    def _read_file(self, fp: Path) -> list[dict]:
        records: list[dict] = []
        if not fp.exists():
            return records
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def query(
        self,
        start_date: str,
        end_date: str,
        stock_code: str | None = None,
        event_type: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Query events within [start_date, end_date] with optional filters."""
        all_records: list[dict] = []
        cur = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        while cur <= end:
            ds = cur.strftime("%Y-%m-%d")
            all_records.extend(self._read_file(self._file_for_date(ds)))
            cur += timedelta(days=1)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)

        if stock_code is not None and "stock_code" in df.columns:
            df = df[df["stock_code"] == stock_code]
        if event_type is not None and "event_type" in df.columns:
            df = df[df["event_type"] == event_type]
        if source is not None and "source" in df.columns:
            df = df[df["source"] == source]

        return df.reset_index(drop=True)

    def query_by_signal_date(
        self,
        signal_date: str,
        stock_code: str | None = None,
        event_type: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Return events whose signal_date matches the given date.

        This is the preferred query method for factors and overlays — it
        returns exactly those events that are actionable on *signal_date*.
        Files are scanned over a 5-day window ending on signal_date to
        catch events filed on the date itself and those that rolled forward
        from earlier calendar days (weekends / after-hours).
        """
        sd = datetime.strptime(signal_date, "%Y-%m-%d")
        all_records: list[dict] = []
        # Scan a window: events may be stored under a calendar date that
        # differs from their computed signal_date (e.g. Friday after-hours
        # event stored under Friday, signal_date = Monday).
        for offset in range(6):  # 0..5 days back
            ds = (sd - timedelta(days=offset)).strftime("%Y-%m-%d")
            all_records.extend(self._read_file(self._file_for_date(ds)))

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        # Filter to matching signal_date
        if "signal_date" in df.columns:
            df = df[df["signal_date"] == signal_date]
        else:
            # Fallback: file has no signal_date yet (pre-migration data)
            df = df[df.get("date", pd.Series(dtype=str)) == signal_date]

        if stock_code is not None and "stock_code" in df.columns:
            df = df[df["stock_code"] == stock_code]
        if event_type is not None and "event_type" in df.columns:
            df = df[df["event_type"] == event_type]
        if source is not None and "source" in df.columns:
            df = df[df["source"] == source]

        return df.drop_duplicates(subset=["_hash"], keep="first").reset_index(drop=True) if "_hash" in df.columns else df.reset_index(drop=True)

    def query_stock(self, stock_code: str, lookback_days: int = 20) -> pd.DataFrame:
        """Recent events for one stock."""
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        return self.query(
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            stock_code=stock_code,
        )

    def compute_event_score(
        self, stock_code: str, date: str, half_life: int = 5, lookback_days: int = 30
    ) -> float:
        """Exponential-decay weighted event score for a stock on a date.

        score = sum_k direction_k * confidence_k * exp(-age_k / half_life)
        """
        dt = datetime.strptime(date, "%Y-%m-%d")
        start = (dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        df = self.query(start, date, stock_code=stock_code)
        if df.empty:
            return 0.0

        score = 0.0
        decay = np.log(2) / half_life
        for _, row in df.iterrows():
            try:
                ev_date = datetime.strptime(str(row.get("date", date)), "%Y-%m-%d")
            except (ValueError, TypeError):
                continue
            age = (dt - ev_date).days
            if age < 0:
                continue
            direction = int(row.get("direction", 0))
            confidence = float(row.get("confidence", 0.5))
            magnitude = float(row.get("magnitude", 0.5))
            score += direction * confidence * magnitude * np.exp(-decay * age)
        return round(score, 4)

    def daily_summary(self, date: str) -> dict:
        """Count events by event_type, source, direction."""
        records = self._read_file(self._file_for_date(date))
        summary: dict[str, dict] = {
            "total": len(records),
            "by_event_type": defaultdict(int),
            "by_source": defaultdict(int),
            "by_direction": defaultdict(int),
        }
        for r in records:
            summary["by_event_type"][r.get("event_type", "unknown")] += 1
            summary["by_source"][r.get("source", "unknown")] += 1
            summary["by_direction"][str(r.get("direction", 0))] += 1
        # Convert defaultdicts to regular dicts for cleaner display
        summary["by_event_type"] = dict(summary["by_event_type"])
        summary["by_source"] = dict(summary["by_source"])
        summary["by_direction"] = dict(summary["by_direction"])
        return summary

    def coverage_report(self, date: str) -> dict:
        """How many stocks have events on this date, broken down by source."""
        records = self._read_file(self._file_for_date(date))
        all_stocks: set[str] = set()
        by_source: dict[str, set[str]] = defaultdict(set)
        for r in records:
            sc = r.get("stock_code", "")
            if sc:
                all_stocks.add(sc)
                by_source[r.get("source", "unknown")].add(sc)
        return {
            "date": date,
            "total_events": len(records),
            "unique_stocks": len(all_stocks),
            "stocks_by_source": {k: len(v) for k, v in sorted(by_source.items())},
        }


# ---------------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------------

LEGACY_DIR = DATA_DIR / "llm_events"


def _convert_legacy_event(raw: dict, file_date: str) -> dict:
    """Convert a legacy llm_events record to the unified schema."""
    # Determine direction from legacy fields
    direction = 0
    if "direction" in raw:
        direction = int(raw["direction"])
    elif "impact_1d" in raw:
        imp = float(raw.get("impact_1d", 0))
        direction = 1 if imp > 0 else (-1 if imp < 0 else 0)

    event_type = raw.get("event_type", "other")
    # If direction is still 0, infer from event_type
    if direction == 0:
        direction = _DIRECTION_MAP.get(event_type, 0)

    # PIT-safe available_date
    publish_time = raw.get("publish_time", "")
    available_date = _pit_available_date(publish_time, file_date)

    # Map source
    source_raw = raw.get("source", "")
    source_category = "news"  # default
    source_lower = source_raw.lower()
    if any(k in source_lower for k in ("公告", "巨潮", "交易所", "上交所", "深交所")):
        source_category = "announcement"
    elif any(k in source_lower for k in ("股吧", "雪球", "forum")):
        source_category = "forum"
    elif any(k in source_lower for k in ("政策", "policy", "国务院", "发改委", "央行")):
        source_category = "policy"

    # Compute magnitude from legacy impact fields
    magnitude = None
    if "impact_1d" in raw and "impact_5d" in raw:
        magnitude = min(1.0, (abs(float(raw["impact_1d"])) + abs(float(raw["impact_5d"]))) / 0.3)
    if "magnitude" in raw:
        magnitude = float(raw["magnitude"])

    # PE-5 (task #144): compute PIT 4-time fields via the canonical helper.
    # Preference order for available_time:
    #   1. raw["available_time"] if present (already PIT-shaped)
    #   2. raw["extract_date"]   (legacy LLM cache provides this)
    #   3. publish_time          (no parser lag known)
    # When publish_time itself is unparseable (truly pre-PIT-era rows)
    # we fall back to file_date so the row still satisfies the contract.
    avail_override = (
        raw.get("available_time")
        or raw.get("extract_date")
        or publish_time
        or file_date
    )
    try:
        pit = _compute_pit_times(
            publish_time or file_date,
            available_time_override=avail_override,
        )
    except PITContractError:
        # Last-resort: use file_date as a midnight available_time so
        # the contract is satisfied. This only fires on truly malformed
        # rows; the migration helper logs + skips when even that fails.
        pit = _compute_pit_times(
            f"{file_date} 00:00:00",
            available_time_override=f"{file_date} 00:00:00",
        )

    unified = {
        "date": available_date,
        "stock_code": raw.get("stock_code", ""),
        "source": source_category,
        "event_type": event_type if event_type in EVENT_TYPES else "other",
        "direction": direction,
        "confidence": max(0.0, min(1.0, float(raw.get("confidence", 0.5)))),
        "summary": str(raw.get("summary", ""))[:200],
        # Optional
        "publish_time": publish_time or f"{file_date} 00:00:00",
        "topic": raw.get("topic", ""),
        # 5 explicit time fields (PE-5)
        "event_time": publish_time or file_date,   # best guess for legacy
        "available_time": pit["available_time"],
        "signal_date": pit["signal_date"],
        "execution_date": pit["execution_date"],
        # Metadata
        "llm_model": raw.get("model_version", ""),
        "prompt_version": raw.get("prompt_version", ""),
        "extract_date": raw.get("extract_date", file_date),
        "source_quality": float(raw.get("source_quality", 0.5)),
        # Preserve original source name
        "source_name": raw.get("source", ""),
        "stock_name": raw.get("stock_name", ""),
    }
    if magnitude is not None:
        unified["magnitude"] = round(magnitude, 3)

    return unified


def migrate_legacy_events(
    legacy_dir: str | Path | None = None,
    store_dir: str | Path | None = None,
) -> int:
    """Read existing llm_events/*.jsonl and write to unified events/ store.

    Returns count of migrated (non-duplicate) events.
    """
    legacy = Path(legacy_dir) if legacy_dir else LEGACY_DIR
    if not legacy.exists():
        logger.info("No legacy events directory at %s", legacy)
        return 0

    store = EventStore(store_dir)
    total = 0

    for fp in sorted(legacy.glob("*.jsonl")):
        file_date = fp.stem  # e.g. "2026-05-22"
        records: list[dict] = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        converted = [_convert_legacy_event(r, file_date) for r in records]
        n = store.add_events(converted)
        total += n
        logger.info("Migrated %s: %d/%d events (new/total)", fp.name, n, len(records))

    logger.info("Migration complete: %d total new events", total)
    return total
