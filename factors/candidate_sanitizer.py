"""Unified candidate sanitizer for production inference paths.

Background — 2026-05-29 incident: ST stocks (ST海王/000078, *ST美丽/000010,
*ST海华/600243) showed up in the morning push despite the training universe
filter (build_tradable_mask) excluding them. Root cause: the inference path
had no equivalent filter; the model emits scores for every stock in the Qlib
dataset and the top-N selector took them at face value.

A patch was applied to scheduler/jobs.py to add an _is_st_or_excluded helper,
but only to the morning candidate-builders. 14:30 _build_intraday_buy_candidates,
22:00 _build_evening_stock_forecasts, the index top_bullish/top_bearish list,
the paper OMS _load_and_filter_predictions path, and the shadow overlays all
had different (or no) filters.

This module is the SINGLE source of truth for "is this stock eligible for a
recommendation/paper trade right now". Every entry point that turns a model
score into a user-visible candidate or a paper-trade order MUST go through it.

Rules enforced:
  1. ST / *ST / 退市整理 — by name pattern OR by st_stock_list.json membership
  2. 北交所 (BJ / 4xx / 8xx / 9xx) — code prefix
  3. Invalid price (≤ 0 or NaN)
  4. Suspended trading today (volume == 0 / NaN / quote missing)
  5. 一字板 (high == low this session — can't actually transact at target price)
  6. Low liquidity (volume below configurable floor)
  7. Stale prediction (prediction date older than freshness window)

Rules NOT enforced (yet — listing-date data isn't in a single location):
  - New IPO < 60 days (tracked in build_tradable_mask; expose via daily file in next iteration)

Fail-closed semantics: if st_stock_list.json fails to load AND the caller can't
provide a name string, the sanitizer treats the candidate as REJECTED. This is
deliberately stricter than the research-time behavior (which was fail-open) per
王总 2026-05-29 feedback — recommendation push has real money implications.

Usage:
    from factors.candidate_sanitizer import CandidateSanitizer

    sanitizer = CandidateSanitizer(today="2026-05-29")
    for code, score in lgb_preds.items():
        ok, reason = sanitizer.check(code, name, quote=spot.get(code))
        if not ok:
            continue
        # build candidate ...
    sanitizer.log_summary(logger, label="14:30 intraday")
"""
import json
import logging
import re
import threading
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Default thresholds (override via CandidateSanitizer kwargs)
DEFAULT_MIN_VOLUME = 100_000.0       # shares — lower than build_tradable_mask's 一字板 check but kept conservative
DEFAULT_MAX_PREDICTION_AGE = 3       # trading days; older than this rejects the candidate

_ST_NAME_RE = re.compile(r"^\s*(\*?ST|退)")


class CandidateSanitizer:
    """Stateful per-pipeline-call sanitizer with reason accounting."""

    def __init__(
        self,
        *,
        today: str | None = None,
        st_list_path: Path | None = None,
        min_volume: float = DEFAULT_MIN_VOLUME,
        max_prediction_age_days: int = DEFAULT_MAX_PREDICTION_AGE,
        allow_bse: bool = False,
        require_quote: bool = True,
        min_listing_days: int = 60,
        crash_probs: dict | None = None,
        crash_threshold: float = 0.65,
        chain_alpha: dict | None = None,
        min_chain_alpha: float = -2.0,
        cooldown_set: set | None = None,
    ):
        self.today = today or datetime.now().strftime("%Y-%m-%d")
        self.min_volume = float(min_volume)
        self.max_prediction_age_days = int(max_prediction_age_days)
        self.allow_bse = bool(allow_bse)
        self.require_quote = bool(require_quote)
        self.min_listing_days = int(min_listing_days)
        # Crash probability hard block (mini-RiskGuard for recommendation path).
        # crash_probs is {code_upper: prob}; codes with prob >= threshold are
        # rejected even before the LGB ranking + sanitizer chain. Threshold
        # 0.65 matches OMS RiskGuard's "cannot_buy" tier.
        self.crash_probs = {k.upper(): float(v) for k, v in (crash_probs or {}).items()}
        self.crash_threshold = float(crash_threshold)
        # Supply-chain alpha negative-event filter (matches RiskGuard's
        # _check_supply_chain_risk): alpha < -2.0 = pending_exit reason.
        # For recommendation: never start a position in a "supply chain
        # negative" stock even before any open position triggers force-sell.
        self.chain_alpha = {k.upper(): float(v) for k, v in (chain_alpha or {}).items()}
        self.min_chain_alpha = float(min_chain_alpha)
        # Codes currently in RiskGuard cooldown (from risk_guard_state.json).
        # Cooldown rules: stop-loss=10 calendar days, event=30, ST=until clear.
        # The set is precomputed for today's date by the caller so the
        # sanitizer doesn't need to read state files itself.
        self.cooldown_set = set(c.upper() for c in (cooldown_set or set()))
        self._st_list_path = st_list_path
        self._st_set: set[str] | None = None
        self._st_load_failed = False
        self._first_dates: dict[str, str] | None = None
        self._first_dates_load_attempted = False
        self._stats = Counter()
        self._lock = threading.Lock()

    # -- ST list loading ------------------------------------------------------

    def _load_st_set(self) -> None:
        """Load st_stock_list.json. On failure leaves _st_set=None and sets
        _st_load_failed=True so the name-only path can still be used while
        list-only candidates (no name available) are fail-closed-rejected."""
        if self._st_set is not None or self._st_load_failed:
            return
        path = self._st_list_path
        if path is None:
            try:
                from config.settings import DATA_DIR
                path = DATA_DIR / "st_stock_list.json"
            except Exception:
                self._st_load_failed = True
                logger.error("CandidateSanitizer: DATA_DIR unavailable, ST list cannot load")
                return
        if not path.exists():
            self._st_load_failed = True
            logger.error("CandidateSanitizer: ST list missing at %s — name-only filter only", path)
            return
        try:
            raw = json.loads(path.read_text())
            self._st_set = set(str(c).upper() for c in raw)
        except Exception as e:
            self._st_load_failed = True
            logger.error("CandidateSanitizer: ST list load failed (%s) — name-only filter only", e)

    def _load_first_dates(self) -> None:
        """Load instrument first_date from qlib all.txt.

        first_date in qlib is the earliest available data date; for stocks
        IPO'd after the qlib history-start (~2020-09-29), this matches the
        actual listing date and lets us enforce the IPO<60-day filter that
        build_tradable_mask applies at training time. Older listings all
        show ~2020-09-29 first_date so they easily pass the floor.
        """
        if self._first_dates is not None or self._first_dates_load_attempted:
            return
        self._first_dates_load_attempted = True
        try:
            from config.settings import DATA_DIR
            inst_file = DATA_DIR / "qlib_data" / "cn_data" / "instruments" / "all.txt"
            if not inst_file.exists():
                logger.warning("CandidateSanitizer: qlib instruments file missing — IPO filter disabled")
                self._first_dates = {}
                return
            d: dict[str, str] = {}
            for line in inst_file.read_text().splitlines():
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    d[parts[0].upper()] = parts[1]
            self._first_dates = d
        except Exception as e:
            logger.warning("CandidateSanitizer: failed to load qlib instruments (%s) — IPO filter disabled", e)
            self._first_dates = {}

    def _is_new_listing(self, code: str) -> bool | None:
        """True if listed less than min_listing_days ago, None if data unknown."""
        if self.min_listing_days <= 0:
            return False
        self._load_first_dates()
        first = (self._first_dates or {}).get(code.upper())
        if not first:
            return None
        try:
            first_dt = datetime.strptime(first[:10], "%Y-%m-%d")
            today_dt = datetime.strptime(self.today[:10], "%Y-%m-%d")
            return (today_dt - first_dt).days < self.min_listing_days
        except (ValueError, TypeError):
            return None

    # -- per-rule helpers -----------------------------------------------------

    def _is_st_by_name(self, name: str) -> bool:
        if not name:
            return False
        return bool(_ST_NAME_RE.search(name.strip()))

    def _is_st_by_code(self, code: str) -> bool | None:
        """Returns True/False if ST list usable, None if list failed to load
        and we have no signal from this path."""
        if self._st_set is None:
            self._load_st_set()
        if self._st_load_failed and self._st_set is None:
            return None
        return code.upper() in (self._st_set or set())

    def _is_bse(self, code: str) -> bool:
        c = code.upper().lstrip()
        if c.startswith("BJ"):
            return True
        # Bare numeric codes (BJ has no SH/SZ prefix in some upstream feeds)
        numeric = c[-6:] if len(c) >= 6 else c
        if numeric.isdigit() and numeric[0] in {"4", "8", "9"}:
            # Filter out STAR (688xxx) — those are sh-prefixed in canonical form
            if c.startswith(("SH", "SZ")):
                return False
            return True
        return False

    # -- main entry -----------------------------------------------------------

    def check(
        self,
        code: str,
        name: str | None = None,
        *,
        quote: dict | None = None,
        prediction_date: str | None = None,
    ) -> tuple[bool, str | None]:
        """Return (passed, reject_reason). reject_reason is None on pass.

        Args:
            code: stock code, any casing, may or may not have SH/SZ/BJ prefix
            name: Chinese name (may be empty if quote not available)
            quote: optional dict with akshare-style fields (最新价, 涨跌幅, 成交量,
                最高, 最低). When None, suspended / 一字板 / low-volume rules are
                skipped (caller is asserting they don't need them — e.g., paper
                OMS prediction load before market open).
            prediction_date: YYYY-MM-DD; if set, rejected if older than
                max_prediction_age_days trading days vs self.today.
        """
        with self._lock:
            return self._check_locked(code, name, quote, prediction_date)

    def _check_locked(self, code, name, quote, prediction_date):
        # 1. ST / 退市
        if name and self._is_st_by_name(name):
            self._stats["st_by_name"] += 1
            return False, "st_by_name"
        st_by_code = self._is_st_by_code(code)
        if st_by_code is True:
            self._stats["st_by_list"] += 1
            return False, "st_by_list"
        if st_by_code is None and not name:
            # Fail-closed: ST list failed AND no name to check
            self._stats["st_unknown_failclosed"] += 1
            return False, "st_unknown_failclosed"

        # 2. BJ / 北交所
        if not self.allow_bse and self._is_bse(code):
            self._stats["bse"] += 1
            return False, "bse"

        # 2.5. New listing (IPO < min_listing_days, default 60)
        new_listing = self._is_new_listing(code)
        if new_listing is True:
            self._stats["new_listing"] += 1
            return False, "new_listing"

        # 2.6. Crash probability hard block
        if self.crash_probs:
            cp = self.crash_probs.get(code.upper())
            if cp is not None and cp >= self.crash_threshold:
                self._stats["high_crash_prob"] += 1
                return False, "high_crash_prob"

        # 2.7. Supply-chain alpha hard block (negative event = don't open)
        if self.chain_alpha:
            ca = self.chain_alpha.get(code.upper())
            if ca is not None and ca < self.min_chain_alpha:
                self._stats["chain_negative"] += 1
                return False, "chain_negative"

        # 2.8. RiskGuard cooldown (stop-loss / event / ST cooldown still active)
        if self.cooldown_set and code.upper() in self.cooldown_set:
            self._stats["in_cooldown"] += 1
            return False, "in_cooldown"

        # 3-6. quote-dependent rules
        if quote is None:
            if self.require_quote:
                self._stats["no_quote"] += 1
                return False, "no_quote"
        else:
            price = self._finite(quote.get("最新价", quote.get("price")))
            high = self._finite(quote.get("最高", quote.get("high")))
            low = self._finite(quote.get("最低", quote.get("low")))
            volume = self._finite(quote.get("成交量", quote.get("volume")))

            if price <= 0:
                self._stats["invalid_price"] += 1
                return False, "invalid_price"
            if volume <= 0:
                self._stats["suspended"] += 1
                return False, "suspended"
            if high > 0 and low > 0 and abs(high - low) < 1e-9:
                self._stats["yizi_ban"] += 1
                return False, "yizi_ban"
            if volume < self.min_volume:
                self._stats["low_volume"] += 1
                return False, "low_volume"

        # 7. stale prediction
        if prediction_date:
            try:
                pred_dt = datetime.strptime(prediction_date[:10], "%Y-%m-%d")
                today_dt = datetime.strptime(self.today[:10], "%Y-%m-%d")
                # Calendar-day age (conservative; trading-day calc would be tighter)
                age = (today_dt - pred_dt).days
                if age > self.max_prediction_age_days:
                    self._stats["stale_prediction"] += 1
                    return False, "stale_prediction"
            except Exception:
                self._stats["bad_prediction_date"] += 1
                return False, "bad_prediction_date"

        self._stats["passed"] += 1
        return True, None

    # -- introspection --------------------------------------------------------

    @staticmethod
    def _finite(v) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        if f != f:  # NaN
            return 0.0
        return f

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def log_summary(self, log=logger, label: str = "") -> None:
        s = self.stats()
        if not s:
            return
        passed = s.get("passed", 0)
        rejected = sum(v for k, v in s.items() if k != "passed")
        prefix = f"CandidateSanitizer[{label}]" if label else "CandidateSanitizer"
        log.info("%s: passed=%d rejected=%d reasons=%s",
                 prefix, passed, rejected,
                 {k: v for k, v in s.items() if k != "passed" and v})
