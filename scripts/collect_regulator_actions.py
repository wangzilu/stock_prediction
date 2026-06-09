"""SPIKE — Collect 证监会 / 沪深交易所 行政处罚 + 自律监管 + 问询函 raw documents.

Channel 4 (regulator penalty + inquiry) of the LLM event pipeline.

Why a dedicated collector
-------------------------
The existing daily_news / event collectors lump regulator actions with
generic announcements. Karpoff-Lott (1993, JLE) and Bhagat-Bizjak (2019)
both show -1.5%~-3% CAR in the [-1, +5] window after enforcement
announcement, with the strongest negative drift on penalties involving
financial misrepresentation and resignation referrals. We therefore
need:

1. **Source-level isolation** so a regulator action is never silently
   demoted to ``routine_announcement`` by the generic LLM extractor's
   keyword gate (factors/event_schema_validator.py).
2. **Stable typed schema** (severity / regulator / topic / fine_amount)
   so RiskGuard's L1.6 + L3 overlays can consume a single column per
   factor without re-parsing free text.
3. **Same-day signal alignment**: regulator filings show up on the
   CSRC / SSE / SZSE portals BEFORE most news terminals, so a dedicated
   poller catches the event same-day, while the generic news collector
   lags by 1-2 sessions.

Data sources (verified 2026-06-09)
----------------------------------
* AKShare keyword probe returned NO direct regulator API:
  ``[m for m in dir(akshare) if 'csrc' in m.lower() or 'punish' in m.lower()
  or 'penalty' in m.lower() or 'inquiry' in m.lower() ...] == []``.
  The only proxies are ``stock_notice_report`` and
  ``stock_zh_a_disclosure_report_cninfo`` — both are FILING dumps that
  include penalty/inquiry receipts mixed with all other filings. Usable
  as a pre-filter but not as the primary source.

* Primary scrape URLs (per task brief):
    - CSRC 行政处罚决定书:
        http://www.csrc.gov.cn/csrc/c100120/zfxxgkml.shtml
    - SSE 自律监管措施:
        http://www.sse.com.cn/disclosure/credibility/supervision/measures/
    - SZSE 自律监管措施:
        http://www.szse.cn/disclosure/supervision/measure/index.html
    - SSE 问询函:
        http://www.sse.com.cn/disclosure/credibility/supervision/inquiries/
    - SZSE 问询函:
        http://www.szse.cn/disclosure/listed/supervision/inquire/index.html

Output
------
``data/storage/regulator_actions/<YYYY-MM-DD>.jsonl``

Each row carries the RAW document (title + body + url + filed_date) so
the LLM extractor can normalise it later. PIT contract: ``event_date``
is the date the regulator filed the document, NOT the date we crawled
it. The LLM extractor's lag (+1 BDay) is applied at factor-build time.

SPIKE caveats (DO NOT ship before addressing)
---------------------------------------------
1. CSRC portal is rendered via ASP/JSP with anti-scrape (User-Agent
   filter + per-IP rate limit + session cookie). Empirical 2025 traffic
   suggests ≤30 GET/min sustained. Throttle ``INTER_REQUEST_DELAY = 2.0``.
2. SSE/SZSE inquiry pages are paginated and tied to ts_code via a
   list view — link harvest must extract ``stock_code`` from the row
   metadata, not from the document title (titles are template-style
   "关于XXXX股份有限公司的监管工作函" which the LLM would have to
   guess at).
3. Backfill window — CSRC only keeps ~3 years of penalty PDFs online,
   exchanges keep ~5 years of inquiries. A historical IC estimate
   needs a one-off snapshot from CSMAR / Wind, NOT a live scrape.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_DIR = DATA_DIR / "regulator_actions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-source URL anchors. The actual list/detail pages are JS-rendered
# on some; the fetcher is structured so each source has a per-source
# adapter so a future swap to a headless renderer is a one-file edit.
SOURCES: dict[str, dict[str, str]] = {
    "csrc_penalty": {
        "list_url": "http://www.csrc.gov.cn/csrc/c100120/zfxxgkml.shtml",
        "regulator": "CSRC",
        "doc_type": "penalty",
    },
    "sse_supervision": {
        "list_url": (
            "http://www.sse.com.cn/disclosure/credibility/supervision/measures/"
        ),
        "regulator": "SSE",
        "doc_type": "supervision",
    },
    "szse_supervision": {
        "list_url": "http://www.szse.cn/disclosure/supervision/measure/index.html",
        "regulator": "SZSE",
        "doc_type": "supervision",
    },
    "sse_inquiry": {
        "list_url": (
            "http://www.sse.com.cn/disclosure/credibility/supervision/inquiries/"
        ),
        "regulator": "SSE",
        "doc_type": "inquiry",
    },
    "szse_inquiry": {
        "list_url": (
            "http://www.szse.cn/disclosure/listed/supervision/inquire/index.html"
        ),
        "regulator": "SZSE",
        "doc_type": "inquiry",
    },
}

# Polite-throttle for gov-CN portals. Empirically the CSRC portal will
# rate-limit a single IP at ≈30 req/min sustained; we use a 2s delay
# baseline. Exchanges are more lenient (≈60 req/min) but we keep the
# same delay for code simplicity.
INTER_REQUEST_DELAY = 2.0

# Recency cutoff for "today's" run: documents filed within this many
# days of target_date are kept. Matches collect_daily_news.py's 7-day
# bound but slightly looser because CSRC sometimes publishes a 2-3 day
# stale batch on Friday afternoon.
RECENCY_DAYS = 10


@dataclass
class RegulatorDoc:
    """One raw regulator document, before LLM extraction.

    All free-text fields are passed through verbatim — the LLM (in
    ``factors/regulator_penalty_extractor.py``) is the only place that
    is allowed to interpret severity / topic / fine amount.
    """
    source_key: str            # csrc_penalty / sse_supervision / ...
    regulator: str             # CSRC / SSE / SZSE
    doc_type: str              # penalty / supervision / inquiry
    filed_date: str            # YYYY-MM-DD, when regulator filed it
    event_date: str            # YYYY-MM-DD, same as filed_date by default
                                 # (SPIKE NOTE: detail page may carry an
                                 # earlier "事实发生日" — TODO L2)
    title: str                 # e.g. 关于对XX股份有限公司的监管工作函
    body: str                  # ≤5000 char body text (after HTML strip)
    url: str                   # canonical document URL
    ts_code: str = ""          # 6-digit if extractable from list page,
                                 # else empty — LLM may infer from body

    def to_dict(self) -> dict:
        return {
            "source_key": self.source_key,
            "regulator": self.regulator,
            "doc_type": self.doc_type,
            "filed_date": self.filed_date,
            "event_date": self.event_date,
            "title": self.title,
            "body": self.body[:5000],
            "url": self.url,
            "ts_code": self.ts_code,
        }


# ---------------------------------------------------------------------
# Per-source fetchers — SKELETON ONLY (raise NotImplementedError).
# Real implementation requires:
#   1. requests session with rotating UA + cookie jar
#   2. lxml/BeautifulSoup link harvest (see scripts/collect_policy_texts.py
#      for the gov.cn pattern — port that here once we land on the same
#      URL family).
#   3. PDF-to-text via pdfplumber for CSRC penalty PDFs
# ---------------------------------------------------------------------
def fetch_csrc_penalty(target_date: str) -> list[RegulatorDoc]:
    """List CSRC penalty decisions filed on or near target_date.

    SPIKE: structure follows scripts/collect_policy_texts.py's
    ``_harvest_links`` pattern. The detail pages are PDF (penalty
    decision书) — we will need ``pdfplumber`` for body extraction.
    Title + filed_date can be read from the list-page TR row directly.

    TODO L1: implement after the spike is approved.
    """
    raise NotImplementedError("SPIKE — implement after approval")


def fetch_exchange_action(source_key: str, target_date: str) -> list[RegulatorDoc]:
    """SSE / SZSE supervision OR inquiry list pages.

    These four list pages share the same row schema (ts_code, 公司名称,
    标题, 文件链接, 公告日期) which is why a single fetcher routes by
    source_key. The detail link is HTML for SSE inquiries but PDF for
    SZSE supervision letters — handle in the body extractor.

    TODO L1: implement after the spike is approved.
    """
    raise NotImplementedError("SPIKE — implement after approval")


SOURCE_FETCHERS = {
    "csrc_penalty": fetch_csrc_penalty,
    "sse_supervision": lambda d: fetch_exchange_action("sse_supervision", d),
    "szse_supervision": lambda d: fetch_exchange_action("szse_supervision", d),
    "sse_inquiry": lambda d: fetch_exchange_action("sse_inquiry", d),
    "szse_inquiry": lambda d: fetch_exchange_action("szse_inquiry", d),
}


# ---------------------------------------------------------------------
# IO + freshness — mirrors collect_daily_news.py's manifest pattern so
# the cron skip-check can verify the file was produced by THIS version
# of the collector (not a pre-spike stub).
# ---------------------------------------------------------------------
COLLECTOR_VERSION = 0  # SPIKE — bump to 1 once fetchers land


def _atomic_write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def collect_regulator_actions(
    target_date: str | None = None,
    sources: list[str] | None = None,
) -> Path:
    """Top-level entrypoint. Writes one JSONL per target_date.

    Args:
        target_date: YYYY-MM-DD, default today. The output file is
            ``regulator_actions/<target_date>.jsonl``.
        sources: subset of ``SOURCES`` keys to run; default = all five.

    Returns:
        Path to the written JSONL (may be empty if no docs match).
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")
    sources = sources or list(SOURCES.keys())
    output_path = OUTPUT_DIR / f"{target_date}.jsonl"
    manifest_path = OUTPUT_DIR / f"{target_date}.manifest.json"

    # Skip rule with manifest version check — mirror collect_daily_news.
    if output_path.exists() and manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            if int(m.get("collector_version", -1)) == COLLECTOR_VERSION:
                logger.info(
                    "regulator_actions already collected for %s "
                    "(collector_version=%d), skipping",
                    target_date, COLLECTOR_VERSION,
                )
                return output_path
        except Exception:
            pass

    all_docs: list[RegulatorDoc] = []
    for src in sources:
        fetcher = SOURCE_FETCHERS.get(src)
        if fetcher is None:
            logger.warning("Unknown source %s, skipping", src)
            continue
        try:
            docs = fetcher(target_date)
            all_docs.extend(docs)
            logger.info("[%s] fetched %d docs for %s", src, len(docs), target_date)
            time.sleep(INTER_REQUEST_DELAY)
        except NotImplementedError:
            logger.warning(
                "[%s] SPIKE stub — fetcher not implemented yet", src,
            )
        except Exception as e:
            logger.error("[%s] fetch failed: %s", src, e)

    _atomic_write_jsonl([d.to_dict() for d in all_docs], output_path)
    manifest = {
        "target_date": target_date,
        "collector_version": COLLECTOR_VERSION,
        "n_docs": len(all_docs),
        "sources_run": sources,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    logger.info("Collected %d regulator docs → %s", len(all_docs), output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Collect 证监会/交易所 regulator actions (SPIKE scaffold).",
    )
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD.")
    parser.add_argument(
        "--source", action="append", default=None,
        choices=list(SOURCES.keys()),
        help="Subset of sources (repeat to add). Default: all.",
    )
    args = parser.parse_args(argv)

    collect_regulator_actions(
        target_date=args.date,
        sources=args.source,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
