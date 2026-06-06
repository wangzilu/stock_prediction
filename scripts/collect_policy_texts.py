"""Collect policy texts — Phase E.1 step 1 (PBOC liquidity overlay).

Fetches monetary-policy texts from the People's Bank of China
(www.pbc.gov.cn) and writes one JSONL row per policy item under
``data/storage/policy_texts/pbc/<YYYY-MM-DD>.jsonl``. Raw HTML for any
item that fails to parse is dropped into
``data/storage/policy_texts/pbc_raw_html/`` so the LLM extract step can
be re-run with a different parser without re-fetching.

Usage
-----
    # Today only (cron mode)
    python scripts/collect_policy_texts.py --source pbc

    # Backfill an explicit window
    python scripts/collect_policy_texts.py --source pbc \\
        --start 2026-06-01 --end 2026-06-05

Constraints
-----------
- A-share isolation: this script lives in the A-share namespace and
  must not pull anything from ``data.crypto`` / ``scheduler.crypto`` /
  ssproxy. All HTTP uses plain ``requests`` with a 15-second timeout
  and a 3-attempt retry wrapper.
- Idempotent: re-running the same date overwrites the day's JSONL
  atomically (``.tmp`` + ``replace``). Existing rows for the same day
  are NOT merged — the day is a single re-fetched snapshot.
- Health: every run calls ``scheduler.data_health.write_health`` so the
  Phase A.7 SLA gate can see fresh / stale / partial state.

Output schema (one JSON object per line)
----------------------------------------
    {
      "publish_date":  "YYYY-MM-DD",   # asof_time of the text
      "policy_type":   "omo|mlf|slf|rrr|lpr|quarterly_report|press_conference|other",
      "title":         "原文标题",
      "url":           "https://www.pbc.gov.cn/...",
      "content":       "原文正文 (UTF-8, plain text, whitespace-normalized)",
      "source":        "pbc.gov.cn",
      "fetch_time":    "YYYY-MM-DDTHH:MM:SSZ"
    }

Phase E.1 downstream (out of scope here):
    extract_policy_events.py  -> LLM fact extraction
    build_policy_factors.py   -> PIT-safe factor parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)

# ── Storage layout ───────────────────────────────────────────────────
POLICY_DIR = DATA_DIR / "policy_texts" / "pbc"
RAW_HTML_DIR = DATA_DIR / "policy_texts" / "pbc_raw_html"

# ── PBOC list pages — kept as named constants so a structural change
# is one-line patchable. URLs verified against the public PBOC
# 货币政策司 column layout (zhengcehuobisi/125207/...).
#
# These are LIST pages: each renders an HTML <ul> of detail-page links
# whose hrefs are relative (``./2026/06/01/...html``) and whose link
# text is the policy title. The list pages tend to outlive any specific
# document URL because they're the directory index, but PBOC has been
# known to renumber the deepest segment — keep the section IDs visible.
PBC_LIST_URLS: dict[str, str] = {
    # OMO daily 公开市场操作业务公告 — highest cadence (daily)
    "omo": "http://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/125475/index.html",
    # LPR 利率 — monthly on the 20th
    "lpr": "http://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/index.html",
    # MLF / SLF / RRR / quarterly report — bonus tier, list pages
    # included for backfill completeness. Parsers are tolerant of
    # 404 / structural drift here; the cron-critical tier is OMO + LPR.
    "mlf": "http://www.pbc.gov.cn/zhengcehuobisi/125207/125213/3963540/index.html",
    "rrr": "http://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125428/index.html",
    "quarterly_report": (
        "http://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/index.html"
    ),
}

PBC_BASE = "http://www.pbc.gov.cn"
USER_AGENT = "Mozilla/5.0 (compatible; StockPrediction-PolicyCollector/1.0)"
REQUEST_TIMEOUT = 15  # seconds per attempt — required by spec
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0
INTER_REQUEST_DELAY = 0.4  # be polite to pbc.gov.cn

HEALTH_SOURCE_NAME = "pbc_policy_texts"


# ─────────────────────────────────────────────────────────────────────
# HTTP helper — retry wrapper required by spec.
# ─────────────────────────────────────────────────────────────────────
def http_get(
    url: str,
    *,
    session: requests.Session | None = None,
    attempts: int = RETRY_ATTEMPTS,
    backoff: float = RETRY_BACKOFF_SECONDS,
) -> requests.Response:
    """GET ``url`` with explicit 15s timeout and 3-attempt retry.

    Raises ``requests.RequestException`` if all attempts fail. Callers
    are expected to catch this so a single bad URL does not poison the
    whole run.
    """
    sess = session or requests
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            resp = sess.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            # Network-layer failure — retry.
            last_exc = e
            if i < attempts - 1:
                time.sleep(backoff * (i + 1))
            continue
        if resp.status_code == 200:
            return resp
        # 4xx → final, do NOT retry (404, 403, 401, etc are not
        # transient and the same URL will keep returning them).
        if 400 <= resp.status_code < 500:
            raise requests.HTTPError(
                f"HTTP {resp.status_code} for {url}", response=resp,
            )
        # 5xx / other → retry with backoff.
        last_exc = requests.HTTPError(
            f"HTTP {resp.status_code} for {url}", response=resp,
        )
        if i < attempts - 1:
            time.sleep(backoff * (i + 1))
    assert last_exc is not None
    raise last_exc


# ─────────────────────────────────────────────────────────────────────
# Parsing — list page → list of (url, title, publish_date)
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PolicyLink:
    url: str
    title: str
    publish_date: str  # YYYY-MM-DD, may be "" if list page only shows the article


# Date hints that PBOC list rows tend to use, in best-to-worst order.
_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})"),
    re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日"),
)


def _try_parse_date(text: str) -> str:
    for pat in _DATE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                y, mo, d = (int(g) for g in m.groups())
                return f"{y:04d}-{mo:02d}-{d:02d}"
            except ValueError:
                continue
    return ""


def _absolutize(href: str, base: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("//"):
        return "http:" + href
    if href.startswith("./"):
        # Relative to the list page directory
        return base.rsplit("/", 1)[0] + href[1:]
    if href.startswith("/"):
        return PBC_BASE + href
    return base.rsplit("/", 1)[0] + "/" + href


def parse_list_page(html: str, base_url: str) -> list[PolicyLink]:
    """Extract (url, title, publish_date) tuples from a PBOC list page.

    Uses BeautifulSoup if available, falls back to a regex pass that
    handles PBOC's canonical ``<a href="./YYYY/MM/DD/...html" ...>title</a>``
    pattern. List rows usually have a sibling ``<span>YYYY-MM-DD</span>``
    that we attach when parseable.
    """
    links: list[PolicyLink] = []
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            title = a.get_text(strip=True)
            if not title or len(title) < 4:
                continue
            # Skip nav anchors / external (e.g. mailto:, http://www.pbc.gov.cn/index.html nav)
            if href.startswith(("javascript:", "mailto:", "#")):
                continue
            # PBOC detail pages live under TWO patterns:
            #   (a) /zhengcehuobisi/.../<YYYYMMDDhhmmssNNN>/index.html
            #       (current live pattern — single timestamp string in path)
            #   (b) /YYYY/MM/DD/...html (legacy slash-separated pattern)
            # 2026-06-06 fix: original parser only matched (b) so the live
            # OMO page (which uses (a)) returned 0 candidates. Match both.
            looks_like_article = (
                href.endswith(".html")
                and (
                    re.search(r"/\d{14,}/index\.html$", href) is not None
                    or re.search(r"/\d{4}/\d{1,2}/\d{1,2}/", href) is not None
                    or re.search(r"_\d{4}\.html$", href) is not None
                )
            )
            if not looks_like_article:
                continue
            url = _absolutize(href, base_url)
            # Look for a date hint near the anchor (sibling span / parent text)
            date_hint = ""
            parent = a.parent
            if parent is not None:
                date_hint = _try_parse_date(parent.get_text(" ", strip=True))
            if not date_hint:
                # Pattern (a): leading YYYYMMDD in the timestamp digit string.
                m_ts = re.search(r"/(\d{4})(\d{2})(\d{2})\d{6,}/index\.html$", href)
                if m_ts:
                    y, mo, d = (int(g) for g in m_ts.groups())
                    date_hint = f"{y:04d}-{mo:02d}-{d:02d}"
            if not date_hint:
                # Pattern (b): slash-separated /YYYY/MM/DD/.
                m = re.search(r"/(\d{4})/(\d{1,2})/(\d{1,2})/", href)
                if m:
                    y, mo, d = (int(g) for g in m.groups())
                    date_hint = f"{y:04d}-{mo:02d}-{d:02d}"
            links.append(PolicyLink(url=url, title=title, publish_date=date_hint))
    except Exception as e:
        logger.warning("BeautifulSoup parse failed (%s); falling back to regex", e)
        # Last-resort regex pass over <a href="...html">title</a>
        for m in re.finditer(
            r'<a[^>]+href="([^"]+\.html)"[^>]*>([^<]{4,200})</a>', html
        ):
            href, title = m.group(1).strip(), m.group(2).strip()
            ts_m = re.search(r"/(\d{4})(\d{2})(\d{2})\d{6,}/index\.html$", href)
            slash_m = re.search(r"/(\d{4})/(\d{1,2})/(\d{1,2})/", href)
            if not (ts_m or slash_m):
                continue
            url = _absolutize(href, base_url)
            date_hint = ""
            if ts_m:
                y, mo, d = (int(g) for g in ts_m.groups())
                date_hint = f"{y:04d}-{mo:02d}-{d:02d}"
            elif slash_m:
                y, mo, d = (int(g) for g in slash_m.groups())
                date_hint = f"{y:04d}-{mo:02d}-{d:02d}"
            links.append(PolicyLink(url=url, title=title, publish_date=date_hint))
    # Deduplicate by URL preserving first-seen order.
    seen: set[str] = set()
    deduped: list[PolicyLink] = []
    for ln in links:
        if ln.url in seen:
            continue
        seen.add(ln.url)
        deduped.append(ln)
    return deduped


# ─────────────────────────────────────────────────────────────────────
# Parsing — article detail page → plain-text content
# ─────────────────────────────────────────────────────────────────────
def parse_article_page(html: str) -> tuple[str, str]:
    """Return (content_text, publish_date_hint) for a single PBOC article.

    PBOC detail pages use ``<div id="zoom">`` (legacy) or
    ``<div class="detail_zoom">`` for the body, plus a header line
    like ``发布时间：2026-06-05``.
    """
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        body_div = (
            soup.find("div", id="zoom")
            or soup.find("div", class_="detail_zoom")
            or soup.find("div", class_="content")
        )
        if body_div is not None:
            text = body_div.get_text("\n", strip=True)
        else:
            text = soup.get_text("\n", strip=True)
        # Normalize whitespace (collapse runs of blanks, but keep paragraph breaks)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Publish-date hint — look at the first 400 chars for "发布时间" / "发布日期"
        head = soup.get_text(" ", strip=True)[:600]
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        head = text[:600]
    date_hint = _try_parse_date(head)
    return text, date_hint


# ─────────────────────────────────────────────────────────────────────
# Main collection
# ─────────────────────────────────────────────────────────────────────
def _slug(text: str, limit: int = 60) -> str:
    """Make a filename-safe slug from a URL or title."""
    s = re.sub(r"[^0-9A-Za-z一-鿿]+", "_", text).strip("_")
    return s[:limit] if s else "untitled"


def _now_utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _date_range(start: str, end: str) -> list[str]:
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    if e < s:
        raise ValueError(f"--end ({end}) must be >= --start ({start})")
    out: list[str] = []
    cur = s
    while cur <= e:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def _save_raw_html_on_failure(url: str, html: str, target_date: str) -> Path:
    """Persist raw HTML so extract_policy_events.py can re-run the parser."""
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{target_date}_{_slug(url)}.html"
    path = RAW_HTML_DIR / fname
    try:
        path.write_text(html, encoding="utf-8")
    except Exception as e:  # pragma: no cover — disk error
        logger.warning("Failed to save raw HTML for %s: %s", url, e)
    return path


def _atomic_write_jsonl(rows: list[dict], path: Path) -> None:
    """Write rows to ``path`` atomically (.tmp + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def collect_pbc(
    *,
    start: str,
    end: str,
    policy_types: Iterable[str] = ("omo", "lpr"),
    bonus_types: Iterable[str] = ("mlf", "rrr", "quarterly_report"),
    http_get_fn: Callable[..., requests.Response] = http_get,
    output_dir: Path | None = None,
) -> dict:
    """Collect PBOC policy texts for every day in [start, end] inclusive.

    Returns a summary dict::

        {
          "rows_by_date":     {date: n_rows, ...},
          "errors":           [{"url": str, "stage": str, "msg": str}, ...],
          "files_written":    [Path, ...],
          "policy_types_seen": {"omo": n, "lpr": n, ...},
          "n_total":           int,
        }

    The function does NOT raise on individual fetch errors — they
    accumulate in ``errors`` and the per-day file is still written
    (possibly with fewer rows). The caller decides whether ``partial``
    in the health record should be True.
    """
    output_root = output_dir or POLICY_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    dates = _date_range(start, end)
    date_set: set[str] = set(dates)

    summary: dict = {
        "rows_by_date": {d: 0 for d in dates},
        "errors": [],
        "files_written": [],
        "policy_types_seen": {},
        "n_total": 0,
    }

    # Step 1: for each list page in (priority ∪ bonus), discover links.
    by_date_rows: dict[str, list[dict]] = {d: [] for d in dates}
    seen_urls: set[str] = set()
    session = requests.Session()

    for pt in list(policy_types) + list(bonus_types):
        list_url = PBC_LIST_URLS.get(pt)
        if not list_url:
            continue
        try:
            resp = http_get_fn(list_url, session=session)
            list_html = resp.content.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning("List page failed for %s: %s", pt, e)
            summary["errors"].append(
                {"url": list_url, "stage": "list", "msg": str(e)}
            )
            continue

        links = parse_list_page(list_html, list_url)
        logger.info("PBOC %s list page: %d candidate links", pt, len(links))

        for ln in links:
            if ln.url in seen_urls:
                continue
            # If we can already filter by publish_date and it's outside
            # the requested window, skip the article fetch entirely.
            if ln.publish_date and ln.publish_date not in date_set:
                continue
            seen_urls.add(ln.url)
            time.sleep(INTER_REQUEST_DELAY)
            try:
                art_resp = http_get_fn(ln.url, session=session)
                art_html = art_resp.content.decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning("Article fetch failed %s: %s", ln.url, e)
                summary["errors"].append(
                    {"url": ln.url, "stage": "article", "msg": str(e)}
                )
                continue
            try:
                content, article_date = parse_article_page(art_html)
            except Exception as e:
                logger.warning("Article parse failed %s: %s", ln.url, e)
                summary["errors"].append(
                    {"url": ln.url, "stage": "parse", "msg": str(e)}
                )
                _save_raw_html_on_failure(
                    ln.url, art_html, ln.publish_date or start,
                )
                continue

            publish_date = ln.publish_date or article_date
            if not publish_date:
                # Cannot place this in a daily file — skip but keep HTML
                _save_raw_html_on_failure(ln.url, art_html, start)
                summary["errors"].append(
                    {"url": ln.url, "stage": "no_date", "msg": "no publish_date"}
                )
                continue
            if publish_date not in date_set:
                continue
            if not content or len(content) < 20:
                # Empty body — keep HTML for re-parse but skip the row.
                _save_raw_html_on_failure(ln.url, art_html, publish_date)
                summary["errors"].append(
                    {"url": ln.url, "stage": "empty_body", "msg": "len(content)<20"}
                )
                continue

            row = {
                "publish_date": publish_date,
                "policy_type": pt,
                "title": ln.title,
                "url": ln.url,
                "content": content,
                "source": "pbc.gov.cn",
                "fetch_time": _now_utc_iso(),
            }
            by_date_rows[publish_date].append(row)
            summary["policy_types_seen"][pt] = (
                summary["policy_types_seen"].get(pt, 0) + 1
            )

    # Step 2: atomic-write per-day JSONL files (always write, even empty,
    # so re-runs are deterministic — overwrites supersede any prior partial).
    for d in dates:
        rows = by_date_rows[d]
        # Sort rows for stable output (idempotent across re-runs).
        rows.sort(key=lambda r: (r["policy_type"], r["url"]))
        out_path = output_root / f"{d}.jsonl"
        _atomic_write_jsonl(rows, out_path)
        summary["rows_by_date"][d] = len(rows)
        summary["files_written"].append(out_path)
        summary["n_total"] += len(rows)
        logger.info("Wrote %d row(s) -> %s", len(rows), out_path)

    return summary


# ─────────────────────────────────────────────────────────────────────
# Health publishing
# ─────────────────────────────────────────────────────────────────────
def publish_health(summary: dict, *, target_date: str) -> None:
    """Write a HealthStatus record via the standard scheduler interface.

    ``partial`` is True when any error rows were recorded but at least
    one item was successfully written. ``success`` is True iff at least
    one item was written (i.e. the day is on file).
    """
    try:
        from scheduler.data_health import HealthStatus, write_health
    except Exception as e:  # pragma: no cover — broken install
        logger.warning("Cannot import scheduler.data_health (%s)", e)
        return

    n_total = int(summary.get("n_total", 0))
    n_errors = len(summary.get("errors", []))
    last_date = ""
    for d in sorted(summary.get("rows_by_date", {}).keys(), reverse=True):
        if summary["rows_by_date"][d] > 0:
            last_date = d
            break
    status = HealthStatus(
        success=n_total > 0,
        n_items=n_total,
        latest_date=last_date or target_date,
        partial=(n_total > 0 and n_errors > 0),
        error_type="" if n_total > 0 else "no_rows",
        error_message=(
            "; ".join(
                f"{e['stage']}:{e['url']}" for e in summary.get("errors", [])[:3]
            )
            if n_errors
            else ""
        ),
        network_profile="ashare",
        extra={
            "policy_types_seen": summary.get("policy_types_seen", {}),
            "rows_by_date": summary.get("rows_by_date", {}),
            "n_errors": n_errors,
        },
    )
    write_health(HEALTH_SOURCE_NAME, status, date=target_date)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Collect policy texts (Phase E.1 step 1)."
    )
    parser.add_argument(
        "--source",
        default="pbc",
        choices=["pbc"],
        help=(
            "Policy source. Only 'pbc' is implemented today; 'gov_cn' "
            "(State Council) and 'nbs' (statistics bureau) are reserved "
            "for Phase E.2 / E.3."
        ),
    )
    parser.add_argument(
        "--start", default=None,
        help="Backfill start date YYYY-MM-DD (default: today).",
    )
    parser.add_argument(
        "--end", default=None,
        help="Backfill end date YYYY-MM-DD (default: today).",
    )
    parser.add_argument(
        "--include-bonus", action="store_true",
        help=(
            "Also fetch MLF / RRR / quarterly_report list pages. "
            "These are bonus-tier sources for E.1 and may 404 without "
            "blocking the run."
        ),
    )
    parser.add_argument(
        "--fail-on-empty", action="store_true",
        help=(
            "2026-06-06 P3 #6 cron-hardening: exit 1 when no rows were "
            "written for any requested date. Without this flag the "
            "script always exits 0 and write_health flags success=False / "
            "no_rows, which a cron that only looks at exit code or file "
            "existence will miss. Enable for production cron, disable "
            "for local exploration."
        ),
    )
    args = parser.parse_args(argv)

    today = datetime.now().strftime("%Y-%m-%d")
    start = args.start or today
    end = args.end or args.start or today

    if args.source != "pbc":
        # Re-raise as a hard error rather than silently no-op. The
        # argparse choices=["pbc"] already prevents this, but defend.
        logger.error("Unsupported --source %s", args.source)
        return 2

    bonus = ("mlf", "rrr", "quarterly_report") if args.include_bonus else ()
    try:
        summary = collect_pbc(
            start=start, end=end,
            policy_types=("omo", "lpr"),
            bonus_types=bonus,
        )
    except Exception as e:
        logger.error("collect_pbc raised: %s", e)
        try:
            from scheduler.data_health import HealthStatus, write_health
            write_health(
                HEALTH_SOURCE_NAME,
                HealthStatus(
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e)[:300],
                    network_profile="ashare",
                ),
                date=end,
            )
        except Exception:
            pass
        return 1

    # Publish health using the LATEST day as the date key so the SLA
    # gate sees today's record on today's run.
    publish_health(summary, target_date=end)

    logger.info(
        "Done. n_total=%d, files=%d, errors=%d, rows_by_date=%s",
        summary["n_total"],
        len(summary["files_written"]),
        len(summary["errors"]),
        summary["rows_by_date"],
    )
    # Surface the per-file paths so the caller / cron log makes it
    # obvious what was written.
    for p in summary["files_written"]:
        logger.info("  -> %s (%d rows)", p, summary["rows_by_date"][p.stem])
    # 2026-06-06 P3 #6 fix: surface a non-zero exit when the user
    # asked for fail-loud and not a single row landed. write_health
    # already records this correctly; this is for cron wrappers that
    # only watch the exit code.
    if args.fail_on_empty and summary["n_total"] == 0:
        logger.error(
            "fail-on-empty: 0 rows written across [%s..%s]. Exiting 1.",
            start, end,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
