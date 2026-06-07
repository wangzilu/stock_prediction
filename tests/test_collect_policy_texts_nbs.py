"""Tests for scripts/collect_policy_texts.py --source nbs.

Phase E.3 (PE-3) step 1 — mirror of test_collect_policy_texts_state_council.py
for the NBS (国家统计局) macro-statistics branch. Covers:

  1. URL pattern fan-out: each of the 4 series LIST URLs gets fetched,
     and the NBS detail URL pattern (gov.cn /YYYY-MM/DD/content_*.html)
     is parsed by the shared parse_list_page.
  2. JSONL schema lock: same 7 documented keys as PBC/PE-2; source field
     is ``stats.gov.cn``; policy_type is one of the NBS enum
     (cpi_monthly / ppi_monthly / pmi_monthly / retail_sales_monthly).
  3. Idempotent re-run: two runs over the same date produce the same
     rows; no .tmp file leftover; no duplicates.
  4. /-rooted href cross-host safety: a stats.gov.cn /-prefixed href
     must absolutize onto www.stats.gov.cn, NOT onto pbc.gov.cn.
  5. NBS URL pattern parse: parse_list_page extracts the date hint
     correctly from the gov.cn ``/YYYY-MM/DD/`` URL layout NBS uses.

All tests are HTTP-mocked. The mock intercepts ``http_get`` so we
never depend on stats.gov.cn being reachable in CI.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from scripts import collect_policy_texts as ctp


# ─────────────────────────────────────────────────────────────────────
# Fixtures — minimal valid NBS list / detail HTML matching gov.cn-style
# layout. Each list page mixes its target series with one off-topic row
# so we can also exercise the title-keyword filter.
# ─────────────────────────────────────────────────────────────────────
CPI_LIST_HTML = """
<html><body>
<ul class="list">
  <li>
    <a href="/sj/zxfb/2026-06/03/content_7100001.html">
      2026年5月份居民消费价格变动情况
    </a>
    <span>2026-06-03</span>
  </li>
  <li>
    <a href="/sj/zxfb/2026-06/05/content_7100002.html">
      关于工业生产者出厂价格的解读
    </a>
    <span>2026-06-05</span>
  </li>
  <li><a href="javascript:void(0)">nav anchor</a></li>
</ul>
</body></html>
"""

PPI_LIST_HTML = """
<html><body>
<a href="/sj/sjjd/2026-06/04/content_7100099.html">
  2026年5月份工业生产者出厂价格同比下降0.6%
</a>
<span>2026-06-04</span>
</body></html>
"""

PMI_LIST_HTML = """
<html><body>
<a href="/xxgk/sjfb/zxfb2020/2026-06/03/content_7200005.html">
  2026年5月中国采购经理指数运行情况（PMI）
</a>
<span>2026-06-03</span>
</body></html>
"""

RETAIL_LIST_HTML = """
<html><body>
<a href="/sj/2026-06/05/content_7300003.html">
  2026年5月份社会消费品零售总额同比增长3.2%
</a>
<span>2026-06-05</span>
</body></html>
"""

CPI_DETAIL_HTML = """
<html><body>
<div class="header">发布时间：2026-06-03</div>
<div id="UCAP-CONTENT">
  <p>2026年5月份，全国居民消费价格（CPI）同比上涨0.3%。环比上涨0.1%。
  其中，城市上涨0.3%，农村上涨0.2%。</p>
  <p>市场普遍预期同比上涨0.5%，实际值低于预期。</p>
</div>
</body></html>
"""

PPI_DETAIL_HTML = """
<html><body>
<div id="UCAP-CONTENT">
  <p>2026年5月份，全国工业生产者出厂价格（PPI）同比下降0.6%，
  环比下降0.2%。市场预期同比下降0.4%。</p>
</div>
</body></html>
"""

PMI_DETAIL_HTML = """
<html><body>
<div id="UCAP-CONTENT">
  <p>2026年5月份，中国制造业采购经理指数（PMI）为50.4%，
  比上月上升0.2个百分点，连续三个月位于扩张区间。</p>
</div>
</body></html>
"""

RETAIL_DETAIL_HTML = """
<html><body>
<div id="UCAP-CONTENT">
  <p>2026年5月份，社会消费品零售总额41326亿元，同比增长3.2%，
  环比增长0.5%。市场预期同比增长3.0%。</p>
</div>
</body></html>
"""


def _mock_response(status: int, body: str) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.content = body.encode("utf-8")
    resp.text = body
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            f"HTTP {status}", response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_http_get(url_to_response: dict[str, MagicMock]):
    """Build a fake ``http_get`` that returns canned responses by URL."""

    def fake_http_get(url, *, session=None, attempts=3, backoff=0.0):
        if url not in url_to_response:
            raise requests.HTTPError(f"unmocked URL: {url}")
        resp = url_to_response[url]
        if resp.status_code >= 400:
            raise requests.HTTPError(f"HTTP {resp.status_code} for {url}")
        return resp

    return fake_http_get


def _abs(list_url: str, href: str) -> str:
    """Helper to absolutize a /-rooted href via the production helper."""
    return ctp._absolutize(href, list_url)


# ─────────────────────────────────────────────────────────────────────
# Test 1 — URL pattern fan-out: 4 series LIST URLs queried, NBS detail
#          pages parsed across mixed gov.cn layouts.
# ─────────────────────────────────────────────────────────────────────
def test_url_pattern_fan_out_across_all_four_series(tmp_path: Path):
    """All 4 registered LIST URLs are queried, NBS YYYY-MM/DD detail
    URL pattern is recognized for each series."""
    cpi_url = ctp.NBS_LIST_URLS["cpi"]
    ppi_url = ctp.NBS_LIST_URLS["ppi"]
    pmi_url = ctp.NBS_LIST_URLS["pmi"]
    retail_url = ctp.NBS_LIST_URLS["retail_sales"]

    url_map = {
        cpi_url: _mock_response(200, CPI_LIST_HTML),
        ppi_url: _mock_response(200, PPI_LIST_HTML),
        pmi_url: _mock_response(200, PMI_LIST_HTML),
        retail_url: _mock_response(200, RETAIL_LIST_HTML),
        # CPI detail (the off-topic PPI row in CPI list must be skipped
        # by the title filter, so we don't mock its URL).
        _abs(cpi_url, "/sj/zxfb/2026-06/03/content_7100001.html"):
            _mock_response(200, CPI_DETAIL_HTML),
        _abs(ppi_url, "/sj/sjjd/2026-06/04/content_7100099.html"):
            _mock_response(200, PPI_DETAIL_HTML),
        _abs(pmi_url, "/xxgk/sjfb/zxfb2020/2026-06/03/content_7200005.html"):
            _mock_response(200, PMI_DETAIL_HTML),
        _abs(retail_url, "/sj/2026-06/05/content_7300003.html"):
            _mock_response(200, RETAIL_DETAIL_HTML),
    }

    summary = ctp.collect_nbs(
        start="2026-06-03", end="2026-06-05",
        http_get_fn=_make_http_get(url_map),
        output_dir=tmp_path,
    )

    # All four series should appear with their NBS-specific policy_type.
    assert summary["n_total"] >= 4, summary
    seen_types = set(summary["policy_types_seen"].keys())
    assert "cpi_monthly" in seen_types, seen_types
    assert "ppi_monthly" in seen_types, seen_types
    assert "pmi_monthly" in seen_types, seen_types
    assert "retail_sales_monthly" in seen_types, seen_types

    # Each window day has a file written.
    for d in ("2026-06-03", "2026-06-04", "2026-06-05"):
        assert (tmp_path / f"{d}.jsonl").exists(), d


# ─────────────────────────────────────────────────────────────────────
# Test 2 — JSONL schema lock: 7 documented keys + stats.gov.cn source
#          + policy_type from NBS enum.
# ─────────────────────────────────────────────────────────────────────
def test_jsonl_schema_matches_pe1_contract_with_stats_gov_cn_source(tmp_path: Path):
    """Every JSONL row has the 7 documented keys and nothing else.
    source is ``stats.gov.cn`` and policy_type is in the NBS enum."""
    cpi_url = ctp.NBS_LIST_URLS["cpi"]
    pmi_url = ctp.NBS_LIST_URLS["pmi"]
    url_map = {
        cpi_url: _mock_response(200, CPI_LIST_HTML),
        ctp.NBS_LIST_URLS["ppi"]: _mock_response(200, "<html></html>"),
        pmi_url: _mock_response(200, PMI_LIST_HTML),
        ctp.NBS_LIST_URLS["retail_sales"]: _mock_response(200, "<html></html>"),
        _abs(cpi_url, "/sj/zxfb/2026-06/03/content_7100001.html"):
            _mock_response(200, CPI_DETAIL_HTML),
        _abs(pmi_url, "/xxgk/sjfb/zxfb2020/2026-06/03/content_7200005.html"):
            _mock_response(200, PMI_DETAIL_HTML),
    }
    summary = ctp.collect_nbs(
        start="2026-06-03", end="2026-06-03",
        http_get_fn=_make_http_get(url_map),
        output_dir=tmp_path,
    )
    assert summary["n_total"] >= 2, summary
    expected_keys = {
        "publish_date", "policy_type", "title", "url",
        "content", "source", "fetch_time",
    }
    rows = [
        json.loads(ln)
        for ln in (tmp_path / "2026-06-03.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if ln.strip()
    ]
    assert rows, "expected at least one row on 2026-06-03"
    for row in rows:
        assert set(row.keys()) == expected_keys, sorted(row.keys())
        assert row["source"] == "stats.gov.cn", (
            "PE-3 rows must source=stats.gov.cn so downstream can split "
            "PBC vs SC vs NBS by source field."
        )
        assert row["policy_type"] in {
            "cpi_monthly", "ppi_monthly",
            "pmi_monthly", "retail_sales_monthly",
        }, row["policy_type"]
        assert row["fetch_time"].endswith("Z")
        assert len(row["content"]) >= 20


# ─────────────────────────────────────────────────────────────────────
# Test 3 — Idempotent re-run: snapshot between runs
# ─────────────────────────────────────────────────────────────────────
def test_idempotent_rerun_does_not_duplicate_rows(tmp_path: Path):
    """Two runs over the same date produce identical rows — no append."""
    cpi_url = ctp.NBS_LIST_URLS["cpi"]
    url_map = {
        cpi_url: _mock_response(200, CPI_LIST_HTML),
        ctp.NBS_LIST_URLS["ppi"]: _mock_response(200, "<html></html>"),
        ctp.NBS_LIST_URLS["pmi"]: _mock_response(200, "<html></html>"),
        ctp.NBS_LIST_URLS["retail_sales"]: _mock_response(200, "<html></html>"),
        _abs(cpi_url, "/sj/zxfb/2026-06/03/content_7100001.html"):
            _mock_response(200, CPI_DETAIL_HTML),
    }
    fake = _make_http_get(url_map)

    s1 = ctp.collect_nbs(
        start="2026-06-03", end="2026-06-03",
        http_get_fn=fake, output_dir=tmp_path,
    )
    out = tmp_path / "2026-06-03.jsonl"
    assert out.exists()
    lines1 = [
        json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    s2 = ctp.collect_nbs(
        start="2026-06-03", end="2026-06-03",
        http_get_fn=fake, output_dir=tmp_path,
    )
    lines2 = [
        json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    def _strip(rows: list[dict]) -> list[dict]:
        return [{k: v for k, v in r.items() if k != "fetch_time"} for r in rows]

    assert _strip(lines1) == _strip(lines2)
    assert s1["n_total"] == s2["n_total"] == len(lines1)
    # No .tmp file should be left over after atomic replace.
    assert not list(tmp_path.glob("*.tmp"))


# ─────────────────────────────────────────────────────────────────────
# Test 4 — /-rooted href cross-host safety: stats.gov.cn /-prefixed href
#          must NOT absolutize onto pbc.gov.cn.
# ─────────────────────────────────────────────────────────────────────
def test_slash_rooted_href_resolves_to_stats_host_not_pbc():
    """A /-rooted href from a stats.gov.cn list page must absolutize onto
    www.stats.gov.cn, not www.pbc.gov.cn. Regression guard for the
    legacy ``_absolutize`` that hardcoded PBC_BASE for /-prefixed hrefs.
    """
    base = "http://www.stats.gov.cn/sj/zxfb/"
    href = "/sj/zxfb/2026-06/03/content_7100001.html"
    resolved = ctp._absolutize(href, base)
    assert resolved.startswith("http://www.stats.gov.cn/"), (
        f"/-rooted href must resolve onto the LIST PAGE host. got {resolved}"
    )
    # Sanity: an stats.gov.cn list-page /-rooted href must NOT land on
    # pbc.gov.cn or gov.cn.
    assert "pbc.gov.cn" not in resolved
    # And the PBC code path still works (no regression for PE-1).
    pbc_base = "http://www.pbc.gov.cn/zhengcehuobisi/x/y/index.html"
    pbc_resolved = ctp._absolutize(
        "/zhengcehuobisi/something/index.html", pbc_base,
    )
    assert pbc_resolved.startswith("http://www.pbc.gov.cn/")


# ─────────────────────────────────────────────────────────────────────
# Test 5 — NBS detail URL pattern parses correctly.
# ─────────────────────────────────────────────────────────────────────
def test_parse_list_page_handles_nbs_url_pattern():
    """The shared parse_list_page extracts the YYYY-MM-DD date hint from
    the NBS ``/YYYY-MM/DD/content_*.html`` URL pattern (same as gov.cn
    State Council pattern). Also verifies the title-keyword filter logic
    is hooked into the collector via _title_matches_series."""
    sample_html = '''
    <html><body>
      <a href="/sj/zxfb/2026-06/03/content_7100001.html">
        2026年5月份居民消费价格变动情况
      </a>
      <a href="/sj/zxfb/2026-05/10/content_7099999.html">
        2026年4月份居民消费价格变动情况
      </a>
      <a href="/index.html">nav link no date</a>
    </body></html>
    '''
    base = "http://www.stats.gov.cn/sj/zxfb/"
    links = ctp.parse_list_page(sample_html, base)
    by_date = {ln.publish_date for ln in links}
    assert "2026-06-03" in by_date, f"got dates {by_date}"
    assert "2026-05-10" in by_date, f"got dates {by_date}"
    # All resolved URLs must be stats.gov.cn.
    for ln in links:
        if "content_710" in ln.url:
            assert "www.stats.gov.cn" in ln.url, ln.url

    # Title-keyword filter unit-test.
    assert ctp._title_matches_series("2026年5月份居民消费价格变动情况", "cpi")
    assert ctp._title_matches_series("CPI rises 0.5% YoY", "cpi")
    assert not ctp._title_matches_series(
        "工业生产者出厂价格同比下降", "cpi"
    ), "PPI title must not match cpi"
    assert ctp._title_matches_series(
        "工业生产者出厂价格同比下降", "ppi"
    )
    assert ctp._title_matches_series(
        "中国采购经理指数运行情况", "pmi"
    )
    assert ctp._title_matches_series(
        "社会消费品零售总额同比增长", "retail_sales"
    )
