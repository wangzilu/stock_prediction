"""Tests for scripts/collect_policy_texts.py --source state_council.

Phase E.2 (PE-2) step 1 — mirror of test_collect_policy_texts.py for
the State Council + ministry branch. Covers:

  1. URL pattern fan-out: each of the 5 LIST URLs gets fetched and the
     gov.cn detail URL patterns (/YYYY-MM/DD/content_*.html plus the
     PBC-style /YYYY/MM/DD/) are both parsed.
  2. JSONL schema lock: same 7 documented keys as PBC; source field
     is ``gov.cn`` not ``pbc.gov.cn``; policy_type is one of the SC
     enum (state_council_doc / state_council_meeting / miit_policy /
     ndrc_policy / mof_policy).
  3. Idempotent re-run: two runs over the same date produce the same
     rows; no .tmp file leftover; no duplicates.
  4. /-rooted href cross-host safety: a gov.cn /-prefixed href must
     absolutize onto www.gov.cn, NOT onto www.pbc.gov.cn.

All tests are HTTP-mocked. The mock intercepts ``http_get`` so we
never depend on gov.cn being reachable in CI.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from scripts import collect_policy_texts as ctp


# ─────────────────────────────────────────────────────────────────────
# Fixtures — minimal valid gov.cn list / detail HTML
# ─────────────────────────────────────────────────────────────────────
STATE_COUNCIL_DOC_LIST_HTML = """
<html><body>
<ul class="list">
  <li>
    <a href="/zhengce/zhengceku/2026-06/03/content_5800001.htm">
      国务院关于推动新能源汽车高质量发展的若干政策
    </a>
    <span>2026-06-03</span>
  </li>
  <li>
    <a href="/zhengce/zhengceku/2026-06/05/content_5800002.html">
      国务院办公厅关于半导体产业升级的指导意见
    </a>
    <span>2026-06-05</span>
  </li>
  <li>
    <a href="/zhengce/zhengceku/2026-05/20/content_5700099.html">老旧政策（窗口外）</a>
    <span>2026-05-20</span>
  </li>
  <li><a href="javascript:void(0)">nav anchor</a></li>
</ul>
</body></html>
"""

MIIT_LIST_HTML = """
<html><body>
<a href="/lianbo/bumen/202606/content_1234567.html">
  工业和信息化部关于加快机器人产业发展的若干措施
</a>
<span>2026-06-04</span>
</body></html>
"""

NDRC_LIST_HTML = """
<html><body>
<a href="/lianbo/bumen/ndrc/202606/05/content_8765432.html">
  发改委关于战略性新兴产业投资清单
</a>
<span>2026-06-05</span>
</body></html>
"""

STATE_COUNCIL_DETAIL_HTML = """
<html><body>
<div class="header">发布时间：2026-06-05</div>
<div id="UCAP-CONTENT">
  <p>为推动半导体产业升级，国务院办公厅决定安排中央财政资金500亿元
  支持先进制程研发。本意见自发布之日起实施。</p>
</div>
</body></html>
"""

MIIT_DETAIL_HTML = """
<html><body>
<div class="pages_content">
  <p>为支持机器人产业高质量发展，工业和信息化部决定对核心零部件
  研发给予税收减免，对采购国产工业机器人的整车企业给予补贴。</p>
</div>
</body></html>
"""

NDRC_DETAIL_HTML = """
<html><body>
<div class="pages_content">
  <p>2026年战略性新兴产业投资清单：新能源、半导体、生物医药、
  人工智能、商业航天。国家发改委将每年评估清单调整。</p>
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
    """Helper to absolutize a /-rooted href using the same logic as the
    production collector. Avoids hardcoding scheme/host in tests."""
    return ctp._absolutize(href, list_url)


# ─────────────────────────────────────────────────────────────────────
# Test 1 — URL pattern fan-out: 5 list URLs queried, all 3 mocked
#          detail patterns parsed.
# ─────────────────────────────────────────────────────────────────────
def test_url_pattern_fan_out_across_state_council_and_ministries(tmp_path: Path):
    """All 5 registered LIST URLs are queried, gov.cn YYYY-MM/DD detail
    URL pattern is recognized, three different detail pages survive."""
    sc_doc_url = ctp.STATE_COUNCIL_LIST_URLS["state_council_doc"]
    sc_meet_url = ctp.STATE_COUNCIL_LIST_URLS["state_council_meeting"]
    miit_url = ctp.STATE_COUNCIL_LIST_URLS["miit_policy"]
    ndrc_url = ctp.STATE_COUNCIL_LIST_URLS["ndrc_policy"]
    mof_url = ctp.STATE_COUNCIL_LIST_URLS["mof_policy"]

    url_map = {
        sc_doc_url: _mock_response(200, STATE_COUNCIL_DOC_LIST_HTML),
        # meeting / MOF list 404 — must NOT kill the run
        sc_meet_url: _mock_response(404, "Not Found"),
        miit_url: _mock_response(200, MIIT_LIST_HTML),
        ndrc_url: _mock_response(200, NDRC_LIST_HTML),
        mof_url: _mock_response(500, "boom"),
        # Detail pages — gov.cn pattern /YYYY-MM/DD/content_*.html
        _abs(sc_doc_url, "/zhengce/zhengceku/2026-06/03/content_5800001.htm"):
            _mock_response(200, STATE_COUNCIL_DETAIL_HTML),
        _abs(sc_doc_url, "/zhengce/zhengceku/2026-06/05/content_5800002.html"):
            _mock_response(200, STATE_COUNCIL_DETAIL_HTML),
        _abs(miit_url, "/lianbo/bumen/202606/content_1234567.html"):
            _mock_response(200, MIIT_DETAIL_HTML),
        _abs(ndrc_url, "/lianbo/bumen/ndrc/202606/05/content_8765432.html"):
            _mock_response(200, NDRC_DETAIL_HTML),
    }

    summary = ctp.collect_state_council(
        start="2026-06-03", end="2026-06-05",
        http_get_fn=_make_http_get(url_map),
        output_dir=tmp_path,
    )

    # SC doc list yielded 2 in-window; MIIT 1; NDRC 1; meeting+MOF
    # errored out gracefully and are recorded in summary["errors"].
    assert summary["n_total"] >= 3, summary
    errors_by_stage = {e["stage"] for e in summary["errors"]}
    assert "list" in errors_by_stage, "404 / 500 list pages must record errors"

    # Three distinct policy_types must appear in the day files.
    seen_types = set(summary["policy_types_seen"].keys())
    assert "state_council_doc" in seen_types
    assert "miit_policy" in seen_types
    assert "ndrc_policy" in seen_types

    # The 2026-05-20 doc is out of the window — must NOT land.
    for d in ("2026-06-03", "2026-06-04", "2026-06-05"):
        # Day file exists (even if 0 rows)
        assert (tmp_path / f"{d}.jsonl").exists(), d
    # The 2026-05-20 doc must not be anywhere
    for d in ("2026-06-03", "2026-06-04", "2026-06-05"):
        rows = [
            json.loads(ln)
            for ln in (tmp_path / f"{d}.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        for r in rows:
            assert "2026-05-20" not in r["url"]


# ─────────────────────────────────────────────────────────────────────
# Test 2 — JSONL schema lock: 7 documented keys + gov.cn source +
#          policy_type from SC enum.
# ─────────────────────────────────────────────────────────────────────
def test_jsonl_schema_matches_pe1_contract_with_gov_cn_source(tmp_path: Path):
    """Every JSONL row has the 7 documented keys and nothing else.
    source is ``gov.cn`` and policy_type is one of the SC enum."""
    sc_doc_url = ctp.STATE_COUNCIL_LIST_URLS["state_council_doc"]
    miit_url = ctp.STATE_COUNCIL_LIST_URLS["miit_policy"]
    url_map = {
        sc_doc_url: _mock_response(200, STATE_COUNCIL_DOC_LIST_HTML),
        ctp.STATE_COUNCIL_LIST_URLS["state_council_meeting"]:
            _mock_response(200, "<html><body></body></html>"),
        miit_url: _mock_response(200, MIIT_LIST_HTML),
        ctp.STATE_COUNCIL_LIST_URLS["ndrc_policy"]:
            _mock_response(200, "<html><body></body></html>"),
        ctp.STATE_COUNCIL_LIST_URLS["mof_policy"]:
            _mock_response(200, "<html><body></body></html>"),
        _abs(sc_doc_url, "/zhengce/zhengceku/2026-06/05/content_5800002.html"):
            _mock_response(200, STATE_COUNCIL_DETAIL_HTML),
        _abs(miit_url, "/lianbo/bumen/202606/content_1234567.html"):
            _mock_response(200, MIIT_DETAIL_HTML),
    }
    summary = ctp.collect_state_council(
        start="2026-06-04", end="2026-06-05",
        http_get_fn=_make_http_get(url_map),
        output_dir=tmp_path,
    )
    assert summary["n_total"] >= 2, summary
    expected_keys = {
        "publish_date", "policy_type", "title", "url",
        "content", "source", "fetch_time",
    }
    for d in ("2026-06-04", "2026-06-05"):
        rows = [
            json.loads(ln)
            for ln in (tmp_path / f"{d}.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        for row in rows:
            assert set(row.keys()) == expected_keys, sorted(row.keys())
            assert row["source"] == "gov.cn", (
                "PE-2 rows must source=gov.cn so the downstream "
                "extractor can split PBC vs SC by source field."
            )
            assert row["policy_type"] in {
                "state_council_doc", "state_council_meeting",
                "miit_policy", "ndrc_policy", "mof_policy",
            }, row["policy_type"]
            assert row["fetch_time"].endswith("Z")
            assert len(row["content"]) >= 20


# ─────────────────────────────────────────────────────────────────────
# Test 3 — Idempotent re-run: snapshot between runs
# ─────────────────────────────────────────────────────────────────────
def test_idempotent_rerun_does_not_duplicate_rows(tmp_path: Path):
    """Two runs over the same date produce identical rows — no append."""
    sc_doc_url = ctp.STATE_COUNCIL_LIST_URLS["state_council_doc"]
    url_map = {
        sc_doc_url: _mock_response(200, STATE_COUNCIL_DOC_LIST_HTML),
        ctp.STATE_COUNCIL_LIST_URLS["state_council_meeting"]:
            _mock_response(200, "<html></html>"),
        ctp.STATE_COUNCIL_LIST_URLS["miit_policy"]:
            _mock_response(200, "<html></html>"),
        ctp.STATE_COUNCIL_LIST_URLS["ndrc_policy"]:
            _mock_response(200, "<html></html>"),
        ctp.STATE_COUNCIL_LIST_URLS["mof_policy"]:
            _mock_response(200, "<html></html>"),
        _abs(sc_doc_url, "/zhengce/zhengceku/2026-06/05/content_5800002.html"):
            _mock_response(200, STATE_COUNCIL_DETAIL_HTML),
    }
    fake = _make_http_get(url_map)

    s1 = ctp.collect_state_council(
        start="2026-06-05", end="2026-06-05",
        http_get_fn=fake, output_dir=tmp_path,
    )
    out = tmp_path / "2026-06-05.jsonl"
    assert out.exists()
    lines1 = [
        json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    s2 = ctp.collect_state_council(
        start="2026-06-05", end="2026-06-05",
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
# Test 4 — /-rooted href cross-host safety: gov.cn /-prefixed href
#          must NOT absolutize onto pbc.gov.cn.
# ─────────────────────────────────────────────────────────────────────
def test_slash_rooted_href_resolves_to_list_page_host_not_pbc():
    """A /-rooted href from a gov.cn list page must absolutize onto
    www.gov.cn (or whatever the list page host is), not onto
    www.pbc.gov.cn. Regression guard for the PE-1/PE-2 host-mixing
    bug: the legacy ``_absolutize`` hardcoded PBC_BASE for /-prefixed
    hrefs, which silently corrupted every gov.cn URL.
    """
    base = "http://www.gov.cn/zhengce/zuixin.htm"
    href = "/zhengce/zhengceku/2026-06/05/content_5800002.html"
    resolved = ctp._absolutize(href, base)
    assert resolved.startswith("http://www.gov.cn/"), (
        f"/-rooted href must resolve onto the LIST PAGE host. "
        f"got {resolved}"
    )
    # Sanity: a pbc.gov.cn list-page /-rooted href still absolutizes
    # onto pbc.gov.cn (no regression for PE-1).
    pbc_base = "http://www.pbc.gov.cn/zhengcehuobisi/x/y/index.html"
    pbc_href = "/zhengcehuobisi/something/index.html"
    pbc_resolved = ctp._absolutize(pbc_href, pbc_base)
    assert pbc_resolved.startswith("http://www.pbc.gov.cn/")


# ─────────────────────────────────────────────────────────────────────
# Test 5 — gov.cn detail URL pattern is parsed by parse_list_page.
#          Regression guard: PBC parser only matched /YYYY/MM/DD/ and
#          /<14-digit timestamp>/. gov.cn uses /YYYY-MM/DD/.
# ─────────────────────────────────────────────────────────────────────
def test_parse_list_page_handles_gov_cn_dash_month_url_pattern():
    sample_html = '''
    <html><body>
      <a href="/zhengce/zhengceku/2026-06/05/content_5800002.html">
        国务院政策一
      </a>
      <a href="/zhengce/zhengceku/2026-06/03/content_5800001.html">
        国务院政策二
      </a>
      <a href="/index.html">nav link no date</a>
    </body></html>
    '''
    base = "http://www.gov.cn/zhengce/zuixin.htm"
    links = ctp.parse_list_page(sample_html, base)
    by_date = {ln.publish_date for ln in links}
    assert "2026-06-05" in by_date, f"got dates {by_date}"
    assert "2026-06-03" in by_date, f"got dates {by_date}"
    # All resolved URLs must be gov.cn, not pbc.gov.cn
    for ln in links:
        if "content_5800" in ln.url:
            assert "www.gov.cn" in ln.url, ln.url
