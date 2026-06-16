# gov.cn sousuo API research & ministry-portal fallback design — 2026-06-16

## TL;DR

The public `https://sousuo.www.gov.cn/search-gov/data` endpoint is **NOT
usable as a JSON data source**. It accepts our requests (HTTP 200) and even
returns aggregate facet counts in `extendresult.groupMap` / `facetMap`, but
the actual document list (`searchVO.listVO`) is **always empty** for any
unauthenticated client. The real query traffic the SPA fires goes through
a different backend (`https://sousuoht.www.gov.cn/athena/forward/<HMAC-token>`)
gated by an RSA-encrypted `athenaAppKey` (Tt) + `T-KEY` / `T-SEC` / `T-T` /
`T-SIGN` HMAC headers, all generated client-side from a public key embedded
in `sousuo/search.js`. That auth chain is brittle anti-bot — keys rotate
and there's no contractual guarantee.

**Recommendation: abandon the sousuo API and replace the 4 dead gov.cn
list URLs with each ministry's own static portal**, all of which still
serve parseable HTML in the gov.cn `/202YMM/tYYYYMMDD_NNNNN.html` and
`/zhengce/content/YYYYMM/content_NNN.htm` patterns the existing
`parse_list_page` already handles.

## 1. sousuo API probing — what we confirmed

### What works
- `GET https://sousuo.www.gov.cn/search-gov/data?t=zhengcelibrary&q=<keyword>&p=1&n=N`
  returns `code=200` and **real aggregate counts** in
  `searchVO.extendresult.groupMap` (e.g. for `q=国发`: 国令 452 / 国发 1128 /
  国办发 1946 / etc., total 5466 in `count`).
- The facet shape `facetMap.{tsbq, pubtimeyear, bmfl, parentid, nodeid}` is
  identical across `t=zhengcelibrary`, `t=gongwen`, `t=zhengce`, `t=gwyzc`,
  `t=changwu`, `t=gwymeeting` — they all hit the same index.
- `bmfl` (部门分类) facet enumerates the ministries we want:
  `工业和信息化部 1383条`, `国家发展和改革委员会 1577条`, `财政部 2206条`,
  `中国人民银行 431条`, etc.

### What does NOT work
- `searchVO.listVO` is always `[]` and `totalCount` is always `0`, regardless
  of:
  - `q=` vs `searchWord=` (53 mentions of `searchWord` in `search.js`)
  - non-empty keyword (`q=人工智能`, `q=国发`, `q=通知`)
  - `dataTypeId=107` (general policy id, hardcoded in gov.cn's global header
    form: `goSearch()` opens `sousuo.www.gov.cn/sousuo/search.shtml?code=17da70961a7&dataTypeId=107&searchWord=…`)
  - `code=17da70961a7` + matching `Referer` + `Origin` + full Chrome
    `sec-ch-ua-*` fingerprint
  - POST `application/x-www-form-urlencoded` instead of GET
  - Date filters `mintime=YYYYMMDD&maxtime=YYYYMMDD` (confirmed in bundle)
- No session cookies are issued by `sousuo.www.gov.cn/sousuo/search.shtml`,
  so there's nothing to replay.

```bash
# Reproduces the "facets work, list withheld" behaviour:
curl -sS -G 'https://sousuo.www.gov.cn/search-gov/data' \
  --data-urlencode 't=zhengcelibrary' \
  --data-urlencode 'q=国发' \
  --data-urlencode 'p=1' --data-urlencode 'n=2' \
  -H 'User-Agent: Mozilla/5.0' | python3 -c \
  "import sys,json; d=json.load(sys.stdin); v=d['searchVO']; \
   print('total=', v['totalCount'], 'list=', len(v['listVO'] or []), \
         'groupMap=', v['extendresult']['groupMap'])"
# → total= 0 list= 0 groupMap= {'国发': 1128, '国办发': 1946, ...}
```

### Why listVO is gated — evidence from `sousuo/search.js`

The SPA's real query does **not** call `search-gov/data`. It calls:

```text
_t = "https://sousuoht.www.gov.cn"
url = _t + "/athena/forward/<HMAC-style 32–96 hex char token>"
headers = { "T-KEY": a, "T-SEC": c, "T-T": s,
            "athenaAppKey": Tt, "athenaAppName": xt,
            "T-SIGN": Dt(a, c, s) }
```

where `Tt = encodeURIComponent(RSA.encrypt(athenaAppKey, embedded-public-key))`
and `Dt()` is a client-side HMAC. `sousuoht.www.gov.cn` directly returns 404
to any path we tried without these signed headers. **This is anti-bot
authentication.** Reverse-engineering it is feasible but fragile: the public
key, app key, and forward token are rotation-vulnerable, and we'd be one
silent key rotation away from another 0-row outage.

## 2. The fallback — ministry portals are still scrapable

All four "missing" categories have a parseable home on either the original
ministry portal or on a still-live `www.gov.cn` sub-path. Verified
2026-06-16:

| policy_type             | List URL (proposed)                                            | Verified |
|-------------------------|----------------------------------------------------------------|----------|
| `state_council_doc`     | `https://www.gov.cn/zhengce/index.htm`                         | 200, 53 article links, pattern `./YYYYMM/content_NNN.htm` redirecting to `https://www.gov.cn/zhengce/content/YYYYMM/content_NNN.htm` (verified one fetch → `<title>国务院关于印发《现代化应急体系建设"十五五"规划》的通知_…</title>`, 55 KB body) |
| `state_council_meeting` | `https://www.gov.cn/zhengce/index.htm` filtered by title keyword `国务院常务会议 / 国务院全体会议 / 国务院专题会议` | same list page — meeting summaries are interleaved with policy docs; reuse `_title_matches_*` helper |
| `miit_policy`           | `https://www.miit.gov.cn/zwgk/zcwj/wjfb/index.html`            | 200, sub-index lists year-month archives; per-article pattern `./art/YYYY/M/D/art_<id>_<seq>.html` |
| `ndrc_policy`           | `https://www.ndrc.gov.cn/xxgk/zcfb/tz/`                        | 200 (43 KB), link pattern `./YYYYMM/tYYYYMMDD_NNNNNNN.html` (verified `./202606/t20260615_1405852.html`) |
| `mof_policy`            | `https://www.mof.gov.cn/zhengwuxinxi/zhengcefabu/`             | 200 (21 KB), link pattern `http://<sub>.mof.gov.cn/.../202YMM/tYYYYMMDD_NNNNNNN.htm` (cross-subdomain) |

**Anti-bot status**: all 5 portals respond to a vanilla
`User-Agent: Mozilla/5.0` GET. `www.gov.cn/lianbo/bumen/` and
`www.gov.cn/guowuyuan/gwy_cwh/` are gone (403/redirect/SPA-shell), but
those were the redundant ministry-aggregation pages — the underlying
ministry portals haven't moved.

## 3. Proposed rewrite of `STATE_COUNCIL_LIST_URLS`

### 3a. Replace the dict with a multi-URL-per-key registry

`scripts/collect_policy_texts.py` lines 167–175. Today it's
`dict[str, str]`. Change to `dict[str, list[str]]` so each policy_type can
fan out across sub-indices (MIIT in particular has multiple `/zwgk/` sub-
directories):

```python
STATE_COUNCIL_LIST_URLS: dict[str, list[str]] = {
    "state_council_doc": [
        "https://www.gov.cn/zhengce/index.htm",
    ],
    "state_council_meeting": [
        "https://www.gov.cn/zhengce/index.htm",  # shared list, filter by title
    ],
    "miit_policy": [
        "https://www.miit.gov.cn/zwgk/zcwj/wjfb/index.html",
        # add more as we map them: tzgg, gzdt, etc.
    ],
    "ndrc_policy": [
        "https://www.ndrc.gov.cn/xxgk/zcfb/tz/",       # 通知
        "https://www.ndrc.gov.cn/xxgk/zcfb/ghxwj/",    # 规范性文件 (verify)
    ],
    "mof_policy": [
        "https://www.mof.gov.cn/zhengwuxinxi/zhengcefabu/",
    ],
}
```

A keyword filter — paralleling `NBS_TITLE_KEYWORDS` /
`_title_matches_series` already at lines 218–230 / 878–889 — splits the
shared `/zhengce/index.htm` page into `state_council_doc` vs
`state_council_meeting`:

```python
STATE_COUNCIL_TITLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "state_council_meeting": ("国务院常务会议", "国务院全体会议",
                              "国务院专题会议", "李强主持召开"),
    # state_council_doc: catch-all (no keyword → True)
}
```

### 3b. Injection point in `collect_state_council`

The existing loop at lines 776–789 reads `STATE_COUNCIL_LIST_URLS.get(pt)`
expecting a `str`. Two minimal changes:

1. Iterate over a list of URLs per `pt` (not a single URL).
2. After `parse_list_page(list_html, list_url)` at line 790, apply the
   title-keyword filter (same helper shape as `_title_matches_series`).
3. **No changes** to `parse_list_page`, `parse_article_page`,
   `_save_raw_html_on_failure`, the per-day write loop, or the JSONL row
   schema. The MIIT/NDRC/MOF article URL patterns
   (`/YYYYMM/tYYYYMMDD_NNNNNNN.htm`, `art/YYYY/M/D/art_...html`,
   `/zhengce/content/YYYYMM/content_NNN.htm`) all already match
   `parse_list_page`'s regex fan-out — the comment at lines 141–151
   explicitly notes the parser handles both the "leading
   `/zhengce/zhengceku/...` State Council layout and the MIIT
   `/n.../c.../content.html` layout".

### 3c. JSONL row shape — unchanged

The existing schema at line 843–851 already carries everything we need:

```python
{
    "publish_date": publish_date,   # ← from parse_article_page / list hint
    "policy_type": pt,              # state_council_doc / _meeting / miit_policy / ndrc_policy / mof_policy
    "title": ln.title,
    "url": ln.url,
    "content": content,
    "source": "gov.cn",             # consider widening to "gov.cn/miit"|"gov.cn/ndrc"|"gov.cn/mof"
    "fetch_time": _now_utc_iso(),
}
```

Optional: set `source` from a per-pt map (`{"miit_policy": "miit.gov.cn"}`)
so downstream provenance is faithful, but the existing extractor keys off
`policy_type`, not `source`, so this is cosmetic.

### 3d. PIT discipline

- `publish_date` must remain the **content date** (`ln.publish_date or
  article_date`, lines 822–832), never `fetch_time`. This is already
  correct in the existing code; the rewrite preserves it because
  `parse_list_page` extracts the date from the URL itself
  (`/202606/t20260615_…`).
- `fetch_time` is the asof-overlay watermark on the JSONL side; that's
  fine — downstream `factors/llm_event_extractor_v2.py` uses
  `publish_date` as the `asof_time` for feature joins.
- The MOF cross-subdomain links (`jjs.mof.gov.cn`, `gss.mof.gov.cn`,
  `sbs.mof.gov.cn`) must be allowed by the URL filter in `parse_list_page`
  — they're plain `http://...mof.gov.cn/...t20260604_3991164.htm` patterns
  that already match the timestamp regex; verify the host-allowlist (if
  any) doesn't reject them.

## 4. Risks & follow-ups

1. **MIIT sub-portal moves**. `/zwgk/zcwj/wjfb/index.html` is a known
   stable URL but MIIT has restructured before — keep `RAW_HTML_DIR_SC`
   dumps on parse failure (already present at line 824) and watch for the
   E.2 SLA breach. Recommend the SLA budget stay at the current 35 days
   from `config/data_sla.py`.
2. **NDRC pagination**. The `tz/` index shows ~10 items per page; gov.cn
   policy publish cadence is bursty but with monthly volume around 20+,
   one page may miss tail items. Add `index_1.html`, `index_2.html` to the
   URL list once we measure the cadence on real data.
3. **MOF subdomain whitelist**. If `parse_list_page` currently enforces a
   host match to the list-URL host, cross-subdomain MOF links will be
   dropped. Confirm before merging.
4. **Don't pursue the sousuo Athena gateway.** If a stronger coverage of
   ministry-by-doc-number filtering is ever needed, prefer subscribing to
   each ministry's RSS / `xxgk` (政府信息公开) JSON feed — those are
   stable contracts. Reverse-engineering the SPA auth chain is dev cost
   we shouldn't pay against rotation risk.

## Appendix — commands used

```bash
# Confirms list withheld but facets returned
curl -sS -G 'https://sousuo.www.gov.cn/search-gov/data' \
  --data-urlencode 't=zhengcelibrary' --data-urlencode 'q=国发' \
  --data-urlencode 'p=1' --data-urlencode 'n=2' -H 'User-Agent: Mozilla/5.0'

# Confirms the SPA real backend is sousuoht.www.gov.cn behind Athena auth
grep -oE 'sousuoht\.www\.gov\.cn|/athena/forward/[A-F0-9]+|athenaAppKey' \
  /tmp/search.js

# Confirms ministry portals still serve static HTML
curl -sS 'https://www.ndrc.gov.cn/xxgk/zcfb/tz/' -H 'User-Agent: Mozilla/5.0' \
  | grep -oE 'href="\./202606/t[0-9_]+\.html"' | head
curl -sS 'https://www.mof.gov.cn/zhengwuxinxi/zhengcefabu/' \
  -H 'User-Agent: Mozilla/5.0' \
  | grep -oE 'href="http://[a-z]+\.mof\.gov\.cn/[^"]+\.htm"' | head
curl -sS 'https://www.gov.cn/zhengce/index.htm' -H 'User-Agent: Mozilla/5.0' \
  | grep -oE 'href="\./202606/content_[0-9]+\.htm"' | head
```
