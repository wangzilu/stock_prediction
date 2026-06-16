# ST_CLIENT Endpoint Audit — 2026-06-16

Background: investigation of today's spot collector failure
(`Both AKShare AND ST_CLIENT spot failed → Tencent partial 50`) revealed
`ST_CLIENT.realtime_list()` returns `{"code": 1, "msg": "该接口为龙虾套餐专属"}` on
the current subscription tier. This audit checks which ST endpoints the
codebase calls and whether each is reachable under the current tier.

## Method

Probed each endpoint from the codebase grep with a recent business date
(`trade_date=20260515`); recorded HTTP code, payload shape, and ST envelope
`code`/`msg`. Endpoints requiring different keyword args (e.g. `ann_date`,
`ts_code`) fall in the "kwarg-contract" bucket — not a tier issue.

## Findings

### Available (current tier) — 13 endpoints
| Endpoint            | Used by                                          |
|---------------------|--------------------------------------------------|
| `adj_factor`        | qlib_data_update, daily-factor backfills          |
| `bak_basic`         | qlib_data_update                                  |
| `block_trade`       | round-7 collectors                                |
| `daily`             | qlib_data_update primary path                     |
| `daily_basic`       | `fetch_st_daily_factors.py`                       |
| `fut_daily`         | regime futures fallback                           |
| `hk_hold`           | northbound flow                                   |
| `limit_list_d`      | regime margin/limit/hsgt critical                 |
| `margin_detail`     | regime margin/limit/hsgt critical (T+1 lag)       |
| `moneyflow`         | st_moneyflow_update                               |
| `moneyflow_hsgt`    | regime critical                                   |
| `moneyflow_ind_dc`  | sector flow                                       |
| `stock_basic`       | universe definition                               |
| `trade_cal`         | calendar generation                               |

### Tier-limited — 1 endpoint
| Endpoint            | Error                                                    | Used by              |
|---------------------|----------------------------------------------------------|----------------------|
| `realtime_list`     | `{code:1, msg:"该接口为龙虾套餐专属，请升级套餐后使用"}`  | `data/collectors/market.py` spot fallback Layer 2 |

**Action:** keep AKShare eastmoney as the spot collector primary
(`stock_zh_a_spot_em`). `realtime_list` stays as fallback Layer 2 with the
new tier-aware silent-skip handler (see `_load_spot_stclient` in market.py
post-2026-06-16). Upgrade to 龙虾 tier would let us flip ST as primary; not
worth doing while AKShare is reliable when proxies are cleared
(`run_network_job --network domestic`).

### Kwarg-contract differences — 7 endpoints
These return `TypeError: got unexpected keyword argument 'trade_date'`
when probed with `trade_date=`. They likely take `ann_date`, `ts_code`, or
`period` instead — NOT a tier issue. Listed here for completeness; check
each call site uses the right kwarg.

- `anns_d`, `broker_recommend`, `forecast`, `fund_portfolio`,
  `fund_share`, `irm_qa_sh`, `irm_qa_sz`, `stk_holdertrade`

### Other errors
- `cyq_perf` → HTTP 400 on 20260515 (intermittent; may work on other dates)
- `stk_factor_pro` → code=None on 20260515 (no data envelope; date-specific)
- `round_trip_rate` → `no-such-method` (endpoint not on client; called by `fetch_st_round6.py` — investigate)

## Decision

The 2026-06-09 directive "ST 优先" applies endpoint-by-endpoint, not as a
blanket primary-source swap. Current state:

- ST is **already primary** for: `daily`, `daily_basic`, `moneyflow`,
  `moneyflow_hsgt`, `stock_basic`, `trade_cal`, `adj_factor`,
  `bak_basic`, `hk_hold`, `limit_list_d`, `margin_detail`,
  `moneyflow_ind_dc`, `fut_daily`, `block_trade`.
- ST is **not viable** for spot realtime (`realtime_list`) — tier limit.
  AKShare stays primary here.
- One bug shipped today: regime_daily script was getting the envelope
  shape wrong (`data` is `{fields, items}` dict, not a list) — fixed in
  `update_regime_daily.py` along with the T+1 publish lag.

## Followups

1. **P3** — audit the 7 kwarg-contract endpoints to confirm each call site
   uses the right argument name.
2. **P3** — `round_trip_rate` call in `fetch_st_round6.py` references a
   non-existent method; check if that script still runs.
3. **P2** — if a tier upgrade ever happens, flip `MarketCollector` to
   ST_CLIENT primary (full 5000+ universe in one call beats AKShare
   eastmoney's 100-stock-per-page pagination).
