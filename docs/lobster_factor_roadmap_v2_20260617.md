# Lobster Factor Roadmap v2 — Full Sweep Results

**Generated**: 2026-06-17 01:15 (overnight)
**Source**: full 244-endpoint sweep via lobster client + lobster token

## Summary

| Tag | Count |
|---|---|
| `OK_LIST` | 217 |
| `UNKNOWN` | 9 |
| `ARG` | 7 |
| `HTTP4` | 6 |
| `NO_METHOD` | 3 |
| `ERR` | 1 |
| `HTTP5` | 1 |

## Working endpoints by category

### 财务三表 (16)

- `balancesheet` — n=101 kw={'ts_code': '000001.SZ'}
- `balancesheet_vip` — n=101 kw={'ts_code': '000001.SZ'}
- `cashflow` — n=82 kw={'ts_code': '000001.SZ'}
- `cashflow_vip` — n=82 kw={'ts_code': '000001.SZ'}
- `disclosure_date` — n=2 kw={'ts_code': '000001.SZ'}
- `express` — n=4 kw={'ts_code': '000001.SZ'}
- `express_vip` — n=4 kw={'ts_code': '000001.SZ'}
- `fina_audit` — n=39 kw={'ts_code': '000001.SZ'}
- `fina_indicator` — n=67 kw={'ts_code': '000001.SZ'}
- `fina_indicator_vip` — n=67 kw={'ts_code': '000001.SZ'}
- `fina_mainbz` — n=274 kw={'ts_code': '000001.SZ'}
- `fina_mainbz_vip` — n=274 kw={'ts_code': '000001.SZ'}
- `forecast` — n=16 kw={'ts_code': '000001.SZ'}
- `forecast_vip` — n=16 kw={'ts_code': '000001.SZ'}
- `income` — n=123 kw={'ts_code': '000001.SZ'}
- `income_vip` — n=123 kw={'ts_code': '000001.SZ'}

### 可转债 (7)

- `cb_basic` — n=0 kw={'ts_code': '000001.SZ'}
- `cb_call` — n=0 kw={'ts_code': '000001.SZ'}
- `cb_daily` — n=0 kw={'ts_code': '000001.SZ'}
- `cb_factor_pro` — n=320 kw={'trade_date': '20260615'}
- `cb_issue` — n=1 kw={'ann_date': '20260615'}
- `cb_rate` — n=0 kw={'ts_code': '000001.SZ'}
- `cb_share` — n=0 kw={'ts_code': '000001.SZ'}

### 港股 (15)

- `ccass_hold` — n=945 kw={'trade_date': '20260615'}
- `ccass_hold_detail` — n=0 kw={'ts_code': '000001.SZ', 'trade_date': '20260615'}
- `ggt_daily` — n=1 kw={'trade_date': '20260615'}
- `ggt_monthly` — n=0 kw={'trade_date': '20260615'}
- `hk_adjfactor` — n=0 kw={'ts_code': '000001.SZ'}
- `hk_balancesheet` — n=0 kw={'ts_code': '000001.SZ'}
- `hk_basic` — n=3554 kw=no-arg
- `hk_cashflow` — n=0 kw={'ts_code': '000001.SZ'}
- `hk_daily` — n=0 kw={'trade_date': '20260615'}
- `hk_daily_adj` — n=0 kw={'trade_date': '20260615'}
- `hk_fina_indicator` — n=0 kw={'ts_code': '000001.SZ'}
- `hk_hold` — n=944 kw={'trade_date': '20260615'}
- `hk_income` — n=0 kw={'ts_code': '000001.SZ'}
- `hk_mins` — n=0 kw={'ts_code': '000001.SZ'}
- `hk_tradecal` — n=0 kw={'start_date': '20260615', 'end_date': '20260615'}

### 美股 (12)

- `us_adjfactor` — n=0 kw={'ts_code': '000001.SZ'}
- `us_balancesheet` — n=0 kw={'ts_code': '000001.SZ'}
- `us_cashflow` — n=0 kw={'ts_code': '000001.SZ'}
- `us_daily` — n=0 kw={'ts_code': '000001.SZ'}
- `us_daily_adj` — n=0 kw={'trade_date': '20260615'}
- `us_fina_indicator` — n=0 kw={'ts_code': '000001.SZ'}
- `us_income` — n=0 kw={'ts_code': '000001.SZ'}
- `us_tbr` — n=1 kw={'start_date': '20260615', 'end_date': '20260615'}
- `us_tltr` — n=1 kw={'start_date': '20260615', 'end_date': '20260615'}
- `us_trltr` — n=1 kw={'start_date': '20260615', 'end_date': '20260615'}
- `us_trycr` — n=1 kw={'start_date': '20260615', 'end_date': '20260615'}
- `us_tycr` — n=1 kw={'start_date': '20260615', 'end_date': '20260615'}

### 基金 (11)

- `fund_adj` — n=0 kw={'ts_code': '000001.SZ'}
- `fund_basic` — n=0 kw={'ts_code': '000001.SZ'}
- `fund_company` — n=15279 kw=no-arg
- `fund_daily` — n=0 kw={'ts_code': '000001.SZ'}
- `fund_div` — n=0 kw={'ts_code': '000001.SZ'}
- `fund_factor_pro` — n=0 kw={'trade_date': '20260615'}
- `fund_manager` — n=0 kw={'ts_code': '000001.SZ'}
- `fund_nav` — n=0 kw={'ts_code': '000001.SZ'}
- `fund_portfolio` — n=0 kw={'ts_code': '000001.SZ'}
- `fund_sales_vol` — n=500 kw=no-arg
- `fund_share` — n=0 kw={'ts_code': '000001.SZ'}

### 期货 (8)

- `ft_limit` — n=869 kw={'trade_date': '20260615'}
- `ft_mins` — n=0 kw={'ts_code': '000001.SZ'}
- `fut_basic` — n=10000 kw=no-arg
- `fut_daily` — n=1075 kw={'trade_date': '20260615'}
- `fut_mapping` — n=202 kw={'trade_date': '20260615'}
- `fut_settle` — n=869 kw={'trade_date': '20260615'}
- `fut_weekly_monthly` — n=0 kw={'ts_code': '000001.SZ'}
- `fut_wsr` — n=816 kw={'trade_date': '20260615'}

### ETF (7)

- `etf_basic` — n=3449 kw=no-arg
- `etf_index` — n=1495 kw=no-arg
- `etf_mins` — n=0 kw={'ts_code': '000001.SZ'}
- `etf_share_size` — n=0 kw={'ts_code': '000001.SZ'}
- `rt_etf_k` — n=0 kw={'ts_code': '000001.SZ'}
- `rt_etf_min` — n=0 kw={'ts_code': '000001.SZ', 'freq': '1min'}
- `rt_etf_tick` — n=0 kw={'ts_code': '000001.SZ'}

### 指数/板块 (28)

- `ci_daily` — n=437 kw={'trade_date': '20260615'}
- `ci_index_member` — n=1 kw={'ts_code': '000001.SZ'}
- `dc_daily` — n=1021 kw={'trade_date': '20260615'}
- `dc_hot` — n=0 kw={'trade_date': '20260615'}
- `dc_index` — n=494 kw={'trade_date': '20260615'}
- `dc_member` — n=0 kw={'ts_code': '000001.SZ'}
- `idx_factor_pro` — n=0 kw={'trade_date': '20260615'}
- `idx_mins` — n=0 kw={'ts_code': '000001.SZ'}
- `index_basic` — n=8000 kw=no-arg
- `index_classify` — n=0 kw={'src': 'em'}
- `index_daily` — n=0 kw={'ts_code': '000001.SZ'}
- `index_dailybasic` — n=6 kw={'trade_date': '20260615'}
- `index_global` — n=0 kw={'ts_code': '000001.SZ'}
- `index_member_all` — n=1 kw={'ts_code': '000001.SZ'}
- `index_monthly` — n=0 kw={'ts_code': '000001.SZ'}
- `index_weekly` — n=0 kw={'ts_code': '000001.SZ'}
- `index_weight` — n=0 kw={'start_date': '20260615', 'end_date': '20260615'}
- `kpl_concept` — n=0 kw={'trade_date': '20260615'}
- `kpl_concept_cons` — n=581 kw={'trade_date': '20260615'}
- `kpl_list` — n=159 kw={'trade_date': '20260615'}
- `sw_daily` — n=0 kw={'trade_date': '20260615'}
- `tdx_daily` — n=616 kw={'trade_date': '20260615'}
- `tdx_index` — n=0 kw={'trade_date': '20260615'}
- `tdx_member` — n=762 kw={'trade_date': '20260615'}
- `ths_daily` — n=3000 kw=no-arg
- `ths_hot` — n=443 kw={'trade_date': '20260615'}
- `ths_index` — n=2010 kw=no-arg
- `ths_member` — n=0 kw={'ts_code': '000001.SZ'}

### 股票基础 (22)

- `namechange` — n=4 kw={'ts_code': '000001.SZ'}
- `stk_account` — n=0 kw={'start_date': '20260615', 'end_date': '20260615'}
- `stk_account_old` — n=0 kw={'start_date': '20260615', 'end_date': '20260615'}
- `stk_ah_comparison` — n=190 kw={'trade_date': '20260615'}
- `stk_auction` — n=339 kw={'ts_code': '000001.SZ'}
- `stk_auction_c` — n=5526 kw={'trade_date': '20260615'}
- `stk_auction_o` — n=5489 kw={'trade_date': '20260615'}
- `stk_factor_pro` — n=1 kw={'ts_code': '000001.SZ', 'trade_date': '20260615'}
- `stk_holdernumber` — n=13 kw={'ts_code': '000001.SZ'}
- `stk_holdertrade` — n=2 kw={'ts_code': '000001.SZ'}
- `stk_limit` — n=1100 kw={'ts_code': '000001.SZ'}
- `stk_managers` — n=193 kw={'ts_code': '000001.SZ'}
- `stk_mins` — n=0 kw={'ts_code': '000001.SZ'}
- `stk_nineturn` — n=5507 kw={'trade_date': '20260615'}
- `stk_premarket` — n=5512 kw={'trade_date': '20260615'}
- `stk_rewards` — n=2773 kw={'ts_code': '000001.SZ'}
- `stk_surv` — n=0 kw={'trade_date': '20260615'}
- `stk_week_month_adj` — n=0 kw={'trade_date': '20260615'}
- `stk_weekly_monthly` — n=6000 kw=no-arg
- `stock_basic` — n=4 kw=no-arg
- `stock_company` — n=4 kw={'ts_code': '000001.SZ'}
- `stock_st` — n=0 kw={'trade_date': '20260615'}

### 资金流/龙虎榜 (17)

- `block_trade` — n=239 kw={'trade_date': '20260615'}
- `hm_detail` — n=274 kw={'trade_date': '20260615'}
- `hm_list` — n=110 kw=no-arg
- `limit_cpt_list` — n=20 kw={'trade_date': '20260615'}
- `limit_list_d` — n=191 kw={'trade_date': '20260615'}
- `limit_list_ths` — n=160 kw={'trade_date': '20260615'}
- `limit_step` — n=0 kw={'trade_date': '20260615'}
- `moneyflow` — n=4004 kw={'ts_code': '000001.SZ'}
- `moneyflow_cnt_ths` — n=383 kw={'trade_date': '20260615'}
- `moneyflow_dc` — n=5903 kw={'trade_date': '20260615'}
- `moneyflow_hsgt` — n=1 kw={'trade_date': '20260615'}
- `moneyflow_ind_dc` — n=0 kw={'trade_date': '20260615'}
- `moneyflow_ind_ths` — n=90 kw={'trade_date': '20260615'}
- `moneyflow_mkt_dc` — n=1 kw={'trade_date': '20260615'}
- `moneyflow_ths` — n=356 kw={'ts_code': '000001.SZ'}
- `top_inst` — n=849 kw={'trade_date': '20260615'}
- `top_list` — n=3071 kw={'trade_date': '20260615'}

### 实时 (10)

- `rt_fut_min` — n=0 kw={'ts_code': '000001.SZ', 'freq': '1min'}
- `rt_hk_k` — n=0 kw={'ts_code': '000001.SZ'}
- `rt_hk_tick` — n=0 kw={'ts_code': '000001.SZ'}
- `rt_idx_k` — n=0 kw={'ts_code': '000001.SZ'}
- `rt_idx_min` — n=0 kw={'ts_code': '000001.SZ', 'freq': 'D'}
- `rt_idx_tick` — n=0 kw={'ts_code': '000001.SZ'}
- `rt_k` — n=0 kw={'ts_code': '00700.HK'}
- `rt_min` — n=0 kw={'ts_code': '000001.SZ', 'freq': 'D'}
- `rt_sw_k` — n=1 kw={'ts_code': '000001.SZ'}
- `rt_tick` — n=0 kw=no-arg

### 宏观 (12)

- `cn_m` — n=580 kw={'start_date': '20260615', 'end_date': '20260615'}
- `cn_pmi` — n=197 kw={'start_date': '20260615', 'end_date': '20260615'}
- `fx_daily` — n=0 kw={'ts_code': '000001.SZ'}
- `gz_index` — n=0 kw={'trade_date': '20260615'}
- `hibor` — n=0 kw={'start_date': '20260615', 'end_date': '20260615'}
- `libor` — n=0 kw={'start_date': '20260615', 'end_date': '20260615'}
- `sge_basic` — n=13 kw=no-arg
- `sge_daily` — n=41 kw={'trade_date': '20260615'}
- `shibor` — n=1 kw={'start_date': '20260615', 'end_date': '20260615'}
- `shibor_lpr` — n=0 kw={'start_date': '20260615', 'end_date': '20260615'}
- `shibor_quote` — n=17 kw={'date': '20260615'}
- `wz_index` — n=0 kw={'trade_date': '20260615'}

### 公司行动 (6)

- `dividend` — n=29 kw={'ts_code': '000001.SZ'}
- `new_share` — n=2000 kw={'ts_code': '000001.SZ'}
- `pledge_detail` — n=0 kw={'ts_code': '000001.SZ'}
- `pledge_stat` — n=12 kw={'ts_code': '000001.SZ'}
- `repurchase` — n=0 kw={'ts_code': '000001.SZ'}
- `share_float` — n=4 kw={'ts_code': '000001.SZ'}

### 另类 (15)

- `anns_d` — n=375 kw={'ts_code': '000001.SZ'}
- `bo_cinema` — n=0 kw={'date': '20260615'}
- `bo_daily` — n=0 kw={'date': '20260615'}
- `bo_monthly` — n=0 kw={'date': '20260615'}
- `bond_blk` — n=0 kw={'start_date': '20260615', 'end_date': '20260615'}
- `broker_recommend` — n=341 kw={'month': '202606'}
- `cctv_news` — n=11 kw={'date': '20260615'}
- `irm_qa_sh` — n=0 kw={'ts_code': '000001.SZ'}
- `irm_qa_sz` — n=277 kw={'ts_code': '000001.SZ'}
- `major_news` — n=0 kw={'src': 'sina'}
- `opt_basic` — n=12000 kw=no-arg
- `opt_daily` — n=15000 kw={'trade_date': '20260615'}
- `opt_mins` — n=0 kw={'ts_code': '000001.SZ'}
- `report_rc` — n=406 kw={'ts_code': '000001.SZ'}
- `research_report` — n=241 kw={'ts_code': '000001.SZ'}

### 票房 (2)

- `film_record` — n=0 kw={'start_date': '20260615', 'end_date': '20260615'}
- `teleplay_record` — n=0 kw={'start_date': '20260615', 'end_date': '20260615'}

### 筹码/CYQ (2)

- `cyq_chips` — n=0 kw={'start_date': '20260615', 'end_date': '20260615'}
- `cyq_perf` — n=715 kw={'ts_code': '000001.SZ'}

### 调节 (3)

- `margin` — n=3 kw={'trade_date': '20260615'}
- `margin_secs` — n=4052 kw={'trade_date': '20260615'}
- `stock_hsgt` — n=191 kw={'ts_code': '000001.SZ'}

### 其它 (24)

- `adj_factor` — n=5530 kw={'trade_date': '20260615'}
- `bak_basic` — n=2367 kw={'ts_code': '000001.SZ'}
- `bak_daily` — n=5525 kw={'trade_date': '20260615'}
- `bc_bestotcqt` — n=0 kw={'ts_code': '000001.SZ'}
- `bc_otcqt` — n=0 kw={'ts_code': '000001.SZ'}
- `bse_mapping` — n=248 kw=no-arg
- `daily` — n=5654 kw={'ts_code': '000001.SZ'}
- `daily_basic` — n=3 kw={'ts_code': '000001.SZ', 'trade_date': '20260615'}
- `daily_info` — n=12 kw={'trade_date': '20260615'}
- `eco_cal` — n=0 kw={'date': '20260615'}
- `monthly` — n=0 kw={'trade_date': '20260615'}
- `npr` — n=0 kw={'start_date': '20260615', 'end_date': '20260615'}
- `pro_bar` — n=1 kw={'ts_code': '000001.SZ', 'start_date': '20260615', 'end_date': '20260615'}
- `repo_daily` — n=50 kw={'trade_date': '20260615'}
- `slb_len` — n=0 kw={'trade_date': '20260615'}
- `slb_len_mm` — n=0 kw={'trade_date': '20260615'}
- `slb_sec` — n=0 kw={'trade_date': '20260615'}
- `suspend_d` — n=0 kw=no-arg
- `sz_daily_info` — n=14 kw={'trade_date': '20260615'}
- `tmt_twincome` — n=0 kw={'start_date': '20260615', 'end_date': '20260615'}
- `tmt_twincomedetail` — n=0 kw={'start_date': '20260615', 'end_date': '20260615'}
- `trade_cal` — n=5217 kw=no-arg
- `weekly` — n=0 kw={'trade_date': '20260615'}
- `yc_cb` — n=0 kw={'trade_date': '20260615'}

## Missing kwargs (server demands more params)

- `cn_cpi` — 参数不能为空
- `cn_gdp` — 参数不能为空
- `cn_ppi` — 参数不能为空
- `fund_sales_ratio` — 参数不能为空
- `fx_obasic` — 参数不能为空
- `realtime_list` — 参数不能为空
- `sf_month` — 参数不能为空

## HTTP errors (gateway broken on lobster tier)

- `fut_holding` — HTTP4 HTTP 400
- `fut_weekly_detail` — HTTP4 HTTP 400
- `news` — HTTP4 HTTP 400
- `realtime_quote` — HTTP4 HTTP 404
- `realtime_tick` — HTTP4 HTTP 404
- `rt_sw_tick` — HTTP4 HTTP 400
- `cb_price_chg` — HTTP5 HTTP 500

## Other errors (per-stock or odd response)

- `bo_weekly` — ERR code=1 msg=接口调用失败
- `bond_blk_detail` — UNKNOWN code=None msg=
- `dc_concept` — UNKNOWN code=None msg=
- `dc_concept_cons` — UNKNOWN code=None msg=
- `margin_detail` — UNKNOWN code=None msg=
- `slb_sec_detail` — UNKNOWN code=None msg=
- `st` — UNKNOWN code=None msg=
- `ths_news` — UNKNOWN code=None msg=
- `us_basic` — UNKNOWN code=None msg=
- `us_tradecal` — UNKNOWN code=None msg=

## Production-grade factor candidates (Tier 1)

Endpoints with `_vip` variants are bulk-by-period — most efficient. Standard endpoints are per-stock or per-date and need batched call orchestration.

| Endpoint | Bulk | Factor purpose |
|---|---|---|
| `forecast_vip` | Y (period) | SUE/PEAD — Bernard&Thomas 1989 |
| `balancesheet_vip` | Y (period) | Total assets / debt / equity ratios |
| `cashflow_vip` | Y (period) | Operating CF / NI ratio (Sloan accruals) |
| `income_vip` | Y (period) | Revenue / NI growth |
| `fina_indicator_vip` | Y (period) | ROE / ROA / margins |
| `dividend` | N (ts_code) | DP yield |
| `repurchase` | N | Insider conviction |
| `share_float` | N (ann_date) | Selling pressure / liquidity |
| `stk_holdertrade` | N | Insider net change — replaces LLM-Ext #3 |
| `report_rc` | N (ann_date) | Analyst revision — replaces LLM-Ext #1 |
| `top_list` | N (trade_date) | Hot money sentiment |
| `top_inst` | N (trade_date) | Institutional net buying |

## Followup

- Backfill the `_vip` endpoints to research parquets — overnight runner step 3.
- After backfill, write a SUE/PEAD factor computation script (per ann_date PIT).
- Then dual-client integration (task #198 / #195) to wire factors into FeatureMerger.
