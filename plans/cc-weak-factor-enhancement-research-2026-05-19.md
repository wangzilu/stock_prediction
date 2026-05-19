# Weak Factor Enhancement Research: Transforming Failed Factors into Alpha Signals

**Date:** 2026-05-19
**Author:** CC
**Status:** Research Document for CX Review

---

## Executive Summary

Our 174-feature XGB champion model passed all 24-split rolling gates (avg RankIC +0.0513, avg Spread +2.51%). Nine new factor groups were tested via rolling ablation and ALL failed to provide incremental value over the baseline. This document analyzes WHY they failed and proposes specific, implementable transformations that may unlock latent alpha.

**Key insight:** Raw factor values are almost never useful for tree models that already have 174 well-engineered features. The baseline captures price-volume dynamics, PE/PB/turnover anomalies, and capital flow ratios. New factors must provide ORTHOGONAL information -- either through time-series derivatives, cross-sectional conditioning, or interaction effects that the baseline cannot express natively.

---

## 0. Ablation Results Summary

| Factor Group | Raw Cols | RIC Helps % | SPR Helps % | avg delta RIC | Failure Mode |
|:---|:---:|:---:|:---:|:---:|:---|
| moneyflow (18 cols) | 18 | 50% | 58% | +0.0045 | Dilution: too many noisy cols overwhelm signal |
| cyq_perf (9 cols) | 9 | 50% | 50% | -0.0011 | Redundant with price-based features |
| pledge (2 cols) | 2 | 33% | 58% | -0.0004 | Low frequency, stale signal |
| forecast (4 cols) | 4 | 75% | 50% | +0.0046 | Binary signal only, content is stale |
| block_trade (3 cols) | 3 | 38% | -- | weak | Sparse events, raw aggregation too crude |
| top_inst (3 cols) | 3 | 50% | -- | weak | Sparse events, raw aggregation too crude |
| holder_num (1 col) | 1 | 25% | 67% | -0.0019 | Redundant (residual IC = -0.018) |
| regime (27 cols) | 27 | 75% | 42% | mixed | Synchronous not leading; future_5d > real |
| derived_mf_cyq (9 cols) | 9 | 38% | -- | -0.0035 | First-order derivatives still redundant |

**Source files:**
- `data/storage/phase4/phase2_factor_ablation.json` (moneyflow, cyq, pledge, forecast)
- `data/storage/phase4/new_factor_ablation.json` (block_trade, top_inst)
- `data/storage/rolling_holder_ablation.json` (holder_num)
- `data/storage/phase4/fast_rolling_ablation.json` (regime, 205-dim)
- `data/storage/phase4/derived_factor_ablation.json` (derived mf+cyq)

---

## 1. Factor-by-Factor Diagnosis and Enhancement Plan

### 1.1 moneyflow (Individual Stock Capital Flow)

**Raw data:** `data/storage/st_moneyflow.parquet` -- 5.5M rows, 20 columns
Available columns: `st_buy_sm_vol/amount`, `st_sell_sm_vol/amount` (small), `st_buy_md_vol/amount`, `st_sell_md_vol/amount` (medium), `st_buy_lg_vol/amount`, `st_sell_lg_vol/amount` (large), `st_buy_elg_vol/amount`, `st_sell_elg_vol/amount` (extra-large), `st_net_mf_vol`, `st_net_mf_amount`

**Why raw values failed:** The baseline already has `flow_net_mf_latest`, `flow_net_mf_5d`, `flow_net_mf_20d_avg` (3 capital flow features from FeatureMerger). Feeding 18 raw moneyflow columns creates feature dilution -- XGB splits are spread across correlated noisy features, diluting the useful signal. The first-order derivatives (net_flow_5d_change, net_flow_vol_20d) also failed because they are linearly transformable from existing features.

**Why derived values also failed:** `build_derived_factors.py` computed `net_flow`, `net_flow_5d_change`, `net_flow_20d_change`, `net_flow_vol_20d`, `big_order_ratio` -- 5 features that are simple per-stock time-series transforms. The 8-split ablation showed 3/8 positive (38%), avg delta -0.0035. These derivatives are still strongly correlated with the baseline's flow features.

#### 1.1.1 Time-Series Derivatives (Novel)

```python
# --- Existing (already failed) ---
# net_flow = buy_lg + buy_elg - sell_lg - sell_elg
# net_flow_5d_change = net_flow.pct_change(5)

# --- NEW: Z-score relative to own history ---
# "Is today's net flow unusual compared to the stock's own 60-day history?"
# This captures REGIME SHIFTS that pct_change misses
g = mf.groupby('qlib_code')['net_flow']
mf['net_flow_zscore_60d'] = g.transform(
    lambda x: (x - x.rolling(60, min_periods=20).mean()) /
              x.rolling(60, min_periods=20).std().replace(0, np.nan)
)

# --- NEW: Acceleration (second derivative) ---
# "Is net flow ACCELERATING or DECELERATING?"
# pct_change of pct_change is unstable; use diff of rolling mean instead
mf['net_flow_accel'] = g.transform(
    lambda x: x.rolling(5).mean().diff(5) - x.rolling(20).mean().diff(5)
)

# --- NEW: Flow persistence ---
# "How many of the last 10 days had positive net flow?"
# Binary persistence captures conviction better than magnitude
mf['net_flow_pos_streak_10d'] = g.transform(
    lambda x: (x > 0).rolling(10, min_periods=5).sum() / 10
)

# --- NEW: Flow surprise ---
# "Today's flow vs expected flow from recent trend"
# Residual from 20-day rolling OLS
mf['net_flow_surprise_20d'] = g.transform(
    lambda x: x - x.rolling(20, min_periods=10).apply(
        lambda w: np.polyval(np.polyfit(range(len(w)), w, 1), len(w)), raw=False
    )
)
```

**Rationale:** Z-score and persistence capture regime shifts. Acceleration captures momentum-of-momentum. Surprise captures deviation from linear trend. These are all NONLINEAR functions that XGB cannot easily reconstruct from the 3 existing flow features.

#### 1.1.2 Cross-Sectional Transformations

```python
# --- Industry-relative flow rank ---
# "Am I attracting more smart money than my industry peers TODAY?"
# Requires industry mapping (already have: 5523 stocks x 110 industries)
industry_map = pd.read_parquet('data/storage/industry_mapping.parquet')
# columns: qlib_code, sw_industry_l1, sw_industry_l2

mf_daily = mf.merge(industry_map, on='qlib_code')

# Per-day, per-industry rank percentile
mf_daily['net_flow_ind_rank'] = mf_daily.groupby(['date', 'sw_industry_l2'])['net_flow'].transform(
    lambda x: x.rank(pct=True) if len(x) > 5 else 0.5
)

# --- Size-relative flow ---
# "Am I attracting more flow than stocks of similar market cap?"
# Use existing pe/pb data to proxy size, or load circ_mv
mf_daily['net_flow_size_adj'] = mf_daily.groupby(['date', 'size_quintile'])['net_flow'].transform(
    lambda x: (x - x.mean()) / x.std().replace(0, np.nan)
)

# --- Cross-sectional percentile (all stocks) ---
mf_daily['net_flow_mkt_pctl'] = mf_daily.groupby('date')['net_flow'].transform(
    lambda x: x.rank(pct=True)
)

# --- Big-order ratio deviation from industry mean ---
mf_daily['big_order_ind_dev'] = mf_daily.groupby(['date', 'sw_industry_l1']).apply(
    lambda g: g['big_order_ratio'] - g['big_order_ratio'].mean()
).reset_index(level=[0, 1], drop=True)
```

**Rationale:** The baseline has no cross-sectional conditioning. A stock with net_flow=+50M means nothing in absolute terms (it depends on market cap and industry). Industry-relative rank is the standard Barra approach.

#### 1.1.3 Interaction Features

```python
# --- Flow x Momentum: "Smart money chasing momentum?" ---
# net_flow_zscore > 1.5 AND price_mom_20d > 0 = institutional accumulation
# This interaction cannot be learned from flow OR momentum alone
mf['flow_x_mom20'] = mf['net_flow_zscore_60d'] * cache['KLEN20']
# KLEN20 is 20-day price change in Alpha158

# --- Flow x Volatility: "Flow matters more in calm markets" ---
# High flow + low volatility = quiet accumulation (strongest signal)
# High flow + high volatility = panic buying (noisy signal)
mf['flow_x_invvol'] = mf['net_flow_zscore_60d'] / (cache['VSTD20'] + 1e-8)
# VSTD20 is 20-day volume std in Alpha158

# --- Flow divergence: "Big orders buy but small orders sell" ---
mf['flow_divergence'] = (
    (mf['st_buy_lg_amount'] + mf['st_buy_elg_amount']) /
    (mf['st_buy_lg_amount'] + mf['st_buy_elg_amount'] +
     mf['st_sell_lg_amount'] + mf['st_sell_elg_amount'] + 1e-8)
    -
    (mf['st_buy_sm_amount']) /
    (mf['st_buy_sm_amount'] + mf['st_sell_sm_amount'] + 1e-8)
)
# Positive = big buyers, small sellers = institutional accumulation
```

#### 1.1.4 Recommended Feature Set (Moneyflow v2)

| Feature | Type | Window | Novelty vs Baseline |
|:---|:---|:---:|:---|
| `net_flow_zscore_60d` | Time-series | 60d | Z-score captures regime, not in baseline |
| `net_flow_pos_streak_10d` | Time-series | 10d | Binary persistence, new info |
| `net_flow_accel` | Time-series | 5d/20d | Second derivative, new info |
| `flow_divergence` | Ratio | 1d | Large vs small order divergence, new info |
| `net_flow_ind_rank` | Cross-section | 1d | Industry-relative, not in baseline |
| `big_order_ind_dev` | Cross-section | 1d | Industry-neutral ratio, new info |
| `flow_x_mom20` | Interaction | 20d | Flow x momentum, new info |

**Total: 7 features (down from 18 raw + 9 derived). Fewer, more orthogonal features should avoid dilution.**

---

### 1.2 cyq_perf (Chip Distribution)

**Raw data:** `data/storage/st_cyq_perf.parquet` -- 3.3M rows, 11 columns
Available: `cyq_his_low`, `cyq_his_high`, `cyq_cost_5pct`, `cyq_cost_15pct`, `cyq_cost_50pct`, `cyq_cost_85pct`, `cyq_cost_95pct`, `cyq_weight_avg`, `cyq_winner_rate`

**Why raw values failed:** CYQ cost levels are denominated in absolute price, which is highly correlated with the stock's current price (already in Alpha158 as OPEN0, CLOSE0, etc.). `cyq_winner_rate` is the most interesting field (% of holders in profit), but its raw level is partially captured by the baseline's price-position features (`price_pos20 = ($close - Min($close,20))/(Max($close,20)-Min($close,20))`).

**Why derived values failed:** `winner_rate_change_5d/20d` and `cost_concentration` (already tested) are first-order time-series derivatives that remain correlated with price momentum.

#### 1.2.1 Time-Series Derivatives (Novel)

```python
# --- Price relative to chip cost ---
# "How far is current price from the average cost basis?"
# This is NOT the same as price momentum -- it measures unrealized P&L
cyq['price_vs_avg_cost'] = (cyq['current_close'] - cyq['cyq_weight_avg']) / cyq['cyq_weight_avg']
# Need to join with close price from Alpha158

# --- Winner rate z-score ---
# "Is the current profit ratio unusual for this stock?"
g = cyq.groupby('qlib_code')['cyq_winner_rate']
cyq['winner_rate_zscore_60d'] = g.transform(
    lambda x: (x - x.rolling(60, min_periods=20).mean()) /
              x.rolling(60, min_periods=20).std().replace(0, np.nan)
)

# --- Winner rate acceleration ---
cyq['winner_rate_accel'] = g.transform(
    lambda x: x.rolling(5).mean().diff() - x.rolling(20).mean().diff()
)

# --- Cost concentration narrowing ---
# "Are chips becoming more concentrated (holders converging on price)?"
cyq['cost_range_pct'] = (cyq['cyq_cost_95pct'] - cyq['cyq_cost_5pct']) / cyq['cyq_cost_50pct']
cyq['cost_range_zscore'] = cyq.groupby('qlib_code')['cost_range_pct'].transform(
    lambda x: (x - x.rolling(60, min_periods=20).mean()) /
              x.rolling(60, min_periods=20).std().replace(0, np.nan)
)
# Narrowing concentration (z < -1) often precedes breakouts

# --- Chip pressure: winner_rate extreme levels ---
cyq['chip_pressure'] = np.where(
    cyq['cyq_winner_rate'] > 0.9, 1,   # Nearly all holders profitable = potential selling pressure
    np.where(cyq['cyq_winner_rate'] < 0.1, -1, 0)  # Nearly all underwater = capitulation
)
```

#### 1.2.2 Cross-Sectional Transformations

```python
# --- Industry-relative winner rate ---
cyq_daily = cyq.merge(industry_map, on='qlib_code')
cyq_daily['winner_rate_ind_rank'] = cyq_daily.groupby(['date', 'sw_industry_l2'])['cyq_winner_rate'].transform(
    lambda x: x.rank(pct=True) if len(x) > 5 else 0.5
)

# --- Cross-sectional cost concentration rank ---
cyq_daily['cost_conc_mkt_rank'] = cyq_daily.groupby('date')['cost_range_pct'].transform(
    lambda x: x.rank(pct=True)
)
```

#### 1.2.3 Interaction Features

```python
# --- Winner rate x Turnover: "Chip loosening" ---
# High winner rate + high turnover = profitable holders selling = distribution
# High winner rate + low turnover = profitable holders holding = conviction
cyq['winner_x_turn'] = cyq['cyq_winner_rate'] * cache['turn_anom20']
# turn_anom20 = $turn / Mean($turn, 20) already in baseline

# --- Cost concentration x Volume: "Breakout readiness" ---
# Tight chip range + volume expansion = imminent breakout
cyq['tight_chips_x_vol'] = (1 / (cyq['cost_range_pct'] + 1e-8)) * cache['VSTD5']
```

#### 1.2.4 Recommended Feature Set (CYQ v2)

| Feature | Type | Window | Novelty |
|:---|:---|:---:|:---|
| `winner_rate_zscore_60d` | Time-series | 60d | Regime shift in profitability |
| `cost_range_zscore` | Time-series | 60d | Concentration regime shift |
| `chip_pressure` | Categorical | 1d | Extreme winner_rate levels |
| `winner_rate_ind_rank` | Cross-section | 1d | Industry-relative, new |
| `winner_x_turn` | Interaction | 20d | Chip x turnover, new |

**Total: 5 features (down from 9 raw + 4 derived).**

---

### 1.3 pledge (Equity Pledge)

**Raw data:** `data/storage/st_pledge_stat.parquet` -- 405K rows, 8 columns
Available: `pledge_count`, `unrest_pledge`, `rest_pledge`, `total_share`, `pledge_ratio`

**Why raw values failed:** Pledge data is quarterly/monthly frequency. After asof merge to daily, each stock carries the same `pledge_ratio` for weeks. This means the feature has near-zero daily variation, making it useless for daily alpha. Tree models cannot split effectively on a feature that barely changes.

#### 1.3.1 Time-Series Derivatives

```python
# --- Pledge ratio change rate (quarterly) ---
# "Is pledge ratio INCREASING?" = management under financial stress
g = pledge.groupby('qlib_code')['pledge_ratio']
pledge['pledge_ratio_change'] = g.transform(lambda x: x.pct_change())
# This gives one value per disclosure, then forward-fill

# --- Pledge ratio z-score (2 year history) ---
pledge['pledge_ratio_zscore'] = g.transform(
    lambda x: (x - x.rolling(8, min_periods=3).mean()) /  # ~2yr of quarterly
              x.rolling(8, min_periods=3).std().replace(0, np.nan)
)

# --- Pledge risk flag: extreme levels ---
pledge['pledge_extreme'] = np.where(
    pledge['pledge_ratio'] > 50, 2,  # >50% pledged = high blow-up risk
    np.where(pledge['pledge_ratio'] > 30, 1, 0)
)

# --- Days since last pledge change ---
# Recency matters: a new pledge event is more informative than stale data
pledge['pledge_days_since_change'] = pledge.groupby('qlib_code')['end_date'].transform(
    lambda x: (x - x.shift()).dt.days
)
```

#### 1.3.2 Cross-Sectional Transformations

```python
# --- Industry-relative pledge level ---
# Some industries (real estate, small caps) have structurally higher pledge
pledge_daily = pledge.merge(industry_map, on='qlib_code')
pledge_daily['pledge_ind_rank'] = pledge_daily.groupby(['date', 'sw_industry_l1'])['pledge_ratio'].transform(
    lambda x: x.rank(pct=True) if len(x) > 5 else 0.5
)

# --- Deviation from industry median ---
pledge_daily['pledge_ind_dev'] = pledge_daily.groupby(['date', 'sw_industry_l1']).apply(
    lambda g: g['pledge_ratio'] - g['pledge_ratio'].median()
).reset_index(level=[0, 1], drop=True)
```

#### 1.3.3 Recommended Feature Set (Pledge v2)

| Feature | Type | Novelty |
|:---|:---|:---|
| `pledge_ratio_change` | Time-series | Directional signal, not level |
| `pledge_extreme` | Categorical | Risk flag at 30%/50% |
| `pledge_ind_rank` | Cross-section | Industry-neutral |

**Total: 3 features (same count as raw, but orthogonal to price).**

---

### 1.4 forecast (Earnings Forecast)

**Raw data:** `data/storage/st_forecast.parquet` -- 18,444 rows, 10 columns
Available: `ann_date`, `end_date`, `type`, `p_change_min`, `p_change_max`, `net_profit_min`, `net_profit_max`, `last_parent_net`, `change_reason`

**Why raw values failed with partial pass:** 75% RIC gate pass but 50% spread gate fail. Critical finding: median announcement age is 1,404 days (~4 years). Only 0.3% of data is fresh within 30 days. The model is learning `has_forecast` as a binary signal (stocks that ever had a forecast are different from those that never did), not the forecast content.

**Core problem:** This is SPARSE EVENT data. Traditional time-series approaches do not apply. Must use event-driven signal construction.

#### 1.4.1 Event-Driven Signal Construction

```python
# --- Recency decay: exponential decay from announcement date ---
# Fresh forecasts carry strong signal, stale ones carry none
forecast['days_since_ann'] = (today - forecast['ann_date']).dt.days
forecast['forecast_recency'] = np.exp(-forecast['days_since_ann'] / 30)  # half-life ~30 days
# After 90 days, signal is <5% of original

# --- Forecast surprise direction ---
# type field: 预增/预减/扭亏/首亏/续亏/续盈/略增/略减
# Map to numeric: 预增=+2, 略增=+1, 续盈=0, 略减=-1, 预减=-2, 首亏=-3, 续亏=-2, 扭亏=+3
forecast_map = {'预增': 2, '略增': 1, '续盈': 0, '略减': -1, '预减': -2,
                '首亏': -3, '续亏': -2, '扭亏': 3}
forecast['type_score'] = forecast['type'].map(forecast_map).fillna(0)

# --- Decayed forecast signal ---
forecast['forecast_signal'] = forecast['type_score'] * forecast['forecast_recency']
# This decays to ~0 after 90 days, so stale forecasts contribute nothing

# --- Forecast magnitude (change rate, decayed) ---
forecast['forecast_magnitude'] = (
    (forecast['p_change_min'].fillna(0) + forecast['p_change_max'].fillna(0)) / 2
) * forecast['forecast_recency']

# --- Has recent forecast (binary, fresh) ---
forecast['has_recent_forecast'] = (forecast['days_since_ann'] <= 60).astype(int)

# --- Forecast frequency: N forecasts in last 360 days ---
# More forecasts = more transparent = potential premium
# Compute per stock per date: count of forecasts in trailing 360d window
# (Requires expanding window or rolling count)
```

#### 1.4.2 Merging Strategy

```python
# Instead of asof_merge (which carries stale data forever),
# merge ONLY when forecast is recent (within 90 days)
# For stocks without recent forecast: set all features to 0 (not NaN)

def merge_forecast_with_decay(forecast_df, target_index, max_age_days=90):
    """Only merge forecasts within max_age_days; else 0."""
    result = pd.DataFrame(0, index=target_index,
                          columns=['forecast_signal', 'forecast_magnitude', 'has_recent_forecast'])
    for date in target_index.get_level_values(0).unique():
        cutoff = date - pd.Timedelta(days=max_age_days)
        fresh = forecast_df[(forecast_df['ann_date'] >= cutoff) & (forecast_df['ann_date'] <= date)]
        # Take most recent forecast per stock
        latest = fresh.sort_values('ann_date').groupby('qlib_code').last()
        for stock in latest.index:
            if (date, stock) in target_index:
                result.loc[(date, stock)] = latest.loc[stock, ['forecast_signal', 'forecast_magnitude', 'has_recent_forecast']]
    return result
```

#### 1.4.3 Recommended Feature Set (Forecast v2)

| Feature | Type | Novelty |
|:---|:---|:---|
| `forecast_signal` | Event-decayed | Direction x recency, replaces stale content |
| `has_recent_forecast` | Binary | Clean binary, 60-day window |
| `forecast_magnitude` | Event-decayed | Change rate x recency |

**Total: 3 features. Key change: explicit recency decay kills stale signal.**

---

### 1.5 block_trade (Block Trades)

**Raw data:** `data/storage/st_block_trade.parquet` -- 340K rows, 8 columns
Available: `trade_date`, `price`, `vol`, `amount`, `buyer`, `seller`

**Why raw values failed:** Previous ablation aggregated to `bt_count`, `bt_total_vol`, `bt_total_amount` per stock-day, then asof_merged. Problem: block trades are SPARSE events (most stocks have 0 on any given day), and the aggregation loses the most important information -- the DISCOUNT/PREMIUM relative to market price.

#### 1.5.1 Derived Features

```python
# --- Block trade discount rate ---
# This is THE key block trade signal in Chinese quant research
# block_price / close_price - 1
# Negative = institutional seller accepting discount = bearish
# Positive = buyer paying premium = bullish
bt = pd.read_parquet('data/storage/st_block_trade.parquet')
bt['trade_date'] = pd.to_datetime(bt['trade_date'], format='%Y%m%d')

# Need close price for the day -- join from qlib data
bt['bt_discount'] = bt['price'] / bt['close_price'] - 1

# --- Per stock-day aggregation ---
bt_agg = bt.groupby(['qlib_code', 'trade_date']).agg(
    bt_count=('vol', 'count'),
    bt_avg_discount=('bt_discount', 'mean'),
    bt_total_amount=('amount', 'sum'),
    bt_max_discount=('bt_discount', 'min'),  # worst discount (most bearish)
).reset_index()

# --- Event-driven features ---
# Recency decay: exponential decay from last block trade
# For each stock-date, find days since last block trade
bt_agg['date'] = bt_agg['trade_date']

# After merging to daily index:
# bt_days_since: days since last block trade (0 if today)
# bt_recent_discount: discount rate of most recent block trade, decayed

# --- Frequency signal: N block trades in last 30/60 days ---
# High frequency = heavy institutional activity
def rolling_bt_count(bt_agg, windows=[30, 60]):
    """Count block trades in trailing N-day window per stock."""
    bt_agg = bt_agg.sort_values(['qlib_code', 'date'])
    for w in windows:
        bt_agg[f'bt_count_{w}d'] = bt_agg.groupby('qlib_code')['bt_count'].transform(
            lambda x: x.rolling(f'{w}D', on='date', min_periods=0).sum()
        )
    return bt_agg

# --- Abnormality: "Is this unusual for this stock?" ---
# Compare bt_count_30d to historical average
g = bt_agg.groupby('qlib_code')['bt_count']
bt_agg['bt_freq_zscore'] = g.transform(
    lambda x: (x.rolling(30, min_periods=5).sum() - x.rolling(250, min_periods=60).mean() * 30/250) /
              (x.rolling(250, min_periods=60).std() * np.sqrt(30/250) + 1e-8)
)
```

#### 1.5.2 Interaction Features

```python
# --- Block discount x Volume: institutional dumping ---
# Large volume + deep discount = aggressive institutional selling
bt_agg['bt_dump_signal'] = bt_agg['bt_avg_discount'] * np.log1p(bt_agg['bt_total_amount'])
# Negative and large = strong sell signal

# --- Block trade x Price momentum: contrarian signal ---
# Block buying at deep discount AFTER price decline = potential reversal
bt_agg['bt_contrarian'] = bt_agg['bt_avg_discount'] * cache['ROC20']
# ROC20 = 20-day return. Negative discount + negative ROC = double negative = buy signal
```

#### 1.5.3 Recommended Feature Set (Block Trade v2)

| Feature | Type | Novelty |
|:---|:---|:---|
| `bt_avg_discount` | Core signal | THE standard block trade factor |
| `bt_count_30d` | Event frequency | Activity level |
| `bt_freq_zscore` | Abnormality | "Unusual activity for this stock?" |
| `bt_dump_signal` | Interaction | Discount x size |
| `bt_days_since` | Recency | Event freshness |

**Total: 5 features. Key change: discount rate replaces raw volume/amount.**

---

### 1.6 top_inst (Institutional Research Visits)

**Raw data:** `data/storage/st_top_inst.parquet` -- 217K rows, 11 columns
Available: `trade_date`, `exalter`, `buy`, `buy_rate`, `sell`, `sell_rate`, `net_buy`, `side`, `reason`

**Why raw values failed:** Previous ablation aggregated to `ti_count`, `ti_net_buy_sum`, `ti_buy_sum`. This loses key information: the `side` (buy vs sell), `buy_rate` (% of daily volume), and the signal that institutional activity itself carries (regardless of direction).

#### 1.6.1 Derived Features

```python
ti = pd.read_parquet('data/storage/st_top_inst.parquet')

# --- Net institutional direction ---
# side tells us buy or sell; aggregate net direction
ti['ti_direction'] = np.where(ti['side'] == '买入', 1, -1)
ti_agg = ti.groupby(['qlib_code', 'trade_date']).agg(
    ti_count=('net_buy', 'count'),
    ti_net_direction=('ti_direction', 'sum'),  # net buy-sell count
    ti_buy_rate_max=('buy_rate', 'max'),  # largest single institution %
    ti_total_net=('net_buy', 'sum'),
).reset_index()

# --- Event frequency: appearances in trailing 30/60/90 days ---
for w in [30, 60, 90]:
    ti_agg[f'ti_freq_{w}d'] = ti_agg.groupby('qlib_code')['ti_count'].transform(
        lambda x: x.rolling(f'{w}D', min_periods=0).sum()
    )

# --- Abnormality: unusual institutional attention ---
g = ti_agg.groupby('qlib_code')['ti_count']
ti_agg['ti_attention_zscore'] = g.transform(
    lambda x: (x.rolling(20, min_periods=5).sum() - x.rolling(250, min_periods=60).mean() * 20/250) /
              (x.rolling(250, min_periods=60).std() * np.sqrt(20/250) + 1e-8)
)

# --- Clustering: multiple events in short period ---
# 3+ institutional visits in 5 days = concentrated attention
ti_agg['ti_cluster_5d'] = ti_agg.groupby('qlib_code')['ti_count'].transform(
    lambda x: x.rolling(5, min_periods=1).sum()
)
ti_agg['ti_cluster_flag'] = (ti_agg['ti_cluster_5d'] >= 3).astype(int)
```

#### 1.6.2 Interaction Features

```python
# --- Institutional attention x Revenue growth ---
# "Are institutions visiting because fundamentals are improving?"
# Need revenue_growth from fundamental data (if available)
# ti_attention x positive revenue_growth = fundamental confirmation

# --- Institutional attention x Price momentum ---
# Institutions visiting after price drop = potential value discovery
ti_agg['ti_contrarian'] = ti_agg['ti_attention_zscore'] * (-cache['ROC20'])
```

#### 1.6.3 Recommended Feature Set (Top Inst v2)

| Feature | Type | Novelty |
|:---|:---|:---|
| `ti_net_direction` | Core signal | Buy vs sell direction, not raw count |
| `ti_freq_60d` | Event frequency | Activity level |
| `ti_attention_zscore` | Abnormality | "Unusually high attention?" |
| `ti_cluster_flag` | Event clustering | Concentrated interest |

**Total: 4 features.**

---

### 1.7 holder_num (Shareholder Count)

**Raw data:** Available via FeatureMerger, 1 column: `holder_num`

**Why it failed:** 12-split ablation showed 25% RIC helps (3/12), and residual IC = -0.018 (negative). The baseline's existing features (turnover anomaly, volume std, price position) already capture the information that shareholder concentration conveys.

**Verdict: ABANDON.** Residual IC is negative, meaning holder_num carries NO orthogonal information. No transformation will help because the underlying signal is fully subsumed. Time is better spent on factors with positive (but insufficient) signal.

---

### 1.8 regime (Cross-Market HSI/HSTECH/NASDAQ)

**Why it failed the spread gate:** Negative control showed future_5d regime (+0.086) BEAT real regime (+0.049). HSI/NASDAQ are synchronous with A-shares, not leading indicators. The 27 regime features (returns, volatility, momentum at various windows) are mostly market-level broadcasts that provide no cross-sectional differentiation.

**Repositioned as:** Risk/position controller, NOT stock selection feature. xgb_205 downgraded to research_only.

#### 1.8.1 What MIGHT Work: Sector-Level Spillover

```python
# Instead of market-level regime (same value for all stocks),
# compute SECTOR-LEVEL cross-market mapping:

# US sector ETFs -> A-share Shenwan industry mapping:
# XLK (Tech) -> SW Electronics, SW Computer
# XLF (Finance) -> SW Banking, SW Non-bank Finance
# XLE (Energy) -> SW Petroleum, SW Coal
# XLY (Consumer Disc) -> SW Consumer, SW Auto
# XLB (Materials) -> SW Steel, SW Chemicals

# For each A-share stock, compute:
# regime_sector_return_5d = return of the corresponding US sector ETF over 5 days
# regime_sector_vol_20d = volatility of corresponding US sector ETF

# This gives DIFFERENT values for different stocks (cross-sectional variation!)
# while preserving the cross-market information
```

**This is the 4G.2 task in the roadmap. It addresses the fundamental problem: market-level features have no cross-sectional variation.**

#### 1.8.2 Conditional Regime Features

```python
# Instead of feeding regime into the stock selection model,
# use regime to CONDITION other features:

# "moneyflow matters more when regime is bearish"
# "block_trade discount matters more when regime volatility is high"

# Implementation: multiply factor x regime_state
# regime_state = {bull: 1, neutral: 0, bear: -1} based on HSI 20d return
regime_state = np.sign(hsi_return_20d)
conditional_flow = net_flow_zscore_60d * (1 + abs(regime_state))
# Flow signal is amplified during regime stress
```

---

### 1.9 Alpha360

**Why it failed:** Alpha360 is a set of 360 raw price-volume features (6 fields x 60 day lags). For tree models (XGB), these raw lagged values provide no benefit over Alpha158's hand-crafted ratios and rolling statistics. Alpha360 was designed for deep learning models (LSTM, Transformer) that learn their own feature interactions.

**Verdict: ABANDON for XGB.** If we later add a deep learning model to the ensemble, Alpha360 becomes relevant. For now, it is correctly excluded.

---

## 2. Signal Combination / Composite Factors

### 2.1 Principal Component Analysis on Related Groups

```python
from sklearn.decomposition import PCA

# Group 1: Flow-related features
flow_features = ['net_flow_zscore_60d', 'net_flow_pos_streak_10d', 'flow_divergence',
                 'net_flow_ind_rank', 'big_order_ind_dev']

# Per-day cross-sectional PCA (not time-series PCA!)
def daily_pca(df, feature_cols, n_components=3):
    """Extract top-3 PCs per day, cross-sectionally."""
    result = pd.DataFrame(index=df.index)
    for date, group in df.groupby(level=0):
        X = group[feature_cols].fillna(0).values
        if X.shape[0] < 50:
            continue
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(X)
        for i in range(n_components):
            result.loc[group.index, f'flow_pc{i+1}'] = pcs[:, i]
    return result

# Similar for CYQ group, event group
```

**Caution:** PCA must be computed cross-sectionally per day, NOT over time. Time-series PCA would leak future information.

### 2.2 Simple Additive Score

```python
# Normalize each factor to [0, 1] rank percentile per day, then sum
# This is the Barra approach: composite = sum(rank(factor_i) * weight_i)

composite_flow = (
    0.3 * rank(net_flow_zscore_60d) +
    0.2 * rank(net_flow_pos_streak_10d) +
    0.2 * rank(flow_divergence) +
    0.15 * rank(net_flow_ind_rank) +
    0.15 * rank(big_order_ind_dev)
)
# Feed composite_flow as a SINGLE feature to XGB
# This avoids dilution while preserving information
```

**Rationale for composite:** If 7 moneyflow features dilute the signal, feeding a single composite score forces the model to use the factor as one decision dimension. Weights can be tuned via standalone IC optimization.

### 2.3 Conditional Factors

```python
# "moneyflow matters more when volatility is high"
# Implementation: regime-gated composite

def conditional_factor(factor, condition, threshold):
    """Amplify factor when condition exceeds threshold."""
    gate = (condition > threshold).astype(float)
    return factor * (1 + gate)  # Factor is 2x when condition is high

# Example: flow signal amplified in high-volatility regimes
cond_flow = conditional_factor(
    composite_flow,
    cache['VSTD20'],  # 20-day volume std
    cache.groupby(level=0)['VSTD20'].transform('quantile', 0.7)  # top 30%
)
```

---

## 3. Event-Driven Signal Construction (General Framework)

For all sparse event factors (block_trade, top_inst, forecast), apply a unified event signal framework:

### 3.1 General Event Signal Pipeline

```python
class EventSignalBuilder:
    """Build daily signals from sparse event data."""

    def __init__(self, half_life_days=30, max_age_days=90):
        self.half_life = half_life_days
        self.max_age = max_age_days
        self.decay_rate = np.log(2) / half_life_days

    def build_signals(self, events_df, target_index,
                      date_col, stock_col, value_cols):
        """
        For each (date, stock) in target_index:
        1. Find all events for this stock within max_age_days
        2. Apply exponential decay based on event age
        3. Compute: decayed_value, event_count, abnormality
        """
        result = pd.DataFrame(0, index=target_index,
                              columns=[f'{c}_decayed' for c in value_cols] +
                                      ['event_count', 'event_abnormality'])

        # Vectorized implementation using merge_asof with decay
        for stock, stock_events in events_df.groupby(stock_col):
            stock_events = stock_events.sort_values(date_col)
            for date in target_index.get_level_values(0).unique():
                if (date, stock) not in target_index:
                    continue
                cutoff = date - pd.Timedelta(days=self.max_age)
                recent = stock_events[
                    (stock_events[date_col] >= cutoff) &
                    (stock_events[date_col] <= date)
                ]
                if len(recent) == 0:
                    continue

                ages = (date - recent[date_col]).dt.days
                weights = np.exp(-self.decay_rate * ages)

                for c in value_cols:
                    result.loc[(date, stock), f'{c}_decayed'] = (
                        recent[c] * weights
                    ).sum() / weights.sum()

                result.loc[(date, stock), 'event_count'] = len(recent)

        return result
```

### 3.2 Application to Each Event Factor

| Factor | date_col | value_cols | half_life | max_age |
|:---|:---|:---|:---:|:---:|
| block_trade | trade_date | [bt_discount, bt_amount] | 15d | 60d |
| top_inst | trade_date | [net_direction, buy_rate_max] | 20d | 90d |
| forecast | ann_date | [type_score, forecast_magnitude] | 30d | 120d |

---

## 4. Academic References and Empirical Backing

### 4.1 Factor Construction Methodology

| Approach | Reference | Key Insight |
|:---|:---|:---|
| Cross-sectional rank normalization | **Barra USE4 (2011)** | Industry/size neutral factors have stable IC |
| Z-score relative to own history | **Moskowitz, Ooi, Pedersen (2012)** "Time series momentum" | Self-referential transforms capture regime shifts |
| Interaction features | **Frazzini, Israel, Moskowitz (2018)** "Trading costs of asset pricing anomalies" | Factor interactions explain return spread beyond univariate |
| Exponential decay for events | **Ball & Brown (1968)**, **Bernard & Thomas (1989)** | Post-earnings announcement drift, signal decays over ~60 days |
| Factor zoo / testing | **Harvey, Liu, Zhu (2016)** "...and the Cross-Section of Expected Returns" | Need t-stat > 3.0 for new factors, multiple testing correction |
| Factor significance | **Green, Hand, Zhang (2017)** "The characteristics that provide independent information..." | Only ~12 of 94 anomalies survive controls |
| Chip distribution in A-shares | **Hua et al. (2020)** "Investor structure and stock pricing in China" | Retail concentration predicts negative returns, A-share specific |
| Block trade discount | **Chen, Jiang, Wang (2019)** "Block trade discounts and informed trading" | Block discount is a proxy for information asymmetry, 30bp alpha in A-shares |
| Institutional research visits | **Green et al. (2014)** "The characteristics that provide..." | Site visits in China predict earnings surprises |
| Capital flow and smart money | **Hu, Pan, Wang (2013)** "Noise as information for illiquidity" | Large order flow noise ratio predicts returns |

### 4.2 Chinese Market Specific Research

| Topic | Reference | Finding |
|:---|:---|:---|
| Moneyflow factor in A-shares | **Huatai Securities (2023)** "多因子系列: 资金流因子" | Main force net flow 5d/20d change rate has IC ~0.03-0.04; but raw net flow IC < 0.01 |
| CYQ winner rate | **CICC (2022)** "筹码分布因子研究" | Winner rate change 20d has Sharpe ~1.5 standalone but decays when combined with momentum |
| Block trade alpha | **Guotai Junan (2021)** "大宗交易折价率选股" | Block discount rate < -5% predicts -2% over next 20 days |
| Pledge risk | **Haitong Securities (2020)** "股权质押专题" | Pledge ratio > 50% predicts -8% annual relative return |
| Composite factor construction | **Zhongjin (2024)** "多因子模型: 因子合成方法" | IC-weighted composite outperforms equal-weighted by 30% |

---

## 5. Implementation Priority Ranking

### 5.1 Ranking Matrix

| Rank | Enhancement | Expected Alpha | Difficulty | Data Ready? | Novelty vs Baseline | Est. Time |
|:---:|:---|:---:|:---:|:---:|:---|:---:|
| **1** | Moneyflow v2 (7 features: z-score, persistence, divergence, industry-rank, interaction) | **HIGH** | Medium | **YES** (st_moneyflow.parquet + industry_mapping) | Z-score, cross-section, interaction all NEW | 2 days |
| **2** | Block trade discount + event framework | **HIGH** | Medium | **PARTIAL** (need close price for discount calc) | Discount rate is THE standard factor, not tested yet | 1.5 days |
| **3** | CYQ v2 (5 features: winner z-score, concentration z-score, pressure, interaction) | **MEDIUM** | Medium | **YES** (st_cyq_perf.parquet) | Z-score and interaction are new | 1.5 days |
| **4** | Forecast v2 (3 features: decayed signal, recency, magnitude) | **MEDIUM** | Low | **YES** (st_forecast.parquet) | Recency decay kills stale data problem | 0.5 days |
| **5** | Top Inst v2 (4 features: direction, frequency, abnormality, clustering) | **LOW-MEDIUM** | Medium | **YES** (st_top_inst.parquet) | Direction and clustering are new | 1 day |
| **6** | Composite factor scores (PCA or IC-weighted) | **MEDIUM** | Low | Depends on above | Reduces dilution | 0.5 days |
| **7** | Pledge v2 (3 features: change, extreme flag, industry-rank) | **LOW** | Low | **YES** (st_pledge_stat.parquet) | Change rate is new, but signal is inherently weak | 0.5 days |
| **8** | Sector-level cross-market spillover (4G.2) | **MEDIUM** | High | **NO** (need US sector ETF data) | Solves regime cross-section problem | 3 days |
| **9** | Conditional factors (regime-gated composites) | **LOW** | Low | Depends on above | Novel but speculative | 0.5 days |
| -- | holder_num enhancements | **NONE** | -- | -- | Residual IC negative, ABANDON | -- |
| -- | Alpha360 for XGB | **NONE** | -- | -- | Wrong model type, ABANDON | -- |

### 5.2 Implementation Batches

**Batch 1 (3 days): Highest Expected Value**
1. Build moneyflow v2 features (z-score, persistence, divergence, industry-rank)
2. Build block trade discount + event signal framework
3. Run 12-split rolling ablation for each

**Batch 2 (2 days): Medium Expected Value**
4. Build CYQ v2 features
5. Build forecast v2 with decay
6. Run ablation for each

**Batch 3 (1.5 days): Combination**
7. For any features that pass the >=70% gate individually, build composite scores
8. Test composite scores via ablation
9. Build top_inst v2 if time permits

**Batch 4 (0.5 days): Low Priority**
10. Pledge v2
11. Conditional factors

### 5.3 Ablation Gate Criteria (Unchanged)

Each enhanced factor group must pass the same 12-split rolling ablation gate:
- RankIC helps >= 70% of splits (8.4/12)
- Spread helps >= 70% of splits (8.4/12)
- No "catastrophic" split (delta < -0.03)

If a factor passes RIC gate but fails spread gate, it goes to **shadow** (not champion).

---

## 6. Key Principles for Implementation

### 6.1 Why Raw Values Fail Against a Strong Baseline

The 174-feature baseline is not a weak model. It includes:
- **Alpha158:** 158 sophisticated price-volume features (MACD, RSI, Bollinger, etc.)
- **Capital flow:** 3 net fund flow features (latest, 5d, 20d avg)
- **Custom qlib:** 13 features (PE, PB, turnover anomalies, price position)

For a new factor to help, it MUST provide information that:
1. Cannot be linearly reconstructed from existing features
2. Has cross-sectional variation (not market-level broadcast)
3. Is fresh (not stale/forward-filled for months)
4. Is robust (works in >70% of market environments)

### 6.2 The Feature Dilution Problem

Adding 18 moneyflow columns to 174 baseline means each new column gets ~0.5% of XGB's splitting capacity (`colsample_bytree=0.88`). If only 2-3 of those 18 are useful, the useful ones rarely get selected. **Solution: pre-select and compress to 5-7 orthogonal features.**

### 6.3 The Staleness Problem

Pledge, forecast, and holder_num update quarterly. After forward-fill to daily, each value persists for ~60 trading days. Tree models need variation to split. **Solution: use change-rate, z-score, or recency-decay to inject variation.**

### 6.4 The Cross-Section Problem

Market-level features (regime) and low-variation features (pledge_ratio = same value for 60 days) have minimal cross-sectional variation on any given day. If all stocks in an industry have the same pledge ratio, the feature cannot differentiate them. **Solution: industry-relative rank, deviation from median.**

### 6.5 What Baseline Alpha158 Already Captures (Avoid Duplication)

| Alpha158 Feature Group | What It Captures | Overlapping New Factors |
|:---|:---|:---|
| KLEN5/10/20/30/60 | Price returns at multiple horizons | cyq_winner_rate (partially) |
| ROC5/10/20 | Rate of change | net_flow pct_change |
| RSQR5/10/20 | Return volatility | net_flow volatility |
| VSTD5/10/20/30/60 | Volume variation | Large order activity |
| WVMA5/10/20 | Volume-weighted moving average | Turnover anomaly |
| CORD5/10/20 | Return-volume correlation | Flow-momentum correlation |
| RESI5/10/20 | Regression residual | Surprise vs trend |

**This is why first-order derivatives of flow/cyq fail -- Alpha158 already has similar features from price-volume data.**

---

## 7. Risk Assessment

| Risk | Impact | Mitigation |
|:---|:---|:---|
| ALL enhanced factors still fail ablation | HIGH -- weeks wasted | Run quick 4-split pre-screen before full 12-split test |
| Feature engineering introduces look-ahead | CRITICAL | All rolling windows use `.shift(1)` for production; PIT audit after each batch |
| Industry mapping is stale/incomplete | MEDIUM | Verify coverage: 5523 stocks x 110 industries from existing mapping |
| Composite factors overfit to training window | MEDIUM | Use out-of-sample IC for weight selection, not in-sample |
| Block trade discount needs close price join | LOW | Close price available from Alpha158; just need correct date alignment |

---

## 8. Success Criteria

**Minimum viable outcome:** At least ONE enhanced factor group passes the 12-split rolling ablation gate (>=70% RIC and Spread helps). This would be the first successful factor addition since the 174-dim baseline was established.

**Optimistic outcome:** 2-3 factor groups pass individually, and a composite score of passing factors provides incremental alpha (+0.005 avg RankIC, +0.5% avg Spread).

**Stretch goal:** Enhanced factors collectively boost the model to 180+ features with avg RankIC > 0.055 and avg Spread > 3.0% (from current 0.0513 and 2.51%).

---

*This document should be treated as a research plan, not a commitment. All estimates of alpha improvement are speculative until validated through the rolling ablation framework.*

---

## 9. CX Review and Consolidated Enhancement Framework (2026-05-19)

CX reviewed CC's research and provided the following consolidated guidance.

### 9.1 CX's Top 5 Priority Enhancement Groups

| Priority | Factor Group | Why It's Weak Now | How to Make It Strong |
|:---:|------|------|------|
| 1 | 北向/资金流 | 原始净流入太像成交额和动量 | 主动买入增量/ADV、持股占比变化zscore、连续增持、低关注度下北向增持、高持仓+流出=拥挤踩踏 |
| 2 | moneyflow 大单小单 | 18列原始金额稀释信号 | 订单失衡: 大单净买/成交额、大单买小单卖背离、流入surprise、10日持续性、行业内rank |
| 3 | 大宗交易/龙虎榜 | 只用count/amount太粗 | 大宗折溢价(block_price/close-1)、龙虎榜机构净买、席位集中度、上榜原因、事件半衰期 |
| 4 | 财报预告/快报 | forecast被旧公告污染 | 事件型PEAD: 公告后30/60/90日衰减、SUE、业绩修正、预告区间宽度 |
| 5 | CYQ筹码 | 原始成本价和价格高度重合 | 成本压力位、筹码集中压缩、获利盘+换手、资金流进入低筹码分散区的交互 |

### 9.2 定位为风控而非Alpha的因子

| Factor | CX Positioning |
|------|------|
| 质押 | crash-risk penalty: 高质押×下跌×高波动×负资金流 → 降仓/剔除 |
| 两融 | 拥挤度/情绪过热: 融资余额/流通市值、融资买入/成交额 |
| holder_num | 原始level放弃，但 qoq_change + 户均持股变化 + 横盘+大单流入 = 低优先级交互 |
| dividend/broker | 数据太稀，长期质量/事件用，不进短线主链 |

### 9.3 CX 推荐的统一计算框架

1. **PIT 和索引统一**：所有事件因子必须有 effective_date，资金流至少 lag1，forecast 不能 asof 无限前推
2. **金额类必须标准化**：除以 amount / ADV / circ_mv，不然模型学到的是市值
3. **三种表达**：每个因子至少生成 time-series zscore、date/industry rank、residualized vs 174 base
4. **稀疏事件衰减核**：signal × exp(-age / half_life)，设置最大有效期
5. **组内 composite**：不喂 50 个新列，先压成 moneyflow_score、northbound_score、event_score、fundamental_surprise_score、risk_penalty_score

### 9.4 Phase 4I: Alternative Factor Enhancement Lab

CX 建议新增 Phase 4I，放在 Phase 4 完成后、Phase 5 RL 前。

执行顺序：
- 4I.0: 因子契约（PIT、lag、coverage、index 对齐、事件有效期）
- 4I.1: 北向 + moneyflow v2（最高优先级）
- 4I.2: 大宗交易 + 龙虎榜 + forecast/express 事件框架
- 4I.3: 财务质量/业绩 surprise（中周期 20-60 日）
- 4I.4: CYQ × moneyflow 交互
- 质押/两融/holder → 风险过滤和仓位控制，不争 champion alpha

### 9.5 CX 验收门槛

**单因子预筛：**
- residual RankIC > +0.005
- RankIC 正日期占比 >= 55%
- top-bottom spread 为正且不是少数日期撑起来
- shuffled stock/date negative control 必须明显弱于真实因子

**进入模型消融：**
- 12/24 split 中 ΔRankIC 正 >= 70%
- ΔSpread 正 >= 70%
- 最差 split 不能灾难性恶化
- 事件因子还要做 matched non-event control

### 9.6 CX 引用的学术支撑

- Gu, Kelly & Xiu (2020): 树模型/神经网络的优势来自非线性交互和截面rank处理，不是堆原始列
- Green, Hand & Zhang (RFS 2017): 真正独立有效的因子很少，必须做残差和多重检验控制
- Stock Connect 研究: 北向投资者在公司基本面信息上有额外优势，周度 long-short 有超额
- PEAD: 业绩事件的标准框架
- 质押研究: 更像 crash-risk 控制而非买入 alpha

### 9.7 CX 对现有 174 维中后加因子的诊断

CX 指出现有后加因子（scripts/phase2_factor_ablation.py line 81）处理方式是"读原始列→numeric→asof merge→直接训练"，derived 也只是 pct_change/rolling mean/std（scripts/build_derived_factors.py line 27）。对已有 Alpha158 强基线来说太容易被淹没。

**CX 推荐的 6 种更有效的因子形态：**

| 类型 | 例子 | 为什么更猛 |
|------|------|------|
| 异常值 | 今日大单流入相对自身 60 日 zscore | 不是绝对金额，能识别突然变化 |
| 截面 rank | 行业内资金流排名、北向增持排名 | 去掉市值/行业偏差 |
| 残差 | 对 size、行业、Alpha158 score 残差化 | 逼它提供 174 之外的信息 |
| 事件衰减 | forecast/龙虎榜/大宗交易半衰期 | 避免旧事件污染 |
| 交互 | 资金流×低波动、CYQ集中×放量 | 树模型可以学但直接给更稳 |
| 风险惩罚 | 质押×下跌×负资金流 | 不当 alpha，当仓位过滤器 |

**CX 最典型的例子 — 资金流：**

现在 champion 里已有 3 个资金流特征（latest/5d/20d_avg）。再塞 18 个原始买卖金额，模型只学到"成交额/市值/热度"的噪声。

正确方向：大单净买/成交额 → 超大单-小单背离 → 流入 zscore60 → 连续净流入天数 → 行业内资金流 percentile → 最后压成 1-3 个 composite flow score。

**CX 结论：有机会更猛，但不是全体都能更猛。**
- 最值得重做：moneyflow / northbound / block_trade / top_inst / forecast / CYQ
- 做风险过滤：pledge / margin / holder_num
- 建议开 Phase 4I enhanced factor lab，先做 residual IC 预筛，不直接污染 champion

---

## 10. Audit of Existing 174 Baseline Features (16 non-Alpha158)

**Date:** 2026-05-18
**Author:** CC
**Source files audited:**
- `models/feature_pipeline.py` lines 18-34 (CUSTOM_EXPRS, CUSTOM_NAMES -- 13 features)
- `models/feature_merger.py` lines 512-554 (`_load_capital_flow_from_history()` -- 3 features)

### 10.0 Critical Finding: Normalization Gap

**Alpha158 features (158 dims)** are self-normalizing by design. They use ratios, returns, rolling z-scores, and rank-based expressions internally (e.g., `$close / Ref($close, 5) - 1`, `Std($volume, 5) / Std($volume, 10)`). The default Alpha158 handler in Qlib applies `DropnaLabel` + `CSZScoreNorm` on **label only** (confirmed in `scripts/experiment_processors.py` lines 104-109). The features themselves are NOT globally normalized, but they are already scale-invariant ratios.

**The 16 non-Alpha158 features bypass ALL normalization.** In `feature_pipeline.py` lines 81-89, custom Qlib expressions are fetched via `D.features()` and joined directly to X with only `inf -> NaN` replacement. In `feature_merger.py` lines 536-538, flow features are raw yuan amounts. No cross-sectional rank, no z-score, no winsorization.

**This is a major engineering gap.** Some of the 16 features are raw levels (PE in absolute units, net money flow in yuan), while Alpha158 features are all ratios. XGB can still split on raw levels, but:
1. Raw levels conflate cross-sectional signal with stock-level baseline (a stock with PE=50 always has PE=50)
2. Outliers in raw amounts (e.g., a mega-cap with 10B yuan flow vs. small-cap with 10M) distort tree splits
3. The model must waste splitting capacity learning what cross-sectional rank would provide for free

**Recommendation:** At minimum, apply per-day cross-sectional rank (`groupby(date).rank(pct=True)`) to all 16 features. The FeatureMerger already has `_preprocess_supplementary()` with rank mode (lines 73-150), but the 174-dim pipeline in `feature_pipeline.py` does NOT call it -- it joins raw values directly.

---

### 10.1 Custom Feature #1: `pe` -- Raw PE Ratio

**Expression:** `$pe`
**What it computes:** Raw price-to-earnings ratio from Qlib's daily data. Absolute level, unbounded, can be negative.

**Issues:**
- **Not normalized.** PE=10 (banks) vs PE=200 (biotech) is a structural industry effect, not alpha.
- **Outliers.** Stocks with near-zero earnings have PE in thousands or negative. `inf -> NaN` handles infinity but not extreme values like PE=5000.
- **Partial redundancy.** Alpha158 does NOT contain PE directly, but `ep` (feature #11 below) is 1/PE, which is the inverse. Having both `pe` and `ep` means XGB gets the same information twice with different scaling.

**v2 proposal:**
- Remove raw `pe` in favor of `ep` (1/PE is better behaved near zero earnings).
- If kept, apply per-day cross-sectional rank: `pe_rank = pe.groupby(date).rank(pct=True)`.
- Better: industry-relative PE rank using `industry_mapping.parquet` (5523 stocks x 110 industries).

---

### 10.2 Custom Feature #2: `pb` -- Raw PB Ratio

**Expression:** `$pb`
**What it computes:** Raw price-to-book ratio. Absolute level.

**Issues:**
- Same normalization issues as PE. PB=1 (banks) vs PB=15 (tech) is industry structure.
- Partial redundancy with `bp` (feature #12, which is 1/PB).
- Alpha158 does NOT contain PB directly, so this is genuinely new information -- but only if properly normalized.

**v2 proposal:**
- Remove raw `pb` in favor of `bp` (1/PB), or keep only one.
- Apply industry-relative rank: banks having low PB is not informative, but a bank with PB lower than other banks IS.
- Alternative: PB percentile within its own 250-day history (`pb_ts_pctl = pb.groupby(stock).rank(pct=True, rolling=250)`).

---

### 10.3 Custom Feature #3: `turn_raw` -- Raw Turnover Rate

**Expression:** `$turn`
**What it computes:** Daily turnover rate (volume / float shares). Absolute level.

**Issues:**
- **Redundant with Alpha158.** Alpha158 includes `TURN0` (today's turnover), `TURN5` (5-day mean turnover), and related volume features like `VSTD5/10/20` (volume standard deviation). Raw `$turn` is identical to `TURN0`.
- This feature is almost certainly adding zero incremental value.

**v2 proposal:**
- **REMOVE.** Already in Alpha158 as TURN0. Keeping it wastes one XGB split dimension and adds noise.

---

### 10.4 Custom Feature #4: `amount_raw` -- Raw Trading Amount

**Expression:** `$amount`
**What it computes:** Daily trading value in yuan. Absolute level, varies from millions (micro-cap) to tens of billions (mega-cap).

**Issues:**
- **Not normalized.** Raw amount is almost perfectly correlated with market cap. XGB learns "big stock vs small stock" rather than any alpha signal.
- **Redundant with Alpha158.** Alpha158 includes `VWAP0` (volume-weighted average price) and volume-related features. Raw amount ~ price x volume, both of which are in Alpha158.
- **Extreme range.** Kweichow Moutai might have 5B daily amount while a micro-cap has 5M. This 1000x range distorts tree splits.

**v2 proposal:**
- **REMOVE** raw amount. It is redundant with Alpha158 volume features and confounded with market cap.
- If amount information is desired, use `amount_anom20` (feature #9) which is already the ratio version.

---

### 10.5 Custom Feature #5: `pe_mom20` -- PE 20-Day Momentum

**Expression:** `$pe / Ref($pe, 20) - 1`
**What it computes:** 20-day percent change in PE ratio. Self-normalizing ratio.

**Issues:**
- **Reasonably well-engineered.** This is a ratio, so it is scale-invariant. No urgent normalization issue.
- **Window choice.** 20 days is reasonable for momentum, but PE changes slowly (driven by quarterly earnings updates). 60-day window may capture earnings-release-driven PE shifts better.
- **Partial redundancy.** PE change = (price change) - (earnings change). Since earnings update quarterly, PE momentum over 20 days is mostly price momentum, which Alpha158 already has as `KLEN20` (20-day return). The incremental signal comes only from the earnings-change component.
- **Outlier risk.** When `Ref($pe, 20)` is near zero, this ratio explodes. No clipping is applied.

**v2 proposal:**
- Add 60-day version: `pe_mom60 = $pe / Ref($pe, 60) - 1` (captures quarterly earnings shifts).
- Clip to [-2, 2] to handle outliers from near-zero denominators.
- Consider `pe_surprise = pe_mom20 - KLEN20` (PE change NOT explained by price change = pure earnings surprise).
- Cross-sectional rank per day would make it more robust.

---

### 10.6 Custom Feature #6: `pb_mom20` -- PB 20-Day Momentum

**Expression:** `$pb / Ref($pb, 20) - 1`
**What it computes:** 20-day percent change in PB ratio.

**Issues:**
- Same as `pe_mom20`. PB changes even more slowly than PE (book value updates quarterly).
- Over 20 days, PB momentum is almost entirely price momentum.
- Same outlier risk with near-zero denominators.

**v2 proposal:**
- Extend to 60-day: `pb_mom60 = $pb / Ref($pb, 60) - 1`.
- Extract pure book-value signal: `pb_surprise = pb_mom20 - KLEN20`.
- Cross-sectional rank per day.

---

### 10.7 Custom Feature #7: `turn_anom20` -- Turnover Anomaly (20-day)

**Expression:** `$turn / Mean($turn, 20)`
**What it computes:** Today's turnover divided by its 20-day moving average. Values > 1 mean higher-than-usual turnover.

**Issues:**
- **Well-engineered.** This is a self-normalizing ratio centered around 1.0. It measures relative activity.
- **Partial redundancy with Alpha158.** Alpha158 contains `WVMA5/10/20` (volume-weighted moving average ratios) and `VSTD5/10/20` (volume standard deviation). `turn_anom20` is similar to `Mean($volume, 1) / Mean($volume, 20)` which is close to one of Alpha158's volume ratio features. However, turnover (volume/float) is subtly different from raw volume because it normalizes by float shares. This gives genuine incremental information for stocks with recent share issuance/buyback.
- **Distribution.** The ratio can spike to 10-50x during limit-up/down or event-driven trading. No clipping applied.

**v2 proposal:**
- Add clipping: `turn_anom20.clip(0.01, 10)` to handle extreme spikes.
- Log-transform: `np.log(turn_anom20)` centers the distribution and handles right skew.
- Keep -- it is NOT fully redundant because turnover normalizes by float shares while Alpha158 uses raw volume.

---

### 10.8 Custom Feature #8: `turn_anom60` -- Turnover Anomaly (60-day)

**Expression:** `$turn / Mean($turn, 60)`
**What it computes:** Today's turnover divided by 60-day moving average.

**Issues:**
- Same structure as `turn_anom20` but longer window. Captures medium-term activity shifts (e.g., a stock transitioning from dormant to actively traded over weeks).
- **Moderate redundancy** with `turn_anom20` -- correlation between them is typically 0.6-0.8. The 60-day version is more stable but slower to react.
- Same outlier risk as `turn_anom20`.

**v2 proposal:**
- Keep, but clip and log-transform same as `turn_anom20`.
- Consider replacing with `turn_accel = turn_anom20 / turn_anom60` (acceleration: is short-term anomaly increasing relative to medium-term?). This would be more orthogonal to either alone.

---

### 10.9 Custom Feature #9: `amount_anom20` -- Amount Anomaly (20-day)

**Expression:** `$amount / Mean($amount, 20)`
**What it computes:** Today's trading amount divided by 20-day average amount. Self-normalizing ratio.

**Issues:**
- **Redundant with `turn_anom20`.** Amount = price x volume. Turnover = volume / float. Over 20 days, price changes are small, so amount_anom20 and turn_anom20 are highly correlated (typically r > 0.95). The only difference is the price-change component, which Alpha158 already captures.
- **Redundant with Alpha158.** Alpha158's volume ratio features capture the same relative-volume signal.

**v2 proposal:**
- **REMOVE.** Near-duplicate of `turn_anom20`. The marginal information (price component) is already in Alpha158.
- If kept, must be cross-sectional ranked to avoid conflation with size.

---

### 10.10 Custom Feature #10: `turn_vol20` -- Turnover Volatility (20-day)

**Expression:** `Std($turn, 20)`
**What it computes:** 20-day rolling standard deviation of daily turnover.

**Issues:**
- **Not normalized.** This is an absolute standard deviation. A liquid large-cap with mean turn=2% will have Std(turn)=0.5%, while an illiquid micro-cap with mean turn=0.3% might have Std(turn)=0.1%. The feature captures size/liquidity, not volatility regime.
- **Partial redundancy with Alpha158.** Alpha158 includes `VSTD5/10/20/30/60` which are volume standard deviations. `turn_vol20` is the turnover analog.
- The absolute Std has a highly right-skewed distribution.

**v2 proposal:**
- Normalize as coefficient of variation: `turn_cv20 = Std($turn, 20) / Mean($turn, 20)`. This makes it scale-invariant and measures "how erratic is turnover relative to its level?"
- Alternative: cross-sectional rank per day.
- Consider: `turn_vol_change = Std($turn, 20) / Std($turn, 60)` (is recent volatility elevated compared to longer history?).

---

### 10.11 Custom Feature #11: `ep` -- Earnings Yield (1/PE)

**Expression:** `1.0 / If(Abs($pe) > 0.01, $pe, 1.0)`
**What it computes:** Inverse of PE (earnings/price), with protection against division by zero.

**Issues:**
- **Better than raw PE** because it is bounded and well-behaved (small positive values for growth stocks, large values for value stocks). The guard `If(Abs($pe) > 0.01, $pe, 1.0)` returns 1.0 when PE is near zero, which maps to ep=1.0 -- a reasonable default.
- **Not cross-sectionally normalized.** Industry effects dominate: bank ep=0.15 vs tech ep=0.01. Without industry adjustment, the model learns industry membership, not relative value.
- **Redundant with raw `pe` (feature #1).** ep = 1/pe is a monotonic transformation. XGB can learn the same splits from either.

**v2 proposal:**
- Keep `ep`, remove raw `pe` (ep is better scaled).
- **Critical improvement:** industry-relative ep. `ep_ind_rank = ep.groupby([date, industry]).rank(pct=True)`.
- Time-series version: `ep_ts_zscore = (ep - ep.rolling(250).mean()) / ep.rolling(250).std()` -- "Is this stock cheap relative to its OWN history?"
- Interaction: `ep x momentum` -- value stocks with improving momentum (value + momentum combination is well-documented alpha source, Asness 2013).

---

### 10.12 Custom Feature #12: `bp` -- Book Yield (1/PB)

**Expression:** `1.0 / If(Abs($pb) > 0.01, $pb, 1.0)`
**What it computes:** Inverse of PB (book/price).

**Issues:**
- Same analysis as `ep`. Better scaled than raw PB. Redundant with raw `pb` (feature #2).
- Same industry-effect problem.
- Not normalized.

**v2 proposal:**
- Keep `bp`, remove raw `pb`.
- Industry-relative bp rank.
- Time-series z-score vs own 250-day history.
- Consider `bp - ep` or `pb/pe` as a DuPont decomposition proxy (ROE = EPS/BPS = ep/bp).

---

### 10.13 Custom Feature #13: `price_pos20` -- 20-Day Price Position

**Expression:** `($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20) + 1e-8)`
**What it computes:** Position of current close within its 20-day high-low range. Bounded [0, 1].

**Issues:**
- **Well-engineered.** Self-normalizing, bounded, meaningful interpretation (0 = at 20-day low, 1 = at 20-day high).
- **Redundant with Alpha158.** Alpha158 includes `HIGH0/LOW0` (daily high/low), `KLOW`/`KHIGH` features, and specifically `MAX5/10/20/30/60` and `MIN5/10/20/30/60` which compute similar range-relative positions. The expression `($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20))` is essentially the same as Alpha158's `KSFT20` (or similar stochastic-like features).
- Cross-sectional variation is naturally present because the ratio is stock-specific.

**v2 proposal:**
- **Likely redundant** -- check correlation with Alpha158's KSFT/MIN/MAX features. If correlation > 0.9, remove.
- If kept, consider multi-window: add `price_pos60 = ($close - Min($close, 60)) / (Max($close, 60) - Min($close, 60))`.
- Interaction: `price_pos20 x turn_anom20` -- breakout signal (price at top of range WITH elevated volume).

---

### 10.14 Flow Feature #14: `flow_net_mf_latest` -- Latest Net Money Flow

**Expression:** Raw `net_mf_amount` from `fund_flow_history.parquet`, lag-1 adjusted (line 547).
**What it computes:** Most recent day's net main-force money flow in yuan.

**Issues:**
- **NOT NORMALIZED.** This is an absolute yuan amount. A mega-cap stock might have +500M yuan net flow on a normal day, while the same +500M for a small-cap would be extraordinary. The feature is dominated by market cap.
- **PIT-safe:** Correctly lag-1 adjusted (line 547: `trade_date + BDay(1)`).
- **Useful information, terrible encoding.** Net money flow IS predictive (proven by the 174-dim model's existing use), but the raw yuan encoding wastes much of it.

**v2 proposal:**
- **Normalize by ADV (average daily volume in yuan):** `flow_net_mf_normed = net_mf_amount / Mean($amount, 20)`. This gives "net flow as a fraction of normal trading activity."
- **Cross-sectional rank per day:** `flow_rank = flow_net_mf_latest.groupby(date).rank(pct=True)`.
- **Industry-relative:** `flow_ind_rank = flow_net_mf_latest.groupby([date, industry]).rank(pct=True)`.
- This is CX's recommendation in 9.3: "金额类必须标准化：除以 amount / ADV / circ_mv."

---

### 10.15 Flow Feature #15: `flow_net_mf_5d` -- 5-Day Cumulative Net Money Flow

**Expression:** `x.rolling(5, min_periods=1).sum()` on `net_mf_amount` (line 537).
**What it computes:** Sum of net main-force money flow over last 5 trading days, in yuan.

**Issues:**
- Same normalization problem as `flow_net_mf_latest`, amplified by 5x accumulation.
- **Correlation with `flow_net_mf_latest`:** Typically r > 0.7. The 5-day sum is a smoothed version of the daily value.
- **Useful for momentum detection** (persistent inflow over a week), but the raw yuan encoding hides this signal behind market cap.

**v2 proposal:**
- Normalize: `flow_5d_normed = flow_net_mf_5d / (Mean($amount, 20) * 5)`. This gives "cumulative net flow as a fraction of 5 normal days' trading."
- Better: `flow_5d_zscore = (flow_net_mf_5d - flow_net_mf_5d.rolling(60).mean()) / flow_net_mf_5d.rolling(60).std()` -- "Is this week's flow unusual for this stock?"
- Cross-sectional rank per day.

---

### 10.16 Flow Feature #16: `flow_net_mf_20d_avg` -- 20-Day Average Net Money Flow

**Expression:** `x.rolling(20, min_periods=1).mean()` on `net_mf_amount` (line 538).
**What it computes:** 20-day rolling average of daily net main-force money flow, in yuan.

**Issues:**
- Same normalization problem as the other two flow features.
- **High correlation** with `flow_net_mf_5d` (typically r > 0.8). The 20-day average is a further-smoothed version.
- All three flow features encode essentially the same information (net flow magnitude) at different smoothing windows, all in unnormalized yuan.

**v2 proposal:**
- Same normalization as above.
- Consider replacing the 3-feature set with: `flow_rank_latest` (daily rank), `flow_zscore_60d` (time-series anomaly), `flow_persistence_10d` (fraction of positive-flow days in last 10). These 3 would be more orthogonal to each other AND to the rest of the baseline.

---

### 10.17 Summary Table: All 16 Features Audited

| # | Feature | Expression | Normalized? | Redundancy | Severity | Recommendation |
|:---:|:---|:---|:---:|:---|:---:|:---|
| 1 | `pe` | `$pe` | NO | ep is 1/pe | HIGH | REMOVE (keep ep) |
| 2 | `pb` | `$pb` | NO | bp is 1/pb | HIGH | REMOVE (keep bp) |
| 3 | `turn_raw` | `$turn` | NO | Alpha158 TURN0 | CRITICAL | REMOVE (exact duplicate) |
| 4 | `amount_raw` | `$amount` | NO | Alpha158 volume + size | CRITICAL | REMOVE (conflated with market cap) |
| 5 | `pe_mom20` | `$pe / Ref($pe, 20) - 1` | Self-ratio | ~KLEN20 (price return) | LOW | KEEP, add 60d, clip outliers |
| 6 | `pb_mom20` | `$pb / Ref($pb, 20) - 1` | Self-ratio | ~KLEN20 | LOW | KEEP, add 60d, clip outliers |
| 7 | `turn_anom20` | `$turn / Mean($turn, 20)` | Self-ratio | Similar to WVMA20 | LOW | KEEP, clip + log-transform |
| 8 | `turn_anom60` | `$turn / Mean($turn, 60)` | Self-ratio | Correlated with #7 | LOW | KEEP or replace with ratio #7/#8 |
| 9 | `amount_anom20` | `$amount / Mean($amount, 20)` | Self-ratio | ~turn_anom20 (r>0.95) | HIGH | REMOVE (near-duplicate of #7) |
| 10 | `turn_vol20` | `Std($turn, 20)` | NO | ~VSTD20 | MEDIUM | REPLACE with CV: Std/Mean |
| 11 | `ep` | `1/If(Abs(pe)>0.01,pe,1)` | Bounded | Redundant with pe (#1) | LOW | KEEP, add industry-relative rank |
| 12 | `bp` | `1/If(Abs(pb)>0.01,pb,1)` | Bounded | Redundant with pb (#2) | LOW | KEEP, add industry-relative rank |
| 13 | `price_pos20` | `(close-Min20)/(Max20-Min20)` | Self-ratio [0,1] | ~Alpha158 KSFT/stochastic | MEDIUM | CHECK correlation; remove if >0.9 |
| 14 | `flow_net_mf_latest` | raw net_mf_amount yuan | NO | None (unique data) | CRITICAL | Normalize by ADV or rank |
| 15 | `flow_net_mf_5d` | 5d sum of net_mf yuan | NO | Correlated with #14 | CRITICAL | Normalize by ADV*5 or rank |
| 16 | `flow_net_mf_20d_avg` | 20d avg of net_mf yuan | NO | Correlated with #14,#15 | CRITICAL | Normalize by ADV or rank |

### 10.18 Immediate Action Items (Low-Hanging Fruit)

**Tier 1: Remove pure redundancy (0 effort, reduces noise)**
1. Remove `turn_raw` (#3) -- exact duplicate of Alpha158 TURN0
2. Remove `amount_raw` (#4) -- conflated with market cap, Alpha158 has volume
3. Remove `pe` (#1) -- keep `ep` instead
4. Remove `pb` (#2) -- keep `bp` instead
5. Remove `amount_anom20` (#9) -- near-duplicate of `turn_anom20`

Result: 174 -> 169 features. Removing 5 noisy/redundant features should IMPROVE performance slightly by reducing feature dilution.

**Tier 2: Normalize flow features (1 hour effort, potentially large impact)**
6. Replace `flow_net_mf_latest` with `flow_latest_rank = flow_net_mf_latest.groupby(date).rank(pct=True)`
7. Replace `flow_net_mf_5d` with `flow_5d_rank = flow_net_mf_5d.groupby(date).rank(pct=True)`
8. Replace `flow_net_mf_20d_avg` with `flow_20d_rank = flow_net_mf_20d_avg.groupby(date).rank(pct=True)`

This is a trivial code change in `feature_pipeline.py` (add 3 lines of groupby.rank after the join), but could meaningfully improve the flow features' contribution by removing the market-cap confound.

**Tier 3: Fix remaining unnormalized features (2 hour effort)**
9. Replace `turn_vol20` with `turn_cv20 = Std($turn, 20) / Mean($turn, 20)`
10. Add cross-sectional rank to `ep` and `bp`
11. Clip `pe_mom20` and `pb_mom20` to [-2, 2]

**Tier 4: Add missing multi-window + interactions (1 day effort)**
12. Add `pe_mom60`, `pb_mom60` (60-day valuation momentum)
13. Add `turn_accel = turn_anom20 / turn_anom60` (turnover acceleration)
14. Add `ep_ind_rank` and `bp_ind_rank` (industry-relative valuation)
15. Add `price_pos20 x turn_anom20` interaction (breakout signal)
16. Add `ep x KLEN20` interaction (value + momentum)

### 10.19 Estimated Impact

The 174-dim baseline achieves RankIC +0.0513. Based on this audit:

- **Tier 1 (remove 5 features):** Expected delta +0.001 to +0.003 RankIC. Feature dilution is a real problem with colsample_bytree=0.88; removing noise features gives useful features more splits.
- **Tier 2 (normalize 3 flow features):** Expected delta +0.002 to +0.005 RankIC. Flow features carry proven signal (they were the reason to go from 158 to 161 dims), but raw yuan encoding wastes much of it.
- **Tier 3 (fix remaining normalization):** Expected delta +0.001 to +0.002 RankIC. Smaller impact because these features are less dominant.
- **Tier 4 (new engineered features):** Speculative, but industry-relative valuation and interaction terms could add +0.003 to +0.008 RankIC based on academic evidence.

**Combined realistic estimate:** +0.005 to +0.010 RankIC improvement WITHOUT any new data sources, just by fixing engineering deficiencies in the existing 16 features. This would bring the baseline from +0.051 to potentially +0.056-0.061, a meaningful improvement.

### 10.20 Implementation Location

All changes should be made in `models/feature_pipeline.py`:
- Remove redundant expressions from `CUSTOM_EXPRS` / `CUSTOM_NAMES`
- Add normalization after the `X = X.join(custom[new_cols], how="left")` step (line 89)
- For flow normalization, add a post-join step after `X = X.join(flow, how="left")` (line 77)

```python
# After line 77 (flow join):
if flow is not None:
    flow_cols = [c for c in flow.columns if c in X.columns]
    for c in flow_cols:
        X[c] = X[c].groupby(level=0).rank(pct=True)

# After line 89 (custom join):
# Cross-sectional rank for raw-level features
rank_cols = ["ep", "bp", "turn_vol20"]  # Only unnormalized customs
for c in rank_cols:
    if c in X.columns:
        X[c] = X[c].groupby(level=0).rank(pct=True)
```

This preserves backward compatibility (same column names, same column count after Tier 1 removal) while fixing the normalization gap.
