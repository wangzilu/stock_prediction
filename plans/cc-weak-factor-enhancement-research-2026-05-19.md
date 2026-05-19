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
