"""Phase 4J: Institutional Metric Gate for xgb_174 champion model.

Computes institutional-grade metrics across 24 rolling splits:
- Signal layer: IC/ICIR, RankIC/RankICIR, IC decay, multi-layer spreads
- Portfolio layer: excess return vs CSI500, IR, turnover, cost drag

Usage:
    python scripts/phase4j_institutional_gate.py
"""
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import xgboost as xgb

from config.qlib_runtime import init_qlib
from utils.json_utils import json_default

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")

# Rolling config
N_SPLITS = 24
TEST_DAYS = 20
VALID_DAYS = 60
TRAIN_DAYS = 750  # ~3 years

# XGB params (tuned)
XGB_PARAMS = {
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8789,
    "colsample_bytree": 0.8879,
    "reg_alpha": 205.6999,
    "reg_lambda": 580.9768,
    "objective": "reg:squarederror",
    "nthread": 12,
    "verbosity": 0,
    "seed": 42,
}
NUM_BOOST_ROUND = 400
EARLY_STOPPING_ROUNDS = 50

# Cost model
COMMISSION = 0.0003 * 2   # buy + sell
STAMP_TAX = 0.0005         # sell side
SLIPPAGE = 0.001 * 2       # buy + sell
TOTAL_COST_PER_TRADE = COMMISSION + STAMP_TAX + SLIPPAGE  # ~0.0026

# Multi-layer spread tiers
SPREAD_TIERS = [20, 50, 100, 300]

LABEL_COL = "__label_5d"
PNL_COL = "__pnl_return_1d"


def load_cache():
    """Load feature cache and separate features / labels / pnl."""
    logger.info("Loading feature cache...")
    cache = pd.read_parquet(str(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"))
    logger.info(f"  Cache shape: {cache.shape}")

    # Feature columns: exclude __ prefixed meta columns, cross-market regime cols
    feat_cols = [c for c in cache.columns
                 if not c.startswith("__") and not c.startswith("_")
                 and not c.startswith("hsi_") and not c.startswith("hstech_")
                 and not c.startswith("nasdaq_")]
    logger.info(f"  Feature cols: {len(feat_cols)}")

    return cache, feat_cols


def load_benchmark(trade_dates):
    """Load CSI500 benchmark daily returns from Qlib."""
    from qlib.data import D
    start = str(trade_dates[0])[:10]
    end = str(trade_dates[-1])[:10]
    bm = D.features(
        ["sh000905"],
        ["Ref($close, -1) / $close - 1"],
        start_time=start,
        end_time=end,
    )
    if bm is not None and not bm.empty:
        bm.columns = ["bm_return"]
        # Flatten to date-indexed series
        bm = bm.droplevel(0) if bm.index.nlevels > 1 else bm
        bm = bm.replace([np.inf, -np.inf], np.nan).dropna()
        return bm["bm_return"]
    logger.warning("Benchmark data unavailable")
    return pd.Series(dtype=float)


def compute_ic_decay(pred_series, label_series, max_lag=20):
    """Compute IC at different lags (signal persistence)."""
    from scipy.stats import spearmanr
    results = {}
    dates = sorted(pred_series.index.get_level_values(0).unique())

    # Build date -> predictions map
    pred_by_date = {}
    label_by_date = {}
    for d in dates:
        mask_d = pred_series.index.get_level_values(0) == d
        p = pred_series[mask_d]
        l = label_series[mask_d]
        if len(p) > 20:
            pred_by_date[d] = p
            label_by_date[d] = l

    sorted_dates = sorted(pred_by_date.keys())
    lags = [1, 5, 10, 20]

    for lag in lags:
        if lag > max_lag:
            break
        ics = []
        for i in range(len(sorted_dates) - lag):
            d_signal = sorted_dates[i]
            d_label = sorted_dates[i + lag]
            p = pred_by_date[d_signal]
            l = label_by_date.get(d_label)
            if l is None:
                continue
            # Align instruments
            common = p.index.get_level_values(1).intersection(l.index.get_level_values(1))
            if len(common) < 20:
                continue
            p_aligned = p.loc[p.index.get_level_values(1).isin(common)]
            l_aligned = l.loc[l.index.get_level_values(1).isin(common)]
            # Sort by instrument for alignment
            p_vals = p_aligned.sort_index(level=1).values
            l_vals = l_aligned.sort_index(level=1).values
            if len(p_vals) == len(l_vals) and len(p_vals) > 10:
                corr, _ = spearmanr(p_vals, l_vals)
                if np.isfinite(corr):
                    ics.append(corr)
        results[f"lag_{lag}"] = round(float(np.mean(ics)), 6) if ics else None
    return results


def compute_multi_layer_spread(pred_series, pnl_series):
    """Compute top-K vs bottom-K spread for multiple tiers."""
    df = pd.DataFrame({"pred": pred_series, "pnl": pnl_series}).dropna()
    results = {}

    for k in SPREAD_TIERS:
        spreads = []
        for date, g in df.groupby(level=0):
            if len(g) < k * 2:
                continue
            s = g.sort_values("pred", ascending=False)
            top_ret = s.head(k)["pnl"].mean()
            bot_ret = s.tail(k)["pnl"].mean()
            spreads.append(top_ret - bot_ret)

        if spreads:
            results[f"top{k}"] = {
                "daily_spread_mean": round(float(np.mean(spreads)), 6),
                "daily_spread_std": round(float(np.std(spreads)), 6),
                "annualized_spread": round(float(np.mean(spreads) * 252), 6),
                "spread_pos_pct": round(float(np.mean([s > 0 for s in spreads])), 4),
                "n_days": len(spreads),
            }
    return results


def compute_portfolio_metrics(pred_series, pnl_series, bm_series):
    """Compute portfolio layer metrics: excess return, IR, turnover, cost drag."""
    df = pd.DataFrame({"pred": pred_series, "pnl": pnl_series}).dropna()
    dates = sorted(df.index.get_level_values(0).unique())

    # Daily top-20 equal-weight portfolio return
    daily_port_ret = {}
    daily_top_sets = {}
    for date in dates:
        g = df.loc[date]
        if len(g) < 40:
            continue
        s = g.sort_values("pred", ascending=False)
        top20 = s.head(20)
        daily_port_ret[date] = float(top20["pnl"].mean())
        daily_top_sets[date] = set(top20.index.get_level_values(-1) if top20.index.nlevels > 1
                                   else top20.index)

    port_ret = pd.Series(daily_port_ret).sort_index()

    # Align benchmark
    common_dates = port_ret.index.intersection(bm_series.index)
    if len(common_dates) == 0:
        logger.warning("No overlapping dates with benchmark")
        bm_aligned = pd.Series(0.0, index=port_ret.index)
    else:
        bm_aligned = bm_series.reindex(port_ret.index).fillna(0.0)

    excess = port_ret - bm_aligned

    # Annualized metrics
    ann_excess = float(excess.mean() * 252)
    ann_tracking_error = float(excess.std() * np.sqrt(252))
    ir = ann_excess / ann_tracking_error if ann_tracking_error > 1e-8 else 0.0

    # Turnover: fraction of top-20 set that changes day-to-day
    sorted_dates_with_sets = sorted(daily_top_sets.keys())
    turnovers = []
    for i in range(1, len(sorted_dates_with_sets)):
        prev_set = daily_top_sets[sorted_dates_with_sets[i - 1]]
        curr_set = daily_top_sets[sorted_dates_with_sets[i]]
        if len(prev_set) > 0 and len(curr_set) > 0:
            changed = len(curr_set - prev_set)
            turnover_rate = changed / max(len(curr_set), 1)
            turnovers.append(turnover_rate)

    daily_turnover = float(np.mean(turnovers)) if turnovers else 0.0
    annual_turnover = daily_turnover * 252  # one-way annual turnover

    # Cost drag
    cost_drag_annual = annual_turnover * TOTAL_COST_PER_TRADE

    return {
        "port_daily_ret_mean": round(float(port_ret.mean()), 6),
        "port_daily_ret_std": round(float(port_ret.std()), 6),
        "bm_daily_ret_mean": round(float(bm_aligned.mean()), 6),
        "excess_daily_mean": round(float(excess.mean()), 6),
        "excess_daily_std": round(float(excess.std()), 6),
        "ann_excess_return": round(ann_excess, 6),
        "ann_tracking_error": round(ann_tracking_error, 6),
        "information_ratio": round(ir, 4),
        "daily_turnover": round(daily_turnover, 4),
        "annual_turnover": round(annual_turnover, 2),
        "cost_per_trade": round(TOTAL_COST_PER_TRADE, 6),
        "cost_drag_annual": round(cost_drag_annual, 6),
        "net_ann_excess": round(ann_excess - cost_drag_annual, 6),
        "n_days": len(port_ret),
    }


def main():
    init_qlib(QLIB_DATA)
    cache, feat_cols = load_cache()

    trade_dates = sorted(cache.index.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1
    logger.info(f"Trading calendar: {len(trade_dates)} dates, "
                f"{str(trade_dates[0])[:10]} ~ {str(trade_dates[-1])[:10]}")

    # Load benchmark
    bm_series = load_benchmark(trade_dates)
    logger.info(f"Benchmark (CSI500): {len(bm_series)} days loaded")

    dl = cache.index.get_level_values(0)

    all_split_metrics = []
    all_pred_series = []
    all_label_series = []
    all_pnl_series = []
    t_total = time.time()

    for split_idx in range(N_SPLITS):
        test_end_idx = today_idx - split_idx * TEST_DAYS
        test_start_idx = test_end_idx - TEST_DAYS
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - VALID_DAYS
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - TRAIN_DAYS

        if train_start_idx < 0 or test_end_idx >= len(trade_dates):
            logger.warning(f"Split {split_idx+1}: out of bounds, stopping")
            break

        test_end = trade_dates[test_end_idx]
        test_start = trade_dates[test_start_idx]
        valid_end = trade_dates[valid_end_idx]
        valid_start = trade_dates[valid_start_idx]
        train_end = trade_dates[train_end_idx]
        train_start = trade_dates[train_start_idx]

        logger.info(f"\nSplit {split_idx+1}/{N_SPLITS}: "
                    f"test {str(test_start)[:10]}~{str(test_end)[:10]} "
                    f"(train {str(train_start)[:10]}~{str(train_end)[:10]})")

        try:
            t0 = time.time()

            # Slice data
            train_mask = (dl >= train_start) & (dl <= train_end)
            valid_mask = (dl >= valid_start) & (dl <= valid_end)
            test_mask = (dl >= test_start) & (dl <= test_end)

            train_df = cache.loc[train_mask]
            valid_df = cache.loc[valid_mask]
            test_df = cache.loc[test_mask]

            # Prepare arrays
            def prep(df):
                X = df[feat_cols].values.astype(np.float32)
                y = df[LABEL_COL].values.astype(np.float32)
                mask = np.isfinite(y)
                return X[mask], y[mask], df.index[mask]

            X_tr, y_tr, idx_tr = prep(train_df)
            X_va, y_va, idx_va = prep(valid_df)
            X_te, y_te, idx_te = prep(test_df)

            if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
                logger.warning(f"  Split {split_idx+1}: empty segment, skipping")
                continue

            # Train XGB
            dt = xgb.DMatrix(X_tr, label=y_tr)
            dv = xgb.DMatrix(X_va, label=y_va)
            model = xgb.train(
                XGB_PARAMS, dt,
                num_boost_round=NUM_BOOST_ROUND,
                evals=[(dv, "valid")],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=0,
            )
            pred = model.predict(xgb.DMatrix(X_te))

            # --- Signal layer metrics ---
            from qlib.contrib.eva.alpha import calc_ic
            mask = np.isfinite(pred) & np.isfinite(y_te)
            ps = pd.Series(pred[mask], index=idx_te[mask])
            ls = pd.Series(y_te[mask], index=idx_te[mask])
            ic_s, ric_s = calc_ic(ps, ls)

            ic_mean = float(ic_s.mean())
            ic_std = float(ic_s.std())
            icir = ic_mean / (ic_std + 1e-8)
            ric_mean = float(ric_s.mean())
            ric_std = float(ric_s.std())
            ricir = ric_mean / (ric_std + 1e-8)

            # PnL returns for this split
            pnl_col_data = test_df.loc[idx_te[mask], PNL_COL] if PNL_COL in cache.columns else None

            elapsed = time.time() - t0

            split_result = {
                "split": split_idx + 1,
                "test_start": str(test_start)[:10],
                "test_end": str(test_end)[:10],
                "train_start": str(train_start)[:10],
                "n_train": len(X_tr),
                "n_test": len(X_te),
                "ic_mean": round(ic_mean, 6),
                "ic_std": round(ic_std, 6),
                "icir": round(icir, 4),
                "rank_ic_mean": round(ric_mean, 6),
                "rank_ic_std": round(ric_std, 6),
                "rank_icir": round(ricir, 4),
                "time_s": round(elapsed, 1),
            }
            all_split_metrics.append(split_result)
            all_pred_series.append(ps)
            all_label_series.append(ls)
            if pnl_col_data is not None:
                all_pnl_series.append(pnl_col_data)

            logger.info(f"  IC={ic_mean:+.4f} ICIR={icir:+.3f} "
                        f"RankIC={ric_mean:+.4f} RankICIR={ricir:+.3f} "
                        f"[{elapsed:.0f}s]")

        except Exception as e:
            logger.error(f"  Split {split_idx+1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_split_metrics:
        logger.error("No valid splits")
        sys.exit(1)

    total_time = time.time() - t_total
    n = len(all_split_metrics)

    # === Concatenate all predictions ===
    all_pred = pd.concat(all_pred_series)
    all_label = pd.concat(all_label_series)
    all_pnl = pd.concat(all_pnl_series) if all_pnl_series else pd.Series(dtype=float)

    # === Aggregate signal metrics ===
    agg_ic_mean = float(np.mean([s["ic_mean"] for s in all_split_metrics]))
    agg_ic_std = float(np.mean([s["ic_std"] for s in all_split_metrics]))
    agg_icir = float(np.mean([s["icir"] for s in all_split_metrics]))
    agg_ric_mean = float(np.mean([s["rank_ic_mean"] for s in all_split_metrics]))
    agg_ric_std = float(np.mean([s["rank_ic_std"] for s in all_split_metrics]))
    agg_ricir = float(np.mean([s["rank_icir"] for s in all_split_metrics]))

    # === IC Decay ===
    logger.info("\nComputing IC decay...")
    ic_decay = compute_ic_decay(all_pred, all_label)
    logger.info(f"  IC decay: {ic_decay}")

    # === Multi-layer spreads ===
    logger.info("Computing multi-layer spreads...")
    if not all_pnl.empty:
        multi_spreads = compute_multi_layer_spread(all_pred, all_pnl)
    else:
        multi_spreads = {}
    for tier, vals in multi_spreads.items():
        logger.info(f"  {tier}: daily spread={vals['daily_spread_mean']*100:+.3f}% "
                    f"ann={vals['annualized_spread']*100:+.1f}% "
                    f"pos={vals['spread_pos_pct']:.0%}")

    # === Portfolio layer ===
    logger.info("Computing portfolio metrics...")
    if not all_pnl.empty:
        port_metrics = compute_portfolio_metrics(all_pred, all_pnl, bm_series)
    else:
        port_metrics = {}
        logger.warning("No PnL data available for portfolio metrics")

    # ============================================================
    # Print summary
    # ============================================================
    logger.info(f"\n{'='*100}")
    logger.info(f"PHASE 4J INSTITUTIONAL METRIC GATE: xgb_174 ({n} splits x {TEST_DAYS} days)")
    logger.info(f"{'='*100}")

    # Per-split table
    logger.info(f"\n{'Split':<6} {'Test Period':<24} {'IC':>8} {'ICIR':>7} {'RankIC':>8} "
                f"{'RICIR':>7} {'N_test':>7}")
    logger.info("-" * 75)
    for r in all_split_metrics:
        logger.info(
            f"{r['split']:<6} {r['test_start']}~{r['test_end']}  "
            f"{r['ic_mean']:+.4f} {r['icir']:+.3f}  {r['rank_ic_mean']:+.4f}  "
            f"{r['rank_icir']:+.3f}  {r['n_test']:>6}"
        )

    # Aggregate signal
    logger.info(f"\n{'='*100}")
    logger.info("SIGNAL LAYER AGGREGATE")
    logger.info(f"{'='*100}")
    logger.info(f"  IC  mean={agg_ic_mean:+.4f}  std={agg_ic_std:.4f}  ICIR={agg_icir:+.3f}")
    logger.info(f"  RIC mean={agg_ric_mean:+.4f}  std={agg_ric_std:.4f}  RICIR={agg_ricir:+.3f}")

    # IC Decay
    logger.info(f"\n  IC Decay (signal persistence):")
    for lag, val in ic_decay.items():
        logger.info(f"    {lag}: {val:+.4f}" if val is not None else f"    {lag}: N/A")

    # Multi-layer spreads
    logger.info(f"\n  Multi-layer spreads (daily, using 1d PnL returns):")
    for tier, vals in multi_spreads.items():
        logger.info(f"    {tier}: mean={vals['daily_spread_mean']*100:+.3f}%  "
                    f"ann={vals['annualized_spread']*100:+.1f}%  "
                    f"pos={vals['spread_pos_pct']:.0%}  n={vals['n_days']}")

    # Portfolio layer
    if port_metrics:
        logger.info(f"\n{'='*100}")
        logger.info("PORTFOLIO LAYER (Top-20 EW vs CSI500)")
        logger.info(f"{'='*100}")
        logger.info(f"  Portfolio daily mean:  {port_metrics['port_daily_ret_mean']*100:+.4f}%")
        logger.info(f"  Benchmark daily mean:  {port_metrics['bm_daily_ret_mean']*100:+.4f}%")
        logger.info(f"  Excess daily mean:     {port_metrics['excess_daily_mean']*100:+.4f}%")
        logger.info(f"  Ann excess return:     {port_metrics['ann_excess_return']*100:+.2f}%")
        logger.info(f"  Ann tracking error:    {port_metrics['ann_tracking_error']*100:.2f}%")
        logger.info(f"  Information Ratio:     {port_metrics['information_ratio']:+.3f}")
        logger.info(f"  Daily turnover:        {port_metrics['daily_turnover']:.1%}")
        logger.info(f"  Annual turnover:       {port_metrics['annual_turnover']:.1f}x")
        logger.info(f"  Cost per trade:        {port_metrics['cost_per_trade']*100:.2f}%")
        logger.info(f"  Cost drag (annual):    {port_metrics['cost_drag_annual']*100:.2f}%")
        logger.info(f"  Net ann excess:        {port_metrics['net_ann_excess']*100:+.2f}%")

    logger.info(f"\n  Total time: {total_time/60:.1f} min")

    # ============================================================
    # Save results
    # ============================================================
    out_dir = DATA_DIR / "phase4"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "institutional_gate_xgb_174.json"

    summary = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "xgb_174",
        "config": {
            "n_splits": n,
            "test_days": TEST_DAYS,
            "valid_days": VALID_DAYS,
            "train_days": TRAIN_DAYS,
            "num_boost_round": NUM_BOOST_ROUND,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "xgb_params": XGB_PARAMS,
            "cost_model": {
                "commission": COMMISSION,
                "stamp_tax": STAMP_TAX,
                "slippage": SLIPPAGE,
                "total_per_trade": TOTAL_COST_PER_TRADE,
            },
        },
        "signal_aggregate": {
            "ic_mean": round(agg_ic_mean, 6),
            "ic_std": round(agg_ic_std, 6),
            "icir": round(agg_icir, 4),
            "rank_ic_mean": round(agg_ric_mean, 6),
            "rank_ic_std": round(agg_ric_std, 6),
            "rank_icir": round(agg_ricir, 4),
        },
        "ic_decay": ic_decay,
        "multi_layer_spreads": multi_spreads,
        "portfolio": port_metrics,
        "per_split": all_split_metrics,
    }

    with open(str(out_path), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=json_default)
    logger.info(f"\nSaved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
