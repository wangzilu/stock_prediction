"""Phase 4 Track A: Formal rolling gate for XGB 174.

24+ split rolling validation with:
- Trading-day aligned windows (not calendar days)
- Multiple training window comparison (2yr/3yr/5yr)
- Regime breakdown (bull/bear/high-vol/low-vol)
- Promotion gate check

Usage:
    python scripts/phase4_rolling_gate.py
    python scripts/phase4_rolling_gate.py --n-splits 24 --train-years 3
    python scripts/phase4_rolling_gate.py --n-splits 24 --train-years 5
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config.settings import PREDICTION_HORIZON_DAYS
from config.qlib_runtime import init_qlib
from models.feature_merger import FeatureMerger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"
SEED = 42

CUSTOM_EXPRS = [
    "$pe", "$pb", "$turn", "$amount",
    "$pe / Ref($pe, 20) - 1", "$pb / Ref($pb, 20) - 1",
    "$turn / Mean($turn, 20)", "$turn / Mean($turn, 60)",
    "$amount / Mean($amount, 20)", "Std($turn, 20)",
    "1.0 / If(Abs($pe) > 0.01, $pe, 1.0)",
    "1.0 / If(Abs($pb) > 0.01, $pb, 1.0)",
    "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20) + 1e-8)",
]
CUSTOM_NAMES = [
    "pe", "pb", "turn_raw", "amount_raw",
    "pe_mom20", "pb_mom20", "turn_anom20", "turn_anom60",
    "amount_anom20", "turn_vol20", "ep", "bp", "price_pos20",
]

# Promotion gate thresholds (from CX plan, CC adjusted)
GATE = {
    "avg_rank_ic": 0.04,       # >= 0.04 (CC: tighter than CX's 0.035)
    "avg_spread": 0.012,       # >= 1.2%
    "rank_ic_pos_pct": 0.65,   # >= 65% splits RankIC > 0
    "spread_pos_pct": 0.65,    # >= 65% splits Spread > 0
    "worst20_spread": -0.015,  # worst 20% avg spread > -1.5%
}


def get_trading_dates():
    """Get trading date calendar from Qlib."""
    from qlib.data import D
    # Use a liquid stock to get trade dates
    cal = D.calendar(start_time="2020-01-01", end_time=datetime.now().strftime("%Y-%m-%d"))
    return sorted(cal)


def prepare_174(dataset, seg, merger):
    """Prepare 174-dim features: Alpha158 + flow + custom."""
    from qlib.data import D
    X = dataset.prepare(seg, col_set="feature")
    y = dataset.prepare(seg, col_set="label")
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    flow = merger._load_capital_flow(X.index)
    if flow is not None:
        X = X.join(flow, how="left")

    insts = list(set(str(c) for c in X.index.get_level_values(1)))
    dates = sorted(X.index.get_level_values(0).unique())
    custom = D.features(insts, CUSTOM_EXPRS,
                        start_time=str(min(dates))[:10], end_time=str(max(dates))[:10])
    if custom is not None and not custom.empty:
        custom.columns = CUSTOM_NAMES
        custom = custom.swaplevel().sort_index().reindex(X.index)
        custom = custom.replace([np.inf, -np.inf], np.nan)
        new_cols = [c for c in custom.columns if c not in set(X.columns)]
        if new_cols:
            X = X.join(custom[new_cols], how="left")
    return X, y


def train_xgb(X_train, y_train, X_valid, y_valid):
    import xgboost as xgb
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_valid, label=y_valid)
    params = {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
              "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
              "objective": "reg:squarederror", "nthread": 4, "verbosity": 0, "seed": SEED}
    model = xgb.train(params, dt, num_boost_round=500,
                      evals=[(dv, "valid")], early_stopping_rounds=50, verbose_eval=0)
    return model


def evaluate_split(pred, label, index):
    """Full evaluation for one split."""
    from qlib.contrib.eva.alpha import calc_ic
    mask = np.isfinite(pred) & np.isfinite(label)
    ps = pd.Series(pred[mask], index=index[mask])
    ls = pd.Series(label[mask], index=index[mask])
    ic, ric = calc_ic(ps, ls)

    # Per-day spreads and returns
    spreads = []
    top20_returns = []
    for _, g in pd.DataFrame({"pred": ps, "label": ls}).groupby(level=0):
        if len(g) < 40:
            continue
        s = g.sort_values("pred", ascending=False)
        top_ret = s.head(20)["label"].mean()
        bot_ret = s.tail(20)["label"].mean()
        spreads.append(top_ret - bot_ret)
        top20_returns.append(top_ret)

    # Regime classification based on market return (avg of all stocks)
    daily_mkt = []
    for _, g in pd.DataFrame({"label": ls}).groupby(level=0):
        daily_mkt.append(g["label"].mean())
    avg_mkt_return = np.mean(daily_mkt) if daily_mkt else 0
    mkt_vol = np.std(daily_mkt) if len(daily_mkt) > 1 else 0

    # Simple regime: bull/bear × high/low vol
    if avg_mkt_return > 0.001:
        trend = "bull"
    elif avg_mkt_return < -0.001:
        trend = "bear"
    else:
        trend = "neutral"
    vol_regime = "high_vol" if mkt_vol > 0.015 else "low_vol"

    return {
        "ic_mean": round(float(ic.mean()), 6),
        "icir": round(float(ic.mean()) / (float(ic.std()) + 1e-8), 4),
        "rank_ic_mean": round(float(ric.mean()), 6),
        "rank_ic_pos": round(float((ric > 0).mean()), 4),
        "top20_spread": round(float(np.mean(spreads)) if spreads else 0, 6),
        "spread_pos": round(float(np.mean([s > 0 for s in spreads])) if spreads else 0, 4),
        "top20_avg_return": round(float(np.mean(top20_returns)) if top20_returns else 0, 6),
        "n_days": len(spreads),
        "regime_trend": trend,
        "regime_vol": vol_regime,
        "mkt_return": round(avg_mkt_return * 100, 4),
        "mkt_vol": round(mkt_vol * 100, 4),
    }


def check_gate(results):
    """Check promotion gate thresholds."""
    ric_values = [r["rank_ic_mean"] for r in results]
    spread_values = [r["top20_spread"] for r in results]
    n = len(results)

    avg_ric = np.mean(ric_values)
    avg_spread = np.mean(spread_values)
    ric_pos_pct = sum(1 for r in ric_values if r > 0) / n
    spread_pos_pct = sum(1 for s in spread_values if s > 0) / n

    # Worst 20% spread
    sorted_spreads = sorted(spread_values)
    worst_n = max(1, int(n * 0.2))
    worst20_avg = np.mean(sorted_spreads[:worst_n])

    checks = {
        "avg_rank_ic": (avg_ric, avg_ric >= GATE["avg_rank_ic"]),
        "avg_spread": (avg_spread, avg_spread >= GATE["avg_spread"]),
        "rank_ic_pos_pct": (ric_pos_pct, ric_pos_pct >= GATE["rank_ic_pos_pct"]),
        "spread_pos_pct": (spread_pos_pct, spread_pos_pct >= GATE["spread_pos_pct"]),
        "worst20_spread": (worst20_avg, worst20_avg >= GATE["worst20_spread"]),
    }

    all_pass = all(passed for _, passed in checks.values())
    return checks, all_pass


def main():
    import xgboost as xgb

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=24)
    parser.add_argument("--test-days", type=int, default=20,
                        help="Trading days per test window")
    parser.add_argument("--valid-days", type=int, default=60)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--model", type=str, default="xgb_174",
                        choices=["xgb_174", "xgb_175"])
    args = parser.parse_args()

    from qlib.utils import init_instance_by_config
    init_qlib(QLIB_DATA)
    merger = FeatureMerger(DATA_DIR)

    # Get trading date calendar for proper alignment
    trade_dates = get_trading_dates()
    logger.info(f"Trading calendar: {len(trade_dates)} days, "
                f"{str(trade_dates[0])[:10]} ~ {str(trade_dates[-1])[:10]}")

    # Find the latest date index
    today_idx = len(trade_dates) - 1

    all_results = []
    t_total = time.time()

    for split_idx in range(args.n_splits):
        # Walk backward by trading days
        test_end_idx = today_idx - split_idx * args.test_days
        test_start_idx = test_end_idx - args.test_days
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - args.valid_days
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - (args.train_years * 250)  # ~250 trading days/year

        # Bounds check
        if train_start_idx < 0 or test_end_idx >= len(trade_dates):
            logger.warning(f"Split {split_idx+1}: out of bounds, stopping")
            break

        test_end = str(trade_dates[test_end_idx])[:10]
        test_start = str(trade_dates[test_start_idx])[:10]
        valid_end = str(trade_dates[valid_end_idx])[:10]
        valid_start = str(trade_dates[valid_start_idx])[:10]
        train_end = str(trade_dates[train_end_idx])[:10]
        train_start = str(trade_dates[train_start_idx])[:10]

        dates = {
            "train": (train_start, train_end),
            "valid": (valid_start, valid_end),
            "test": (test_start, test_end),
        }

        logger.info(f"\nSplit {split_idx+1}/{args.n_splits}: "
                    f"test {test_start}~{test_end} "
                    f"(train {train_start}~{train_end})")

        try:
            t0 = time.time()
            dataset = init_instance_by_config({
                "class": "DatasetH", "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
                        "kwargs": {"start_time": train_start, "end_time": test_end,
                                   "instruments": "all", "label": [LABEL_EXPR]},
                    },
                    "segments": {
                        "train": dates["train"],
                        "valid": dates["valid"],
                        "test": dates["test"],
                    },
                },
            })

            # Prepare features
            segs = {}
            for seg in ["train", "valid", "test"]:
                X, y = prepare_174(dataset, seg, merger)

                # Add holder for 175 model
                if args.model == "xgb_175":
                    holder = merger._load_st_holder_number(X.index)
                    if holder is not None:
                        X = X.join(holder, how="left")

                Xn = X.values.astype(np.float32)
                yn = y.values.astype(np.float32)
                mask = np.isfinite(yn)
                segs[seg] = (Xn[mask], yn[mask], X.index[mask])

            X_train, y_train, _ = segs["train"]
            X_valid, y_valid, _ = segs["valid"]
            X_test, y_test, test_idx = segs["test"]

            # Train
            model = train_xgb(X_train, y_train, X_valid, y_valid)
            pred = model.predict(xgb.DMatrix(X_test))

            # Evaluate
            metrics = evaluate_split(pred, y_test, test_idx)
            elapsed = time.time() - t0

            result = {
                "split": split_idx + 1,
                "test_start": test_start,
                "test_end": test_end,
                "train_start": train_start,
                "n_features": X_train.shape[1],
                "time_s": round(elapsed, 1),
                **metrics,
            }
            all_results.append(result)

            logger.info(f"  IC={metrics['ic_mean']:+.4f} ICIR={metrics['icir']:+.3f} "
                        f"RankIC={metrics['rank_ic_mean']:+.4f} "
                        f"Spread={metrics['top20_spread']*100:+.3f}% "
                        f"({metrics['regime_trend']}/{metrics['regime_vol']}) "
                        f"[{elapsed:.0f}s]")

        except Exception as e:
            logger.error(f"  Split {split_idx+1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        logger.error("No valid splits")
        sys.exit(1)

    # === Summary ===
    n = len(all_results)
    total_time = time.time() - t_total

    logger.info(f"\n{'='*90}")
    logger.info(f"PHASE 4 ROLLING GATE: {args.model} ({n} splits × {args.test_days} trading days)")
    logger.info(f"{'='*90}")
    logger.info(f"{'Split':<6} {'Test Period':<24} {'IC':>7} {'ICIR':>7} {'RankIC':>8} "
                f"{'Spread':>8} {'Regime':<16} {'Top20 Ret':>9}")
    logger.info("-" * 90)
    for r in all_results:
        logger.info(
            f"{r['split']:<6} {r['test_start']}~{r['test_end']}  "
            f"{r['ic_mean']:+.4f} {r['icir']:+.3f}  {r['rank_ic_mean']:+.4f}  "
            f"{r['top20_spread']*100:+.3f}%  "
            f"{r['regime_trend']}/{r['regime_vol']:<8} "
            f"{r['top20_avg_return']*100:+.3f}%"
        )

    # Gate check
    checks, all_pass = check_gate(all_results)

    logger.info(f"\n{'='*90}")
    logger.info("PROMOTION GATE CHECK")
    logger.info(f"{'='*90}")
    for name, (value, passed) in checks.items():
        threshold = GATE[name]
        if "pct" in name:
            logger.info(f"  {name:<20} {value:.1%} {'>=':>3} {threshold:.1%}  "
                        f"{'✅ PASS' if passed else '❌ FAIL'}")
        else:
            logger.info(f"  {name:<20} {value:+.4f} {'>=':>3} {threshold:+.4f}  "
                        f"{'✅ PASS' if passed else '❌ FAIL'}")

    logger.info(f"\n  Overall: {'✅ ALL GATES PASS' if all_pass else '❌ SOME GATES FAILED'}")
    logger.info(f"  Total time: {total_time/60:.1f} min")

    # Regime breakdown
    regimes = {}
    for r in all_results:
        key = f"{r['regime_trend']}/{r['regime_vol']}"
        regimes.setdefault(key, []).append(r)

    logger.info(f"\n{'='*90}")
    logger.info("REGIME BREAKDOWN")
    logger.info(f"{'='*90}")
    for regime, splits in sorted(regimes.items()):
        rics = [s["rank_ic_mean"] for s in splits]
        sprs = [s["top20_spread"] for s in splits]
        logger.info(f"  {regime:<18} n={len(splits):>2}  "
                    f"avg RankIC={np.mean(rics):+.4f}  "
                    f"avg Spread={np.mean(sprs)*100:+.3f}%")

    # Save
    summary = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "n_splits": n,
        "test_days": args.test_days,
        "valid_days": args.valid_days,
        "train_years": args.train_years,
        "total_time_min": round(total_time / 60, 1),
        "gate_thresholds": GATE,
        "gate_results": {name: {"value": round(float(v), 6), "pass": p}
                         for name, (v, p) in checks.items()},
        "gate_pass": all_pass,
        "aggregate": {
            "avg_ic": round(float(np.mean([r["ic_mean"] for r in all_results])), 6),
            "avg_rank_ic": round(float(np.mean([r["rank_ic_mean"] for r in all_results])), 6),
            "avg_spread": round(float(np.mean([r["top20_spread"] for r in all_results])), 6),
            "avg_icir": round(float(np.mean([r["icir"] for r in all_results])), 4),
        },
        "regime_breakdown": {
            regime: {
                "n_splits": len(splits),
                "avg_rank_ic": round(float(np.mean([s["rank_ic_mean"] for s in splits])), 6),
                "avg_spread": round(float(np.mean([s["top20_spread"] for s in splits])), 6),
            }
            for regime, splits in regimes.items()
        },
        "splits": all_results,
    }

    out_path = DATA_DIR / f"phase4_rolling_gate_{args.model}_{args.train_years}yr.json"
    with open(str(out_path), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved: {out_path}")

    # Push
    try:
        from push.wechat import WeChatPusher
        agg = summary["aggregate"]
        lines = [
            f"🏛️ Phase 4 Rolling Gate: {args.model}",
            f"{'✅ ALL PASS' if all_pass else '❌ FAILED'}",
            f"{n} splits × {args.test_days} trading days, train={args.train_years}yr",
            "",
            f"avg IC:     {agg['avg_ic']:+.4f}",
            f"avg RankIC: {agg['avg_rank_ic']:+.4f} (gate: ≥{GATE['avg_rank_ic']})",
            f"avg Spread: {agg['avg_spread']*100:+.3f}% (gate: ≥{GATE['avg_spread']*100:.1f}%)",
            f"RankIC>0:   {checks['rank_ic_pos_pct'][0]:.0%} (gate: ≥{GATE['rank_ic_pos_pct']:.0%})",
            f"Spread>0:   {checks['spread_pos_pct'][0]:.0%} (gate: ≥{GATE['spread_pos_pct']:.0%})",
        ]
        WeChatPusher().send("\n".join(lines), title=f"Phase4 Gate: {args.model}")
    except Exception as e:
        logger.warning(f"Push failed: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
