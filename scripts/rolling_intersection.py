"""Rolling validation of intersection strategy: XGB Top50 ∩ Ranker Top50.

Tests whether the intersection approach is stable across 12+ time windows.

Usage:
    python scripts/rolling_intersection.py
    python scripts/rolling_intersection.py --n-splits 16 --topn 50
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


def prepare_174(dataset, seg, merger):
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


def train_ranker(X_train, y_train, X_valid, y_valid, train_idx, valid_idx):
    import lightgbm as lgb
    s = pd.Series(y_train, index=train_idx)
    y_train_int = s.groupby(level=0).transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")).fillna(0).astype(np.int32).values
    s2 = pd.Series(y_valid, index=valid_idx)
    y_valid_int = s2.groupby(level=0).transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")).fillna(0).astype(np.int32).values

    train_groups = train_idx.get_level_values(0).value_counts().sort_index().values.tolist()
    valid_groups = valid_idx.get_level_values(0).value_counts().sort_index().values.tolist()
    dtrain = lgb.Dataset(X_train, label=y_train_int, group=train_groups)
    dvalid = lgb.Dataset(X_valid, label=y_valid_int, group=valid_groups, reference=dtrain)
    params = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [20, 50],
              "num_leaves": 128, "learning_rate": 0.05, "subsample": 0.85,
              "colsample_bytree": 0.85, "reg_alpha": 200, "reg_lambda": 500,
              "verbose": -1, "seed": SEED}
    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dvalid], callbacks=[lgb.early_stopping(50)])
    return model


def compute_spreads(df, topN):
    """Compute daily spreads for XGB, Ranker, and Intersection strategies."""
    spreads_xgb = []
    spreads_ranker = []
    spreads_inter = []
    n_picks_per_day = []

    for _, g in df.groupby(level=0):
        if len(g) < 40:
            continue

        # XGB Top20
        s = g.sort_values("xgb", ascending=False)
        spr_xgb = s.head(20)["label"].mean() - s.tail(20)["label"].mean()
        spreads_xgb.append(spr_xgb)

        # Ranker Top20
        s = g.sort_values("ranker", ascending=False)
        spr_ranker = s.head(20)["label"].mean() - s.tail(20)["label"].mean()
        spreads_ranker.append(spr_ranker)

        # Intersection
        xgb_top = set(g.nlargest(topN, "xgb").index)
        ranker_top = set(g.nlargest(topN, "ranker").index)
        inter = list(xgb_top & ranker_top)

        if len(inter) >= 3:
            sub = g.loc[inter]
            bot20 = g.nsmallest(20, "xgb")
            spr_inter = sub["label"].mean() - bot20["label"].mean()
            spreads_inter.append(spr_inter)
            n_picks_per_day.append(len(inter))
        else:
            spreads_inter.append(np.nan)
            n_picks_per_day.append(0)

    return spreads_xgb, spreads_ranker, spreads_inter, n_picks_per_day


def main():
    import xgboost as xgb

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=12)
    parser.add_argument("--test-days", type=int, default=20)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--topn", type=int, default=50)
    args = parser.parse_args()

    from qlib.utils import init_instance_by_config
    init_qlib(QLIB_DATA)
    merger = FeatureMerger(DATA_DIR)

    today = datetime.now()
    all_results = []

    for split_idx in range(args.n_splits):
        test_end = today - timedelta(days=split_idx * args.test_days)
        test_start = test_end - timedelta(days=args.test_days)
        valid_end = test_start - timedelta(days=1)
        valid_start = valid_end - timedelta(days=60)
        train_end = valid_start - timedelta(days=1)
        train_start = train_end - timedelta(days=365 * args.train_years)

        dates = {
            "train": (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
            "valid": (valid_start.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")),
            "test": (test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")),
        }

        logger.info(f"\nSplit {split_idx+1}/{args.n_splits}: test {dates['test'][0]}~{dates['test'][1]}")

        try:
            dataset = init_instance_by_config({
                "class": "DatasetH", "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
                        "kwargs": {"start_time": dates["train"][0], "end_time": dates["test"][1],
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
                Xn = X.values.astype(np.float32)
                yn = y.values.astype(np.float32)
                mask = np.isfinite(yn)
                segs[seg] = (Xn[mask], yn[mask], X.index[mask])

            X_train, y_train, train_idx = segs["train"]
            X_valid, y_valid, valid_idx = segs["valid"]
            X_test, y_test, test_idx = segs["test"]

            # Train both models
            t0 = time.time()
            model_xgb = train_xgb(X_train, y_train, X_valid, y_valid)
            model_ranker = train_ranker(X_train, y_train, X_valid, y_valid, train_idx, valid_idx)
            train_time = time.time() - t0

            # Predict on test
            pred_xgb = model_xgb.predict(xgb.DMatrix(X_test))
            pred_ranker = model_ranker.predict(X_test)

            df = pd.DataFrame({"xgb": pred_xgb, "ranker": pred_ranker, "label": y_test},
                              index=test_idx)

            # Compute spreads
            spr_xgb, spr_ranker, spr_inter, n_picks = compute_spreads(df, args.topn)

            # Filter out NaN from intersection
            spr_inter_clean = [s for s in spr_inter if not np.isnan(s)]

            result = {
                "split": split_idx + 1,
                "test_period": f"{dates['test'][0]}~{dates['test'][1]}",
                "xgb_spread": round(float(np.mean(spr_xgb)) * 100, 3) if spr_xgb else 0,
                "ranker_spread": round(float(np.mean(spr_ranker)) * 100, 3) if spr_ranker else 0,
                "inter_spread": round(float(np.mean(spr_inter_clean)) * 100, 3) if spr_inter_clean else 0,
                "inter_pos_pct": round(float(np.mean([s > 0 for s in spr_inter_clean])), 3) if spr_inter_clean else 0,
                "avg_picks": round(float(np.mean([n for n in n_picks if n > 0])), 1) if any(n > 0 for n in n_picks) else 0,
                "n_days": len(spr_xgb),
                "train_time_s": round(train_time, 1),
            }
            all_results.append(result)

            logger.info(f"  XGB Spread:   {result['xgb_spread']:+.3f}%")
            logger.info(f"  Ranker Spread: {result['ranker_spread']:+.3f}%")
            logger.info(f"  Inter Spread:  {result['inter_spread']:+.3f}% "
                        f"(>0: {result['inter_pos_pct']:.0%}, avg {result['avg_picks']:.0f} picks/day)")

        except Exception as e:
            logger.error(f"  Split {split_idx+1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        logger.error("No valid splits")
        sys.exit(1)

    # Summary
    n = len(all_results)
    xgb_spreads = [r["xgb_spread"] for r in all_results]
    ranker_spreads = [r["ranker_spread"] for r in all_results]
    inter_spreads = [r["inter_spread"] for r in all_results]

    inter_pos = sum(1 for s in inter_spreads if s > 0) / n
    xgb_pos = sum(1 for s in xgb_spreads if s > 0) / n
    ranker_pos = sum(1 for s in ranker_spreads if s > 0) / n
    inter_beats_xgb = sum(1 for i, x in zip(inter_spreads, xgb_spreads) if i > x) / n
    inter_beats_ranker = sum(1 for i, r in zip(inter_spreads, ranker_spreads) if i > r) / n

    logger.info(f"\n{'='*80}")
    logger.info(f"ROLLING INTERSECTION (Top{args.topn}): {n} splits, {args.test_days} days each")
    logger.info(f"{'='*80}")
    logger.info(f"{'Split':<6} {'Test Period':<24} {'XGB':>8} {'Ranker':>8} {'Inter':>8} {'Inter>0':>8} {'Picks':>6}")
    logger.info("-" * 75)
    for r in all_results:
        logger.info(f"{r['split']:<6} {r['test_period']:<24} "
                    f"{r['xgb_spread']:+.3f}%  {r['ranker_spread']:+.3f}%  "
                    f"{r['inter_spread']:+.3f}%  {r['inter_pos_pct']:.0%}      "
                    f"{r['avg_picks']:.0f}")

    logger.info(f"\n{'='*80}")
    logger.info("AGGREGATE")
    logger.info(f"{'='*80}")
    logger.info(f"  XGB avg Spread:    {np.mean(xgb_spreads):+.3f}% (>0 in {xgb_pos:.0%} splits)")
    logger.info(f"  Ranker avg Spread: {np.mean(ranker_spreads):+.3f}% (>0 in {ranker_pos:.0%} splits)")
    logger.info(f"  Inter avg Spread:  {np.mean(inter_spreads):+.3f}% (>0 in {inter_pos:.0%} splits)")
    logger.info(f"  Inter beats XGB:    {inter_beats_xgb:.0%} of splits")
    logger.info(f"  Inter beats Ranker: {inter_beats_ranker:.0%} of splits")
    logger.info(f"  Avg picks/day:      {np.mean([r['avg_picks'] for r in all_results]):.1f}")

    gate_pass = inter_pos >= 0.70
    logger.info(f"\n  Gate (Inter Spread>0 ≥70%): {'✅ PASS' if gate_pass else '❌ FAIL'} ({inter_pos:.0%})")

    # Save
    summary = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "n_splits": n,
        "test_days": args.test_days,
        "topN": args.topn,
        "aggregate": {
            "xgb_avg_spread": round(float(np.mean(xgb_spreads)), 4),
            "ranker_avg_spread": round(float(np.mean(ranker_spreads)), 4),
            "inter_avg_spread": round(float(np.mean(inter_spreads)), 4),
            "inter_spread_pos_pct": round(inter_pos, 4),
            "inter_beats_xgb_pct": round(inter_beats_xgb, 4),
            "inter_beats_ranker_pct": round(inter_beats_ranker, 4),
            "gate_pass": gate_pass,
        },
        "splits": all_results,
    }

    out_path = DATA_DIR / "rolling_intersection_results.json"
    with open(str(out_path), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved: {out_path}")

    # Push
    try:
        from push.wechat import WeChatPusher
        lines = [
            f"🔬 Rolling Intersection Top{args.topn} ({n} splits)",
            f"Gate: {'✅ PASS' if gate_pass else '❌ FAIL'}",
            "",
            f"XGB avg:    {np.mean(xgb_spreads):+.3f}% (>0: {xgb_pos:.0%})",
            f"Ranker avg: {np.mean(ranker_spreads):+.3f}% (>0: {ranker_pos:.0%})",
            f"Inter avg:  {np.mean(inter_spreads):+.3f}% (>0: {inter_pos:.0%})",
            f"Inter beats XGB: {inter_beats_xgb:.0%}, Ranker: {inter_beats_ranker:.0%}",
            f"Avg picks/day: {np.mean([r['avg_picks'] for r in all_results]):.1f}",
        ]
        WeChatPusher().send("\n".join(lines), title="Rolling Intersection")
    except Exception as e:
        logger.warning(f"Push failed: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
