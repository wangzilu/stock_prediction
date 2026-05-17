"""Rolling ablation: 174-dim vs 175-dim (174+holder) across 12+ splits.

Tests whether holder_num is a stable alpha source, not single-window luck.
Phase 2 gate: ≥70% of splits must have RankIC>0 AND Spread>0.

Usage:
    python scripts/rolling_holder_ablation.py
    python scripts/rolling_holder_ablation.py --n-splits 16 --test-days 20
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
    """Prepare 174-dim: Alpha158(158) + flow(3) + custom(13)."""
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
                        start_time=str(min(dates))[:10],
                        end_time=str(max(dates))[:10])
    if custom is not None and not custom.empty:
        custom.columns = CUSTOM_NAMES
        custom = custom.swaplevel().sort_index().reindex(X.index)
        custom = custom.replace([np.inf, -np.inf], np.nan)
        new_cols = [c for c in custom.columns if c not in set(X.columns)]
        if new_cols:
            X = X.join(custom[new_cols], how="left")

    return X, y


def evaluate(pred, label, index):
    from qlib.contrib.eva.alpha import calc_ic
    mask = np.isfinite(pred) & np.isfinite(label)
    ps = pd.Series(pred[mask], index=index[mask])
    ls = pd.Series(label[mask], index=index[mask])
    ic, ric = calc_ic(ps, ls)
    spreads = []
    for _, g in pd.DataFrame({"pred": ps, "label": ls}).groupby(level=0):
        if len(g) < 40:
            continue
        s = g.sort_values("pred", ascending=False)
        spreads.append(s.head(20)["label"].mean() - s.tail(20)["label"].mean())
    return {
        "ic_mean": round(float(ic.mean()), 6),
        "icir": round(float(ic.mean()) / (float(ic.std()) + 1e-8), 4),
        "rank_ic_mean": round(float(ric.mean()), 6),
        "rank_ic_pos": round(float((ric > 0).mean()), 4),
        "top20_spread": round(float(np.mean(spreads)) if spreads else 0, 6),
        "spread_pos": round(float(np.mean([s > 0 for s in spreads])) if spreads else 0, 4),
    }


def train_xgb(X_train, y_train, X_valid, y_valid):
    import xgboost as xgb
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_valid, label=y_valid)
    params = {
        "max_depth": 8, "learning_rate": 0.05,
        "subsample": 0.8789, "colsample_bytree": 0.8879,
        "reg_alpha": 205.6999, "reg_lambda": 580.9768,
        "objective": "reg:squarederror", "nthread": 4,
        "verbosity": 0, "seed": SEED,
    }
    model = xgb.train(params, dt, num_boost_round=500,
                      evals=[(dv, "valid")], early_stopping_rounds=50, verbose_eval=0)
    return model


def main():
    import xgboost as xgb

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=12)
    parser.add_argument("--test-days", type=int, default=20)
    parser.add_argument("--train-years", type=int, default=3)
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

        logger.info(f"\n{'='*60}")
        logger.info(f"Split {split_idx+1}/{args.n_splits}: test {dates['test'][0]}~{dates['test'][1]}")

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

            # Prepare 174-dim and 175-dim for all segments
            base = {}
            segs_175 = {}
            for seg in ["train", "valid", "test"]:
                X, y = prepare_174(dataset, seg, merger)
                yn = y.values.astype(np.float32)
                mask = np.isfinite(yn)

                # 174-dim
                Xn = X.values.astype(np.float32)
                base[seg] = (Xn[mask], yn[mask], X.index[mask])

                # 175-dim: join holder before filtering
                h = merger._load_st_holder_number(X.index)
                if h is not None:
                    X175 = X.join(h, how="left")
                else:
                    X175 = X
                X175n = X175.values.astype(np.float32)
                segs_175[seg] = (X175n[mask], yn[mask], X.index[mask])

            # --- Experiment A: 174-dim base ---
            t0 = time.time()
            model_174 = train_xgb(base["train"][0], base["train"][1],
                                  base["valid"][0], base["valid"][1])
            pred_174 = model_174.predict(xgb.DMatrix(base["test"][0]))
            m174 = evaluate(pred_174, base["test"][1], base["test"][2])
            t174 = time.time() - t0

            t0 = time.time()
            model_175 = train_xgb(segs_175["train"][0], segs_175["train"][1],
                                  segs_175["valid"][0], segs_175["valid"][1])
            pred_175 = model_175.predict(xgb.DMatrix(segs_175["test"][0]))
            m175 = evaluate(pred_175, segs_175["test"][1], segs_175["test"][2])
            t175 = time.time() - t0

            n174 = base["train"][0].shape[1]
            n175 = segs_175["train"][0].shape[1]

            split_result = {
                "split": split_idx + 1,
                "test_period": f"{dates['test'][0]}~{dates['test'][1]}",
                "174": {"n_feat": n174, **m174, "time_s": round(t174, 1)},
                "175": {"n_feat": n175, **m175, "time_s": round(t175, 1)},
                "delta_rank_ic": round(m175["rank_ic_mean"] - m174["rank_ic_mean"], 6),
                "delta_spread": round(m175["top20_spread"] - m174["top20_spread"], 6),
            }
            all_results.append(split_result)

            logger.info(f"  174: IC={m174['ic_mean']:+.4f} RankIC={m174['rank_ic_mean']:+.4f} "
                        f"Spread={m174['top20_spread']*100:+.3f}%")
            logger.info(f"  175: IC={m175['ic_mean']:+.4f} RankIC={m175['rank_ic_mean']:+.4f} "
                        f"Spread={m175['top20_spread']*100:+.3f}%")
            logger.info(f"  Δ RankIC={split_result['delta_rank_ic']:+.4f} "
                        f"Δ Spread={split_result['delta_spread']*100:+.3f}%")

        except Exception as e:
            logger.error(f"  Split {split_idx+1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        logger.error("No valid splits completed")
        sys.exit(1)

    # ============ Summary ============
    ric_174 = [r["174"]["rank_ic_mean"] for r in all_results]
    ric_175 = [r["175"]["rank_ic_mean"] for r in all_results]
    spr_174 = [r["174"]["top20_spread"] for r in all_results]
    spr_175 = [r["175"]["top20_spread"] for r in all_results]
    delta_ric = [r["delta_rank_ic"] for r in all_results]
    delta_spr = [r["delta_spread"] for r in all_results]

    n = len(all_results)
    ric175_pos = sum(1 for r in ric_175 if r > 0) / n
    spr175_pos = sum(1 for s in spr_175 if s > 0) / n
    delta_ric_pos = sum(1 for d in delta_ric if d > 0) / n
    delta_spr_pos = sum(1 for d in delta_spr if d > 0) / n

    # Phase 2 gate
    pass_ric = ric175_pos >= 0.70
    pass_spr = spr175_pos >= 0.70
    pass_gate = pass_ric and pass_spr

    logger.info(f"\n{'='*80}")
    logger.info(f"ROLLING ABLATION: 174 vs 175 (174+holder)")
    logger.info(f"{'='*80}")
    logger.info(f"{'Split':<8} {'Test Period':<26} {'174 RankIC':>10} {'175 RankIC':>10} "
                f"{'Δ RankIC':>10} {'174 Spread':>10} {'175 Spread':>10} {'Δ Spread':>10}")
    logger.info("-" * 95)

    for r in all_results:
        logger.info(
            f"{r['split']:<8} {r['test_period']:<26} "
            f"{r['174']['rank_ic_mean']:+.4f}     {r['175']['rank_ic_mean']:+.4f}     "
            f"{r['delta_rank_ic']:+.4f}     "
            f"{r['174']['top20_spread']*100:+.3f}%    {r['175']['top20_spread']*100:+.3f}%    "
            f"{r['delta_spread']*100:+.3f}%"
        )

    logger.info(f"\n{'='*80}")
    logger.info("AGGREGATE")
    logger.info(f"{'='*80}")
    logger.info(f"  174 avg RankIC: {np.mean(ric_174):+.4f}  175 avg RankIC: {np.mean(ric_175):+.4f}")
    logger.info(f"  174 avg Spread: {np.mean(spr_174)*100:+.3f}%  175 avg Spread: {np.mean(spr_175)*100:+.3f}%")
    logger.info(f"  175 RankIC>0: {ric175_pos:.0%} (gate: ≥70%)  {'PASS' if pass_ric else 'FAIL'}")
    logger.info(f"  175 Spread>0: {spr175_pos:.0%} (gate: ≥70%)  {'PASS' if pass_spr else 'FAIL'}")
    logger.info(f"  Δ RankIC>0:   {delta_ric_pos:.0%} of splits  (holder helps)")
    logger.info(f"  Δ Spread>0:   {delta_spr_pos:.0%} of splits  (holder helps)")
    logger.info(f"\n  Phase 2 Gate: {'✅ PASS' if pass_gate else '❌ FAIL'}")

    # Save
    summary = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "n_splits": n,
        "test_days": args.test_days,
        "train_years": args.train_years,
        "aggregate": {
            "174_avg_rank_ic": round(float(np.mean(ric_174)), 6),
            "175_avg_rank_ic": round(float(np.mean(ric_175)), 6),
            "174_avg_spread": round(float(np.mean(spr_174)), 6),
            "175_avg_spread": round(float(np.mean(spr_175)), 6),
            "175_rank_ic_pos_pct": round(ric175_pos, 4),
            "175_spread_pos_pct": round(spr175_pos, 4),
            "delta_rank_ic_pos_pct": round(delta_ric_pos, 4),
            "delta_spread_pos_pct": round(delta_spr_pos, 4),
            "phase2_gate_pass": pass_gate,
        },
        "splits": all_results,
    }

    out_path = DATA_DIR / "rolling_holder_ablation.json"
    with open(str(out_path), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved: {out_path}")

    # Push
    try:
        from push.wechat import WeChatPusher
        lines = [
            f"🔬 Rolling Holder Ablation ({n} splits)",
            f"Gate: {'✅ PASS' if pass_gate else '❌ FAIL'}",
            "",
            f"174 avg RankIC: {np.mean(ric_174):+.4f}  Spread: {np.mean(spr_174)*100:+.3f}%",
            f"175 avg RankIC: {np.mean(ric_175):+.4f}  Spread: {np.mean(spr_175)*100:+.3f}%",
            f"175 RankIC>0: {ric175_pos:.0%}  Spread>0: {spr175_pos:.0%}",
            f"Holder helps: Δ RankIC>0 in {delta_ric_pos:.0%}, Δ Spread>0 in {delta_spr_pos:.0%}",
        ]
        WeChatPusher().send("\n".join(lines), title="Rolling Holder Ablation")
        logger.info("✅ 推送成功")
    except Exception as e:
        logger.warning(f"Push failed: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
