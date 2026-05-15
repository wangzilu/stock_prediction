"""Factor group ablation: test each supplementary factor group's marginal contribution.

Runs against Alpha158 baseline with the best model (XGB, raw preprocessing).
Includes shuffled negative control for each group.

Groups:
  1. valuation:  PE/PB/PS/EP/BP/SP/PCF (7 dims)
  2. flow:       net_mf_latest/5d/20d_avg (3 dims)
  3. macro:      bond/rate/commodity/PMI/CPI (10 dims)
  4. shareholder: total_share/liquid_share/liquid_ratio (3 dims)
  5. northbound:  hold_change/ratio (4 dims)
  6. quality:     ROE/margin/growth (8 dims)
  7. qlib_custom: PE/PB/Turn expressions (13 dims)

Outputs: data/storage/factor_ablation_v2.json

Usage:
    python scripts/train_factor_ablation_v2.py
"""
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
OUTPUT_PATH = DATA_DIR / "factor_ablation_v2.json"

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


def evaluate(pred, label, index):
    from qlib.contrib.eva.alpha import calc_ic
    mask = np.isfinite(pred) & np.isfinite(label)
    pred_s = pd.Series(pred[mask], index=index[mask], name="score")
    label_s = pd.Series(label[mask], index=index[mask], name="label")
    ic, ric = calc_ic(pred_s, label_s)
    df = pd.DataFrame({"pred": pred_s, "label": label_s})
    spreads = []
    for d, g in df.groupby(level=0):
        if len(g) < 40:
            continue
        s = g.sort_values("pred", ascending=False)
        spreads.append(s.head(20)["label"].mean() - s.tail(20)["label"].mean())
    return {
        "ic_mean": round(float(ic.mean()), 6),
        "icir": round(float(ic.mean()) / (float(ic.std()) + 1e-8), 4),
        "rank_ic_mean": round(float(ric.mean()), 6),
        "rank_ic_pos_ratio": round(float((ric > 0).mean()), 4),
        "top20_spread": round(float(np.mean(spreads)) if spreads else 0, 6),
        "spread_pos_ratio": round(float(np.mean([s > 0 for s in spreads])) if spreads else 0, 4),
    }


def train_xgb(X_train, y_train, X_valid, y_valid):
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    params = {
        "max_depth": 8, "learning_rate": 0.05,
        "subsample": 0.8789, "colsample_bytree": 0.8879,
        "reg_alpha": 205.6999, "reg_lambda": 580.9768,
        "objective": "reg:squarederror", "nthread": 4,
        "verbosity": 0, "seed": SEED,
    }
    model = xgb.train(params, dtrain, num_boost_round=500,
                      evals=[(dvalid, "valid")],
                      early_stopping_rounds=50, verbose_eval=0)
    return model


def predict_xgb(model, X):
    import xgboost as xgb
    return model.predict(xgb.DMatrix(X))


def shuffle_within_date(df, index):
    """Shuffle factor values within each date, preserving daily distribution.
    This is the correct negative control: breaks stock-factor mapping but keeps
    the same cross-sectional distribution per day."""
    result = df.copy()
    rng = np.random.RandomState(SEED)
    dates = index.get_level_values(0)
    for date in dates.unique():
        mask = dates == date
        n = mask.sum()
        for col in result.columns:
            vals = result.loc[mask, col].values.copy()
            rng.shuffle(vals)
            result.loc[mask, col] = vals
    return result


def main():
    from qlib.utils import init_instance_by_config
    from qlib.data import D

    init_qlib(QLIB_DATA)

    today = datetime.now()
    train_start = (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=89)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")

    logger.info("Loading Alpha158 dataset...")
    dataset = init_instance_by_config({
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
                "kwargs": {"start_time": train_start, "end_time": test_end,
                           "instruments": "all", "label": [LABEL_EXPR]},
            },
            "segments": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (test_start, test_end),
            },
        },
    })

    # Prepare base Alpha158 features
    logger.info("Preparing base features...")
    segs_base = {}
    for seg in ["train", "valid", "test"]:
        X = dataset.prepare(seg, col_set="feature")
        y = dataset.prepare(seg, col_set="label")
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        segs_base[seg] = (X, y)

    # Load all supplementary factor groups separately
    logger.info("Loading supplementary factor groups...")
    merger = FeatureMerger(DATA_DIR)
    train_index = segs_base["train"][0].index

    groups = {}

    # Each group loaded independently
    g = merger._load_capital_flow(train_index)
    if g is not None and not g.empty:
        groups["flow"] = g.columns.tolist()
    g = merger._load_valuation(train_index)
    if g is not None and not g.empty:
        groups["valuation"] = g.columns.tolist()
    g = merger._load_macro(train_index)
    if g is not None and not g.empty:
        groups["macro"] = g.columns.tolist()
    g = merger._load_shareholder(train_index)
    if g is not None and not g.empty:
        groups["shareholder"] = g.columns.tolist()
    g = merger._load_northbound(train_index)
    if g is not None and not g.empty:
        groups["northbound"] = g.columns.tolist()
    g = merger._load_quality(train_index)
    if g is not None and not g.empty:
        groups["quality"] = g.columns.tolist()

    logger.info(f"Factor groups: {', '.join(f'{k}({len(v)})' for k,v in groups.items())}")

    # Load all supplementary at once for column reference
    all_supp = {}
    for seg in ["train", "valid", "test"]:
        idx = segs_base[seg][0].index
        supp = merger._load_supplementary(idx)
        all_supp[seg] = supp

    # Also load Qlib custom factors
    logger.info("Loading Qlib custom factors...")
    qlib_custom_segs = {}
    for seg in ["train", "valid", "test"]:
        X = segs_base[seg][0]
        instruments = list(set(str(c) for c in X.index.get_level_values(1)))
        dates = sorted(X.index.get_level_values(0).unique())
        custom = D.features(instruments, CUSTOM_EXPRS,
                            start_time=str(min(dates))[:10],
                            end_time=str(max(dates))[:10])
        if custom is not None and not custom.empty:
            custom.columns = CUSTOM_NAMES
            custom = custom.swaplevel().sort_index().reindex(X.index)
            custom = custom.replace([np.inf, -np.inf], np.nan)
            qlib_custom_segs[seg] = custom
        else:
            qlib_custom_segs[seg] = None
    groups["qlib_custom"] = CUSTOM_NAMES

    # Build experiment list
    experiments = []

    # 1. Base: Alpha158 only
    experiments.append({"name": "base_158", "add_groups": [], "shuffle": False})

    # 2. Base + all supplementary (202 dims)
    experiments.append({"name": "base_all_202", "add_groups": list(groups.keys()), "shuffle": False})

    # 3. Base + each group individually
    for gname in groups:
        experiments.append({"name": f"base+{gname}", "add_groups": [gname], "shuffle": False})
        experiments.append({"name": f"base+{gname}_SHUFFLED", "add_groups": [gname], "shuffle": True})

    logger.info(f"\nRunning {len(experiments)} experiments...")
    results = []

    for exp in experiments:
        name = exp["name"]
        logger.info(f"\n{'='*50}")
        logger.info(f"Experiment: {name}")
        logger.info(f"{'='*50}")

        t0 = time.time()
        exp_segs = {}

        for seg in ["train", "valid", "test"]:
            X_base = segs_base[seg][0].copy()
            y = segs_base[seg][1]

            # Add requested factor groups
            for gname in exp["add_groups"]:
                if gname == "qlib_custom":
                    custom = qlib_custom_segs[seg]
                    if custom is not None:
                        existing = set(X_base.columns)
                        new_cols = [c for c in custom.columns if c not in existing]
                        if new_cols:
                            to_add = custom[new_cols]
                            if exp["shuffle"] and seg == "train":
                                to_add = shuffle_within_date(to_add, X_base.index)
                            X_base = X_base.join(to_add, how="left")
                else:
                    supp = all_supp[seg]
                    if supp is not None:
                        group_cols = [c for c in groups.get(gname, []) if c in supp.columns]
                        if group_cols:
                            to_add = supp[group_cols]
                            if exp["shuffle"] and seg == "train":
                                to_add = shuffle_within_date(to_add, X_base.index)
                            X_base = X_base.join(to_add, how="left")

            X_np = X_base.values.astype(np.float32)
            y_np = y.values.astype(np.float32)
            mask = np.isfinite(y_np)
            exp_segs[seg] = (X_np[mask], y_np[mask], X_base.index[mask])

        n_feat = exp_segs["train"][0].shape[1]
        logger.info(f"  Features: {n_feat}")

        # Train XGB
        model = train_xgb(exp_segs["train"][0], exp_segs["train"][1],
                          exp_segs["valid"][0], exp_segs["valid"][1])
        pred = predict_xgb(model, exp_segs["test"][0])
        metrics = evaluate(pred, exp_segs["test"][1], exp_segs["test"][2])

        result = {
            "name": name,
            "n_features": n_feat,
            "groups": exp["add_groups"],
            "shuffled": exp["shuffle"],
            **metrics,
            "time_s": round(time.time() - t0, 1),
        }
        results.append(result)

        logger.info(f"  IC:     {metrics['ic_mean']:+.4f}  ICIR: {metrics['icir']:+.3f}")
        logger.info(f"  RankIC: {metrics['rank_ic_mean']:+.4f}  RIC>0: {metrics['rank_ic_pos_ratio']:.0%}")
        logger.info(f"  Spread: {metrics['top20_spread']*100:+.3f}%  Sprd>0: {metrics['spread_pos_ratio']:.0%}")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("ABLATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Name':<30} {'Feat':>5} {'IC':>8} {'ICIR':>7} {'RankIC':>8} {'Spread':>9} {'RIC>0':>6}")
    logger.info("-" * 75)

    base_ric = None
    for r in results:
        if r["name"] == "base_158":
            base_ric = r["rank_ic_mean"]

    for r in results:
        delta = ""
        if base_ric is not None and r["name"] != "base_158":
            d = r["rank_ic_mean"] - base_ric
            delta = f" ({d:+.4f})"
        shuf = " ⚠NEG" if r["shuffled"] else ""
        logger.info(
            f"{r['name']:<30} {r['n_features']:>5} "
            f"{r['ic_mean']:+.4f}  {r['icir']:+.3f}  "
            f"{r['rank_ic_mean']:+.4f}{delta}  "
            f"{r['top20_spread']*100:+.3f}%  {r['rank_ic_pos_ratio']:.0%}{shuf}"
        )

    # Save
    with open(str(OUTPUT_PATH), "w") as f:
        json.dump({
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "baseline_rank_ic": base_ric,
            "results": results,
        }, f, indent=2)
    logger.info(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
