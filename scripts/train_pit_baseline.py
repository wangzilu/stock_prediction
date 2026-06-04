"""PIT-safe baseline comparison: multiple models × feature sets.

Supports: LGB, XGB, CatBoost, LGBRanker
Dimensions: 158 (Alpha158 only), 174 (+ flow + qlib_custom)

Usage:
    python scripts/train_pit_baseline.py
    python scripts/train_pit_baseline.py --models lgb,xgb,lgb_ranker --dims 174
    python scripts/train_pit_baseline.py --models lgb_ranker --dims 174
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
OUTPUT_PATH = DATA_DIR / "pit_baseline_comparison.json"

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


def load_dataset():
    from qlib.utils import init_instance_by_config
    today = datetime.now()
    train_start = (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=89)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")
    dates = {"train": (train_start, train_end), "valid": (valid_start, valid_end),
             "test": (test_start, test_end)}
    dataset = init_instance_by_config({
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {"class": "Alpha158", "module_path": "qlib.contrib.data.handler",
                        "kwargs": {"start_time": train_start, "end_time": test_end,
                                   "instruments": "all", "label": [LABEL_EXPR]}},
            "segments": dates}})
    return dataset, dates


def prepare_features(dataset, segment, dim_mode, preprocess="raw"):
    from qlib.data import D
    X = dataset.prepare(segment, col_set="feature")
    y = dataset.prepare(segment, col_set="label")
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    if dim_mode in ("174", "175"):
        merger = FeatureMerger(DATA_DIR)
        flow = merger._load_capital_flow(X.index)
        if flow is not None:
            X = X.join(flow, how="left")

        instruments = list(set(str(c) for c in X.index.get_level_values(1)))
        dates = sorted(X.index.get_level_values(0).unique())
        custom = D.features(instruments, CUSTOM_EXPRS,
                            start_time=str(min(dates))[:10], end_time=str(max(dates))[:10])
        if custom is not None and not custom.empty:
            custom.columns = CUSTOM_NAMES
            custom = custom.swaplevel().sort_index().reindex(X.index)
            custom = custom.replace([np.inf, -np.inf], np.nan)
            new_cols = [c for c in custom.columns if c not in set(X.columns)]
            if new_cols:
                X = X.join(custom[new_cols], how="left")

        if dim_mode == "175":
            holder = merger._load_st_holder_number(X.index)
            if holder is not None:
                X = X.join(holder, how="left")

    elif dim_mode == "202":
        merger = FeatureMerger(DATA_DIR)
        # research/ablation script — explicit opt-in to "load every loader"
        # (P0-e: production must use PRODUCTION_SUPPLEMENTARY_GROUPS instead)
        from config.production_features import RESEARCH_ALL_LOADERS
        supp = merger._load_supplementary(X.index,
                                          groups=RESEARCH_ALL_LOADERS)
        if supp is not None and not supp.empty:
            X = X.join(supp, how="left")
        instruments = list(set(str(c) for c in X.index.get_level_values(1)))
        dates = sorted(X.index.get_level_values(0).unique())
        custom = D.features(instruments, CUSTOM_EXPRS,
                            start_time=str(min(dates))[:10], end_time=str(max(dates))[:10])
        if custom is not None and not custom.empty:
            custom.columns = CUSTOM_NAMES
            custom = custom.swaplevel().sort_index().reindex(X.index)
            custom = custom.replace([np.inf, -np.inf], np.nan)
            new_cols = [c for c in custom.columns if c not in set(X.columns)]
            if new_cols:
                X = X.join(custom[new_cols], how="left")

    elif dim_mode == "production_242":
        # Mirrors the live champion EXACTLY: Alpha158 + the 11 groups in
        # PRODUCTION_SUPPLEMENTARY_GROUPS, no qlib_custom extras. This is
        # the contract train_lgb.py ships today (post-P0-c). Side-by-side
        # vs dim=158 answers "is the 84-col supplementary block actually
        # earning its keep on hold-out?" (task #102 / 174-vs-242 ask).
        merger = FeatureMerger(DATA_DIR)
        from config.production_features import (
            PRODUCTION_SUPPLEMENTARY_GROUPS,
        )
        supp = merger._load_supplementary(
            X.index, groups=PRODUCTION_SUPPLEMENTARY_GROUPS,
        )
        if supp is not None and not supp.empty:
            supp = supp.replace([np.inf, -np.inf], np.nan)
            X = X.join(supp, how="left")

    feature_names = list(X.columns)
    X_np = X.values.astype(np.float32)
    y_np = y.values.astype(np.float32)
    mask = np.isfinite(y_np)
    return X_np[mask], y_np[mask], X.index[mask], feature_names


def evaluate(pred, label, index):
    from qlib.contrib.eva.alpha import calc_ic
    mask = np.isfinite(pred) & np.isfinite(label)
    ps = pd.Series(pred[mask], index=index[mask], name="score")
    ls = pd.Series(label[mask], index=index[mask], name="label")
    ic, ric = calc_ic(ps, ls)
    df = pd.DataFrame({"pred": ps, "label": ls})
    spreads = []
    for d, g in df.groupby(level=0):
        if len(g) < 40:
            continue
        s = g.sort_values("pred", ascending=False)
        spreads.append(s.head(20)["label"].mean() - s.tail(20)["label"].mean())

    # 2026-06-04 cx round 12 P1-1: emit BOTH label-spread (default,
    # may be CSZScoreNorm-processed) and raw-forward-return spread when
    # the label is normalised. Without a raw_return reference here we
    # can only tag the unit; downstream backtest scripts that load this
    # JSON must check ``label_unit`` before reading top20_spread as a
    # percentage. The "+%.3f%%" printout in main() also gets a unit
    # disclaimer.
    label_unit = "alpha_qlib_csz_norm"
    # ``label_unit`` is "alpha_qlib_csz_norm" because the dataset
    # config uses Qlib's CSZScoreNorm pre-processor — the spread
    # is on standardised label, NOT raw 5-day return. To get raw
    # return spread, the caller must re-evaluate with a label that
    # bypasses CSZScoreNorm (planned: train_pit_baseline_raw.py
    # — task #102 follow-up).
    # cx round 12 P1-3: embed execution_schema so the reader can
    # verify the artifact came from the same execution rules as any
    # other artifact being compared.
    from config.experiment import EXECUTION_SCHEMA
    return {
        "ic_mean": round(float(ic.mean()), 6),
        "ic_std": round(float(ic.std()), 6),
        "icir": round(float(ic.mean()) / (float(ic.std()) + 1e-8), 4),
        "rank_ic_mean": round(float(ric.mean()), 6),
        "rank_ic_pos_ratio": round(float((ric > 0).mean()), 4),
        # cx round 11 P2-5: rename to make the unit obvious. Old key
        # ``top20_spread`` kept as alias for back-compat with existing
        # consumers reading the JSON.
        "top20_label_spread": round(float(np.mean(spreads)) if spreads else 0, 6),
        "top20_spread": round(float(np.mean(spreads)) if spreads else 0, 6),
        "label_unit": label_unit,
        "top20_raw_return_spread": None,  # populated when raw-return label is wired
        "spread_pos_ratio": round(float(np.mean([s > 0 for s in spreads])) if spreads else 0, 4),
        "n_test_days": len(spreads),
        "n_samples": int(mask.sum()),
        "execution_schema": dict(EXECUTION_SCHEMA),
    }


def train_lgb(X_train, y_train, X_valid, y_valid):
    import lightgbm as lgb
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
    params = {"objective": "regression", "metric": "mse", "num_leaves": 128,
              "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85,
              "reg_alpha": 200, "reg_lambda": 500, "verbose": -1, "seed": SEED}
    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dvalid], callbacks=[lgb.early_stopping(50)])
    return model


def _returns_to_rank_label(y, index, n_bins=5):
    """Discretize continuous returns to int relevance grades (0..n_bins-1) per day.

    LightGBM lambdarank requires integer labels.  We bin returns into
    per-day quantile buckets so that the ranking target is cross-sectionally
    comparable across different dates.
    """
    s = pd.Series(y, index=index)
    labels = s.groupby(level=0).transform(
        lambda x: pd.qcut(x, n_bins, labels=False, duplicates="drop")
    )
    return labels.fillna(0).astype(np.int32).values


def train_lgb_ranker(X_train, y_train, X_valid, y_valid, train_index, valid_index):
    """LGBMRanker: directly optimizes ranking (lambdarank)."""
    import lightgbm as lgb

    # Discretize continuous returns to integer relevance grades
    y_train_int = _returns_to_rank_label(y_train, train_index)
    y_valid_int = _returns_to_rank_label(y_valid, valid_index)

    # Group sizes: number of stocks per day (query group)
    train_dates = train_index.get_level_values(0)
    train_groups = train_dates.value_counts().sort_index().values.tolist()
    valid_dates = valid_index.get_level_values(0)
    valid_groups = valid_dates.value_counts().sort_index().values.tolist()

    dtrain = lgb.Dataset(X_train, label=y_train_int, group=train_groups)
    dvalid = lgb.Dataset(X_valid, label=y_valid_int, group=valid_groups, reference=dtrain)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [20, 50],
        "num_leaves": 128,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 200,
        "reg_lambda": 500,
        "verbose": -1,
        "seed": SEED,
    }
    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dvalid], callbacks=[lgb.early_stopping(50)])
    return model


def train_xgb(X_train, y_train, X_valid, y_valid, feature_names):
    import xgboost as xgb
    fnames = feature_names if len(feature_names) == X_train.shape[1] else None
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=fnames)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=fnames)
    params = {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
              "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
              "objective": "reg:squarederror", "nthread": 4, "verbosity": 0, "seed": SEED}
    model = xgb.train(params, dtrain, num_boost_round=500,
                      evals=[(dvalid, "valid")], early_stopping_rounds=50, verbose_eval=0)
    return model


def train_catboost(X_train, y_train, X_valid, y_valid):
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=8,
                              l2_leaf_reg=500, subsample=0.85, random_seed=SEED,
                              verbose=0, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    return model


def predict_model(model, X_test, model_name, feature_names):
    if model_name == "xgb":
        import xgboost as xgb
        fnames = feature_names if len(feature_names) == X_test.shape[1] else None
        return model.predict(xgb.DMatrix(X_test, feature_names=fnames))
    elif model_name in ("lgb", "lgb_ranker"):
        return model.predict(X_test)
    elif model_name == "catboost":
        return model.predict(X_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="lgb,xgb,catboost,lgb_ranker")
    parser.add_argument("--dims", type=str, default="158,174")
    parser.add_argument("--preprocess", type=str, default="raw")
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",")]
    dim_list = [d.strip() for d in args.dims.split(",")]
    preprocess = args.preprocess

    logger.info("=" * 60)
    logger.info("PIT-SAFE BASELINE COMPARISON")
    logger.info(f"Models: {model_list}")
    logger.info(f"Dims: {dim_list}")
    logger.info(f"Preprocess: {preprocess}")
    logger.info(f"Label: {LABEL_EXPR}")
    logger.info(f"Seed: {SEED}")
    logger.info("=" * 60)

    init_qlib(QLIB_DATA)

    logger.info("Loading Alpha158 dataset...")
    t0 = time.time()
    dataset, dates = load_dataset()
    logger.info(f"Dataset loaded: {time.time()-t0:.1f}s")

    features_cache = {}
    for dim in dim_list:
        logger.info(f"\nPreparing features: {dim} dims...")
        t1 = time.time()
        segs = {}
        for seg in ["train", "valid", "test"]:
            X, y, idx, fnames = prepare_features(dataset, seg, dim, preprocess)
            segs[seg] = (X, y, idx, fnames)
            logger.info(f"  {seg}: {X.shape}")
        features_cache[dim] = segs
        logger.info(f"  Done: {time.time()-t1:.1f}s")

    results = []
    for dim in dim_list:
        segs = features_cache[dim]
        X_train, y_train, train_idx, fnames = segs["train"]
        X_valid, y_valid, valid_idx, _ = segs["valid"]
        X_test, y_test, test_idx, _ = segs["test"]
        n_feat = X_train.shape[1]

        for model_name in model_list:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training: {model_name.upper()} × {dim} dims ({n_feat} features)")
            logger.info(f"{'='*60}")

            t2 = time.time()
            try:
                if model_name == "lgb":
                    model = train_lgb(X_train, y_train, X_valid, y_valid)
                elif model_name == "lgb_ranker":
                    model = train_lgb_ranker(X_train, y_train, X_valid, y_valid,
                                            train_idx, valid_idx)
                elif model_name == "xgb":
                    model = train_xgb(X_train, y_train, X_valid, y_valid, fnames)
                elif model_name == "catboost":
                    model = train_catboost(X_train, y_train, X_valid, y_valid)
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    continue

                train_time = time.time() - t2
                logger.info(f"  Trained in {train_time:.1f}s")

                pred = predict_model(model, X_test, model_name, fnames)
                metrics = evaluate(pred, y_test, test_idx)

                result = {"model": model_name, "dim_mode": dim, "n_features": n_feat,
                          "train_time_s": round(train_time, 1), "pit_safe": True,
                          "label": LABEL_EXPR, "seed": SEED,
                          "dates": {k: list(v) for k, v in dates.items()}, **metrics}
                results.append(result)

                logger.info(f"  IC:       {metrics['ic_mean']:+.4f}")
                logger.info(f"  ICIR:     {metrics['icir']:.3f}")
                logger.info(f"  RankIC:   {metrics['rank_ic_mean']:+.4f}")
                logger.info(f"  RankIC>0: {metrics['rank_ic_pos_ratio']:.1%}")
                # cx round 12 P1-1 / round 11 P2-5: label-spread is NOT
                # a raw return %. Print with explicit unit so readers do
                # not misinterpret the number as "0.96% per day return".
                logger.info(
                    f"  LabelSpread (unit=%s):   %+.3f (×100 for normalised-label display only)" %
                    (metrics.get("label_unit", "?"), metrics['top20_spread']*100,)
                )
                logger.info(f"  Sprd>0:   {metrics['spread_pos_ratio']:.1%}")

            except Exception as e:
                logger.error(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                results.append({"model": model_name, "dim_mode": dim,
                                "n_features": n_feat, "error": str(e)})

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Model':<14} {'Dims':<6} {'IC':>8} {'ICIR':>8} {'RankIC':>8} {'Spread':>10} {'RIC>0':>8}")
    logger.info("-" * 64)
    for r in results:
        if "error" in r:
            logger.info(f"{r['model']:<14} {r['dim_mode']:<6} ERROR: {r['error'][:30]}")
            continue
        logger.info(
            f"{r['model']:<14} {r['dim_mode']:<6} "
            f"{r['ic_mean']:+.4f}  {r['icir']:+.3f}  "
            f"{r['rank_ic_mean']:+.4f}  {r['top20_spread']*100:+.3f}%  "
            f"{r['rank_ic_pos_ratio']:.1%}"
        )

    with open(str(OUTPUT_PATH), "w") as f:
        json.dump({"evaluated_at": datetime.now().isoformat(timespec="seconds"),
                    "results": results}, f, indent=2)
    logger.info(f"\nSaved: {OUTPUT_PATH}")

    # Push results
    try:
        from push.wechat import WeChatPusher
        lines = ["📊 PIT-safe 模型对比", ""]
        lines.append(f"{'模型':<14} {'维度':<5} {'IC':>8} {'ICIR':>7} {'RankIC':>8} {'Spread':>9}")
        lines.append("-" * 55)
        for r in results:
            if "error" in r:
                continue
            lines.append(f"{r['model']:<14} {r['dim_mode']:<5} "
                         f"{r['ic_mean']:+.4f} {r['icir']:+.3f} "
                         f"{r['rank_ic_mean']:+.4f} {r['top20_spread']*100:+.3f}%")
        WeChatPusher().send("\n".join(lines), title="模型训练对比结果")
        logger.info("✅ 推送成功")
    except Exception as e:
        logger.warning(f"Push failed: {e}")


if __name__ == "__main__":
    main()
