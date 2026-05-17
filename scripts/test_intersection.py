"""Test intersection ensemble strategies: XGB ∩ Ranker."""
import os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config.settings import PREDICTION_HORIZON_DAYS
from config.qlib_runtime import init_qlib
from qlib.utils import init_instance_by_config
from qlib.contrib.eva.alpha import calc_ic
from qlib.data import D
from models.feature_merger import FeatureMerger

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


def evaluate_topk(df_day, pred_col, k=20):
    """Evaluate Top-K spread for a single day."""
    if len(df_day) < 40:
        return None
    s = df_day.sort_values(pred_col, ascending=False)
    return s.head(k)["label"].mean() - s.tail(k)["label"].mean()


def main():
    init_qlib(QLIB_DATA)
    merger = FeatureMerger(DATA_DIR)

    today = datetime.now()
    train_start = (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=89)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")

    print(f"Test: {test_start}~{test_end}")

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

    segs = {}
    for seg in ["train", "valid", "test"]:
        print(f"Preparing {seg}...")
        X, y = prepare_174(dataset, seg, merger)
        Xn = X.values.astype(np.float32)
        yn = y.values.astype(np.float32)
        mask = np.isfinite(yn)
        segs[seg] = (Xn[mask], yn[mask], X.index[mask])

    X_train, y_train, train_idx = segs["train"]
    X_valid, y_valid, valid_idx = segs["valid"]
    X_test, y_test, test_idx = segs["test"]

    # Train XGB
    import xgboost as xgb
    print("Training XGB...")
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_valid, label=y_valid)
    params = {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
              "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
              "objective": "reg:squarederror", "nthread": 4, "verbosity": 0, "seed": SEED}
    model_xgb = xgb.train(params, dt, num_boost_round=500,
                           evals=[(dv, "valid")], early_stopping_rounds=50, verbose_eval=0)

    # Train Ranker
    import lightgbm as lgb
    print("Training Ranker...")
    s = pd.Series(y_train, index=train_idx)
    y_train_int = s.groupby(level=0).transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")).fillna(0).astype(np.int32).values
    s2 = pd.Series(y_valid, index=valid_idx)
    y_valid_int = s2.groupby(level=0).transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")).fillna(0).astype(np.int32).values

    train_groups = train_idx.get_level_values(0).value_counts().sort_index().values.tolist()
    valid_groups = valid_idx.get_level_values(0).value_counts().sort_index().values.tolist()
    dtrain_r = lgb.Dataset(X_train, label=y_train_int, group=train_groups)
    dvalid_r = lgb.Dataset(X_valid, label=y_valid_int, group=valid_groups, reference=dtrain_r)
    params_r = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [20, 50],
                "num_leaves": 128, "learning_rate": 0.05, "subsample": 0.85,
                "colsample_bytree": 0.85, "reg_alpha": 200, "reg_lambda": 500,
                "verbose": -1, "seed": SEED}
    model_ranker = lgb.train(params_r, dtrain_r, num_boost_round=500,
                             valid_sets=[dvalid_r], callbacks=[lgb.early_stopping(50)])

    # Predict
    pred_xgb = model_xgb.predict(xgb.DMatrix(X_test))
    pred_ranker = model_ranker.predict(X_test)

    df = pd.DataFrame({"xgb": pred_xgb, "ranker": pred_ranker, "label": y_test}, index=test_idx)

    # === Evaluate strategies ===
    print(f"\n{'='*70}")
    print("INTERSECTION STRATEGIES (test)")
    print(f"{'='*70}")

    strategies = {}

    # Baseline: XGB alone
    spreads_xgb = [evaluate_topk(g, "xgb") for _, g in df.groupby(level=0) if len(g) >= 40]
    spreads_xgb = [s for s in spreads_xgb if s is not None]

    # Baseline: Ranker alone
    spreads_ranker = [evaluate_topk(g, "ranker") for _, g in df.groupby(level=0) if len(g) >= 40]
    spreads_ranker = [s for s in spreads_ranker if s is not None]

    print(f"\n  XGB alone:    Spread={np.mean(spreads_xgb)*100:+.3f}% "
          f"(>0: {np.mean([s>0 for s in spreads_xgb]):.0%})")
    print(f"  Ranker alone: Spread={np.mean(spreads_ranker)*100:+.3f}% "
          f"(>0: {np.mean([s>0 for s in spreads_ranker]):.0%})")

    # Intersection strategies
    for topN in [30, 40, 50, 60]:
        spreads_inter = []
        n_picks = []
        for _, g in df.groupby(level=0):
            if len(g) < 40:
                continue
            xgb_top = set(g.nlargest(topN, "xgb").index)
            ranker_top = set(g.nlargest(topN, "ranker").index)
            inter = list(xgb_top & ranker_top)
            if len(inter) < 5:
                continue
            sub = g.loc[inter].copy()
            # Average rank as score
            top20 = sub.nlargest(min(20, len(sub)), "xgb")  # within intersection, rank by XGB
            bot20 = g.nsmallest(20, "xgb")
            spread = top20["label"].mean() - bot20["label"].mean()
            spreads_inter.append(spread)
            n_picks.append(len(inter))

        if spreads_inter:
            print(f"  Inter Top{topN} (rank by XGB): Spread={np.mean(spreads_inter)*100:+.3f}% "
                  f"(>0: {np.mean([s>0 for s in spreads_inter]):.0%}, "
                  f"avg picks={np.mean(n_picks):.0f})")

    # Quality gate: Ranker picks, XGB filters
    print(f"\n  --- Ranker picks + XGB quality gate ---")
    for gate in [50, 80, 100, 150]:
        spreads_gate = []
        for _, g in df.groupby(level=0):
            if len(g) < 40:
                continue
            xgb_gate = set(g.nlargest(gate, "xgb").index)
            # From ranker's ranking, only keep those in XGB gate
            ranker_sorted = g.sort_values("ranker", ascending=False)
            passed = ranker_sorted[ranker_sorted.index.isin(xgb_gate)].head(20)
            bot20 = g.nsmallest(20, "ranker")
            spread = passed["label"].mean() - bot20["label"].mean()
            spreads_gate.append(spread)

        if spreads_gate:
            print(f"  Ranker Top20 | XGB gate={gate}: Spread={np.mean(spreads_gate)*100:+.3f}% "
                  f"(>0: {np.mean([s>0 for s in spreads_gate]):.0%})")

    # Max of two: per day, pick whichever model's top20 has higher conviction
    print(f"\n  --- Per-day model selection ---")
    spreads_max = []
    for _, g in df.groupby(level=0):
        if len(g) < 40:
            continue
        s_xgb = evaluate_topk(g, "xgb")
        s_ranker = evaluate_topk(g, "ranker")
        # Can't know in advance, but shows upper bound
        spreads_max.append(max(s_xgb, s_ranker))
    print(f"  Oracle (best of two): Spread={np.mean(spreads_max)*100:+.3f}% (upper bound)")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
