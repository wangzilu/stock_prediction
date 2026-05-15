"""Train 174-dim (proven factors) + ST_CLIENT new data, compare with baseline.

174 base = Alpha158(158) + flow(3) + qlib_custom(13)
+ ST margin_detail: 融资融券 (7 factors, daily per-stock)
+ ST moneyflow_hsgt: 北向资金 (4 factors, daily market-level broadcast)
+ ST limit/top: 涨跌停/龙虎榜 (sparse, converted to recent-event flags)
+ ST fina_indicator: 财务指标 (roe/margin/eps/bps/debt/turnover)
+ ST holder_number: 股东户数

v2: enhanced preprocessing (ts-derivatives + mcap neutralization + rank)

Usage:
    python scripts/train_174_plus_st.py
    python scripts/train_174_plus_st.py --preprocess enhanced
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


# ========== ST data loaders ==========

def load_st_margin(index):
    """Load 融资融券 daily data, asof merge to training index."""
    path = DATA_DIR / "st_margin_detail.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty or "qlib_code" not in df.columns:
        return None

    factor_cols = ["rzye", "rqye", "rzmre", "rzche", "rqmcl", "rqchl", "rzrqye"]
    factor_cols = [c for c in factor_cols if c in df.columns]
    if not factor_cols:
        return None

    for c in factor_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Rename with prefix
    out_cols = [f"margin_{c}" for c in factor_cols]
    ts = df[["qlib_code", "date"] + factor_cols].copy()
    ts.columns = ["qlib_code", "date"] + out_cols

    merger = FeatureMerger(DATA_DIR)
    result = merger._asof_merge_timeseries(ts, index, "date", out_cols)
    if result is not None:
        logger.info(f"  ST margin: {len(out_cols)} factors, asof merged")
    return result


def load_st_hsgt(index):
    """Load 北向资金汇总 (market-level), broadcast to all stocks."""
    path = DATA_DIR / "st_moneyflow_hsgt.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty or "date" not in df.columns:
        return None

    factor_cols = ["hgt", "sgt", "north_money", "south_money"]
    factor_cols = [c for c in factor_cols if c in df.columns]
    if not factor_cols:
        return None

    for c in factor_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Market-level: need to match by date only (broadcast to all stocks)
    date_level = 0
    train_dates = pd.to_datetime(index.get_level_values(date_level))

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # merge_asof on date only
    left = pd.DataFrame({"date": train_dates, "_pos": range(len(train_dates))})
    left = left.sort_values("date")
    right = df[["date"] + factor_cols].drop_duplicates("date").sort_values("date")

    merged = pd.merge_asof(left, right, on="date", direction="backward")

    result_arrays = {}
    for col in factor_cols:
        arr = np.full(len(index), np.nan)
        for _, row in merged.iterrows():
            arr[int(row["_pos"])] = row[col] if pd.notna(row.get(col)) else np.nan
        result_arrays[f"hsgt_{col}"] = arr

    out_cols = [f"hsgt_{c}" for c in factor_cols]
    logger.info(f"  ST hsgt: {len(out_cols)} factors, date-broadcast")
    return pd.DataFrame(result_arrays, index=index)


def load_st_limit_flags(index):
    """Convert 涨跌停 sparse data to per-stock rolling flags (vectorized):
       - limit_up_5d: hit limit-up in last 5 trading days (0/1)
       - limit_up_count_20d: count of limit-up days in last 20 calendar days
    """
    path = DATA_DIR / "st_limit_list_d.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty or "qlib_code" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["qlib_code"] = df["qlib_code"].str.upper()

    # Build per-stock daily indicator: 1 on days with limit event
    daily = df.groupby(["qlib_code", "date"]).size().reset_index(name="_hit")
    daily["_hit"] = 1
    daily = daily.pivot_table(index="date", columns="qlib_code", values="_hit",
                              fill_value=0).sort_index()

    # Rolling windows
    any_5d = daily.rolling(5, min_periods=1).max()    # 1 if any hit in last 5 days
    count_20d = daily.rolling(20, min_periods=1).sum()  # count in last 20 days

    # Align to training index
    date_level = 0
    inst_level = 1 if index.nlevels > 1 else 0
    train_dates = index.get_level_values(date_level)
    train_insts = index.get_level_values(inst_level).astype(str).str.upper()

    # Vectorized lookup via reindex
    arr_5d = np.zeros(len(index), dtype=np.float32)
    arr_20d = np.zeros(len(index), dtype=np.float32)

    # Build (date, inst) -> position mapping using merge
    lookup = pd.DataFrame({"date": train_dates, "inst": train_insts,
                           "_pos": range(len(index))})

    for stock in lookup["inst"].unique():
        if stock not in any_5d.columns:
            continue
        mask = lookup["inst"] == stock
        sub = lookup.loc[mask]
        # Use searchsorted for fast date alignment
        dates_arr = any_5d.index
        idxs = dates_arr.searchsorted(sub["date"].values, side="right") - 1
        valid = (idxs >= 0) & (idxs < len(dates_arr))
        positions = sub["_pos"].values[valid]
        arr_5d[positions] = any_5d[stock].values[idxs[valid]]
        arr_20d[positions] = count_20d[stock].values[idxs[valid]]

    logger.info(f"  ST limit flags: 2 factors, {int(arr_5d.sum())} samples with recent limit-up")
    return pd.DataFrame({
        "limit_up_5d": arr_5d,
        "limit_up_count_20d": arr_20d,
    }, index=index)


def load_st_toplist_flags(index):
    """Convert 龙虎榜 sparse data to per-stock rolling flags (vectorized):
       - toplist_5d: appeared on dragon-tiger list in last 5 days (0/1)
       - toplist_net_5d: rolling 5-day sum of net buy amount
    """
    path = DATA_DIR / "st_top_list.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty or "qlib_code" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["qlib_code"] = df["qlib_code"].str.upper()
    df["net_amount"] = pd.to_numeric(df.get("net_amount", 0), errors="coerce").fillna(0)

    # Aggregate per (stock, date): flag + net amount
    agg = df.groupby(["qlib_code", "date"]).agg(
        _flag=("net_amount", "size"),
        _net=("net_amount", "sum"),
    ).reset_index()
    agg["_flag"] = 1

    flag_pivot = agg.pivot_table(index="date", columns="qlib_code",
                                 values="_flag", fill_value=0).sort_index()
    net_pivot = agg.pivot_table(index="date", columns="qlib_code",
                                values="_net", fill_value=0).sort_index()

    any_5d = flag_pivot.rolling(5, min_periods=1).max()
    net_5d = net_pivot.rolling(5, min_periods=1).sum()

    # Align to training index
    date_level = 0
    inst_level = 1 if index.nlevels > 1 else 0
    train_dates = index.get_level_values(date_level)
    train_insts = index.get_level_values(inst_level).astype(str).str.upper()

    arr_flag = np.zeros(len(index), dtype=np.float32)
    arr_net = np.zeros(len(index), dtype=np.float32)

    lookup = pd.DataFrame({"date": train_dates, "inst": train_insts,
                           "_pos": range(len(index))})

    for stock in lookup["inst"].unique():
        if stock not in any_5d.columns:
            continue
        mask = lookup["inst"] == stock
        sub = lookup.loc[mask]
        dates_arr = any_5d.index
        idxs = dates_arr.searchsorted(sub["date"].values, side="right") - 1
        valid = (idxs >= 0) & (idxs < len(dates_arr))
        positions = sub["_pos"].values[valid]
        arr_flag[positions] = any_5d[stock].values[idxs[valid]]
        arr_net[positions] = net_5d[stock].values[idxs[valid]]

    logger.info(f"  ST toplist flags: 2 factors, {int(arr_flag.sum())} samples with recent toplist")
    return pd.DataFrame({
        "toplist_5d": arr_flag,
        "toplist_net_5d": arr_net,
    }, index=index)


def load_st_fina(index):
    """Load 财务指标 (quarterly, PIT-safe via ann_date), pick key factors."""
    path = DATA_DIR / "st_fina_indicator.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty or "qlib_code" not in df.columns:
        return None

    # Select high-coverage, economically meaningful factors
    factor_cols = [c for c in [
        "roe", "grossprofit_margin", "netprofit_margin", "eps", "bps",
        "current_ratio", "debt_to_assets", "assets_turn", "ocfps", "roe_dt",
    ] if c in df.columns]
    if not factor_cols:
        return None

    for c in factor_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # PIT-safe: use ann_date (announcement date) as effective date
    df["ann_date"] = pd.to_datetime(df["ann_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["ann_date"])

    out_cols = [f"fina_{c}" for c in factor_cols]
    ts = df[["qlib_code", "ann_date"] + factor_cols].copy()
    ts.columns = ["qlib_code", "ann_date"] + out_cols
    ts = ts.sort_values(["qlib_code", "ann_date"]).drop_duplicates(
        ["qlib_code", "ann_date"], keep="last")

    merger = FeatureMerger(DATA_DIR)
    result = merger._asof_merge_timeseries(ts, index, "ann_date", out_cols)
    if result is not None:
        logger.info(f"  ST fina: {len(out_cols)} factors, asof merged via ann_date")
    return result


def load_st_holder(index):
    """Load 股东户数 (quarterly, PIT-safe via ann_date)."""
    path = DATA_DIR / "st_holder_number.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty or "qlib_code" not in df.columns:
        return None

    df["holder_num"] = pd.to_numeric(df.get("holder_num"), errors="coerce")
    df["ann_date"] = pd.to_datetime(df["ann_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["ann_date", "holder_num"])

    ts = df[["qlib_code", "ann_date", "holder_num"]].copy()
    ts = ts.sort_values(["qlib_code", "ann_date"]).drop_duplicates(
        ["qlib_code", "ann_date"], keep="last")

    merger = FeatureMerger(DATA_DIR)
    result = merger._asof_merge_timeseries(ts, index, "ann_date", ["holder_num"])
    if result is not None:
        logger.info(f"  ST holder: 1 factor, asof merged via ann_date")
    return result


def preprocess_st_factors(st_df, index, mode="raw"):
    """Apply preprocessing to ST factors before merging with base features.

    Args:
        st_df: DataFrame of raw ST factors
        mode: "raw", "rank", or "enhanced"
    """
    if st_df is None or st_df.empty or mode == "raw":
        return st_df

    merger = FeatureMerger(DATA_DIR)
    return merger._preprocess_supplementary(st_df, index, mode=mode)


# ========== Training ==========

def evaluate(pred, label, index):
    from qlib.contrib.eva.alpha import calc_ic
    mask = np.isfinite(pred) & np.isfinite(label)
    ps = pd.Series(pred[mask], index=index[mask])
    ls = pd.Series(label[mask], index=index[mask])
    ic, ric = calc_ic(ps, ls)
    df = pd.DataFrame({"pred": ps, "label": ls})
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", type=str, default="enhanced",
                        choices=["raw", "rank", "enhanced"],
                        help="Preprocessing for ST factors: raw/rank/enhanced")
    args = parser.parse_args()
    pp_mode = args.preprocess

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

    logger.info(f"Preprocess mode: {pp_mode}")
    logger.info("Loading Alpha158...")
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

    # Prepare 174 base features for each segment
    merger = FeatureMerger(DATA_DIR)
    base_segs = {}

    for seg in ["train", "valid", "test"]:
        logger.info(f"Preparing {seg}...")
        X = dataset.prepare(seg, col_set="feature")
        y = dataset.prepare(seg, col_set="label")
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        # Add flow (proven)
        flow = merger._load_capital_flow(X.index)
        if flow is not None:
            X = X.join(flow, how="left")

        # Add qlib_custom (proven)
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

        base_segs[seg] = (X, y)
        logger.info(f"  {seg} base: {X.shape}")

    # Load ST data once (for each segment's index)
    logger.info("Loading ST data...")
    all_loaders = [
        ("margin", load_st_margin),
        ("hsgt", load_st_hsgt),
        ("limit", load_st_limit_flags),
        ("toplist", load_st_toplist_flags),
        ("fina", load_st_fina),
        ("holder", load_st_holder),
    ]

    st_data = {}
    for name, loader in all_loaders:
        for seg in ["train", "valid", "test"]:
            key = f"{name}_{seg}"
            d = loader(base_segs[seg][0].index)
            st_data[key] = d
            if d is not None and seg == "train":
                logger.info(f"  {name}: {d.shape[1]} cols, "
                            f"{d.notna().any(axis=1).sum()} non-null rows")

    # Experiments: raw factors first, then preprocessed
    experiments = [
        ("174_base", []),
        ("174+fina", ["fina"]),
        ("174+holder", ["holder"]),
        ("174+margin", ["margin"]),
        ("174+limit", ["limit"]),
        ("174+all_daily", ["margin", "hsgt", "limit", "toplist"]),
        ("174+fina+holder", ["fina", "holder"]),
        ("174+all", ["margin", "hsgt", "limit", "toplist", "fina", "holder"]),
    ]

    results = []
    import xgboost as xgb

    for exp_name, st_groups in experiments:
        logger.info(f"\n{'='*50}")
        logger.info(f"Experiment: {exp_name} (preprocess={pp_mode})")

        segs = {}
        for seg in ["train", "valid", "test"]:
            X = base_segs[seg][0].copy()
            y = base_segs[seg][1]

            # Collect raw ST factors for this experiment
            st_frames = []
            for g in st_groups:
                d = st_data.get(f"{g}_{seg}")
                if d is not None:
                    st_frames.append(d)

            if st_frames:
                st_combined = pd.concat(st_frames, axis=1)
                # Apply preprocessing
                st_processed = preprocess_st_factors(st_combined, X.index, mode=pp_mode)
                if st_processed is not None and not st_processed.empty:
                    X = X.join(st_processed, how="left")

            Xn = X.values.astype(np.float32)
            yn = y.values.astype(np.float32)
            mask = np.isfinite(yn)
            segs[seg] = (Xn[mask], yn[mask], X.index[mask])

        n_feat = segs["train"][0].shape[1]
        logger.info(f"  Features: {n_feat}")

        t0 = time.time()
        model = train_xgb(segs["train"][0], segs["train"][1],
                          segs["valid"][0], segs["valid"][1])
        pred = model.predict(xgb.DMatrix(segs["test"][0]))
        metrics = evaluate(pred, segs["test"][1], segs["test"][2])
        elapsed = time.time() - t0

        result = {"name": exp_name, "n_features": n_feat, "preprocess": pp_mode,
                  **metrics, "time_s": round(elapsed, 1)}
        results.append(result)

        logger.info(f"  IC:     {metrics['ic_mean']:+.4f}  ICIR: {metrics['icir']:+.3f}")
        logger.info(f"  RankIC: {metrics['rank_ic_mean']:+.4f}  RIC>0: {metrics['rank_ic_pos_ratio']:.0%}")
        logger.info(f"  Spread: {metrics['top20_spread']*100:+.3f}%  Sprd>0: {metrics['spread_pos_ratio']:.0%}")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"SUMMARY: 174 base vs +ST data (preprocess={pp_mode})")
    logger.info(f"{'='*80}")
    logger.info(f"{'Name':<22} {'Feat':>5} {'IC':>8} {'ICIR':>7} {'RankIC':>8} {'Spread':>9} {'RIC>0':>6}")
    logger.info("-" * 70)

    base_ric = results[0]["rank_ic_mean"] if results else 0
    for r in results:
        delta = r["rank_ic_mean"] - base_ric
        logger.info(
            f"{r['name']:<22} {r['n_features']:>5} "
            f"{r['ic_mean']:+.4f}  {r['icir']:+.3f}  "
            f"{r['rank_ic_mean']:+.4f} ({delta:+.4f})  "
            f"{r['top20_spread']*100:+.3f}%  {r['rank_ic_pos_ratio']:.0%}"
        )

    # Save
    out = DATA_DIR / "train_174_plus_st_results_v2.json"
    with open(str(out), "w") as f:
        json.dump({"evaluated_at": datetime.now().isoformat(timespec="seconds"),
                    "preprocess": pp_mode,
                    "results": results}, f, indent=2)
    logger.info(f"\nSaved: {out}")

    # Push
    try:
        from push.wechat import WeChatPusher
        lines = [f"📊 174维+ST v2 ({pp_mode})", "=" * 40, ""]
        lines.append(f"{'版本':<22} {'维度':>4} {'IC':>8} {'ICIR':>7} {'RankIC':>8} {'Spread':>9}")
        lines.append("-" * 65)
        for r in results:
            lines.append(
                f"{r['name']:<22} {r['n_features']:>4} "
                f"{r['ic_mean']:+.4f}  {r['icir']:+.3f}  "
                f"{r['rank_ic_mean']:+.4f}  {r['top20_spread']*100:+.3f}%"
            )
        lines.append("")
        lines.append(f"基线 174维: RankIC={base_ric:+.4f}, "
                     f"Spread={results[0]['top20_spread']*100:+.3f}%")
        best = max(results, key=lambda r: r["rank_ic_mean"])
        lines.append(f"最佳: {best['name']} RankIC={best['rank_ic_mean']:+.4f}")

        msg = "\n".join(lines)
        print(msg)
        pusher = WeChatPusher()
        if pusher.send(msg, title=f"174+ST v2 ({pp_mode})"):
            logger.info("✅ 推送成功")
        else:
            logger.info("❌ 推送失败")
    except Exception as e:
        logger.warning(f"Push failed: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
