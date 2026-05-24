"""Train a regime predictor: current signals → future 5-day market move.

Input: 12 regime scores (today)
Output: probability of >3% market drop or >3% market rise in next 5 days

Uses LightGBM binary classifier with walk-forward validation.

Usage:
    python scripts/train_regime_predictor.py
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"


def build_regime_features() -> pd.DataFrame:
    """Build daily regime feature matrix from all available data sources."""
    logger.info("Building daily regime feature matrix...")

    # Use Shibor dates as timeline (most granular daily data)
    shibor = pd.read_parquet(DATA_DIR / "st_shibor.parquet")
    shibor["date"] = pd.to_datetime(shibor["date"], format="%Y%m%d", errors="coerce")
    shibor = shibor.dropna(subset=["date"]).sort_values("date").set_index("date")
    for col in ["on", "1w", "1m", "3m", "1y"]:
        if col in shibor.columns:
            shibor[col] = pd.to_numeric(shibor[col], errors="coerce")

    features = pd.DataFrame(index=shibor.index)

    # 1. Shibor features
    features["shibor_on"] = shibor.get("on")
    features["shibor_3m"] = shibor.get("3m")
    features["shibor_spread"] = shibor.get("3m", 0) - shibor.get("on", 0)
    features["shibor_on_5d_chg"] = features["shibor_on"].diff(5)
    features["shibor_on_20d_chg"] = features["shibor_on"].diff(20)

    # 2. Margin (融资余额)
    try:
        md = pd.read_parquet(DATA_DIR / "st_margin_detail.parquet")
        md["trade_date"] = pd.to_datetime(md["trade_date"], format="%Y%m%d", errors="coerce")
        md["rzye"] = pd.to_numeric(md["rzye"], errors="coerce")
        daily_margin = md.groupby("trade_date").agg(
            total_rzye=("rzye", "sum"), n_stocks=("rzye", "count")
        )
        daily_margin = daily_margin[daily_margin["n_stocks"] >= 3000]
        features["margin_total"] = daily_margin["total_rzye"].reindex(features.index)
        features["margin_5d_chg"] = features["margin_total"].pct_change(5)
        features["margin_20d_chg"] = features["margin_total"].pct_change(20)
    except Exception:
        pass

    # 3. Limit-down count
    try:
        ld = pd.read_parquet(DATA_DIR / "st_limit_list_d.parquet")
        ld["trade_date"] = pd.to_datetime(ld["trade_date"], format="%Y%m%d", errors="coerce")
        if "limit" in ld.columns:
            down = ld[ld["limit"].astype(str).str.upper() == "D"]
        else:
            ld["pct_chg"] = pd.to_numeric(ld.get("pct_chg", pd.Series()), errors="coerce")
            down = ld[ld["pct_chg"] < -9]
        daily_ld = down.groupby("trade_date").size()
        features["limitdown_count"] = daily_ld.reindex(features.index).fillna(0)
        features["limitdown_5d_avg"] = features["limitdown_count"].rolling(5).mean()
    except Exception:
        pass

    # 4. IC futures
    try:
        ic = pd.read_parquet(DATA_DIR / "ak_futures_ic0.parquet")
        ic["日期"] = pd.to_datetime(ic["日期"], errors="coerce")
        ic = ic.set_index("日期").sort_index()
        ic["收盘价"] = pd.to_numeric(ic["收盘价"], errors="coerce")
        features["ic_close"] = ic["收盘价"].reindex(features.index)
        features["ic_5d_ret"] = features["ic_close"].pct_change(5)
        features["ic_20d_ret"] = features["ic_close"].pct_change(20)
        features["ic_20d_vol"] = features["ic_close"].pct_change().rolling(20).std()
    except Exception:
        pass

    # 5. USD/CNY
    try:
        fx = pd.read_parquet(DATA_DIR / "ak_usdcny.parquet")
        # Find rate and date columns
        rate_col, date_col = None, None
        for col in fx.columns:
            vals = pd.to_numeric(fx[col], errors="coerce").dropna()
            if len(vals) > 100 and 5 < vals.mean() < 10:
                rate_col = col
            if "日期" in col or "date" in col.lower():
                date_col = col
        if rate_col and date_col:
            fx[date_col] = pd.to_datetime(fx[date_col], errors="coerce")
            fx = fx.dropna(subset=[date_col]).set_index(date_col).sort_index()
            fx[rate_col] = pd.to_numeric(fx[rate_col], errors="coerce")
            features["usdcny"] = fx[rate_col].reindex(features.index)
            features["usdcny_5d_chg"] = features["usdcny"].pct_change(5)
    except Exception:
        pass

    # 6. M2
    try:
        m2 = pd.read_parquet(DATA_DIR / "st_cn_m.parquet")
        m2["month"] = pd.to_datetime(m2["month"], format="%Y%m", errors="coerce")
        m2["m2_yoy"] = pd.to_numeric(m2["m2_yoy"], errors="coerce")
        m2["m1_yoy"] = pd.to_numeric(m2["m1_yoy"], errors="coerce")
        m2 = m2.set_index("month").sort_index()
        features["m2_yoy"] = m2["m2_yoy"].reindex(features.index, method="ffill")
        features["m1_yoy"] = m2["m1_yoy"].reindex(features.index, method="ffill")
        features["m1_m2_gap"] = features["m1_yoy"] - features["m2_yoy"]
    except Exception:
        pass

    # 7. Cross-market (纳指 from feature cache)
    try:
        cache = pd.read_parquet(
            DATA_DIR / "feature_cache_174_holder_regime_ma.parquet",
            columns=["nasdaq_ret1d", "nasdaq_ret5d"]
        )
        # Average across all stocks per date (it's broadcast)
        for col in ["nasdaq_ret1d", "nasdaq_ret5d"]:
            daily = cache.groupby(level=0)[col].first()
            features[col] = daily.reindex(features.index)
    except Exception:
        pass

    # Forward fill and clean
    features = features.ffill().replace([np.inf, -np.inf], np.nan)
    features = features.dropna(how="all")
    logger.info(f"  Features: {features.shape}, date range: {features.index.min()} ~ {features.index.max()}")

    return features


def build_labels(features_index) -> pd.DataFrame:
    """Build future 5-day market return labels."""
    logger.info("Building labels...")

    try:
        cache = pd.read_parquet(
            DATA_DIR / "feature_cache_174_holder_regime_ma.parquet",
            columns=["__pnl_return_1d"]
        )
        # Market daily return = cross-sectional mean
        market_ret = cache.groupby(level=0)["__pnl_return_1d"].mean()

        labels = pd.DataFrame(index=features_index)
        labels["market_ret_1d"] = market_ret.reindex(features_index)

        # Future 5-day cumulative return
        labels["future_5d_ret"] = labels["market_ret_1d"].rolling(5).sum().shift(-5)

        # Binary labels
        labels["is_crash"] = (labels["future_5d_ret"] < -0.03).astype(float)  # >3% drop
        labels["is_rally"] = (labels["future_5d_ret"] > 0.03).astype(float)   # >3% rise

        logger.info(f"  Labels: crash={labels['is_crash'].sum():.0f} days, "
                    f"rally={labels['is_rally'].sum():.0f} days, "
                    f"total={len(labels)} days")

        return labels
    except Exception as e:
        logger.error(f"  Label build failed: {e}")
        return pd.DataFrame()


def main():
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    t_start = time.time()

    features = build_regime_features()
    labels = build_labels(features.index)

    if labels.empty:
        return

    # Merge
    df = features.join(labels[["future_5d_ret", "is_crash", "is_rally"]])
    df = df.dropna(subset=["future_5d_ret"])

    feature_cols = [c for c in features.columns if c in df.columns]
    logger.info(f"\nDataset: {len(df)} days, {len(feature_cols)} features")
    logger.info(f"Crash days: {df['is_crash'].sum():.0f} ({df['is_crash'].mean()*100:.1f}%)")
    logger.info(f"Rally days: {df['is_rally'].sum():.0f} ({df['is_rally'].mean()*100:.1f}%)")

    # Walk-forward validation
    train_size = int(len(df) * 0.6)
    valid_size = int(len(df) * 0.2)

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    for target_name, target_col in [("crash", "is_crash"), ("rally", "is_rally")]:
        logger.info(f"\n=== Training {target_name} predictor ===")
        y = df[target_col].values

        X_train, y_train = X[:train_size], y[:train_size]
        X_valid, y_valid = X[train_size:train_size+valid_size], y[train_size:train_size+valid_size]
        X_test, y_test = X[train_size+valid_size:], y[train_size+valid_size:]

        logger.info(f"  Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")
        logger.info(f"  Train positive rate: {y_train.mean():.1%}")

        train_ds = lgb.Dataset(X_train, label=y_train)
        valid_ds = lgb.Dataset(X_valid, label=y_valid, reference=train_ds)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "max_depth": 4,
            "num_leaves": 15,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "verbose": -1,
            "seed": 42,
        }

        model = lgb.train(
            params, train_ds, num_boost_round=200,
            valid_sets=[valid_ds],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        # Test set evaluation
        prob_test = model.predict(X_test)
        pred_test = (prob_test > 0.5).astype(int)

        if y_test.sum() > 0:
            acc = accuracy_score(y_test, pred_test)
            prec = precision_score(y_test, pred_test, zero_division=0)
            rec = recall_score(y_test, pred_test, zero_division=0)
            f1 = f1_score(y_test, pred_test, zero_division=0)
            logger.info(f"  Test: acc={acc:.1%}, prec={prec:.1%}, recall={rec:.1%}, F1={f1:.3f}")
            logger.info(f"  Test positive rate: {y_test.mean():.1%}, predicted positive: {pred_test.mean():.1%}")
        else:
            logger.info(f"  Test: no positive samples")

        # Feature importance
        imp = model.feature_importance(importance_type="gain")
        top_feats = sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:8]
        logger.info(f"  Top features:")
        for fname, fval in top_feats:
            logger.info(f"    {fname}: {fval:.0f}")

        # Current prediction
        X_latest = X[-1:].copy()
        prob_latest = model.predict(X_latest)[0]
        logger.info(f"\n  Current {target_name} probability: {prob_latest:.1%}")
        if prob_latest > 0.5:
            logger.warning(f"  ⚠️ {target_name.upper()} SIGNAL ACTIVE!")

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
