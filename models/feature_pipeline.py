"""Shared feature preparation pipeline for XGB 174-dim model.

Centralizes the Alpha158 + flow + custom feature preparation that was
duplicated across 5+ scripts. All training/evaluation scripts should
import from here.

Usage:
    from models.feature_pipeline import prepare_features_174, XGB_PARAMS

    X, y = prepare_features_174(dataset, "train", merger)
"""
import numpy as np
import pandas as pd

from models.feature_merger import FeatureMerger

# Qlib bin custom factor expressions (PE/PB/Turn that Alpha158 doesn't provide)
CUSTOM_EXPRS = [
    "$pe", "$pb", "$turn", "$amount",
    "$pe / Ref($pe, 20) - 1",
    "$pb / Ref($pb, 20) - 1",
    "$turn / Mean($turn, 20)",
    "$turn / Mean($turn, 60)",
    "$amount / Mean($amount, 20)",
    "Std($turn, 20)",
    "1.0 / If(Abs($pe) > 0.01, $pe, 1.0)",
    "1.0 / If(Abs($pb) > 0.01, $pb, 1.0)",
    "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20) + 1e-8)",
]
CUSTOM_NAMES = [
    "pe", "pb", "turn_raw", "amount_raw",
    "pe_mom20", "pb_mom20", "turn_anom20", "turn_anom60",
    "amount_anom20", "turn_vol20", "ep", "bp", "price_pos20",
]

# Standard XGB hyperparameters (tuned via earlier experiments)
XGB_PARAMS = {
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8789,
    "colsample_bytree": 0.8879,
    "reg_alpha": 205.6999,
    "reg_lambda": 580.9768,
    "objective": "reg:squarederror",
    "nthread": 4,
    "verbosity": 0,
    "seed": 42,
}


def prepare_features_174(dataset, segment: str, merger: FeatureMerger,
                         include_holder: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare 174-dim (or 175-dim with holder) features for a dataset segment.

    Returns unfiltered (X DataFrame, y Series) — caller is responsible for NaN masking.

    Args:
        dataset: Qlib DatasetH object
        segment: "train", "valid", or "test"
        merger: FeatureMerger instance
        include_holder: if True, adds holder_num (175-dim)

    Returns:
        (X: pd.DataFrame, y: pd.Series) — raw, not NaN-filtered
    """
    from qlib.data import D

    X = dataset.prepare(segment, col_set="feature")
    y = dataset.prepare(segment, col_set="label")
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # Capital flow (proven, 3 dims)
    flow = merger._load_capital_flow(X.index)
    if flow is not None:
        X = X.join(flow, how="left")

    # Qlib custom expressions (proven, 13 dims)
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

    # Optional: holder_num (1 dim, for 175 model)
    if include_holder:
        holder = merger._load_st_holder_number(X.index)
        if holder is not None:
            X = X.join(holder, how="left")

    return X, y


def prepare_segment_numpy(dataset, segment: str, merger: FeatureMerger,
                          include_holder: bool = False
                          ) -> tuple[np.ndarray, np.ndarray, pd.MultiIndex, list[str]]:
    """Prepare features and filter NaN labels, return numpy arrays ready for training.

    Returns:
        (X_np, y_np, index, feature_names) — NaN-filtered
    """
    X, y = prepare_features_174(dataset, segment, merger, include_holder=include_holder)
    Xn = X.values.astype(np.float32)
    yn = y.values.astype(np.float32)
    mask = np.isfinite(yn)
    return Xn[mask], yn[mask], X.index[mask], list(X.columns)


def train_xgb(X_train, y_train, X_valid, y_valid, params=None):
    """Train XGB with standard params and early stopping."""
    import xgboost as xgb
    p = params or XGB_PARAMS
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_valid, label=y_valid)
    model = xgb.train(p, dt, num_boost_round=500,
                      evals=[(dv, "valid")], early_stopping_rounds=50, verbose_eval=0)
    return model


def load_daily_returns(index: pd.MultiIndex, execution_price: str = "close") -> pd.Series:
    """Load 1-day realized returns for portfolio PnL accounting.

    Args:
        index: MultiIndex (datetime, instrument) defining the universe.
        execution_price: "close" or "open".
            - "close": close-to-close return = (D+1 close) / (D close) - 1.
              Signal at D close, PnL from D close to D+1 close.
            - "open": open-to-open return = (D+1 open) / (D open) - 1.
              Signal at D-1 close, execute at D open, PnL from D open to D+1 open.
              More realistic: avoids lookahead on execution price.

    IMPORTANT: This is NOT the model training label (which is N-day forward).
    Never use model labels as PnL returns.
    """
    from qlib.data import D

    if execution_price not in ("close", "open"):
        raise ValueError(f"execution_price must be 'close' or 'open', got '{execution_price}'")

    insts = sorted(set(str(c) for c in index.get_level_values(1)))
    dates = sorted(index.get_level_values(0).unique())
    start = str(min(dates))[:10]
    end = str(max(dates))[:10]

    if execution_price == "close":
        ret = D.features(
            insts,
            ["Ref($close, -1) / $close - 1"],
            start_time=start,
            end_time=end,
        )
        ret.columns = ["pnl_return_1d"]
    else:
        # Open-to-open: load both open prices and compute manually.
        # Ref($open, -1) is tomorrow's open; Ref($open, -1) / $open - 1
        # gives the return from today's open to tomorrow's open.
        ret = D.features(
            insts,
            ["Ref($open, -1) / $open - 1"],
            start_time=start,
            end_time=end,
        )
        ret.columns = ["pnl_return_1d"]

    ret = ret.swaplevel().sort_index()
    return ret.replace([np.inf, -np.inf], np.nan).dropna()


def evaluate_predictions(pred, label, index) -> dict:
    """Standard evaluation: IC, ICIR, RankIC, Top20 Spread."""
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
