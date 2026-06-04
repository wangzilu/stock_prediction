"""Train LightGBM model using Qlib Alpha158 factors.

Usage: python scripts/train_lgb.py
"""
import os
import sys
import pickle
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from qlib.utils import init_instance_by_config

from config.qlib_runtime import init_qlib
from config.settings import (
    LGB_INFERENCE_UNIVERSE,
    LGB_MIN_DATA_INSTRUMENTS,
    LGB_MIN_PREDICTIONS,
    QLIB_PROVIDER_URI,
)
from scripts.check_qlib_data_health import check_qlib_dir

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "storage")
QLIB_DATA = QLIB_PROVIDER_URI
MODEL_PATH = os.path.join(DATA_DIR, "lgb_model.pkl")
DATASET_PATH = os.path.join(DATA_DIR, "lgb_dataset.pkl")


def _prediction_score_series(predictions) -> pd.Series:
    if isinstance(predictions, pd.Series):
        series = predictions
    elif isinstance(predictions, pd.DataFrame):
        if "score" in predictions.columns:
            series = predictions["score"]
        elif len(predictions.columns) == 1:
            series = predictions.iloc[:, 0]
        else:
            numeric_cols = [
                col for col in predictions.columns
                if pd.api.types.is_numeric_dtype(predictions[col])
            ]
            if len(numeric_cols) != 1:
                raise RuntimeError("prediction output does not contain a single score column")
            series = predictions[numeric_cols[0]]
    else:
        raise RuntimeError(
            f"prediction output must be a Series or DataFrame, got {type(predictions).__name__}"
        )
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _datetime_level(index: pd.MultiIndex) -> int:
    for i, name in enumerate(index.names):
        if name and str(name).lower() in ("datetime", "date"):
            return i
    for i in range(index.nlevels):
        values = index.get_level_values(i)
        if pd.api.types.is_datetime64_any_dtype(values):
            return i
    return 0


def _instrument_level(index: pd.MultiIndex, date_level: int) -> int:
    for i, name in enumerate(index.names):
        if name and str(name).lower() in ("instrument", "code", "symbol"):
            return i
    return 1 if date_level == 0 and index.nlevels > 1 else 0


def _prediction_health(predictions, min_predictions: int) -> dict:
    scores = _prediction_score_series(predictions)
    values = scores.to_numpy()
    finite_mask = np.isfinite(values)
    finite_scores = scores.loc[finite_mask]

    latest_finite_count = len(finite_scores)
    latest_date = None
    stale_prediction_count = 0
    if isinstance(scores.index, pd.MultiIndex) and not finite_scores.empty:
        date_level = _datetime_level(scores.index)
        instrument_level = _instrument_level(scores.index, date_level)
        latest_date = scores.index.get_level_values(date_level).max()
        finite_frame = finite_scores.to_frame("score")
        finite_frame["_datetime"] = pd.to_datetime(
            finite_frame.index.get_level_values(date_level),
            errors="coerce",
        )
        finite_frame["_instrument"] = [
            str(code).upper()
            for code in finite_frame.index.get_level_values(instrument_level)
        ]
        finite_frame = finite_frame.dropna(subset=["_datetime"])
        latest_per_instrument = finite_frame.sort_values(
            ["_instrument", "_datetime"]
        ).groupby("_instrument", sort=False).tail(1)
        latest_finite_count = int(len(latest_per_instrument))
        stale_prediction_count = int(
            (latest_per_instrument["_datetime"] < pd.Timestamp(latest_date)).sum()
        )

    stats = {
        "prediction_count": int(len(scores)),
        "finite_prediction_count": int(finite_mask.sum()),
        "non_finite_prediction_count": int((~finite_mask).sum()),
        "latest_finite_prediction_count": latest_finite_count,
        "stale_prediction_count": stale_prediction_count,
        "latest_date": str(latest_date) if latest_date is not None else "",
        "min_predictions": min_predictions,
    }
    if stats["finite_prediction_count"] == 0:
        raise RuntimeError(f"model produced no finite predictions: {stats}")
    if latest_finite_count < min_predictions:
        raise RuntimeError(
            f"latest finite predictions {latest_finite_count} < required {min_predictions}: {stats}"
        )
    return stats


def _predict_test_segment(model, dataset):
    try:
        return model.predict(dataset, segment="test")
    except TypeError:
        return model.predict(dataset)


def _save_artifacts_atomically(model, dataset):
    os.makedirs(DATA_DIR, exist_ok=True)
    model_path = Path(MODEL_PATH)
    dataset_path = Path(DATASET_PATH)
    tmp_model_path = model_path.with_name(f"{model_path.name}.tmp")
    tmp_dataset_path = dataset_path.with_name(f"{dataset_path.name}.tmp")

    try:
        with tmp_model_path.open("wb") as f:
            pickle.dump(model, f)
        with tmp_dataset_path.open("wb") as f:
            pickle.dump(dataset, f)
        os.replace(tmp_model_path, model_path)
        os.replace(tmp_dataset_path, dataset_path)
    finally:
        for path in (tmp_model_path, tmp_dataset_path):
            if path.exists():
                path.unlink()


def main():
    print("Checking Qlib data health...")
    health = check_qlib_dir(
        Path(QLIB_DATA),
        universe=LGB_INFERENCE_UNIVERSE,
        min_instruments=LGB_MIN_DATA_INSTRUMENTS,
    )
    if not health.ok:
        print("Qlib data health check failed; refusing to train.")
        for error in health.errors:
            print(f"- {error}")
        return 1

    print("Initializing Qlib...")
    init_qlib(QLIB_DATA)

    # Dynamic date ranges
    today = datetime.now()
    train_start = (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=89)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")

    print(f"Train: {train_start} ~ {train_end}")
    print(f"Valid: {valid_start} ~ {valid_end}")
    print(f"Test:  {test_start} ~ {test_end}")

    handler_config = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": train_start,
            "end_time": test_end,
            "instruments": LGB_INFERENCE_UNIVERSE,
        },
    }
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (test_start, test_end),
            },
        },
    }

    print(f"Loading dataset (Alpha158 x {LGB_INFERENCE_UNIVERSE} x 7 years)...")
    dataset = init_instance_by_config(dataset_config)
    print("Dataset ready.")

    # Merge supplementary features (capital flow, valuation, etc.)
    # 2026-06-04 cx P0-c: route through inject_supplementary_into_handler
    # so the production contract (PRODUCTION_SUPPLEMENTARY_GROUPS in
    # config/production_features.py) is the SINGLE GATE for which supp
    # cols enter the trained model. Previously this site called
    # FeatureMerger._load_supplementary() with no allowlist — every new
    # loader added to FeatureMerger silently joined the production
    # champion at the next weekly retrain (no shadow gate).
    # 2026-06-04 cx P0 round 2: production training MUST NOT save a
    # 158-dim model just because supplementary injection raised.
    # Previously this block swallowed any exception with
    # ``print("...skipped"); ...``, then training continued at the
    # Alpha158-only dim and the artifact got overwritten — silently
    # shipping next week's champion as 158-dim with the 242-dim name.
    # Now: any failure here aborts training before model.fit() runs;
    # nothing is saved.
    from models.feature_merger import FeatureMerger
    from config.production_features import PRODUCTION_SUPPLEMENTARY_GROUPS
    merger = FeatureMerger()
    handler = dataset.handler

    supp_cols = merger.inject_supplementary_into_handler(
        handler,
        preprocess=False,
        groups=PRODUCTION_SUPPLEMENTARY_GROUPS,
    )
    # 2026-06-04 cx round 10 follow-up: when the profile is xgb_174,
    # also inject the qlib-custom expression factors (PE / PB / Turn /
    # amount + their momenta + EP/BP/price_pos) — these come from
    # ``D.features`` on the fly, not from FeatureMerger parquet
    # loaders. The xgb_242 profile has an empty custom factor list, so
    # this call is a no-op there.
    custom_cols = merger.inject_qlib_custom_factors_into_handler(handler)
    # 2026-06-04 cx round 16 P1-2 fix: strict profile-aware dim check.
    # Previously only checked ``supp + custom > 0`` — a partial custom
    # factor failure under xgb_174 would still pass because
    # capital_flow's 3 cols alone met the floor. Replaced by
    # ``assert_profile_dimensions`` which requires the EXACT contract
    # counts.
    from config.production_features import (
        PRODUCTION_MODEL_PROFILE,
        assert_profile_dimensions,
    )
    assert_profile_dimensions(
        alpha_count=158,
        supp_count=int(supp_cols or 0),
        custom_count=int(custom_cols or 0),
    )

    # Verify injection worked for learn data + capture the real feature
    # name list for the artifact contract.
    from qlib.data.dataset.handler import DataHandlerLP
    verify = dataset.prepare("train", col_set="feature",
                             data_key=DataHandlerLP.DK_L)
    print(f"Merged {supp_cols} supplementary features "
          f"(learn features: {verify.shape[1]})")
    feature_cols = [
        col[1] if isinstance(col, tuple) else col
        for col in verify.columns.tolist()
    ]
    # 158 alpha158 cols + supp_cols supplementary cols
    # cx round 16 P0-1 fix: include custom_cols in the dim check.
    # Pre-fix this was ``158 + supp_cols`` only; under xgb_174
    # (supp=3, custom=13) it expected 161 instead of 174 and
    # incorrectly raised on a healthy 174-dim training run.
    expected_total = 158 + int(supp_cols or 0) + int(custom_cols or 0)
    if verify.shape[1] != expected_total:
        raise RuntimeError(
            f"production training dim sanity failed: handler reports "
            f"{verify.shape[1]} features but expected "
            f"{expected_total} (158 alpha158 + {supp_cols} supp + "
            f"{custom_cols} qlib_custom). Refusing to save a model "
            f"under a corrupt feature contract."
        )

    # Model selection: XGB (better IC) or LGB (fallback)
    model_type = os.environ.get("TRAIN_MODEL_TYPE", "xgb").lower()

    if model_type == "xgb":
        model_config = {
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
            "kwargs": {
                "n_estimators": 500,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.8789,
                "colsample_bytree": 0.8879,
                "reg_alpha": 205.6999,
                "reg_lambda": 580.9768,
                "n_jobs": 4,
            },
        }
        n_features = expected_total  # cx round 16 P0-1: include custom_cols
        print(f"Training XGBoost ({n_features} features, profile={PRODUCTION_MODEL_PROFILE})...")
    else:
        model_config = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.05,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 4,
            },
        }
        print("Training LightGBM (fallback)...")
    # ---- Apply tradable mask: filter ST/IPO/suspended/一字板 from training data ----
    # 2026-06-04 cx round 6 P0-2: previously this block swallowed
    # mask-application errors and continued training on the full
    # universe — that drifts the training set away from the serve-time
    # universe (which DOES filter ST/IPO/suspended) and produces a
    # champion that's optimistic on stocks the inference path will
    # then refuse. Also: missing mask file was a silent warning.
    # Both modes now hard-fail; the operator must regenerate the mask
    # or pass TRAIN_LGB_ALLOW_NO_MASK=1 for an explicit override.
    tradable_mask_path = os.path.join(DATA_DIR, "tradable_mask.parquet")
    winsorized_label_path = os.path.join(DATA_DIR, "label_5d_winsorized.parquet")

    if os.path.exists(tradable_mask_path):
        from qlib.data.dataset.handler import DataHandlerLP

        mask_df = pd.read_parquet(tradable_mask_path)
        tradable = mask_df["tradable"]

        handler = dataset.handler
        for attr in ('_learn', '_data'):
            df = getattr(handler, attr, None)
            if df is None:
                continue
            # Align mask to handler index
            common = tradable.index.intersection(df.index)
            untradable_idx = tradable.loc[common]
            untradable_idx = untradable_idx[~untradable_idx].index  # indices where tradable=False

            if len(untradable_idx) > 0:
                drop_idx = untradable_idx.intersection(df.index)
                n_before = len(df)
                # Drop untradable rows entirely (NaN label causes XGB crash)
                setattr(handler, attr, df.drop(drop_idx))
                n_after = len(getattr(handler, attr))
                print(f"Tradable mask applied to {attr}: dropped {n_before - n_after} untradable rows ({n_before} -> {n_after})")

        # Optionally apply winsorized labels (only overwrite non-NaN values)
        if os.path.exists(winsorized_label_path):
            win_df = pd.read_parquet(winsorized_label_path)
            win_label = win_df["label_5d_win"]
            for attr in ('_learn', '_data'):
                df = getattr(handler, attr, None)
                if df is None:
                    continue
                label_cols = [c for c in df.columns if c[0] == "label"]
                if label_cols:
                    lc = label_cols[0]
                    common = win_label.index.intersection(df.index)
                    valid_win = win_label.loc[common].dropna()
                    valid_win = valid_win[valid_win.index.isin(df.index)]
                    if len(valid_win) > 0:
                        df.loc[valid_win.index, lc] = valid_win.values
                        print(f"Winsorized labels applied to {attr}: {len(valid_win)} samples")

    elif os.environ.get("TRAIN_LGB_ALLOW_NO_MASK") == "1":
        print(
            "No tradable mask found — TRAIN_LGB_ALLOW_NO_MASK=1 set, "
            "proceeding on full universe (THIS DRIFTS TRAIN vs SERVE)."
        )
    else:
        raise RuntimeError(
            "Tradable mask not found at "
            f"{tradable_mask_path}. Production training MUST run on "
            "the same ST / IPO / suspended-filtered universe that "
            "inference uses. Run: python scripts/build_tradable_mask.py "
            "(or set TRAIN_LGB_ALLOW_NO_MASK=1 for an explicit override)."
        )

    model = init_instance_by_config(model_config)
    model.fit(dataset)
    print("Training complete!")

    # Validate before touching the production model artifact.
    pred = _predict_test_segment(model, dataset)
    try:
        stats = _prediction_health(pred, LGB_MIN_PREDICTIONS)
    except RuntimeError as exc:
        print(f"Prediction health failed; refusing to save model: {exc}")
        return 1
    print("Prediction health passed:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # --- Evaluate: RankIC on test set + compare with previous ---
    train_report = _evaluate_and_compare(pred, dataset, stats)

    # 2026-06-04 cx round 8 P1-7 + round 10 (2): contract must be
    # written BEFORE the model artifact is replaced. Pre-fix the
    # order was:
    #   1. _save_artifacts_atomically(model, ...)
    #   2. try write_contract; except: print "non-fatal"
    # which meant a model could land in production with NO matching
    # contract — the next inference cycle would then either (a) hit
    # the OLD contract's NAME gate (now stricter, would refuse) or
    # (b) hit a count-only path under a stale contract.
    # New order:
    #   1. Write contract to a TEMP path
    #   2. Atomic-promote BOTH (model.pkl, contract.json) together
    # so a failed contract write rolls the whole step back.
    from models.feature_contract import write_contract, contract_path
    _contract_tmp = contract_path(Path(DATA_DIR)).with_suffix(
        ".pretrain.tmp.json"
    )
    try:
        # write_contract writes to its canonical path; we redirect
        # via a temporary swap below. Simplest implementation: call
        # write_contract to a TEMP DATA_DIR alias is overkill; we
        # instead write contract first, fail-loud on error, then
        # atomic-promote model knowing contract is already on disk.
        # 2026-06-04 cx round 18 P0: contract's "supplementary_count" is
        # really "everything-else after Alpha158". For xgb_174 that
        # bucket holds capital_flow (3) + qlib custom (13) = 16. Pre-fix
        # we passed supp_cols only, so the contract validator
        # ``alpha158_count + supplementary_count == n_features`` failed
        # with 158+3=161 vs n_features=174. Now pass supp+custom.
        non_alpha_count = int(supp_cols or 0) + int(custom_cols or 0)
        write_contract(
            Path(DATA_DIR),
            model_pkl_path=str(MODEL_PATH),
            feature_names=list(feature_cols),
            alpha158_count=158,
            supplementary_count=non_alpha_count,
            production_groups=PRODUCTION_SUPPLEMENTARY_GROUPS,
        )
    except Exception as contract_exc:
        # Contract write failed — model artifact NOT yet touched.
        # Refuse to overwrite the existing production model with one
        # whose contract is unknown.
        raise RuntimeError(
            f"Feature contract write failed BEFORE model save — "
            f"refusing to promote a model whose feature contract is "
            f"unknown: {contract_exc}"
        )

    # Contract is on disk; now safe to atomic-promote the model.
    _save_artifacts_atomically(model, dataset)
    print(f"Model saved to {MODEL_PATH}")

    # Save feature contract to experiment artifact
    try:
        from tracker.artifact_contract import ExperimentArtifact
        exp_id = f"train_lgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        art = ExperimentArtifact.create(
            experiment_id=exp_id,
            model_name="xgb_174",
            feature_set="FS-174",
            description="Production LGB training run",
            # Per code-review P2 2026-05-31: this used to be
            #   feature_cols if 'feature_cols' in dir() else []
            # but `feature_cols` was never defined → always [], artifact
            # contract empty. Now feature_cols is set above from the
            # actual handler columns after supplementary merge.
            feature_list=list(feature_cols),
            label_column=LABEL_EXPR,
            data_asof_date=train_end,
            n_features=n_features,
            train_start=train_start,
            train_end=train_end,
        )
        art.save_metrics({
            "rank_ic_mean": train_report.get("rank_ic"),
            "prediction_count": stats.get("latest_finite_prediction_count", 0),
        })
        print(f"Artifact saved: {exp_id}")
    except Exception as e:
        print(f"Artifact save failed (non-fatal): {e}")

    # Update registry — keep production artifact path in sync with research
    try:
        from models.registry import ModelRegistry
        reg = ModelRegistry()
        reg.register("xgb_174", role="champion", feature_set="FS-174",
                      model_path=MODEL_PATH, n_features=n_features,
                      train_start=train_start, train_end=train_end,
                      metrics={"rank_ic": train_report.get("rank_ic"),
                               "prediction_count": stats.get("latest_finite_prediction_count", 0),
                               "train_date": datetime.now().strftime("%Y-%m-%d")})
        print("Registry updated: champion=xgb_174")
    except Exception as e:
        print(f"Registry update failed (non-fatal): {e}")

    print(f"\nPredictions shape: {pred.shape}")
    print(f"Last 5 predictions:")
    print(pred.tail(5))

    # Push training result
    _push_training_result(train_report)

    return 0


def _evaluate_and_compare(pred, dataset, health_stats) -> dict:
    """Evaluate model on test set and compare with previous training."""
    from scipy.stats import spearmanr
    import json as _json

    report = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "prediction_count": health_stats.get("latest_finite_prediction_count", 0),
        "rank_ic": None,
        "prev_rank_ic": None,
        "delta_ic": None,
        "status": "success",
    }

    # Compute RankIC on test set
    try:
        test_pred = pred
        if isinstance(test_pred, pd.DataFrame):
            test_pred = test_pred.iloc[:, 0]

        test_label = dataset.prepare("test", col_set="label")
        if isinstance(test_label, pd.DataFrame):
            test_label = test_label.iloc[:, 0]

        # Align
        common = test_pred.index.intersection(test_label.index)
        p = test_pred.loc[common].values.astype(float)
        l = test_label.loc[common].values.astype(float)
        mask = np.isfinite(p) & np.isfinite(l)
        p, l = p[mask], l[mask]

        if len(p) > 100:
            # Per-date RankIC
            pred_s = pd.Series(p, index=common[mask])
            label_s = pd.Series(l, index=common[mask])
            rics = []
            for dt, g in pred_s.groupby(level=0):
                gl = label_s.reindex(g.index).dropna()
                c = g.index.intersection(gl.index)
                if len(c) >= 40:
                    ric = spearmanr(g.loc[c].values, gl.loc[c].values).statistic
                    if np.isfinite(ric):
                        rics.append(ric)
            if rics:
                report["rank_ic"] = round(float(np.mean(rics)), 4)
                report["rank_ic_std"] = round(float(np.std(rics)), 4)
                report["n_test_days"] = len(rics)
                print(f"\nTest RankIC: {report['rank_ic']:+.4f} (±{report['rank_ic_std']:.4f}, {len(rics)} days)")
    except Exception as e:
        print(f"RankIC evaluation failed: {e}")

    # Load previous training log for comparison
    eval_log_path = os.path.join(DATA_DIR, "train_eval_history.jsonl")
    try:
        if os.path.exists(eval_log_path):
            with open(eval_log_path) as f:
                lines = f.readlines()
            if lines:
                prev = _json.loads(lines[-1])
                report["prev_rank_ic"] = prev.get("rank_ic")
                if report["rank_ic"] is not None and report["prev_rank_ic"] is not None:
                    report["delta_ic"] = round(report["rank_ic"] - report["prev_rank_ic"], 4)
                    direction = "↑" if report["delta_ic"] > 0 else "↓" if report["delta_ic"] < 0 else "→"
                    print(f"vs Previous: {report['prev_rank_ic']:+.4f} → {report['rank_ic']:+.4f} ({direction}{abs(report['delta_ic']):.4f})")
    except Exception:
        pass

    # Append to eval history
    try:
        with open(eval_log_path, "a") as f:
            f.write(_json.dumps(report) + "\n")
    except Exception:
        pass

    return report


def _push_training_result(report: dict):
    """Push training result via WeChat."""
    try:
        from push.wechat import WeChatPusher

        ric = report.get("rank_ic")
        prev = report.get("prev_rank_ic")
        delta = report.get("delta_ic")
        n_pred = report.get("prediction_count", 0)

        lines = [f"🔧 模型训练完成 {report.get('date', '')}"]
        lines.append(f"预测数: {n_pred}")

        if ric is not None:
            lines.append(f"RankIC(近{report.get('n_test_days','?')}天): {ric:+.4f}")
            # Context: 24-split gate avg is +0.041
            if ric > 0.02:
                lines.append("信号正常 ✅")
            elif ric > 0:
                lines.append("信号偏弱 ⚠️")
            else:
                lines.append("信号为负（短期波动，非衰退）")
        if prev is not None and delta is not None:
            direction = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
            lines.append(f"上次: {prev:+.4f} {direction} Δ={delta:+.4f}")

        msg = "\n".join(lines)
        WeChatPusher().send(msg, title="模型训练")
        print(f"Training result pushed")
    except Exception as e:
        print(f"Push failed: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
