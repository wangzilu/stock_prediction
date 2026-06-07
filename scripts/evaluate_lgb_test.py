"""Evaluate LGB model quality on test data with IC, RankIC, and bucket analysis.

Outputs data/storage/lgb_eval_latest.json with metrics for production gating.

Usage:
    python scripts/evaluate_lgb_test.py
    python scripts/evaluate_lgb_test.py --json
"""
import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    LGB_MIN_PREDICTIONS,
    PREDICTION_HORIZON_DAYS,
)
from config.qlib_runtime import init_qlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
# cx round 23 E.P2 #6: resolve to the ACTIVE profile's binary instead of
# the legacy alias. Evaluating the legacy path while the live champion is
# a different profile produces numbers that don't represent reality.
try:
    from config.production_features import production_model_filename
    MODEL_PATH = str(DATA_DIR / production_model_filename())
except Exception:
    MODEL_PATH = str(DATA_DIR / "lgb_model.pkl")
EVAL_PATH = DATA_DIR / "lgb_eval_latest.json"
EVAL_HISTORY_PATH = DATA_DIR / "lgb_eval_history.json"

LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"


def evaluate(
    model_path: str = MODEL_PATH,
    qlib_data: str = QLIB_DATA,
    universe: str = "all",
    min_predictions: int = LGB_MIN_PREDICTIONS,
    test_days: int = 30,
    topk: int = 20,
    profile: str | None = None,
) -> dict:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return

    if not os.path.exists(model_path):
        return {"ok": False, "error": f"Model file not found: {model_path}"}

    today = datetime.now()
    test_start = (today - timedelta(days=test_days)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")

    logger.info(f"Test period: {test_start} ~ {test_end}")
    logger.info(f"Label: {LABEL_EXPR}")

    # 2026-06-04 cx round 8 P1-3: use the production-safe loader so
    # the model evaluation sees the SAME 242-dim feature shape the
    # live champion sees. Pre-fix this script built a bare Alpha158
    # (158-dim) dataset and called pickle.load on the 242-dim model
    # — XGB walked default-direction branches on the missing 84 cols
    # and produced silent-garbage IC/spread that misled "is 242 ok?"
    # decisions.
    from models.production_inference import load_production_model
    # cx round 23 E.P2 #6: thread profile= through so a candidate binary
    # is evaluated against ITS contract, not the live champion's.
    model, dataset = load_production_model(
        test_start, test_end,
        model_path=model_path,
        instruments=universe,
        label_expr=LABEL_EXPR,
        profile=profile,
    )

    logger.info("Running predictions...")
    pred = model.predict(dataset=dataset)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")

    label = dataset.prepare("test", col_set="label")
    if isinstance(label, pd.DataFrame):
        label = label.iloc[:, 0]

    # Align
    common = pred.index.intersection(label.index)
    pred_s = pred.loc[common, "score"] if "score" in pred.columns else pred.loc[common].iloc[:, 0]
    label_s = label.loc[common]

    mask = pred_s.notna() & label_s.notna() & np.isfinite(pred_s) & np.isfinite(label_s)
    pred_clean = pred_s[mask]
    label_clean = label_s[mask]

    n_samples = len(pred_clean)
    logger.info(f"Aligned samples: {n_samples}")

    if n_samples < 100:
        return {"ok": False, "error": f"Too few aligned samples: {n_samples}"}

    # IC / RankIC
    ic, rank_ic = calc_ic(pred_clean, label_clean)
    ic_mean = float(ic.mean())
    ic_std = float(ic.std())
    icir = ic_mean / (ic_std + 1e-8)
    rank_ic_mean = float(rank_ic.mean())
    rank_ic_pos_ratio = float((rank_ic > 0).mean())
    n_dates = len(ic)

    # Long-short return
    try:
        ls_ret, long_ret = calc_long_short_return(pred_clean, label_clean, quantile=0.2)
        ls_mean = float(ls_ret.mean())
        long_mean = float(long_ret.mean())
    except Exception as e:
        logger.warning(f"Long-short calc error: {e}")
        ls_mean = 0.0
        long_mean = 0.0

    # TopK / BottomK bucket analysis
    df = pd.DataFrame({"pred": pred_clean, "label": label_clean})
    bucket_results = []
    for date, group in df.groupby(level=0):
        if len(group) < topk * 2:
            continue
        sorted_g = group.sort_values("pred", ascending=False)
        top = sorted_g.head(topk)["label"].mean()
        bot = sorted_g.tail(topk)["label"].mean()
        bucket_results.append({
            "date": str(date),
            "top_return": float(top),
            "bot_return": float(bot),
            "spread": float(top - bot),
            "universe_return": float(group["label"].mean()),
        })

    br = pd.DataFrame(bucket_results)
    if not br.empty:
        top_mean = float(br["top_return"].mean())
        bot_mean = float(br["bot_return"].mean())
        spread_mean = float(br["spread"].mean())
        spread_pos_ratio = float((br["spread"] > 0).mean())
        excess_mean = float((br["top_return"] - br["universe_return"]).mean())
    else:
        top_mean = bot_mean = spread_mean = spread_pos_ratio = excess_mean = 0.0

    # Quality judgment
    quality = "normal"
    warnings = []
    if ic_mean < 0.02:
        quality = "weak"
        warnings.append(f"IC {ic_mean:.4f} < 0.02")
    elif ic_mean < 0.03:
        quality = "marginal"
        warnings.append(f"IC {ic_mean:.4f} < 0.03")
    if spread_mean <= 0:
        if quality != "weak":
            quality = "marginal"
        warnings.append(f"Top-Bottom spread {spread_mean:.6f} <= 0")

    result = {
        "ok": True,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": model_path,
        "test_start": test_start,
        "test_end": test_end,
        "label_expression": LABEL_EXPR,
        "prediction_horizon_days": PREDICTION_HORIZON_DAYS,
        "n_samples": n_samples,
        "n_dates": n_dates,
        "topk": topk,
        "quality": quality,
        "warnings": warnings,
        "metrics": {
            "ic_mean": round(ic_mean, 6),
            "ic_std": round(ic_std, 6),
            "icir": round(icir, 4),
            "rank_ic_mean": round(rank_ic_mean, 6),
            "rank_ic_pos_ratio": round(rank_ic_pos_ratio, 4),
            "long_short_return_mean": round(ls_mean, 6),
            "long_return_mean": round(long_mean, 6),
            f"top{topk}_return_mean": round(top_mean, 6),
            f"bot{topk}_return_mean": round(bot_mean, 6),
            f"top{topk}_bot{topk}_spread": round(spread_mean, 6),
            "spread_pos_ratio": round(spread_pos_ratio, 4),
            f"top{topk}_excess_vs_universe": round(excess_mean, 6),
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate LGB model quality")
    # cx round 23 E.P2 #6: --model-path defaults to None so production-safe
    # semantics (active-profile binary) kick in when omitted; --profile lets
    # the caller evaluate a specific candidate without aliasing through the
    # legacy lgb_model.pkl symlink.
    parser.add_argument(
        "--model-path", default=None,
        help=f"Override the model binary. Default: active-profile binary ({MODEL_PATH}).",
    )
    parser.add_argument(
        "--profile", default=None,
        help="Candidate profile name (e.g. xgb_242). Default: active profile.",
    )
    parser.add_argument("--universe", default="all")
    parser.add_argument("--min-predictions", type=int, default=LGB_MIN_PREDICTIONS)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # Resolve effective model path: explicit --model-path > profile binary
    # > active-profile default. load_production_model will refuse to fall
    # back to the live champion contract when model_path is custom.
    if args.model_path is not None:
        effective_model_path = args.model_path
    elif args.profile is not None:
        try:
            from config.production_features import production_model_filename
            effective_model_path = str(DATA_DIR / production_model_filename(args.profile))
        except Exception as exc:
            print(json.dumps({"ok": False, "error": f"profile resolution failed: {exc}"}))
            sys.exit(1)
    else:
        effective_model_path = MODEL_PATH

    result = evaluate(
        model_path=effective_model_path,
        universe=args.universe,
        min_predictions=args.min_predictions,
        test_days=args.test_days,
        topk=args.topk,
        profile=args.profile,
    )

    # Save latest
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = EVAL_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    os.replace(tmp, EVAL_PATH)

    # Append to history
    history = []
    if EVAL_HISTORY_PATH.exists():
        try:
            history = json.loads(EVAL_HISTORY_PATH.read_text())
        except Exception:
            history = []
    history.append(result)
    # Keep last 90 entries, atomic write
    history = history[-90:]
    hist_tmp = EVAL_HISTORY_PATH.with_suffix(".tmp")
    hist_tmp.write_text(json.dumps(history, ensure_ascii=False, indent=2))
    os.replace(hist_tmp, EVAL_HISTORY_PATH)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        m = result.get("metrics", {})
        print(f"Quality: {result.get('quality', '?')}")
        print(f"IC: {m.get('ic_mean', 0):.4f}  ICIR: {m.get('icir', 0):.4f}")
        print(f"RankIC: {m.get('rank_ic_mean', 0):.4f}  (>0 ratio: {m.get('rank_ic_pos_ratio', 0):.1%})")
        print(f"Top{args.topk} spread: {m.get(f'top{args.topk}_bot{args.topk}_spread', 0)*100:.3f}%")
        print(f"Top{args.topk} excess: {m.get(f'top{args.topk}_excess_vs_universe', 0)*100:.3f}%")
        if result.get("warnings"):
            for w in result["warnings"]:
                print(f"WARNING: {w}")

    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
