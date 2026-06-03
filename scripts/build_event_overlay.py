"""Build event overlay score — CX-approved gated approach.

Rules (from CX review):
  1. Raw LLM impact as prior (not calibrated — in-sample calibration degrades RICIR)
  2. Filter: other/routine_announcement → impact = 0
  3. Unstable buckets (earnings_negative, industry_trend_positive, product_launch,
     analyst_upgrade) → weight = 0.2 (not flipped, just dampened)
  4. Only apply to Top500/Top1000 liquid stocks
  5. final_score = zscore(xgb_score) + alpha * zscore(event_alpha)

Usage:
    python scripts/build_event_overlay.py --date 2026-05-22 --alpha 1.0
    python scripts/build_event_overlay.py --backtest  # test on historical data
"""
import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"

# Noise event types → impact forced to 0
NOISE_TYPES = {"other", "routine_announcement", "reorganize"}

# Unstable buckets (LLM direction may be wrong) → weight = 0.2
UNSTABLE_TYPES = {"earnings_negative", "industry_trend_positive",
                  "product_launch", "analyst_upgrade"}

UNSTABLE_WEIGHT = 0.2


def _get_liquid_pool(top_n: int = 500) -> set:
    """Get top N liquid stocks by average daily amount from feature cache."""
    try:
        cache = pd.read_parquet(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet",
                                columns=["amount_raw"])
        # Use latest date
        dates = cache.index.get_level_values(0)
        latest = dates.max()
        day_amount = cache.loc[latest, "amount_raw"].dropna()
        top = day_amount.nlargest(top_n).index
        return set(str(x) for x in top)
    except Exception:
        # Fallback: return empty (overlay all stocks)
        return set()


def load_events_for_date(date: str) -> dict[str, float]:
    """Load gated event alpha for a date.

    Returns {stock_code: gated_impact}
    """
    events_path = DATA_DIR / "llm_events" / f"{date}.jsonl"
    if not events_path.exists():
        return {}

    stock_impacts = defaultdict(list)
    for line in open(events_path):
        e = json.loads(line)
        etype = e.get("event_type", "other")
        impact = e.get("impact_1d", 0)
        code = e.get("stock_code", "")
        if not code:
            continue

        # Gate 1: noise types → 0
        if etype in NOISE_TYPES:
            continue

        # Gate 2: unstable types → dampen
        if etype in UNSTABLE_TYPES:
            impact *= UNSTABLE_WEIGHT

        stock_impacts[code].append(impact)

    # Average per stock
    return {code: np.mean(vals) for code, vals in stock_impacts.items() if vals}


def apply_overlay(xgb_predictions: dict, event_alphas: dict,
                  alpha: float = 1.0, top_n_liquid: int = 500) -> dict:
    """Apply event overlay to XGB predictions.

    final_score = zscore(xgb) + alpha * zscore(event)
    Only overlays stocks in top_n_liquid by |xgb_score| (proxy for liquidity).

    Args:
        xgb_predictions: {code: xgb_score}
        event_alphas: {code: gated_event_impact}
        alpha: blending weight
        top_n_liquid: only overlay top N stocks

    Returns:
        {code: final_score}
    """
    if not event_alphas or alpha == 0:
        return dict(xgb_predictions)

    # Determine liquid pool by actual trading volume (not XGB score)
    # CX fix: abs(xgb_score) is NOT liquidity
    liquid_pool = _get_liquid_pool(top_n_liquid)

    # Z-score XGB predictions
    xgb_vals = np.array(list(xgb_predictions.values()))
    xgb_mean, xgb_std = np.mean(xgb_vals), np.std(xgb_vals) + 1e-8

    # Z-score event alphas (only non-zero)
    event_vals = [v for v in event_alphas.values() if v != 0]
    if not event_vals:
        return dict(xgb_predictions)
    evt_mean, evt_std = np.mean(event_vals), np.std(event_vals) + 1e-8

    # Blend
    result = {}
    n_overlaid = 0
    for code, xgb_score in xgb_predictions.items():
        xgb_z = (xgb_score - xgb_mean) / xgb_std

        # Only overlay if stock is in liquid pool AND has event
        code_lower = code.lower()  # sh600519
        code_6 = code_lower[2:] if len(code_lower) > 2 else code_lower  # 600519
        event_impact = event_alphas.get(code_6, 0)

        in_pool = (not liquid_pool) or (code_lower in liquid_pool)
        if event_impact != 0 and in_pool:
            evt_z = (event_impact - evt_mean) / evt_std
            result[code] = xgb_z + alpha * evt_z
            n_overlaid += 1
        else:
            result[code] = xgb_z

    logger.info(f"  Overlay: {n_overlaid} stocks adjusted out of {len(result)}")
    return result


def exploratory_replay_overlay():
    """Exploratory replay of overlay on historical event dates.

    THIS IS NOT A BACKTEST. The output of this function is NOT valid
    promotion evidence. Per code-review P1 2026-05-31 (cx finding): the
    previous name `backtest_overlay` + CLI flag `--backtest` led readers
    to assume PIT-safe backtest output, but it isn't.

    What it actually does: uses a single snapshot of the current XGB
    predictions (lgb_latest_predictions.json) across ALL historical event
    dates. The XGB scores were trained on data through "today", so for
    any earlier event date the scores contain future-training info
    relative to that date. Useful as "current model × historical events"
    sanity exploration only.

    For valid historical evidence each date must use as-of predictions
    from the rolling training pipeline (per-date prediction artifacts via
    tracker/artifact_contract.py). The full rolling train → predict →
    overlay pipeline does that work.

    Promotion gate (`tracker/promotion_gate.py`) MUST NOT accept the
    output of this function as evidence — it's exploratory only.

    TODO: Once artifact_contract stores per-split pred.pkl, refactor this
    to load date-appropriate predictions for each replay day.
    """
    from config.qlib_runtime import init_qlib
    from qlib.data import D
    from scipy import stats

    init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

    events_dir = DATA_DIR / "llm_events"
    dates = sorted(f.stem for f in events_dir.glob("*.jsonl"))
    logger.warning(
        "============================================================\n"
        "  EXPLORATORY REPLAY — NOT A PIT BACKTEST                    \n"
        "  Using SINGLE latest XGB predictions across ALL %d dates    \n"
        "  Output is NOT valid promotion evidence (cx review P1)      \n"
        "============================================================",
        len(dates),
    )

    # Load XGB predictions — SAME model for all dates (exploratory only)
    xgb_preds = json.loads(open(DATA_DIR / "lgb_latest_predictions.json").read())["predictions"]

    # Load returns
    all_qlib = [v.lower() for v in xgb_preds.keys()]
    ret = D.features(all_qlib, ["Ref($close, -1) / $close - 1"],
                     start_time="2026-04-25", end_time="2026-05-22")
    ret.columns = ["ret"]
    ret_lookup = {}
    for idx, row in ret.iterrows():
        if np.isfinite(row["ret"]):
            ret_lookup[(idx[0], idx[1].strftime("%Y-%m-%d"))] = float(row["ret"])

    # Test different alpha values
    for alpha_val in [0, 0.5, 1.0, 2.0]:
        rics = []
        for date in dates:
            event_alphas = load_events_for_date(date)
            if not event_alphas:
                continue

            blended = apply_overlay(xgb_preds, event_alphas,
                                    alpha=alpha_val, top_n_liquid=1000)

            # Match with returns
            preds, actuals = [], []
            for code, score in blended.items():
                qlib = code.lower()
                if (qlib, date) in ret_lookup:
                    preds.append(score)
                    actuals.append(ret_lookup[(qlib, date)])

            if len(preds) >= 500:
                ric = stats.spearmanr(preds, actuals).statistic
                if np.isfinite(ric):
                    rics.append(ric)

        if rics:
            avg = np.mean(rics)
            ricir = avg / (np.std(rics) + 1e-8)
            pos = np.mean([r > 0 for r in rics]) * 100
            logger.info(f"  alpha={alpha_val:<4} AvgRIC={avg:+.4f} RICIR={ricir:+.3f} "
                        f"RIC>0={pos:.0f}% ({len(rics)} days)")


def main():
    parser = argparse.ArgumentParser(description="Event overlay")
    parser.add_argument("--date", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--exploratory-replay", action="store_true",
        help=(
            "Run exploratory replay (NOT a valid PIT backtest). Uses "
            "single latest XGB predictions across all historical event "
            "dates. Output is NOT valid promotion evidence."
        ),
    )
    # Backwards-compat: --backtest is the historical flag name. Per
    # code-review P1 2026-05-31 the name was misleading. Accept it but
    # warn loudly and route to the same exploratory_replay_overlay.
    parser.add_argument(
        "--backtest", action="store_true",
        help=(
            "DEPRECATED: use --exploratory-replay instead. The old name "
            "misleadingly suggested PIT-safe backtest output. Output is "
            "NOT valid promotion evidence."
        ),
    )
    args = parser.parse_args()

    if args.backtest and not args.exploratory_replay:
        logger.warning(
            "--backtest is deprecated (cx review P1 2026-05-31): the name "
            "misleadingly suggests PIT-safe backtest, but the function only "
            "does exploratory replay. Use --exploratory-replay going forward."
        )
    if args.backtest or args.exploratory_replay:
        exploratory_replay_overlay()
        return

    date = args.date or datetime.now().strftime("%Y-%m-%d")
    event_alphas = load_events_for_date(date)
    logger.info(f"Date {date}: {len(event_alphas)} gated event stocks")

    # Load XGB predictions
    xgb_preds = json.loads(open(DATA_DIR / "lgb_latest_predictions.json").read())["predictions"]
    blended = apply_overlay(xgb_preds, event_alphas, alpha=args.alpha)
    logger.info(f"Blended: {len(blended)} stocks")


if __name__ == "__main__":
    main()
