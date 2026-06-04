"""Brinson attribution: decompose portfolio return into allocation vs selection.

Runs fresh prediction on test period, then decomposes TopK excess return.

Saves to: data/storage/brinson_attribution.json
Usage:
    python scripts/run_brinson_attribution.py
"""
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
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("QLIB_DEBUG_SAFE", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
OUTPUT_PATH = DATA_DIR / "brinson_attribution.json"


def run_attribution():
    from config.qlib_runtime import init_qlib
    from config.settings import (
        LGB_INFERENCE_UNIVERSE,
        PREDICTION_HORIZON_DAYS,
        QLIB_PROVIDER_URI,
    )
    from models.portfolio_policy import sector_from_code

    # 2026-06-04 cx round 8 P1-3: use the production-safe loader so
    # Brinson attribution sees the SAME 242-dim feature shape as
    # the live champion. Pre-fix the script unpickled the 242-dim
    # model against a 158-dim Alpha158 dataset → silent default-leaf
    # predictions → garbage sector/timing attribution numbers.
    from models.production_inference import load_production_model

    label_expr = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"

    model_path = DATA_DIR / "lgb_model.pkl"
    if not model_path.exists():
        logger.error("Model not found")
        return None

    today = datetime.now()
    test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")

    logger.info(f"Loading dataset for {test_start} ~ {test_end}...")
    model, dataset = load_production_model(
        test_start, test_end,
        model_path=str(model_path),
        instruments=LGB_INFERENCE_UNIVERSE,
        label_expr=label_expr,
    )

    pred = model.predict(dataset=dataset)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")

    label = dataset.prepare("test", col_set="label")
    if isinstance(label, pd.DataFrame):
        label = label.iloc[:, 0]

    common = pred.index.intersection(label.index)
    pred_s = pred.loc[common, "score"] if "score" in pred.columns else pred.loc[common].iloc[:, 0]
    label_s = label.loc[common]
    mask = pred_s.notna() & label_s.notna() & np.isfinite(pred_s) & np.isfinite(label_s)
    pred_s = pred_s[mask]
    label_s = label_s[mask]
    logger.info(f"Aligned predictions/labels: {len(pred_s)}")

    top_k = 20
    dates = pred_s.index.get_level_values(0).unique().sort_values()
    daily_excess = []
    daily_allocation = []
    daily_selection = []
    daily_interaction = []

    for date in dates:
        try:
            day_pred = pred_s.xs(date, level=0)
            day_label = label_s.xs(date, level=0)
            day = pd.DataFrame({"pred": day_pred, "label": day_label}).dropna()
            if len(day) < top_k * 2:
                continue

            portfolio = day.sort_values("pred", ascending=False).head(top_k).copy()
            benchmark = day.copy()
            portfolio["sector"] = [sector_from_code(str(c)) for c in portfolio.index]
            benchmark["sector"] = [sector_from_code(str(c)) for c in benchmark.index]

            portfolio_ret = portfolio["label"].mean()
            benchmark_ret = benchmark["label"].mean()
            daily_excess.append(float(portfolio_ret - benchmark_ret))

            port_weight = portfolio.groupby("sector").size() / len(portfolio)
            bench_weight = benchmark.groupby("sector").size() / len(benchmark)
            port_return = portfolio.groupby("sector")["label"].mean()
            bench_return = benchmark.groupby("sector")["label"].mean()

            allocation = 0.0
            selection = 0.0
            interaction = 0.0
            for sector in sorted(set(port_weight.index) | set(bench_weight.index)):
                w_p = float(port_weight.get(sector, 0.0))
                w_b = float(bench_weight.get(sector, 0.0))
                r_b = float(bench_return.get(sector, 0.0))
                r_p = float(port_return.get(sector, r_b))
                allocation += (w_p - w_b) * r_b
                selection += w_b * (r_p - r_b)
                interaction += (w_p - w_b) * (r_p - r_b)

            daily_allocation.append(float(allocation))
            daily_selection.append(float(selection))
            daily_interaction.append(float(interaction))
        except Exception as exc:
            logger.debug(f"Attribution skipped for {date}: {exc}")

    if not daily_excess:
        logger.error("No attribution data")
        return None

    alloc_total = sum(daily_allocation)
    select_total = sum(daily_selection)
    interact_total = sum(daily_interaction)
    abs_total = abs(alloc_total) + abs(select_total) + abs(interact_total)

    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "test_period": f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}",
        "n_days": len(daily_excess),
        "top_k": top_k,
        "cumulative_excess_pct": round(sum(daily_excess) * 100, 2),
        "allocation_effect_pct": round(alloc_total * 100, 4),
        "selection_effect_pct": round(select_total * 100, 4),
        "interaction_effect_pct": round(interact_total * 100, 4),
        "explained_excess_pct": round((alloc_total + select_total + interact_total) * 100, 4),
        "allocation_share": round(abs(alloc_total) / abs_total * 100, 1) if abs_total > 0 else 50,
        "selection_share": round(abs(select_total) / abs_total * 100, 1) if abs_total > 0 else 50,
        "interaction_share": round(abs(interact_total) / abs_total * 100, 1) if abs_total > 0 else 0,
        "avg_daily_excess_pct": round(np.mean(daily_excess) * 100, 4),
        "label": label_expr,
    }

    if result["selection_share"] > 60:
        result["recommendation"] = "Alpha 主要来自选股，模型个股区分能力强"
    elif result["allocation_share"] > 60:
        result["recommendation"] = "Alpha 主要来自板块配置，建议检查行业暴露"
    else:
        result["recommendation"] = "选股、板块配置与交互项贡献较均衡"

    return result


def main():
    result = run_attribution()
    if result is None:
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    os.replace(tmp, OUTPUT_PATH)

    print(f"Brinson Attribution ({result['test_period']})")
    print(f"  Cumulative excess: {result['cumulative_excess_pct']:+.2f}%")
    print(f"  Allocation: {result['allocation_effect_pct']:+.4f}% ({result['allocation_share']:.0f}%)")
    print(f"  Selection:  {result['selection_effect_pct']:+.4f}% ({result['selection_share']:.0f}%)")
    print(f"  Interaction:{result['interaction_effect_pct']:+.4f}% ({result['interaction_share']:.0f}%)")
    print(f"  {result['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
