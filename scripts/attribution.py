"""Simple Brinson-style attribution: industry allocation vs stock selection.

Decomposes TopK portfolio return into:
- Allocation effect: did we pick the right industries?
- Selection effect: did we pick the right stocks within industries?

Usage:
    python scripts/attribution.py
    python scripts/attribution.py --json
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
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config.settings import PREDICTION_HORIZON_DAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
MODEL_PATH = str(DATA_DIR / "lgb_model.pkl")
ATTRIBUTION_PATH = DATA_DIR / "lgb_attribution_latest.json"


def get_industry_map() -> dict:
    """Get stock -> industry mapping from AKShare."""
    try:
        import akshare as ak
        df = ak.stock_board_industry_name_em()
        if df is None or df.empty:
            return {}

        industry_map = {}
        for _, row in df.iterrows():
            industry = row.get("板块名称", "")
            try:
                cons = ak.stock_board_industry_cons_em(symbol=industry)
                if cons is not None:
                    for _, stock in cons.iterrows():
                        code = str(stock.get("代码", "")).zfill(6)
                        prefix = "SH" if code.startswith("6") else "SZ"
                        industry_map[f"{prefix}{code}"] = industry
            except Exception:
                pass
        return industry_map
    except Exception:
        return {}


def simple_attribution(topk: int = 20, test_days: int = 30) -> dict:
    """Run simplified Brinson attribution."""
    import qlib
    from qlib.constant import REG_CN
    from qlib.utils import init_instance_by_config

    qlib.init(provider_uri=QLIB_DATA, region=REG_CN)

    today = datetime.now()
    test_start = (today - timedelta(days=test_days)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")

    handler_config = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": test_start,
            "end_time": test_end,
            "instruments": "all",
            "label": [f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"],
        },
    }
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {"test": (test_start, test_end)},
        },
    }

    dataset = init_instance_by_config(dataset_config)

    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not found"}

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    pred = model.predict(dataset=dataset)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")

    label = dataset.prepare("test", col_set="label")
    if isinstance(label, pd.DataFrame):
        label = label.iloc[:, 0]

    # Build industry map (simplified: use code prefix as sector proxy)
    # SH60xxxx = 沪市主板, SZ00xxxx = 深市主板, SZ30xxxx = 创业板, SH68xxxx = 科创板
    def sector_from_code(code):
        code = str(code).upper()
        if code.startswith("SH68"):
            return "科创板"
        elif code.startswith("SH60"):
            return "沪市主板"
        elif code.startswith("SZ00"):
            return "深市主板"
        elif code.startswith("SZ30"):
            return "创业板"
        elif code.startswith("SZ002"):
            return "中小板"
        else:
            return "其他"

    common = pred.index.intersection(label.index)
    df = pd.DataFrame({
        "pred": pred.loc[common, "score"] if "score" in pred.columns else pred.loc[common].iloc[:, 0],
        "label": label.loc[common],
    })
    df = df.dropna()

    dates = sorted(df.index.get_level_values(0).unique())

    allocation_effects = []
    selection_effects = []

    for date in dates:
        day = df.loc[date].copy()
        if len(day) < topk * 2:
            continue

        # Instrument level
        inst_level = day.index
        day["sector"] = [sector_from_code(str(c)) for c in inst_level]

        # Portfolio: top K by prediction
        day_sorted = day.sort_values("pred", ascending=False)
        portfolio = day_sorted.head(topk)
        benchmark = day  # equal-weight universe

        # Allocation effect: sector weight diff × sector benchmark return
        port_sector_weight = portfolio.groupby("sector").size() / len(portfolio)
        bench_sector_weight = benchmark.groupby("sector").size() / len(benchmark)
        bench_sector_return = benchmark.groupby("sector")["label"].mean()

        alloc = 0.0
        for sector in set(port_sector_weight.index) | set(bench_sector_weight.index):
            w_p = port_sector_weight.get(sector, 0)
            w_b = bench_sector_weight.get(sector, 0)
            r_b = bench_sector_return.get(sector, 0)
            alloc += (w_p - w_b) * r_b

        # Selection effect: benchmark weight × (portfolio return - benchmark return) per sector
        port_sector_return = portfolio.groupby("sector")["label"].mean()
        select = 0.0
        for sector in port_sector_weight.index:
            w_b = bench_sector_weight.get(sector, 0)
            r_p = port_sector_return.get(sector, 0)
            r_b = bench_sector_return.get(sector, 0)
            select += w_b * (r_p - r_b)

        allocation_effects.append(alloc)
        selection_effects.append(select)

    if not allocation_effects:
        return {"error": "No valid dates for attribution"}

    total_alloc = float(np.mean(allocation_effects))
    total_select = float(np.mean(selection_effects))
    total_excess = total_alloc + total_select

    result = {
        "attribution_at": datetime.now().isoformat(timespec="seconds"),
        "test_start": test_start,
        "test_end": test_end,
        "n_dates": len(allocation_effects),
        "topk": topk,
        "allocation_effect_pct": round(total_alloc * 100, 4),
        "selection_effect_pct": round(total_select * 100, 4),
        "total_excess_pct": round(total_excess * 100, 4),
        "allocation_share": round(total_alloc / (abs(total_excess) + 1e-8) * 100, 1),
        "selection_share": round(total_select / (abs(total_excess) + 1e-8) * 100, 1),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Brinson attribution")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = simple_attribution(topk=args.topk, test_days=args.test_days)

    # Save
    ATTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = ATTRIBUTION_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    os.replace(tmp, ATTRIBUTION_PATH)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Brinson Attribution ({result['n_dates']} days, Top{args.topk})")
            print(f"  行业配置贡献: {result['allocation_effect_pct']:+.4f}% ({result['allocation_share']:.0f}%)")
            print(f"  个股选择贡献: {result['selection_effect_pct']:+.4f}% ({result['selection_share']:.0f}%)")
            print(f"  总超额收益:   {result['total_excess_pct']:+.4f}%")

    sys.exit(0 if "error" not in result else 1)


if __name__ == "__main__":
    main()
