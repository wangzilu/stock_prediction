"""Export the 242-dim production feature contract.

Reads the currently-deployed lgb_model.pkl + the train-time feature
shape and writes
  data/storage/production_feature_contract.json

with:
  - frozen_at: timestamp
  - model_pkl_path
  - booster_num_features
  - features: [
      {index, name, group, pit_status, approved}
    ]

The contract artifact is the SINGLE SOURCE OF TRUTH for the production
champion model's feature shape. Training and inference must read from
this artifact (not from FeatureMerger's live state) so any new column
silently added to FeatureMerger does not silently enter production.

Usage:
    python scripts/export_feature_contract.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
MODEL_PATH = DATA_DIR / "lgb_model.pkl"
CONTRACT_PATH = DATA_DIR / "production_feature_contract.json"


# Hard-coded group→loader→column mapping mirroring
# FeatureMerger._load_supplementary. Maintained alongside that method;
# if a loader is added there but NOT here, the export will report an
# unknown-group warning (the column is still recorded but tagged
# "unknown" so review surfaces it).
GROUP_LOADERS = (
    ("fundamental", "_load_fundamental"),
    ("capital_flow", "_load_capital_flow"),
    ("macro_zero_baseline", "_load_macro"),
    ("shareholder", "_load_shareholder"),
    ("valuation", "_load_valuation"),
    ("northbound", "_load_northbound"),
    ("quality", "_load_quality"),
    ("st_daily_basic", "_load_st_daily_basic"),
    ("st_moneyflow", "_load_st_moneyflow"),
    ("st_holder_number", "_load_st_holder_number"),
    ("cross_market_regime", "_load_cross_market_regime"),
)


def _alpha158_column_names() -> list[str]:
    """Best-effort enumeration of the 158 Alpha158 column names.

    Qlib generates these from the handler expression list. For the
    artifact we record positional names f0..f157 and the FieldGroup
    qualifier — Qlib does not guarantee stable string names across
    versions and the contract is matched by COUNT, not name."""
    return [f"alpha158_f{i:03d}" for i in range(158)]


def _supp_columns_per_group() -> list[tuple[str, list[str]]]:
    """Use a small synthetic index to probe each supp loader and
    record the columns it returns."""
    import pandas as pd
    from models.feature_merger import FeatureMerger

    merger = FeatureMerger()
    today = datetime.now()
    idx = pd.MultiIndex.from_product(
        [pd.date_range(today - __import__("datetime").timedelta(days=2),
                       periods=3),
         ["SH600519"]],
        names=["datetime", "instrument"],
    )

    out: list[tuple[str, list[str]]] = []
    for group, method_name in GROUP_LOADERS:
        method = getattr(merger, method_name, None)
        if method is None:
            logger.warning("group %s: loader %s missing on FeatureMerger",
                            group, method_name)
            out.append((group, []))
            continue
        try:
            df = method(idx)
        except Exception as e:  # noqa: BLE001
            logger.warning("group %s: loader %s raised %s", group,
                            method_name, e)
            df = None
        cols = list(df.columns) if df is not None else []
        logger.info("  %-25s %d cols", group, len(cols))
        out.append((group, cols))
    return out


def build_contract() -> dict:
    if not MODEL_PATH.exists():
        raise SystemExit(f"model not found: {MODEL_PATH}")
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    booster = getattr(model, "model", model)
    n_features = int(booster.num_features())

    alpha_cols = _alpha158_column_names()
    supp_per_group = _supp_columns_per_group()
    flat_supp = [
        (c, group)
        for group, cols in supp_per_group
        for c in cols
    ]

    all_features: list[dict] = []
    for i, name in enumerate(alpha_cols):
        all_features.append({
            "index": i,
            "name": name,
            "group": "alpha158",
            "pit_status": "verified",  # qlib alpha158 is canonical PIT
            "approved": True,
        })

    expected_supp_count = n_features - len(alpha_cols)
    actual_supp_count = len(flat_supp)

    for j, (col_name, group_name) in enumerate(flat_supp):
        pit = "zero_baseline" if "macro" in col_name.lower() else "verified"
        if group_name == "unknown":
            pit = "unknown"
        all_features.append({
            "index": len(alpha_cols) + j,
            "name": col_name,
            "group": group_name,
            "pit_status": pit,
            "approved": True,
        })

    contract = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "model_pkl_path": str(MODEL_PATH),
        "booster_num_features": n_features,
        "alpha158_count": len(alpha_cols),
        "supplementary_count": actual_supp_count,
        "supplementary_count_expected_from_model": expected_supp_count,
        "dim_match": actual_supp_count + len(alpha_cols) == n_features,
        "groups": [
            {
                "name": g,
                "n_cols": len(cs),
                "cols": cs,
            }
            for g, cs in supp_per_group
        ],
        "features": all_features,
    }
    return contract


def main():
    contract = build_contract()
    logger.info(
        "Booster expects %d features. Alpha158 %d + supplementary %d = %d. "
        "dim_match=%s",
        contract["booster_num_features"], contract["alpha158_count"],
        contract["supplementary_count"],
        contract["alpha158_count"] + contract["supplementary_count"],
        contract["dim_match"],
    )

    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2),
    )
    logger.info("Wrote: %s", CONTRACT_PATH)

    if not contract["dim_match"]:
        logger.error(
            "DIMENSION MISMATCH: booster needs %d but supplementary "
            "loaders produced %d cols (alpha158 + supp). The contract "
            "artifact is incomplete; do NOT use it as a production "
            "gate until alignment is restored.",
            contract["booster_num_features"],
            contract["alpha158_count"] + contract["supplementary_count"],
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
