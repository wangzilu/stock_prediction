"""Champion feature path configuration and promotion pipeline.

⚠️  REGISTRY SPLIT WARNING (cx round 10, 2026-06-04) ⚠️

This module is the RESEARCH-LINE registry — it describes the
``xgb_174`` champion profile (Alpha158 + 16 qlib_custom + a few
one-off cols, 205 columns total). It is NOT the runtime production
contract any more.

What actually runs in production today:
  - lgb_model.pkl is 242-dim (Alpha158 + 11 supplementary loader
    groups). See ``config.production_features.PRODUCTION_MODEL_PROFILE``
    and ``PRODUCTION_SUPPLEMENTARY_GROUPS``.
  - The feature contract artifact at
    ``data/storage/production_feature_contract.json`` is the
    runtime source of truth.

How the split happened:
  - commit 95cd256 (2026-05-12) opened the ``_load_supplementary``
    暗道 in ``scripts/train_lgb.py`` without an allowlist.
  - The first weekly retrain after that (~2026-05-23) wrote a
    242-dim model over the previous ``xgb_174`` champion binary,
    but this CHAMPION_PATH dict still names ``xgb_174``.
  - Task #112 tracks the formal resolution (retrain xgb_174 +
    24-split + cost-adjusted backtest → choose default).

Until #112 lands, treat this module as the SHADOW / RESEARCH
description of what xgb_174 was. Do not use CHAMPION_PATH as the
truth for inference / training without first cross-checking against
``config.production_features``.

Defines exactly what the champion model uses, what supplement data sources
are available but not yet promoted, and the official pipeline for promoting
new factors into the champion cache.

Usage:
    from config.feature_path import (
        CHAMPION_PATH, SUPPLEMENT_SOURCES, OVERLAY_SOURCES,
        PROMOTION_PIPELINE, get_champion_features, get_feature_group,
        is_in_champion, get_promotion_status,
    )
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Feature group definitions (ordered as they appear in the cache)
# ---------------------------------------------------------------------------

# Qlib Alpha158 OHLCV-derived features (columns 0-157)
_ALPHA158_COLS = [
    "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",
    "OPEN0", "HIGH0", "LOW0", "VWAP0",
    "ROC5", "ROC10", "ROC20", "ROC30", "ROC60",
    "MA5", "MA10", "MA20", "MA30", "MA60",
    "STD5", "STD10", "STD20", "STD30", "STD60",
    "BETA5", "BETA10", "BETA20", "BETA30", "BETA60",
    "RSQR5", "RSQR10", "RSQR20", "RSQR30", "RSQR60",
    "RESI5", "RESI10", "RESI20", "RESI30", "RESI60",
    "MAX5", "MAX10", "MAX20", "MAX30", "MAX60",
    "MIN5", "MIN10", "MIN20", "MIN30", "MIN60",
    "QTLU5", "QTLU10", "QTLU20", "QTLU30", "QTLU60",
    "QTLD5", "QTLD10", "QTLD20", "QTLD30", "QTLD60",
    "RANK5", "RANK10", "RANK20", "RANK30", "RANK60",
    "RSV5", "RSV10", "RSV20", "RSV30", "RSV60",
    "IMAX5", "IMAX10", "IMAX20", "IMAX30", "IMAX60",
    "IMIN5", "IMIN10", "IMIN20", "IMIN30", "IMIN60",
    "IMXD5", "IMXD10", "IMXD20", "IMXD30", "IMXD60",
    "CORR5", "CORR10", "CORR20", "CORR30", "CORR60",
    "CORD5", "CORD10", "CORD20", "CORD30", "CORD60",
    "CNTP5", "CNTP10", "CNTP20", "CNTP30", "CNTP60",
    "CNTN5", "CNTN10", "CNTN20", "CNTN30", "CNTN60",
    "CNTD5", "CNTD10", "CNTD20", "CNTD30", "CNTD60",
    "SUMP5", "SUMP10", "SUMP20", "SUMP30", "SUMP60",
    "SUMN5", "SUMN10", "SUMN20", "SUMN30", "SUMN60",
    "SUMD5", "SUMD10", "SUMD20", "SUMD30", "SUMD60",
    "VMA5", "VMA10", "VMA20", "VMA30", "VMA60",
    "VSTD5", "VSTD10", "VSTD20", "VSTD30", "VSTD60",
    "WVMA5", "WVMA10", "WVMA20", "WVMA30", "WVMA60",
    "VSUMP5", "VSUMP10", "VSUMP20", "VSUMP30", "VSUMP60",
    "VSUMN5", "VSUMN10", "VSUMN20", "VSUMN30", "VSUMN60",
    "VSUMD5", "VSUMD10", "VSUMD20", "VSUMD30", "VSUMD60",
]

# Capital flow (fund_flow_history.parquet, 3 dims)
_CAPITAL_FLOW_COLS = [
    "flow_net_mf_latest",
    "flow_net_mf_5d",
    "flow_net_mf_20d_avg",
]

# Qlib custom expressions (PE/PB/turnover derived, 13 dims)
_CUSTOM_COLS = [
    "pe", "pb", "turn_raw", "amount_raw",
    "pe_mom20", "pb_mom20", "turn_anom20", "turn_anom60",
    "amount_anom20", "turn_vol20", "ep", "bp", "price_pos20",
]

# Holder count (1 dim)
_HOLDER_COLS = [
    "holder_num",
]

# Cross-market regime signals: HSI, HSTECH, NASDAQ (9 each = 27 dims)
_CROSS_MARKET_REGIME_COLS = [
    "hsi_ret1d", "hsi_ret5d", "hsi_ret20d",
    "hsi_vol5d", "hsi_vol20d",
    "hsi_mom5d", "hsi_mom20d",
    "hsi_up_ratio_10d", "hsi_dd20d",
    "hstech_ret1d", "hstech_ret5d", "hstech_ret20d",
    "hstech_vol5d", "hstech_vol20d",
    "hstech_mom5d", "hstech_mom20d",
    "hstech_up_ratio_10d", "hstech_dd20d",
    "nasdaq_ret1d", "nasdaq_ret5d", "nasdaq_ret20d",
    "nasdaq_vol5d", "nasdaq_vol20d",
    "nasdaq_mom5d", "nasdaq_mom20d",
    "nasdaq_up_ratio_10d", "nasdaq_dd20d",
]

# MA auxiliary (for timing backtest, not XGB input, 3 dims)
_MA_AUXILIARY_COLS = [
    "_close",
    "_ma5",
    "_ma20",
]

# Build column-to-group lookup
_COL_TO_GROUP: dict[str, str] = {}
for _col in _ALPHA158_COLS:
    _COL_TO_GROUP[_col] = "alpha158"
for _col in _CAPITAL_FLOW_COLS:
    _COL_TO_GROUP[_col] = "capital_flow"
for _col in _CUSTOM_COLS:
    _COL_TO_GROUP[_col] = "custom"
for _col in _HOLDER_COLS:
    _COL_TO_GROUP[_col] = "holder"
for _col in _CROSS_MARKET_REGIME_COLS:
    _COL_TO_GROUP[_col] = "cross_market_regime"
for _col in _MA_AUXILIARY_COLS:
    _COL_TO_GROUP[_col] = "ma_auxiliary"

# All champion feature columns (order matches cache)
_ALL_CHAMPION_COLS = (
    _ALPHA158_COLS
    + _CAPITAL_FLOW_COLS
    + _CUSTOM_COLS
    + _HOLDER_COLS
    + _CROSS_MARKET_REGIME_COLS
    + _MA_AUXILIARY_COLS
)

# ---------------------------------------------------------------------------
# 1. CHAMPION_PATH — the official champion configuration
# ---------------------------------------------------------------------------

CHAMPION_PATH = {
    "name": "xgb_174",
    "cache_file": "feature_cache_174_holder_regime_ma.parquet",
    "builder": "models.feature_pipeline.prepare_features_174",
    "feature_groups": {
        "alpha158": len(_ALPHA158_COLS),
        "capital_flow": len(_CAPITAL_FLOW_COLS),
        "custom": len(_CUSTOM_COLS),
        "holder": len(_HOLDER_COLS),
        "cross_market_regime": len(_CROSS_MARKET_REGIME_COLS),
        "ma_auxiliary": len(_MA_AUXILIARY_COLS),
    },
    "total_features": len(_ALL_CHAMPION_COLS),
    "label": "__label_5d",
    "pnl_col": "__pnl_return_1d",
    "status": "champion",
}

# Sanity check at import time
assert CHAMPION_PATH["total_features"] == 205, (
    f"Expected 205 champion features, got {CHAMPION_PATH['total_features']}"
)
assert sum(CHAMPION_PATH["feature_groups"].values()) == 205

# ---------------------------------------------------------------------------
# 2. SUPPLEMENT_SOURCES — available but non-champion data sources
# ---------------------------------------------------------------------------

SUPPLEMENT_SOURCES = {
    "st_daily_basic": {
        "loader": "feature_merger._load_st_daily_basic",
        "parquet": "st_daily_basic.parquet",
        "status": "available_not_promoted",
        "pit_safe": True,
        "notes": "PE/PB/PS/turnover/MV daily time-series. BDay(1) lag enforced.",
    },
    "st_moneyflow": {
        "loader": "feature_merger._load_st_moneyflow",
        "parquet": "st_moneyflow.parquet",
        "status": "failed_gate",
        "pit_safe": True,
        "notes": ("Per-stock capital flow (buy/sell by size). BDay(1) lag enforced. "
                  "All variants showed negative IC in tearsheet. Likely contrarian signal "
                  "or size effect — retail buying = negative alpha. Not promoted."),
    },
    "northbound": {
        "loader": "feature_merger._load_northbound",
        "parquet": "northbound_history.parquet",
        "status": "available_not_promoted",
        "pit_safe": True,
        "notes": "Northbound per-stock holdings (vol, ratio). BDay(1) lag enforced.",
    },
    "fundamental_quality": {
        "loader": "feature_merger._load_quality",
        "parquet": "fundamental_quality.parquet",
        "status": "available_not_promoted",
        "pit_safe": True,
        "notes": "ROE/margins/growth from financial reports via ann_date.",
    },
    "fundamental_valuation": {
        "loader": "feature_merger._load_valuation",
        "parquet": "fundamental_valuation.parquet",
        "status": "available_not_promoted",
        "pit_safe": True,
        "notes": "PE/PB/PS valuation features from financial reports via ann_date.",
    },
    "shareholder": {
        "loader": "feature_merger._load_shareholder",
        "parquet": "shareholder_features.parquet",
        "status": "available_not_promoted",
        "pit_safe": True,
        "notes": "Shareholder concentration metrics from financial reports.",
    },
    "macro": {
        "loader": "feature_merger._load_macro",
        "parquet": "macro_features.parquet",
        "status": "available_not_promoted",
        "pit_safe": False,
        "notes": "Macro features (currently single-row broadcast, PIT-unsafe).",
    },
}

# ---------------------------------------------------------------------------
# 3. OVERLAY_SOURCES — factors for overlay/rerank, NOT XGB input
# ---------------------------------------------------------------------------

OVERLAY_SOURCES = {
    "llm_event": {
        "status": "shadow",
        "usage": "overlay_rerank",
        "notes": "LLM-parsed news events. Only 18 days of data (2026-04-27+). PIT-unsafe.",
    },
    "guba_sentiment": {
        "status": "insufficient_data",
        "usage": "overlay_rerank",
        "notes": "Guba popularity ranking. Only 1 file collected. PIT-unsafe.",
    },
    "regime_controller": {
        "status": "active",
        "usage": "overlay_gate",
        "notes": "Composite regime score from macro + micro signals. Used as position gate.",
    },
    "holder_decrease": {
        "status": "research_only",
        "usage": "overlay_rerank",
        "notes": ("24-split rolling gate PASS but marginal. Regime-weighted overlay FAIL. "
                  "Keep as overlay candidate — do not promote to champion XGB features."),
    },
}

# ---------------------------------------------------------------------------
# 4. PROMOTION_PIPELINE — steps for a factor to go from candidate to champion
# ---------------------------------------------------------------------------

PROMOTION_PIPELINE = [
    "candidate_factors/ registration",
    "Alpha Factory tearsheet (IC/RankIC/spread/coverage/negative_control)",
    "Residual IC vs champion (marginal value > 0.005)",
    "12+ splits positive delta",
    "Supplement ablation (add to cache, retrain, compare)",
    "24-split rolling gate",
    "Shadow 20 paper days",
    "Champion cache rebuild",
]

# Track where each candidate currently sits in the pipeline.
# Keys are factor source names, values are the step index (0-based) they have
# completed, or -1 if not yet registered.
_CANDIDATE_STATUS: dict[str, int] = {
    # Supplement sources that have been through tearsheet but not further
    "st_daily_basic": 1,       # tearsheet done, residual IC pending
    "st_moneyflow": -1,        # FAILED: all variants negative IC, contrarian/size effect
    "northbound": 1,           # tearsheet done, residual IC pending
    "fundamental_quality": 0,  # registered, tearsheet pending
    "fundamental_valuation": 0,  # registered, tearsheet pending
    "shareholder": 0,          # registered, tearsheet pending
    "macro": -1,               # PIT-unsafe, not registered
    "holder_decrease": 4,      # 24-split marginal, overlay candidate only (research_only)
}


# ---------------------------------------------------------------------------
# 5. Utility functions
# ---------------------------------------------------------------------------

def get_champion_features() -> list[str]:
    """Return the ordered list of feature column names from the champion cache.

    Excludes meta columns (__label_5d, __pnl_return_1d, datetime, instrument).
    """
    return list(_ALL_CHAMPION_COLS)


def get_feature_group(col_name: str) -> str:
    """Return which feature group a column belongs to.

    Raises KeyError if the column is not in the champion feature set.
    """
    if col_name not in _COL_TO_GROUP:
        raise KeyError(
            f"Column '{col_name}' is not in the champion feature set. "
            f"Use is_in_champion() to check first."
        )
    return _COL_TO_GROUP[col_name]


def is_in_champion(col_name: str) -> bool:
    """Check whether a column name is part of the champion feature set."""
    return col_name in _COL_TO_GROUP


def get_promotion_status(factor_name: str) -> str:
    """Return a human-readable promotion status for a factor.

    Args:
        factor_name: Name of the factor/source (e.g. "st_daily_basic").

    Returns:
        A string describing where in the promotion pipeline the factor is,
        or "champion" if it is already in the champion set.
    """
    # Check if it is already champion
    if factor_name in CHAMPION_PATH["feature_groups"]:
        return "champion"

    # Check overlay sources
    if factor_name in OVERLAY_SOURCES:
        src = OVERLAY_SOURCES[factor_name]
        return f"overlay ({src['status']})"

    # Check candidate pipeline
    step = _CANDIDATE_STATUS.get(factor_name, None)
    if step is None:
        return "unknown (not registered)"
    if step < 0:
        return "not_registered"
    if step >= len(PROMOTION_PIPELINE):
        return "promoted (pending cache rebuild)"
    completed = PROMOTION_PIPELINE[step]
    next_step = (
        PROMOTION_PIPELINE[step + 1]
        if step + 1 < len(PROMOTION_PIPELINE)
        else "DONE"
    )
    return f"step {step}/{len(PROMOTION_PIPELINE)-1}: completed '{completed}', next: '{next_step}'"


def get_champion_cache_path(data_dir: str | Path | None = None) -> Path:
    """Return the absolute path to the champion cache parquet file.

    Args:
        data_dir: Path to data/storage. If None, uses project default.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data" / "storage"
    return Path(data_dir) / CHAMPION_PATH["cache_file"]
