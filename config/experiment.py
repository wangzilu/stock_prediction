"""Frozen experiment configuration — single source of truth for all evaluations.

All training/evaluation/backtest scripts MUST import from here to ensure
experiments are reproducible and comparable.

CX Iteration 0: "先让所有实验可比"
"""
from pathlib import Path

# ============================================================
# FROZEN PARAMETERS — do not change without versioning
# ============================================================

# Random seed for all models and splits
SEED = 42

# Universe
UNIVERSE = "all"  # Full A-share

# Label (model training target)
PREDICTION_HORIZON_DAYS = 5  # imported from config.settings, duplicated here for clarity
TARGET_LABEL_EXPR = "Ref($close, -5) / Ref($close, -1) - 1"  # 5-day forward return
TARGET_LABEL_NAME = "target_label_5d"

# Daily PnL return (portfolio accounting, NOT model label)
PNL_RETURN_EXPR = "Ref($close, -1) / $close - 1"  # 1-day close-to-close
PNL_RETURN_NAME = "pnl_return_1d"

# Execution assumption
EXECUTION_ASSUMPTION = "T+1 close-to-close"  # Signal on T close, PnL from T+1 close to T+2 close

# ============================================================
# SPLIT CONFIGURATION
# ============================================================

# Training window
DEFAULT_TRAIN_YEARS = 3  # 3 years of training data (~750 trading days)

# Validation window
DEFAULT_VALID_DAYS = 60  # 60 trading days

# Test window per rolling split
DEFAULT_TEST_DAYS = 20  # 20 trading days (~1 month)

# Number of rolling splits
DEFAULT_N_SPLITS = 24

# ============================================================
# MODEL HYPERPARAMETERS (XGB champion)
# ============================================================

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
    "seed": SEED,
}

# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SETS = {
    "FS-174": {
        "handler": "Alpha158",
        "supplements": ["flow", "custom"],
        "include_holder": False,
        "description": "Champion: Alpha158 + flow(3) + custom(13)",
    },
    "FS-175": {
        "handler": "Alpha158",
        "supplements": ["flow", "custom", "holder"],
        "include_holder": True,
        "description": "Shadow: FS-174 + holder_num(1)",
    },
    "FS-360": {
        "handler": "Alpha360",
        "supplements": [],
        "include_holder": False,
        "description": "Sequence baseline: 60-day raw OHLCV",
    },
    "FS-534": {
        "handler": "Alpha158+Alpha360",
        "supplements": ["flow", "custom"],
        "include_holder": False,
        "description": "Combined: FS-174 + Alpha360",
    },
    "FS-535": {
        "handler": "Alpha158+Alpha360",
        "supplements": ["flow", "custom", "holder"],
        "include_holder": True,
        "description": "Full: FS-175 + Alpha360",
    },
}

# ============================================================
# PROMOTION GATE THRESHOLDS
# ============================================================

ROLLING_GATE = {
    "avg_rank_ic": 0.04,
    "avg_spread": 0.012,
    "rank_ic_pos_pct": 0.65,
    "spread_pos_pct": 0.65,
    "worst20_spread": -0.015,
}

BACKTEST_GATE = {
    "annual_return": 0.0,       # > 0
    "sharpe": 0.8,              # >= 0.8
    "max_drawdown": -0.20,      # >= -20%
    "avg_turnover": 0.35,       # <= 35%
    "cost_to_return": 0.35,     # <= 35%
}

EXPOSURE_GATE = {
    "max_stock_weight": 0.08,   # <= 8%
    "max_industry_weight": 0.40,  # <= 40% (A股行业轮动是alpha来源，不强制中性化)
    "max_adv_participation": 0.02,  # <= 2%
}

GOVERNANCE_GATE = {
    "shadow_min_days": 20,      # >= 20 trading days
    "paper_min_days": 20,       # >= 20 trading days
}

# ============================================================
# OUTPUT PATHS
# ============================================================

PHASE4_OUTPUT_DIR = Path("data/storage/phase4")

# Ensure directory exists on import
PHASE4_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# COST MODEL DEFAULTS
# ============================================================

COST_COMMISSION_RATE = 0.0003    # 万三
COST_STAMP_TAX_RATE = 0.0005    # 万五 (sell only)
COST_SLIPPAGE_RATE = 0.001      # 单边 0.1%

# ============================================================
# PORTFOLIO DEFAULTS (biweekly+dropout — Track B winner)
# ============================================================

PORTFOLIO_TOP_K = 20
PORTFOLIO_REBALANCE_FREQ = 10   # Every 10 trading days
PORTFOLIO_DROPOUT_K = 15        # Only sell if falls below top 35
PORTFOLIO_HOLD_BONUS = 0.0


def experiment_metadata(
    cache_path=None,
    model_version: str = "xgb_174",
) -> dict:
    """Return a metadata dict to embed in every experiment artifact.

    Includes full version binding (data, code, config, qlib calendar)
    so every result can be traced to its exact inputs.

    Parameters
    ----------
    cache_path : Path or str, optional
        Feature cache parquet path (for data_version fingerprint).
    model_version : str
        Human-readable model tag, e.g. "xgb_174".
    """
    from utils.versioning import get_experiment_metadata

    version_info = get_experiment_metadata(
        cache_path=cache_path,
        model_version=model_version,
    )
    return {
        # --- version binding ---
        **version_info,
        # --- experiment parameters ---
        "seed": SEED,
        "universe": UNIVERSE,
        "target_label": TARGET_LABEL_EXPR,
        "target_horizon_days": PREDICTION_HORIZON_DAYS,
        "pnl_return": PNL_RETURN_EXPR,
        "pnl_horizon_days": 1,
        "execution": EXECUTION_ASSUMPTION,
        "train_years": DEFAULT_TRAIN_YEARS,
        "valid_days": DEFAULT_VALID_DAYS,
        "test_days": DEFAULT_TEST_DAYS,
        "n_splits": DEFAULT_N_SPLITS,
        "feature_set": "FS-174",
        "model": "XGB",
        "xgb_params": XGB_PARAMS,
    }
