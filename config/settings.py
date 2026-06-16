import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Load .env file if it exists
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA_DIR = DATA_DIR / "qlib_data"
DB_PATH = DATA_DIR / "tracker.db"
SPOT_CACHE_PATH = DATA_DIR / "a_share_spot_cache.csv"
SPOT_CACHE_META_PATH = DATA_DIR / "a_share_spot_cache.meta.json"
SPOT_CACHE_TTL_SECONDS = int(os.environ.get("SPOT_CACHE_TTL_SECONDS", "300"))

QLIB_PROVIDER_URI = os.environ.get("QLIB_PROVIDER_URI", str(QLIB_DATA_DIR / "cn_data"))
QLIB_DATA_PROVIDER = os.environ.get("QLIB_DATA_PROVIDER", "auto")
QLIB_UNIVERSE_SOURCE = os.environ.get("QLIB_UNIVERSE_SOURCE", "auto")

HIGH_THRESHOLD = 0.7
MID_THRESHOLD = 0.3

# Push notification (pushplus.plus)
PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN", "")

# LLM API (MiniMax)
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")

# StockToday API (Tushare-compatible)
ST_TOKEN = os.environ.get("ST_TOKEN", "Np7IlxHtqsLEjyDEYD2xcaoaoiv7qS93")

RECOMMENDATION_TIME = "14:00"
MARKET_CLOSE_TIME = "15:00"
DATA_CUTOFF_TIME = "13:00"

MAX_RECOMMENDATIONS_PER_DAY = 5
MAX_PUSH_PER_STOCK_PER_DAY = 2

PREDICTION_HORIZON_DAYS = 5
TOP_K_STOCKS = 5
LGB_MIN_PREDICTIONS = int(os.environ.get("LGB_MIN_PREDICTIONS", "4500"))
LGB_INFERENCE_UNIVERSE = os.environ.get("LGB_INFERENCE_UNIVERSE", "all")
LGB_MIN_DATA_INSTRUMENTS = int(os.environ.get("LGB_MIN_DATA_INSTRUMENTS", "4500"))
# 2026-06-04 cx round 13 P2-6: default tightened from 7 → 1 trading day.
# 7 was research-grade leniency that effectively let last week's
# predictions feed today's live recommendation. Set
# LGB_CACHE_MAX_AGE_DAYS=7 explicitly via the env var for
# research/diagnostic paths that need the old behavior.
LGB_CACHE_MAX_AGE_DAYS = int(os.environ.get("LGB_CACHE_MAX_AGE_DAYS", "1"))

# Model paths
RL_MODEL_PATH = DATA_DIR / "rl_model.pt"
# 2026-06-04 cx round 10 Option B: LGB_MODEL_PATH now resolves through
# the profile machinery. The active profile's binary lives at
# ``data/storage/lgb_model_{profile}.pkl``. A legacy symlink at
# ``lgb_model.pkl`` points to the active profile so any caller that
# still hardcodes the legacy filename keeps working during migration.
def _resolve_lgb_model_path():
    from config.production_features import production_model_filename
    return DATA_DIR / production_model_filename()
LGB_MODEL_PATH = _resolve_lgb_model_path()
LGB_LEGACY_MODEL_PATH = DATA_DIR / "lgb_model.pkl"  # legacy alias / symlink target
LGB_PREDICTION_CACHE_PATH = DATA_DIR / "lgb_latest_predictions.json"
MID_MODEL_PATH = DATA_DIR / "mid_model.pt"
OVERNIGHT_STOCK_SNAPSHOT_PATH = DATA_DIR / "overnight_stock_forecasts.json"

# Sell check thresholds
TAKE_PROFIT_PCT = 8.0   # sell if gain >= 8%
STOP_LOSS_PCT = 5.0      # sell if loss >= 5%
LGB_FLIP_THRESHOLD = -0.02  # sell if LGB score drops below this
