import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA_DIR = DATA_DIR / "qlib_data"
DB_PATH = DATA_DIR / "tracker.db"

QLIB_PROVIDER_URI = str(QLIB_DATA_DIR / "cn_data")

HIGH_THRESHOLD = 0.7
MID_THRESHOLD = 0.3

# Push notification (pushplus.plus)
PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN", "543dffd7b291435bbba636f2b7e3499b")

RECOMMENDATION_TIME = "14:00"
MARKET_CLOSE_TIME = "15:00"
DATA_CUTOFF_TIME = "13:00"

MAX_RECOMMENDATIONS_PER_DAY = 5
MAX_PUSH_PER_STOCK_PER_DAY = 2

PREDICTION_HORIZON_DAYS = 5
TOP_K_STOCKS = 5
