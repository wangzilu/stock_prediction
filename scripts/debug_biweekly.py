"""Debug biweekly+dropout strategy — sanity check the +281% result."""
import sys, os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config.settings import PREDICTION_HORIZON_DAYS
from config.qlib_runtime import init_qlib
from qlib.utils import init_instance_by_config
from models.feature_merger import FeatureMerger
from models.feature_pipeline import prepare_features_174, train_xgb
from backtest.cost_model import CostModel
from backtest.portfolio_backtest import PortfolioBacktest
import xgboost as xgb

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"

init_qlib(QLIB_DATA)
merger = FeatureMerger(DATA_DIR)

today = datetime.now()
test_end = today.strftime("%Y-%m-%d")
test_start = (today - timedelta(days=180)).strftime("%Y-%m-%d")
valid_end = (today - timedelta(days=181)).strftime("%Y-%m-%d")
valid_start = (today - timedelta(days=241)).strftime("%Y-%m-%d")
train_end = (today - timedelta(days=242)).strftime("%Y-%m-%d")
train_start = (today - timedelta(days=365*3+242)).strftime("%Y-%m-%d")

print(f"Test: {test_start}~{test_end}")

dataset = init_instance_by_config({
    "class": "DatasetH", "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
            "kwargs": {"start_time": train_start, "end_time": test_end,
                       "instruments": "all", "label": [LABEL_EXPR]},
        },
        "segments": {
            "train": (train_start, train_end),
            "valid": (valid_start, valid_end),
            "test": (test_start, test_end),
        },
    },
})

print("Preparing features...")
X_train_df, y_train_s = prepare_features_174(dataset, "train", merger)
X_valid_df, y_valid_s = prepare_features_174(dataset, "valid", merger)
X_test_df, y_test_s = prepare_features_174(dataset, "test", merger)

y_train = y_train_s.values.astype(np.float32)
mask_train = np.isfinite(y_train)
y_valid = y_valid_s.values.astype(np.float32)
mask_valid = np.isfinite(y_valid)

print("Training...")
model = train_xgb(
    X_train_df.values.astype(np.float32)[mask_train], y_train[mask_train],
    X_valid_df.values.astype(np.float32)[mask_valid], y_valid[mask_valid])

X_test_np = X_test_df.values.astype(np.float32)
y_test_np = y_test_s.values.astype(np.float32)
pred_raw = model.predict(xgb.DMatrix(X_test_np))

predictions = pd.Series(pred_raw, index=X_test_df.index, name="score")
predictions = predictions[np.isfinite(predictions)]
returns = pd.Series(y_test_np, index=X_test_df.index, name="return")
returns = returns[np.isfinite(returns)]

print(f"Predictions: {len(predictions)}, Returns: {len(returns)}")

# ========== Sanity checks ==========
dates = sorted(predictions.index.get_level_values(0).unique())
print(f"Test dates: {len(dates)}")

# Check 1: What does the label distribution look like?
print(f"\nLabel stats (forward {PREDICTION_HORIZON_DAYS}-day return):")
print(f"  mean={returns.mean()*100:.3f}% std={returns.std()*100:.2f}%")
print(f"  min={returns.min()*100:.2f}% max={returns.max()*100:.2f}%")

# Check 2: Run biweekly+dropout and trace holdings
cost = CostModel()
bt = PortfolioBacktest(top_k=20, cost_model=cost, rebalance_freq=10, dropout_k=15)
result = bt.run(predictions=predictions.to_frame("score"), returns=returns.to_frame("return"))

print(f"\nBiweekly+dropout result:")
print(f"  n_days={result.n_days}, avg_holdings={result.avg_holdings:.1f}")
print(f"  raw_total={result.raw_total_return*100:.1f}%")
print(f"  net_total={result.total_return*100:.1f}%")
print(f"  avg_turnover={result.avg_turnover*100:.1f}%")

# Check 3: Daily PnL distribution
pnl = result.daily_pnl
print(f"\n  Daily PnL stats:")
print(f"    mean={pnl.mean()*100:.3f}% std={pnl.std()*100:.2f}%")
print(f"    min={pnl.min()*100:.2f}% max={pnl.max()*100:.2f}%")
print(f"    >0: {(pnl>0).sum()}/{len(pnl)} = {(pnl>0).mean()*100:.0f}%")

# Check 4: Is there a single day driving everything?
sorted_pnl = pnl.sort_values(ascending=False)
print(f"\n  Top 5 days:")
for d, v in sorted_pnl.head(5).items():
    print(f"    {str(d)[:10]}: {v*100:+.2f}%")
print(f"  Bottom 5 days:")
for d, v in sorted_pnl.tail(5).items():
    print(f"    {str(d)[:10]}: {v*100:+.2f}%")

# Check 5: Compare with simple weekly (no dropout)
bt2 = PortfolioBacktest(top_k=20, cost_model=cost, rebalance_freq=5, dropout_k=0)
r2 = bt2.run(predictions=predictions.to_frame("score"), returns=returns.to_frame("return"))
print(f"\nWeekly (no dropout) for comparison:")
print(f"  net_total={r2.total_return*100:.1f}% sharpe={r2.sharpe_ratio:.3f}")

# Check 6: The KEY question — is the return attribution correct?
# The label is N-day FORWARD return. If we hold for 10 days without rebalancing,
# are we double-counting returns?
print(f"\n=== CRITICAL CHECK ===")
print(f"PREDICTION_HORIZON_DAYS = {PREDICTION_HORIZON_DAYS}")
print(f"Rebalance freq = 10 days")
print(f"If label = {PREDICTION_HORIZON_DAYS}-day forward return and we hold 10 days,")
print(f"we might be attributing the SAME return to multiple holding days!")
if PREDICTION_HORIZON_DAYS > 1:
    print(f"⚠️  LIKELY BUG: label is {PREDICTION_HORIZON_DAYS}-day return but backtest")
    print(f"    treats it as 1-day return per day. This INFLATES results for low-turnover strategies!")
    print(f"    The 'return' column should be DAILY return, not {PREDICTION_HORIZON_DAYS}-day return.")

print("\nDone!")
