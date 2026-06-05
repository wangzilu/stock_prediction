#!/bin/bash
# 24-split xgb-only validation runner.
#
# Reproduces the 2026-05-26 baseline windows (end-date 2026-05-19)
# with TODAY's code + TODAY's feature cache so we can attribute any
# RankIC delta to code/cache drift instead of regime / window choice.
#
# Output goes to a tagged checkpoint dir so it does NOT collide with
# the historical 5-25 3-model checkpoints at data/storage/phase4e_24split/.
#
# Usage: bash scripts/run_24split_xgb_validation.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs/24split"
mkdir -p "$LOG_DIR"

ts="$(date +%Y%m%d_%H%M%S)"
log="$LOG_DIR/full_24split_xgb_only_${ts}.log"

echo "[wrapper] $(date '+%F %T') launching 24-split xgb-only validation" | tee -a "$log"
echo "[wrapper] log: $log"
echo "[wrapper] checkpoint dir: data/storage/phase4e_24split_xgb/"

# Match 5-26 baseline windows (end_date=2026-05-19) so per-split RankIC
# is apples-to-apples comparable to xgb175_24split_20260526_001941.
# xgb-only cuts runtime ~3× (no LGB/CatBoost retrain).
# Early stopping (default 30 rounds) cuts another ~20-40%.
python scripts/phase4e_24split_ensemble.py \
    --preset 24split \
    --models xgb \
    --n-estimators 500 \
    --early-stopping-rounds 30 \
    --end-date 2026-05-19 \
    --checkpoint-tag 24split_xgb \
    >> "$log" 2>&1

rc=$?
echo "[wrapper] $(date '+%F %T') exited rc=$rc" | tee -a "$log"
exit $rc
