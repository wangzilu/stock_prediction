#!/bin/bash
# Phase B.1: leave-one-out ablation across the 11 supplementary
# loader groups in PRODUCTION_SUPPLEMENTARY_GROUPS, on a 6-split fast
# screen. After the run completes, ``three_way_compare.py`` (and
# the ledger) shows the LOO rows side by side with the baseline.
#
# Cache: data/storage/feature_cache_242_production.parquet (242 cols).
# Window: --end-date 2026-05-19 (matches the existing same-exam runs).
#
# Output: one ledger row per LOO group + one baseline row, plus
#         per-checkpoint dirs under data/storage/phase4e_loo_<group>_6split/.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs/loo"
mkdir -p "$LOG_DIR"

CACHE="data/storage/feature_cache_242_production.parquet"
END_DATE="2026-05-19"

# Groups with > 0 columns in the current cache. fundamental + northbound
# return 0 cols today (parquets absent in this build) so dropping them
# is a no-op; skipped to save 14 minutes.
GROUPS=(
    "capital_flow"
    "macro_zero_baseline"
    "shareholder"
    "valuation"
    "quality"
    "st_daily_basic"
    "st_moneyflow"
    "st_holder_number"
    "cross_market_regime"
)

run_one() {
    local label="$1"
    local extra_args="$2"
    local log="$LOG_DIR/${label}.log"
    echo "[loo] $(date '+%F %T') START $label  log=$log"
    python scripts/phase4e_24split_ensemble.py \
        --preset 6split --models xgb \
        --n-estimators 500 --early-stopping-rounds 30 \
        --end-date "$END_DATE" \
        --cache-path "$CACHE" \
        --checkpoint-tag "loo_${label}_6split" \
        $extra_args \
        > "$log" 2>&1
    local rc=$?
    echo "[loo] $(date '+%F %T') END   $label rc=$rc"
}

# 1. Baseline (no drop) — comparison anchor.
run_one "baseline_full242" ""

# 2. One LOO per group.
for g in "${GROUPS[@]}"; do
    run_one "drop_${g}" "--drop-group $g"
done

echo "[loo] $(date '+%F %T') ALL DONE"
