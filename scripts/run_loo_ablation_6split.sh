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

# 2026-06-06 cx review (P1): track failed LOO runs so the wrapper
# does NOT print "ALL DONE" when one of them crashed. loo_analysis.py
# already requires a baseline; the wrapper must fail loudly when the
# group set is incomplete so the post-mortem cannot mistake a
# half-finished sweep for a full sweep.
FAILED=()

# Groups with > 0 columns in the current cache. fundamental + northbound
# return 0 cols today (parquets absent in this build) so dropping them
# is a no-op; skipped to save 14 minutes.
LOO_GROUPS=(
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
    if [ "$rc" -ne 0 ]; then
        FAILED+=("$label(rc=$rc)")
    fi
}

# 1. Baseline (no drop) — comparison anchor. Skip when its checkpoint
# dir already exists; the resumable runner would re-emit the same
# numbers and write a duplicate ledger row anyway.
if [ -f "$PROJECT_ROOT/data/storage/phase4e_loo_baseline_full242_6split/summary.json" ]; then
    echo "[loo] $(date '+%F %T') SKIP  baseline_full242 (already has summary.json)"
else
    run_one "baseline_full242" ""
fi

# 2. One LOO per group.
for g in "${LOO_GROUPS[@]}"; do
    run_one "drop_${g}" "--drop-group $g"
done

if [ "${#FAILED[@]}" -ne 0 ]; then
    echo "[loo] $(date '+%F %T') PARTIAL — ${#FAILED[@]} LOO runs failed: ${FAILED[*]}"
    echo "[loo] Refusing to print ALL DONE so the post-mortem (loo_analysis.py)"
    echo "[loo] does not mistake a half-finished sweep for a clean result."
    exit 1
fi
echo "[loo] $(date '+%F %T') ALL DONE (no failed LOOs)"
