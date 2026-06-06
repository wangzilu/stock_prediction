#!/usr/bin/env bash
# Phase B.5 — 24-split Bucket B sweep
#
# Phase B 6-split LOO flagged 4 groups as neutral/marginal:
#   macro_zero_baseline, st_holder_number, valuation, st_daily_basic
# This wrapper runs each as a 24-split LOO on the production 242-cache
# at the same end-date (2026-05-19) so we can decide whether to drop
# any of them in a future xgb_205 / xgb_201 etc. candidate.
#
# 2026-06-06 cx review (P1) parity: FAILED[] tracker + refuse-ALL-DONE
# semantics so a half-finished sweep cannot be mistaken for a full one.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs/phase_b5"
mkdir -p "$LOG_DIR"

CACHE="data/storage/feature_cache_242_production.parquet"
END_DATE="2026-05-19"

# Bucket B groups from Phase B 6-split verdict (docs/phase_b_loo_audit_20260606.md)
B5_GROUPS=(
  "macro_zero_baseline"
  "st_holder_number"
  "valuation"
  "st_daily_basic"
)

FAILED=()

run_one() {
    local group="$1"
    local label="drop_${group}"
    local log="$LOG_DIR/${label}.log"
    echo "[b5] $(date '+%F %T') START $label  log=$log"
    PYTHONUNBUFFERED=1 python -u scripts/phase4e_24split_ensemble.py \
        --preset 24split --models xgb \
        --n-estimators 500 --early-stopping-rounds 30 \
        --end-date "$END_DATE" \
        --cache-path "$CACHE" \
        --drop-group "$group" \
        --checkpoint-tag "b5_${label}_24split" \
        > "$log" 2>&1
    local rc=$?
    echo "[b5] $(date '+%F %T') END   $label rc=$rc"
    if [ "$rc" -ne 0 ]; then
        FAILED+=("$label(rc=$rc)")
    fi
}

for group in "${B5_GROUPS[@]}"; do
    run_one "$group"
done

if [ "${#FAILED[@]}" -ne 0 ]; then
    echo "[b5] $(date '+%F %T') PARTIAL — ${#FAILED[@]} runs failed: ${FAILED[*]}"
    echo "[b5] Refusing ALL DONE so the verdict cannot consume a half-finished sweep."
    exit 1
fi
echo "[b5] $(date '+%F %T') ALL DONE (no failed runs)"
