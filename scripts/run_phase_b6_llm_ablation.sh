#!/usr/bin/env bash
# Phase B.6 — xgb_209_llm vs xgb_209 ablation (6-split fast screen).
#
# Question: does the LLM event factor group (5 cols) add value when
# joined to xgb_209? Phase B.4 promoted xgb_209 (158 + 51 supp = 209)
# as the new champion. xgb_209_llm = 209 + 5 LLM cols. If
# ΔRankIC > +0.005 when keeping LLM vs dropping it, promote xgb_209_llm
# as the next candidate. If neutral, LLM stays shadow.
#
# Sequential — baseline (full 214 cache) then drop_llm_event (= 209)
# so we don't fight CPU between runs.

set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
LOG_DIR="$PROJECT_ROOT/logs/phase_b6"
mkdir -p "$LOG_DIR"

CACHE="data/storage/feature_cache_209_llm.parquet"
END_DATE="2026-05-19"

FAILED=()

run_one() {
    local label="$1"
    local drop_args="$2"
    local log="$LOG_DIR/${label}.log"
    echo "[b6] $(date '+%F %T') START $label  log=$log"
    PYTHONUNBUFFERED=1 python -u scripts/phase4e_24split_ensemble.py \
        --preset 6split --models xgb \
        --n-estimators 500 --early-stopping-rounds 30 \
        --end-date "$END_DATE" \
        --cache-path "$CACHE" \
        --checkpoint-tag "$label" \
        $drop_args \
        > "$log" 2>&1
    local rc=$?
    echo "[b6] $(date '+%F %T') END   $label rc=$rc"
    if [ "$rc" -ne 0 ]; then
        FAILED+=("$label(rc=$rc)")
    fi
}

# Skip baseline if already running externally with the same tag.
if [ -f data/storage/phase4e_xgb_209_llm_baseline_6split/summary.json ]; then
    echo "[b6] $(date '+%F %T') SKIP baseline (already has summary)"
else
    run_one xgb_209_llm_baseline_6split ""
fi

run_one xgb_209_llm_drop_6split "--drop-group llm_event"

if [ "${#FAILED[@]}" -ne 0 ]; then
    echo "[b6] $(date '+%F %T') PARTIAL — ${#FAILED[@]} failed: ${FAILED[*]}"
    exit 1
fi
echo "[b6] $(date '+%F %T') ALL DONE"
