#!/usr/bin/env bash
# Phase B.9 — post-#174 chain LLM 24-split LOO.
#
# Question: after #174 density uplift (case-insensitive topic resolver
# + shrink-mode + topic infer), does the 4-col global_chain_llm group
# add ΔRankIC > +0.005 vs xgb_209 baseline?
#
# Apples-to-apples: ONE cache (feature_cache_209_chain_llm.parquet,
# built 2026-06-07 22:31 post-#174 step 3 22:25 bak), two runs:
#   1. baseline = drop global_chain_llm group → 209 dims
#   2. candidate = full cache                → 213 dims
# Same end-date 2026-05-19 as B.4/B.5/B.8 for verdict parity.

set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs/phase_b9"
mkdir -p "$LOG_DIR"

CACHE="data/storage/feature_cache_209_chain_llm.parquet"
END_DATE="2026-05-19"

if [ ! -f "$CACHE" ]; then
  echo "[b9] cache missing: $CACHE" >&2
  exit 2
fi

FAILED=()

run_one() {
    local label="$1"; local extra="$2"; local log="$LOG_DIR/${label}.log"
    echo "[b9] $(date '+%F %T') START $label  log=$log"
    PYTHONUNBUFFERED=1 python -u scripts/phase4e_24split_ensemble.py \
        --preset 24split --models xgb \
        --n-estimators 500 --early-stopping-rounds 30 \
        --end-date "$END_DATE" \
        --cache-path "$CACHE" \
        --checkpoint-tag "$label" \
        $extra \
        > "$log" 2>&1
    local rc=$?
    echo "[b9] $(date '+%F %T') END   $label rc=$rc"
    if [ "$rc" -ne 0 ]; then FAILED+=("$label(rc=$rc)"); fi
}

# Sequential — avoid CPU contention on macOS
run_one "b9_baseline_drop_chain_llm" "--drop-group global_chain_llm"
run_one "b9_candidate_with_chain_llm" ""

echo "[b9] $(date '+%F %T') ALL DONE"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "[b9] FAILED: ${FAILED[*]}" >&2
  exit 1
fi
exit 0
