#!/usr/bin/env bash
# L6 followup — waits for backfill to finish, then rebuilds factor cache
# and reruns the B.6 ablation smoke test.
#
# Triggered manually OR via a tail-watcher; the script polls the backfill
# log for the "[L6 backfill done]" marker (also accepts the backfill
# process exiting). Once done, runs in sequence:
#   a. scripts/build_llm_event_factors.py
#   b. scripts/build_feature_cache_209_llm.py
#   c. scripts/phase4e_24split_ensemble.py --preset 6split --models xgb \
#         --cache-path data/storage/feature_cache_209_llm.parquet \
#         --checkpoint-tag l6_after_backfill
#   d. Append a one-line summary to docs/llm_l6_backfill_log.md
#
# Idempotency:
#   - Each rebuild step writes its own output file; if file mtime >
#     the backfill marker we treat it as already done and skip.
#   - The smoke run writes to data/storage/phase4e_xgb_l6_after_backfill_6split/
#     — if a summary.json already exists there we don't re-run.
#
# Usage:
#   scripts/run_l6_followup.sh                # poll log forever, then run
#   scripts/run_l6_followup.sh --skip-wait    # backfill already finished
#   scripts/run_l6_followup.sh --force        # ignore idempotency markers

set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

BACKFILL_LOG="$PROJECT_ROOT/logs/llm_l6_backfill.log"
COMPLETION_MARKER="$PROJECT_ROOT/data/storage/llm_l6_backfill_done.json"
SMOKE_LOG="$PROJECT_ROOT/logs/l6_smoke.log"
LOG_DOC="$PROJECT_ROOT/docs/llm_l6_backfill_log.md"
LLM_FACTORS="$PROJECT_ROOT/data/storage/llm_event_factors.parquet"
LLM_CACHE="$PROJECT_ROOT/data/storage/feature_cache_209_llm.parquet"
SMOKE_DIR="$PROJECT_ROOT/data/storage/phase4e_xgb_l6_after_backfill_6split"

SKIP_WAIT=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --skip-wait) SKIP_WAIT=1 ;;
        --force)     FORCE=1 ;;
        --help|-h)
            sed -n '2,28p' "$0"; exit 0 ;;
        *) echo "[l6_followup] unknown arg: $arg" >&2; exit 2 ;;
    esac
done

log() { echo "[l6_followup] $(date '+%F %T') $*"; }
fail() { log "FAIL: $*"; exit 1; }

# ── Step 0: wait for backfill ──────────────────────────────────────────
wait_for_backfill() {
    if [ -f "$COMPLETION_MARKER" ]; then
        log "completion marker already exists at $COMPLETION_MARKER — skipping wait"
        return 0
    fi
    if [ "$SKIP_WAIT" = "1" ]; then
        log "skipping wait (per --skip-wait); backfill assumed done"
        return 0
    fi
    if [ ! -f "$BACKFILL_LOG" ]; then
        fail "no backfill log at $BACKFILL_LOG — did you launch run_l6_backfill.py?"
    fi
    log "watching $BACKFILL_LOG for [L6 backfill done] marker..."
    # Tail-follow until marker shows up OR the writing process exits.
    while true; do
        if grep -q "\[L6 backfill done\]" "$BACKFILL_LOG"; then
            log "backfill log shows done marker"
            return 0
        fi
        # Check if any python process is still writing to the log
        if ! pgrep -fl "run_l6_backfill" > /dev/null; then
            log "no run_l6_backfill process found — exit detected"
            # Final check: maybe marker landed in the last second
            if grep -q "\[L6 backfill done\]" "$BACKFILL_LOG"; then
                return 0
            fi
            log "WARNING: process exited without writing [L6 backfill done] marker — proceeding anyway"
            return 0
        fi
        sleep 30
    done
}

# ── Step a: rebuild llm_event_factors.parquet ──────────────────────────
rebuild_factors() {
    if [ "$FORCE" != "1" ] && [ -f "$LLM_FACTORS" ] && [ -f "$COMPLETION_MARKER" ] && \
       [ "$LLM_FACTORS" -nt "$COMPLETION_MARKER" ]; then
        log "skip rebuild_factors — $LLM_FACTORS newer than backfill marker"
        return 0
    fi
    log "step a: rebuilding $LLM_FACTORS"
    PYTHONUNBUFFERED=1 python -m scripts.build_llm_event_factors \
        2>&1 | tee -a "$PROJECT_ROOT/logs/l6_followup.log" \
        || fail "build_llm_event_factors failed"
}

# ── Step b: rebuild feature_cache_209_llm.parquet ──────────────────────
rebuild_cache() {
    if [ "$FORCE" != "1" ] && [ -f "$LLM_CACHE" ] && [ -f "$LLM_FACTORS" ] && \
       [ "$LLM_CACHE" -nt "$LLM_FACTORS" ]; then
        log "skip rebuild_cache — $LLM_CACHE newer than $LLM_FACTORS"
        return 0
    fi
    log "step b: rebuilding $LLM_CACHE"
    PYTHONUNBUFFERED=1 python -m scripts.build_feature_cache_209_llm \
        2>&1 | tee -a "$PROJECT_ROOT/logs/l6_followup.log" \
        || fail "build_feature_cache_209_llm failed"
}

# ── Step c: smoke re-run of B.6 with denser cache ─────────────────────
run_smoke() {
    if [ "$FORCE" != "1" ] && [ -f "$SMOKE_DIR/summary.json" ]; then
        log "skip run_smoke — $SMOKE_DIR/summary.json already exists"
        return 0
    fi
    log "step c: running 6-split smoke (xgb_209_llm post-L6) → $SMOKE_LOG"
    PYTHONUNBUFFERED=1 python -u scripts/phase4e_24split_ensemble.py \
        --preset 6split --models xgb \
        --cache-path "$LLM_CACHE" \
        --checkpoint-tag l6_after_backfill \
        > "$SMOKE_LOG" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        fail "phase4e_24split_ensemble failed (rc=$rc); see $SMOKE_LOG"
    fi
}

# ── Step d: append one-line summary to docs ────────────────────────────
append_summary() {
    log "step d: appending summary to $LOG_DOC"
    mkdir -p "$(dirname "$LOG_DOC")"
    if [ ! -f "$LOG_DOC" ]; then
        cat > "$LOG_DOC" <<'EOF'
# LLM L6 Backfill Log

Append-only log of L6 backfill + ablation rerun events.
Format: `YYYY-MM-DD HH:MM | <event> | <metrics-or-path>`
EOF
    fi
    local ts="$(date '+%F %H:%M')"
    local summary_line
    if [ -f "$SMOKE_DIR/summary.json" ]; then
        # Pull RankIC + Spread20 from summary.json — keep grep simple to
        # avoid jq dependency
        local rankic=$(python -c "import json; print(json.load(open('$SMOKE_DIR/summary.json')).get('aggregate',{}).get('rankic_mean','?'))" 2>/dev/null || echo "?")
        local sp20=$(python -c "import json; print(json.load(open('$SMOKE_DIR/summary.json')).get('aggregate',{}).get('spread20_bps','?'))" 2>/dev/null || echo "?")
        summary_line="$ts | l6_after_backfill 6split done | RankIC=$rankic Sp20=${sp20}bps log=$SMOKE_LOG"
    else
        summary_line="$ts | l6 followup ran but smoke summary missing | log=$SMOKE_LOG"
    fi
    echo "$summary_line" >> "$LOG_DOC"
    log "appended: $summary_line"
}

# ── Main ───────────────────────────────────────────────────────────────
mkdir -p "$PROJECT_ROOT/logs"
log "L6 followup start"
wait_for_backfill
rebuild_factors
rebuild_cache
run_smoke
append_summary
log "L6 followup DONE"
