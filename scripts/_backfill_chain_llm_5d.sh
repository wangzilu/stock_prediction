#!/usr/bin/env bash
# One-shot manual backfill for global_chain_events_llm v1 — 5 days
# (2026-06-08 to 2026-06-12). chain_llm cron never existed historically;
# this fills the gap so B.9 production shadow has fresh signal.
set -uo pipefail
cd "$(dirname "$0")/.."

DATES="2026-06-08 2026-06-09 2026-06-10 2026-06-11 2026-06-12"
LOG_DIR="logs/manual/chain_llm_backfill_$(date +%H%M)"
mkdir -p "$LOG_DIR"

for d in $DATES; do
  echo "[$(date '+%T')] extract LLM events for $d"
  python scripts/extract_global_chain_llm.py \
    --date "$d" --schema v1 --max-candidates 200 --max-llm 80 \
    > "$LOG_DIR/extract_${d}.log" 2>&1
  rc=$?
  echo "[$(date '+%T')] extract $d rc=$rc"
done

echo "[$(date '+%T')] build chain_llm factors (rebuilds whole parquet)"
python scripts/build_global_chain_factors.py --source llm \
  > "$LOG_DIR/build_factors.log" 2>&1
echo "[$(date '+%T')] build rc=$?"

echo "[$(date '+%T')] backfill done"
