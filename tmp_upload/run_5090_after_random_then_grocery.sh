#!/usr/bin/env bash
# Priority overnight chain on 5090:
#   1) Wait Beauty/Toys GTS-Random
#   2) Beauty Early60 Last+Random (existing ew_* ckpts, ~1h)
#   3) Grocery GTS train + Last+Random (aligned protocol)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
QLOG="logs/after_random_grocery_queue.log"
mkdir -p logs
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

wait_for_done() {
  local marker="$1" label="$2"
  log "Waiting for $marker ($label)..."
  while [[ ! -f "$marker" ]]; do
    sleep 60
  done
  log "$label done."
}

# ---- P0: Beauty/Toys Random (already running or done) ----
if [[ ! -f logs/gts_random.DONE ]]; then
  log "Waiting for logs/gts_random.DONE (Beauty/Toys Random)..."
  while [[ ! -f logs/gts_random.DONE ]]; do
    if ! pgrep -f "scripts/eval_gts_random.py --ckpt_list logs/gts_last_ckpt_list" >/dev/null 2>&1; then
      N=$(python3 - <<'PY'
from pathlib import Path
p=Path("logs/gts_random_eval.csv")
if not p.exists():
    print(0)
else:
    print(max(0, sum(1 for _ in p.open())-1))
PY
)
      if [[ "$N" -ge 30 ]]; then
        log "Random process gone but CSV has $N rows → mark DONE"
        touch logs/gts_random.DONE
        cp -f logs/gts_random_eval.csv paper_recsys/gts_random_eval.csv 2>/dev/null || true
        break
      fi
    fi
    sleep 60
  done
fi
log "P0 Beauty/Toys Random ready."
cp -f logs/gts_random_eval.csv paper_recsys/gts_random_eval.csv 2>/dev/null || true

# Rebuild main Last/Random tables if builder present
if [[ -f scripts/build_main_tables_gts.py ]]; then
  log "Building main_table_gts_last_random.md ..."
  python3 scripts/build_main_tables_gts.py \
    --last paper_recsys/gts_last_eval.csv \
    --random paper_recsys/gts_random_eval.csv \
    --out_md paper_recsys/main_table_gts_last_random.md \
    --out_csv paper_recsys/main_table_gts_last_random.csv \
    >> "$QLOG" 2>&1 || log "WARN: build_main_tables_gts.py failed (non-fatal)"
fi

# ---- P1: Early60 Last+Random (RQ2 close) ----
if [[ -f logs/early60_last_random.DONE ]]; then
  log "SKIP Early60 Last/Random (already DONE)"
else
  log "======== P1 Early60 Last+Random ========"
  bash "$ROOT/run_5090_early60_last_random.sh"
fi

# ---- P2: Grocery train + Last/Random ----
if [[ -f logs/grocery_gts_wrapup.DONE ]]; then
  log "SKIP Grocery wrap-up (already DONE)"
else
  log "======== P2 Grocery GTS wrap-up ========"
  bash "$ROOT/run_5090_grocery_gts_wrapup.sh"
fi

touch logs/overnight_priority_chain.DONE
log "======== ALL PRIORITY GAPS CHAIN COMPLETE ========"
log "Artifacts:"
log "  Beauty/Toys Random: paper_recsys/gts_random_eval.csv"
log "  Early60 Last/Random: paper_recsys/gts_last_eval_early60.csv / gts_random_eval_early60.csv"
log "  Grocery Last/Random: paper_recsys/gts_last_eval_grocery.csv / gts_random_eval_grocery.csv"
