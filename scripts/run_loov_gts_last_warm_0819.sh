#!/usr/bin/env bash
# GTS-Last warm LOOV on existing cutoff-fit MV ckpts (no retraining).
# Beauty: wrap_beauty_mv_*  |  Toys: wrap_toys_mv_*
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

GPU_ID="${GPU_ID:-0}"
CSV="$ROOT/paper_recsys/loov_gts_last_warm.csv"
LOG="$ROOT/logs/loov_gts_last_warm_0819.log"
: > "$LOG"
rm -f "$CSV"

log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

run_one() {
  local DS="$1" CKPT="$2" SEED="$3" SHORT="$4"
  local OUT="$ROOT/paper_recsys/loov_gts_last_warm_${SHORT}_seed${SEED}.md"
  log ">>> $DS seed=$SEED ckpt=$CKPT"
  python3 "$ROOT/scripts/analyze_leave_one_view_out.py" \
    --ckpt "$CKPT" --dataset "$DS" --output "$OUT" \
    --gpu_id "$GPU_ID" --min_train_interactions 5 \
    --protocol gts_last_warm --csv "$CSV" \
    >> "$LOG" 2>&1
  log "  done $OUT"
}

log "======== LOOV GTS-Last warm START ========"
for SEED in 42 2024 2026; do
  run_one Amazon_Beauty \
    "$ROOT/saved/wrap_beauty_mv_lr5e4_do0.3_seed${SEED}" \
    "$SEED" beauty
done
for SEED in 42 2024 2026; do
  run_one Amazon_Toys_and_Games \
    "$ROOT/saved/wrap_toys_mv_lr5e4_do0.3_seed${SEED}" \
    "$SEED" toys
done
log "======== LOOV GTS-Last warm ALL DONE ========"
log "CSV: $CSV"
touch "$ROOT/logs/loov_gts_last_warm_0819.DONE"
