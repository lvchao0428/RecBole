#!/usr/bin/env bash
#
# Beauty Early60 (TS[0.6,0.1,0.3]) — GTS-Last + GTS-Random re-eval.
# Uses existing ew_beauty_* checkpoints (no retraining).
# Seeds in original Early60 run: 2025, 42, 2024.
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/recbole_env.sh" 2>/dev/null || true

GPU_ID="${GPU_ID:-0}"
QLOG="logs/early60_last_random_queue.log"
CKPT_LIST="logs/gts_early60_ckpt_list.txt"
mkdir -p logs paper_recsys

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

cat > "$CKPT_LIST" << 'EOF'
# Beauty Early60 TF/LLM/MV × seeds {2025,42,2024}
saved/ew_beauty_tf_lr5e4_do0.3_seed2025
saved/ew_beauty_tf_lr5e4_do0.3_seed42
saved/ew_beauty_tf_lr5e4_do0.3_seed2024
saved/ew_beauty_llm_lr1e4_do0.1_seed2025
saved/ew_beauty_llm_lr1e4_do0.1_seed42
saved/ew_beauty_llm_lr1e4_do0.1_seed2024
saved/ew_beauty_mv_lr5e4_do0.3_seed2025
saved/ew_beauty_mv_lr5e4_do0.3_seed42
saved/ew_beauty_mv_lr5e4_do0.3_seed2024
EOF

N=0
MISSING=0
while read -r line; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  N=$((N+1))
  if [[ ! -d "$line" ]] || ! compgen -G "$line/*.pth" >/dev/null; then
    log "MISSING ckpt: $line"
    MISSING=$((MISSING+1))
  fi
done < "$CKPT_LIST"
if [[ "$MISSING" -gt 0 ]]; then
  log "FATAL: $MISSING/$N Early60 checkpoints missing — abort"
  exit 1
fi

log "======== Early60 GTS-Last START ($N ckpts) ========"
python3 scripts/eval_gts_last.py \
  --ckpt_list "$CKPT_LIST" \
  --out logs/gts_last_eval_early60.csv \
  --min_train_interactions 5 \
  --gpu_id "$GPU_ID" 2>&1 | tee logs/gts_last_eval_early60.log
cp logs/gts_last_eval_early60.csv paper_recsys/gts_last_eval_early60.csv
log "Early60 Last → logs/gts_last_eval_early60.csv"

log "======== Early60 GTS-Random START ========"
python3 scripts/eval_gts_random.py \
  --ckpt_list "$CKPT_LIST" \
  --out logs/gts_random_eval_early60.csv \
  --min_train_interactions 5 \
  --sample_seed 42 \
  --gpu_id "$GPU_ID" 2>&1 | tee logs/gts_random_eval_early60.log
cp logs/gts_random_eval_early60.csv paper_recsys/gts_random_eval_early60.csv
log "Early60 Random → logs/gts_random_eval_early60.csv"

touch logs/early60_last_random.DONE
log "======== Early60 Last+Random COMPLETE ========"
