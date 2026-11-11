#!/usr/bin/env bash
# P2 — UniSRec Toys on cutoff-fit LLM embedding.
# Beauty/Grocery UniSRec ckpts already record *.ts.npy; Toys ckpt records .npy.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

GPU_ID="${GPU_ID:-0}"
MIN5=5
SEEDS=(42 2024 2026)
DS="Amazon_Toys_and_Games"
DS_DIR="$ROOT/dataset/${DS}"
LLM_TS="${DS_DIR}/item_text_emb.qwen2.5_7b.base.ts.npy"
YAML="$ROOT/unisrec_toys_stratified_ts.yaml"
[[ -f "$YAML" ]] || YAML="$ROOT/unisrec_toys_stratified.yaml"
LOG="$ROOT/logs/unisrec_toys_ts_0818.log"
CKPT_LIST="$ROOT/logs/unisrec_toys_ts_ckpts.txt"
TRAIN="$ROOT/scripts/two_phase_train.py"

log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

if [[ ! -f "$LLM_TS" ]]; then
  log "FATAL missing $LLM_TS"
  exit 1
fi

: > "$CKPT_LIST"
log "======== UniSRec Toys TS START yaml=$YAML ========"
for SEED in "${SEEDS[@]}"; do
  TAG="unisrec_toys_gts_ts_seed${SEED}"
  CKPT="$ROOT/saved/${TAG}"
  LOGFILE="$ROOT/logs/${TAG}.log"
  log ">>> $TAG"
  python3 "$TRAIN" \
    --model UniSRec --dataset "$DS" \
    --config_files "$YAML" \
    --config_dict "{'seed':${SEED},'item_text_emb_path':'${LLM_TS}'}" \
    --gpu_id "$GPU_ID" --min_train_interactions $MIN5 \
    --phase_a_epochs 50 --phase_a_eval_step 5 --phase_a_valid_metric MRR@10 \
    --only_phase_a --checkpoint_dir "$CKPT" --seed "$SEED" \
    --watchdog_disable --save \
    > "$LOGFILE" 2>&1
  echo "$CKPT" >> "$CKPT_LIST"
  log "  done $TAG"
done

python3 "$ROOT/scripts/eval_gts_last.py" \
  --ckpt_list "$CKPT_LIST" --gpu_id "$GPU_ID" --min_train_interactions 5 \
  --out "$ROOT/paper_recsys/gts_last_eval_unisrec_toys_ts.csv"

for SAMPLE_SEED in 42 123 456; do
  python3 "$ROOT/eval_gts_random.py" \
    --ckpt_list "$CKPT_LIST" --gpu_id "$GPU_ID" --min_train_interactions 5 \
    --sample_seed "$SAMPLE_SEED" \
    --out "$ROOT/paper_recsys/gts_random_eval_unisrec_toys_ts_seed${SAMPLE_SEED}.csv"
done

log "======== UniSRec Toys TS ALL DONE ========"
