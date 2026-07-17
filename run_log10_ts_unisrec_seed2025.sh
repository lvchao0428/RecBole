#!/usr/bin/env bash
# log10 (1080 Ti 11GB): UniSRec Beauty portability — seed=2025, min5
# 低显存任务：batch=256；结果由 5090 侧 sync_log10_results_to_5090.sh 回收
#
# Usage (on log10):
#   nohup bash run_log10_ts_unisrec_seed2025.sh > logs/ts_unisrec_seed2025_nohup.log 2>&1 &
#
# Or from 5090:
#   bash scripts/remote_start_log10_unisrec.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
MIN5=5
SEED=2025
# 1080 Ti: 降 batch 防 OOM
BATCH="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH="${EVAL_BATCH_SIZE:-256}"

B_TFIDF="$ROOT/dataset/Amazon_Beauty/item_text_emb.base.ts.npy"
B_QWEN="$ROOT/dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts.npy"
B_VIEWS="$ROOT/dataset/Amazon_Beauty/qwen2.5_7b_4views_ts"

LOG="logs/ts_unisrec_seed2025.log"
mkdir -p logs saved
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======== UniSRec Beauty seed=$SEED min5 on log10 (batch=$BATCH) ========"

run_one() {
  local MODEL="$1" CONFIG="$2" LABEL="$3" CKPT="$4" LOGFILE="$5" VFEAT="$6" EXTRA_DICT="$7"
  log ">>> $LABEL"
  t0=$SECONDS
  python scripts/two_phase_train.py \
    --model "$MODEL" \
    --dataset Amazon_Beauty \
    --config_files "$CONFIG" \
    --config_dict "{'freeze_backbone': False, 'train_batch_size': ${BATCH}, 'eval_batch_size': ${EVAL_BATCH}${EXTRA_DICT}}" \
    --gpu_id "$GPU_ID" \
    --min_train_interactions "$MIN5" \
    --phase_a_epochs 50 \
    --phase_a_eval_step 5 \
    --phase_a_valid_metric "MRR@10" \
    --only_phase_a \
    --checkpoint_dir "$CKPT" \
    --seed "$SEED" \
    --variant_features "$VFEAT" \
    --watchdog_disable \
    --save \
    >> "$LOGFILE" 2>&1
  log "  done ($((SECONDS - t0))s)"
}

# Base: use TS-aware Qwen emb
run_one UniSRec unisrec_beauty_stratified_ts.yaml \
  "UniSRec Base" "./saved/ts_unisrec_base_seed${SEED}" \
  "logs/ts_unisrec_base_seed${SEED}.log" "unisrec,base,beauty,ts,min5,seed${SEED},log10" \
  ", 'item_text_emb_path': '${B_QWEN}'"

# AlignV3: TS-aware base + llm
run_one UniSRecAlignV3 unisrec_align_v3_beauty_stratified_ts.yaml \
  "UniSRec +AlignV3" "./saved/ts_unisrec_alignv3_seed${SEED}" \
  "logs/ts_unisrec_alignv3_seed${SEED}.log" "unisrec,alignv3,beauty,ts,min5,seed${SEED},log10" \
  ", 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_path_llm': '${B_QWEN}', 'item_plm_emb_path': '${B_QWEN}'"

# MV-V3: TS-aware multi-view
run_one UniSRecAlignMultiViewV3 unisrec_align_multiview_v3_beauty_stratified_ts.yaml \
  "UniSRec +MV-V3" "./saved/ts_unisrec_mv_v3_seed${SEED}" \
  "logs/ts_unisrec_mv_v3_seed${SEED}.log" "unisrec,mv_v3,no_se,beauty,ts,min5,seed${SEED},log10" \
  ", 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_split_dir': '${B_VIEWS}', 'item_plm_emb_path': '${B_QWEN}', 'use_text_view_senet': false"

log "======== UniSRec Beauty seed=$SEED COMPLETE (log10) ========"
touch logs/ts_unisrec_seed2025.DONE
