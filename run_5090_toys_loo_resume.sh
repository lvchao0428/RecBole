#!/usr/bin/env bash
# Resume Toys Clean LOO from loo_toys_llm_seed2026 + loo_toys_mv_seed2026
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
MIN5=5
PA_EP=20
PB_EP=50
SEED=2026

CSV="logs/toys_loo_results.csv"
QLOG="logs/toys_loo_queue.log"
mkdir -p logs saved

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

extract_mrr() {
  local LOG_FILE=$1
  local VALID TEST
  VALID=$(python3 -c "
import re
t=open('$LOG_FILE',errors='ignore').read()
m=re.search(r'phase_b_result.*?best_valid_score.: ([0-9.]+)', t, re.S)
if m: print(m.group(1))
else:
  m2=re.search(r'best_valid_score.: ([0-9.]+)', t)
  print(m2.group(1) if m2 else '0')
")
  TEST=$(python3 -c "
import re
t=open('$LOG_FILE',errors='ignore').read()
m=re.findall(r'\"mrr@10\": ([0-9.]+)', t)
print(m[-1] if m else '0')
")
  echo "${VALID:-0},${TEST:-0}"
}

run_two_phase() {
  local TAG="$1" MODEL="$2" DATASET="$3" YAML="$4" LR="$5" DO="$6" SEED="$7" EXTRA="$8"
  local CKPT="./saved/${TAG}"
  local LOGFILE="logs/${TAG}.log"
  local CFG="{learning_rate:${LR},attn_dropout_prob:${DO},hidden_dropout_prob:${DO},weight_decay:0.0,cold_text_boost:0.0,infer_boost:0.0,use_cross:false,use_seq_text_cross:false,use_multiview_text_cross:false,text_cross_layer_num:0${EXTRA}}"

  log ">>> $TAG"
  rm -rf "$CKPT"
  local t0=$SECONDS
  python scripts/two_phase_train.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --config_files "$ROOT/$YAML" \
    --config_dict "$CFG" \
    --gpu_id "$GPU_ID" \
    --min_train_interactions $MIN5 \
    --phase_a_grid --align_grid "0.10" --tau_grid "0.05" \
    --backbone_burnin_epochs 0 --burnin_eval_step 2 \
    --phase_a_epochs $PA_EP --phase_a_eval_step 1 --phase_a_valid_metric "MRR@10" \
    --metric_baseline 0.0 --metric_gain_threshold 0.0 \
    --lr_text_head 2e-3 --lr_dnn_cross 5e-4 \
    --phase_a_auto_to_b --phase_b_epochs $PB_EP --backbone_lr_scale 0.1 \
    --checkpoint_dir "$CKPT" --seed "$SEED" \
    --watchdog_disable --save \
    > "$LOGFILE" 2>&1 || true

  local elapsed=$(( SECONDS - t0 ))
  local METRICS VALID TEST
  METRICS=$(extract_mrr "$LOGFILE")
  VALID=${METRICS%,*}
  TEST=${METRICS#*,}
  log "  done (${elapsed}s) valid=${VALID} test=${TEST}"
  echo "${TAG},${MODEL},${DATASET},${LR},${DO},${SEED},${VALID},${TEST}" >> "$CSV"
}

LLM_EXTRA=",item_text_emb_path_base:/dataset/Amazon_Toys_and_Games/item_text_emb.base.npy,item_text_emb_path_llm:/dataset/Amazon_Toys_and_Games/item_text_emb.qwen2.5_7b.base.npy,use_llm:true,text_mode:both"
MV_EXTRA=",item_text_emb_path_base:/dataset/Amazon_Toys_and_Games/item_text_emb.base.npy,item_text_emb_split_dir:/dataset/Amazon_Toys_and_Games/qwen2.5_7b_4views,use_llm:false,use_text_view_split:true,mv_include_base:true"

LLM_YAML="sasrec_align_toys_qwen3_stratified_v3.yaml"
MV_YAML="sasrec_align_multi_view_v3_toys_stratified_7b.yaml"

log "======== Toys LOO RESUME (seed=2026 LLM + MV) ========"

run_two_phase "loo_toys_llm_lr1e4_do0.1_seed${SEED}" \
  SASRecAlignV3 Amazon_Toys_and_Games "$LLM_YAML" \
  0.0001 0.1 "$SEED" "$LLM_EXTRA"

run_two_phase "loo_toys_mv_lr5e4_do0.3_seed${SEED}" \
  SASRecAlignMultiViewV3 Amazon_Toys_and_Games "$MV_YAML" \
  0.0005 0.3 "$SEED" "$MV_EXTRA"

log "======== Toys LOO RESUME COMPLETE ========"
