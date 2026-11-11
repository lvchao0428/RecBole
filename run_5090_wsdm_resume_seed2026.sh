#!/usr/bin/env bash
# WSDM wrap-up RESUME — only Toys seed=2026 LLM + MV
# Appends to existing CSV; does NOT overwrite prior results.
#
# Usage:
#   nohup bash run_5090_wsdm_resume_seed2026.sh > logs/wsdm_resume_seed2026_nohup.log 2>&1 &

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

CSV="logs/wsdm_wrapup_toys_gts.csv"
QLOG="logs/wsdm_wrapup_5090_queue.log"
mkdir -p logs saved

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

extract_mrr() {
  local LOG_FILE=$1
  local VALID TEST
  VALID=$(python3 - <<PY
import re
t=open("$LOG_FILE",errors="ignore").read()
m=re.search(r'phase_b_result.*?best_valid_score.: ([0-9.]+)', t, re.S)
print(m.group(1) if m else "")
PY
)
  TEST=$(python3 - <<PY
import re
t=open("$LOG_FILE",errors="ignore").read()
m=re.search(r'phase_b_result.*?"test_result": \{.*?"mrr@10": ([0-9.]+)', t, re.S)
print(m.group(1) if m else "")
PY
)
  if [[ -z "$VALID" ]]; then
    VALID=$(grep -oP 'best_valid_score.: \K[0-9.]+' "$LOG_FILE" | tail -1 || echo 0)
  fi
  if [[ -z "$TEST" ]]; then
    TEST=$(grep -oP '"mrr@10": \K[0-9.]+' "$LOG_FILE" | tail -1 || echo 0)
  fi
  echo "${VALID:-0},${TEST:-0}"
}

run_two_phase() {
  local TAG="$1" MODEL="$2" DATASET="$3" YAML="$4" LR="$5" DO="$6" SEED="$7" EXTRA="$8"
  local CKPT="./saved/${TAG}"
  local LOGFILE="logs/${TAG}.log"
  local CFG="{'learning_rate':${LR},'attn_dropout_prob':${DO},'hidden_dropout_prob':${DO},'weight_decay':0.0,'cold_text_boost':0.0,'infer_boost':0.0,'use_cross':false,'use_seq_text_cross':false,'use_multiview_text_cross':false,'text_cross_layer_num':0${EXTRA}}"

  log ">>> $TAG"
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

log "======== RESUME Toys seed=$SEED (LLM + MV) ========"

rm -rf saved/wrap_toys_llm_lr1e4_do0.1_seed${SEED} 2>/dev/null || true

LLM_EXTRA=",'item_text_emb_path_base':'${ROOT}/dataset/Amazon_Toys_and_Games/item_text_emb.base.ts.npy','item_text_emb_path_llm':'${ROOT}/dataset/Amazon_Toys_and_Games/item_text_emb.qwen2.5_7b.base.ts.npy','use_llm':true,'text_mode':'both'"
MV_EXTRA=",'item_text_emb_path_base':'${ROOT}/dataset/Amazon_Toys_and_Games/item_text_emb.base.ts.npy','item_text_emb_split_dir':'${ROOT}/dataset/Amazon_Toys_and_Games/qwen2.5_7b_4views_ts','use_llm':false,'use_text_view_split':true,'mv_include_base':true"

run_two_phase "wrap_toys_llm_lr1e4_do0.1_seed${SEED}" \
  SASRecAlignV3 Amazon_Toys_and_Games \
  "sasrec_align_toys_qwen3_stratified_v3_ts.yaml" \
  0.0001 0.1 "$SEED" "$LLM_EXTRA"

run_two_phase "wrap_toys_mv_lr5e4_do0.3_seed${SEED}" \
  SASRecAlignMultiViewV3 Amazon_Toys_and_Games \
  "sasrec_align_multi_view_v3_toys_stratified_7b_ts.yaml" \
  0.0005 0.3 "$SEED" "$MV_EXTRA"

log "======== RESUME COMPLETE ========"
log "CSV: $CSV"
sort -t',' -k8 -rn "$CSV" | head -20 | tee -a "$QLOG"
touch logs/wsdm_wrapup_5090.DONE
