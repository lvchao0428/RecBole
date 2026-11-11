#!/usr/bin/env bash
# WSDM Clean LOO — Beauty ID/TF/LLM/MV × 3 seeds
#
# 用修正后的统一代码重跑 LOO，与 GTS 结果做 protocol 对照。
# 使用非 .ts 的全量 embedding 文件（LOO 不做 train-only 特征处理）。
# no-boost / no-Cross / per-source align。
# 超参锁定：ID/LLM lr=1e-4 do=0.1，TF/MV lr=5e-4 do=0.3（同 GTS Beauty grid 最优）。
#
# Usage:
#   nohup bash run_5090_beauty_loo_serial.sh > logs/beauty_loo_nohup.log 2>&1 &

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
SEEDS=(42 2024 2026)

CSV="logs/beauty_loo_results.csv"
QLOG="logs/beauty_loo_queue.log"
mkdir -p logs saved

echo "tag,model,dataset,lr,dropout,seed,valid_mrr,test_mrr" > "$CSV"
: > "$QLOG"

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

run_id_only() {
  local TAG="$1" DATASET="$2" YAML="$3" LR="$4" DO="$5" SEED="$6"
  local CKPT="./saved/${TAG}"
  local LOGFILE="logs/${TAG}.log"
  local CFG="{'learning_rate':${LR},'attn_dropout_prob':${DO},'hidden_dropout_prob':${DO},'weight_decay':0.0,'freeze_backbone':false,'disable_text_feature':true,'fuse_text_feature':false,'cold_text_boost':0.0,'infer_boost':0.0}"

  log ">>> $TAG (ID-only)"
  local t0=$SECONDS
  python scripts/two_phase_train.py \
    --model SASRecAlign \
    --dataset "$DATASET" \
    --config_files "$ROOT/$YAML" \
    --config_dict "$CFG" \
    --gpu_id "$GPU_ID" \
    --min_train_interactions $MIN5 \
    --phase_a_epochs 50 --phase_a_eval_step 5 --phase_a_valid_metric "MRR@10" \
    --only_phase_a \
    --checkpoint_dir "$CKPT" --seed "$SEED" \
    --watchdog_disable --save \
    > "$LOGFILE" 2>&1 || true

  local elapsed=$(( SECONDS - t0 ))
  local VALID TEST
  VALID=$(grep -oP 'best_valid_score.: \K[0-9.]+' "$LOGFILE" | tail -1 || echo 0)
  TEST=$(python3 -c "import re;t=open('$LOGFILE',errors='ignore').read();m=re.findall(r'\"mrr@10\": ([0-9.]+)',t);print(m[-1] if m else '0')")
  log "  done (${elapsed}s) valid=${VALID:-0} test=${TEST:-0}"
  echo "${TAG},SASRecAlign,${DATASET},${LR},${DO},${SEED},${VALID:-0},${TEST:-0}" >> "$CSV"
}

log "======== Beauty Clean LOO (5090) started ========"
log "GPU=$GPU_ID Seeds=${SEEDS[*]}"

# LOO 使用非 .ts 全量 embedding
TF_EXTRA=",'item_text_emb_path_base':'${ROOT}/dataset/Amazon_Beauty/item_text_emb.base.npy','use_llm':false,'text_mode':'base'"
LLM_EXTRA=",'item_text_emb_path_base':'${ROOT}/dataset/Amazon_Beauty/item_text_emb.base.npy','item_text_emb_path_llm':'${ROOT}/dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.npy','use_llm':true,'text_mode':'both'"
MV_EXTRA=",'item_text_emb_path_base':'${ROOT}/dataset/Amazon_Beauty/item_text_emb.base.npy','item_text_emb_split_dir':'${ROOT}/dataset/Amazon_Beauty/qwen2.5_7b_4views','use_llm':false,'use_text_view_split':true,'mv_include_base':true"

# LOO YAML 使用 LS split
ID_YAML="sasrec_baseline_50ep_stratified.yaml"
TF_YAML="sasrec_align_base_stratified_v3.yaml"
LLM_YAML="sasrec_align_qwen3_stratified_v3.yaml"
MV_YAML="sasrec_align_multi_view_v3_stratified.yaml"

for SEED in "${SEEDS[@]}"; do
  log "-------- Beauty LOO seed=$SEED --------"

  run_id_only "loo_beauty_id_lr1e4_do0.1_seed${SEED}" \
    Amazon_Beauty "$ID_YAML" \
    0.0001 0.1 "$SEED"

  run_two_phase "loo_beauty_tfidf_lr5e4_do0.3_seed${SEED}" \
    SASRecAlignV3 Amazon_Beauty "$TF_YAML" \
    0.0005 0.3 "$SEED" "$TF_EXTRA"

  run_two_phase "loo_beauty_llm_lr1e4_do0.1_seed${SEED}" \
    SASRecAlignV3 Amazon_Beauty "$LLM_YAML" \
    0.0001 0.1 "$SEED" "$LLM_EXTRA"

  run_two_phase "loo_beauty_mv_lr5e4_do0.3_seed${SEED}" \
    SASRecAlignMultiViewV3 Amazon_Beauty "$MV_YAML" \
    0.0005 0.3 "$SEED" "$MV_EXTRA"
done

log "======== Beauty Clean LOO COMPLETE ========"
log "CSV: $CSV"
sort -t',' -k8 -rn "$CSV" | head -20 | tee -a "$QLOG"
touch logs/beauty_loo.DONE
