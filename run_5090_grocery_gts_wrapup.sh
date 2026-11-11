#!/usr/bin/env bash
#
# Grocery (= Amazon_Grocery_and_Gourmet_Food) GTS wrap-up on 5090.
# Aligns with Beauty/Toys WSDM protocol:
#   - TS split [0.8,0.1,0.1], min_train=5
#   - train-cutoff TS embeddings (*.ts.npy / qwen2.5_7b_4views_ts)
#   - no cold boost / no Cross
#   - seeds {42,2024,2026}; Beauty-locked lr/dropout
# Then re-eval GTS-Last + GTS-Random on the new checkpoints.
#
# Usage:
#   nohup bash run_5090_grocery_gts_wrapup.sh > logs/grocery_gts_wrapup_nohup.log 2>&1 &
#
# Or wait for Beauty/Toys Random first:
#   nohup bash run_5090_after_random_then_grocery.sh > logs/after_random_grocery_nohup.log 2>&1 &
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
MIN5=5
PA_EP=20
PB_EP=50
SEEDS=(42 2024 2026)
DS="Amazon_Grocery_and_Gourmet_Food"
DS_DIR="$ROOT/dataset/Amazon_Grocery_and_Gourmet_Food"

CSV="logs/wsdm_wrapup_grocery_gts.csv"
QLOG="logs/grocery_gts_wrapup_queue.log"
CKPT_LIST="logs/gts_grocery_ckpt_list.txt"
mkdir -p logs saved paper_recsys

echo "tag,model,dataset,lr,dropout,seed,valid_mrr,test_mrr" > "$CSV"
: > "$QLOG"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

# Sanity: Amazon Grocery + timestamp + TS emb
if [[ ! -f "$DS_DIR/Amazon_Grocery_and_Gourmet_Food.inter" ]]; then
  log "FATAL: missing $DS_DIR inter file"
  exit 1
fi
if ! head -1 "$DS_DIR/Amazon_Grocery_and_Gourmet_Food.inter" | grep -q timestamp; then
  log "FATAL: Grocery .inter has no timestamp field — cannot GTS"
  exit 1
fi
for f in item_text_emb.base.ts.npy item_text_emb.qwen2.5_7b.base.ts.npy \
         qwen2.5_7b_4views_ts/view_0.npy; do
  if [[ ! -e "$DS_DIR/$f" ]]; then
    log "FATAL: missing TS feature $DS_DIR/$f"
    exit 1
  fi
done
log "Grocery dataset OK (Amazon + timestamp + TS embeddings)"

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
m=re.search(r'phase_b_result.*?\"test_result\": \{.*?\"mrr@10\": ([0-9.]+)', t, re.S)
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
  local TAG="$1" MODEL="$2" YAML="$3" LR="$4" DO="$5" SEED="$6" EXTRA="$7"
  local CKPT="./saved/${TAG}"
  local LOGFILE="logs/${TAG}.log"
  if [[ -d "$CKPT" ]] && compgen -G "$CKPT/*.pth" > /dev/null; then
    log "SKIP train (ckpt exists): $TAG"
    return 0
  fi
  local CFG="{'learning_rate':${LR},'attn_dropout_prob':${DO},'hidden_dropout_prob':${DO},'weight_decay':0.0,'cold_text_boost':0.0,'infer_boost':0.0,'use_cross':false,'use_seq_text_cross':false,'use_multiview_text_cross':false,'text_cross_layer_num':0${EXTRA}}"

  log ">>> TRAIN $TAG"
  local t0=$SECONDS
  python scripts/two_phase_train.py \
    --model "$MODEL" \
    --dataset "$DS" \
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
  log "  done (${elapsed}s) valid=${VALID} test_all=${TEST}"
  echo "${TAG},${MODEL},${DS},${LR},${DO},${SEED},${VALID},${TEST}" >> "$CSV"
}

run_id_only() {
  local TAG="$1" YAML="$2" LR="$3" DO="$4" SEED="$5"
  local CKPT="./saved/${TAG}"
  local LOGFILE="logs/${TAG}.log"
  if [[ -d "$CKPT" ]] && compgen -G "$CKPT/*.pth" > /dev/null; then
    log "SKIP train (ckpt exists): $TAG"
    return 0
  fi
  local CFG="{'learning_rate':${LR},'attn_dropout_prob':${DO},'hidden_dropout_prob':${DO},'weight_decay':0.0,'freeze_backbone':false,'disable_text_feature':true,'fuse_text_feature':false,'cold_text_boost':0.0,'infer_boost':0.0}"

  log ">>> TRAIN $TAG (ID-only)"
  local t0=$SECONDS
  python scripts/two_phase_train.py \
    --model SASRecAlign \
    --dataset "$DS" \
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
  TEST=$(python3 - <<PY
import re
t=open("$LOGFILE",errors="ignore").read()
m=re.search(r'"test_result": \{.*"mrr@10": ([0-9.]+)', t, re.S)
print(m.group(1) if m else "0")
PY
)
  log "  done (${elapsed}s) valid=${VALID} test_all=${TEST}"
  echo "${TAG},SASRecAlign,${DS},${LR},${DO},${SEED},${VALID},${TEST}" >> "$CSV"
}

TF_EXTRA=",'item_text_emb_path_base':'${DS_DIR}/item_text_emb.base.ts.npy','use_llm':false,'text_mode':'base'"
LLM_EXTRA=",'item_text_emb_path_base':'${DS_DIR}/item_text_emb.base.ts.npy','item_text_emb_path_llm':'${DS_DIR}/item_text_emb.qwen2.5_7b.base.ts.npy','use_llm':true,'text_mode':'both'"
MV_EXTRA=",'item_text_emb_path_base':'${DS_DIR}/item_text_emb.base.ts.npy','item_text_emb_split_dir':'${DS_DIR}/qwen2.5_7b_4views_ts','use_llm':false,'use_text_view_split':true,'mv_include_base':true,'use_text_view_senet':false"

log "======== Grocery GTS wrap-up TRAIN START ========"
log "Protocol: TS[0.8,0.1,0.1] min5=5 no-boost no-Cross seeds=${SEEDS[*]}"

: > "$CKPT_LIST"
for SEED in "${SEEDS[@]}"; do
  log "-------- Grocery seed=$SEED --------"

  run_id_only "wrap_grocery_id_lr1e4_do0.1_seed${SEED}" \
    "sasrec_baseline_50ep_stratified_grocery_ts.yaml" \
    0.0001 0.1 "$SEED"
  echo "saved/wrap_grocery_id_lr1e4_do0.1_seed${SEED}" >> "$CKPT_LIST"

  run_two_phase "wrap_grocery_tf_lr5e4_do0.3_seed${SEED}" \
    SASRecAlignV3 \
    "sasrec_align_grocery_base_stratified_v3_ts.yaml" \
    0.0005 0.3 "$SEED" "$TF_EXTRA"
  echo "saved/wrap_grocery_tf_lr5e4_do0.3_seed${SEED}" >> "$CKPT_LIST"

  run_two_phase "wrap_grocery_llm_lr1e4_do0.1_seed${SEED}" \
    SASRecAlignV3 \
    "sasrec_align_grocery_qwen_stratified_v3_ts.yaml" \
    0.0001 0.1 "$SEED" "$LLM_EXTRA"
  echo "saved/wrap_grocery_llm_lr1e4_do0.1_seed${SEED}" >> "$CKPT_LIST"

  run_two_phase "wrap_grocery_mv_lr5e4_do0.3_seed${SEED}" \
    SASRecAlignMultiViewV3 \
    "sasrec_align_multi_view_v3_grocery_stratified_7b_ts.yaml" \
    0.0005 0.3 "$SEED" "$MV_EXTRA"
  echo "saved/wrap_grocery_mv_lr5e4_do0.3_seed${SEED}" >> "$CKPT_LIST"
done

log "======== TRAIN DONE → Last/Random re-eval ========"

log "[eval] GTS-Last on Grocery ckpts"
python3 scripts/eval_gts_last.py \
  --ckpt_list "$CKPT_LIST" \
  --out logs/gts_last_eval_grocery.csv \
  --min_train_interactions 5 \
  --gpu_id "$GPU_ID" 2>&1 | tee logs/gts_last_eval_grocery.log
cp logs/gts_last_eval_grocery.csv paper_recsys/gts_last_eval_grocery.csv

log "[eval] GTS-Random on Grocery ckpts"
python3 scripts/eval_gts_random.py \
  --ckpt_list "$CKPT_LIST" \
  --out logs/gts_random_eval_grocery.csv \
  --min_train_interactions 5 \
  --sample_seed 42 \
  --gpu_id "$GPU_ID" 2>&1 | tee logs/gts_random_eval_grocery.log
cp logs/gts_random_eval_grocery.csv paper_recsys/gts_random_eval_grocery.csv

touch logs/grocery_gts_wrapup.DONE
log "======== Grocery GTS wrap-up COMPLETE ========"
log "Artifacts:"
log "  train CSV: $CSV"
log "  Last:   logs/gts_last_eval_grocery.csv"
log "  Random: logs/gts_random_eval_grocery.csv"
