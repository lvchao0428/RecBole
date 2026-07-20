#!/usr/bin/env bash
# WSDM V4 Stop-Gate: Beauty 3-seed with grid-selected best configs
#
# By valid (0717/0720 规范):
#   LLM: lr=1e-4, dropout=0.1
#   TF:  lr=5e-4, dropout=0.3
#   MV:  lr=5e-4, dropout=0.3  (主报)
#   MV*: lr=5e-4, dropout=0.5  (附带: valid/test 选参翻转诊断)
#
# Seeds: 42, 2024, 2026  (seed=2025 网格结果可作对照)
# Protocol: no-boost, no-Cross, per-source align, proper Phase-A
#
# Usage (on 5090):
#   nohup bash run_5090_stopgate_3seed.sh > logs/stopgate_3seed_nohup.log 2>&1 &

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

CSV="logs/stopgate_3seed_beauty.csv"
LOG="logs/stopgate_3seed_nohup.log"
mkdir -p logs saved

echo "tag,model,lr,dropout,seed,valid_mrr,test_mrr" > "$CSV"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run_one() {
  local LABEL="$1" MODEL="$2" YAML="$3" LR="$4" DO="$5" SEED="$6" TAG="$7"
  local EXTRA="${8:-}"
  local CKPT="./saved/${TAG}"
  local LOGFILE="logs/${TAG}.log"
  local CFG="{'learning_rate':${LR},'attn_dropout_prob':${DO},'hidden_dropout_prob':${DO},'weight_decay':0.0,'cold_text_boost':0.0,'infer_boost':0.0,'use_cross':false,'use_seq_text_cross':false,'text_cross_layer_num':0${EXTRA}}"

  log ">>> $LABEL | seed=$SEED lr=$LR do=$DO"
  local t0=$SECONDS

  python scripts/two_phase_train.py \
    --model "$MODEL" \
    --dataset Amazon_Beauty \
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
  local VALID_MRR TEST_MRR
  VALID_MRR=$(grep -oP 'valid.*?mrr@10\s*[:=]\s*\K[0-9.]+' "$LOGFILE" | tail -1 || true)
  TEST_MRR=$(grep -oP 'test.*?mrr@10\s*[:=]\s*\K[0-9.]+' "$LOGFILE" | tail -1 || true)
  if [[ -z "${VALID_MRR:-}" ]]; then
    VALID_MRR=$(grep -oP 'mrr@10\s*[:=]\s*\K[0-9.]+' "$LOGFILE" | tail -2 | head -1 || echo 0)
  fi
  if [[ -z "${TEST_MRR:-}" ]]; then
    TEST_MRR=$(grep -oP 'mrr@10\s*[:=]\s*\K[0-9.]+' "$LOGFILE" | tail -1 || echo 0)
  fi

  log "  done (${elapsed}s) valid=${VALID_MRR:-N/A} test=${TEST_MRR:-N/A}"
  echo "${TAG},${MODEL},${LR},${DO},${SEED},${VALID_MRR:-0},${TEST_MRR:-0}" >> "$CSV"
}

log "======== Stop-Gate 3-seed Beauty started ========"
log "Seeds: ${SEEDS[*]}"
log "Configs: LLM(1e-4/0.1), TF(5e-4/0.3), MV(5e-4/0.3), MV*(5e-4/0.5)"

for SEED in "${SEEDS[@]}"; do
  log "-------- Seed=$SEED --------"

  run_one "LLM" "SASRecAlignV3" "sasrec_align_qwen3_stratified_v3_ts.yaml" \
    0.0001 0.1 "$SEED" "sg_beauty_llm_lr1e4_do0.1_seed${SEED}"

  run_one "TF" "SASRecAlignV3" "sasrec_align_base_stratified_v3_ts.yaml" \
    0.0005 0.3 "$SEED" "sg_beauty_tf_lr5e4_do0.3_seed${SEED}"

  run_one "MV" "SASRecAlignMultiViewV3" "sasrec_align_multi_view_v3_stratified_ts.yaml" \
    0.0005 0.3 "$SEED" "sg_beauty_mv_lr5e4_do0.3_seed${SEED}"

  # Diagnostic: MV do=0.5 (test-best under seed2025; not for main selection)
  run_one "MV*" "SASRecAlignMultiViewV3" "sasrec_align_multi_view_v3_stratified_ts.yaml" \
    0.0005 0.5 "$SEED" "sg_beauty_mv_lr5e4_do0.5_seed${SEED}"
done

log "======== Stop-Gate 3-seed completed ========"
log "CSV: $CSV"
sort -t',' -k6 -rn "$CSV" | head -20 | tee -a "$LOG"
