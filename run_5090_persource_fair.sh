#!/usr/bin/env bash
# Per-source align 公平对比队列
# 三模型统一 per-source align，统一 no-boost(cb0/ib0)
# Round 1: no-Cross  (TF → LLM → MV)
# Round 2: +Cross    (TF → LLM → MV)
#
# Usage:
#   nohup bash run_5090_persource_fair.sh > logs/persource_fair_nohup.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID=0
PA_EP=20
PB_EP=50
MIN5=5
SEED=2025

B_TFIDF="$ROOT/dataset/Amazon_Beauty/item_text_emb.base.ts.npy"
B_LLM="$ROOT/dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts.npy"
B_VIEWS="$ROOT/dataset/Amazon_Beauty/qwen2.5_7b_4views_ts"

LOG="logs/persource_fair_queue.log"
mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

COMMON_NOBOOST="'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0, 'align_weight': 0.1, 'text_weight': 1.0"

run_v3() {
  local LABEL="$1" MODEL="$2" YAML="$3" CKPT="$4" LOGFILE="$5" VFEAT="$6" EXTRA="$7"
  log "--- $LABEL ---"
  t0=$SECONDS
  python scripts/two_phase_train.py \
    --model "$MODEL" \
    --dataset Amazon_Beauty \
    --config_files "$YAML" \
    --config_dict "{${EXTRA}}" \
    --gpu_id $GPU_ID --min_train_interactions $MIN5 \
    --phase_a_grid --align_grid "0.10" --tau_grid "0.05" \
    --backbone_burnin_epochs 0 --burnin_eval_step 2 \
    --phase_a_epochs $PA_EP --phase_a_eval_step 1 --phase_a_valid_metric "MRR@10" \
    --metric_baseline 0.0 --metric_gain_threshold 0.0 \
    --lr_text_head 2e-3 --lr_dnn_cross 5e-4 \
    --phase_a_auto_to_b --phase_b_epochs $PB_EP --backbone_lr_scale 0.1 \
    --checkpoint_dir "$CKPT" --seed $SEED \
    --variant_features "$VFEAT" \
    --watchdog_disable --save \
    2>&1 | tee "$LOGFILE"
  elapsed=$(( SECONDS - t0 ))
  log "  完成: $LABEL (${elapsed}s)"
}

log "======== Round 1: no-Cross, no-boost, per-source align ========"

# TF no-Cross no-boost
run_v3 \
  "TF no-Cross no-boost (per-source)" \
  "SASRecAlignV3" \
  "sasrec_align_base_stratified_v3_ts.yaml" \
  "./saved/ps_beauty_tf_nc_noboost_seed${SEED}" \
  "logs/ps_beauty_tf_nc_noboost_seed${SEED}.log" \
  "ts,tfidf,no_cross,no_boost,per_source,min5,seed${SEED}" \
  "${COMMON_NOBOOST}, 'use_cross': false, 'use_seq_text_cross': false, 'text_cross_layer_num': 0, 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_path_llm': ''"

# LLM no-Cross no-boost
run_v3 \
  "LLM no-Cross no-boost (per-source)" \
  "SASRecAlignV3" \
  "sasrec_align_qwen3_stratified_v3_ts.yaml" \
  "./saved/ps_beauty_llm_nc_noboost_seed${SEED}" \
  "logs/ps_beauty_llm_nc_noboost_seed${SEED}.log" \
  "ts,llm,no_cross,no_boost,per_source,min5,seed${SEED}" \
  "${COMMON_NOBOOST}, 'use_cross': false, 'use_seq_text_cross': false, 'text_cross_layer_num': 0, 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_path_llm': '${B_LLM}'"

# MV no-Cross no-boost — 已跑过(0.0158)，但为保证完全公平同批次对比，重跑
run_v3 \
  "MV no-Cross no-boost (per-source)" \
  "SASRecAlignMultiViewV3" \
  "sasrec_align_multi_view_v3_beauty_ts_nocross.yaml" \
  "./saved/ps_beauty_mv_nc_noboost_seed${SEED}" \
  "logs/ps_beauty_mv_nc_noboost_seed${SEED}.log" \
  "ts,mv,no_cross,no_boost,per_source,min5,seed${SEED}" \
  "${COMMON_NOBOOST}, 'use_cross': false, 'use_multiview_text_cross': false, 'use_text_view_split': true, 'mv_include_base': true, 'use_per_view_align': true, 'text_view_indices': null, 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_path_llm': '', 'item_text_emb_split_dir': '${B_VIEWS}'"

log "======== Round 2: +Cross, no-boost, per-source align ========"

# TF +Cross no-boost
run_v3 \
  "TF +Cross no-boost (per-source)" \
  "SASRecAlignV3" \
  "sasrec_align_base_stratified_v3_ts.yaml" \
  "./saved/ps_beauty_tf_cross_noboost_seed${SEED}" \
  "logs/ps_beauty_tf_cross_noboost_seed${SEED}.log" \
  "ts,tfidf,cross,no_boost,per_source,min5,seed${SEED}" \
  "${COMMON_NOBOOST}, 'use_cross': true, 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_path_llm': ''"

# LLM +Cross no-boost
run_v3 \
  "LLM +Cross no-boost (per-source)" \
  "SASRecAlignV3" \
  "sasrec_align_qwen3_stratified_v3_ts.yaml" \
  "./saved/ps_beauty_llm_cross_noboost_seed${SEED}" \
  "logs/ps_beauty_llm_cross_noboost_seed${SEED}.log" \
  "ts,llm,cross,no_boost,per_source,min5,seed${SEED}" \
  "${COMMON_NOBOOST}, 'use_cross': true, 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_path_llm': '${B_LLM}'"

# MV +Cross no-boost
run_v3 \
  "MV +Cross no-boost (per-source)" \
  "SASRecAlignMultiViewV3" \
  "sasrec_align_multi_view_v3_stratified_ts.yaml" \
  "./saved/ps_beauty_mv_cross_noboost_seed${SEED}" \
  "logs/ps_beauty_mv_cross_noboost_seed${SEED}.log" \
  "ts,mv,cross,no_boost,per_source,min5,seed${SEED}" \
  "${COMMON_NOBOOST}, 'use_cross': true, 'use_multiview_text_cross': true, 'use_text_view_split': true, 'mv_include_base': true, 'use_per_view_align': true, 'text_view_indices': null, 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_path_llm': '', 'item_text_emb_split_dir': '${B_VIEWS}'"

log "======== 全部 per-source 公平对比完成 ========"
