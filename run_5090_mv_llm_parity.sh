#!/usr/bin/env bash
# MV fair fusion vs LLM both-mode
# - SENet removed
# - TF(256) ∥ 4views(4×64=256) = 512 → Linear(512→H) → ID∥text concat
#   (= LLM text_mode=both: TF∥LLM = 512)
# - Per-view align: each view independently aligned with ID via shared proj (64→H)
# - text_view_indices for single-view ablation; mv_include_base=false for views-only
#
# Usage:
#   nohup bash run_5090_mv_llm_parity.sh > logs/mv_tf_views_nohup.log 2>&1 &

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
B_VIEWS="$ROOT/dataset/Amazon_Beauty/qwen2.5_7b_4views_ts"

LOG="logs/mv_tf_views_queue.log"
mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

COMMON_MV="'use_text_view_split': true, 'mv_include_base': true, 'use_per_view_align': true, 'text_view_indices': null, 'align_weight': 0.1, 'text_weight': 1.0, 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_path_llm': '', 'item_text_emb_split_dir': '${B_VIEWS}'"

log "======== MV fair queue (TF∥views=512 ≡ LLM both, SENet removed) ========"

run_mv() {
  local LABEL="$1" YAML="$2" CKPT="$3" LOGFILE="$4" VFEAT="$5" EXTRA="$6"
  log "--- $LABEL ---"
  t0=$SECONDS
  python scripts/two_phase_train.py \
    --model SASRecAlignMultiViewV3 \
    --dataset Amazon_Beauty \
    --config_files "$YAML" \
    --config_dict "{${EXTRA}, ${COMMON_MV}}" \
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

run_mv \
  "MV TF∥views no-Cross, no-boost" \
  "sasrec_align_multi_view_v3_beauty_ts_nocross.yaml" \
  "./saved/ts_beauty_mv_nc_tfviews_seed${SEED}" \
  "logs/ts_beauty_mv_nc_tfviews_seed${SEED}.log" \
  "ts,mv,no_boost,no_cross,tf_views,min5,seed${SEED}" \
  "'use_cross': false, 'use_multiview_text_cross': false, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0"

run_mv \
  "MV TF∥views no-Cross, cb3+infer" \
  "sasrec_align_multi_view_v3_beauty_ts_nocross.yaml" \
  "./saved/ts_beauty_mv_nc_boost_tfviews_seed${SEED}" \
  "logs/ts_beauty_mv_nc_boost_tfviews_seed${SEED}.log" \
  "ts,mv,boost,no_cross,tf_views,min5,seed${SEED}" \
  "'use_cross': false, 'use_multiview_text_cross': false, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10"

run_mv \
  "MV TF∥views +Cross, no-boost" \
  "sasrec_align_multi_view_v3_stratified_ts.yaml" \
  "./saved/ts_beauty_mv_cross_tfviews_seed${SEED}" \
  "logs/ts_beauty_mv_cross_tfviews_seed${SEED}.log" \
  "ts,mv,no_boost,cross,tf_views,min5,seed${SEED}" \
  "'use_cross': true, 'use_multiview_text_cross': true, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0"

log "======== 全部 MV TF∥views 实验完成 ========"
