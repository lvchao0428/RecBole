#!/usr/bin/env bash
# MV-Align 参数对齐重跑 (Beauty TS, seed=2025, min5)
#
# 修复项：
#   text_weight:       0.7 → 1.0 (对齐 TF/LLM)
#   cross_dropout_prob: 0.15 → 0.2 (对齐 TF/LLM)
#   use_text_view_senet: false (永久关闭)
#
# 4 组实验，串行：
#   1. MV no-boost + no-Cross    (20 PA + 50 PB)
#   2. MV no-boost + Cross       (20 PA + 50 PB)
#   3. MV boost + no-Cross       (20 PA + 50 PB)
#   4. MV boost + Cross          (20 PA + 50 PB)
#
# 预计耗时: ~2h/组 × 4 = ~8h
#
# Usage:
#   nohup bash run_5090_mv_aligned_rerun.sh > logs/mv_aligned_rerun_nohup.log 2>&1 &

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

LOG="logs/mv_aligned_rerun.log"
mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

COMMON_MV="'use_text_view_senet': false, 'align_weight': 0.1, 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_split_dir': '${B_VIEWS}'"

log "======== MV-Align 参数对齐重跑 (text_weight=1.0, senet=off, cross_dropout=0.2) ========"

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
    --checkpoint_dir "$CKPT" \
    --seed $SEED --variant_features "$VFEAT" \
    --watchdog_disable --save \
    >> "$LOGFILE" 2>&1
  log "  done ($((SECONDS - t0))s)"
}

# 1. no-boost + no-Cross
run_mv "[1/4] MV no-boost no-Cross" \
  sasrec_align_multi_view_v3_beauty_ts_nocross.yaml \
  "./saved/ts_beauty_mv_nc_aligned_seed${SEED}" \
  "logs/ts_beauty_mv_nc_aligned_seed${SEED}.log" \
  "ts,mv,no_boost,no_cross,aligned,min5,seed${SEED}" \
  "'use_cross': false, 'use_multiview_text_cross': false, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0"

# 2. no-boost + Cross
run_mv "[2/4] MV no-boost Cross" \
  sasrec_align_multi_view_v3_stratified_ts.yaml \
  "./saved/ts_beauty_mv_cross_aligned_seed${SEED}" \
  "logs/ts_beauty_mv_cross_aligned_seed${SEED}.log" \
  "ts,mv,no_boost,cross,aligned,min5,seed${SEED}" \
  "'use_cross': true, 'use_multiview_text_cross': true, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0"

# 3. boost + no-Cross
run_mv "[3/4] MV boost no-Cross" \
  sasrec_align_multi_view_v3_beauty_ts_nocross.yaml \
  "./saved/ts_beauty_mv_nc_boost_aligned_seed${SEED}" \
  "logs/ts_beauty_mv_nc_boost_aligned_seed${SEED}.log" \
  "ts,mv,boost,no_cross,aligned,min5,seed${SEED}" \
  "'use_cross': false, 'use_multiview_text_cross': false, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10"

# 4. boost + Cross
run_mv "[4/4] MV boost Cross" \
  sasrec_align_multi_view_v3_stratified_ts.yaml \
  "./saved/ts_beauty_mv_cross_boost_aligned_seed${SEED}" \
  "logs/ts_beauty_mv_cross_boost_aligned_seed${SEED}.log" \
  "ts,mv,boost,cross,aligned,min5,seed${SEED}" \
  "'use_cross': true, 'use_multiview_text_cross': true, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10"

log "======== MV-Align 对齐重跑 COMPLETE ========"
