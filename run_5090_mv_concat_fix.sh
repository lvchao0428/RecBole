#!/usr/bin/env bash
# MV concat-predictor 修复后重跑三组实验
# 修复内容：MV no-Cross 融合从简单加法改为 concat+predictor（与 V3 TF/LLM 对齐）
# 优先级：最高（公平对比验证）
# 预计耗时：~6h（每组 ~2h）
#
# Usage:
#   nohup bash run_5090_mv_concat_fix.sh > logs/mv_concat_fix_nohup.log 2>&1 &

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

LOG="logs/mv_concat_fix_queue.log"
mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

COMMON_MV="'use_text_view_senet': false, 'align_weight': 0.1, 'text_weight': 1.0, 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_split_dir': '${B_VIEWS}'"

log "======== MV concat-fix 实验队列启动 (concat+predictor, 3 组) ========"

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

# ────────────────────────────────────────────
# 实验 1: MV no-Cross, no-boost (关键公平对比)
# ────────────────────────────────────────────
run_mv \
  "MV no-Cross, no-boost (concat-fix)" \
  "sasrec_align_multi_view_v3_beauty_ts_nocross.yaml" \
  "./saved/ts_beauty_mv_nc_concatfix_seed${SEED}" \
  "logs/ts_beauty_mv_nc_concatfix_seed${SEED}.log" \
  "ts,mv,no_boost,no_cross,concat_fix,min5,seed${SEED}" \
  "'use_cross': false, 'use_multiview_text_cross': false, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0"

# ────────────────────────────────────────────
# 实验 2: MV +Cross, no-boost
# ────────────────────────────────────────────
run_mv \
  "MV +Cross, no-boost (concat-fix baseline)" \
  "sasrec_align_multi_view_v3_stratified_ts.yaml" \
  "./saved/ts_beauty_mv_cross_concatfix_seed${SEED}" \
  "logs/ts_beauty_mv_cross_concatfix_seed${SEED}.log" \
  "ts,mv,no_boost,cross,concat_fix,min5,seed${SEED}" \
  "'use_cross': true, 'use_multiview_text_cross': true, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0"

# ────────────────────────────────────────────
# 实验 3: MV +Cross, cb3+infer
# ────────────────────────────────────────────
run_mv \
  "MV +Cross, cb3+infer (concat-fix)" \
  "sasrec_align_multi_view_v3_stratified_ts.yaml" \
  "./saved/ts_beauty_mv_cross_boost_concatfix_seed${SEED}" \
  "logs/ts_beauty_mv_cross_boost_concatfix_seed${SEED}.log" \
  "ts,mv,boost,cross,concat_fix,min5,seed${SEED}" \
  "'use_cross': true, 'use_multiview_text_cross': true, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10"

log "======== 全部 3 组 MV concat-fix 实验完成 ========"
