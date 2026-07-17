#!/usr/bin/env bash
# GTS (Global Time-based Split) 验证实验 — Beauty × 4 配置 × 1 seed
#
# 目的：验证 TS 切分下 MV > LLM > TF-IDF > ID 的相对排序是否与 LOO 一致
# 预计耗时：~8h（4 配置串行，每个 ~2h）
#
# Usage (on 5090):
#   nohup bash run_5090_ts_beauty_validation.sh \
#     > logs/ts_beauty_validation_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2025}"
PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-20}"
PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-50}"
LOG="logs/ts_beauty_validation.log"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "======== GTS Validation: Beauty × 4 configs (seed=$SEED) ========"
log "GPU=$GPU_ID  Split=TS[0.8,0.1,0.1]"

# ============================================================
# 1. ID-only baseline
# ============================================================
log ">>> [1/4] ID-only baseline (SASRecAlign, phase_a only)"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlign \
  --dataset Amazon_Beauty \
  --config_files "sasrec_baseline_50ep_stratified_ts.yaml" \
  --config_dict "{'freeze_backbone': False}" \
  --gpu_id "$GPU_ID" \
  --phase_a_epochs 50 \
  --phase_a_eval_step 5 \
  --phase_a_valid_metric "MRR@10" \
  --only_phase_a \
  --checkpoint_dir ./saved/ts_beauty_id_only_seed${SEED} \
  --seed "$SEED" \
  --variant_features "sasrec,id_only,beauty,ts,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_id_only_seed${SEED}.log" 2>&1
log "  ✅ ID-only done ($((SECONDS - t0))s)"

# ============================================================
# 2. TF-IDF
# ============================================================
log ">>> [2/4] TF-IDF (SASRecAlignV3)"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_base_stratified_v3_ts.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}" \
  --gpu_id "$GPU_ID" \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs ${PHASE_A_EPOCHS} \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_auto_to_b \
  --phase_b_epochs ${PHASE_B_EPOCHS} \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/ts_beauty_tfidf_seed${SEED} \
  --seed "$SEED" \
  --variant_features "sasrec,tfidf,v3,beauty,ts,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_tfidf_seed${SEED}.log" 2>&1
log "  ✅ TF-IDF done ($((SECONDS - t0))s)"

# ============================================================
# 3. TF-IDF + LLM (single-view)
# ============================================================
log ">>> [3/4] TF-IDF + LLM (SASRecAlignV3)"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_qwen3_stratified_v3_ts.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}" \
  --gpu_id "$GPU_ID" \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs ${PHASE_A_EPOCHS} \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_auto_to_b \
  --phase_b_epochs ${PHASE_B_EPOCHS} \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/ts_beauty_tfidf_llm_seed${SEED} \
  --seed "$SEED" \
  --variant_features "sasrec,tfidf,llm,v3,beauty,ts,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_tfidf_llm_seed${SEED}.log" 2>&1
log "  ✅ TF-IDF+LLM done ($((SECONDS - t0))s)"

# ============================================================
# 4. Multi-View (MV-Align)
# ============================================================
log ">>> [4/4] MV-Align (SASRecAlignMultiViewV3)"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v3_stratified_ts.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}" \
  --gpu_id "$GPU_ID" \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs ${PHASE_A_EPOCHS} \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_auto_to_b \
  --phase_b_epochs ${PHASE_B_EPOCHS} \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/ts_beauty_mv_seed${SEED} \
  --seed "$SEED" \
  --variant_features "sasrec,multiview_v3,7b,4views,beauty,ts,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_mv_seed${SEED}.log" 2>&1
log "  ✅ MV-Align done ($((SECONDS - t0))s)"

log "======== GTS Validation complete ========"
log "Results saved in logs/ts_beauty_*_seed${SEED}.log"
log ""
log "Compare with LOO results to check ranking consistency:"
log "  Expected LOO ranking: MV > LLM > TF-IDF > ID (on MRR/NDCG)"
log "  Check: does TS split preserve this ranking?"
