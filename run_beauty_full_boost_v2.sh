#!/usr/bin/env bash
# Beauty TS V2 FULL BOOST 验证实验
# 目的：验证 cold_boost=3.0 + infer_boost=0.6 + V2 TS-aware 无泄漏特征下是否恢复层级
# 4 configs × seed=2025：ID-only / TF-IDF / TF-IDF+LLM / MV-Align
# 预计耗时：~8h（TF-IDF ~80min, LLM ~100min, MV ~200min）
#
# 优先级：插队到当前 queue_20260712 完成后立即运行
# Usage: nohup bash run_beauty_full_boost_v2.sh > logs/beauty_full_boost_v2_nohup.log 2>&1 &

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

# V2 TS-aware feature paths (无泄漏)
B_TFIDF="$ROOT/dataset/Amazon_Beauty/item_text_emb.base.ts.npy"
B_QWEN="$ROOT/dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts.npy"
B_VIEWS="$ROOT/dataset/Amazon_Beauty/qwen2.5_7b_4views_ts"

LOG="logs/beauty_full_boost_v2.log"
mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======== Beauty TS V2 FULL BOOST (cold_boost=3.0 + infer_boost=0.6) ========"
log "Features: V2 TS-aware (leakage-free), min_train>=$MIN5, seed=$SEED"

# ── 1. ID-only (baseline, boost N/A) ──────────────────────────────────────────
log "--- [1/4] ID-only (no text, no boost) ---"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlign \
  --dataset Amazon_Beauty \
  --config_files sasrec_baseline_50ep_stratified_ts.yaml \
  --config_dict "{'freeze_backbone': false}" \
  --gpu_id $GPU_ID \
  --min_train_interactions $MIN5 \
  --phase_a_epochs 50 \
  --phase_a_eval_step 5 \
  --phase_a_valid_metric "MRR@10" \
  --only_phase_a \
  --checkpoint_dir "./saved/ts_beauty_fb_id_only_seed${SEED}" \
  --seed $SEED \
  --variant_features "id_only,ts,min5,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_fb_id_only_seed${SEED}.log" 2>&1
log "  done ($((SECONDS - t0))s)"

# ── 2. TF-IDF + full boost ─────────────────────────────────────────────────────
log "--- [2/4] TF-IDF + cold_boost=3.0 + infer_boost=0.6 ---"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignV3 \
  --dataset Amazon_Beauty \
  --config_files sasrec_align_base_stratified_v3_ts.yaml \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10, 'item_text_emb_path_base': '${B_TFIDF}'}" \
  --gpu_id $GPU_ID \
  --min_train_interactions $MIN5 \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs $PA_EP \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_auto_to_b \
  --phase_b_epochs $PB_EP \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir "./saved/ts_beauty_fb_tfidf_seed${SEED}" \
  --seed $SEED \
  --variant_features "ts,full_boost,cb3_ib0.6,min5,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_fb_tfidf_seed${SEED}.log" 2>&1
log "  done ($((SECONDS - t0))s)"

# ── 3. TF-IDF+LLM + full boost ────────────────────────────────────────────────
log "--- [3/4] TF-IDF+LLM + cold_boost=3.0 + infer_boost=0.6 ---"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignV3 \
  --dataset Amazon_Beauty \
  --config_files sasrec_align_qwen3_stratified_v3_ts.yaml \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10, 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_path_llm': '${B_QWEN}'}" \
  --gpu_id $GPU_ID \
  --min_train_interactions $MIN5 \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs $PA_EP \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_auto_to_b \
  --phase_b_epochs $PB_EP \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir "./saved/ts_beauty_fb_llm_seed${SEED}" \
  --seed $SEED \
  --variant_features "ts,full_boost,cb3_ib0.6,min5,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_fb_llm_seed${SEED}.log" 2>&1
log "  done ($((SECONDS - t0))s)"

# ── 4. MV-Align + full boost ───────────────────────────────────────────────────
log "--- [4/4] MV-Align + cold_boost=3.0 + infer_boost=0.6 ---"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV3 \
  --dataset Amazon_Beauty \
  --config_files sasrec_align_multi_view_v3_stratified_ts.yaml \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10, 'use_text_view_senet': false, 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_split_dir': '${B_VIEWS}'}" \
  --gpu_id $GPU_ID \
  --min_train_interactions $MIN5 \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs $PA_EP \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_auto_to_b \
  --phase_b_epochs $PB_EP \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir "./saved/ts_beauty_fb_mv_seed${SEED}" \
  --seed $SEED \
  --variant_features "ts,full_boost,cb3_ib0.6,min5,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_fb_mv_seed${SEED}.log" 2>&1
log "  done ($((SECONDS - t0))s)"

log "======== Beauty TS V2 FULL BOOST COMPLETE ========"
