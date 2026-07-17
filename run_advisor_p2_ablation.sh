#!/usr/bin/env bash
# =============================================================================
# Advisor P(2) Ablation — Cross on/off under strict leakage-free protocol
#
# 导师指导 (0712zhidao.txt, 第11条):
#   简化模型：SE、cold boost 从主方法删掉；保留 multi-view/align；
#   Cross 作为机制旋钮，验证是否带来稳定净收益。
#
# 实验设计：V2 TS-aware 特征，no-SE，no-boost，seed=2025
#   补充 3 组（C、E、G-fix）：
#     [C] TF-IDF      + no-Cross
#     [E] TF-IDF+LLM  + no-Cross
#     [G-fix] MV      + no-Cross   重跑修复（旧 G 格子结果不可信）
#
# 训练方式说明（与 B/D/F 协议完全对齐）：
#   所有模型统一走两阶段（two_phase_train.py）：
#     Phase-A frozen backbone（metric_gain_threshold=0，强制进入 Phase-B）
#     Phase-B unfrozen 联合训练
#   no-cross 情况 Phase-A 因信息通路断裂会得到很低的 MRR，但 metric_gain_threshold=0
#   保证无论 Phase-A 得分多低都会进入 Phase-B，Phase-B 从该 checkpoint 继续训练。
#   唯一变量：use_cross 开/关，训练协议完全一致。
#
#   已有数据（无需重跑）：
#     [A] ID-only     → ts_beauty_id_only_min5_seed2025.log      MRR=0.0112
#     [B] TF-IDF+Cross → ts_beauty_tfidf_v2_seed2025.log          MRR=0.0157
#     [D] TF-IDF+LLM+Cross → ts_beauty_tfidf_llm_v2_seed2025.log  MRR=0.0157
#     [F] MV+Cross    → ts_beauty_mv_v2_seed2025.log              MRR=0.0157
#
# 耗时估计：C ~80min + E ~100min + G-fix ~100min ≈ 4.5h
#
# Usage:
#   nohup bash run_advisor_p2_ablation.sh > logs/advisor_p2_ablation_nohup.log 2>&1 &
# =============================================================================

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

# V2 TS-aware leakage-free features
B_TFIDF="$ROOT/dataset/Amazon_Beauty/item_text_emb.base.ts.npy"
B_QWEN="$ROOT/dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts.npy"

LOG="logs/advisor_p2_ablation.log"
mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======== Advisor P(2) Ablation: Cross on/off (V2 TS-aware, no-boost, seed=$SEED) ========"
log "Purpose: verify if Cross brings stable gain under leakage-free protocol"
log "New runs: [C] TF-IDF no-cross  [E] TF-IDF+LLM no-cross  [G-fix] MV no-cross (rerun)"
log "Training: all use two_phase_train.py (metric_gain_threshold=0 forces Phase-B regardless of Phase-A score)"
log "  no-cross Phase-A may collapse, but threshold=0 guarantees Phase-B runs from that checkpoint"
log "  Identical protocol to B/D/F — only variable is use_cross on/off"
log ""
log "Already done (no rerun needed):"
log "  [A] ID-only          → ts_beauty_id_only_min5_seed2025.log       MRR=0.0112"
log "  [B] TF-IDF+Cross     → ts_beauty_tfidf_v2_seed2025.log           MRR=0.0157"
log "  [D] TF-IDF+LLM+Cross → ts_beauty_tfidf_llm_v2_seed2025.log       MRR=0.0157"
log "  [F] MV+Cross         → ts_beauty_mv_v2_seed2025.log               MRR=0.0157"
log ""

# ──────────────────────────────────────────────────────────────────────────────
# [C] TF-IDF + no-Cross  — 两阶段，metric_gain_threshold=0 强制进 Phase-B
# 对照 B: TF-IDF+Cross（ts_beauty_tfidf_v2_seed2025.log, MRR=0.0157）
# ──────────────────────────────────────────────────────────────────────────────
log "--- [C/3] TF-IDF + NO-CROSS (seed=$SEED) ---"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignV3 \
  --dataset Amazon_Beauty \
  --config_files sasrec_align_base_stratified_v3_ts.yaml \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0, 'use_cross': false, 'use_seq_text_cross': false, 'text_cross_layer_num': 0, 'item_text_emb_path_base': '${B_TFIDF}'}" \
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
  --metric_gain_threshold 0.0 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_auto_to_b \
  --phase_b_epochs $PB_EP \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir "./saved/ts_beauty_tfidf_nocross_v2_seed${SEED}" \
  --seed $SEED \
  --variant_features "ts,tfidf,no_cross,noboost,min5,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_tfidf_nocross_v2_seed${SEED}.log" 2>&1
log "  [C] done ($((SECONDS - t0))s)"

# ──────────────────────────────────────────────────────────────────────────────
# [E] TF-IDF+LLM + no-Cross  — 两阶段，metric_gain_threshold=0
# 对照 D: TF-IDF+LLM+Cross（ts_beauty_tfidf_llm_v2_seed2025.log, MRR=0.0157）
# ──────────────────────────────────────────────────────────────────────────────
log "--- [E/3] TF-IDF+LLM + NO-CROSS (seed=$SEED) ---"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignV3 \
  --dataset Amazon_Beauty \
  --config_files sasrec_align_qwen3_stratified_v3_ts.yaml \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0, 'use_cross': false, 'use_seq_text_cross': false, 'text_cross_layer_num': 0, 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_path_llm': '${B_QWEN}'}" \
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
  --metric_gain_threshold 0.0 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_auto_to_b \
  --phase_b_epochs $PB_EP \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir "./saved/ts_beauty_llm_nocross_v2_seed${SEED}" \
  --seed $SEED \
  --variant_features "ts,llm,no_cross,noboost,min5,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_llm_nocross_v2_seed${SEED}.log" 2>&1
log "  [E] done ($((SECONDS - t0))s)"

# ──────────────────────────────────────────────────────────────────────────────
# [G-fix] MV + no-Cross  — 两阶段，metric_gain_threshold=0
# 对照 F: MV+Cross（ts_beauty_mv_v2_seed2025.log, MRR=0.0157）
# 旧结果 ts_beauty_mv_nocross_v2_seed2025.log 作废：
#   旧跑 metric_gain_threshold=0.01，Phase-A MRR=0.0002 未达阈值 → 但实际还是跑了 Phase-B
#   然而 Phase-A 的极差 checkpoint 污染了 Phase-B 起点
#   修复：threshold=0 保证进 Phase-B，行为与其他格子完全一致
# ──────────────────────────────────────────────────────────────────────────────
B_QWEN_VIEWS="$ROOT/dataset/Amazon_Beauty/qwen2.5_7b_4views_ts"
log "--- [G-fix/3] MV + NO-CROSS (fixed, seed=$SEED) ---"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV3 \
  --dataset Amazon_Beauty \
  --config_files sasrec_align_multi_view_v3_beauty_ts_nocross.yaml \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0, 'use_text_view_senet': false, 'use_cross': false, 'use_multiview_text_cross': false, 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_split_dir': '${B_QWEN_VIEWS}'}" \
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
  --metric_gain_threshold 0.0 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_auto_to_b \
  --phase_b_epochs $PB_EP \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir "./saved/ts_beauty_mv_nocross_v2_fixed_seed${SEED}" \
  --seed $SEED \
  --variant_features "ts,mv,no_cross,noboost,min5,seed${SEED},fixed" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_mv_nocross_v2_fixed_seed${SEED}.log" 2>&1
log "  [G-fix] done ($((SECONDS - t0))s)"

log "======== Advisor P(2) Ablation COMPLETE ========"
log ""
log "=== RESULT COLLECTION GUIDE ==="
log "Run the following to collect all 7 cells:"
log "  python3 /tmp/collect_p2_ablation.py"
