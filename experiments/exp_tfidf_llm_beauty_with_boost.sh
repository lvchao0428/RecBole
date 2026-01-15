#!/usr/bin/env bash
#
# exp_tfidf_llm_beauty_with_boost.sh - TF-IDF+LLM with cold-start boost for Beauty
#
# 目的: 修复层级反转问题 (TF-IDF+LLM HR_new 与 TF-IDF 接近)
# 方案: 为 TF-IDF+LLM 添加 cold_boost 和 infer_boost
# 预期: HR_new@10 提升, 满足 TF-IDF+LLM > TF-IDF
#
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "TF-IDF+LLM with Cold-start Boost (Beauty)"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_boost: 2.0 (新增)"
echo "  - inference_cold_text_boost: 1.0 (新增)"
echo "  - 预期: 修复 HR_new 层级反转"
echo ""

python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_llm_7b_stratified.yaml" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs 20 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0207 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_text_gate_reg_l2 0.01 \
  --phase_b_alignment_weight 0.10 \
  --phase_b_text_gate_reg_l2 0.03 \
  --phase_b_text_weight 1.0 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --config_dict "{'cold_start_align_boost': 2.0, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 1.0}" \
  --checkpoint_dir ./saved/tfidf_llm_beauty_with_boost \
  --seed 2025 \
  --variant_features "sasrec,tfidf_llm,beauty,with_boost" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/tfidf_llm_beauty_with_boost/"

