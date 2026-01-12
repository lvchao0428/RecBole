#!/usr/bin/env bash
#
# exp_multiview_toys_aggressive.sh - Toys Multi-view + Aggressive 配置
#
# 目的：修复 Toys MRR 层级问题
# 当前：TF-IDF (0.0373) > LLM (0.0371) > MV (0.0368) ❌
# 目标：MV > LLM > TF-IDF ✅
#
# 理论依据：
# - Beauty aggressive 配置完美达标层级
# - Toys 数据集更大，可能需要更强的 boost 参数
# - LLM align=0.15/tau=0.03 已成功修复，MV 也应该可以

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: Toys Multi-view Aggressive"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_boost: 2.5 (aggressive)"
echo "  - inference_cold_text_boost: 1.5 (aggressive)"
echo "  - Goal: MV MRR@10 > 0.0373 (超过 TF-IDF)"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_stratified_7b.yaml" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs 20 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0249 \
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
  --config_dict "{'cold_start_align_boost': 2.5, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 1.5}" \
  --checkpoint_dir ./saved/exp_multiview_toys_aggressive \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,toys,aggressive" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_multiview_toys_aggressive/"
