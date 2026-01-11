#!/usr/bin/env bash
#
# exp_inference_boost_threshold3.sh - threshold=3 精准匹配 new [0,3)
#
# Threshold 对比：
# - threshold=10: 覆盖 new + few (当前默认)
# - threshold=5:  覆盖 new + 部分 few (正在测试)
# - threshold=3:  精准覆盖 new [0,3) (本实验)
#
# 预期：
# - 更精准的 boost 定义可能提升 HR_new 效果
# - 避免对 few 项目的过度干预

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: Inference Boost threshold=3 (Beauty)"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_boost: 2.0"
echo "  - inference_cold_text_boost: 1.0"
echo "  - cold_start_align_threshold: 3  <-- 精准匹配 new [0,3)"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v2_inference_boost_threshold3.yaml" \
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
  --checkpoint_dir ./saved/exp_inference_boost_threshold3 \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,beauty,inference_boost_threshold3" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_inference_boost_threshold3/"
