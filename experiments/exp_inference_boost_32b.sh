#!/usr/bin/env bash
#
# exp_inference_boost_32b.sh - Beauty 32B + CHANGE-9 (修复 Scale Law 失效)
#
# 当前发现：
# - Beauty 7B: MRR@10=0.0320, 14B: 0.0322, 32B: 0.0314 (Scale Law 失效)
# - inference_boost 在 7B 上成功修复 HR_new
#
# 预期：
# - 32B + CHANGE-9 能否提升 MRR 到超过 14B？
# - 32B + CHANGE-9 的 HR_new 能否转正？

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: Inference Boost 32B (Beauty)"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_boost: 2.0"
echo "  - inference_cold_text_boost: 1.0"
echo "  - cold_start_align_threshold: 10"
echo "  - Model: 32B"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v2_inference_boost_32b.yaml" \
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
  --checkpoint_dir ./saved/exp_inference_boost_32b \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,32b,beauty,inference_boost" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_inference_boost_32b/"
