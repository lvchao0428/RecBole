#!/usr/bin/env bash
#
# exp_inference_boost_threshold5.sh - 精准新品定义
#
# 目的：测试 threshold=5 是否比 threshold=10 更精准
# 原理：threshold=5 更聚焦于真正的 new 商品 (pop < 5)
#       而 threshold=10 覆盖了 new + few

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: Inference Boost Threshold=5 (Beauty)"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_threshold: 5 (was 10)"
echo "  - More focused on true 'new' items"
echo ""
echo "Boost coverage:"
echo "  - threshold=10: pop < 10 → new + few"
echo "  - threshold=5:  pop < 5  → mostly new"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v2_inference_boost_threshold5.yaml" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs 20 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0272 \
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
  --checkpoint_dir ./saved/exp_inference_boost_threshold5 \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,beauty,inference_boost,threshold5" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_inference_boost_threshold5/"
