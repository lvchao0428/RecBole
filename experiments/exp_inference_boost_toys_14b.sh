#!/usr/bin/env bash
#
# exp_inference_boost_toys_14b.sh - Toys 14B + CHANGE-9 (Scale Law 验证)
#
# 当前发现：
# - Toys HR_new 呈现 Scale Law: 7B(+5%) < 14B(+6%) < 32B(+9%)
# - Toys MV-14B MRR@10=0.0373 (最好)
#
# 预期：
# - CHANGE-9 能否进一步提升 Toys 14B 的 HR_new？
# - MRR 是否能保持或提升？

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: Inference Boost 14B (Toys)"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_boost: 2.0"
echo "  - inference_cold_text_boost: 1.0"
echo "  - cold_start_align_threshold: 10"
echo "  - Model: 14B"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_inference_boost_14b.yaml" \
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
  --checkpoint_dir ./saved/exp_inference_boost_toys_14b \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,14b,toys,inference_boost" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_inference_boost_toys_14b/"
