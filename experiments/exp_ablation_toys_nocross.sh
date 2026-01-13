#!/usr/bin/env bash
#
# exp_ablation_toys_nocross.sh - Toys 消融: 移除 Cross Network
#
# 目的：验证 Cross Network 对最终性能的贡献 (RQ4)
# 基准：Toys 7B aggressive (MRR=0.0376)
# 配置：使用最佳参数 (cold=2.5, infer=1.5)，但禁用 Cross
#
# 对应 main.tex 中的 RQ4: Cross and Align ablation
# 预期：MRR 会显著下降（Cross 对 ranking 至关重要）
#

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Ablation: Toys 7B - No Cross Network"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - use_cross: False (禁用 Cross Network)"
echo "  - cold_start_align_boost: 2.5"
echo "  - inference_cold_text_boost: 1.5"
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
  --config_dict "{'cold_start_align_boost': 2.5, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 1.5, 'use_cross': False}" \
  --checkpoint_dir ./saved/exp_ablation_toys_nocross \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,toys,ablation,nocross" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_ablation_toys_nocross/"
