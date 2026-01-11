#!/usr/bin/env bash
#
# exp_inference_boost.sh - 测试推理时冷启动文本权重增强 (CHANGE-9)
#
# 目标：平衡提升所有指标，特别是 HR_new
# - 训练侧：cold_start_align_boost=2.0 (温和)
# - 推理侧：inference_cold_text_boost=1.0 (新品文本权重翻倍)
#
# 对比基线：exp_burn0 (无任何冷启动增强)
#
# 预期效果：
#   - HR_new:      -2% → +10~15%
#   - HR_few:      -1% → +5~10%  
#   - HR_frequent: +35% → +30~35% (略降)
#   - MRR_new:     +39% → +35~40% (保持)
#   - MRR_frequent:+61% → +55~60% (保持)

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# GPU ID support
GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: Inference Cold Text Boost"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo ""
echo "Key Parameters:"
echo "  - backbone_burnin_epochs: 0 (no burn-in)"
echo "  - cold_start_align_boost: 2.0 (training)"
echo "  - inference_cold_text_boost: 1.0 (inference, NEW!)"
echo "  - alignment_weight: 0.10"
echo "  - temperature: 0.05"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v2_inference_boost.yaml" \
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
  --checkpoint_dir ./saved/exp_inference_boost \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,beauty,inference_boost" \
  --watchdog_disable \
  --save

echo ""
echo "========================================="
echo "✅ Training Done!"
echo "========================================="
echo "Check results in: saved/exp_inference_boost/"
echo ""
