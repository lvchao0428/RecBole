#!/usr/bin/env bash
#
# exp_inference_boost_aggressive.sh - 激进版推理时冷启动增强
#
# 对比 exp_inference_boost (cold=2.0, infer_boost=1.0):
# - cold_start_align_boost: 2.5 (更强的训练侧增强)
# - inference_cold_text_boost: 1.5 (新品文本权重 +150%)
#
# 预期效果：
#   - HR_new:      +15~25% (比温和版更强)
#   - MRR_new:     +30~40% (保持)
#   - MRR_frequent: 可能略降 (trade-off)

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# GPU ID support
GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: Inference Boost AGGRESSIVE"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo ""
echo "Key Parameters (vs standard inference_boost):"
echo "  - cold_start_align_boost: 2.5 (was 2.0)"
echo "  - inference_cold_text_boost: 1.5 (was 1.0)"
echo ""
echo "Expected text weight for new items:"
echo "  - pop=0:  factor = 1 + 1.5 = 2.5x (vs 2.0x)"
echo "  - pop=5:  factor = 1 + 0.75 = 1.75x (vs 1.5x)"
echo "  - pop=10+: factor = 1.0 (unchanged)"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v2_inference_boost_aggressive.yaml" \
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
  --checkpoint_dir ./saved/exp_inference_boost_aggressive \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,beauty,inference_boost_aggressive" \
  --watchdog_disable \
  --save

echo ""
echo "========================================="
echo "✅ Training Done!"
echo "========================================="
echo "Check results in: saved/exp_inference_boost_aggressive/"
echo ""
