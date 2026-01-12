#!/usr/bin/env bash
#
# exp_boost_beauty_14b_infer2.sh - Beauty 14B + Higher Inference Boost
#
# 目的：验证 Beauty 14B 能否通过更高的 inference_cold_text_boost 超越 7B
#
# 当前问题：Beauty Scale Law 失效
# - 7B aggressive MRR@10 = 0.0327, HR@10 = 0.0607
# - 14B aggressive MRR@10 = 0.0322, HR@10 = 0.0595
#
# 假设：14B 模型需要更强的 inference boost 才能释放潜力
#
# 配置变更：
# - cold_start_align_boost: 2.5 (保持 aggressive)
# - inference_cold_text_boost: 1.5 → 2.0 ⭐
#
# 预期：
# - 如果 MRR > 0.0327：参数是 Scale Law 的关键
# - 如果 MRR ≤ 0.0327：Beauty 数据特性导致 Scale Law 失效

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: Beauty 14B + Higher Inference Boost"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_boost: 2.5"
echo "  - inference_cold_text_boost: 2.0 (from 1.5)"
echo "  - Model: 14B"
echo "  - Target: MRR@10 > 0.0327 (beat 7B aggressive)"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v2_inference_boost_14b_infer2.yaml" \
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
  --checkpoint_dir ./saved/exp_boost_beauty_14b_infer2 \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,14b,beauty,infer2" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_boost_beauty_14b_infer2/"
echo ""
echo "📊 Compare with:"
echo "  - 7B aggressive: MRR@10 = 0.0327, HR@10 = 0.0607"
echo "  - 14B aggressive: MRR@10 = 0.0322, HR@10 = 0.0595"
