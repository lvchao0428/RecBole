#!/usr/bin/env bash
#
# exp_fair_multiview_no_boost_beauty.sh - Multi-view 无 boost (Beauty)
#
# 目的：公平对比 - Multi-view 关闭所有 boost 参数
# 与 TF-IDF/TF-IDF+LLM (cold=0) 公平对比，证明多视角结构本身的价值

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Fair Comparison: Multi-view NO Boost (Beauty)"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_boost: 0 (关闭)"
echo "  - inference_cold_text_boost: 0 (关闭)"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v2_stratified.yaml" \
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
  --config_dict "{'cold_start_align_boost': 0, 'inference_cold_text_boost': 0}" \
  --checkpoint_dir ./saved/exp_fair_multiview_no_boost_beauty \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,beauty,fair_no_boost" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_fair_multiview_no_boost_beauty/"
