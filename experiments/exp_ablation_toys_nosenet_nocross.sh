#!/usr/bin/env bash
#
# exp_ablation_toys_nosenet_nocross.sh - Toys 消融: 同时移除 SENet + Cross
#
# 目的：验证 SENet + Cross 的联合贡献 (RQ2 + RQ4)
# 基准：Toys 7B Aggressive (HR=6.92, MRR=3.76, NDCG=4.51)
# 配置：使用 Aggressive 参数 (cold=2.5, infer=1.5)，同时禁用 SENet 和 Cross
# 注意：主参数选择 Aggressive（层级优先原则，MV 7B Agg 是唯一全满足层级的配置）
#
# 对应 main.tex 中的消融表格 (- SENet - Cross 行)
# 预期：性能下降最大，验证两个组件的联合重要性
#

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Ablation: Toys 7B - No SENet - No Cross"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - use_text_view_senet: False (禁用 SENet)"
echo "  - use_cross: False (禁用 Cross Network)"
echo "  - cold_start_align_boost: 2.5 (Aggressive)"
echo "  - inference_cold_text_boost: 1.5 (Aggressive)"
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
  --config_dict "{'cold_start_align_boost': 2.5, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 1.5, 'use_text_view_senet': False, 'use_cross': False}" \
  --checkpoint_dir ./saved/exp_ablation_toys_nosenet_nocross \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,toys,ablation,nosenet_nocross" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_ablation_toys_nosenet_nocross/"

