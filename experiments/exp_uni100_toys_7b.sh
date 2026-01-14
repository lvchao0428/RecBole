#!/usr/bin/env bash
#
# exp_uni100_toys_7b.sh - Uni100 Sampled Evaluation (Multi-View 7B)
#
# 目的: 验证 uni100 采样评估 vs full-ranking 的一致性
# 配置: Toys 7B Aggressive (二阶段训练)
# 对比: 与 full-ranking 结果比较 (HR@10=6.92, MRR@10=3.76)
#
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Uni100 Evaluation: Multi-View 7B (Toys)"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - Training: Two-phase (Phase-A + Phase-B)"
echo "  - Evaluation: uni100 (100 sampled negatives)"
echo "  - Configuration: Aggressive (cold=2.5, infer=1.5)"
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
  --config_dict "{'cold_start_align_boost': 2.5, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 1.5, 'eval_args': {'mode': 'uni100', 'order': 'RO'}}" \
  --checkpoint_dir ./saved/uni100_toys_7b \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,agg,toys,uni100" \
  --watchdog_disable \
  --save

echo ""
echo "✅ Done! Results saved to: saved/uni100_toys_7b/"
echo ""
echo "Compare with full-ranking:"
echo "  Full: HR@10=6.92, MRR@10=3.76, HR_new@10=1.99"
echo "  Uni100: Check evaluation results above"
