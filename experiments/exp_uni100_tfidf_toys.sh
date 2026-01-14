#!/usr/bin/env bash
#
# exp_uni100_tfidf_toys.sh - Uni100 Sampled Evaluation (TF-IDF only)
#
# 目的: 验证 uni100 采样评估 vs full-ranking 的一致性
# 配置: TF-IDF baseline (Toys)
# 对比: 与 full-ranking 结果比较 (HR@10=6.60, MRR@10=3.73)
#
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Uni100 Evaluation: TF-IDF (Toys)"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - Training: Two-phase (Phase-A + Phase-B)"
echo "  - Evaluation: uni100 (100 sampled negatives)"
echo ""

python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_toys_base_stratified.yaml" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs 20 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "HR@10" \
  --metric_baseline 0.0597 \
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
  --config_dict "{'cold_start_align_boost': 2.0, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 1.0, 'eval_args': {'mode': 'uni100', 'order': 'RO'}}" \
  --checkpoint_dir ./saved/uni100_tfidf_toys \
  --seed 2025 \
  --variant_features "sasrec,tfidf,toys,uni100" \
  --watchdog_disable \
  --save

echo ""
echo "✅ Done! Results saved to: saved/uni100_tfidf_toys/"
echo ""
echo "Compare with full-ranking:"
echo "  Full: HR@10=6.60, MRR@10=3.73, HR_new@10=1.83"
echo "  Uni100: Check evaluation results above"
