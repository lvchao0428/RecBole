#!/usr/bin/env bash
#
# exp_fair_tfidf_llm_cold2_toys.sh - TF-IDF+LLM + cold_boost=2.0 (Toys)
#
# 目的：公平对比 - TF-IDF+LLM 使用与 Multi-view 相同的 cold_start_align_boost
# 证明：Multi-view 的提升来自多视角结构，而非 cold_boost 参数

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Fair Comparison: TF-IDF+LLM + cold_boost=2.0 (Toys)"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_boost: 2.0"
echo "  - inference_cold_text_boost: 0 (不支持)"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlign \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_toys_qwen3_stratified.yaml" \
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
  --config_dict "{'cold_start_align_boost': 2.0, 'cold_start_align_threshold': 10}" \
  --checkpoint_dir ./saved/exp_fair_tfidf_llm_cold2_toys \
  --seed 2025 \
  --variant_features "sasrec,tfidf_llm,toys,fair_cold2" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_fair_tfidf_llm_cold2_toys/"
