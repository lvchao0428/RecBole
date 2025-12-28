#!/usr/bin/env bash
#set -euo pipefail

# Two-phase TF-IDF+LLM with Stratified Evaluation (Toys Dataset)
# 与 Multi-View V2 14B 公平对比的配置
#
# 对齐项（与 two_phase_run_multiview_v2_toys_stratified_14b.sh 保持一致）：
# ============================================================
# Shell 脚本参数:
#   - lr_text_head: 2e-3 (vs 原 1e-3)
#   - lr_dnn_cross: 1e-3 (vs 原 5e-4)
#   - phase_b_alignment_weight: 0.10 (vs 原 0.05)
#
# YAML 配置:
#   - text_weight: 1.0 (vs 原 0.8)
#   - text_gate_init: 0.7 (vs 原 0.5)
#   - text_gate_reg_l2: 0.01 (vs 原 0.05)
# ============================================================

#base_dir='/home/ubuntu/own/RecBole'
base_dir='/home/charlie/project/RecBole'
cd ${base_dir}
#yaml_dir='/home/ubuntu/own/RecBole/toy_train'
yaml_dir='/home/charlie/project/RecBole/toy_train'
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================="
echo "TF-IDF+LLM (14B Fair) with Stratified Metrics (Toys)"
echo "========================================="
echo "Dataset: Amazon_Toys_and_Games"
echo "LLM Model: Qwen2.5-14B-Instruct"
echo ""
echo "Fair Comparison with MultiView V2 14B:"
echo "  - lr_text_head: 2e-3 (2x higher)"
echo "  - lr_dnn_cross: 1e-3 (2x higher)"
echo "  - phase_b_alignment_weight: 0.10 (2x higher)"
echo "  - text_weight: 1.0, text_gate_init: 0.7"
echo "  - text_gate_reg_l2: 0.01 (less regularization)"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Toys_and_Games \
  --config_files "${yaml_dir}/sasrec_align_toys_qwen3_stratified_14b_fair.yaml" \
  --phase_a_grid \
  --align_grid "0.03" \
  --tau_grid "0.1" \
  --backbone_burnin_epochs 10 \
  --burnin_eval_step 2 \
  --phase_a_epochs 15 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0272 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 1e-3 \
  --phase_a_text_gate_reg_l2 0.01 \
  --phase_b_alignment_weight 0.10 \
  --phase_b_text_gate_reg_l2 0.01 \
  --phase_b_text_weight 1.0 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs_tfidf_llm_toys_14b_fair \
  --seed 2025 \
  --variant_features "sasrec,tfidf,llm,14b,toys,stratified,fair_compare" \
  --watchdog_disable \
  --save

echo ""
echo "========================================="
echo "✅ Training Done!"
echo "========================================="
echo "Fair Comparison Settings Applied:"
echo "  - lr_text_head=2e-3, lr_dnn_cross=1e-3 (same as 14B MultiView)"
echo "  - phase_b_alignment_weight=0.10 (same as 14B MultiView)"
echo "  - text_weight=1.0, text_gate_init=0.7 (same as 14B MultiView)"
echo "  - text_gate_reg_l2=0.01 (same as 14B MultiView)"
echo ""
echo "Stratified metrics in results:"
echo "  - Recall_new@10, Recall_few@10, Recall_frequent@10"
echo "  - NDCG_new@10, NDCG_few@10, NDCG_frequent@10"
echo "  - Coverage_new@10, Coverage_few@10, Coverage_frequent@10"
echo ""

