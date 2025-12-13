#!/usr/bin/env bash
#set -euo pipefail

# Two-phase TF-IDF baseline with Stratified Evaluation (Amazon_Movies_and_TV Dataset)
# 按交互次数分档评估：new [1,3), few [3,10), frequent [10,+inf)

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================="
echo "TF-IDF Baseline with Stratified Metrics (Movies)"
echo "========================================="
echo "Dataset: Amazon_Movies_and_TV"
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""
echo "Metrics:"
echo "  - Standard: Recall, NDCG, MRR, etc."
echo "  - Stratified: Recall_new@K, NDCG_few@K, etc."
echo "  - Coverage: Coverage_new@K, Coverage_frequent@K, etc."
echo ""

python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Movies_and_TV \
  --config_files "sasrec_align_movies_base_stratified.yaml" \
  --phase_a_grid \
  --align_grid "0.05" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 10 \
  --burnin_eval_step 2 \
  --phase_a_epochs 15 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "Recall@10" \
  --metric_baseline 0.03 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 1e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_text_gate_reg_l2 0.05 \
  --phase_b_alignment_weight 0.05 \
  --phase_b_text_gate_reg_l2 0.05 \
  --phase_b_text_weight 0.8 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs_movies_stratified \
  --seed 2025 \
  --variant_features "sasrec,tfidf,movies,stratified" \
  --watchdog_disable \
  --save

echo ""
echo "✅ Training Done! Stratified metrics available in results."
echo ""

