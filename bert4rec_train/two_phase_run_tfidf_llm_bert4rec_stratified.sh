#!/usr/bin/env bash
#set -euo pipefail

# Two-phase TF-IDF+LLM with Stratified Evaluation + Cold-Start Alignment Boost for BERT4Rec
# 按交互次数分档评估：new [1,3), few [3,10), frequent [10,+inf)

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================="
echo "BERT4Rec TF-IDF+LLM with Stratified Metrics + Cold-Start Boost"
echo "========================================="
echo "Cold-Start Alignment Boost:"
echo "  - boost: 3.0 (cold items get up to 4x align weight)"
echo "  - threshold: 10 (items with popularity < 10)"
echo ""
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


#base_dir='/home/charlie/project/RecBole'
base_dir='/home/ubuntu/own/RecBole/'
cd ${base_dir}
#yaml_dir='/home/charlie/project/RecBole/beauty_train'
yaml_dir='/home/ubuntu/own/RecBole/bert4rec_train'

python scripts/two_phase_train.py \
  --model BERT4RecAlign \
  --dataset Amazon_Beauty \
  --config_files "${yaml_dir}/bert4rec_align_qwen3_stratified.yaml" \
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
  --lr_text_head 1e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_text_gate_reg_l2 0.05 \
  --phase_b_alignment_weight 0.05 \
  --phase_b_text_gate_reg_l2 0.05 \
  --phase_b_text_weight 0.8 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs_bert4rec_stratified \
  --seed 2025 \
  --variant_features "bert4rec,tfidf,llm,beauty,stratified,cold_start_align_boost" \
  --watchdog_disable \
  --save

echo ""
echo "========================================="
echo "✅ Training Done!"
echo "========================================="
echo "Cold-Start Alignment Boost applied:"
echo "  - cold_start_align_boost=3.0 (cold items get up to 4x weight)"
echo "  - cold_start_align_threshold=10 (popularity < 10 = cold)"
echo ""
echo "Stratified metrics in results:"
echo "  - Recall_new@10, Recall_few@10, Recall_frequent@10"
echo "  - NDCG_new@10, NDCG_few@10, NDCG_frequent@10"
echo "  - Coverage_new@10, Coverage_few@10, Coverage_frequent@10"
echo ""

