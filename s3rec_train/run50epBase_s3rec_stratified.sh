#!/usr/bin/env bash
# Pure S3Rec baseline (50 epochs finetune) with Stratified Evaluation
# Amazon_Beauty dataset
# NOTE: S3Rec requires pre-training first. Make sure you have the pretrained model.

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "========================================="
echo "S3Rec Baseline (50ep finetune) with Stratified Metrics"
echo "========================================="
echo "Dataset: Amazon_Beauty"
echo "Model: Pure S3Rec (no text features)"
echo ""
echo "NOTE: S3Rec requires pre-training first!"
echo "Pre-trained model path: ./saved/S3Rec-pretrain.pth"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python run_recbole.py \
  --model S3RecAlign \
  --dataset Amazon_Beauty \
  --config_files "s3rec_baseline_50ep_stratified.yaml"

echo ""
echo "✅ Training Done! Check results for stratified metrics."
echo ""

