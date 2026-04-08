#!/usr/bin/env bash
# Pure SASRec baseline (60 epochs) with Stratified Evaluation - Multi-seed version
# book-crossing dataset
# 参考 run50epBase_toys_stratified_multiseed.sh 风格，使用 two_phase_train.py --only_phase_a
#
# 用法（项目根目录）:
#   bash run60epBase_book_crossing_stratified_multiseed.sh
# 可选:
#   SEEDS="42 2024 2025 2026"  GPU_ID=0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID=${GPU_ID:-0}
SEEDS="${SEEDS:-42 2024 2025 2026}"

echo "========================================="
echo "SASRec Baseline (60ep) with Stratified Metrics (book-crossing)"
echo "========================================="
echo "Dataset: book-crossing"
echo "Model: Pure SASRec (no text features)"
echo "GPU: $GPU_ID"
echo "Seeds: $SEEDS"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

for SEED in $SEEDS; do
    echo "========================================="
    echo "Running seed: $SEED"
    echo "========================================="

    python scripts/two_phase_train.py \
        --model SASRecAlign \
        --dataset book-crossing \
        --config_files "sasrec_baseline_60ep_book_crossing_stratified.yaml" \
        --config_dict "{'freeze_backbone': False}" \
        --gpu_id $GPU_ID \
        --phase_a_epochs 60 \
        --phase_a_eval_step 5 \
        --phase_a_valid_metric "MRR@10" \
        --only_phase_a \
        --checkpoint_dir ./saved/baseline_60ep_book_crossing_stratified_seed${SEED} \
        --seed $SEED \
        --variant_features "sasrec,id_only,book_crossing,seed${SEED}" \
        --watchdog_disable \
        --save

    echo ""
    echo "✅ Seed $SEED Done!"
    echo ""
done

echo "========================================="
echo "✅ All seeds completed!"
echo "========================================="
