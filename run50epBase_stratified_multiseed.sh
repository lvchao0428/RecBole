#!/usr/bin/env bash
# Pure SASRec baseline (50 epochs) with Stratified Evaluation - Multi-seed version
# Amazon_Beauty dataset

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
SEEDS=${SEEDS:-"2026 2023 2025 2024 42"}

echo "========================================="
echo "SASRec Baseline (50ep) with Stratified Metrics"
echo "========================================="
echo "Dataset: Amazon_Beauty"
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
        --dataset Amazon_Beauty \
        --config_files "sasrec_baseline_50ep_stratified.yaml" \
        --config_dict "{'freeze_backbone': False}" \
        --gpu_id $GPU_ID \
        --phase_a_epochs 50 \
        --phase_a_eval_step 5 \
        --phase_a_valid_metric "MRR@10" \
        --only_phase_a \
        --checkpoint_dir ./saved/baseline_beauty_seed${SEED} \
        --seed $SEED \
        --variant_features "sasrec,id_only,beauty,seed${SEED}" \
        --watchdog_disable \
        --save
    
    echo ""
    echo "✅ Seed $SEED Done!"
    echo ""
done

echo "========================================="
echo "✅ All seeds completed!"
echo "========================================="
