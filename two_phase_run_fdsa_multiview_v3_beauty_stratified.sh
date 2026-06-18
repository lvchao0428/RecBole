#!/usr/bin/env bash

# Multi-View FDSA V3 with Stratified Evaluation (Amazon_Beauty, Qwen2.5 7B 4-view)

# Run from project root. SEED via env var, default 2025.



set -euo pipefail



ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"



export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"



GPU_ID=${GPU_ID:-0}

METRIC_BASELINE=${METRIC_BASELINE:-0.0272}

SEED="${SEED:-2025}"



echo "Using GPU: $GPU_ID"

echo "METRIC_BASELINE (MRR@10): $METRIC_BASELINE"

echo "SEED: $SEED"



echo "========================================="

echo "Multi-View FDSA V3 with Stratified Metrics (Amazon_Beauty, Qwen2.5 7B 4-view)"

echo "========================================="

echo "Model: FDSAAlignMultiViewV3"

echo "Dataset: Amazon_Beauty"

echo ""



python scripts/two_phase_train.py \

	--model FDSAAlignMultiViewV3 \

	--dataset Amazon_Beauty \

	--config_files "fdsa_align_multi_view_v3_beauty_stratified.yaml" \

	--config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}" \

	--gpu_id $GPU_ID \

	--phase_a_grid \

	--align_grid "0.10" \

	--tau_grid "0.05" \

	--backbone_burnin_epochs 0 \

	--burnin_eval_step 2 \

	--phase_a_epochs 20 \

	--phase_a_eval_step 1 \

	--phase_a_valid_metric "MRR@10" \

	--metric_baseline $METRIC_BASELINE \

	--metric_gain_threshold 0.01 \

	--lr_text_head 2e-3 \

	--lr_dnn_cross 5e-4 \

	--phase_a_auto_to_b \

	--phase_b_epochs 40 \

	--backbone_lr_scale 0.1 \

	--checkpoint_dir "./saved/two_phase_run_fdsa_multiview_v3_beauty_stratified_seed${SEED}" \

	--seed "$SEED" \

	--variant_features "fdsa,multiview_v3,qwen3,beauty,stratified,seed_${SEED}" \

	--watchdog_disable \

	--save



echo ""

echo "========================================="

echo "Training Done!"

echo "========================================="

