#!/usr/bin/env bash
# Pure FDSA baseline (50 epochs) with Stratified Evaluation (book-crossing)
# Uses FDSAAlignMultiViewV3 skeleton (text disabled = pure ID)
#
# Run from project root. SEED via env var, default 2025.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID=${GPU_ID:-0}
SEED="${SEED:-2025}"

echo "Using GPU: $GPU_ID"
echo "SEED: $SEED"

echo "========================================="
echo "FDSA Baseline (50ep, V3) with Stratified Metrics (book-crossing)"
echo "========================================="
echo "Dataset: book-crossing"
echo "Model: FDSAAlignMultiViewV3 (text disabled = pure ID)"
echo ""

python scripts/two_phase_train.py \
	--model FDSAAlignMultiViewV3 \
	--dataset book-crossing \
	--config_files "fdsa_baseline_book_crossing_stratified_v3.yaml" \
	--config_dict "{'freeze_backbone': False}" \
	--gpu_id $GPU_ID \
	--phase_a_epochs 50 \
	--phase_a_eval_step 5 \
	--phase_a_valid_metric "MRR@10" \
	--only_phase_a \
	--checkpoint_dir "./saved/baseline_fdsa_v3_book_crossing_stratified_seed${SEED}" \
	--seed "$SEED" \
	--variant_features "fdsa,id_only,v3,book_crossing,stratified,seed_${SEED}" \
	--watchdog_disable \
	--save

echo ""
echo "========================================="
echo "Training Done (seed=$SEED)."
echo "========================================="
