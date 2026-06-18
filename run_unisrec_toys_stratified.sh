#!/usr/bin/env bash
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
echo "UniSRec (transductive, no pretrain) - Toys"
echo "========================================="

python scripts/two_phase_train.py \
	--model UniSRec \
	--dataset Amazon_Toys_and_Games \
	--config_files "unisrec_toys_stratified.yaml" \
	--config_dict "{'freeze_backbone': False}" \
	--gpu_id $GPU_ID \
	--phase_a_epochs 50 \
	--phase_a_eval_step 5 \
	--phase_a_valid_metric "MRR@10" \
	--only_phase_a \
	--checkpoint_dir "./saved/unisrec_toys_stratified_seed${SEED}" \
	--seed "$SEED" \
	--variant_features "unisrec,transductive,toys,stratified,seed_${SEED}" \
	--watchdog_disable \
	--save
