#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
GPU_ID=${GPU_ID:-0}
SEED="${SEED:-2025}"
BATCH_CFG=""
if [[ -n "${TRAIN_BATCH_SIZE:-}" ]]; then
  BATCH_CFG="'train_batch_size': ${TRAIN_BATCH_SIZE}, 'eval_batch_size': ${EVAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}, "
fi
echo "Using GPU: $GPU_ID"
echo "SEED: $SEED"
echo "========================================="
echo "GRU4Rec Baseline (50ep, V3) with Stratified Metrics (toys)"
echo "========================================="

python scripts/two_phase_train.py \
	--model GRU4RecAlignMultiViewV3 \
	--dataset Amazon_Toys_and_Games \
	--config_files "gru4rec_baseline_toys_stratified_v3.yaml" \
	--config_dict "{${BATCH_CFG}'freeze_backbone': False}" \
	--gpu_id $GPU_ID \
	--phase_a_epochs 50 \
	--phase_a_eval_step 5 \
	--phase_a_valid_metric "MRR@10" \
	--only_phase_a \
	--checkpoint_dir "./saved/gru4rec_baseline_v3_toys_stratified_seed${SEED}" \
	--seed "$SEED" \
	--variant_features "gru4rec,id_only,v3,toys,stratified,seed_${SEED}" \
	--watchdog_disable \
	--save
