#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID=${GPU_ID:-0}

METRIC_BASELINE=${METRIC_BASELINE:-0.0272}

SEED="${SEED:-2025}"

BATCH_CFG=""
if [[ -n "${TRAIN_BATCH_SIZE:-}" ]]; then
  BATCH_CFG="'train_batch_size': ${TRAIN_BATCH_SIZE}, 'eval_batch_size': ${EVAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}, "
fi

python scripts/two_phase_train.py \

	--model GRU4RecAlignV3 \

	--dataset Amazon_Beauty \

	--config_files "gru4rec_align_beauty_base_stratified_v3.yaml" \

	--config_dict "{${BATCH_CFG}'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}" \

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

	--checkpoint_dir "./saved/gru4rec_tfidf_v3_beauty_stratified_seed${SEED}" \

	--seed "$SEED" \

	--variant_features "gru4rec,tfidf,v3,beauty,stratified,seed_${SEED}" \

	--watchdog_disable \

	--save

