#!/usr/bin/env bash
# UniSRecAlignMultiViewV3 Beauty · balanced infer_boost=0.6 (主表对齐)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2024}"
CKPT="./saved/unisrec_align_multiview_v3_beauty_stratified_balanced_seed${SEED}"

if [[ -d "$CKPT" ]] && compgen -G "${CKPT}/*.pth" >/dev/null; then
  echo "skip UniSRecAlignMultiViewV3 balanced (ckpt exists): $CKPT"
  exit 0
fi

echo "Using GPU: $GPU_ID · SEED: $SEED · infer_boost=0.6"
python scripts/two_phase_train.py \
  --model UniSRecAlignMultiViewV3 \
  --dataset Amazon_Beauty \
  --config_files "unisrec_align_multiview_v3_beauty_stratified.yaml" \
  --config_dict "{'freeze_backbone': False, 'infer_boost': 0.6}" \
  --gpu_id "$GPU_ID" \
  --phase_a_epochs 50 \
  --phase_a_eval_step 5 \
  --phase_a_valid_metric "MRR@10" \
  --only_phase_a \
  --checkpoint_dir "$CKPT" \
  --seed "$SEED" \
  --variant_features "unisrec_align_multiview_v3,cross,align,mv,balanced,beauty,stratified,seed_${SEED}" \
  --watchdog_disable \
  --save
