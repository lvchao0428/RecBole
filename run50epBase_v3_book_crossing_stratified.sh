#!/usr/bin/env bash
# Pure SASRec baseline (50 epochs) with Stratified Evaluation (book-crossing)
# 使用 SASRecAlignMultiViewV3 骨架（与 V3 文本实验对齐），禁用全部文本特征
#
# 在项目根目录执行。SEED 由环境变量传入，默认 2025。
# 多 seed：run50epBase_v3_book_crossing_stratified_multiseed.sh

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
echo "SASRec Baseline (50ep, V3) with Stratified Metrics (book-crossing)"
echo "========================================="
echo "Dataset: book-crossing"
echo "Model: SASRecAlignMultiViewV3 (text disabled = pure ID)"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +inf) interactions"
echo ""

python scripts/two_phase_train.py \
	--model SASRecAlignMultiViewV3 \
	--dataset book-crossing \
	--config_files "sasrec_baseline_50ep_book_crossing_stratified_v3.yaml" \
	--config_dict "{'freeze_backbone': False}" \
	--gpu_id $GPU_ID \
	--phase_a_epochs 50 \
	--phase_a_eval_step 5 \
	--phase_a_valid_metric "MRR@10" \
	--only_phase_a \
	--checkpoint_dir "./saved/baseline_v3_book_crossing_stratified_seed${SEED}" \
	--seed "$SEED" \
	--variant_features "sasrec,id_only,v3,book_crossing,stratified,seed_${SEED}" \
	--watchdog_disable \
	--save

echo ""
echo "========================================="
echo "Training Done (seed=$SEED)."
echo "========================================="
