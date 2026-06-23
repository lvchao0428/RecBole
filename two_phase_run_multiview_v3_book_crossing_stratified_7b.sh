#!/usr/bin/env bash
# Multi-View V3 Training with Stratified Evaluation (book-crossing)
# 脚本名沿用 toys 的 _7b 命名；向量目录为 Qwen3 四视图: dataset/book-crossing/qwen3_4views
# 依赖: dataset/book-crossing/item_text_emb.base.npy 与 qwen3_4views/ 下分视图向量
# SEED 由环境变量传入，默认 2025。多 seed：two_phase_run_multiview_v3_book_crossing_stratified_7b_multiseed.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID=${GPU_ID:-0}
PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-20}"
PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-50}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-phaseb50}"
METRIC_BASELINE=${METRIC_BASELINE:-0.015}
SEED="${SEED:-2025}"

echo "Using GPU: $GPU_ID"
echo "METRIC_BASELINE (MRR@10): $METRIC_BASELINE"
echo "SEED: $SEED"

echo "========================================="
echo "Multi-View V3 with Stratified Metrics (book-crossing, Qwen3 4-view)"
echo "========================================="
echo "Model: SASRecAlignMultiViewV3"
echo "Dataset: book-crossing"
echo "Split dir: dataset/book-crossing/qwen3_4views"
echo "V3 Simplified Weights: align_weight, cold_text_boost, infer_boost"
echo ""

python scripts/two_phase_train.py \
	--model SASRecAlignMultiViewV3 \
	--dataset book-crossing \
	--config_files "sasrec_align_multi_view_v3_book_crossing_stratified_7b.yaml" \
	--config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}" \
	--gpu_id $GPU_ID \
	--phase_a_grid \
	--align_grid "0.10" \
	--tau_grid "0.05" \
	--backbone_burnin_epochs 0 \
	--burnin_eval_step 2 \
	--phase_a_epochs ${PHASE_A_EPOCHS} \
	--phase_a_eval_step 1 \
	--phase_a_valid_metric "MRR@10" \
	--metric_baseline $METRIC_BASELINE \
	--metric_gain_threshold 0.01 \
	--lr_text_head 2e-3 \
	--lr_dnn_cross 5e-4 \
	--phase_a_auto_to_b \
	--phase_b_epochs ${PHASE_B_EPOCHS} \
	--backbone_lr_scale 0.1 \
	--checkpoint_dir "./saved/two_phase_run_multiview_v3_book_crossing_stratified_7b_${CHECKPOINT_TAG}_seed${SEED}" \
	--seed "$SEED" \
	--variant_features "sasrec,multiview_v3,qwen3,book_crossing,stratified,${CHECKPOINT_TAG},seed_${SEED}" \
	--watchdog_disable \
	--save

echo ""
echo "========================================="
echo "✅ Training Done!"
echo "========================================="
