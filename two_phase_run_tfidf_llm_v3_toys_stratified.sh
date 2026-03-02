#!/usr/bin/env bash
# Two-phase TF-IDF+LLM V3 with Stratified Evaluation (Toys)

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "Using GPU: $GPU_ID"

echo "========================================="
echo "TF-IDF+LLM V3 with Stratified Metrics (Toys)"
echo "========================================="
echo "Model: SASRecAlignV3"
echo "Dataset: Amazon_Toys_and_Games"
echo "V3 Simplified Weights: align_weight, cold_text_boost, infer_boost"
echo ""

python scripts/two_phase_train.py \
	--model SASRecAlignV3 \
	--dataset Amazon_Toys_and_Games \
	--config_files "sasrec_align_toys_qwen3_stratified_v3.yaml" \
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
	--metric_baseline 0.0272 \
	--metric_gain_threshold 0.01 \
	--lr_text_head 2e-3 \
	--lr_dnn_cross 5e-4 \
	--phase_a_auto_to_b \
	--phase_b_epochs 40 \
	--backbone_lr_scale 0.1 \
	--checkpoint_dir ./saved/two_phase_run_tfidf_llm_v3_toys_stratified \
	--seed 2025 \
	--variant_features "sasrec,tfidf,llm,v3,toys,stratified" \
	--watchdog_disable \
	--save

echo ""
echo "========================================="
echo "✅ Training Done!"
echo "========================================="
