#!/usr/bin/env bash
# Minimal re-run for per-user paired t-test significance.
#
# Re-trains only the 2 models compared in the main table (Table 2):
#   - TF-IDF+LLM (strongest baseline)
#   - MV-Align 7B (proposed)
# on both datasets (Beauty, Toys), single seed, with --save_peruser_topk.
#
# After training, run:
#   python paper_sigir/compute_peruser_significance.py \
#       --model_a saved/peruser/<baseline>.npy \
#       --model_b saved/peruser/<proposed>.npy
#
# Total: 4 training runs (vs 16+ for full multi-seed).

set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
SEED=2025
PERUSER_DIR="saved/peruser"

echo "=============================================="
echo "Per-user significance test: 2 models × 2 datasets"
echo "Seed: $SEED   GPU: $GPU_ID"
echo "Output: $PERUSER_DIR/"
echo "=============================================="

# ---- 1. Beauty: TF-IDF+LLM (strongest baseline) ----
echo ""
echo "[1/4] Beauty — TF-IDF+LLM"
python scripts/two_phase_train.py \
  --model SASRecAlignV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_qwen3_stratified_v3.yaml" \
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
  --checkpoint_dir ./saved/peruser_runs/beauty_tfidf_llm \
  --seed $SEED \
  --variant_features "beauty_tfidf_llm" \
  --watchdog_disable \
  --save \
  --save_peruser_topk \
  --peruser_output_dir "$PERUSER_DIR"

# ---- 2. Beauty: MV-Align 7B (proposed) ----
echo ""
echo "[2/4] Beauty — MV-Align 7B"
python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v3_stratified.yaml" \
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
  --checkpoint_dir ./saved/peruser_runs/beauty_mv_7b \
  --seed $SEED \
  --variant_features "beauty_mv_7b" \
  --watchdog_disable \
  --save \
  --save_peruser_topk \
  --peruser_output_dir "$PERUSER_DIR"

# ---- 3. Toys: TF-IDF+LLM (strongest baseline) ----
echo ""
echo "[3/4] Toys — TF-IDF+LLM"
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
  --checkpoint_dir ./saved/peruser_runs/toys_tfidf_llm \
  --seed $SEED \
  --variant_features "toys_tfidf_llm" \
  --watchdog_disable \
  --save \
  --save_peruser_topk \
  --peruser_output_dir "$PERUSER_DIR"

# ---- 4. Toys: MV-Align 7B (proposed) ----
echo ""
echo "[4/4] Toys — MV-Align 7B"
python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV3 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v3_toys_stratified_7b.yaml" \
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
  --checkpoint_dir ./saved/peruser_runs/toys_mv_7b \
  --seed $SEED \
  --variant_features "toys_mv_7b" \
  --watchdog_disable \
  --save \
  --save_peruser_topk \
  --peruser_output_dir "$PERUSER_DIR"

echo ""
echo "=============================================="
echo "All 4 runs done. Per-user topk files in: $PERUSER_DIR/"
echo ""
echo "Now run paired t-tests:"
echo ""
echo "  # Beauty"
echo "  python paper_sigir/compute_peruser_significance.py \\"
echo "      --model_a $PERUSER_DIR/beauty_tfidf_llm_topk.npy \\"
echo "      --model_b $PERUSER_DIR/beauty_mv_7b_topk.npy \\"
echo "      --label_a 'TF-IDF+LLM' --label_b 'MV-Align(7B)'"
echo ""
echo "  # Toys"
echo "  python paper_sigir/compute_peruser_significance.py \\"
echo "      --model_a $PERUSER_DIR/toys_tfidf_llm_topk.npy \\"
echo "      --model_b $PERUSER_DIR/toys_mv_7b_topk.npy \\"
echo "      --label_a 'TF-IDF+LLM' --label_b 'MV-Align(7B)'"
echo "=============================================="
