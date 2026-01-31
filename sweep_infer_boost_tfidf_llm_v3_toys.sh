#!/usr/bin/env bash
# Infer Boost Sweep - TF-IDF + LLM V3 Toys

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ==========================================
# 配置区域
# ==========================================
CHECKPOINT="./saved/phase_runs_tfidf_llm_v3_toys_stratified/SASRecAlignV3-Amazon_Toys_and_Games-Jan25-2026.pth"
MODEL="SASRecAlignV3"
DATASET="Amazon_Toys_and_Games"
CONFIG_FILES="sasrec_align_tfidf_llm_v3_toys_stratified.yaml"
INFER_BOOST_GRID="0.0,0.5,0.8,1.0,1.2,1.5,2.0"
VALID_METRIC="MRR@10"
VARIANT_LABEL="tfidf_llm_v3_toys"
GPU_ID=${GPU_ID:-0}

# ==========================================
# 检查 checkpoint
# ==========================================
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $CHECKPOINT"
    echo "查找: ls -lht saved/phase_runs_tfidf_llm_v3_toys_stratified/*.pth"
    exit 1
fi

echo "=========================================="
echo "Infer Boost Sweep - TF-IDF + LLM V3 Toys"
echo "=========================================="

python scripts/infer_boost_sweep.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --config_files "$CONFIG_FILES" \
  --checkpoint "$CHECKPOINT" \
  --infer_boost_grid "$INFER_BOOST_GRID" \
  --valid_metric "$VALID_METRIC" \
  --variant_label "$VARIANT_LABEL" \
  --gpu_id "$GPU_ID"

echo "✅ Done!"
