#!/usr/bin/env bash
# Infer Boost Sweep - TF-IDF V3 Beauty

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ==========================================
# 配置区域
# ==========================================
CHECKPOINT="./saved/two_phase_run_tfidf_v3_stratified/SASRecAlignV3-Jan-29-2026_14-05-45.pth"
MODEL="SASRecAlignV3"
DATASET="Amazon_Beauty"
CONFIG_FILES="sasrec_align_base_stratified_v3.yaml"
INFER_BOOST_GRID="0.0,0.5,0.8,1.0,1.2,1.5,2.0"
VALID_METRIC="MRR@10"
VARIANT_LABEL="tfidf_v3_beauty"
GPU_ID=${GPU_ID:-0}

# ==========================================
# 检查 checkpoint
# ==========================================
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $CHECKPOINT"
    echo "查找: ls -lht saved/phase_runs_tfidf_v3_stratified/*.pth"
    exit 1
fi

echo "=========================================="
echo "Infer Boost Sweep - TF-IDF V3 Beauty"
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
