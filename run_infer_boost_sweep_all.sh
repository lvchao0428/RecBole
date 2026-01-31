#!/usr/bin/env bash
# ============================================================
# Infer Boost Sweep - 所有 V3 模型的 infer_boost 网格搜索
# ============================================================
#
# 用途：
#   对 run_0125Batch_v3.sh 中训练的所有模型进行 infer_boost 网格搜索
#   在 validation set 上找到最优 γ*，然后固定 γ* 报告 test 结果
#
# 使用方法：
#   1. 确保所有模型已经训练完成（运行过 run_0125Batch_v3.sh）
#   2. 修改下面的 CHECKPOINT_DIR 和各模型的 checkpoint 路径
#   3. 运行: sh run_infer_boost_sweep_all.sh
#
# 网格搜索策略：
#   - 粗扫：{0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0} (7个点)
#   - 细扫：在最优附近 ±0.2，步长 0.1 (约2-3个点)
#   - 总计约 6-10 个点，符合 "每个模型都用同样的 search budget"
#
# ============================================================

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# GPU ID (可通过环境变量设置)
GPU_ID=${GPU_ID:-0}

# 输出目录
OUTPUT_DIR="./run_metrics/infer_boost_sweep"
mkdir -p "$OUTPUT_DIR"

# Infer boost 网格（粗扫）
COARSE_GRID="0.0,0.5,0.8,1.0,1.2,1.5,2.0"

# 细扫参数
FINE_RANGE=0.2
FINE_STEP=0.1

# 验证指标
VALID_METRIC="MRR@10"

# ============================================================
# Checkpoint 目录配置
# ============================================================
# 请根据实际训练输出修改这些路径
CHECKPOINT_BASE="./saved"

# Beauty 数据集的 checkpoint 目录
BEAUTY_TFIDF_CKPT_DIR="${CHECKPOINT_BASE}/phase_runs_tfidf_v3_stratified"
BEAUTY_TFIDF_LLM_CKPT_DIR="${CHECKPOINT_BASE}/phase_runs_tfidf_llm_v3_stratified"
BEAUTY_MV_7B_CKPT_DIR="${CHECKPOINT_BASE}/phase_runs_multiview_v3_stratified"
BEAUTY_MV_14B_CKPT_DIR="${CHECKPOINT_BASE}/phase_runs_multiview_v3_stratified_14b"

# Toys 数据集的 checkpoint 目录
TOYS_TFIDF_CKPT_DIR="${CHECKPOINT_BASE}/phase_runs_tfidf_v3_toys_stratified"
TOYS_TFIDF_LLM_CKPT_DIR="${CHECKPOINT_BASE}/phase_runs_tfidf_llm_v3_toys_stratified"
TOYS_MV_7B_CKPT_DIR="${CHECKPOINT_BASE}/phase_runs_multiview_v3_toys_stratified_7b"
TOYS_MV_14B_CKPT_DIR="${CHECKPOINT_BASE}/phase_runs_multiview_v3_toys_stratified_14b"

# ============================================================
# 辅助函数
# ============================================================

find_latest_checkpoint() {
    local dir="$1"
    local pattern="$2"
    
    if [ ! -d "$dir" ]; then
        echo ""
        return
    fi
    
    # 查找最新的 checkpoint 文件
    local ckpt=$(ls -t "$dir"/${pattern}*.pth 2>/dev/null | head -1)
    echo "$ckpt"
}

run_sweep() {
    local model="$1"
    local dataset="$2"
    local config_file="$3"
    local checkpoint="$4"
    local variant_label="$5"
    
    if [ -z "$checkpoint" ] || [ ! -f "$checkpoint" ]; then
        echo "❌ Checkpoint not found for $variant_label"
        echo "   Expected: $checkpoint"
        echo "   Skipping..."
        echo ""
        return 1
    fi
    
    echo "=============================================="
    echo "Running sweep for: $variant_label"
    echo "  Model: $model"
    echo "  Dataset: $dataset"
    echo "  Checkpoint: $checkpoint"
    echo "=============================================="
    
    python scripts/infer_boost_sweep.py \
        --model "$model" \
        --dataset "$dataset" \
        --config_files "$config_file" \
        --checkpoint "$checkpoint" \
        --gpu_id "$GPU_ID" \
        --infer_boost_grid "$COARSE_GRID" \
        --fine_sweep_range "$FINE_RANGE" \
        --fine_sweep_step "$FINE_STEP" \
        --valid_metric "$VALID_METRIC" \
        --output_dir "$OUTPUT_DIR" \
        --variant_label "$variant_label"
    
    local status=$?
    
    if [ $status -eq 0 ]; then
        echo "✅ Completed: $variant_label"
    else
        echo "❌ Failed: $variant_label"
    fi
    echo ""
    
    return $status
}

# ============================================================
# 主流程
# ============================================================

echo "=============================================="
echo "Infer Boost Sweep - All V3 Models"
echo "=============================================="
echo "GPU: $GPU_ID"
echo "Coarse Grid: $COARSE_GRID"
echo "Fine Range: ±$FINE_RANGE, Step: $FINE_STEP"
echo "Valid Metric: $VALID_METRIC"
echo "Output: $OUTPUT_DIR"
echo "=============================================="
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# ========== Beauty Dataset ==========
echo "========== Beauty Dataset =========="
echo ""

# 1. Beauty TF-IDF V3
CKPT=$(find_latest_checkpoint "$BEAUTY_TFIDF_CKPT_DIR" "SASRecAlignV3")
run_sweep \
    "SASRecAlignV3" \
    "Amazon_Beauty" \
    "sasrec_align_v3_stratified.yaml" \
    "$CKPT" \
    "Beauty_TFIDF_V3"

# 2. Beauty TF-IDF + LLM V3
CKPT=$(find_latest_checkpoint "$BEAUTY_TFIDF_LLM_CKPT_DIR" "SASRecAlignV3")
run_sweep \
    "SASRecAlignV3" \
    "Amazon_Beauty" \
    "sasrec_align_v3_llm_stratified.yaml" \
    "$CKPT" \
    "Beauty_TFIDF_LLM_V3"

# 3. Beauty Multi-View V3 (7B)
CKPT=$(find_latest_checkpoint "$BEAUTY_MV_7B_CKPT_DIR" "SASRecAlignMultiViewV3")
run_sweep \
    "SASRecAlignMultiViewV3" \
    "Amazon_Beauty" \
    "sasrec_align_multi_view_v3_stratified.yaml" \
    "$CKPT" \
    "Beauty_MultiView_V3_7B"

# 4. Beauty Multi-View V3 (14B)
CKPT=$(find_latest_checkpoint "$BEAUTY_MV_14B_CKPT_DIR" "SASRecAlignMultiViewV3")
run_sweep \
    "SASRecAlignMultiViewV3" \
    "Amazon_Beauty" \
    "sasrec_align_multi_view_v3_stratified_14b.yaml" \
    "$CKPT" \
    "Beauty_MultiView_V3_14B"

# ========== Toys Dataset ==========
echo ""
echo "========== Toys Dataset =========="
echo ""

# 5. Toys TF-IDF V3
CKPT=$(find_latest_checkpoint "$TOYS_TFIDF_CKPT_DIR" "SASRecAlignV3")
run_sweep \
    "SASRecAlignV3" \
    "Amazon_Toys_and_Games" \
    "sasrec_align_v3_toys_stratified.yaml" \
    "$CKPT" \
    "Toys_TFIDF_V3"

# 6. Toys TF-IDF + LLM V3
CKPT=$(find_latest_checkpoint "$TOYS_TFIDF_LLM_CKPT_DIR" "SASRecAlignV3")
run_sweep \
    "SASRecAlignV3" \
    "Amazon_Toys_and_Games" \
    "sasrec_align_v3_llm_toys_stratified.yaml" \
    "$CKPT" \
    "Toys_TFIDF_LLM_V3"

# 7. Toys Multi-View V3 (7B)
CKPT=$(find_latest_checkpoint "$TOYS_MV_7B_CKPT_DIR" "SASRecAlignMultiViewV3")
run_sweep \
    "SASRecAlignMultiViewV3" \
    "Amazon_Toys_and_Games" \
    "sasrec_align_multi_view_v3_toys_stratified_7b.yaml" \
    "$CKPT" \
    "Toys_MultiView_V3_7B"

# 8. Toys Multi-View V3 (14B)
CKPT=$(find_latest_checkpoint "$TOYS_MV_14B_CKPT_DIR" "SASRecAlignMultiViewV3")
run_sweep \
    "SASRecAlignMultiViewV3" \
    "Amazon_Toys_and_Games" \
    "sasrec_align_multi_view_v3_toys_stratified_14b.yaml" \
    "$CKPT" \
    "Toys_MultiView_V3_14B"

# ============================================================
# 完成
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "✅ All Infer Boost Sweeps Completed!"
echo "=============================================="
echo "Total time: ${ELAPSED}s"
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Check the JSON files for detailed results:"
echo "  ls -la $OUTPUT_DIR/*.json"
echo ""
