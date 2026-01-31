#!/usr/bin/env bash
# ============================================================
# 一键执行所有 V3 模型的 Infer Boost Sweep
# ============================================================
#
# 对应 run_0125Batch_v3.sh 中的所有模型：
#   Beauty: TF-IDF, TF-IDF+LLM, MultiView-7B, MultiView-14B
#   Toys:   TF-IDF, TF-IDF+LLM, MultiView-7B, MultiView-14B
#
# 使用方法：
#   1. 确保所有训练已完成并保存了 checkpoint
#   2. 修改各个 sweep 脚本中的 CHECKPOINT 路径
#   3. 运行本脚本: sh run_all_infer_boost_sweep.sh
#
# 注意：
#   - 每个模型使用相同的 search budget (7个点: 0.0,0.5,0.8,1.0,1.2,1.5,2.0)
#   - 不重新训练，只加载 checkpoint 做推理评估
#   - 结果保存在 ./run_metrics/infer_boost_sweep/

set -e  # 遇到错误时停止

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# GPU 设置（可通过环境变量覆盖）
GPU_ID=${GPU_ID:-0}
export GPU_ID

echo "============================================================"
echo "  Infer Boost Sweep - All V3 Models"
echo "============================================================"
echo "GPU: $GPU_ID"
echo "Grid: 0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0 (7 points)"
echo "Metric: MRR@10"
echo "============================================================"
echo ""

# 创建结果目录
mkdir -p ./run_metrics/infer_boost_sweep

# 记录开始时间
START_TIME=$(date +%s)

# ========== Beauty Dataset ==========
echo ""
echo "========== Beauty Dataset =========="
echo ""

echo "[1/8] Beauty - TF-IDF V3"
if [ -f "./sweep_infer_boost_tfidf_v3.sh" ]; then
    sh sweep_infer_boost_tfidf_v3.sh 2>&1 | tee -a ./run_metrics/infer_boost_sweep/sweep_tfidf_v3_beauty.log
else
    echo "  ⚠️ Script not found, skipping..."
fi
echo ""

echo "[2/8] Beauty - TF-IDF + LLM V3"
if [ -f "./sweep_infer_boost_tfidf_llm_v3.sh" ]; then
    sh sweep_infer_boost_tfidf_llm_v3.sh 2>&1 | tee -a ./run_metrics/infer_boost_sweep/sweep_tfidf_llm_v3_beauty.log
else
    echo "  ⚠️ Script not found, skipping..."
fi
echo ""

echo "[3/8] Beauty - Multi-View V3 (7B)"
if [ -f "./sweep_infer_boost_multiview_v3_7b.sh" ]; then
    sh sweep_infer_boost_multiview_v3_7b.sh 2>&1 | tee -a ./run_metrics/infer_boost_sweep/sweep_multiview_v3_7b_beauty.log
else
    echo "  ⚠️ Script not found, skipping..."
fi
echo ""

echo "[4/8] Beauty - Multi-View V3 (14B)"
if [ -f "./sweep_infer_boost_multiview_v3_14b.sh" ]; then
    sh sweep_infer_boost_multiview_v3_14b.sh 2>&1 | tee -a ./run_metrics/infer_boost_sweep/sweep_multiview_v3_14b_beauty.log
else
    echo "  ⚠️ Script not found, skipping..."
fi
echo ""

# ========== Toys Dataset ==========
echo ""
echo "========== Toys Dataset =========="
echo ""

echo "[5/8] Toys - TF-IDF V3"
if [ -f "./sweep_infer_boost_tfidf_v3_toys.sh" ]; then
    sh sweep_infer_boost_tfidf_v3_toys.sh 2>&1 | tee -a ./run_metrics/infer_boost_sweep/sweep_tfidf_v3_toys.log
else
    echo "  ⚠️ Script not found, skipping..."
fi
echo ""

echo "[6/8] Toys - TF-IDF + LLM V3"
if [ -f "./sweep_infer_boost_tfidf_llm_v3_toys.sh" ]; then
    sh sweep_infer_boost_tfidf_llm_v3_toys.sh 2>&1 | tee -a ./run_metrics/infer_boost_sweep/sweep_tfidf_llm_v3_toys.log
else
    echo "  ⚠️ Script not found, skipping..."
fi
echo ""

echo "[7/8] Toys - Multi-View V3 (7B)"
if [ -f "./sweep_infer_boost_multiview_v3_7b_toys.sh" ]; then
    sh sweep_infer_boost_multiview_v3_7b_toys.sh 2>&1 | tee -a ./run_metrics/infer_boost_sweep/sweep_multiview_v3_7b_toys.log
else
    echo "  ⚠️ Script not found, skipping..."
fi
echo ""

echo "[8/8] Toys - Multi-View V3 (14B)"
if [ -f "./sweep_infer_boost_multiview_v3_14b_toys.sh" ]; then
    sh sweep_infer_boost_multiview_v3_14b_toys.sh 2>&1 | tee -a ./run_metrics/infer_boost_sweep/sweep_multiview_v3_14b_toys.log
else
    echo "  ⚠️ Script not found, skipping..."
fi
echo ""

# 计算耗时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "============================================================"
echo "  All Infer Boost Sweeps Completed!"
echo "============================================================"
echo ""
echo "Total time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo ""
echo "Results saved to:"
echo "  ./run_metrics/infer_boost_sweep/"
echo ""
echo "JSON result files:"
ls -lht ./run_metrics/infer_boost_sweep/*.json 2>/dev/null || echo "  (no JSON files found)"
echo ""
echo "Log files:"
ls -lht ./run_metrics/infer_boost_sweep/*.log 2>/dev/null || echo "  (no log files found)"
echo ""
echo "============================================================"
