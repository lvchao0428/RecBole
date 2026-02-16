#!/usr/bin/env bash
# V3 Batch Training Script
# 使用简化的 V3 模型和权重配置
# 对应原始 run_0125Batch.sh 的 V3 版本

echo "=============================================="
echo "V3 Batch Training - Simplified Weight Config"
echo "=============================================="
echo "Models:"
echo "  - SASRecAlignV3 (single-view)"
echo "  - SASRecAlignMultiViewV3 (multi-view)"
echo ""
echo "V3 Core Weights:"
echo "  - align_weight: 全局对齐权重"
echo "  - cold_text_boost: 冷启动训练增强"
echo "  - infer_boost: 推理时冷启动增强"
echo "=============================================="
echo ""

# ========== Beauty Dataset ==========
echo "========== Beauty Dataset =========="

echo "1. [Beauty] TF-IDF V3"
nohup sh two_phase_run_tfidf_v3_stratified.sh > two_phase_run_tfidf_v3_stratified.log 2>&1
echo "   Done!"

echo "2. [Beauty] TF-IDF + LLM V3"
nohup sh two_phase_run_tfidf_llm_v3_stratified.sh > two_phase_run_tfidf_llm_v3_stratified.log 2>&1
echo "   Done!"

echo "3. [Beauty] Multi-View V3 (7B)"
nohup sh two_phase_run_multiview_v3_stratified.sh > two_phase_run_multiview_v3_stratified.log 2>&1
echo "   Done!"

echo "4. [Beauty] Multi-View V3 (14B)"
nohup sh two_phase_run_multiview_v3_stratified_14b.sh > two_phase_run_multiview_v3_stratified_14b.log 2>&1
echo "   Done!"

# ========== Toys Dataset ==========
echo ""
echo "========== Toys Dataset =========="

echo "5. [Toys] TF-IDF V3"
nohup sh two_phase_run_tfidf_v3_toys_stratified.sh > two_phase_run_tfidf_v3_toys_stratified.log 2>&1
echo "   Done!"

echo "6. [Toys] TF-IDF + LLM V3"
nohup sh two_phase_run_tfidf_llm_v3_toys_stratified.sh > two_phase_run_tfidf_llm_v3_toys_stratified.log 2>&1
echo "   Done!"

echo "7. [Toys] Multi-View V3 (7B)"
nohup sh two_phase_run_multiview_v3_toys_stratified_7b.sh > two_phase_run_multiview_v3_toys_stratified_7b.log 2>&1
echo "   Done!"

echo "8. [Toys] Multi-View V3 (14B)"
nohup sh two_phase_run_multiview_v3_toys_stratified_14b.sh > two_phase_run_multiview_v3_toys_stratified_14b.log 2>&1
echo "   Done!"

echo ""
echo "=============================================="
echo "✅ All V3 Training Tasks Submitted!"
echo "=============================================="
echo ""
echo "Check logs:"
echo "  - two_phase_run_tfidf_v3_stratified.log"
echo "  - two_phase_run_tfidf_llm_v3_stratified.log"
echo "  - two_phase_run_multiview_v3_stratified.log"
echo "  - two_phase_run_multiview_v3_stratified_14b.log"
echo "  - two_phase_run_tfidf_v3_toys_stratified.log"
echo "  - two_phase_run_tfidf_llm_v3_toys_stratified.log"
echo "  - two_phase_run_multiview_v3_toys_stratified_7b.log"
echo "  - two_phase_run_multiview_v3_toys_stratified_14b.log"
echo ""
