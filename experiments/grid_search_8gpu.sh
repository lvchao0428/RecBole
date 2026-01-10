#!/usr/bin/env bash
# ============================================================
# 8-GPU Grid Search for TF-IDF+LLM and Multi-View Optimization
# 目标：
#   1. TF-IDF+LLM > TF-IDF（整体指标）
#   2. Multi-View 7B > TF-IDF+LLM（冷启动略强）
#   3. 为 14B/32B Scale Law 找到最佳基础配置
# ============================================================
#
# 实验设计（精简版）:
# ============================================================
# TF-IDF+LLM (GPU 0-3):
#   A: 原版复现 - tau=0.05, align=0.15, text_wt=1.0, cold=0
#   C: 中等对齐 - tau=0.05, align=0.10, text_wt=1.0, cold=2.0
#   D: 高温度   - tau=0.07, align=0.12, text_wt=1.0, cold=2.0
#   H: 综合最优 - tau=0.05, align=0.12, text_wt=1.0, cold=2.5
#
# Multi-View 7B (GPU 4-7):
#   M1: 基础    - tau=0.05, align=0.10, text_wt=1.0, cold=2.0
#   M2: 冷启动  - tau=0.05, align=0.12, text_wt=1.0, cold=3.0
#   M3: 高对齐  - tau=0.05, align=0.15, text_wt=0.9, cold=2.0
#   M4: 激进    - tau=0.05, align=0.12, text_wt=1.0, cold=4.0
# ============================================================
#
# 运行方式（手动启动每个实验）：
#   bash experiments/exp_A_original.sh 0     # GPU 0
#   bash experiments/exp_C_moderate.sh 1     # GPU 1
#   bash experiments/exp_D_high_tau.sh 2     # GPU 2
#   bash experiments/exp_H_optimal.sh 3      # GPU 3
#   bash experiments/exp_M1_mv_base.sh 4     # GPU 4
#   bash experiments/exp_M2_mv_cold.sh 5     # GPU 5
#   bash experiments/exp_M3_mv_high_align.sh 6  # GPU 6
#   bash experiments/exp_M4_mv_aggressive.sh 7  # GPU 7
#
# 参数说明：
#   每个脚本接受一个可选的 GPU_ID 参数
#   例如: bash experiments/exp_A_original.sh 3  # 在 GPU 3 上运行
#
# 预期结果：
#   - Exp A/C/D/H 中找到让 TF-IDF+LLM > TF-IDF 的配置
#   - Exp M1/M2/M3/M4 中找到让 Multi-View > TF-IDF+LLM 的配置
#   - 冷启动指标 (MRR_new, NDCG_new) 明显提升
# ============================================================

echo "请运行: bash experiments/run_all_8gpu.sh"
echo "或查看各实验脚本: ls experiments/exp_*.sh"
