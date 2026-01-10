#!/usr/bin/env bash
# ============================================================
# 8-GPU 实验设计 (2026-01-11 更新)
# ============================================================
# 核心修改: burn-in 从 10 epochs 改为 2 epochs (折中方案)
# 原因: 10 epoch burn-in 导致 ID 过拟合，Phase-A 无法追上
# ============================================================

echo "=========================================="
echo "8-GPU 实验设计 - Burn-in 折中方案"
echo "=========================================="
echo ""
echo "实验列表:"
echo "  GPU 0: exp_burn0     - 无 burn-in 测试 (对照组)"
echo "  GPU 1: exp_burn2     - 折中方案测试 (burn-in=2)"
echo "  GPU 2: exp_A         - TF-IDF+LLM 原版 (burn-in=2)"
echo "  GPU 3: exp_H         - TF-IDF+LLM 最优 (burn-in=2)"
echo "  GPU 4: exp_M1        - Multi-View 基础 (burn-in=2)"
echo "  GPU 5: exp_M2        - Multi-View 冷启动 (burn-in=2)"
echo "  GPU 6: exp_M3        - Multi-View 高对齐 (burn-in=2)"
echo "  GPU 7: exp_M4        - Multi-View 激进 (burn-in=2)"
echo ""
echo "手动启动命令:"
echo "----------------------------------------"
echo "bash experiments/exp_burn0_no_burnin.sh 0   # GPU 0: 无 burn-in"
echo "bash experiments/exp_burn2_compromise.sh 1  # GPU 1: burn-in=2"
echo "bash experiments/exp_A_original.sh 2        # GPU 2: LLM 原版"
echo "bash experiments/exp_H_optimal.sh 3         # GPU 3: LLM 最优"
echo "bash experiments/exp_M1_mv_base.sh 4        # GPU 4: MV 基础"
echo "bash experiments/exp_M2_mv_cold.sh 5        # GPU 5: MV 冷启动"
echo "bash experiments/exp_M3_mv_high_align.sh 6  # GPU 6: MV 高对齐"
echo "bash experiments/exp_M4_mv_aggressive.sh 7  # GPU 7: MV 激进"
echo "----------------------------------------"
echo ""
echo "预期结果:"
echo "  1. exp_burn0 vs exp_burn2: 验证 burn-in 是否必要"
echo "  2. 所有 burn-in=2 实验: Phase-A 应能持续涨点"
echo "  3. Multi-View (M1-M4): 应超过 TF-IDF+LLM (A, H)"
echo ""
echo "关键观察指标:"
echo "  - Phase-A valid_score 是否持续提升"
echo "  - Phase-B 开始时 loss 是否正常 (不应大幅跳升)"
echo "  - 最终 MRR@10, MRR_new@10 表现"
echo "=========================================="
