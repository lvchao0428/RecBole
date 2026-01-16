#!/bin/bash
# ============================================================
# 新增实验批次 - 完善论文Ablation
# 执行时间: 2026-01-16
# 总计: 5个核心实验 (可扩展到8个)
# ============================================================

# ============================================================
# 方案A: 5个核心实验 (推荐)
# ============================================================

echo "启动5个核心ablation实验..."

# 5090 (1张卡)
echo "5090-0: Beauty No-Whiten"
GPU_ID=0 nohup bash experiments/exp_ablation_beauty_nowhiten.sh > logs/ablation_beauty_nowhiten.log 2>&1 &

# 4090 (4张卡)
echo "4090-0: Beauty No-Cross"
GPU_ID=0 nohup bash experiments/exp_ablation_beauty_nocross.sh > logs/ablation_beauty_nocross.log 2>&1 &

echo "4090-1: Toys No Cold-start Reweight"
GPU_ID=1 nohup bash experiments/exp_ablation_no_cold_reweight_toys.sh > logs/ablation_no_cold_reweight.log 2>&1 &

echo "4090-2: Toys Center-only Normalization"
GPU_ID=2 nohup bash experiments/exp_ablation_center_only_toys.sh > logs/ablation_center_only.log 2>&1 &

echo "4090-3: TF-IDF Toys with Boost"
GPU_ID=3 nohup bash experiments/exp_tfidf_toys_with_boost.sh > logs/tfidf_toys_boost.log 2>&1 &

echo "核心5个实验已启动！"
echo "预计完成时间: ~2-3小时"

# ============================================================
# 方案B: 扩展到8个实验 (如需修复层级反转问题)
# ============================================================
# 
# 额外3个实验:
# echo "4090-4: TF-IDF+LLM Toys with Boost"
# GPU_ID=4 nohup bash experiments/exp_tfidf_llm_toys_with_boost.sh > logs/tfidf_llm_toys_boost.log 2>&1 &
# 
# echo "4090-5: TF-IDF Beauty with Boost"
# GPU_ID=5 nohup bash experiments/exp_tfidf_beauty_with_boost.sh > logs/tfidf_beauty_boost.log 2>&1 &
# 
# echo "4090-6: TF-IDF+LLM Beauty with Boost"
# GPU_ID=6 nohup bash experiments/exp_tfidf_llm_beauty_with_boost.sh > logs/tfidf_llm_beauty_boost.log 2>&1 &

# ============================================================
# 检查实验状态
# ============================================================

echo ""
echo "检查运行中的实验:"
ps aux | grep "exp_ablation\|exp_tfidf.*with_boost" | grep -v grep

echo ""
echo "实时监控日志 (选择一个):"
echo "  tail -f logs/ablation_beauty_nowhiten.log"
echo "  tail -f logs/ablation_beauty_nocross.log"
echo "  tail -f logs/ablation_no_cold_reweight.log"
echo "  tail -f logs/ablation_center_only.log"
echo "  tail -f logs/tfidf_toys_boost.log"

# ============================================================
# 实验说明
# ============================================================

cat << 'EOF'

实验目的说明:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

实验1: Beauty No-Whiten
  - 目的: 补全Table ablation_whiten的Beauty数据
  - 预期: 验证whitening在Beauty上的效果
  - 论文位置: line 784-802

实验2: Beauty No-Cross  
  - 目的: 补全Table ablation_senet_cross的Beauty数据
  - 预期: 验证cross network的HR-MRR trade-off
  - 论文位置: line 732-751

实验3: No Cold-start Reweight
  - 目的: 兑现Appendix line 1173承诺
  - 预期: 验证cold-start reweighting的必要性
  - 论文位置: Appendix line 1171-1177

实验4: Center-only Normalization
  - 目的: 兑现Appendix line 1175承诺
  - 预期: 对比ZCA whitening vs center-only
  - 论文位置: Appendix line 1171-1177

实验5: TF-IDF with Boost
  - 目的: 修复层级反转问题 (TF-IDF HR_new < ID-only)
  - 预期: fair comparison下TF-IDF应超过ID-only
  - 论文位置: 可能更新main table或Discussion

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

实验完成后的论文提升:
✅ 兑现所有Appendix承诺
✅ Beauty和Toys的ablation数据对称完整
✅ 增强论文的robustness claims
✅ 修复潜在的reviewer concern
✅ 提供更深入的component analysis

EOF
