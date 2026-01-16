#!/bin/bash
# ============================================================
# 批次 2: No-Whiten Ablation实验 (Aggressive配置)
# 执行时间: 2026-01-16
# 总计: 4个实验 (Beauty 2个 + Toys 2个)
# ============================================================

echo "=========================================="
echo "启动 No-Whiten Ablation 实验"
echo "配置: Aggressive (cold=2.5, infer=1.5)"
echo "目的: 补全论文Table ablation_whiten"
echo "=========================================="
echo ""

# 创建日志目录
mkdir -p logs

# Beauty数据集 (2个实验)
echo "[1/4] 启动: Beauty TF-IDF+LLM No-Whiten"
GPU_ID=0 nohup bash beauty_train/two_phase_run_tfidf_llm_stratified_no_whiten.sh > logs/beauty_tfidf_llm_no_whiten.log 2>&1 &
PID1=$!
echo "  PID: $PID1"
echo "  日志: logs/beauty_tfidf_llm_no_whiten.log"
echo ""

sleep 2

echo "[2/4] 启动: Beauty MV-7B No-Whiten"
GPU_ID=1 nohup bash beauty_train/two_phase_run_multiview_v2_stratified_no_whiten.sh > logs/beauty_mv_no_whiten.log 2>&1 &
PID2=$!
echo "  PID: $PID2"
echo "  日志: logs/beauty_mv_no_whiten.log"
echo ""

sleep 2

# Toys数据集 (2个实验)
echo "[3/4] 启动: Toys TF-IDF+LLM No-Whiten"
GPU_ID=2 nohup bash toy_train/two_phase_run_tfidf_llm_toys_stratified_no_whiten.sh > logs/toys_tfidf_llm_no_whiten.log 2>&1 &
PID3=$!
echo "  PID: $PID3"
echo "  日志: logs/toys_tfidf_llm_no_whiten.log"
echo ""

sleep 2

echo "[4/4] 启动: Toys MV-7B No-Whiten"
GPU_ID=3 nohup bash toy_train/two_phase_run_multiview_v2_toys_stratified_7b_no_whiten.sh > logs/toys_mv_no_whiten.log 2>&1 &
PID4=$!
echo "  PID: $PID4"
echo "  日志: logs/toys_mv_no_whiten.log"
echo ""

echo "=========================================="
echo "✅ 所有4个实验已启动！"
echo "=========================================="
echo ""

# 显示进程信息
echo "运行中的实验进程:"
ps aux | grep "two_phase_run.*no_whiten" | grep -v grep | awk '{print "  PID: "$2" | GPU: "$12" | Script: "$14}'
echo ""

# 监控命令
echo "=========================================="
echo "监控命令:"
echo "=========================================="
echo "查看所有日志:"
echo "  tail -f logs/beauty_tfidf_llm_no_whiten.log"
echo "  tail -f logs/beauty_mv_no_whiten.log"
echo "  tail -f logs/toys_tfidf_llm_no_whiten.log"
echo "  tail -f logs/toys_mv_no_whiten.log"
echo ""
echo "检查进程状态:"
echo "  ps aux | grep 'no_whiten' | grep -v grep"
echo ""
echo "检查GPU使用:"
echo "  nvidia-smi"
echo ""

# 预期结果说明
cat << 'EOF'
========================================== 
预期结果
==========================================

📊 论文Table ablation_whiten (line 784-802)
当前状态: 仅有Toys数据，缺Beauty

实验完成后将补充:
┌─────────────────────────────────────┐
│ Beauty TF-IDF+LLM                   │
│  w/ Whiten:  5.74% HR@10 (已有)     │
│  w/o Whiten: ?.??% HR@10 (新增)     │
│                                      │
│ Beauty MV-7B                         │
│  w/ Whiten:  6.07% HR@10 (已有)     │
│  w/o Whiten: ?.??% HR@10 (新增)     │
└─────────────────────────────────────┘

预期效果:
- Multi-view: no-whiten应降低性能
- Single-view: no-whiten可能略微提升
- 与Toys数据趋势一致

论文更新:
1. 扩展Table ablation_whiten为双数据集对比
2. 增强whitening对multi-view的必要性claim
3. 解释单/多view对whitening的不同响应

预计完成时间: ~2-3小时

==========================================
EOF

# 实时状态检查函数
echo ""
echo "按Ctrl+C退出监控，实验将继续在后台运行"
echo ""

# 简单的状态轮询
while true; do
    sleep 60
    RUNNING=$(ps aux | grep "two_phase_run.*no_whiten" | grep -v grep | wc -l)
    echo "[$(date '+%H:%M:%S')] 运行中的实验: $RUNNING/4"
    
    if [ $RUNNING -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✅ 所有实验已完成！"
        echo "=========================================="
        echo "请检查日志文件提取结果"
        break
    fi
done
