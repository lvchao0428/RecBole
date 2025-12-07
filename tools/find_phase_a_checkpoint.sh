#!/usr/bin/env bash
# 辅助工具：查找 Phase A checkpoint

echo "========================================"
echo "查找 Phase A Checkpoint"
echo "========================================"
echo ""

# 默认搜索目录
CHECKPOINT_DIRS=(
    "./saved/phase_runs_multiview_4views"
    "./saved"
    "./saved/phase_runs"
)

echo "搜索目录："
for dir in "${CHECKPOINT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir"
    else
        echo "  ❌ $dir (不存在)"
    fi
done
echo ""

# 查找所有 Phase A checkpoint
echo "查找 *-phase_a.pth 文件..."
echo ""

FOUND_CHECKPOINTS=()
for dir in "${CHECKPOINT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        while IFS= read -r -d '' file; do
            FOUND_CHECKPOINTS+=("$file")
        done < <(find "$dir" -name "*-phase_a.pth" -type f -print0 2>/dev/null)
    fi
done

if [ ${#FOUND_CHECKPOINTS[@]} -eq 0 ]; then
    echo "❌ 未找到 Phase A checkpoint"
    echo ""
    echo "可能的原因："
    echo "  1. Phase A 还未运行完成"
    echo "  2. 运行 Phase A 时未使用 --save 参数"
    echo "  3. Checkpoint 保存在其他目录"
    echo ""
    echo "建议："
    echo "  1. 检查 Phase A 运行日志"
    echo "  2. 查找 '[Phase-A] Saved checkpoint:' 输出"
    echo "  3. 手动搜索: find ./saved -name '*-phase_a.pth'"
    echo ""
    exit 1
fi

echo "找到 ${#FOUND_CHECKPOINTS[@]} 个 Phase A checkpoint："
echo ""

# 按时间排序并显示
for i in "${!FOUND_CHECKPOINTS[@]}"; do
    ckpt="${FOUND_CHECKPOINTS[$i]}"
    size=$(ls -lh "$ckpt" | awk '{print $5}')
    mtime=$(ls -l "$ckpt" | awk '{print $6, $7, $8}')
    echo "[$((i+1))] $ckpt"
    echo "    大小: $size"
    echo "    修改时间: $mtime"
    echo ""
done

# 推荐最新的
echo "========================================"
echo "推荐使用："
echo "========================================"

# 找到最新的文件
LATEST_CKPT=""
LATEST_TIME=0
for ckpt in "${FOUND_CHECKPOINTS[@]}"; do
    ckpt_time=$(stat -f "%m" "$ckpt" 2>/dev/null || stat -c "%Y" "$ckpt" 2>/dev/null)
    if [ "$ckpt_time" -gt "$LATEST_TIME" ]; then
        LATEST_TIME=$ckpt_time
        LATEST_CKPT=$ckpt
    fi
done

echo "最新的 checkpoint："
echo "  $LATEST_CKPT"
echo ""
echo "使用方法："
echo "  1. 编辑 two_phase_run_multiview_split_phaseB_only.sh"
echo "  2. 设置 PHASE_A_CHECKPOINT=\"$LATEST_CKPT\""
echo "  3. 运行: bash two_phase_run_multiview_split_phaseB_only.sh"
echo ""

# 也可以输出一个临时脚本
TEMP_SCRIPT="./run_phaseB_from_latest.sh"
cat > "$TEMP_SCRIPT" << EOF
#!/usr/bin/env bash
# 自动生成的 Phase B 启动脚本
# 生成时间: $(date '+%Y-%m-%d %H:%M:%S')

cd /home/charlie/project/RecBole
export PYTHONPATH="\$(pwd):\${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PHASE_A_CHECKPOINT="$LATEST_CKPT"

echo "从 Phase A checkpoint 启动 Phase B:"
echo "  \$PHASE_A_CHECKPOINT"
echo ""

python scripts/two_phase_train.py \\
  --model SASRecAlignMultiView \\
  --dataset Amazon_Beauty \\
  --config_files "sasrec_align_multi_view.yaml" \\
  --only_phase_b \\
  --resume_from "\$PHASE_A_CHECKPOINT" \\
  --phase_b_alignment_weight 0.05 \\
  --phase_b_text_gate_reg_l2 0.05 \\
  --phase_b_text_weight 0.8 \\
  --phase_b_epochs 40 \\
  --lr_text_head 1e-3 \\
  --lr_dnn_cross 5e-4 \\
  --backbone_lr_scale 0.1 \\
  --checkpoint_dir ./saved/phase_runs_multiview_4views \\
  --seed 2025 \\
  --variant_features "sasrec,multiview,4views,per_view_align,qwen3,phase_b_only" \\
  --watchdog_disable \\
  --save

echo ""
echo "✅ Phase B 完成！"
EOF

chmod +x "$TEMP_SCRIPT"

echo "快速启动："
echo "  已生成临时脚本: $TEMP_SCRIPT"
echo "  直接运行: bash $TEMP_SCRIPT"
echo ""
echo "========================================"

