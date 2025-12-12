#!/bin/bash
# 完整的数据集分析工作流
# 包括统计分析、可视化和表格生成

echo "================================="
echo "数据集完整分析工作流"
echo "================================="
echo ""

# 检查参数
if [ "$1" == "example" ]; then
    echo "[1/3] 分析示例数据集..."
    python3 dataset_statistics.py --example --output analysis_results.json
    
elif [ "$1" == "full" ]; then
    if [ -z "$2" ]; then
        echo "错误: 请提供服务器数据集路径"
        echo "用法: $0 full /path/to/server/datasets"
        exit 1
    fi
    echo "[1/3] 分析服务器完整数据集..."
    echo "服务器路径: $2"
    python3 dataset_statistics.py --full --server-path "$2" --output analysis_results.json
    
else
    echo "用法:"
    echo "  $0 example              # 分析示例数据集"
    echo "  $0 full <server_path>   # 分析服务器完整数据集"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "错误: 统计分析失败"
    exit 1
fi

echo ""
echo "[2/3] 生成可视化图表和表格..."
python3 visualize_dataset_stats.py \
    --input analysis_results.json \
    --output-prefix analysis \
    --markdown \
    --latex

if [ $? -ne 0 ]; then
    echo "警告: 可视化生成部分失败（可能缺少matplotlib）"
fi

echo ""
echo "[3/3] 生成完整报告..."

# 创建报告文件
REPORT_FILE="analysis_report.txt"

cat > $REPORT_FILE << 'EOF'
================================================================================
数据集统计分析报告
================================================================================

生成时间: $(date)

EOF

echo "生成时间: $(date)" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "生成文件列表:" >> $REPORT_FILE
echo "  - analysis_results.json       (详细统计数据)" >> $REPORT_FILE

if [ -f "analysis_table.md" ]; then
    echo "  - analysis_table.md          (Markdown表格)" >> $REPORT_FILE
fi

if [ -f "analysis_table.tex" ]; then
    echo "  - analysis_table.tex         (LaTeX表格)" >> $REPORT_FILE
fi

if [ -f "analysis_basic_stats.png" ]; then
    echo "  - analysis_basic_stats.png   (基础统计图表)" >> $REPORT_FILE
    echo "  - analysis_distribution.png  (分布图表)" >> $REPORT_FILE
    echo "  - analysis_gini.png          (Gini系数图表)" >> $REPORT_FILE
fi

echo "  - $REPORT_FILE                (本报告)" >> $REPORT_FILE
echo "" >> $REPORT_FILE

# 如果生成了Markdown表格，添加到报告中
if [ -f "analysis_table.md" ]; then
    echo "数据集对比表格:" >> $REPORT_FILE
    echo "--------------------------------------------------------------------------------" >> $REPORT_FILE
    cat analysis_table.md >> $REPORT_FILE
    echo "" >> $REPORT_FILE
fi

echo "=================================================================================" >> $REPORT_FILE
echo "分析完成!" >> $REPORT_FILE
echo "=================================================================================" >> $REPORT_FILE

echo ""
echo "================================="
echo "✓ 分析完成!"
echo "================================="
echo ""
echo "生成的文件:"
ls -lh analysis_* 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
echo ""
echo "查看报告: cat $REPORT_FILE"
echo "查看JSON: cat analysis_results.json | python3 -m json.tool"
if [ -f "analysis_table.md" ]; then
    echo "查看表格: cat analysis_table.md"
fi
echo ""


