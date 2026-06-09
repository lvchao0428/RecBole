#!/usr/bin/env bash
# 编译 TikZ 架构图为 PDF

set -e

cd "$(dirname "$0")"

echo "========================================"
echo "编译 MV-Align 架构图"
echo "========================================"
echo ""

if ! command -v pdflatex &> /dev/null; then
    echo "❌ 错误: pdflatex 未安装"
    echo ""
    echo "安装方法:"
    echo "  macOS:   brew install --cask mactex-no-gui"
    echo "  Ubuntu:  sudo apt install texlive-latex-base texlive-latex-extra"
    echo ""
    echo "或使用在线工具: https://www.overleaf.com/"
    exit 1
fi

echo "[1/2] 编译 framework.tex ..."
pdflatex -interaction=nonstopmode framework.tex > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 编译成功!"
    echo ""
    echo "生成文件:"
    echo "  - framework.pdf  (论文用)"
    echo ""
    
    # 清理临时文件
    rm -f framework.aux framework.log
    
    echo "[2/2] 预览 PDF..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open framework.pdf
    elif command -v xdg-open &> /dev/null; then
        xdg-open framework.pdf
    else
        echo "请手动打开: paper_sigir/figures/framework.pdf"
    fi
else
    echo "❌ 编译失败，请检查 LaTeX 错误"
    cat framework.log | tail -20
    exit 1
fi

echo ""
echo "========================================"
echo "✅ 完成!"
echo "========================================"
