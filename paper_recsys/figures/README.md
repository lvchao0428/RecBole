# 架构图生成指南

## 方法1: 本地编译 (推荐)

```bash
cd paper_sigir/figures
pdflatex framework.tex
```

生成的 `framework.pdf` 可直接在论文中使用。

## 方法2: Overleaf 在线编译 (最简单)

1. 访问 https://www.overleaf.com/
2. 创建新项目 → Upload Project
3. 上传 `framework.tex`
4. 自动编译，实时预览
5. 下载 PDF

## 方法3: 使用 draw.io + TikZ 插件

虽然 TikZ 是学术界标准，但如果需要更直观的可视化编辑：

1. 访问 https://app.diagrams.net/
2. 创建架构图，导出为 PDF
3. 使用 Inkscape 添加公式和细节

## 方法4: Python + Matplotlib (快速原型)

对于快速迭代，可以用 Python 生成：

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 见 generate_framework_matplotlib.py
```

## 推荐工作流

**学术论文最佳实践：TikZ**
- ✅ 矢量图，完美缩放
- ✅ 与 LaTeX 数学公式无缝集成
- ✅ 审稿人和出版社偏好
- ✅ 可编程，便于版本控制

**当前文件说明：**
- `framework.tex` - TikZ 源码（可直接编译）
- `framework.pdf` - 编译后的图（论文会自动引用）

**已在论文中配置：**
```latex
\IfFileExists{figures/framework.pdf}{
  \includegraphics[width=\linewidth]{figures/framework.pdf}
}{
  % 占位符
}
```

## 颜色方案

当前使用 SIGIR 风格学术配色：
- **ID Embedding**: 专业蓝 (RGB 65,105,225)
- **TF-IDF**: 深橙 (RGB 255,140,0)
- **LLM**: 海绿 (RGB 60,179,113)
- **Multi-view**: 粉/紫/青/金 四色
- **Cross Network**: 深红 (RGB 220,20,60)
- **Alignment**: 琥珀金 (RGB 255,191,0)

## 自定义修改

编辑 `framework.tex`，调整：
- 颜色: `\definecolor{...}{RGB}{...}`
- 布局: `node distance=0.7cm`
- 字体大小: `font=\small`
- 阴影效果: `drop shadow={...}`

编译后刷新即可看到效果。
