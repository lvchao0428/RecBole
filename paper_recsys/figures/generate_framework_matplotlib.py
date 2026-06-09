#!/usr/bin/env python3
"""
使用 Matplotlib 生成 MV-Align 架构图 (无需 LaTeX)

优点: 
- 无需安装 LaTeX
- Python 环境即可运行
- 可快速迭代调整

缺点:
- 不如 TikZ 精细
- 数学公式支持有限

使用:
    python generate_framework_matplotlib.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# 学术风格配色
COLORS = {
    'id': '#4169E1',        # 蓝色 - ID
    'tfidf': '#FF8C00',     # 橙色 - TF-IDF
    'llm': '#3CB371',       # 绿色 - LLM
    'view1': '#FF69B4',     # 粉色
    'view2': '#9370DB',     # 紫色
    'view3': '#48D1CC',     # 青色
    'view4': '#DAA520',     # 金色
    'cross': '#DC143C',     # 红色 - Cross
    'align': '#FFD700',     # 金色 - Align
}

def draw_box(ax, x, y, width, height, text, color, **kwargs):
    """绘制圆角矩形框"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.05", 
        edgecolor='black', facecolor=color,
        linewidth=1.5, zorder=2,
        **kwargs
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', 
            fontsize=9, fontweight='normal', zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, style='solid', color='black'):
    """绘制箭头"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', mutation_scale=15,
        linewidth=1.5, color=color,
        linestyle=style, zorder=1
    )
    ax.add_patch(arrow)

def main():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 11)
    ax.axis('off')
    
    # ========== 标题 ==========
    ax.text(3, 10, '(a) Single-View', fontsize=14, fontweight='bold')
    ax.text(10, 10, '(b) Multi-View (MV-Align)', fontsize=14, fontweight='bold')
    
    # ========== Single-View (左侧) ==========
    y_start = 9
    
    # Layer 1: 输入
    draw_box(ax, 2, y_start, 1.2, 0.5, 'TF-IDF', COLORS['tfidf'] + '40')
    draw_box(ax, 4, y_start, 1.2, 0.5, 'LLM\n7B', COLORS['llm'] + '40')
    
    # Layer 2: 投影
    draw_box(ax, 2, y_start-1.2, 1, 0.4, 'Proj', '#F0F0F0')
    draw_box(ax, 4, y_start-1.2, 1, 0.4, 'SVD', '#F0F0F0')
    
    # Layer 3: SENet
    draw_box(ax, 2, y_start-2.2, 1, 0.4, 'SENet', '#F0F0F0')
    draw_box(ax, 4, y_start-2.2, 1, 0.4, 'SENet', '#F0F0F0')
    
    # Layer 4: L2
    draw_box(ax, 4, y_start-3, 1, 0.4, 'L2', '#F0F0F0')
    ax.text(5.2, y_start-3, 'No Whiten', fontsize=7, style='italic', color='gray')
    
    # Layer 5: Concat
    draw_box(ax, 3, y_start-4, 2, 0.4, 'Concat', '#FFFFFF')
    
    # ID + Cross
    draw_box(ax, 0.8, y_start-4, 1, 0.5, 'ID', COLORS['id'] + '40')
    draw_box(ax, 3, y_start-5.2, 3, 0.6, 'Cross Network', COLORS['cross'] + '20')
    
    # 输出
    draw_box(ax, 3, y_start-6.5, 2, 0.5, 'Fused Emb', COLORS['cross'] + '40')
    
    # Single-view 箭头
    for x, y1, y2 in [(2, y_start-0.25, y_start-1), (4, y_start-0.25, y_start-1),
                       (2, y_start-1.4, y_start-2), (4, y_start-1.4, y_start-2),
                       (4, y_start-2.4, y_start-2.8), (2, y_start-2.4, y_start-3.8),
                       (4, y_start-3.2, y_start-3.8)]:
        draw_arrow(ax, x, y1, x, y2)
    
    draw_arrow(ax, 3, y_start-4.2, 3, y_start-4.9)
    draw_arrow(ax, 1.3, y_start-4, 2, y_start-4.9)
    draw_arrow(ax, 3, y_start-5.5, 3, y_start-6.2)
    
    # ========== Multi-View (右侧) ==========
    x_offset = 7
    
    # Layer 1: 输入
    draw_box(ax, x_offset, y_start, 1, 0.5, 'TF-IDF', COLORS['tfidf'] + '40')
    draw_box(ax, x_offset+1.5, y_start, 0.9, 0.5, 'V1', COLORS['view1'] + '60')
    draw_box(ax, x_offset+2.6, y_start, 0.9, 0.5, 'V2', COLORS['view2'] + '60')
    draw_box(ax, x_offset+3.7, y_start, 0.9, 0.5, 'V3', COLORS['view3'] + '60')
    draw_box(ax, x_offset+4.8, y_start, 0.9, 0.5, 'V4', COLORS['view4'] + '60')
    
    # 视图标签
    ax.text(x_offset+3.15, y_start+0.6, 'LLM 4 Views', fontsize=7, ha='center', color='gray')
    for i, label in enumerate(['Identity', 'Function', 'Audience', 'Category']):
        ax.text(x_offset+1.5+i*1.1, y_start-0.4, label, 
                fontsize=6, ha='center', style='italic', color='gray')
    
    # Layer 2: SVD
    for i, x in enumerate([x_offset+1.5+j*1.1 for j in range(4)]):
        draw_box(ax, x, y_start-1.4, 0.85, 0.35, 'SVD\n64-d', '#F5F5F5')
    
    # Layer 3: Whitening
    for i, x in enumerate([x_offset+1.5+j*1.1 for j in range(4)]):
        draw_box(ax, x, y_start-2.3, 0.85, 0.35, 'Whiten', '#F5F5F5')
    
    ax.text(x_offset+6, y_start-2.3, 'ZCA', fontsize=7, style='italic', color='gray')
    
    # Layer 4: SENet
    draw_box(ax, x_offset, y_start-2, 0.9, 0.35, 'Proj', '#F5F5F5')
    draw_box(ax, x_offset, y_start-3, 0.9, 0.35, 'SENet', '#F5F5F5')
    
    for i, x in enumerate([x_offset+1.5+j*1.1 for j in range(4)]):
        draw_box(ax, x, y_start-3.6, 0.85, 0.35, 'SENet', '#F5F5F5')
    
    # Layer 5: Concat
    draw_box(ax, x_offset+2.9, y_start-4.7, 3, 0.4, 'Concat & Project', '#FFFFFF')
    
    # ID + Cross
    draw_box(ax, x_offset-1.5, y_start-4.7, 1, 0.5, 'ID', COLORS['id'] + '40')
    draw_box(ax, x_offset+2.9, y_start-5.9, 4, 0.6, 'Cross Network (DCN-V2)', COLORS['cross'] + '20')
    
    # 输出
    draw_box(ax, x_offset+2.9, y_start-7.2, 2, 0.5, 'Fused Emb', COLORS['cross'] + '40')
    
    # Alignment 模块
    draw_box(ax, x_offset+6.5, y_start-2.9, 1.3, 1.5, 
             'Multi-view\nAlignment\n\nInfoNCE\nLoss', COLORS['align'] + '30')
    
    # Multi-view 箭头 (主流程)
    draw_arrow(ax, x_offset, y_start-0.25, x_offset, y_start-1.8)
    draw_arrow(ax, x_offset, y_start-2.2, x_offset, y_start-2.8)
    
    for i in range(4):
        x = x_offset + 1.5 + i * 1.1
        draw_arrow(ax, x, y_start-0.25, x, y_start-1.2)
        draw_arrow(ax, x, y_start-1.6, x, y_start-2.1)
        draw_arrow(ax, x, y_start-2.5, x, y_start-3.4)
        draw_arrow(ax, x, y_start-3.8, x, y_start-4.5)
    
    draw_arrow(ax, x_offset+2.9, y_start-4.9, x_offset+2.9, y_start-5.6)
    draw_arrow(ax, x_offset-1, y_start-4.7, x_offset+1, y_start-5.6)
    draw_arrow(ax, x_offset+2.9, y_start-6.2, x_offset+2.9, y_start-6.9)
    
    # Alignment 虚线
    for i, y in enumerate([y_start-2.3-j*0.35 for j in range(4)]):
        x = x_offset + 1.5 + i * 1.1
        draw_arrow(ax, x+0.4, y, x_offset+5.8, y_start-2.9+0.7-i*0.3, 
                   style='dashed', color=COLORS['align'])
    
    draw_arrow(ax, x_offset-1.5, y_start-4.5, x_offset+5.8, y_start-2.9+0.75,
               style='dashed', color=COLORS['align'])
    
    # 分隔线
    ax.plot([6, 6], [10.5, 0], 'k--', linewidth=1, alpha=0.3)
    
    # 底部说明
    ax.text(7, -0.5, 'Total budget: 256-d (Single: 1×256-d, Multi: 4×64-d)', 
            fontsize=8, ha='center', style='italic', color='gray')
    
    # 保存
    plt.tight_layout()
    plt.savefig('framework_matplotlib.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('framework_matplotlib.png', dpi=300, bbox_inches='tight')
    
    print("✅ 生成完成:")
    print("   - framework_matplotlib.pdf")
    print("   - framework_matplotlib.png")
    print("")
    print("提示: TikZ 版本更适合学术论文投稿")

if __name__ == '__main__':
    main()
