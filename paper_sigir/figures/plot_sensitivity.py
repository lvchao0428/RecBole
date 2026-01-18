#!/usr/bin/env python3
"""
Sensitivity Analysis Plot for MV-Align Paper

生成4个子图，展示不同超参数对 HR@10 和 HR_new@10 的影响：
(a) λ (alignment weight) 变化
(b) τ (temperature) 变化  
(c) cold_start_align_boost 变化
(d) inference_cold_text_boost 变化

Usage:
    python plot_sensitivity.py
    
Output:
    sensitivity.pdf
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置论文风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# ==================== 数据 ====================
# 从 Table: Sensitivity analysis on Toys 7B (Aggressive)
# Baseline: λ=0.10, τ=0.05, cold=2.5, infer=1.5

# (a) Alignment weight (λ)
lambda_vals = [0.05, 0.10, 0.15]
lambda_hr = [6.82, 6.92, 6.93]
lambda_hr_new = [1.98, 1.99, 2.04]

# (b) Temperature (τ)
tau_vals = [0.03, 0.05, 0.10]
tau_hr = [6.94, 6.92, 6.90]
tau_hr_new = [2.01, 1.99, 2.01]

# (c) Cold-start boost
cold_vals = [1.5, 2.5, 3.0]
cold_hr = [6.90, 6.92, 6.89]
cold_hr_new = [1.95, 1.99, 2.02]

# (d) Inference boost
infer_vals = [0.5, 1.0, 1.5, 2.0, 2.5]
infer_hr = [6.70, 6.85, 6.92, 6.90, 6.71]
infer_hr_new = [1.95, 1.97, 1.99, 1.99, 2.03]

# ==================== 作图 ====================
fig, axes = plt.subplots(2, 2, figsize=(7, 6.5))  # 增加高度

# 颜色方案
color_hr = '#2E86AB'       # 蓝色 - HR@10
color_hr_new = '#E94F37'   # 红色 - HR_new@10
marker_size = 8
linewidth = 1.8

def plot_dual_axis(ax, x_vals, y1, y2, xlabel, title, baseline_x=None, x_labels=None):
    """绘制双Y轴图"""
    ax2 = ax.twinx()
    
    # 左轴: HR@10
    line1, = ax.plot(x_vals, y1, 'o-', color=color_hr, 
                     markersize=marker_size, linewidth=linewidth, label='HR@10')
    ax.set_ylabel('HR@10 (%)', color=color_hr)
    ax.tick_params(axis='y', labelcolor=color_hr)
    
    # 右轴: HR_new@10
    line2, = ax2.plot(x_vals, y2, 's--', color=color_hr_new,
                      markersize=marker_size, linewidth=linewidth, label='HR_new@10')
    ax2.set_ylabel('HR_new@10 (%)', color=color_hr_new)
    ax2.tick_params(axis='y', labelcolor=color_hr_new)
    
    # 标记baseline
    if baseline_x is not None:
        ax.axvline(x=baseline_x, color='gray', linestyle=':', alpha=0.7, linewidth=1.2)
    
    ax.set_xlabel(xlabel, labelpad=3)  # 减少xlabel下方间距
    ax.set_title(title, fontweight='bold', pad=8)  # 增加标题上方间距
    
    # 设置x轴刻度标签
    if x_labels is not None:
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_labels)
    
    # 调整Y轴范围，留出空间
    y1_margin = (max(y1) - min(y1)) * 0.3
    y2_margin = (max(y2) - min(y2)) * 0.3
    ax.set_ylim(min(y1) - y1_margin, max(y1) + y1_margin)
    ax2.set_ylim(min(y2) - y2_margin, max(y2) + y2_margin)
    
    return line1, line2

# (a) λ (alignment weight)
l1, l2 = plot_dual_axis(
    axes[0, 0], lambda_vals, lambda_hr, lambda_hr_new,
    xlabel=r'$\lambda$ (alignment weight)',
    title=r'(a) Alignment Weight $\lambda$',
    baseline_x=0.10
)

# (b) τ (temperature)
plot_dual_axis(
    axes[0, 1], tau_vals, tau_hr, tau_hr_new,
    xlabel=r'$\tau$ (temperature)',
    title=r'(b) Temperature $\tau$',
    baseline_x=0.05
)

# (c) Cold-start boost
plot_dual_axis(
    axes[1, 0], cold_vals, cold_hr, cold_hr_new,
    xlabel='cold_start_align_boost',
    title='(c) Cold-Start Boost',
    baseline_x=2.5
)

# (d) Inference boost
plot_dual_axis(
    axes[1, 1], infer_vals, infer_hr, infer_hr_new,
    xlabel='inference_cold_text_boost',
    title='(d) Inference Boost',
    baseline_x=1.5
)

# 添加图例 (只在第一个子图)
fig.legend([l1, l2], ['HR@10 (Overall)', 'HR_new@10 (Cold-start)'], 
           loc='upper center', ncol=2, frameon=True, 
           bbox_to_anchor=(0.5, 0.02), fancybox=True, shadow=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.10, top=0.95, hspace=0.45, wspace=0.50)

# 保存
plt.savefig('sensitivity.pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('sensitivity.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
print("Saved: sensitivity.pdf, sensitivity.png")

plt.show()
