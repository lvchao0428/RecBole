#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集统计可视化工具
Dataset Statistics Visualization Tool

读取 dataset_comparison.json 生成对比图表
"""

import json
import argparse
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 无GUI后端
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib未安装，仅生成文本报告")
    print("安装方法: pip install matplotlib")


def load_stats(json_file):
    """加载统计数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_comparison_charts(data, output_prefix='dataset_comparison'):
    """创建对比图表"""
    if not HAS_MATPLOTLIB:
        print("跳过图表生成（需要matplotlib）")
        return
    
    datasets = list(data.keys())
    n_datasets = len(datasets)
    
    # 图1: 基础统计对比 (柱状图)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('数据集基础统计对比', fontsize=16, fontproperties='SimHei')
    
    metrics = [
        ('n_users', '用户数'),
        ('n_items', '物品数'),
        ('n_interactions', '交互总数'),
        ('avg_interactions_per_user', '用户平均交互'),
        ('sparsity', '稀疏度 (%)'),
        ('density', '密度 (%)')
    ]
    
    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        values = [data[ds].get(metric, 0) for ds in datasets]
        
        bars = ax.bar(range(n_datasets), values, alpha=0.7)
        ax.set_xticks(range(n_datasets))
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        
        # 在柱状图上显示数值
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if metric in ['sparsity', 'density']:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
            elif metric == 'avg_interactions_per_user':
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(val):,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_basic_stats.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已生成: {output_prefix}_basic_stats.png")
    plt.close()
    
    # 图2: 交互分布对比 (箱线图)
    if all('user_p25' in data[ds] for ds in datasets):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('交互分布对比', fontsize=16)
        
        # 用户交互分布
        user_percentiles = ['user_p25', 'user_p50', 'user_p75', 'user_p90', 'user_p95', 'user_p99']
        user_data = []
        for ds in datasets:
            user_data.append([data[ds].get(p, 0) for p in user_percentiles])
        
        x = np.arange(len(user_percentiles))
        width = 0.8 / n_datasets
        
        for i, ds in enumerate(datasets):
            offset = (i - n_datasets/2) * width + width/2
            ax1.bar(x + offset, user_data[i], width, label=ds, alpha=0.7)
        
        ax1.set_xlabel('百分位数')
        ax1.set_ylabel('交互次数')
        ax1.set_title('用户交互分布')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['P25', 'P50', 'P75', 'P90', 'P95', 'P99'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 物品交互分布
        item_percentiles = ['item_p25', 'item_p50', 'item_p75', 'item_p90', 'item_p95', 'item_p99']
        item_data = []
        for ds in datasets:
            item_data.append([data[ds].get(p, 0) for p in item_percentiles])
        
        for i, ds in enumerate(datasets):
            offset = (i - n_datasets/2) * width + width/2
            ax2.bar(x + offset, item_data[i], width, label=ds, alpha=0.7)
        
        ax2.set_xlabel('百分位数')
        ax2.set_ylabel('交互次数')
        ax2.set_title('物品交互分布')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['P25', 'P50', 'P75', 'P90', 'P95', 'P99'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ 已生成: {output_prefix}_distribution.png")
        plt.close()
    
    # 图3: Gini系数对比 (如果有)
    if all('user_gini' in data[ds] for ds in datasets):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        x = np.arange(n_datasets)
        width = 0.35
        
        user_gini = [data[ds].get('user_gini', 0) for ds in datasets]
        item_gini = [data[ds].get('item_gini', 0) for ds in datasets]
        
        bars1 = ax.bar(x - width/2, user_gini, width, label='用户Gini', alpha=0.7)
        bars2 = ax.bar(x + width/2, item_gini, width, label='物品Gini', alpha=0.7)
        
        ax.set_xlabel('数据集')
        ax.set_ylabel('Gini系数')
        ax.set_title('交互分布不均匀程度 (Gini系数)')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_gini.png', dpi=300, bbox_inches='tight')
        print(f"✓ 已生成: {output_prefix}_gini.png")
        plt.close()


def generate_latex_table(data, output_file='dataset_table.tex'):
    """生成LaTeX表格"""
    datasets = list(data.keys())
    
    lines = []
    lines.append("% LaTeX表格 - 数据集统计对比")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{数据集统计对比}")
    lines.append("\\label{tab:dataset_stats}")
    lines.append("\\begin{tabular}{l" + "r" * len(datasets) + "}")
    lines.append("\\hline")
    
    # 表头
    header = "指标 & " + " & ".join(datasets) + " \\\\"
    lines.append(header)
    lines.append("\\hline")
    
    # 数据行
    metrics = [
        ('n_users', '用户数'),
        ('n_items', '物品数'),
        ('n_interactions', '交互总数'),
        ('avg_interactions_per_user', '用户平均交互'),
        ('sparsity', '稀疏度 (\\%)'),
        ('density', '密度 (\\%)'),
    ]
    
    for metric, label in metrics:
        row = label + " & "
        values = []
        for ds in datasets:
            val = data[ds].get(metric, 0)
            if metric in ['sparsity', 'density']:
                values.append(f"{val:.4f}")
            elif metric == 'avg_interactions_per_user':
                values.append(f"{val:.2f}")
            else:
                values.append(f"{int(val):,}")
        row += " & ".join(values) + " \\\\"
        lines.append(row)
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"✓ 已生成LaTeX表格: {output_file}")


def generate_markdown_table(data, output_file='dataset_table.md'):
    """生成Markdown表格"""
    datasets = list(data.keys())
    
    lines = []
    lines.append("# 数据集统计对比\n")
    
    # 表头
    header = "| 指标 | " + " | ".join(datasets) + " |"
    lines.append(header)
    separator = "|" + "|".join([" --- "] * (len(datasets) + 1)) + "|"
    lines.append(separator)
    
    # 数据行
    metrics = [
        ('n_users', '用户数'),
        ('n_items', '物品数'),
        ('n_interactions', '交互总数'),
        ('avg_interactions_per_user', '用户平均交互'),
        ('median_interactions_per_user', '用户交互中位数'),
        ('sparsity', '稀疏度 (%)'),
        ('density', '密度 (%)'),
        ('user_gini', '用户Gini系数'),
        ('item_gini', '物品Gini系数'),
    ]
    
    for metric, label in metrics:
        if not any(metric in data[ds] for ds in datasets):
            continue
            
        row = f"| {label} | "
        values = []
        for ds in datasets:
            val = data[ds].get(metric, 0)
            if metric in ['sparsity', 'density']:
                values.append(f"{val:.4f}%")
            elif metric in ['user_gini', 'item_gini', 'avg_interactions_per_user', 'median_interactions_per_user']:
                values.append(f"{val:.2f}")
            else:
                values.append(f"{int(val):,}")
        row += " | ".join(values) + " |"
        lines.append(row)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"✓ 已生成Markdown表格: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='数据集统计可视化工具')
    parser.add_argument('--input', type=str, default='dataset_comparison.json',
                        help='输入JSON文件 (默认: dataset_comparison.json)')
    parser.add_argument('--output-prefix', type=str, default='dataset_comparison',
                        help='输出文件前缀 (默认: dataset_comparison)')
    parser.add_argument('--no-plot', action='store_true',
                        help='不生成图表')
    parser.add_argument('--latex', action='store_true',
                        help='生成LaTeX表格')
    parser.add_argument('--markdown', action='store_true',
                        help='生成Markdown表格')
    
    args = parser.parse_args()
    
    try:
        data = load_stats(args.input)
        print(f"\n已加载统计数据: {args.input}")
        print(f"数据集数量: {len(data)}")
        print(f"数据集: {', '.join(data.keys())}\n")
        
        if not args.no_plot:
            create_comparison_charts(data, args.output_prefix)
        
        if args.latex:
            generate_latex_table(data, f'{args.output_prefix}_table.tex')
        
        if args.markdown:
            generate_markdown_table(data, f'{args.output_prefix}_table.md')
        
        # 默认生成markdown
        if not args.latex and not args.markdown:
            generate_markdown_table(data, f'{args.output_prefix}_table.md')
        
        print("\n✓ 完成!")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {args.input}")
        print("请先运行 dataset_statistics.py 生成统计数据")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


