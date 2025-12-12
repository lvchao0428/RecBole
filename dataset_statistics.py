#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集统计分析工具
Dataset Statistics Analysis Tool

用于分析RecBole格式的数据集统计信息，支持对比多个数据集
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import argparse
from typing import Dict, List, Tuple
import json


class DatasetAnalyzer:
    """数据集分析器"""
    
    def __init__(self, dataset_path: str, dataset_name: str):
        """
        初始化数据集分析器
        
        Args:
            dataset_path: 数据集文件夹路径
            dataset_name: 数据集名称（用于读取文件）
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.inter_file = os.path.join(dataset_path, f"{dataset_name}.inter")
        self.item_file = os.path.join(dataset_path, f"{dataset_name}.item")
        self.user_file = os.path.join(dataset_path, f"{dataset_name}.user")
        
        self.stats = {}
        
    def load_interaction_data(self) -> pd.DataFrame:
        """加载交互数据"""
        if not os.path.exists(self.inter_file):
            raise FileNotFoundError(f"交互文件不存在: {self.inter_file}")
        
        # 读取数据，跳过类型定义行
        df = pd.read_csv(self.inter_file, sep='\t')
        return df
    
    def compute_basic_stats(self) -> Dict:
        """计算基础统计信息"""
        df = self.load_interaction_data()
        
        # 基础统计
        n_users = df['user_id:token'].nunique()
        n_items = df['item_id:token'].nunique()
        n_interactions = len(df)
        
        # 用户交互统计
        user_interactions = df.groupby('user_id:token').size()
        avg_interactions_per_user = user_interactions.mean()
        median_interactions_per_user = user_interactions.median()
        min_interactions_per_user = user_interactions.min()
        max_interactions_per_user = user_interactions.max()
        std_interactions_per_user = user_interactions.std()
        
        # 物品交互统计
        item_interactions = df.groupby('item_id:token').size()
        avg_interactions_per_item = item_interactions.mean()
        median_interactions_per_item = item_interactions.median()
        min_interactions_per_item = item_interactions.min()
        max_interactions_per_item = item_interactions.max()
        std_interactions_per_item = item_interactions.std()
        
        # 稀疏度计算
        total_possible_interactions = n_users * n_items
        sparsity = 100 * (1 - n_interactions / total_possible_interactions)
        density = 100 * (n_interactions / total_possible_interactions)
        
        # 评分统计（如果有rating列）
        rating_stats = {}
        if 'rating:float' in df.columns:
            rating_stats = {
                'avg_rating': df['rating:float'].mean(),
                'median_rating': df['rating:float'].median(),
                'min_rating': df['rating:float'].min(),
                'max_rating': df['rating:float'].max(),
                'std_rating': df['rating:float'].std(),
            }
        
        # Gini系数 - 衡量交互分布的不均匀程度
        def gini_coefficient(x):
            """计算Gini系数"""
            sorted_x = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(sorted_x)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        user_gini = gini_coefficient(user_interactions.values)
        item_gini = gini_coefficient(item_interactions.values)
        
        stats = {
            'n_users': n_users,
            'n_items': n_items,
            'n_interactions': n_interactions,
            'avg_interactions_per_user': avg_interactions_per_user,
            'median_interactions_per_user': median_interactions_per_user,
            'min_interactions_per_user': min_interactions_per_user,
            'max_interactions_per_user': max_interactions_per_user,
            'std_interactions_per_user': std_interactions_per_user,
            'avg_interactions_per_item': avg_interactions_per_item,
            'median_interactions_per_item': median_interactions_per_item,
            'min_interactions_per_item': min_interactions_per_item,
            'max_interactions_per_item': max_interactions_per_item,
            'std_interactions_per_item': std_interactions_per_item,
            'sparsity': sparsity,
            'density': density,
            'user_gini': user_gini,
            'item_gini': item_gini,
        }
        
        stats.update(rating_stats)
        self.stats = stats
        return stats
    
    def get_distribution_stats(self) -> Dict:
        """获取分布统计信息"""
        df = self.load_interaction_data()
        
        user_interactions = df.groupby('user_id:token').size()
        item_interactions = df.groupby('item_id:token').size()
        
        # 计算百分位数
        user_percentiles = {
            'user_p25': user_interactions.quantile(0.25),
            'user_p50': user_interactions.quantile(0.50),
            'user_p75': user_interactions.quantile(0.75),
            'user_p90': user_interactions.quantile(0.90),
            'user_p95': user_interactions.quantile(0.95),
            'user_p99': user_interactions.quantile(0.99),
        }
        
        item_percentiles = {
            'item_p25': item_interactions.quantile(0.25),
            'item_p50': item_interactions.quantile(0.50),
            'item_p75': item_interactions.quantile(0.75),
            'item_p90': item_interactions.quantile(0.90),
            'item_p95': item_interactions.quantile(0.95),
            'item_p99': item_interactions.quantile(0.99),
        }
        
        return {**user_percentiles, **item_percentiles}
    
    def analyze(self) -> Dict:
        """执行完整分析"""
        basic_stats = self.compute_basic_stats()
        distribution_stats = self.get_distribution_stats()
        return {**basic_stats, **distribution_stats}


def format_number(num: float, decimal: int = 2) -> str:
    """格式化数字显示"""
    if isinstance(num, int) or num == int(num):
        return f"{int(num):,}"
    else:
        return f"{num:,.{decimal}f}"


def compare_datasets(datasets_info: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """
    比较多个数据集
    
    Args:
        datasets_info: List of (dataset_path, dataset_name, display_name)
    
    Returns:
        比较结果DataFrame
    """
    results = {}
    
    for dataset_path, dataset_name, display_name in datasets_info:
        print(f"\n分析数据集: {display_name}")
        print(f"路径: {dataset_path}")
        
        analyzer = DatasetAnalyzer(dataset_path, dataset_name)
        stats = analyzer.analyze()
        results[display_name] = stats
        
        print(f"✓ 完成")
    
    return pd.DataFrame(results).T


def create_comparison_table(df: pd.DataFrame, metrics: List[str] = None) -> str:
    """
    创建对比表格
    
    Args:
        df: 统计结果DataFrame
        metrics: 要显示的指标列表
    
    Returns:
        格式化的对比表格字符串
    """
    if metrics is None:
        metrics = [
            'n_users', 'n_items', 'n_interactions',
            'avg_interactions_per_user', 'sparsity', 'density'
        ]
    
    metric_names = {
        'n_users': '用户数',
        'n_items': '物品数',
        'n_interactions': '交互总数',
        'avg_interactions_per_user': '用户平均交互',
        'median_interactions_per_user': '用户交互中位数',
        'avg_interactions_per_item': '物品平均交互',
        'median_interactions_per_item': '物品交互中位数',
        'sparsity': '稀疏度 (%)',
        'density': '密度 (%)',
        'user_gini': '用户Gini系数',
        'item_gini': '物品Gini系数',
        'avg_rating': '平均评分',
    }
    
    lines = []
    lines.append("\n" + "="*100)
    lines.append("数据集统计对比")
    lines.append("="*100)
    
    # 表头
    header = "指标".ljust(25)
    for col in df.index:
        header += f"{col}".ljust(25)
    lines.append(header)
    lines.append("-"*100)
    
    # 数据行
    for metric in metrics:
        if metric in df.columns:
            row = metric_names.get(metric, metric).ljust(25)
            for dataset in df.index:
                value = df.loc[dataset, metric]
                if metric in ['sparsity', 'density']:
                    formatted = f"{value:.4f}%".ljust(25)
                elif metric in ['user_gini', 'item_gini', 'avg_rating']:
                    formatted = f"{value:.4f}".ljust(25)
                elif metric == 'avg_interactions_per_user':
                    formatted = f"{value:.2f}".ljust(25)
                else:
                    formatted = format_number(value).ljust(25)
                row += formatted
            lines.append(row)
    
    lines.append("="*100)
    
    # 添加差异分析
    if len(df) == 2:
        lines.append("\n差异分析:")
        lines.append("-"*100)
        dataset1, dataset2 = df.index[0], df.index[1]
        
        # 用户数对比
        ratio = df.loc[dataset2, 'n_users'] / df.loc[dataset1, 'n_users']
        lines.append(f"• 用户数: {dataset2} 是 {dataset1} 的 {ratio:.2f} 倍")
        
        # 物品数对比
        ratio = df.loc[dataset2, 'n_items'] / df.loc[dataset1, 'n_items']
        lines.append(f"• 物品数: {dataset2} 是 {dataset1} 的 {ratio:.2f} 倍")
        
        # 交互数对比
        ratio = df.loc[dataset1, 'n_interactions'] / df.loc[dataset2, 'n_interactions']
        if ratio > 1:
            lines.append(f"• 交互总数: {dataset1} 是 {dataset2} 的 {ratio:.2f} 倍")
        else:
            ratio = 1 / ratio
            lines.append(f"• 交互总数: {dataset2} 是 {dataset1} 的 {ratio:.2f} 倍")
        
        # 用户交互密度对比
        ratio = df.loc[dataset1, 'avg_interactions_per_user'] / df.loc[dataset2, 'avg_interactions_per_user']
        if ratio > 1:
            lines.append(f"• 用户交互密度: {dataset1} 用户平均交互是 {dataset2} 的 {ratio:.2f} 倍")
        else:
            ratio = 1 / ratio
            lines.append(f"• 用户交互密度: {dataset2} 用户平均交互是 {dataset1} 的 {ratio:.2f} 倍")
        
        # 稀疏度对比
        sparse1 = df.loc[dataset1, 'sparsity']
        sparse2 = df.loc[dataset2, 'sparsity']
        if sparse1 > sparse2:
            lines.append(f"• 稀疏度: {dataset1} ({sparse1:.4f}%) > {dataset2} ({sparse2:.4f}%)")
        else:
            lines.append(f"• 稀疏度: {dataset2} ({sparse2:.4f}%) > {dataset1} ({sparse1:.4f}%)")
        
        # 密度对比
        lines.append(f"• 信息密度: {dataset1} ({df.loc[dataset1, 'density']:.4f}%) vs {dataset2} ({df.loc[dataset2, 'density']:.4f}%)")
        
        lines.append("="*100)
    
    return "\n".join(lines)


def save_detailed_stats(df: pd.DataFrame, output_file: str = "dataset_comparison.json"):
    """保存详细统计信息到JSON文件"""
    # 转换DataFrame为字典
    data = df.to_dict(orient='index')
    
    # 转换numpy类型为Python原生类型
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        else:
            return obj
    
    data = convert_types(data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细统计信息已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='数据集统计分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析示例数据集
  python dataset_statistics.py --example
  
  # 分析服务器上的完整数据集
  python dataset_statistics.py --full --server-path /path/to/datasets
  
  # 自定义数据集分析
  python dataset_statistics.py --custom \\
      --datasets "path1:name1:display1" "path2:name2:display2"
        """
    )
    
    parser.add_argument('--example', action='store_true',
                        help='分析示例数据集 (ml-1m-example 和 Amazon_Beauty-example)')
    parser.add_argument('--full', action='store_true',
                        help='分析完整数据集 (需要指定 --server-path)')
    parser.add_argument('--server-path', type=str,
                        help='服务器上数据集的根路径')
    parser.add_argument('--datasets', nargs='+',
                        help='自定义数据集列表，格式: "路径:数据集名:显示名"')
    parser.add_argument('--output', type=str, default='dataset_comparison.json',
                        help='输出JSON文件路径 (默认: dataset_comparison.json)')
    parser.add_argument('--metrics', nargs='+',
                        help='要显示的指标列表')
    
    args = parser.parse_args()
    
    datasets_info = []
    
    if args.example:
        # 分析示例数据集
        base_path = os.path.dirname(os.path.abspath(__file__))
        datasets_info = [
            (os.path.join(base_path, 'dataset/ml-1m-example'), 'ml-1m', 'MovieLens-1M'),
            (os.path.join(base_path, 'dataset/Amazon_Beauty-example'), 'Amazon_Beauty', 'Amazon Beauty'),
        ]
    
    elif args.full:
        # 分析完整数据集
        if not args.server_path:
            print("错误: 使用 --full 时必须指定 --server-path")
            return
        
        datasets_info = [
            (os.path.join(args.server_path, 'ml-1m'), 'ml-1m', 'MovieLens-1M (Full)'),
            (os.path.join(args.server_path, 'Amazon_Beauty'), 'Amazon_Beauty', 'Amazon Beauty (Full)'),
        ]
    
    elif args.datasets:
        # 自定义数据集
        for ds in args.datasets:
            parts = ds.split(':')
            if len(parts) != 3:
                print(f"错误: 数据集格式不正确: {ds}")
                print("正确格式: '路径:数据集名:显示名'")
                return
            datasets_info.append(tuple(parts))
    
    else:
        parser.print_help()
        return
    
    # 执行分析
    print("\n" + "="*100)
    print("开始数据集统计分析")
    print("="*100)
    
    try:
        df = compare_datasets(datasets_info)
        
        # 显示对比表格
        table = create_comparison_table(df, args.metrics)
        print(table)
        
        # 保存详细统计
        save_detailed_stats(df, args.output)
        
        # 显示更多详细信息
        print("\n" + "="*100)
        print("详细分布统计")
        print("="*100)
        
        distribution_metrics = [
            'user_p25', 'user_p50', 'user_p75', 'user_p90', 'user_p95', 'user_p99',
            'item_p25', 'item_p50', 'item_p75', 'item_p90', 'item_p95', 'item_p99',
        ]
        
        for dataset in df.index:
            print(f"\n{dataset} - 用户交互分布:")
            for metric in ['user_p25', 'user_p50', 'user_p75', 'user_p90', 'user_p95', 'user_p99']:
                if metric in df.columns:
                    percentile = metric.replace('user_p', 'P')
                    print(f"  {percentile}: {df.loc[dataset, metric]:.2f}")
            
            print(f"\n{dataset} - 物品交互分布:")
            for metric in ['item_p25', 'item_p50', 'item_p75', 'item_p90', 'item_p95', 'item_p99']:
                if metric in df.columns:
                    percentile = metric.replace('item_p', 'P')
                    print(f"  {percentile}: {df.loc[dataset, metric]:.2f}")
        
        print("\n" + "="*100)
        print("分析完成!")
        print("="*100)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()


