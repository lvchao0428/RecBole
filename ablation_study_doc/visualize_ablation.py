#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验可视化脚本
================

用于生成论文中的消融实验可视化图表：
1. Score分布对比图（需要模型推理，见 ScoreDistributionAnalyzer）
2. Recall vs MRR Trade-off曲线（直接使用实验数据）
3. 冷启动 vs 高频商品对比柱状图（直接使用实验数据）

Usage:
    python visualize_ablation.py --plot-all         # 生成所有图表
    python visualize_ablation.py --plot-tradeoff    # 只生成Trade-off曲线
    python visualize_ablation.py --plot-coldstart   # 只生成冷启动对比图
    
    # 生成分数分布图（需要模型checkpoint路径）
    python visualize_ablation.py --plot-score-dist \
        --baseline-checkpoint /path/to/baseline.pth \
        --ablation-checkpoint /path/to/ablation.pth \
        --config /path/to/config.yaml
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器

# 设置中文字体支持（可选）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 实验数据（硬编码，来自消融实验结果）
# ============================================================================

# Baseline: sasrec + tfidf + multi-view v2 + coldstart weighted + text_tail_threshold 0
BASELINE = {
    'recall': {'5': 0.0486, '10': 0.0652, '20': 0.0849},
    'mrr': {'5': 0.0293, '10': 0.0315, '20': 0.0329},
    'ndcg': {'5': 0.0341, '10': 0.0394, '20': 0.0444},
    'hit': {'5': 0.0486, '10': 0.0652, '20': 0.0849},
    'precision': {'5': 0.0097, '10': 0.0065, '20': 0.0042},
    # 冷启动
    'recall_new': {'5': 0.0161, '10': 0.0204, '20': 0.0237},
    'mrr_new': {'5': 0.0097, '10': 0.0103, '20': 0.0105},
    'ndcg_new': {'5': 0.0113, '10': 0.0127, '20': 0.0135},
    # 少样本
    'recall_few': {'5': 0.0303, '10': 0.0401, '20': 0.051},
    'mrr_few': {'5': 0.0198, '10': 0.0211, '20': 0.0218},
    'ndcg_few': {'5': 0.0224, '10': 0.0255, '20': 0.0283},
    # 高频
    'recall_frequent': {'5': 0.0839, '10': 0.1132, '20': 0.1489},
    'mrr_frequent': {'5': 0.0499, '10': 0.0538, '20': 0.0562},
    'ndcg_frequent': {'5': 0.0583, '10': 0.0678, '20': 0.0768},
}

# Ablation: 去掉 Cross & SENet
ABLATION_NO_CROSS_SENET = {
    'recall': {'5': 0.0491, '10': 0.0688, '20': 0.0915},
    'mrr': {'5': 0.0226, '10': 0.0252, '20': 0.0268},
    'ndcg': {'5': 0.0292, '10': 0.0355, '20': 0.0413},
    'hit': {'5': 0.0491, '10': 0.0688, '20': 0.0915},
    'precision': {'5': 0.0098, '10': 0.0069, '20': 0.0046},
    # 冷启动
    'recall_new': {'5': 0.015, '10': 0.0216, '20': 0.0265},
    'mrr_new': {'5': 0.0061, '10': 0.0069, '20': 0.0073},
    'ndcg_new': {'5': 0.0083, '10': 0.0104, '20': 0.0117},
    # 少样本
    'recall_few': {'5': 0.0311, '10': 0.0435, '20': 0.0573},
    'mrr_few': {'5': 0.0134, '10': 0.0151, '20': 0.016},
    'ndcg_few': {'5': 0.0178, '10': 0.0218, '20': 0.0253},
    # 高频
    'recall_frequent': {'5': 0.0849, '10': 0.1188, '20': 0.1591},
    'mrr_frequent': {'5': 0.0397, '10': 0.0442, '20': 0.047},
    'ndcg_frequent': {'5': 0.051, '10': 0.0619, '20': 0.0721},
}

# Ablation: 只去掉 SENet
ABLATION_NO_SENET = {
    'recall': {'5': 0.048, '10': 0.064, '20': 0.0829},
    'mrr': {'5': 0.0292, '10': 0.0313, '20': 0.0326},
    'ndcg': {'5': 0.0339, '10': 0.039, '20': 0.0438},
    'hit': {'5': 0.048, '10': 0.064, '20': 0.0829},
    'precision': {'5': 0.0096, '10': 0.0064, '20': 0.0041},
    # 冷启动
    'recall_new': {'5': 0.0174, '10': 0.0209, '20': 0.0239},
    'mrr_new': {'5': 0.0105, '10': 0.011, '20': 0.0112},
    'ndcg_new': {'5': 0.0122, '10': 0.0133, '20': 0.0141},
    # 少样本
    'recall_few': {'5': 0.0299, '10': 0.0402, '20': 0.0513},
    'mrr_few': {'5': 0.0198, '10': 0.0212, '20': 0.0219},
    'ndcg_few': {'5': 0.0224, '10': 0.0256, '20': 0.0284},
    # 高频
    'recall_frequent': {'5': 0.0824, '10': 0.1105, '20': 0.1446},
    'mrr_frequent': {'5': 0.0494, '10': 0.0531, '20': 0.0554},
    'ndcg_frequent': {'5': 0.0576, '10': 0.0666, '20': 0.0752},
}


def compute_relative_change(baseline, ablation):
    """计算相对变化百分比"""
    return {
        k: {kk: (ablation[k][kk] - baseline[k][kk]) / baseline[k][kk] * 100 
            for kk in baseline[k]}
        for k in baseline
    }


# ============================================================================
# 可视化函数
# ============================================================================

def plot_tradeoff_curve(output_dir: str, filename: str = "recall_mrr_tradeoff.png"):
    """
    绘制 Recall vs MRR Trade-off 曲线
    
    展示去掉Cross+SENet后，Recall上升但MRR下降的trade-off关系
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ks = ['5', '10', '20']
    x = np.arange(len(ks))
    width = 0.35
    
    # 计算相对变化
    changes = compute_relative_change(BASELINE, ABLATION_NO_CROSS_SENET)
    
    recall_changes = [changes['recall'][k] for k in ks]
    mrr_changes = [changes['mrr'][k] for k in ks]
    ndcg_changes = [changes['ndcg'][k] for k in ks]
    
    # 绘制柱状图
    bars1 = ax.bar(x - width/2, recall_changes, width, label='Recall', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, mrr_changes, width, label='MRR', color='#e74c3c', alpha=0.8)
    
    # 添加NDCG折线
    ax.plot(x, ndcg_changes, 'o-', color='#3498db', linewidth=2, markersize=8, label='NDCG')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, -15), textcoords="offset points",
                    ha='center', va='top', fontsize=10)
    
    # 添加零线
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    ax.set_xlabel('@K', fontsize=12)
    ax.set_ylabel('Relative Change (%)', fontsize=12)
    ax.set_title('Ablation: Removing Cross Network + SENet\nRecall ↑ vs MRR ↓ Trade-off', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'@{k}' for k in ks])
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 设置y轴范围
    ax.set_ylim(-30, 15)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {output_path}")
    return output_path


def plot_coldstart_comparison(output_dir: str, filename: str = "coldstart_comparison.png"):
    """
    绘制冷启动 vs 少样本 vs 高频商品的 MRR 下降对比柱状图
    
    展示Cross+SENet对不同类型商品的影响差异
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ks = ['5', '10', '20']
    changes = compute_relative_change(BASELINE, ABLATION_NO_CROSS_SENET)
    
    # 子图配置
    configs = [
        ('MRR', ['mrr_new', 'mrr_few', 'mrr_frequent'], 
         ['Cold-Start', 'Few-Shot', 'Frequent'], '#e74c3c'),
        ('Recall', ['recall_new', 'recall_few', 'recall_frequent'], 
         ['Cold-Start', 'Few-Shot', 'Frequent'], '#2ecc71'),
        ('NDCG', ['ndcg_new', 'ndcg_few', 'ndcg_frequent'], 
         ['Cold-Start', 'Few-Shot', 'Frequent'], '#3498db'),
    ]
    
    for ax, (metric_name, metrics, labels, color) in zip(axes, configs):
        x = np.arange(len(ks))
        width = 0.25
        
        for i, (metric, label) in enumerate(zip(metrics, labels)):
            values = [changes[metric][k] for k in ks]
            bars = ax.bar(x + (i - 1) * width, values, width, label=label, alpha=0.8)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                offset = 3 if height >= 0 else -3
                ax.annotate(f'{height:+.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, offset), textcoords="offset points",
                            ha='center', va=va, fontsize=8, rotation=45)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('@K', fontsize=11)
        ax.set_ylabel(f'{metric_name} Change (%)', fontsize=11)
        ax.set_title(f'{metric_name} by Item Frequency', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'@{k}' for k in ks])
        ax.legend(loc='best', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Ablation Impact on Different Item Types (w/o Cross + SENet)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {output_path}")
    return output_path


def plot_module_contribution(output_dir: str, filename: str = "module_contribution.png"):
    """
    绘制各模块贡献对比图
    
    对比: Baseline vs 去掉SENet vs 去掉Cross+SENet
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ks = ['5', '10', '20']
    
    # 计算相对变化
    changes_no_senet = compute_relative_change(BASELINE, ABLATION_NO_SENET)
    changes_no_both = compute_relative_change(BASELINE, ABLATION_NO_CROSS_SENET)
    
    # 左图: Recall变化
    ax = axes[0]
    x = np.arange(len(ks))
    width = 0.35
    
    recall_no_senet = [changes_no_senet['recall'][k] for k in ks]
    recall_no_both = [changes_no_both['recall'][k] for k in ks]
    
    bars1 = ax.bar(x - width/2, recall_no_senet, width, label='w/o SENet', color='#f39c12', alpha=0.8)
    bars2 = ax.bar(x + width/2, recall_no_both, width, label='w/o Cross+SENet', color='#e74c3c', alpha=0.8)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('@K', fontsize=11)
    ax.set_ylabel('Recall Change (%)', fontsize=11)
    ax.set_title('Recall@K Change', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'@{k}' for k in ks])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # 右图: MRR变化
    ax = axes[1]
    
    mrr_no_senet = [changes_no_senet['mrr'][k] for k in ks]
    mrr_no_both = [changes_no_both['mrr'][k] for k in ks]
    
    bars1 = ax.bar(x - width/2, mrr_no_senet, width, label='w/o SENet', color='#f39c12', alpha=0.8)
    bars2 = ax.bar(x + width/2, mrr_no_both, width, label='w/o Cross+SENet', color='#e74c3c', alpha=0.8)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, -15), textcoords="offset points",
                    ha='center', va='top', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, -15), textcoords="offset points",
                    ha='center', va='top', fontsize=9)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('@K', fontsize=11)
    ax.set_ylabel('MRR Change (%)', fontsize=11)
    ax.set_title('MRR@K Change', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'@{k}' for k in ks])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-30, 5)
    
    plt.suptitle('Module Contribution: SENet vs Cross Network', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {output_path}")
    return output_path


# ============================================================================
# Score分布分析器（需要模型推理）
# ============================================================================

class ScoreDistributionAnalyzer:
    """
    分析模型预测分数的分布
    
    用于对比有/无Cross+SENet时，Top-K商品的分数分布差异
    
    使用方法:
        1. 在评估脚本中，保存分数到文件
        2. 使用此类加载并可视化
        
    或者:
        直接提供模型checkpoint，自动进行推理
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.baseline_scores = None
        self.ablation_scores = None
    
    def load_scores_from_file(self, baseline_path: str, ablation_path: str):
        """从保存的文件加载分数"""
        self.baseline_scores = np.load(baseline_path)
        self.ablation_scores = np.load(ablation_path)
        print(f"[✓] Loaded baseline scores: {self.baseline_scores.shape}")
        print(f"[✓] Loaded ablation scores: {self.ablation_scores.shape}")
    
    def plot_score_distribution(self, top_k: int = 10, filename: str = "score_distribution.png"):
        """
        绘制Top-K商品的分数分布对比图
        
        Args:
            top_k: 分析Top多少个商品的分数
            filename: 输出文件名
        """
        if self.baseline_scores is None or self.ablation_scores is None:
            print("[!] Please load scores first using load_scores_from_file()")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图: 分数分布直方图
        ax = axes[0]
        
        # 取Top-K分数
        baseline_topk = np.sort(self.baseline_scores, axis=1)[:, -top_k:]
        ablation_topk = np.sort(self.ablation_scores, axis=1)[:, -top_k:]
        
        # 绘制分布
        ax.hist(baseline_topk.flatten(), bins=50, alpha=0.6, label='With Cross+SENet', color='#3498db')
        ax.hist(ablation_topk.flatten(), bins=50, alpha=0.6, label='W/O Cross+SENet', color='#e74c3c')
        
        ax.set_xlabel('Score', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Top-{top_k} Score Distribution', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # 右图: 分数方差对比
        ax = axes[1]
        
        baseline_std = np.std(baseline_topk, axis=1)
        ablation_std = np.std(ablation_topk, axis=1)
        
        ax.hist(baseline_std, bins=50, alpha=0.6, label='With Cross+SENet', color='#3498db')
        ax.hist(ablation_std, bins=50, alpha=0.6, label='W/O Cross+SENet', color='#e74c3c')
        
        ax.set_xlabel('Score Std Dev', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Top-{top_k} Score Variance', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加均值标注
        ax.axvline(x=baseline_std.mean(), color='#3498db', linestyle='--', 
                   label=f'Baseline Mean: {baseline_std.mean():.4f}')
        ax.axvline(x=ablation_std.mean(), color='#e74c3c', linestyle='--',
                   label=f'Ablation Mean: {ablation_std.mean():.4f}')
        
        plt.suptitle('Score Distribution Analysis: Cross+SENet Effect', fontsize=14, y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[✓] Saved: {output_path}")
        return output_path


def save_scores_during_eval():
    """
    在评估过程中保存分数的示例代码片段
    
    将此代码添加到评估脚本中，在 full_sort_predict 后保存分数
    """
    code_snippet = '''
# === 在评估脚本中添加以下代码 ===

import numpy as np

# 在 full_sort_predict 调用后:
scores = model.full_sort_predict(interaction)  # [batch_size, n_items]

# 保存分数
scores_np = scores.cpu().numpy()

# 如果是baseline模型:
np.save('ablation_study_doc/scores_baseline.npy', scores_np)

# 如果是ablation模型:
np.save('ablation_study_doc/scores_ablation.npy', scores_np)

# === 或者在 Evaluator 中修改 ===
# 找到 recbole/evaluator/evaluator.py
# 在 evaluate() 方法中，scores 变量保存的就是预测分数
'''
    return code_snippet


def generate_synthetic_scores_for_demo(output_dir: str):
    """
    生成合成的分数数据用于演示
    
    模拟有/无Cross+SENet时的分数分布差异
    """
    np.random.seed(42)
    n_samples = 1000
    n_items = 5000
    
    # Baseline: 分数分布更"尖锐"，Top-1和其他差距大
    baseline_scores = np.random.randn(n_samples, n_items) * 0.5
    # 给正确答案加上更大的分数
    correct_idx = np.random.randint(0, n_items, n_samples)
    for i, idx in enumerate(correct_idx):
        baseline_scores[i, idx] += 3.0  # 大幅领先
    
    # Ablation: 分数分布更"平坦"，各商品分数差距小
    ablation_scores = np.random.randn(n_samples, n_items) * 0.8
    # 正确答案的领先优势变小
    for i, idx in enumerate(correct_idx):
        ablation_scores[i, idx] += 1.5  # 领先优势减小
    
    # 保存
    baseline_path = os.path.join(output_dir, 'scores_baseline_demo.npy')
    ablation_path = os.path.join(output_dir, 'scores_ablation_demo.npy')
    np.save(baseline_path, baseline_scores)
    np.save(ablation_path, ablation_scores)
    
    print(f"[✓] Generated demo scores: {baseline_path}")
    print(f"[✓] Generated demo scores: {ablation_path}")
    
    return baseline_path, ablation_path


def plot_score_distribution_demo(output_dir: str):
    """使用合成数据演示分数分布可视化"""
    baseline_path, ablation_path = generate_synthetic_scores_for_demo(output_dir)
    
    analyzer = ScoreDistributionAnalyzer(output_dir)
    analyzer.load_scores_from_file(baseline_path, ablation_path)
    analyzer.plot_score_distribution(top_k=10, filename="score_distribution_demo.png")


def plot_score_distribution_from_files(
    output_dir: str,
    baseline_scores_path: str,
    ablation_scores_path: str,
    baseline_label: str = "With Cross+SENet",
    ablation_label: str = "W/O Cross+SENet",
    filename: str = "score_distribution_real.png"
):
    """
    使用真实保存的分数文件绘制分数分布对比图
    
    Args:
        output_dir: 输出目录
        baseline_scores_path: Baseline模型的topk_scores.npy路径
        ablation_scores_path: Ablation模型的topk_scores.npy路径
        baseline_label: Baseline图例标签
        ablation_label: Ablation图例标签
        filename: 输出文件名
    """
    # 加载分数（优先使用topk_scores，如果不存在则用全量scores）
    def load_scores(path):
        topk_path = path.replace('_scores.npy', '_topk_scores.npy')
        if os.path.exists(topk_path):
            return np.load(topk_path)
        elif os.path.exists(path):
            return np.load(path)
        else:
            raise FileNotFoundError(f"Score file not found: {path}")
    
    baseline_scores = load_scores(baseline_scores_path)
    ablation_scores = load_scores(ablation_scores_path)
    
    print(f"[✓] Loaded baseline scores: {baseline_scores.shape}")
    print(f"[✓] Loaded ablation scores: {ablation_scores.shape}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图1: Top-10分数分布直方图
    ax = axes[0]
    top_k = min(10, baseline_scores.shape[1])
    baseline_topk = baseline_scores[:, :top_k]
    ablation_topk = ablation_scores[:, :top_k]
    
    ax.hist(baseline_topk.flatten(), bins=50, alpha=0.6, label=baseline_label, color='#3498db')
    ax.hist(ablation_topk.flatten(), bins=50, alpha=0.6, label=ablation_label, color='#e74c3c')
    ax.set_xlabel('Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Top-{top_k} Score Distribution', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # 图2: 分数方差对比
    ax = axes[1]
    baseline_std = np.std(baseline_topk, axis=1)
    ablation_std = np.std(ablation_topk, axis=1)
    
    ax.hist(baseline_std, bins=50, alpha=0.6, label=baseline_label, color='#3498db')
    ax.hist(ablation_std, bins=50, alpha=0.6, label=ablation_label, color='#e74c3c')
    ax.axvline(x=baseline_std.mean(), color='#3498db', linestyle='--', linewidth=2)
    ax.axvline(x=ablation_std.mean(), color='#e74c3c', linestyle='--', linewidth=2)
    ax.set_xlabel('Score Std Dev (within Top-K)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Top-{top_k} Score Variance\n(Higher = More Discriminative)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # 图3: Top-1 vs Top-2 得分差距
    ax = axes[2]
    baseline_gap = baseline_scores[:, 0] - baseline_scores[:, 1]
    ablation_gap = ablation_scores[:, 0] - ablation_scores[:, 1]
    
    ax.hist(baseline_gap, bins=50, alpha=0.6, label=baseline_label, color='#3498db')
    ax.hist(ablation_gap, bins=50, alpha=0.6, label=ablation_label, color='#e74c3c')
    ax.axvline(x=baseline_gap.mean(), color='#3498db', linestyle='--', linewidth=2, 
               label=f'{baseline_label} Mean: {baseline_gap.mean():.3f}')
    ax.axvline(x=ablation_gap.mean(), color='#e74c3c', linestyle='--', linewidth=2,
               label=f'{ablation_label} Mean: {ablation_gap.mean():.3f}')
    ax.set_xlabel('Score Gap (Top-1 - Top-2)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Top-1 Confidence Gap\n(Higher = More Confident Top-1)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Score Distribution Analysis: Cross+SENet Effect on Ranking Precision', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {output_path}")
    
    # 打印统计摘要
    print("\n=== Score Distribution Summary ===")
    print(f"Top-{top_k} Score Std Dev:")
    print(f"  {baseline_label}: mean={baseline_std.mean():.4f}, std={baseline_std.std():.4f}")
    print(f"  {ablation_label}: mean={ablation_std.mean():.4f}, std={ablation_std.std():.4f}")
    print(f"  Difference: {(baseline_std.mean() - ablation_std.mean()) / ablation_std.mean() * 100:+.1f}%")
    print(f"\nTop-1 vs Top-2 Gap:")
    print(f"  {baseline_label}: mean={baseline_gap.mean():.4f}")
    print(f"  {ablation_label}: mean={ablation_gap.mean():.4f}")
    print(f"  Difference: {(baseline_gap.mean() - ablation_gap.mean()) / ablation_gap.mean() * 100:+.1f}%")
    
    return output_path


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='消融实验可视化')
    parser.add_argument('--output-dir', type=str, default='ablation_study_doc',
                        help='输出目录')
    parser.add_argument('--plot-all', action='store_true', 
                        help='生成所有图表')
    parser.add_argument('--plot-tradeoff', action='store_true',
                        help='生成Recall vs MRR trade-off曲线')
    parser.add_argument('--plot-coldstart', action='store_true',
                        help='生成冷启动对比图')
    parser.add_argument('--plot-contribution', action='store_true',
                        help='生成模块贡献对比图')
    parser.add_argument('--plot-score-dist-demo', action='store_true',
                        help='使用合成数据生成分数分布演示图')
    # 真实分数可视化参数
    parser.add_argument('--plot-score-dist', action='store_true',
                        help='使用真实保存的分数文件生成分数分布图')
    parser.add_argument('--baseline-scores', type=str, default=None,
                        help='Baseline模型的分数文件路径 (e.g., scores/baseline_phase_b_topk_scores.npy)')
    parser.add_argument('--ablation-scores', type=str, default=None,
                        help='Ablation模型的分数文件路径')
    parser.add_argument('--baseline-label', type=str, default='With Cross+SENet',
                        help='Baseline图例标签')
    parser.add_argument('--ablation-label', type=str, default='W/O Cross+SENet',
                        help='Ablation图例标签')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.plot_all or args.plot_tradeoff:
        plot_tradeoff_curve(args.output_dir)
    
    if args.plot_all or args.plot_coldstart:
        plot_coldstart_comparison(args.output_dir)
    
    if args.plot_all or args.plot_contribution:
        plot_module_contribution(args.output_dir)
    
    if args.plot_score_dist_demo:
        plot_score_distribution_demo(args.output_dir)
    
    if args.plot_score_dist:
        if not args.baseline_scores or not args.ablation_scores:
            print("[!] Error: --plot-score-dist requires --baseline-scores and --ablation-scores")
            print("    Example:")
            print("    python visualize_ablation.py --plot-score-dist \\")
            print("        --baseline-scores scores/multiview_v2_phase_b_topk_scores.npy \\")
            print("        --ablation-scores scores/multiview_v2_no_cross_senet_phase_b_topk_scores.npy")
            return
        plot_score_distribution_from_files(
            args.output_dir,
            args.baseline_scores,
            args.ablation_scores,
            baseline_label=args.baseline_label,
            ablation_label=args.ablation_label
        )
    
    if not any([args.plot_all, args.plot_tradeoff, args.plot_coldstart, 
                args.plot_contribution, args.plot_score_dist_demo, args.plot_score_dist]):
        print("消融实验可视化工具")
        print("=" * 60)
        print("\n基础图表 (使用硬编码的实验数据):")
        print("  python visualize_ablation.py --plot-all           # 生成所有基础图表")
        print("  python visualize_ablation.py --plot-tradeoff      # Trade-off曲线")
        print("  python visualize_ablation.py --plot-coldstart     # 冷启动对比图")
        print("  python visualize_ablation.py --plot-contribution  # 模块贡献图")
        print("\n分数分布图 (需要真实分数文件):")
        print("  python visualize_ablation.py --plot-score-dist-demo  # 使用合成数据演示")
        print("  python visualize_ablation.py --plot-score-dist \\")
        print("      --baseline-scores scores/baseline_topk_scores.npy \\")
        print("      --ablation-scores scores/ablation_topk_scores.npy")
        print("\n如何获取真实分数文件:")
        print("  在训练脚本中添加 --save_test_scores 参数:")
        print("  1. run_recbole.py --save_test_scores --variant_name baseline")
        print("  2. two_phase_train.py --save_test_scores")
        print("  分数将保存到 ablation_study_doc/scores/ 目录")


if __name__ == '__main__':
    main()

