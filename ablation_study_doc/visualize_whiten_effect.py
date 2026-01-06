#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Whiten效果可视化脚本
==================

展示Beauty vs Toys数据集的embedding分布差异，以及whiten对不同数据集的影响。

核心观点：
- Beauty: 物品间相似度高，需要whiten来"拉开距离"
- Toys: 物品间相似度低，whiten可能破坏原有的良好结构

可视化内容：
1. t-SNE/UMAP降维可视化：展示embedding空间分布
2. 相似度分布直方图：展示物品间余弦相似度分布
3. 协方差矩阵热力图：展示特征维度间的相关性
4. Whiten前后对比：展示whiten对分布的影响

Usage:
    # 在有embedding文件的服务器上运行
    python visualize_whiten_effect.py --dataset beauty --compare-whiten
    python visualize_whiten_effect.py --dataset toys --compare-whiten
    python visualize_whiten_effect.py --compare-datasets  # 对比Beauty vs Toys
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配色方案
COLORS = {
    'beauty': '#E74C3C',      # 红色
    'toys': '#3498DB',        # 蓝色
    'whiten': '#2ECC71',      # 绿色
    'no_whiten': '#F39C12',   # 橙色
}


# ============================================================================
# Embedding加载和统计
# ============================================================================

def load_embedding(path: str) -> np.ndarray:
    """加载embedding文件"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")
    
    emb = np.load(path)
    print(f"[✓] Loaded embedding: {path}")
    print(f"    Shape: {emb.shape}, dtype: {emb.dtype}")
    
    # 跳过PAD token (index 0)
    emb = emb[1:]
    
    # 转换为float32
    if emb.dtype == np.float16:
        emb = emb.astype(np.float32)
    
    return emb


def compute_similarity_stats(emb: np.ndarray, n_samples: int = 5000) -> Dict:
    """
    计算embedding的相似度统计
    
    Args:
        emb: embedding矩阵 [N, D]
        n_samples: 采样数量（避免全量计算太慢）
    
    Returns:
        统计字典
    """
    # 采样
    if len(emb) > n_samples:
        indices = np.random.choice(len(emb), n_samples, replace=False)
        sample_emb = emb[indices]
    else:
        sample_emb = emb
    
    # L2归一化
    norms = np.linalg.norm(sample_emb, axis=1, keepdims=True)
    sample_emb_normed = sample_emb / np.clip(norms, 1e-8, None)
    
    # 计算余弦相似度矩阵
    sim_matrix = sample_emb_normed @ sample_emb_normed.T
    
    # 取上三角（排除对角线和重复）
    triu_indices = np.triu_indices(len(sample_emb), k=1)
    similarities = sim_matrix[triu_indices]
    
    stats = {
        'mean': similarities.mean(),
        'std': similarities.std(),
        'median': np.median(similarities),
        'q25': np.percentile(similarities, 25),
        'q75': np.percentile(similarities, 75),
        'max': similarities.max(),
        'min': similarities.min(),
        'high_sim_ratio': (similarities > 0.5).mean(),  # 高相似度占比
        'similarities': similarities,  # 保存用于绘图
    }
    
    return stats


def compute_covariance_stats(emb: np.ndarray, n_samples: int = 5000) -> Dict:
    """
    计算embedding的协方差矩阵统计
    
    用于展示whiten效果：
    - 理想whiten后：对角线≈1.0，非对角线≈0
    """
    # 采样
    if len(emb) > n_samples:
        indices = np.random.choice(len(emb), n_samples, replace=False)
        sample_emb = emb[indices]
    else:
        sample_emb = emb
    
    # 中心化
    centered = sample_emb - sample_emb.mean(axis=0)
    
    # 计算协方差矩阵
    cov = (centered.T @ centered) / len(centered)
    
    # 统计
    diag = np.diag(cov)
    off_diag_mask = ~np.eye(cov.shape[0], dtype=bool)
    off_diag = cov[off_diag_mask]
    
    stats = {
        'diag_mean': diag.mean(),
        'diag_std': diag.std(),
        'off_diag_mean': np.abs(off_diag).mean(),
        'off_diag_max': np.abs(off_diag).max(),
        'cov_matrix': cov,  # 保存用于热力图
    }
    
    return stats


# ============================================================================
# 可视化函数
# ============================================================================

def plot_similarity_comparison(
    beauty_emb: np.ndarray,
    toys_emb: np.ndarray,
    output_path: str,
    n_samples: int = 5000
):
    """
    图1：相似度分布对比
    
    展示Beauty vs Toys的物品间相似度分布差异
    """
    print("\n[1] Computing similarity distributions...")
    
    beauty_stats = compute_similarity_stats(beauty_emb, n_samples)
    toys_stats = compute_similarity_stats(toys_emb, n_samples)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：相似度分布直方图
    ax = axes[0]
    ax.hist(beauty_stats['similarities'], bins=100, alpha=0.6, 
            label=f"Beauty (mean={beauty_stats['mean']:.4f})", color=COLORS['beauty'], density=True)
    ax.hist(toys_stats['similarities'], bins=100, alpha=0.6, 
            label=f"Toys (mean={toys_stats['mean']:.4f})", color=COLORS['toys'], density=True)
    
    # 添加均值线
    ax.axvline(x=beauty_stats['mean'], color=COLORS['beauty'], linestyle='--', linewidth=2)
    ax.axvline(x=toys_stats['mean'], color=COLORS['toys'], linestyle='--', linewidth=2)
    
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Item-Item Similarity Distribution\n(Higher mean = More correlated items)', fontsize=13)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # 右图：箱线图对比
    ax = axes[1]
    bp = ax.boxplot([beauty_stats['similarities'], toys_stats['similarities']], 
                    labels=['Beauty', 'Toys'],
                    patch_artist=True, showmeans=True)
    
    bp['boxes'][0].set_facecolor(COLORS['beauty'])
    bp['boxes'][1].set_facecolor(COLORS['toys'])
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_alpha(0.6)
    
    # 添加统计值标注
    for i, (stats, name) in enumerate([(beauty_stats, 'Beauty'), (toys_stats, 'Toys')]):
        ax.annotate(f"Mean: {stats['mean']:.4f}\nHigh-sim: {stats['high_sim_ratio']*100:.2f}%",
                    xy=(i+1, stats['q75']), xytext=(10, 20),
                    textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Similarity Statistics Comparison\n(Beauty items are more similar to each other)', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Beauty vs Toys: Item Embedding Similarity', fontsize=15, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Saved: {output_path}")
    
    # 打印统计
    print("\n=== Similarity Statistics ===")
    print(f"{'Metric':<25} {'Beauty':>12} {'Toys':>12} {'Diff':>12}")
    print("-" * 63)
    print(f"{'Mean Similarity':<25} {beauty_stats['mean']:>12.4f} {toys_stats['mean']:>12.4f} "
          f"{(beauty_stats['mean'] - toys_stats['mean']) / toys_stats['mean'] * 100:>+10.1f}%")
    print(f"{'Std Dev':<25} {beauty_stats['std']:>12.4f} {toys_stats['std']:>12.4f}")
    print(f"{'High-sim Ratio (>0.5)':<25} {beauty_stats['high_sim_ratio']*100:>11.2f}% {toys_stats['high_sim_ratio']*100:>11.2f}%")
    
    return beauty_stats, toys_stats


def plot_tsne_comparison(
    beauty_emb: np.ndarray,
    toys_emb: np.ndarray,
    output_path: str,
    n_samples: int = 2000,
    perplexity: int = 30
):
    """
    图2：t-SNE降维可视化
    
    展示embedding空间的分布形态
    """
    print("\n[2] Running t-SNE visualization...")
    
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[!] sklearn not found, skipping t-SNE visualization")
        return None
    
    # 采样
    beauty_sample = beauty_emb[np.random.choice(len(beauty_emb), min(n_samples, len(beauty_emb)), replace=False)]
    toys_sample = toys_emb[np.random.choice(len(toys_emb), min(n_samples, len(toys_emb)), replace=False)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Beauty t-SNE
    print("  - Computing t-SNE for Beauty...")
    tsne_beauty = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    beauty_2d = tsne_beauty.fit_transform(beauty_sample)
    
    ax = axes[0]
    ax.scatter(beauty_2d[:, 0], beauty_2d[:, 1], c=COLORS['beauty'], alpha=0.5, s=10)
    ax.set_title(f'Beauty Dataset\n(Items cluster together)', fontsize=13)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_aspect('equal')
    
    # Toys t-SNE
    print("  - Computing t-SNE for Toys...")
    tsne_toys = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    toys_2d = tsne_toys.fit_transform(toys_sample)
    
    ax = axes[1]
    ax.scatter(toys_2d[:, 0], toys_2d[:, 1], c=COLORS['toys'], alpha=0.5, s=10)
    ax.set_title(f'Toys Dataset\n(Items more spread out)', fontsize=13)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_aspect('equal')
    
    plt.suptitle('t-SNE Visualization: Embedding Space Distribution', fontsize=15, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Saved: {output_path}")
    return beauty_2d, toys_2d


def plot_covariance_heatmap(
    emb_whiten: np.ndarray,
    emb_no_whiten: np.ndarray,
    dataset_name: str,
    output_path: str,
    n_samples: int = 5000
):
    """
    图3：协方差矩阵热力图对比
    
    展示whiten前后的协方差矩阵变化
    - 理想whiten：对角线=1，非对角线=0（单位矩阵）
    """
    print(f"\n[3] Computing covariance matrices for {dataset_name}...")
    
    stats_whiten = compute_covariance_stats(emb_whiten, n_samples)
    stats_no_whiten = compute_covariance_stats(emb_no_whiten, n_samples)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 只显示前64维的协方差矩阵（完整矩阵太大）
    dim_show = 64
    
    # 左图：No Whiten协方差矩阵
    ax = axes[0]
    cov_show = stats_no_whiten['cov_matrix'][:dim_show, :dim_show]
    im = ax.imshow(cov_show, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_title(f'No Whiten (first {dim_show} dims)\nDiag: {stats_no_whiten["diag_mean"]:.3f}±{stats_no_whiten["diag_std"]:.3f}', 
                 fontsize=12)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 中图：With Whiten协方差矩阵
    ax = axes[1]
    cov_show = stats_whiten['cov_matrix'][:dim_show, :dim_show]
    im = ax.imshow(cov_show, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_title(f'With Whiten (first {dim_show} dims)\nDiag: {stats_whiten["diag_mean"]:.3f}±{stats_whiten["diag_std"]:.3f}', 
                 fontsize=12)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 右图：对角线值分布对比
    ax = axes[2]
    diag_no_whiten = np.diag(stats_no_whiten['cov_matrix'])
    diag_whiten = np.diag(stats_whiten['cov_matrix'])
    
    ax.hist(diag_no_whiten, bins=50, alpha=0.6, label='No Whiten', color=COLORS['no_whiten'])
    ax.hist(diag_whiten, bins=50, alpha=0.6, label='With Whiten', color=COLORS['whiten'])
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Ideal (1.0)')
    ax.set_xlabel('Diagonal Value (Variance)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Diagonal Values Distribution\n(Whiten should → 1.0)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'{dataset_name}: Covariance Matrix Comparison (Whiten Effect)', 
                 fontsize=15, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Saved: {output_path}")
    
    # 打印统计
    print(f"\n=== Covariance Statistics ({dataset_name}) ===")
    print(f"{'Metric':<30} {'No Whiten':>15} {'With Whiten':>15}")
    print("-" * 62)
    print(f"{'Diagonal Mean (ideal=1.0)':<30} {stats_no_whiten['diag_mean']:>15.4f} {stats_whiten['diag_mean']:>15.4f}")
    print(f"{'Diagonal Std (ideal=0)':<30} {stats_no_whiten['diag_std']:>15.4f} {stats_whiten['diag_std']:>15.4f}")
    print(f"{'Off-diagonal Mean (ideal=0)':<30} {stats_no_whiten['off_diag_mean']:>15.4f} {stats_whiten['off_diag_mean']:>15.4f}")
    
    return stats_whiten, stats_no_whiten


def plot_whiten_similarity_impact(
    emb_whiten: np.ndarray,
    emb_no_whiten: np.ndarray,
    dataset_name: str,
    output_path: str,
    n_samples: int = 5000
):
    """
    图4：Whiten对相似度分布的影响
    
    核心可视化：展示whiten如何改变物品间的相似度分布
    """
    print(f"\n[4] Analyzing whiten impact on similarity for {dataset_name}...")
    
    stats_whiten = compute_similarity_stats(emb_whiten, n_samples)
    stats_no_whiten = compute_similarity_stats(emb_no_whiten, n_samples)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 左图：相似度分布对比
    ax = axes[0]
    ax.hist(stats_no_whiten['similarities'], bins=100, alpha=0.6, 
            label=f"No Whiten (mean={stats_no_whiten['mean']:.4f})", 
            color=COLORS['no_whiten'], density=True)
    ax.hist(stats_whiten['similarities'], bins=100, alpha=0.6, 
            label=f"With Whiten (mean={stats_whiten['mean']:.4f})", 
            color=COLORS['whiten'], density=True)
    
    ax.axvline(x=stats_no_whiten['mean'], color=COLORS['no_whiten'], linestyle='--', linewidth=2)
    ax.axvline(x=stats_whiten['mean'], color=COLORS['whiten'], linestyle='--', linewidth=2)
    
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{dataset_name}: Similarity Distribution\n(Whiten changes the distribution)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # 中图：累积分布函数 (CDF)
    ax = axes[1]
    sorted_no_whiten = np.sort(stats_no_whiten['similarities'])
    sorted_whiten = np.sort(stats_whiten['similarities'])
    cdf_y = np.arange(1, len(sorted_no_whiten) + 1) / len(sorted_no_whiten)
    
    ax.plot(sorted_no_whiten, cdf_y, label='No Whiten', color=COLORS['no_whiten'], linewidth=2)
    ax.plot(sorted_whiten, cdf_y, label='With Whiten', color=COLORS['whiten'], linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution Function\n(CDF shows distribution shift)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # 右图：变化量统计
    ax = axes[2]
    metrics = ['Mean Sim', 'Std Dev', 'High-sim\nRatio (%)']
    no_whiten_vals = [stats_no_whiten['mean'], stats_no_whiten['std'], stats_no_whiten['high_sim_ratio']*100]
    whiten_vals = [stats_whiten['mean'], stats_whiten['std'], stats_whiten['high_sim_ratio']*100]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, no_whiten_vals, width, label='No Whiten', color=COLORS['no_whiten'], alpha=0.8)
    bars2 = ax.bar(x + width/2, whiten_vals, width, label='With Whiten', color=COLORS['whiten'], alpha=0.8)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title('Similarity Statistics Comparison', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'{dataset_name}: Whiten Effect on Item Similarity', 
                 fontsize=15, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Saved: {output_path}")
    
    # 关键分析输出
    mean_change = (stats_whiten['mean'] - stats_no_whiten['mean']) / stats_no_whiten['mean'] * 100
    print(f"\n=== Whiten Impact Analysis ({dataset_name}) ===")
    print(f"Mean Similarity Change: {mean_change:+.1f}%")
    if mean_change < -10:
        print(f"[!] Whiten significantly REDUCES similarity → Good for high-correlation datasets (like Beauty)")
    elif mean_change > 10:
        print(f"[!] Whiten significantly INCREASES similarity → May be harmful for low-correlation datasets")
    else:
        print(f"[!] Whiten has minimal effect on similarity distribution")
    
    return stats_whiten, stats_no_whiten


def plot_comprehensive_comparison(
    beauty_whiten: np.ndarray,
    beauty_no_whiten: np.ndarray,
    toys_whiten: np.ndarray,
    toys_no_whiten: np.ndarray,
    output_path: str,
    n_samples: int = 3000
):
    """
    图5：综合对比图（核心可视化）
    
    在一张图中展示：
    - Beauty vs Toys的差异
    - Whiten对两者的不同影响
    """
    print("\n[5] Creating comprehensive comparison...")
    
    # 计算所有统计
    beauty_w = compute_similarity_stats(beauty_whiten, n_samples)
    beauty_nw = compute_similarity_stats(beauty_no_whiten, n_samples)
    toys_w = compute_similarity_stats(toys_whiten, n_samples)
    toys_nw = compute_similarity_stats(toys_no_whiten, n_samples)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 使用GridSpec创建复杂布局
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # ========== 第一行：相似度分布对比 ==========
    # 左：Beauty (whiten vs no_whiten)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(beauty_nw['similarities'], bins=80, alpha=0.5, label='No Whiten', color=COLORS['no_whiten'], density=True)
    ax1.hist(beauty_w['similarities'], bins=80, alpha=0.5, label='With Whiten', color=COLORS['whiten'], density=True)
    ax1.axvline(x=beauty_nw['mean'], color=COLORS['no_whiten'], linestyle='--', linewidth=2)
    ax1.axvline(x=beauty_w['mean'], color=COLORS['whiten'], linestyle='--', linewidth=2)
    ax1.set_title(f'Beauty: Whiten Effect\n(No Whiten mean={beauty_nw["mean"]:.4f} → Whiten mean={beauty_w["mean"]:.4f})', 
                  fontsize=12, fontweight='bold', color=COLORS['beauty'])
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 右：Toys (whiten vs no_whiten)
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.hist(toys_nw['similarities'], bins=80, alpha=0.5, label='No Whiten', color=COLORS['no_whiten'], density=True)
    ax2.hist(toys_w['similarities'], bins=80, alpha=0.5, label='With Whiten', color=COLORS['whiten'], density=True)
    ax2.axvline(x=toys_nw['mean'], color=COLORS['no_whiten'], linestyle='--', linewidth=2)
    ax2.axvline(x=toys_w['mean'], color=COLORS['whiten'], linestyle='--', linewidth=2)
    ax2.set_title(f'Toys: Whiten Effect\n(No Whiten mean={toys_nw["mean"]:.4f} → Whiten mean={toys_w["mean"]:.4f})', 
                  fontsize=12, fontweight='bold', color=COLORS['toys'])
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # ========== 第二行：数据集对比（相同处理方式） ==========
    # 左：No Whiten下的Beauty vs Toys
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.hist(beauty_nw['similarities'], bins=80, alpha=0.5, label=f'Beauty (mean={beauty_nw["mean"]:.4f})', 
             color=COLORS['beauty'], density=True)
    ax3.hist(toys_nw['similarities'], bins=80, alpha=0.5, label=f'Toys (mean={toys_nw["mean"]:.4f})', 
             color=COLORS['toys'], density=True)
    ax3.set_title('No Whiten: Beauty vs Toys\n(Beauty items are more correlated)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Cosine Similarity')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 右：With Whiten下的Beauty vs Toys
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.hist(beauty_w['similarities'], bins=80, alpha=0.5, label=f'Beauty (mean={beauty_w["mean"]:.4f})', 
             color=COLORS['beauty'], density=True)
    ax4.hist(toys_w['similarities'], bins=80, alpha=0.5, label=f'Toys (mean={toys_w["mean"]:.4f})', 
             color=COLORS['toys'], density=True)
    ax4.set_title('With Whiten: Beauty vs Toys\n(Whiten reduces Beauty correlation more)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Cosine Similarity')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # ========== 第三行：统计对比柱状图 ==========
    ax5 = fig.add_subplot(gs[2, :2])
    
    # 对比Mean Similarity变化
    datasets = ['Beauty', 'Toys']
    no_whiten_means = [beauty_nw['mean'], toys_nw['mean']]
    whiten_means = [beauty_w['mean'], toys_w['mean']]
    changes = [(w - nw) / nw * 100 for w, nw in zip(whiten_means, no_whiten_means)]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, no_whiten_means, width, label='No Whiten', color=COLORS['no_whiten'], alpha=0.8)
    bars2 = ax5.bar(x + width/2, whiten_means, width, label='With Whiten', color=COLORS['whiten'], alpha=0.8)
    
    # 标注变化百分比
    for i, (bar1, bar2, change) in enumerate(zip(bars1, bars2, changes)):
        ax5.annotate(f'{change:+.1f}%', xy=(bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                     xytext=(15, 5), textcoords="offset points", fontsize=11, fontweight='bold',
                     color='green' if change < 0 else 'red')
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(datasets)
    ax5.set_ylabel('Mean Cosine Similarity')
    ax5.set_title('Whiten Impact Comparison\n(Negative change = More decorrelation)', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # 右侧：解释文字
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.axis('off')
    
    explanation = """
    KEY FINDINGS:
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    📊 Beauty Dataset (High Correlation):
       • Original items are highly similar (semantic clustering)
       • Whiten HELPS: reduces redundant correlations
       • Effect: Spreads items apart in embedding space
       • Result: Better discrimination for cold-start items
    
    📊 Toys Dataset (Low Correlation):
       • Items are already well-separated (diverse products)
       • Whiten may HURT: forces uniform variance
       • Effect: May distort natural semantic structure
       • Result: SENet loses discriminative signals
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    💡 RECOMMENDATION:
       • Beauty → Use Whiten (items need decorrelation)
       • Toys   → Skip Whiten (preserve natural structure)
    """
    ax6.text(0.1, 0.95, explanation, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))
    
    plt.suptitle('Whiten Effect Analysis: Beauty vs Toys', fontsize=16, y=0.98, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Saved: {output_path}")
    
    return {
        'beauty_whiten': beauty_w,
        'beauty_no_whiten': beauty_nw,
        'toys_whiten': toys_w,
        'toys_no_whiten': toys_nw,
    }


# ============================================================================
# 主函数
# ============================================================================

def get_default_paths():
    """获取默认的embedding文件路径"""
    base_dir = "/home/ubuntu/own/RecBole/dataset"
    
    return {
        'beauty_whiten': f"{base_dir}/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.npy",
        'beauty_no_whiten': f"{base_dir}/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.no_whiten.npy",
        'toys_whiten': f"{base_dir}/Amazon_Toys_and_Games/item_text_emb.qwen2.5_7b.base.npy",
        'toys_no_whiten': f"{base_dir}/Amazon_Toys_and_Games/item_text_emb.qwen2.5_7b.base.no_whiten.npy",
    }


def main():
    parser = argparse.ArgumentParser(description='Whiten效果可视化')
    parser.add_argument('--output-dir', type=str, default='ablation_study_doc',
                        help='输出目录')
    
    # 文件路径参数
    parser.add_argument('--beauty-whiten', type=str, default=None,
                        help='Beauty数据集whiten embedding路径')
    parser.add_argument('--beauty-no-whiten', type=str, default=None,
                        help='Beauty数据集no_whiten embedding路径')
    parser.add_argument('--toys-whiten', type=str, default=None,
                        help='Toys数据集whiten embedding路径')
    parser.add_argument('--toys-no-whiten', type=str, default=None,
                        help='Toys数据集no_whiten embedding路径')
    
    # 可视化选项
    parser.add_argument('--compare-datasets', action='store_true',
                        help='对比Beauty vs Toys的相似度分布')
    parser.add_argument('--compare-whiten', action='store_true',
                        help='对比Whiten vs No Whiten的效果')
    parser.add_argument('--comprehensive', action='store_true',
                        help='生成综合对比图（需要所有4个文件）')
    parser.add_argument('--tsne', action='store_true',
                        help='生成t-SNE可视化')
    parser.add_argument('--all', action='store_true',
                        help='生成所有可视化')
    
    parser.add_argument('--n-samples', type=int, default=5000,
                        help='采样数量（用于加速计算）')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 使用默认路径
    default_paths = get_default_paths()
    beauty_whiten_path = args.beauty_whiten or default_paths['beauty_whiten']
    beauty_no_whiten_path = args.beauty_no_whiten or default_paths['beauty_no_whiten']
    toys_whiten_path = args.toys_whiten or default_paths['toys_whiten']
    toys_no_whiten_path = args.toys_no_whiten or default_paths['toys_no_whiten']
    
    print("=" * 70)
    print("Whiten Effect Visualization")
    print("=" * 70)
    
    # 根据参数执行不同的可视化
    if args.all or args.comprehensive:
        # 需要所有4个文件
        try:
            beauty_w = load_embedding(beauty_whiten_path)
            beauty_nw = load_embedding(beauty_no_whiten_path)
            toys_w = load_embedding(toys_whiten_path)
            toys_nw = load_embedding(toys_no_whiten_path)
            
            plot_comprehensive_comparison(
                beauty_w, beauty_nw, toys_w, toys_nw,
                os.path.join(args.output_dir, 'whiten_comprehensive_comparison.png'),
                args.n_samples
            )
            
            if args.all:
                plot_similarity_comparison(
                    beauty_nw, toys_nw,
                    os.path.join(args.output_dir, 'similarity_beauty_vs_toys.png'),
                    args.n_samples
                )
                
                plot_whiten_similarity_impact(
                    beauty_w, beauty_nw, 'Beauty',
                    os.path.join(args.output_dir, 'whiten_impact_beauty.png'),
                    args.n_samples
                )
                
                plot_whiten_similarity_impact(
                    toys_w, toys_nw, 'Toys',
                    os.path.join(args.output_dir, 'whiten_impact_toys.png'),
                    args.n_samples
                )
                
                plot_covariance_heatmap(
                    beauty_w, beauty_nw, 'Beauty',
                    os.path.join(args.output_dir, 'covariance_beauty.png'),
                    args.n_samples
                )
                
                plot_covariance_heatmap(
                    toys_w, toys_nw, 'Toys',
                    os.path.join(args.output_dir, 'covariance_toys.png'),
                    args.n_samples
                )
                
        except FileNotFoundError as e:
            print(f"[!] Error: {e}")
            print("    Please provide valid embedding file paths")
            return
    
    elif args.compare_datasets:
        try:
            beauty_emb = load_embedding(beauty_no_whiten_path)
            toys_emb = load_embedding(toys_no_whiten_path)
            
            plot_similarity_comparison(
                beauty_emb, toys_emb,
                os.path.join(args.output_dir, 'similarity_beauty_vs_toys.png'),
                args.n_samples
            )
            
            if args.tsne:
                plot_tsne_comparison(
                    beauty_emb, toys_emb,
                    os.path.join(args.output_dir, 'tsne_beauty_vs_toys.png'),
                    min(2000, args.n_samples)
                )
                
        except FileNotFoundError as e:
            print(f"[!] Error: {e}")
            return
    
    elif args.compare_whiten:
        # 需要指定数据集
        print("[!] Please use --comprehensive or --all for whiten comparison")
        print("    Example: python visualize_whiten_effect.py --comprehensive")
    
    else:
        # 打印帮助信息
        print("""
Whiten效果可视化工具
=====================

用法示例:

1. 生成综合对比图（推荐，需要4个embedding文件）:
   python visualize_whiten_effect.py --comprehensive

2. 生成所有可视化:
   python visualize_whiten_effect.py --all

3. 只对比Beauty vs Toys的相似度:
   python visualize_whiten_effect.py --compare-datasets

4. 自定义文件路径:
   python visualize_whiten_effect.py --comprehensive \\
       --beauty-whiten /path/to/beauty_whiten.npy \\
       --beauty-no-whiten /path/to/beauty_no_whiten.npy \\
       --toys-whiten /path/to/toys_whiten.npy \\
       --toys-no-whiten /path/to/toys_no_whiten.npy

默认文件路径:
  - Beauty Whiten: {beauty_whiten}
  - Beauty No Whiten: {beauty_no_whiten}
  - Toys Whiten: {toys_whiten}
  - Toys No Whiten: {toys_no_whiten}
        """.format(**default_paths))


if __name__ == '__main__':
    main()


