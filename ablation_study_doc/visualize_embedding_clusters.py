#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Embedding聚类和降维可视化脚本
============================

使用t-SNE、PCA、UMAP等方法对Beauty和Toys的embedding进行降维可视化，
分析whiten对聚类结构的影响。

可视化内容：
1. t-SNE散点图：展示embedding空间的聚类结构
2. PCA分析：展示主成分方差分布
3. 特征维度分析：方差分布、有效维度等
4. K-Means聚类：展示聚类紧密度差异
5. 局部结构分析：最近邻距离分布

Usage:
    python visualize_embedding_clusters.py --tsne           # t-SNE可视化
    python visualize_embedding_clusters.py --pca            # PCA分析
    python visualize_embedding_clusters.py --cluster        # 聚类分析
    python visualize_embedding_clusters.py --all            # 所有分析
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配色方案
COLORS = {
    'beauty': '#E74C3C',
    'beauty_light': '#F5B7B1',
    'toys': '#3498DB', 
    'toys_light': '#AED6F1',
    'whiten': '#2ECC71',
    'no_whiten': '#F39C12',
    'cluster_colors': ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12', 
                       '#1ABC9C', '#E67E22', '#34495E', '#95A5A6', '#D35400'],
}


# ============================================================================
# 工具函数
# ============================================================================

def load_embedding(path: str, skip_pad: bool = True) -> np.ndarray:
    """加载embedding文件"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")
    
    emb = np.load(path)
    print(f"[✓] Loaded: {path}")
    print(f"    Shape: {emb.shape}, dtype: {emb.dtype}")
    
    if skip_pad:
        emb = emb[1:]  # 跳过PAD token
    
    if emb.dtype == np.float16:
        emb = emb.astype(np.float32)
    
    return emb


def sample_embeddings(emb: np.ndarray, n_samples: int = 5000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """采样embedding"""
    np.random.seed(seed)
    if len(emb) > n_samples:
        indices = np.random.choice(len(emb), n_samples, replace=False)
        return emb[indices], indices
    return emb, np.arange(len(emb))


def l2_normalize(emb: np.ndarray) -> np.ndarray:
    """L2归一化"""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-8, None)


# ============================================================================
# t-SNE可视化
# ============================================================================

def plot_tsne_comparison(
    beauty_whiten: np.ndarray,
    beauty_no_whiten: np.ndarray,
    toys_whiten: np.ndarray,
    toys_no_whiten: np.ndarray,
    output_path: str,
    n_samples: int = 3000,
    perplexity: int = 30
):
    """
    t-SNE降维对比可视化
    
    展示4个场景：Beauty/Toys × Whiten/No-Whiten
    """
    print("\n" + "="*60)
    print("Running t-SNE visualization...")
    print("="*60)
    
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[!] sklearn not found, please install: pip install scikit-learn")
        return None
    
    # 采样
    bw, _ = sample_embeddings(beauty_whiten, n_samples)
    bnw, _ = sample_embeddings(beauty_no_whiten, n_samples)
    tw, _ = sample_embeddings(toys_whiten, n_samples)
    tnw, _ = sample_embeddings(toys_no_whiten, n_samples)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    datasets = [
        (bnw, 'Beauty (No Whiten)', COLORS['no_whiten'], axes[0, 0]),
        (bw, 'Beauty (With Whiten)', COLORS['whiten'], axes[0, 1]),
        (tnw, 'Toys (No Whiten)', COLORS['no_whiten'], axes[1, 0]),
        (tw, 'Toys (With Whiten)', COLORS['whiten'], axes[1, 1]),
    ]
    
    for emb, title, color, ax in datasets:
        print(f"  Computing t-SNE for {title}...")
        # 兼容不同版本的sklearn
        try:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                        max_iter=1000, learning_rate='auto', init='pca')
        except TypeError:
            # 旧版本sklearn使用n_iter
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                        n_iter=1000)
        emb_2d = tsne.fit_transform(emb)
        
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=color, alpha=0.4, s=8, edgecolors='none')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        
        # 计算并显示聚类紧密度（平均最近邻距离）
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=6)
        nn.fit(emb_2d)
        distances, _ = nn.kneighbors(emb_2d)
        avg_nn_dist = distances[:, 1:].mean()  # 排除自身
        
        ax.text(0.02, 0.98, f'Avg NN dist: {avg_nn_dist:.2f}', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('t-SNE Visualization: Embedding Space Structure\n(Lower NN distance = Tighter clusters)', 
                 fontsize=15, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Saved: {output_path}")
    return output_path


def plot_tsne_overlay(
    beauty_emb: np.ndarray,
    toys_emb: np.ndarray,
    output_path: str,
    n_samples: int = 2000,
    perplexity: int = 30,
    title_suffix: str = ""
):
    """
    t-SNE叠加可视化：Beauty和Toys在同一空间中
    
    展示两个数据集的embedding分布差异
    """
    print(f"\nRunning t-SNE overlay{title_suffix}...")
    
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[!] sklearn not found")
        return None
    
    # 采样
    beauty_sample, _ = sample_embeddings(beauty_emb, n_samples)
    toys_sample, _ = sample_embeddings(toys_emb, n_samples)
    
    # 合并
    combined = np.vstack([beauty_sample, toys_sample])
    labels = np.array(['Beauty'] * len(beauty_sample) + ['Toys'] * len(toys_sample))
    
    print(f"  Running t-SNE on {len(combined)} samples...")
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                    max_iter=1000, learning_rate='auto', init='pca')
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    combined_2d = tsne.fit_transform(combined)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 分别绘制
    beauty_mask = labels == 'Beauty'
    toys_mask = labels == 'Toys'
    
    ax.scatter(combined_2d[beauty_mask, 0], combined_2d[beauty_mask, 1], 
               c=COLORS['beauty'], alpha=0.5, s=15, label='Beauty', edgecolors='none')
    ax.scatter(combined_2d[toys_mask, 0], combined_2d[toys_mask, 1], 
               c=COLORS['toys'], alpha=0.5, s=15, label='Toys', edgecolors='none')
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(f't-SNE: Beauty vs Toys Embedding Distribution{title_suffix}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, markerscale=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Saved: {output_path}")
    return output_path


# ============================================================================
# PCA分析
# ============================================================================

def plot_pca_analysis(
    beauty_whiten: np.ndarray,
    beauty_no_whiten: np.ndarray,
    toys_whiten: np.ndarray,
    toys_no_whiten: np.ndarray,
    output_path: str,
    n_samples: int = 10000
):
    """
    PCA主成分分析
    
    展示：
    1. 累积方差解释比例
    2. 有效维度（达到90%方差需要多少维）
    3. 主成分散点图
    """
    print("\n" + "="*60)
    print("Running PCA analysis...")
    print("="*60)
    
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("[!] sklearn not found")
        return None
    
    # 采样
    bw, _ = sample_embeddings(beauty_whiten, n_samples)
    bnw, _ = sample_embeddings(beauty_no_whiten, n_samples)
    tw, _ = sample_embeddings(toys_whiten, n_samples)
    tnw, _ = sample_embeddings(toys_no_whiten, n_samples)
    
    fig = plt.figure(figsize=(16, 12))
    
    # ========== 上排：累积方差曲线 ==========
    ax1 = fig.add_subplot(2, 2, 1)
    
    datasets = [
        (bnw, 'Beauty No-Whiten', COLORS['beauty'], '--'),
        (bw, 'Beauty Whiten', COLORS['beauty'], '-'),
        (tnw, 'Toys No-Whiten', COLORS['toys'], '--'),
        (tw, 'Toys Whiten', COLORS['toys'], '-'),
    ]
    
    eff_dims = {}
    for emb, label, color, ls in datasets:
        pca = PCA(n_components=min(100, emb.shape[1]))
        pca.fit(emb)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        
        ax1.plot(range(1, len(cumvar)+1), cumvar, label=label, color=color, linestyle=ls, linewidth=2)
        
        # 计算有效维度（达到90%方差）
        eff_dim = np.argmax(cumvar >= 0.9) + 1
        eff_dims[label] = eff_dim
        print(f"  {label}: Effective dim (90% var) = {eff_dim}")
    
    ax1.axhline(y=0.9, color='gray', linestyle=':', alpha=0.7, label='90% threshold')
    ax1.set_xlabel('Number of Principal Components', fontsize=12)
    ax1.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax1.set_title('PCA: Cumulative Variance Explained\n(Fewer dims needed = More concentrated info)', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1.05)
    
    # ========== 右上：有效维度对比柱状图 ==========
    ax2 = fig.add_subplot(2, 2, 2)
    
    labels_bar = ['Beauty\nNo-Whiten', 'Beauty\nWhiten', 'Toys\nNo-Whiten', 'Toys\nWhiten']
    values = [eff_dims['Beauty No-Whiten'], eff_dims['Beauty Whiten'],
              eff_dims['Toys No-Whiten'], eff_dims['Toys Whiten']]
    colors_bar = [COLORS['no_whiten'], COLORS['whiten'], COLORS['no_whiten'], COLORS['whiten']]
    
    bars = ax2.bar(labels_bar, values, color=colors_bar, alpha=0.8, edgecolor='black')
    
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Effective Dimensions (for 90% variance)', fontsize=12)
    ax2.set_title('Effective Dimensionality Comparison\n(Lower = More information concentrated)', fontsize=13)
    ax2.grid(axis='y', alpha=0.3)
    
    # ========== 下排：PC1 vs PC2 散点图 ==========
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Beauty对比
    pca_bnw = PCA(n_components=2).fit_transform(bnw[:2000])
    pca_bw = PCA(n_components=2).fit_transform(bw[:2000])
    
    ax3.scatter(pca_bnw[:, 0], pca_bnw[:, 1], c=COLORS['no_whiten'], alpha=0.3, s=10, label='No Whiten')
    ax3.scatter(pca_bw[:, 0], pca_bw[:, 1], c=COLORS['whiten'], alpha=0.3, s=10, label='Whiten')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_title('Beauty: PCA Projection', fontsize=13)
    ax3.legend()
    
    # Toys对比
    pca_tnw = PCA(n_components=2).fit_transform(tnw[:2000])
    pca_tw = PCA(n_components=2).fit_transform(tw[:2000])
    
    ax4.scatter(pca_tnw[:, 0], pca_tnw[:, 1], c=COLORS['no_whiten'], alpha=0.3, s=10, label='No Whiten')
    ax4.scatter(pca_tw[:, 0], pca_tw[:, 1], c=COLORS['whiten'], alpha=0.3, s=10, label='Whiten')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_title('Toys: PCA Projection', fontsize=13)
    ax4.legend()
    
    plt.suptitle('PCA Analysis: Information Concentration', fontsize=15, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Saved: {output_path}")
    return output_path


# ============================================================================
# 特征维度分析
# ============================================================================

def plot_dimension_analysis(
    beauty_whiten: np.ndarray,
    beauty_no_whiten: np.ndarray,
    toys_whiten: np.ndarray,
    toys_no_whiten: np.ndarray,
    output_path: str,
    n_samples: int = 10000
):
    """
    特征维度分析
    
    展示：
    1. 每个维度的方差分布
    2. 方差分布直方图
    3. 维度相关性热力图
    """
    print("\n" + "="*60)
    print("Running dimension analysis...")
    print("="*60)
    
    # 采样
    bw, _ = sample_embeddings(beauty_whiten, n_samples)
    bnw, _ = sample_embeddings(beauty_no_whiten, n_samples)
    tw, _ = sample_embeddings(toys_whiten, n_samples)
    tnw, _ = sample_embeddings(toys_no_whiten, n_samples)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # ========== 第一行：方差曲线 ==========
    # 左：每个维度的方差（排序后）
    ax = axes[0, 0]
    
    for emb, label, color, ls in [
        (bnw, 'Beauty No-Whiten', COLORS['beauty'], '--'),
        (bw, 'Beauty Whiten', COLORS['beauty'], '-'),
        (tnw, 'Toys No-Whiten', COLORS['toys'], '--'),
        (tw, 'Toys Whiten', COLORS['toys'], '-'),
    ]:
        var = np.var(emb, axis=0)
        var_sorted = np.sort(var)[::-1]
        ax.plot(var_sorted, label=label, color=color, linestyle=ls, linewidth=1.5)
    
    ax.set_xlabel('Dimension Index (sorted by variance)', fontsize=11)
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_title('Per-dimension Variance (Sorted)\n(Whiten should flatten this curve)', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    
    # 中：方差分布直方图
    ax = axes[0, 1]
    
    ax.hist(np.var(bnw, axis=0), bins=50, alpha=0.5, label='Beauty No-Whiten', color=COLORS['beauty'])
    ax.hist(np.var(bw, axis=0), bins=50, alpha=0.5, label='Beauty Whiten', color=COLORS['whiten'])
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Ideal (1.0)')
    ax.set_xlabel('Variance', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Beauty: Dimension Variance Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    ax = axes[0, 2]
    ax.hist(np.var(tnw, axis=0), bins=50, alpha=0.5, label='Toys No-Whiten', color=COLORS['toys'])
    ax.hist(np.var(tw, axis=0), bins=50, alpha=0.5, label='Toys Whiten', color=COLORS['whiten'])
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Ideal (1.0)')
    ax.set_xlabel('Variance', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Toys: Dimension Variance Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # ========== 第二行：相关性矩阵 ==========
    # 计算特征间相关性
    def compute_correlation_matrix(emb, n_dims=64):
        """计算前n_dims维度的相关性矩阵"""
        emb_subset = emb[:, :n_dims]
        centered = emb_subset - emb_subset.mean(axis=0)
        cov = (centered.T @ centered) / len(centered)
        std = np.sqrt(np.diag(cov))
        corr = cov / (np.outer(std, std) + 1e-8)
        return corr
    
    n_dims = 64
    
    # Beauty相关性对比
    ax = axes[1, 0]
    corr_bnw = compute_correlation_matrix(bnw, n_dims)
    im = ax.imshow(np.abs(corr_bnw), cmap='Reds', vmin=0, vmax=1)
    ax.set_title(f'Beauty No-Whiten\nCorrelation (first {n_dims} dims)', fontsize=11)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    ax = axes[1, 1]
    corr_bw = compute_correlation_matrix(bw, n_dims)
    im = ax.imshow(np.abs(corr_bw), cmap='Reds', vmin=0, vmax=1)
    ax.set_title(f'Beauty Whiten\nCorrelation (first {n_dims} dims)', fontsize=11)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 相关性统计对比
    ax = axes[1, 2]
    
    def off_diag_stats(corr):
        mask = ~np.eye(corr.shape[0], dtype=bool)
        off_diag = np.abs(corr[mask])
        return off_diag.mean(), off_diag.max()
    
    stats_data = [
        ('Beauty\nNo-Whiten', *off_diag_stats(corr_bnw), COLORS['no_whiten']),
        ('Beauty\nWhiten', *off_diag_stats(compute_correlation_matrix(bw, n_dims)), COLORS['whiten']),
        ('Toys\nNo-Whiten', *off_diag_stats(compute_correlation_matrix(tnw, n_dims)), COLORS['no_whiten']),
        ('Toys\nWhiten', *off_diag_stats(compute_correlation_matrix(tw, n_dims)), COLORS['whiten']),
    ]
    
    x = np.arange(len(stats_data))
    means = [s[1] for s in stats_data]
    maxes = [s[2] for s in stats_data]
    colors_bar = [s[3] for s in stats_data]
    labels = [s[0] for s in stats_data]
    
    width = 0.35
    bars1 = ax.bar(x - width/2, means, width, label='Mean |corr|', color=colors_bar, alpha=0.7)
    bars2 = ax.bar(x + width/2, maxes, width, label='Max |corr|', color=colors_bar, alpha=0.4, hatch='//')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Correlation', fontsize=11)
    ax.set_title('Off-diagonal Correlation Stats\n(Lower = Better decorrelation)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Dimension-wise Analysis: Variance and Correlation', fontsize=15, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Saved: {output_path}")
    return output_path


# ============================================================================
# K-Means聚类分析
# ============================================================================

def plot_clustering_analysis(
    beauty_whiten: np.ndarray,
    beauty_no_whiten: np.ndarray,
    toys_whiten: np.ndarray,
    toys_no_whiten: np.ndarray,
    output_path: str,
    n_samples: int = 5000,
    n_clusters: int = 10
):
    """
    K-Means聚类分析
    
    展示：
    1. 聚类后的t-SNE可视化
    2. 轮廓系数对比
    3. 簇内距离分布
    """
    print("\n" + "="*60)
    print("Running clustering analysis...")
    print("="*60)
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.manifold import TSNE
    except ImportError:
        print("[!] sklearn not found")
        return None
    
    # 采样
    bw, _ = sample_embeddings(beauty_whiten, n_samples)
    bnw, _ = sample_embeddings(beauty_no_whiten, n_samples)
    tw, _ = sample_embeddings(toys_whiten, n_samples)
    tnw, _ = sample_embeddings(toys_no_whiten, n_samples)
    
    fig = plt.figure(figsize=(16, 12))
    
    datasets = [
        (bnw, 'Beauty No-Whiten', 0),
        (bw, 'Beauty Whiten', 1),
        (tnw, 'Toys No-Whiten', 2),
        (tw, 'Toys Whiten', 3),
    ]
    
    silhouette_scores = {}
    intra_cluster_dists = {}
    
    for emb, label, idx in datasets:
        print(f"  Clustering {label}...")
        
        # K-Means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(emb)
        
        # 轮廓系数
        sil_score = silhouette_score(emb, cluster_labels)
        silhouette_scores[label] = sil_score
        print(f"    Silhouette score: {sil_score:.4f}")
        
        # 簇内平均距离
        intra_dists = []
        for c in range(n_clusters):
            mask = cluster_labels == c
            if mask.sum() > 1:
                cluster_points = emb[mask]
                center = cluster_points.mean(axis=0)
                dists = np.linalg.norm(cluster_points - center, axis=1)
                intra_dists.extend(dists)
        intra_cluster_dists[label] = np.array(intra_dists)
        
        # t-SNE可视化（使用较小样本加速）
        emb_small = emb[:2000]
        labels_small = cluster_labels[:2000]
        
        try:
            tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=500)
        except TypeError:
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        emb_2d = tsne.fit_transform(emb_small)
        
        ax = fig.add_subplot(2, 3, idx + 1)
        for c in range(n_clusters):
            mask = labels_small == c
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                       c=COLORS['cluster_colors'][c % len(COLORS['cluster_colors'])],
                       alpha=0.5, s=10, label=f'Cluster {c}')
        
        ax.set_title(f'{label}\nSilhouette: {sil_score:.3f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
    
    # ========== 右下角：统计对比 ==========
    ax = fig.add_subplot(2, 3, 5)
    
    labels_bar = list(silhouette_scores.keys())
    values = list(silhouette_scores.values())
    colors_bar = [COLORS['no_whiten'], COLORS['whiten'], COLORS['no_whiten'], COLORS['whiten']]
    
    bars = ax.bar(range(len(labels_bar)), values, color=colors_bar, alpha=0.8, edgecolor='black')
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(range(len(labels_bar)))
    ax.set_xticklabels([l.replace(' ', '\n') for l in labels_bar], fontsize=9)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_title(f'Clustering Quality (K={n_clusters})\n(Higher = Better separated clusters)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(values) * 1.2)
    
    # 簇内距离箱线图
    ax = fig.add_subplot(2, 3, 6)
    
    bp_data = [intra_cluster_dists[l] for l in labels_bar]
    bp = ax.boxplot(bp_data, labels=[l.replace(' ', '\n') for l in labels_bar], patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors_bar):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Intra-cluster Distance', fontsize=11)
    ax.set_title('Cluster Compactness\n(Lower = Tighter clusters)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'K-Means Clustering Analysis (K={n_clusters})', fontsize=15, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Saved: {output_path}")
    return output_path


# ============================================================================
# 最近邻分析
# ============================================================================

def plot_nearest_neighbor_analysis(
    beauty_whiten: np.ndarray,
    beauty_no_whiten: np.ndarray,
    toys_whiten: np.ndarray,
    toys_no_whiten: np.ndarray,
    output_path: str,
    n_samples: int = 5000,
    k: int = 10
):
    """
    最近邻距离分析
    
    展示k-NN距离分布，反映局部结构的紧密程度
    """
    print("\n" + "="*60)
    print("Running nearest neighbor analysis...")
    print("="*60)
    
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        print("[!] sklearn not found")
        return None
    
    # 采样
    bw, _ = sample_embeddings(beauty_whiten, n_samples)
    bnw, _ = sample_embeddings(beauty_no_whiten, n_samples)
    tw, _ = sample_embeddings(toys_whiten, n_samples)
    tnw, _ = sample_embeddings(toys_no_whiten, n_samples)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    datasets = [
        (bnw, bw, 'Beauty', axes[0, 0], axes[0, 1]),
        (tnw, tw, 'Toys', axes[1, 0], axes[1, 1]),
    ]
    
    all_stats = {}
    
    for emb_nw, emb_w, name, ax1, ax2 in datasets:
        # 计算k-NN距离
        nn_nw = NearestNeighbors(n_neighbors=k+1)
        nn_nw.fit(l2_normalize(emb_nw))
        dist_nw, _ = nn_nw.kneighbors(l2_normalize(emb_nw))
        dist_nw = dist_nw[:, 1:]  # 排除自身
        
        nn_w = NearestNeighbors(n_neighbors=k+1)
        nn_w.fit(l2_normalize(emb_w))
        dist_w, _ = nn_w.kneighbors(l2_normalize(emb_w))
        dist_w = dist_w[:, 1:]
        
        # 平均k-NN距离
        avg_dist_nw = dist_nw.mean(axis=1)
        avg_dist_w = dist_w.mean(axis=1)
        
        all_stats[f'{name} No-Whiten'] = avg_dist_nw.mean()
        all_stats[f'{name} Whiten'] = avg_dist_w.mean()
        
        # 左图：距离分布直方图
        ax1.hist(avg_dist_nw, bins=50, alpha=0.6, label=f'No Whiten (mean={avg_dist_nw.mean():.4f})', 
                 color=COLORS['no_whiten'], density=True)
        ax1.hist(avg_dist_w, bins=50, alpha=0.6, label=f'Whiten (mean={avg_dist_w.mean():.4f})', 
                 color=COLORS['whiten'], density=True)
        ax1.axvline(x=avg_dist_nw.mean(), color=COLORS['no_whiten'], linestyle='--', linewidth=2)
        ax1.axvline(x=avg_dist_w.mean(), color=COLORS['whiten'], linestyle='--', linewidth=2)
        ax1.set_xlabel(f'Average {k}-NN Distance', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title(f'{name}: k-NN Distance Distribution', fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        
        # 右图：不同k值的距离变化
        k_values = [1, 3, 5, 10, 20, 50]
        mean_dists_nw = []
        mean_dists_w = []
        
        for kv in k_values:
            if kv <= k:
                mean_dists_nw.append(dist_nw[:, :kv].mean())
                mean_dists_w.append(dist_w[:, :kv].mean())
            else:
                nn_temp = NearestNeighbors(n_neighbors=kv+1)
                nn_temp.fit(l2_normalize(emb_nw))
                d, _ = nn_temp.kneighbors(l2_normalize(emb_nw))
                mean_dists_nw.append(d[:, 1:].mean())
                
                nn_temp.fit(l2_normalize(emb_w))
                d, _ = nn_temp.kneighbors(l2_normalize(emb_w))
                mean_dists_w.append(d[:, 1:].mean())
        
        ax2.plot(k_values, mean_dists_nw, 'o-', label='No Whiten', color=COLORS['no_whiten'], linewidth=2)
        ax2.plot(k_values, mean_dists_w, 's-', label='Whiten', color=COLORS['whiten'], linewidth=2)
        ax2.set_xlabel('k (number of neighbors)', fontsize=11)
        ax2.set_ylabel('Mean k-NN Distance', fontsize=11)
        ax2.set_title(f'{name}: Distance vs k', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
    
    plt.suptitle(f'Nearest Neighbor Analysis (k={k})\n(Lower distance = Denser local structure)', 
                 fontsize=15, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Saved: {output_path}")
    
    # 打印统计
    print("\n=== k-NN Distance Statistics ===")
    for name, val in all_stats.items():
        print(f"  {name}: {val:.4f}")
    
    return output_path


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
    parser = argparse.ArgumentParser(description='Embedding聚类和降维可视化')
    parser.add_argument('--output-dir', type=str, default='ablation_study_doc',
                        help='输出目录')
    
    # 文件路径参数
    parser.add_argument('--beauty-whiten', type=str, default=None)
    parser.add_argument('--beauty-no-whiten', type=str, default=None)
    parser.add_argument('--toys-whiten', type=str, default=None)
    parser.add_argument('--toys-no-whiten', type=str, default=None)
    
    # 可视化选项
    parser.add_argument('--tsne', action='store_true', help='t-SNE可视化')
    parser.add_argument('--pca', action='store_true', help='PCA分析')
    parser.add_argument('--dimension', action='store_true', help='维度分析')
    parser.add_argument('--cluster', action='store_true', help='聚类分析')
    parser.add_argument('--knn', action='store_true', help='最近邻分析')
    parser.add_argument('--all', action='store_true', help='所有分析')
    
    parser.add_argument('--n-samples', type=int, default=5000, help='采样数量')
    parser.add_argument('--n-clusters', type=int, default=10, help='聚类数')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载文件路径
    default_paths = get_default_paths()
    paths = {
        'beauty_whiten': args.beauty_whiten or default_paths['beauty_whiten'],
        'beauty_no_whiten': args.beauty_no_whiten or default_paths['beauty_no_whiten'],
        'toys_whiten': args.toys_whiten or default_paths['toys_whiten'],
        'toys_no_whiten': args.toys_no_whiten or default_paths['toys_no_whiten'],
    }
    
    print("=" * 70)
    print("Embedding Clustering and Dimensionality Analysis")
    print("=" * 70)
    
    if not any([args.tsne, args.pca, args.dimension, args.cluster, args.knn, args.all]):
        print("""
用法示例:

  python visualize_embedding_clusters.py --all          # 所有分析
  python visualize_embedding_clusters.py --tsne         # t-SNE可视化
  python visualize_embedding_clusters.py --pca          # PCA分析
  python visualize_embedding_clusters.py --dimension    # 维度分析
  python visualize_embedding_clusters.py --cluster      # 聚类分析
  python visualize_embedding_clusters.py --knn          # 最近邻分析

生成的图表:
  - tsne_comparison.png         : t-SNE降维对比
  - tsne_overlay_*.png          : t-SNE叠加图
  - pca_analysis.png            : PCA主成分分析
  - dimension_analysis.png      : 维度方差和相关性
  - clustering_analysis.png     : K-Means聚类
  - knn_analysis.png            : 最近邻距离分析
        """)
        return
    
    try:
        # 加载embedding
        beauty_w = load_embedding(paths['beauty_whiten'])
        beauty_nw = load_embedding(paths['beauty_no_whiten'])
        toys_w = load_embedding(paths['toys_whiten'])
        toys_nw = load_embedding(paths['toys_no_whiten'])
    except FileNotFoundError as e:
        print(f"[!] Error: {e}")
        print("    Please provide valid embedding file paths")
        return
    
    if args.all or args.tsne:
        plot_tsne_comparison(
            beauty_w, beauty_nw, toys_w, toys_nw,
            os.path.join(args.output_dir, 'tsne_comparison.png'),
            args.n_samples
        )
        
        plot_tsne_overlay(
            beauty_nw, toys_nw,
            os.path.join(args.output_dir, 'tsne_overlay_no_whiten.png'),
            min(2000, args.n_samples),
            title_suffix=' (No Whiten)'
        )
        
        plot_tsne_overlay(
            beauty_w, toys_w,
            os.path.join(args.output_dir, 'tsne_overlay_whiten.png'),
            min(2000, args.n_samples),
            title_suffix=' (With Whiten)'
        )
    
    if args.all or args.pca:
        plot_pca_analysis(
            beauty_w, beauty_nw, toys_w, toys_nw,
            os.path.join(args.output_dir, 'pca_analysis.png'),
            args.n_samples
        )
    
    if args.all or args.dimension:
        plot_dimension_analysis(
            beauty_w, beauty_nw, toys_w, toys_nw,
            os.path.join(args.output_dir, 'dimension_analysis.png'),
            args.n_samples
        )
    
    if args.all or args.cluster:
        plot_clustering_analysis(
            beauty_w, beauty_nw, toys_w, toys_nw,
            os.path.join(args.output_dir, 'clustering_analysis.png'),
            args.n_samples,
            args.n_clusters
        )
    
    if args.all or args.knn:
        plot_nearest_neighbor_analysis(
            beauty_w, beauty_nw, toys_w, toys_nw,
            os.path.join(args.output_dir, 'knn_analysis.png'),
            args.n_samples
        )
    
    print("\n" + "=" * 70)
    print("All visualizations completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()

