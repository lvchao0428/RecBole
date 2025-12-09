#!/usr/bin/env python3
"""
Quick Embedding Statistics

快速统计Beauty和Toys数据集的embedding特征，
用于诊断multi-view在不同数据集上的表现差异。

Usage:
    python tools/quick_emb_stats.py
    python tools/quick_emb_stats.py --dataset Amazon_Beauty
    python tools/quick_emb_stats.py --dataset Amazon_Toys_and_Games
    python tools/quick_emb_stats.py --compare  # 对比两个数据集
"""

import argparse
import numpy as np
from pathlib import Path
import json
import sys

BASE_DIR = Path("/home/charlie/project/RecBole/dataset")


def quick_stats(emb, name, verbose=True):
    """计算embedding的快速统计"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    stats = {
        'name': name,
        'shape': list(emb.shape),
        'mean': float(np.mean(emb)),
        'std': float(np.std(emb)),
        'norm_mean': float(np.mean(np.linalg.norm(emb, axis=1))),
        'norm_std': float(np.std(np.linalg.norm(emb, axis=1))),
    }
    
    # 采样计算inter-item相似度
    n_samples = min(len(emb), 300)
    sampled = emb[:n_samples]
    cos_sim = cosine_similarity(sampled)
    mask = ~np.eye(cos_sim.shape[0], dtype=bool)
    similarities = cos_sim[mask]
    
    stats['inter_item_sim_mean'] = float(np.mean(similarities))
    stats['inter_item_sim_std'] = float(np.std(similarities))
    stats['inter_item_sim_p90'] = float(np.percentile(similarities, 90))
    stats['high_sim_ratio'] = float(np.mean(similarities > 0.9))  # >0.9的比例
    
    if verbose:
        print(f"  {name}:")
        print(f"    Shape: {emb.shape}")
        print(f"    Norm: {stats['norm_mean']:.4f} ± {stats['norm_std']:.4f}")
        print(f"    Inter-item sim: {stats['inter_item_sim_mean']:.4f} ± {stats['inter_item_sim_std']:.4f}")
        print(f"    High sim ratio (>0.9): {stats['high_sim_ratio']:.2%}")
    
    return stats


def compute_view_diversity(views_dict, n_samples=300, verbose=True):
    """计算视图间多样性"""
    from scipy.spatial.distance import cosine
    
    view_names = list(views_dict.keys())
    pairwise_sims = {}
    
    for i, name_i in enumerate(view_names):
        for j, name_j in enumerate(view_names):
            if i >= j:
                continue
            
            view_i = views_dict[name_i]
            view_j = views_dict[name_j]
            n = min(len(view_i), n_samples)
            
            sims = [1 - cosine(view_i[k], view_j[k]) for k in range(n)]
            pair_key = f"{name_i}_vs_{name_j}"
            pairwise_sims[pair_key] = float(np.mean(sims))
    
    avg_sim = np.mean(list(pairwise_sims.values()))
    diversity_score = 1 - avg_sim
    
    if verbose:
        print(f"\n  View Diversity:")
        for pair, sim in sorted(pairwise_sims.items()):
            print(f"    {pair}: {sim:.4f}")
        print(f"    --")
        print(f"    Diversity Score: {diversity_score:.4f} (higher = better)")
    
    return {
        'pairwise': pairwise_sims,
        'avg_similarity': float(avg_sim),
        'diversity_score': float(diversity_score),
    }


def analyze_dataset(dataset_name, verbose=True):
    """分析单个数据集"""
    dataset_dir = BASE_DIR / dataset_name
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
    
    results = {'dataset': dataset_name}
    
    # 1. Base embedding
    base_path = dataset_dir / "item_text_emb.base.npy"
    if base_path.exists():
        base_emb = np.load(base_path)
        results['base'] = quick_stats(base_emb, 'Base (TF-IDF)', verbose)
    
    # 2. Single LLM embedding
    llm_path = dataset_dir / "item_text_emb.qwen3.npy"
    if llm_path.exists():
        llm_emb = np.load(llm_path)
        results['llm'] = quick_stats(llm_emb, 'Single LLM', verbose)
    
    # 3. Multi-view embeddings
    mv_dir = dataset_dir / "qwen3_4views"
    if mv_dir.exists():
        views = {}
        view_names = ['identity', 'function', 'audience', 'category']
        results['views'] = {}
        
        if verbose:
            print(f"\n  Multi-View Stats:")
        
        for view in view_names:
            view_path = mv_dir / f"item_text_emb.qwen3.{view}.npy"
            if view_path.exists():
                view_emb = np.load(view_path)
                views[view] = view_emb
                results['views'][view] = quick_stats(view_emb, f"View[{view}]", verbose)
        
        # 计算视图多样性
        if len(views) > 1:
            results['view_diversity'] = compute_view_diversity(views, verbose=verbose)
    
    return results


def compare_datasets(verbose=True):
    """对比两个数据集"""
    beauty_results = analyze_dataset('Amazon_Beauty', verbose)
    toys_results = analyze_dataset('Amazon_Toys_and_Games', verbose)
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    # 对比关键指标
    print("\n关键指标对比:")
    print(f"{'Metric':<40} {'Beauty':>12} {'Toys':>12} {'Diff':>12}")
    print("-" * 78)
    
    comparisons = []
    
    # LLM inter-item similarity
    if 'llm' in beauty_results and 'llm' in toys_results:
        b_sim = beauty_results['llm']['inter_item_sim_mean']
        t_sim = toys_results['llm']['inter_item_sim_mean']
        diff = t_sim - b_sim
        print(f"{'LLM inter-item sim':<40} {b_sim:>12.4f} {t_sim:>12.4f} {diff:>+12.4f}")
        comparisons.append(('llm_inter_item_sim', b_sim, t_sim))
    
    # View diversity
    if 'view_diversity' in beauty_results and 'view_diversity' in toys_results:
        b_div = beauty_results['view_diversity']['diversity_score']
        t_div = toys_results['view_diversity']['diversity_score']
        diff = t_div - b_div
        print(f"{'View diversity score':<40} {b_div:>12.4f} {t_div:>12.4f} {diff:>+12.4f}")
        comparisons.append(('view_diversity', b_div, t_div))
    
    # High similarity ratio
    if 'llm' in beauty_results and 'llm' in toys_results:
        b_high = beauty_results['llm']['high_sim_ratio']
        t_high = toys_results['llm']['high_sim_ratio']
        diff = t_high - b_high
        print(f"{'High sim ratio (>0.9)':<40} {b_high:>11.2%} {t_high:>11.2%} {diff:>+11.2%}")
        comparisons.append(('high_sim_ratio', b_high, t_high))
    
    # 诊断建议
    print("\n" + "="*60)
    print("诊断建议")
    print("="*60)
    
    issues = []
    
    if 'llm' in beauty_results and 'llm' in toys_results:
        b_sim = beauty_results['llm']['inter_item_sim_mean']
        t_sim = toys_results['llm']['inter_item_sim_mean']
        
        if t_sim > b_sim + 0.02:
            issues.append(f"""
⚠️ Toys的item间相似度更高 ({t_sim:.4f} vs Beauty {b_sim:.4f})
   → 建议增加temperature: 0.07 → 0.10~0.12
   → 建议增加alignment_weight: 0.05 → 0.08~0.10
   → 原因: 需要更强的对比学习来区分相似items
""")
    
    if 'view_diversity' in beauty_results and 'view_diversity' in toys_results:
        b_div = beauty_results['view_diversity']['diversity_score']
        t_div = toys_results['view_diversity']['diversity_score']
        
        if t_div < b_div - 0.02:
            issues.append(f"""
⚠️ Toys的视图多样性较低 ({t_div:.4f} vs Beauty {b_div:.4f})
   → 建议增加text_view_senet_ratio: 4 → 8
   → 建议增加text_weight: 0.8 → 0.9
   → 原因: 视图信息冗余，需要更强的特征增强
""")
        elif t_div > b_div + 0.02:
            issues.append(f"""
✓ Toys的视图多样性实际更高 ({t_div:.4f} vs Beauty {b_div:.4f})
   → 问题可能在fusion策略而非embedding质量
   → 建议检查gate正则化和初始化
""")
    
    if not issues:
        print("\n✓ 两个数据集的embedding特性相似，问题可能在训练策略")
        print("  建议:")
        print("  - 增加phase_b_epochs: 40 → 60")
        print("  - 降低backbone_lr_scale: 0.1 → 0.05")
    else:
        for issue in issues:
            print(issue)
    
    return {
        'beauty': beauty_results,
        'toys': toys_results,
        'comparisons': comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description="Quick Embedding Stats")
    parser.add_argument('--dataset', type=str, default=None,
                       help='分析单个数据集 (Amazon_Beauty 或 Amazon_Toys_and_Games)')
    parser.add_argument('--compare', action='store_true', default=True,
                       help='对比两个数据集 (默认)')
    parser.add_argument('--output', type=str, default=None,
                       help='保存JSON结果')
    parser.add_argument('--quiet', action='store_true',
                       help='仅输出关键信息')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if args.dataset:
        results = analyze_dataset(args.dataset, verbose)
    else:
        results = compare_datasets(verbose)
    
    if args.output:
        # 转换numpy类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        results = convert_numpy(results)
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ 结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
