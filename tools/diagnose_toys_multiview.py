#!/usr/bin/env python3
"""
Toys Multi-View Diagnostic Tool

针对Toys数据集上multi-view表现不如baseline的问题进行深度诊断。

Usage:
    python tools/diagnose_toys_multiview.py
    python tools/diagnose_toys_multiview.py --full  # 完整分析（较慢）
    
需要在服务器上运行，访问embedding文件。
"""

import argparse
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

# 数据目录
BASE_DIR = Path("/home/charlie/project/RecBole/dataset")
BEAUTY_DIR = BASE_DIR / "Amazon_Beauty"
TOYS_DIR = BASE_DIR / "Amazon_Toys_and_Games"


def load_embeddings(dataset_dir, view_names=['identity', 'function', 'audience', 'category']):
    """加载所有embedding"""
    embeddings = {}
    
    # Base (TF-IDF)
    base_path = dataset_dir / "item_text_emb.base.npy"
    if base_path.exists():
        embeddings['base'] = np.load(base_path)
        print(f"  ✓ Base: {embeddings['base'].shape}")
    
    # Single LLM
    llm_path = dataset_dir / "item_text_emb.qwen3.npy"
    if llm_path.exists():
        embeddings['llm'] = np.load(llm_path)
        print(f"  ✓ LLM: {embeddings['llm'].shape}")
    
    # Multi-view
    mv_dir = dataset_dir / "qwen3_4views"
    if mv_dir.exists():
        embeddings['views'] = {}
        for view in view_names:
            view_path = mv_dir / f"item_text_emb.qwen3.{view}.npy"
            if view_path.exists():
                embeddings['views'][view] = np.load(view_path)
                print(f"  ✓ View [{view}]: {embeddings['views'][view].shape}")
    
    return embeddings


def compute_inter_item_similarity(emb, n_samples=500):
    """计算item间相似度分布"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    n_samples = min(len(emb), n_samples)
    sampled = emb[:n_samples]
    cos_sim = cosine_similarity(sampled)
    
    # 移除对角线
    mask = ~np.eye(cos_sim.shape[0], dtype=bool)
    similarities = cos_sim[mask]
    
    return {
        'mean': float(np.mean(similarities)),
        'std': float(np.std(similarities)),
        'median': float(np.median(similarities)),
        'p25': float(np.percentile(similarities, 25)),
        'p75': float(np.percentile(similarities, 75)),
        'p95': float(np.percentile(similarities, 95)),
        'high_sim_ratio': float(np.mean(similarities > 0.9)),  # >0.9的比例
    }


def compute_view_diversity(views_dict, n_samples=500):
    """计算多视图间的多样性"""
    from scipy.spatial.distance import cosine
    
    view_names = list(views_dict.keys())
    n_views = len(view_names)
    
    pairwise_cosine = {}
    
    for i in range(n_views):
        for j in range(i + 1, n_views):
            view_i = views_dict[view_names[i]]
            view_j = views_dict[view_names[j]]
            
            n_samples = min(len(view_i), n_samples)
            
            # 计算每个item在两个view上的cosine相似度
            cos_sims = []
            for k in range(n_samples):
                sim = 1 - cosine(view_i[k], view_j[k])
                cos_sims.append(sim)
            
            pair_name = f"{view_names[i]}_vs_{view_names[j]}"
            pairwise_cosine[pair_name] = {
                'mean': float(np.mean(cos_sims)),
                'std': float(np.std(cos_sims)),
            }
    
    # 计算整体多样性分数
    all_sims = [v['mean'] for v in pairwise_cosine.values()]
    diversity_score = 1 - np.mean(all_sims)  # 越高越好
    
    return {
        'pairwise': pairwise_cosine,
        'diversity_score': float(diversity_score),
        'avg_inter_view_sim': float(np.mean(all_sims)),
    }


def compute_view_vs_llm(views_dict, llm_emb, n_samples=500):
    """比较每个view与single LLM的相似度"""
    from scipy.spatial.distance import cosine
    
    results = {}
    n_samples = min(len(llm_emb), n_samples)
    
    for view_name, view_emb in views_dict.items():
        cos_sims = []
        for k in range(n_samples):
            sim = 1 - cosine(view_emb[k], llm_emb[k])
            cos_sims.append(sim)
        
        results[view_name] = {
            'mean': float(np.mean(cos_sims)),
            'std': float(np.std(cos_sims)),
        }
    
    return results


def compute_effective_rank(emb, n_samples=1000):
    """计算有效秩（Effective Rank）衡量embedding的内在维度"""
    from numpy.linalg import svd
    
    n_samples = min(len(emb), n_samples)
    sampled = emb[:n_samples]
    
    # 中心化
    centered = sampled - sampled.mean(axis=0)
    
    try:
        U, S, Vt = svd(centered, full_matrices=False)
        
        # 有效秩 = exp(Shannon entropy of normalized singular values)
        S_norm = S / S.sum()
        effective_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-12)))
        
        # Top-k奇异值占比
        energy_ratio = {
            'top_10': float(S[:10].sum() / S.sum()),
            'top_50': float(S[:50].sum() / S.sum()) if len(S) >= 50 else 1.0,
            'top_100': float(S[:100].sum() / S.sum()) if len(S) >= 100 else 1.0,
        }
        
        return {
            'effective_rank': float(effective_rank),
            'total_dims': int(len(S)),
            'energy_ratio': energy_ratio,
            'singular_decay': float(S[10] / S[0]) if len(S) > 10 else 0.0,
        }
    except Exception as e:
        print(f"Warning: SVD failed: {e}")
        return None


def analyze_dataset(dataset_name, dataset_dir, full=False):
    """分析单个数据集"""
    print(f"\n{'='*80}")
    print(f"分析数据集: {dataset_name}")
    print(f"{'='*80}")
    
    embeddings = load_embeddings(dataset_dir)
    
    results = {
        'dataset': dataset_name,
        'inter_item_similarity': {},
        'view_diversity': None,
        'view_vs_llm': None,
        'effective_rank': {},
    }
    
    # 1. Inter-item similarity
    print("\n[1] Inter-item 相似度分析...")
    
    if 'llm' in embeddings:
        llm_sim = compute_inter_item_similarity(embeddings['llm'])
        results['inter_item_similarity']['llm'] = llm_sim
        print(f"  LLM: mean={llm_sim['mean']:.4f}, std={llm_sim['std']:.4f}, high_sim={llm_sim['high_sim_ratio']:.2%}")
    
    if 'views' in embeddings:
        for view_name, view_emb in embeddings['views'].items():
            view_sim = compute_inter_item_similarity(view_emb)
            results['inter_item_similarity'][view_name] = view_sim
            print(f"  {view_name}: mean={view_sim['mean']:.4f}, std={view_sim['std']:.4f}")
    
    # 2. View diversity
    if 'views' in embeddings and len(embeddings['views']) > 1:
        print("\n[2] Multi-View 多样性分析...")
        diversity = compute_view_diversity(embeddings['views'])
        results['view_diversity'] = diversity
        print(f"  多样性分数: {diversity['diversity_score']:.4f} (越高越好)")
        print(f"  平均视图间相似度: {diversity['avg_inter_view_sim']:.4f} (越低越好)")
        print(f"  各视图对相似度:")
        for pair, stats in diversity['pairwise'].items():
            print(f"    {pair}: {stats['mean']:.4f}")
    
    # 3. View vs Single LLM
    if 'views' in embeddings and 'llm' in embeddings:
        print("\n[3] 各View vs Single LLM 相似度...")
        view_vs_llm = compute_view_vs_llm(embeddings['views'], embeddings['llm'])
        results['view_vs_llm'] = view_vs_llm
        for view_name, stats in view_vs_llm.items():
            print(f"  {view_name} vs LLM: {stats['mean']:.4f}")
    
    # 4. Effective rank (if full analysis)
    if full:
        print("\n[4] Effective Rank 分析...")
        
        if 'llm' in embeddings:
            rank_info = compute_effective_rank(embeddings['llm'])
            if rank_info:
                results['effective_rank']['llm'] = rank_info
                print(f"  LLM: effective_rank={rank_info['effective_rank']:.2f}, top10_energy={rank_info['energy_ratio']['top_10']:.2%}")
        
        if 'views' in embeddings:
            for view_name, view_emb in embeddings['views'].items():
                rank_info = compute_effective_rank(view_emb)
                if rank_info:
                    results['effective_rank'][view_name] = rank_info
                    print(f"  {view_name}: effective_rank={rank_info['effective_rank']:.2f}")
    
    return results


def compare_and_diagnose(beauty_results, toys_results):
    """对比分析并生成诊断建议"""
    print("\n" + "="*80)
    print("诊断结果与建议")
    print("="*80)
    
    issues = []
    suggestions = []
    
    # 1. 对比inter-item similarity
    if 'llm' in beauty_results['inter_item_similarity'] and 'llm' in toys_results['inter_item_similarity']:
        b_sim = beauty_results['inter_item_similarity']['llm']['mean']
        t_sim = toys_results['inter_item_similarity']['llm']['mean']
        
        print(f"\n[诊断1] Inter-item 相似度对比:")
        print(f"  Beauty LLM: {b_sim:.4f}")
        print(f"  Toys LLM:   {t_sim:.4f}")
        
        if t_sim > b_sim + 0.03:
            issues.append("Toys的item embedding之间相似度更高")
            suggestions.append({
                'issue': 'High inter-item similarity',
                'severity': 'HIGH',
                'reason': f'Toys items更相似 ({t_sim:.4f} vs Beauty {b_sim:.4f})',
                'actions': [
                    '增加temperature (0.07 → 0.10-0.15) 使对比学习关注更难区分的样本',
                    '增加alignment_weight (0.05 → 0.08-0.10)',
                    '考虑更长的训练轮次',
                ],
                'config_changes': {
                    'temperature': 0.12,
                    'alignment_weight': 0.08,
                    'phase_b_epochs': 60,
                }
            })
            print("  ⚠️ Toys items更相似，multi-view难以学到区分性特征")
    
    # 2. 对比view diversity
    if beauty_results['view_diversity'] and toys_results['view_diversity']:
        b_div = beauty_results['view_diversity']['diversity_score']
        t_div = toys_results['view_diversity']['diversity_score']
        
        print(f"\n[诊断2] Multi-View 多样性对比:")
        print(f"  Beauty: {b_div:.4f}")
        print(f"  Toys:   {t_div:.4f}")
        
        if t_div < b_div - 0.02:
            issues.append("Toys的多视图多样性较低")
            suggestions.append({
                'issue': 'Low view diversity',
                'severity': 'HIGH',
                'reason': f'Toys views多样性更低 ({t_div:.4f} vs Beauty {b_div:.4f})',
                'actions': [
                    '增加text_view_senet_ratio (4 → 8) 增强视图特征',
                    '增加text_weight (0.8 → 0.9) 给text分支更大权重',
                    '考虑重新生成embedding用更差异化的prompt',
                ],
                'config_changes': {
                    'text_view_senet_ratio': 8,
                    'text_weight': 0.9,
                }
            })
            print("  ⚠️ Toys的各视图过于相似，信息冗余")
        elif t_div > b_div + 0.02:
            print("  ✓ Toys的视图多样性实际更好，问题可能在融合策略")
            suggestions.append({
                'issue': 'Good diversity but poor fusion',
                'severity': 'MEDIUM',
                'reason': '视图多样性OK，但融合效果不好',
                'actions': [
                    '检查gate初始化和正则化',
                    '尝试不同的fusion策略 (concat vs weighted sum)',
                ],
                'config_changes': {
                    'text_gate_init': 0.3,  # 降低初始gate值
                    'text_gate_reg_l2': 0.08,
                }
            })
    
    # 3. 检查view vs llm相似度
    if beauty_results['view_vs_llm'] and toys_results['view_vs_llm']:
        print(f"\n[诊断3] View vs Single LLM 对比:")
        
        b_avg = np.mean([v['mean'] for v in beauty_results['view_vs_llm'].values()])
        t_avg = np.mean([v['mean'] for v in toys_results['view_vs_llm'].values()])
        
        print(f"  Beauty avg: {b_avg:.4f}")
        print(f"  Toys avg:   {t_avg:.4f}")
        
        if t_avg > b_avg + 0.05:
            issues.append("Toys的multi-view与single LLM过于相似")
            suggestions.append({
                'issue': 'Views too similar to single LLM',
                'severity': 'MEDIUM',
                'reason': f'Toys各view与single LLM更接近 ({t_avg:.4f} vs Beauty {b_avg:.4f})',
                'actions': [
                    'Multi-view可能没有提供比single LLM更多的信息',
                    '考虑使用不同的prompt策略',
                    '或者回退到单一LLM方案 + 更强的regularization',
                ],
                'config_changes': {}
            })
            print("  ⚠️ Toys的multi-view与single LLM差异不大")
    
    # 4. 训练策略建议
    print(f"\n[诊断4] 训练策略建议:")
    suggestions.append({
        'issue': 'Training strategy adjustment',
        'severity': 'MEDIUM',
        'reason': '基于toys数据特性的训练调整',
        'actions': [
            '增加phase_b轮次 (40 → 60)，因为toys的loss下降更慢',
            '考虑更激进的学习率调度 (backbone_lr_scale: 0.05)',
            '增加early stopping patience',
        ],
        'config_changes': {
            'phase_b_epochs': 60,
            'backbone_lr_scale': 0.05,
            'stopping_step': 30,
        }
    })
    
    return {
        'issues': issues,
        'suggestions': suggestions,
    }


def generate_tuning_configs(suggestions, base_config_path):
    """生成调优配置"""
    import yaml
    
    print("\n" + "="*80)
    print("生成调优配置")
    print("="*80)
    
    # 读取基础配置
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    variants = []
    
    # 生成单独变体
    for i, sugg in enumerate(suggestions):
        if sugg['config_changes']:
            variant_name = f"toys_tuning_v{i+1}"
            variant_config = base_config.copy()
            variant_config.update(sugg['config_changes'])
            variants.append({
                'name': variant_name,
                'reason': sugg['issue'],
                'changes': sugg['config_changes'],
                'config': variant_config,
            })
    
    # 生成组合变体
    combined_changes = {}
    for sugg in suggestions:
        combined_changes.update(sugg['config_changes'])
    
    if combined_changes:
        combined_config = base_config.copy()
        combined_config.update(combined_changes)
        variants.append({
            'name': 'toys_tuning_combined',
            'reason': 'All suggested changes combined',
            'changes': combined_changes,
            'config': combined_config,
        })
    
    return variants


def main():
    parser = argparse.ArgumentParser(description="Toys Multi-View Diagnostic")
    parser.add_argument('--full', action='store_true', help='运行完整分析（包括SVD）')
    parser.add_argument('--output', type=str, default='toys_multiview_diagnosis.json', 
                       help='输出JSON文件')
    parser.add_argument('--gen_config', action='store_true', help='生成调优配置文件')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("TOYS MULTI-VIEW 诊断工具")
    print("="*80)
    print("\n目标: 分析为什么multi-view在Toys上表现不如tfidf+llm baseline")
    
    # 分析两个数据集
    beauty_results = analyze_dataset("Amazon_Beauty", BEAUTY_DIR, full=args.full)
    toys_results = analyze_dataset("Amazon_Toys_and_Games", TOYS_DIR, full=args.full)
    
    # 对比诊断
    diagnosis = compare_and_diagnose(beauty_results, toys_results)
    
    # 保存结果
    results = {
        'beauty': beauty_results,
        'toys': toys_results,
        'diagnosis': diagnosis,
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ 诊断结果已保存到: {args.output}")
    
    # 生成调优配置
    if args.gen_config:
        base_config = Path("/home/charlie/project/RecBole/sasrec_align_multi_view_toys.yaml")
        if base_config.exists():
            variants = generate_tuning_configs(diagnosis['suggestions'], base_config)
            
            output_dir = Path("config_toys_tuning")
            output_dir.mkdir(exist_ok=True)
            
            import yaml
            for variant in variants:
                variant_path = output_dir / f"{variant['name']}.yaml"
                with open(variant_path, 'w') as f:
                    f.write(f"# {variant['reason']}\n")
                    f.write(f"# Changes: {variant['changes']}\n\n")
                    yaml.dump(variant['config'], f, default_flow_style=False)
                print(f"  ✓ 生成: {variant_path}")
            
            print(f"\n✓ 配置文件生成在: {output_dir}/")
    
    # 打印最终建议
    print("\n" + "="*80)
    print("快速行动指南")
    print("="*80)
    print("""
基于诊断结果，建议按以下优先级尝试：

1. [立即尝试] 增加训练轮次和调整超参:
   --phase_b_epochs 60
   --temperature 0.10 或 0.12
   --alignment_weight 0.08

2. [立即尝试] 增加text分支权重:
   --phase_b_text_weight 0.9

3. [需要重跑embedding] 如果上述无效:
   - 检查toys的multi-view prompts是否足够差异化
   - 考虑为toys定制不同的prompt策略

4. [对照实验] 运行修复后的baseline对比:
   - 为toys的tfidf+llm baseline也添加SENet
   - 确保公平对比
""")


if __name__ == '__main__':
    main()
