#!/usr/bin/env python3
"""
Multi-View Quality Analysis Tool

Analyzes the quality and characteristics of multi-view embeddings to diagnose
why multi-view performs well on Beauty but not on Toys.

Usage:
    python tools/analyze_multiview_quality.py --dataset Amazon_Beauty
    python tools/analyze_multiview_quality.py --dataset Amazon_Toys_and_Games
"""

import argparse
import numpy as np
import os
from pathlib import Path
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr, spearmanr
import json


def load_embeddings(dataset_name, base_dir="/home/charlie/project/RecBole/dataset"):
    """Load base, LLM, and multi-view embeddings."""
    dataset_path = Path(base_dir) / dataset_name
    
    embeddings = {}
    
    # Load base TF-IDF embedding
    base_path = dataset_path / "item_text_emb.base.npy"
    if base_path.exists():
        emb = np.load(base_path)
        # Convert float16 to float32 to avoid overflow
        embeddings['base'] = emb.astype(np.float32) if emb.dtype == np.float16 else emb
        print(f"✓ Loaded base embedding: {embeddings['base'].shape} (dtype: {embeddings['base'].dtype})")
    else:
        print(f"✗ Base embedding not found: {base_path}")
    
    # Load single LLM embedding (qwen3 base)
    llm_path = dataset_path / "item_text_emb.qwen3.base.npy"
    if llm_path.exists():
        emb = np.load(llm_path)
        # Convert float16 to float32 to avoid overflow
        embeddings['llm'] = emb.astype(np.float32) if emb.dtype == np.float16 else emb
        print(f"✓ Loaded LLM embedding: {embeddings['llm'].shape} (dtype: {embeddings['llm'].dtype})")
    else:
        print(f"✗ LLM embedding not found: {llm_path}")
    
    # Load merged multi-view embedding (for comparison with single LLM)
    multiview_merged_path = dataset_path / "item_text_emb.qwen3.multiview.npy"
    if multiview_merged_path.exists():
        emb = np.load(multiview_merged_path)
        embeddings['multiview_merged'] = emb.astype(np.float32) if emb.dtype == np.float16 else emb
        print(f"✓ Loaded merged multi-view embedding: {embeddings['multiview_merged'].shape} (dtype: {embeddings['multiview_merged'].dtype})")
    else:
        print(f"ℹ Merged multi-view embedding not found: {multiview_merged_path}")
    
    # Load multi-view embeddings
    multiview_dir = dataset_path / "qwen3_4views"
    if multiview_dir.exists():
        # Load views.json to get view name mapping and prompt information
        views_json_path = multiview_dir / "views.json"
        view_mapping = {}
        prompt_info = {}
        
        if views_json_path.exists():
            with open(views_json_path, 'r') as f:
                views_config = json.load(f)
                # views.json can have various formats:
                # 1. {"prompts": [{"index": 0, "prompt": "...", "file": "view_0.npy"}, ...]}
                # 2. {"views": [{"id": 0, "name": "identity"}, ...]}
                # 3. {"0": "identity", "1": "function", ..., "num_items": 123}
                # 4. {"view_names": ["identity", "function", ...]}
                if isinstance(views_config, dict):
                    if "prompts" in views_config and isinstance(views_config["prompts"], list):
                        # Prompts list format (actual format used)
                        for prompt_item in views_config["prompts"]:
                            idx = prompt_item["index"]
                            # Extract a short name from the prompt
                            prompt_text = prompt_item["prompt"]
                            # Use first few words or a descriptive name
                            if "Identify" in prompt_text:
                                name = "identity"
                            elif "function" in prompt_text or "feature" in prompt_text:
                                name = "function"
                            elif "audience" in prompt_text or "user group" in prompt_text:
                                name = "audience"
                            elif "Categorize" in prompt_text or "context" in prompt_text:
                                name = "category"
                            else:
                                name = f"view_{idx}"
                            
                            view_mapping[idx] = name
                            prompt_info[name] = {
                                'prompt': prompt_text,
                                'file': prompt_item.get('file', f'view_{idx}.npy'),
                                'vector_dim': prompt_item.get('vector_dim', 64)
                            }
                    elif "views" in views_config and isinstance(views_config["views"], list):
                        # List format
                        for view_info in views_config["views"]:
                            view_mapping[view_info["id"]] = view_info["name"]
                    elif "view_names" in views_config:
                        # view_names list format
                        for idx, name in enumerate(views_config["view_names"]):
                            view_mapping[idx] = name
                    else:
                        # Direct mapping format - filter only numeric keys
                        for k, v in views_config.items():
                            try:
                                view_mapping[int(k)] = v
                            except (ValueError, TypeError):
                                # Skip non-numeric keys like 'num_items', 'config', etc.
                                pass
        
        if not view_mapping:
            # Default mapping if views.json not found or empty
            print(f"Warning: Could not parse views.json, using default view names")
            view_mapping = {0: 'view_0', 1: 'view_1', 2: 'view_2', 3: 'view_3'}
        else:
            print(f"✓ Parsed {len(view_mapping)} view names from views.json")
        
        embeddings['views'] = {}
        embeddings['view_prompts'] = prompt_info
        
        for view_idx in sorted(view_mapping.keys()):
            view_name = view_mapping[view_idx]
            view_path = multiview_dir / f"view_{view_idx}.npy"
            if view_path.exists():
                emb = np.load(view_path)
                # Convert float16 to float32 to avoid overflow
                embeddings['views'][view_name] = emb.astype(np.float32) if emb.dtype == np.float16 else emb
                
                # Show prompt info if available
                if view_name in prompt_info:
                    prompt_preview = prompt_info[view_name]['prompt'][:50] + '...' if len(prompt_info[view_name]['prompt']) > 50 else prompt_info[view_name]['prompt']
                    print(f"✓ Loaded view_{view_idx} ({view_name}): {embeddings['views'][view_name].shape} (dtype: {embeddings['views'][view_name].dtype})")
                    print(f"  Prompt: \"{prompt_preview}\"")
                else:
                    print(f"✓ Loaded view_{view_idx} ({view_name}): {embeddings['views'][view_name].shape} (dtype: {embeddings['views'][view_name].dtype})")
            else:
                print(f"✗ View {view_idx} not found: {view_path}")
    else:
        print(f"✗ Multi-view directory not found: {multiview_dir}")
    
    return embeddings


def analyze_embedding_statistics(emb, name):
    """Compute basic statistics for an embedding."""
    stats = {
        'name': name,
        'shape': emb.shape,
        'mean': float(np.mean(emb)),
        'std': float(np.std(emb)),
        'min': float(np.min(emb)),
        'max': float(np.max(emb)),
        'norm_mean': float(np.mean(np.linalg.norm(emb, axis=1))),
        'norm_std': float(np.std(np.linalg.norm(emb, axis=1))),
        'sparsity': float(np.mean(np.abs(emb) < 1e-6)),  # Percentage of near-zero values
    }
    
    # Compute variance along feature dimension
    feature_var = np.var(emb, axis=0)
    stats['feature_var_mean'] = float(np.mean(feature_var))
    stats['feature_var_std'] = float(np.std(feature_var))
    stats['feature_var_min'] = float(np.min(feature_var))
    stats['feature_var_max'] = float(np.max(feature_var))
    
    return stats


def compute_view_diversity(views_dict):
    """
    Compute diversity metrics between different views.
    Higher diversity = views capture different aspects.
    """
    view_names = list(views_dict.keys())
    n_views = len(view_names)
    
    diversity_metrics = {
        'pairwise_cosine_similarity': {},
        'pairwise_correlation': {},
        'mean_pairwise_cosine': 0.0,
        'mean_pairwise_correlation': 0.0,
    }
    
    # Compute pairwise similarities
    cosine_sims = []
    correlations = []
    
    for i in range(n_views):
        for j in range(i + 1, n_views):
            view_i = views_dict[view_names[i]]
            view_j = views_dict[view_names[j]]
            
            # Ensure float32
            if view_i.dtype == np.float16:
                view_i = view_i.astype(np.float32)
            if view_j.dtype == np.float16:
                view_j = view_j.astype(np.float32)
            
            # Average cosine similarity across all items (sample for speed)
            sample_size = min(len(view_i), 1000)
            cos_sims_per_item = []
            
            for k in range(sample_size):
                try:
                    sim = 1 - cosine(view_i[k], view_j[k])
                    if not np.isnan(sim) and not np.isinf(sim):
                        cos_sims_per_item.append(sim)
                except:
                    continue
            
            avg_cos_sim = np.mean(cos_sims_per_item) if cos_sims_per_item else 0.0
            
            # Correlation of flattened embeddings (sample for speed)
            try:
                flat_i = view_i.flatten()[:10000]
                flat_j = view_j.flatten()[:10000]
                corr, _ = pearsonr(flat_i, flat_j)
                if np.isnan(corr) or np.isinf(corr):
                    corr = 0.0
            except:
                corr = 0.0
            
            pair_name = f"{view_names[i]}_vs_{view_names[j]}"
            diversity_metrics['pairwise_cosine_similarity'][pair_name] = float(avg_cos_sim)
            diversity_metrics['pairwise_correlation'][pair_name] = float(corr)
            
            cosine_sims.append(avg_cos_sim)
            correlations.append(corr)
    
    if cosine_sims:
        diversity_metrics['mean_pairwise_cosine'] = float(np.mean(cosine_sims))
        diversity_metrics['mean_pairwise_correlation'] = float(np.mean(correlations))
        # Lower similarity = higher diversity (better for multi-view)
        diversity_metrics['diversity_score'] = float(1 - np.mean(cosine_sims))
    else:
        diversity_metrics['mean_pairwise_cosine'] = None
        diversity_metrics['mean_pairwise_correlation'] = None
        diversity_metrics['diversity_score'] = None
    
    return diversity_metrics


def compute_view_vs_single_llm(views_dict, llm_emb, multiview_merged_emb=None):
    """
    Compare multi-view embeddings with single LLM embedding.
    If views are too similar to single LLM, multi-view may not add value.
    
    Note: Individual views may have different dimensions than single LLM,
    so we primarily compare the merged multi-view embedding if available.
    """
    metrics = {}
    
    # If we have merged multi-view embedding, compare it with single LLM
    if multiview_merged_emb is not None and multiview_merged_emb.shape[1] == llm_emb.shape[1]:
        sample_size = min(len(multiview_merged_emb), 1000)
        
        cos_sims = []
        for k in range(sample_size):
            try:
                sim = 1 - cosine(multiview_merged_emb[k], llm_emb[k])
                if not np.isnan(sim) and not np.isinf(sim):
                    cos_sims.append(sim)
            except:
                continue
        
        if cos_sims:
            metrics['multiview_merged'] = {
                'mean_cosine_to_llm': float(np.mean(cos_sims)),
                'std_cosine_to_llm': float(np.std(cos_sims)),
                'comparison': 'Merged multi-view (all 4 views combined) vs Single LLM'
            }
    
    # For individual views, we can't directly compare with single LLM if dimensions differ
    # But we can report their dimensionality
    for view_name, view_emb in views_dict.items():
        metrics[view_name] = {
            'dimension': view_emb.shape[1],
            'note': f'Individual view dimension ({view_emb.shape[1]}D) differs from single LLM ({llm_emb.shape[1]}D)',
        }
    
    return metrics


def analyze_feature_space(emb, name, n_samples=1000):
    """
    Analyze the feature space structure.
    """
    n_samples = min(len(emb), n_samples)
    sampled = emb[:n_samples]
    
    # Ensure float32 for numerical stability
    if sampled.dtype == np.float16:
        sampled = sampled.astype(np.float32)
    
    # Compute pairwise distances
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim_matrix = cosine_similarity(sampled)
    
    # Remove diagonal
    mask = ~np.eye(cos_sim_matrix.shape[0], dtype=bool)
    cos_sims = cos_sim_matrix[mask]
    
    # Filter out nan/inf values
    cos_sims = cos_sims[~np.isnan(cos_sims) & ~np.isinf(cos_sims)]
    
    metrics = {
        'name': name,
        'mean_pairwise_cosine': float(np.mean(cos_sims)) if len(cos_sims) > 0 else None,
        'std_pairwise_cosine': float(np.std(cos_sims)) if len(cos_sims) > 0 else None,
        'min_pairwise_cosine': float(np.min(cos_sims)) if len(cos_sims) > 0 else None,
        'max_pairwise_cosine': float(np.max(cos_sims)) if len(cos_sims) > 0 else None,
        'median_pairwise_cosine': float(np.median(cos_sims)) if len(cos_sims) > 0 else None,
    }
    
    # Compute intrinsic dimensionality (effective rank)
    try:
        from numpy.linalg import svd
        # Center the data
        centered = sampled - sampled.mean(axis=0)
        
        # Ensure float32
        if centered.dtype == np.float16:
            centered = centered.astype(np.float32)
        
        U, S, Vt = svd(centered, full_matrices=False)
        
        # Filter out very small singular values
        S = S[S > 1e-10]
        
        # Effective rank (participation ratio)
        if len(S) > 0:
            S_normalized = S / S.sum()
            effective_rank = np.exp(-np.sum(S_normalized * np.log(S_normalized + 1e-12)))
            
            metrics['effective_rank'] = float(effective_rank)
            metrics['top_10_singular_values'] = S[:10].tolist()
            metrics['singular_value_decay'] = float(S[10] / S[0]) if len(S) > 10 else 0.0
        else:
            metrics['effective_rank'] = None
    except Exception as e:
        print(f"Warning: Could not compute SVD for {name}: {e}")
        metrics['effective_rank'] = None
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-view embedding quality")
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['Amazon_Beauty', 'Amazon_Toys_and_Games'],
                       help='Dataset name')
    parser.add_argument('--base_dir', type=str, 
                       default='/home/charlie/project/RecBole/dataset',
                       help='Base directory for datasets')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (default: auto-generated)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Multi-View Quality Analysis: {args.dataset}")
    print(f"{'='*80}\n")
    
    # Load embeddings
    print("Loading embeddings...")
    embeddings = load_embeddings(args.dataset, args.base_dir)
    
    if not embeddings:
        print("Error: No embeddings loaded!")
        return
    
    results = {
        'dataset': args.dataset,
        'statistics': {},
        'feature_space': {},
        'diversity': None,
        'view_vs_llm': None,
        'view_prompts': embeddings.get('view_prompts', {}),
    }
    
    # 1. Basic statistics for all embeddings
    print("\n" + "="*80)
    print("1. BASIC STATISTICS")
    print("="*80)
    
    if 'base' in embeddings:
        stats = analyze_embedding_statistics(embeddings['base'], 'base')
        results['statistics']['base'] = stats
        print(f"\nBase (TF-IDF):")
        print(f"  Shape: {stats['shape']}")
        print(f"  Mean±Std: {stats['mean']:.4f}±{stats['std']:.4f}")
        print(f"  Norm: {stats['norm_mean']:.4f}±{stats['norm_std']:.4f}")
        print(f"  Sparsity: {stats['sparsity']:.2%}")
        print(f"  Feature variance: {stats['feature_var_mean']:.4f}±{stats['feature_var_std']:.4f}")
    
    if 'llm' in embeddings:
        stats = analyze_embedding_statistics(embeddings['llm'], 'llm')
        results['statistics']['llm'] = stats
        print(f"\nSingle LLM:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Mean±Std: {stats['mean']:.4f}±{stats['std']:.4f}")
        print(f"  Norm: {stats['norm_mean']:.4f}±{stats['norm_std']:.4f}")
        print(f"  Sparsity: {stats['sparsity']:.2%}")
        print(f"  Feature variance: {stats['feature_var_mean']:.4f}±{stats['feature_var_std']:.4f}")
    
    if 'views' in embeddings:
        results['statistics']['views'] = {}
        for view_name, view_emb in embeddings['views'].items():
            stats = analyze_embedding_statistics(view_emb, view_name)
            results['statistics']['views'][view_name] = stats
            print(f"\nView [{view_name}]:")
            print(f"  Shape: {stats['shape']}")
            print(f"  Mean±Std: {stats['mean']:.4f}±{stats['std']:.4f}")
            print(f"  Norm: {stats['norm_mean']:.4f}±{stats['norm_std']:.4f}")
            print(f"  Sparsity: {stats['sparsity']:.2%}")
            print(f"  Feature variance: {stats['feature_var_mean']:.4f}±{stats['feature_var_std']:.4f}")
    
    # 2. Feature space analysis
    print("\n" + "="*80)
    print("2. FEATURE SPACE STRUCTURE")
    print("="*80)
    
    if 'llm' in embeddings:
        fs_metrics = analyze_feature_space(embeddings['llm'], 'llm')
        results['feature_space']['llm'] = fs_metrics
        print(f"\nSingle LLM:")
        print(f"  Mean pairwise cosine: {fs_metrics['mean_pairwise_cosine']:.4f}±{fs_metrics['std_pairwise_cosine']:.4f}")
        print(f"  Effective rank: {fs_metrics.get('effective_rank', 'N/A')}")
        if fs_metrics.get('effective_rank'):
            print(f"  Singular value decay: {fs_metrics['singular_value_decay']:.4f}")
    
    if 'views' in embeddings:
        results['feature_space']['views'] = {}
        for view_name, view_emb in embeddings['views'].items():
            fs_metrics = analyze_feature_space(view_emb, view_name)
            results['feature_space']['views'][view_name] = fs_metrics
            print(f"\nView [{view_name}]:")
            print(f"  Mean pairwise cosine: {fs_metrics['mean_pairwise_cosine']:.4f}±{fs_metrics['std_pairwise_cosine']:.4f}")
            print(f"  Effective rank: {fs_metrics.get('effective_rank', 'N/A')}")
            if fs_metrics.get('effective_rank'):
                print(f"  Singular value decay: {fs_metrics['singular_value_decay']:.4f}")
    
    # 3. View diversity analysis
    if 'views' in embeddings and len(embeddings['views']) > 1:
        print("\n" + "="*80)
        print("3. MULTI-VIEW DIVERSITY ANALYSIS")
        print("="*80)
        
        diversity = compute_view_diversity(embeddings['views'])
        results['diversity'] = diversity
        
        print(f"\nPairwise Cosine Similarities (lower = more diverse):")
        for pair, sim in diversity['pairwise_cosine_similarity'].items():
            print(f"  {pair}: {sim:.4f}")
        
        print(f"\nOverall Metrics:")
        print(f"  Mean pairwise cosine: {diversity['mean_pairwise_cosine']:.4f}")
        print(f"  Mean pairwise correlation: {diversity['mean_pairwise_correlation']:.4f}")
        print(f"  Diversity score: {diversity['diversity_score']:.4f} (higher is better)")
    
    # 4. View vs Single LLM comparison
    if 'views' in embeddings and 'llm' in embeddings:
        print("\n" + "="*80)
        print("4. VIEWS vs SINGLE LLM COMPARISON")
        print("="*80)
        
        multiview_merged = embeddings.get('multiview_merged', None)
        view_vs_llm = compute_view_vs_single_llm(embeddings['views'], embeddings['llm'], multiview_merged)
        results['view_vs_llm'] = view_vs_llm
        
        print("\nComparing multi-view with single LLM embedding:")
        
        if 'multiview_merged' in view_vs_llm and 'mean_cosine_to_llm' in view_vs_llm['multiview_merged']:
            metrics = view_vs_llm['multiview_merged']
            print(f"\n  Merged Multi-View vs Single LLM:")
            print(f"    Cosine similarity: {metrics['mean_cosine_to_llm']:.4f}±{metrics['std_cosine_to_llm']:.4f}")
            print(f"    (High similarity = multi-view is redundant with single LLM)")
        
        print(f"\n  Individual view dimensions:")
        for view_name, metrics in view_vs_llm.items():
            if view_name != 'multiview_merged' and 'dimension' in metrics:
                print(f"    {view_name}: {metrics['dimension']}D")
        
        if 'multiview_merged' not in view_vs_llm or 'mean_cosine_to_llm' not in view_vs_llm.get('multiview_merged', {}):
            print(f"\n  ℹ Note: Individual views have different dimensions than single LLM")
            print(f"    Cannot directly compare. Need merged multi-view embedding for comparison.")
    
    # Save results
    output_file = args.output
    if output_file is None:
        output_file = f"multiview_analysis_{args.dataset.replace('Amazon_', '').lower()}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print(f"✓ Analysis complete! Results saved to: {output_file}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
