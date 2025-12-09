#!/usr/bin/env python3
"""
Dataset Comparison Tool

Compares characteristics between Beauty and Toys datasets to understand
why multi-view works differently on them.

Usage:
    python tools/compare_datasets.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


def load_dataset_info(dataset_name, base_dir="/home/charlie/project/RecBole/dataset"):
    """Load dataset files and extract key statistics."""
    dataset_path = Path(base_dir) / dataset_name
    
    info = {
        'name': dataset_name,
        'interactions': None,
        'items': None,
        'users': None,
    }
    
    # Load interaction file
    inter_file = dataset_path / f"{dataset_name}.inter"
    if inter_file.exists():
        try:
            df = pd.read_csv(inter_file, sep='\t')
            info['interactions'] = {
                'total': len(df),
                'unique_users': df['user_id'].nunique(),
                'unique_items': df['item_id'].nunique(),
                'density': len(df) / (df['user_id'].nunique() * df['item_id'].nunique()),
                'avg_interactions_per_user': len(df) / df['user_id'].nunique(),
                'avg_interactions_per_item': len(df) / df['item_id'].nunique(),
            }
            print(f"✓ Loaded interactions for {dataset_name}: {len(df)} records")
        except Exception as e:
            print(f"✗ Error loading interactions: {e}")
    
    # Load item file (text descriptions)
    item_file = dataset_path / f"{dataset_name}.item"
    if item_file.exists():
        try:
            df_item = pd.read_csv(item_file, sep='\t')
            
            if 'title' in df_item.columns:
                # Analyze text lengths
                title_lengths = df_item['title'].fillna('').str.len()
                info['items'] = {
                    'total': len(df_item),
                    'title_length_mean': float(title_lengths.mean()),
                    'title_length_std': float(title_lengths.std()),
                    'title_length_median': float(title_lengths.median()),
                    'title_length_min': int(title_lengths.min()),
                    'title_length_max': int(title_lengths.max()),
                }
                
                # Word count analysis
                word_counts = df_item['title'].fillna('').str.split().str.len()
                info['items']['words_per_title_mean'] = float(word_counts.mean())
                info['items']['words_per_title_std'] = float(word_counts.std())
                
                print(f"✓ Loaded items for {dataset_name}: {len(df_item)} items")
        except Exception as e:
            print(f"✗ Error loading items: {e}")
    
    return info


def compare_embedding_distributions(dataset1, dataset2, 
                                    base_dir="/home/charlie/project/RecBole/dataset"):
    """
    Compare embedding distributions between two datasets.
    Focuses on how well multi-view embeddings might work.
    """
    results = {
        'dataset1': dataset1,
        'dataset2': dataset2,
        'comparison': {},
    }
    
    for dataset in [dataset1, dataset2]:
        dataset_path = Path(base_dir) / dataset
        
        # Load LLM embedding (qwen3 base)
        llm_path = dataset_path / "item_text_emb.qwen3.base.npy"
        if llm_path.exists():
            llm_emb = np.load(llm_path)
            
            # Convert float16 to float32 for numerical stability
            if llm_emb.dtype == np.float16:
                llm_emb = llm_emb.astype(np.float32)
            
            # Compute inter-item similarities
            from sklearn.metrics.pairwise import cosine_similarity
            n_samples = min(len(llm_emb), 500)
            sampled = llm_emb[:n_samples]
            cos_sim = cosine_similarity(sampled)
            
            # Remove diagonal
            mask = ~np.eye(cos_sim.shape[0], dtype=bool)
            similarities = cos_sim[mask]
            
            # Filter out nan/inf values
            similarities = similarities[~np.isnan(similarities) & ~np.isinf(similarities)]
            
            results['comparison'][dataset] = {
                'llm_embedding_shape': llm_emb.shape,
                'llm_embedding_dtype': str(llm_emb.dtype),
                'inter_item_similarity_mean': float(similarities.mean()) if len(similarities) > 0 else None,
                'inter_item_similarity_std': float(similarities.std()) if len(similarities) > 0 else None,
                'inter_item_similarity_median': float(np.median(similarities)) if len(similarities) > 0 else None,
                'inter_item_similarity_q25': float(np.percentile(similarities, 25)) if len(similarities) > 0 else None,
                'inter_item_similarity_q75': float(np.percentile(similarities, 75)) if len(similarities) > 0 else None,
            }
            
            # Check if items are too similar (might hurt multi-view diversity)
            high_sim_threshold = 0.9
            high_sim_ratio = np.mean(similarities > high_sim_threshold)
            results['comparison'][dataset]['high_similarity_ratio'] = float(high_sim_ratio)
            
            print(f"\n{dataset}:")
            print(f"  Mean inter-item cosine: {similarities.mean():.4f}±{similarities.std():.4f}")
            print(f"  % pairs with similarity > {high_sim_threshold}: {high_sim_ratio:.2%}")
    
    return results


def main():
    print("\n" + "="*80)
    print("DATASET COMPARISON: Beauty vs Toys")
    print("="*80 + "\n")
    
    # Compare basic dataset statistics
    print("1. Basic Dataset Statistics")
    print("-" * 80)
    
    beauty_info = load_dataset_info('Amazon_Beauty')
    toys_info = load_dataset_info('Amazon_Toys_and_Games')
    
    results = {
        'beauty': beauty_info,
        'toys': toys_info,
    }
    
    # Print comparison
    if beauty_info['interactions'] and toys_info['interactions']:
        print("\nInteraction Statistics:")
        print(f"{'Metric':<40} {'Beauty':>15} {'Toys':>15}")
        print("-" * 72)
        
        metrics = [
            ('Total interactions', 'total'),
            ('Unique users', 'unique_users'),
            ('Unique items', 'unique_items'),
            ('Density', 'density'),
            ('Avg interactions/user', 'avg_interactions_per_user'),
            ('Avg interactions/item', 'avg_interactions_per_item'),
        ]
        
        for label, key in metrics:
            b_val = beauty_info['interactions'][key]
            t_val = toys_info['interactions'][key]
            
            if isinstance(b_val, float):
                print(f"{label:<40} {b_val:>15.4f} {t_val:>15.4f}")
            else:
                print(f"{label:<40} {b_val:>15} {t_val:>15}")
    
    if beauty_info['items'] and toys_info['items']:
        print("\nItem Text Statistics:")
        print(f"{'Metric':<40} {'Beauty':>15} {'Toys':>15}")
        print("-" * 72)
        
        metrics = [
            ('Total items', 'total'),
            ('Avg title length (chars)', 'title_length_mean'),
            ('Std title length', 'title_length_std'),
            ('Median title length', 'title_length_median'),
            ('Avg words per title', 'words_per_title_mean'),
            ('Std words per title', 'words_per_title_std'),
        ]
        
        for label, key in metrics:
            b_val = beauty_info['items'][key]
            t_val = toys_info['items'][key]
            
            if isinstance(b_val, float):
                print(f"{label:<40} {b_val:>15.2f} {t_val:>15.2f}")
            else:
                print(f"{label:<40} {b_val:>15} {t_val:>15}")
    
    # Compare embedding distributions
    print("\n" + "="*80)
    print("2. Embedding Distribution Comparison")
    print("="*80)
    
    emb_comparison = compare_embedding_distributions('Amazon_Beauty', 'Amazon_Toys_and_Games')
    results['embedding_comparison'] = emb_comparison
    
    # Analysis and recommendations
    print("\n" + "="*80)
    print("3. Analysis & Insights")
    print("="*80 + "\n")
    
    if 'embedding_comparison' in results:
        beauty_sim = results['embedding_comparison']['comparison'].get('Amazon_Beauty', {})
        toys_sim = results['embedding_comparison']['comparison'].get('Amazon_Toys_and_Games', {})
        
        if beauty_sim and toys_sim:
            b_mean = beauty_sim['inter_item_similarity_mean']
            t_mean = toys_sim['inter_item_similarity_mean']
            
            print(f"Inter-item similarity comparison:")
            print(f"  Beauty: {b_mean:.4f}")
            print(f"  Toys:   {t_mean:.4f}")
            print(f"  Difference: {abs(b_mean - t_mean):.4f}")
            
            if t_mean > b_mean + 0.05:
                print("\n⚠ WARNING: Toys items are significantly more similar to each other!")
                print("  → This may reduce the benefit of multi-view learning")
                print("  → Less diverse views = less complementary information")
                print("  → Recommendation: Try increasing view diversity or adjusting fusion weights")
            
            b_high_sim = beauty_sim.get('high_similarity_ratio', 0)
            t_high_sim = toys_sim.get('high_similarity_ratio', 0)
            
            if t_high_sim > b_high_sim * 1.5:
                print(f"\n⚠ WARNING: Toys has {t_high_sim:.1%} highly similar item pairs")
                print(f"  (vs {b_high_sim:.1%} in Beauty)")
                print("  → Multi-view may struggle to find discriminative features")
                print("  → Recommendation: Consider stronger regularization or different prompts")
    
    # Save results
    output_file = "dataset_comparison_beauty_vs_toys.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print(f"✓ Comparison complete! Results saved to: {output_file}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
