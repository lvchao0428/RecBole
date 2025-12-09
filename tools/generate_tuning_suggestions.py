#!/usr/bin/env python3
"""
Tuning Suggestions Generator

Based on diagnostic analysis, generates concrete hyperparameter tuning suggestions
to improve multi-view performance on Toys dataset.

Usage:
    python tools/generate_tuning_suggestions.py \
      --beauty_analysis multiview_diagnostics_*/multiview_quality_beauty.json \
      --toys_analysis multiview_diagnostics_*/multiview_quality_toys.json
"""

import argparse
import json
from pathlib import Path
import yaml


def load_analysis(file_path):
    """Load analysis JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def compare_diversity(beauty_data, toys_data):
    """Compare view diversity between datasets."""
    suggestions = []
    
    beauty_div = beauty_data.get('diversity', {})
    toys_div = toys_data.get('diversity', {})
    
    if not beauty_div or not toys_div:
        return suggestions
    
    beauty_score = beauty_div.get('diversity_score', 0)
    toys_score = toys_div.get('diversity_score', 0)
    
    print(f"\nView Diversity Comparison:")
    print(f"  Beauty: {beauty_score:.4f}")
    print(f"  Toys:   {toys_score:.4f}")
    print(f"  Difference: {beauty_score - toys_score:.4f}")
    
    if toys_score < beauty_score * 0.8:
        suggestions.append({
            'issue': 'Low view diversity on Toys',
            'severity': 'HIGH',
            'recommendation': 'Views are too similar on Toys dataset',
            'actions': [
                'Try different prompting strategies to increase view diversity',
                'Consider regenerating embeddings with more diverse prompts',
                'Increase per-view SENet ratio from 4 to 8 for stronger feature enhancement',
            ],
            'config_changes': {
                'text_view_senet_ratio': 8,
            }
        })
    
    # Check pairwise similarities
    beauty_pairs = beauty_div.get('pairwise_cosine_similarity', {})
    toys_pairs = toys_div.get('pairwise_cosine_similarity', {})
    
    if beauty_pairs and toys_pairs:
        beauty_avg = sum(beauty_pairs.values()) / len(beauty_pairs)
        toys_avg = sum(toys_pairs.values()) / len(toys_pairs)
        
        if toys_avg > beauty_avg + 0.1:
            suggestions.append({
                'issue': 'High inter-view similarity on Toys',
                'severity': 'MEDIUM',
                'recommendation': f'Average pairwise view similarity: Toys={toys_avg:.3f} vs Beauty={beauty_avg:.3f}',
                'actions': [
                    'Use learnable per-view weights to emphasize diverse views',
                    'Add orthogonality regularization to view projections',
                ],
                'config_changes': {
                    'use_view_orthogonal_reg': True,
                    'view_orthogonal_weight': 0.01,
                }
            })
    
    return suggestions


def compare_feature_space(beauty_data, toys_data):
    """Compare feature space characteristics."""
    suggestions = []
    
    beauty_fs = beauty_data.get('feature_space', {})
    toys_fs = toys_data.get('feature_space', {})
    
    # Compare effective rank
    beauty_llm_rank = beauty_fs.get('llm', {}).get('effective_rank')
    toys_llm_rank = toys_fs.get('llm', {}).get('effective_rank')
    
    if beauty_llm_rank and toys_llm_rank:
        print(f"\nEffective Rank (LLM embeddings):")
        print(f"  Beauty: {beauty_llm_rank:.2f}")
        print(f"  Toys:   {toys_llm_rank:.2f}")
        
        if toys_llm_rank < beauty_llm_rank * 0.7:
            suggestions.append({
                'issue': 'Low effective rank on Toys',
                'severity': 'MEDIUM',
                'recommendation': 'Toys embeddings have lower intrinsic dimensionality',
                'actions': [
                    'Reduce hidden_size to match intrinsic dimensionality',
                    'Add stronger regularization to prevent overfitting',
                    'Consider using PCA/whitening more aggressively',
                ],
                'config_changes': {
                    'weight_decay': 5e-5,  # Increase from 1e-5
                    'cross_dropout_prob': 0.6,  # Increase from 0.5
                }
            })
    
    # Compare pairwise similarities
    beauty_llm_sim = beauty_fs.get('llm', {}).get('mean_pairwise_cosine')
    toys_llm_sim = toys_fs.get('llm', {}).get('mean_pairwise_cosine')
    
    if beauty_llm_sim and toys_llm_sim:
        print(f"\nMean Pairwise Cosine (LLM embeddings):")
        print(f"  Beauty: {beauty_llm_sim:.4f}")
        print(f"  Toys:   {toys_llm_sim:.4f}")
        
        if toys_llm_sim > beauty_llm_sim + 0.05:
            suggestions.append({
                'issue': 'High inter-item similarity on Toys',
                'severity': 'HIGH',
                'recommendation': 'Toys items are more similar, making discrimination harder',
                'actions': [
                    'Increase temperature in alignment loss to focus on hard negatives',
                    'Use contrastive learning with harder negative sampling',
                    'Increase alignment_weight to strengthen feature alignment',
                ],
                'config_changes': {
                    'temperature': 0.1,  # Increase from 0.07
                    'alignment_weight': 0.1,  # Increase from 0.05
                    'use_hard_negative_sampling': True,
                }
            })
    
    return suggestions


def compare_statistics(beauty_data, toys_data):
    """Compare basic embedding statistics."""
    suggestions = []
    
    beauty_stats = beauty_data.get('statistics', {})
    toys_stats = toys_data.get('statistics', {})
    
    # Compare view statistics
    beauty_views = beauty_stats.get('views', {})
    toys_views = toys_stats.get('views', {})
    
    if beauty_views and toys_views:
        # Compare feature variance
        beauty_vars = [v.get('feature_var_mean', 0) for v in beauty_views.values()]
        toys_vars = [v.get('feature_var_mean', 0) for v in toys_views.values()]
        
        beauty_avg_var = sum(beauty_vars) / len(beauty_vars) if beauty_vars else 0
        toys_avg_var = sum(toys_vars) / len(toys_vars) if toys_vars else 0
        
        print(f"\nAverage Feature Variance (across views):")
        print(f"  Beauty: {beauty_avg_var:.6f}")
        print(f"  Toys:   {toys_avg_var:.6f}")
        
        if toys_avg_var < beauty_avg_var * 0.5:
            suggestions.append({
                'issue': 'Low feature variance on Toys',
                'severity': 'MEDIUM',
                'recommendation': 'Toys views have lower feature variance',
                'actions': [
                    'Apply stronger whitening/normalization',
                    'Increase feature dimension before fusion',
                ],
                'config_changes': {
                    'text_proj_norm': True,
                    'fused_item_norm': True,
                }
            })
    
    return suggestions


def generate_config_variants(base_config_path, suggestions):
    """Generate config file variants based on suggestions."""
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    variants = []
    
    # Create a variant for each high-severity suggestion
    for i, sugg in enumerate(suggestions):
        if sugg['severity'] == 'HIGH':
            variant_config = base_config.copy()
            variant_config.update(sugg['config_changes'])
            
            variants.append({
                'name': f"variant_{i+1}_{sugg['issue'].replace(' ', '_').lower()}",
                'description': sugg['recommendation'],
                'config': variant_config,
            })
    
    # Create a combined variant with all changes
    combined_config = base_config.copy()
    for sugg in suggestions:
        combined_config.update(sugg['config_changes'])
    
    variants.append({
        'name': 'variant_combined_all_fixes',
        'description': 'Combined all suggested changes',
        'config': combined_config,
    })
    
    return variants


def main():
    parser = argparse.ArgumentParser(description="Generate tuning suggestions")
    parser.add_argument('--beauty_analysis', type=str, required=True,
                       help='Path to Beauty analysis JSON')
    parser.add_argument('--toys_analysis', type=str, required=True,
                       help='Path to Toys analysis JSON')
    parser.add_argument('--base_config', type=str,
                       default='sasrec_align_multi_view_toys.yaml',
                       help='Base config file to modify')
    parser.add_argument('--output_dir', type=str, default='config_variants',
                       help='Output directory for config variants')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("TUNING SUGGESTIONS GENERATOR")
    print("="*80 + "\n")
    
    # Load analyses
    print(f"Loading Beauty analysis: {args.beauty_analysis}")
    beauty_data = load_analysis(args.beauty_analysis)
    
    print(f"Loading Toys analysis: {args.toys_analysis}")
    toys_data = load_analysis(args.toys_analysis)
    
    # Compare and generate suggestions
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    
    all_suggestions = []
    
    all_suggestions.extend(compare_diversity(beauty_data, toys_data))
    all_suggestions.extend(compare_feature_space(beauty_data, toys_data))
    all_suggestions.extend(compare_statistics(beauty_data, toys_data))
    
    # Print all suggestions
    print("\n" + "="*80)
    print("TUNING SUGGESTIONS")
    print("="*80 + "\n")
    
    for i, sugg in enumerate(all_suggestions, 1):
        print(f"{i}. [{sugg['severity']}] {sugg['issue']}")
        print(f"   {sugg['recommendation']}")
        print(f"\n   Actions:")
        for action in sugg['actions']:
            print(f"     • {action}")
        print(f"\n   Config changes:")
        for key, value in sugg['config_changes'].items():
            print(f"     {key}: {value}")
        print()
    
    # Save suggestions
    suggestions_file = 'tuning_suggestions.json'
    with open(suggestions_file, 'w') as f:
        json.dump(all_suggestions, f, indent=2)
    
    print(f"✓ Suggestions saved to: {suggestions_file}")
    
    # Generate config variants
    if Path(args.base_config).exists():
        print(f"\nGenerating config variants based on {args.base_config}...")
        
        variants = generate_config_variants(args.base_config, all_suggestions)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save variants
        for variant in variants:
            variant_path = output_dir / f"{variant['name']}.yaml"
            
            with open(variant_path, 'w') as f:
                f.write(f"# {variant['description']}\n\n")
                yaml.dump(variant['config'], f, default_flow_style=False)
            
            print(f"  ✓ Created: {variant_path}")
        
        print(f"\n✓ Generated {len(variants)} config variants in {output_dir}/")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    print(f"Total suggestions: {len(all_suggestions)}")
    print(f"  HIGH severity: {sum(1 for s in all_suggestions if s['severity'] == 'HIGH')}")
    print(f"  MEDIUM severity: {sum(1 for s in all_suggestions if s['severity'] == 'MEDIUM')}")
    print(f"\nReview {suggestions_file} for details and test config variants.")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
