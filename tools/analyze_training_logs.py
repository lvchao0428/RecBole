#!/usr/bin/env python3
"""
Training Log Analysis Tool

Analyzes training logs to diagnose multi-view training issues.
Checks for:
- Alignment loss patterns
- Gate values evolution
- Per-view contribution
- Overfitting signals

Usage:
    python tools/analyze_training_logs.py --log_dir saved/phase_runs_multiview_4views_toys
    python tools/analyze_training_logs.py --log_dir saved/phase_runs_multiview_4views
"""

import argparse
import re
from pathlib import Path
import json
import numpy as np


def parse_log_file(log_path):
    """Parse a training log file and extract metrics."""
    metrics = {
        'epochs': [],
        'train_loss': [],
        'valid_recall_10': [],
        'valid_mrr_10': [],
        'valid_ndcg_10': [],
        'test_recall_10': [],
        'test_mrr_10': [],
        'test_ndcg_10': [],
        'alignment_loss': [],
        'gate_values': [],
    }
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Extract epoch number
            epoch_match = re.search(r'epoch (\d+)', line, re.IGNORECASE)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                metrics['epochs'].append(epoch)
            
            # Extract training loss
            loss_match = re.search(r'(?:train|training).*?loss[:\s]+([0-9.]+)', line, re.IGNORECASE)
            if loss_match:
                metrics['train_loss'].append(float(loss_match.group(1)))
            
            # Extract alignment loss (if logged separately)
            align_match = re.search(r'align(?:ment)?.*?loss[:\s]+([0-9.]+)', line, re.IGNORECASE)
            if align_match:
                metrics['alignment_loss'].append(float(align_match.group(1)))
            
            # Extract validation metrics
            if 'valid' in line.lower() or 'validation' in line.lower():
                recall_match = re.search(r'Recall@10[:\s]+([0-9.]+)', line, re.IGNORECASE)
                if recall_match:
                    metrics['valid_recall_10'].append(float(recall_match.group(1)))
                
                mrr_match = re.search(r'MRR@10[:\s]+([0-9.]+)', line, re.IGNORECASE)
                if mrr_match:
                    metrics['valid_mrr_10'].append(float(mrr_match.group(1)))
                
                ndcg_match = re.search(r'NDCG@10[:\s]+([0-9.]+)', line, re.IGNORECASE)
                if ndcg_match:
                    metrics['valid_ndcg_10'].append(float(ndcg_match.group(1)))
            
            # Extract test metrics
            if 'test' in line.lower():
                recall_match = re.search(r'Recall@10[:\s]+([0-9.]+)', line, re.IGNORECASE)
                if recall_match:
                    metrics['test_recall_10'].append(float(recall_match.group(1)))
                
                mrr_match = re.search(r'MRR@10[:\s]+([0-9.]+)', line, re.IGNORECASE)
                if mrr_match:
                    metrics['test_mrr_10'].append(float(mrr_match.group(1)))
                
                ndcg_match = re.search(r'NDCG@10[:\s]+([0-9.]+)', line, re.IGNORECASE)
                if ndcg_match:
                    metrics['test_ndcg_10'].append(float(ndcg_match.group(1)))
            
            # Extract gate values (if logged)
            gate_match = re.search(r'gate.*?([0-9.]+)', line, re.IGNORECASE)
            if gate_match and 'text' in line.lower():
                metrics['gate_values'].append(float(gate_match.group(1)))
    
    return metrics


def analyze_convergence(metrics):
    """Analyze convergence patterns."""
    analysis = {}
    
    if metrics['valid_recall_10']:
        recalls = np.array(metrics['valid_recall_10'])
        
        # Best performance
        best_idx = np.argmax(recalls)
        analysis['best_epoch'] = int(best_idx)
        analysis['best_recall'] = float(recalls[best_idx])
        
        # Check for overfitting
        if len(recalls) > best_idx + 5:
            post_best_mean = recalls[best_idx+1:].mean()
            analysis['overfitting_signal'] = float(recalls[best_idx] - post_best_mean)
        
        # Check for early plateau
        if len(recalls) >= 10:
            first_half_mean = recalls[:len(recalls)//2].mean()
            second_half_mean = recalls[len(recalls)//2:].mean()
            analysis['improvement_trend'] = float(second_half_mean - first_half_mean)
        
        # Stability
        if len(recalls) >= 5:
            analysis['stability_std'] = float(recalls[-5:].std())
    
    if metrics['train_loss']:
        losses = np.array(metrics['train_loss'])
        
        # Check if loss is decreasing
        if len(losses) >= 10:
            first_half_mean = losses[:len(losses)//2].mean()
            second_half_mean = losses[len(losses)//2:].mean()
            analysis['loss_decrease'] = float(first_half_mean - second_half_mean)
    
    if metrics['alignment_loss']:
        align_losses = np.array(metrics['alignment_loss'])
        analysis['avg_alignment_loss'] = float(align_losses.mean())
        analysis['alignment_loss_std'] = float(align_losses.std())
    
    return analysis


def compare_logs(beauty_dir, toys_dir):
    """Compare training logs between Beauty and Toys."""
    comparison = {}
    
    for dataset_name, log_dir in [('Beauty', beauty_dir), ('Toys', toys_dir)]:
        log_dir_path = Path(log_dir)
        
        if not log_dir_path.exists():
            print(f"Warning: {log_dir} does not exist")
            continue
        
        # Find log files
        log_files = list(log_dir_path.glob('*.log'))
        if not log_files:
            log_files = list(log_dir_path.glob('**/*.log'))
        
        if log_files:
            print(f"Found {len(log_files)} log file(s) for {dataset_name}")
            
            # Parse the most recent log
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            print(f"  Analyzing: {latest_log}")
            
            metrics = parse_log_file(latest_log)
            analysis = analyze_convergence(metrics)
            
            comparison[dataset_name] = {
                'log_file': str(latest_log),
                'metrics': {k: v for k, v in metrics.items() if v},  # Only non-empty
                'analysis': analysis,
            }
        else:
            print(f"No log files found for {dataset_name}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Analyze training logs")
    parser.add_argument('--log_dir', type=str, required=False,
                       help='Path to log directory')
    parser.add_argument('--compare', action='store_true',
                       help='Compare Beauty vs Toys logs')
    parser.add_argument('--beauty_dir', type=str,
                       default='saved/phase_runs_multiview_4views',
                       help='Beauty log directory')
    parser.add_argument('--toys_dir', type=str,
                       default='saved/phase_runs_multiview_4views_toys',
                       help='Toys log directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("TRAINING LOG ANALYSIS")
    print("="*80 + "\n")
    
    results = {}
    
    if args.compare:
        print("Comparing Beauty vs Toys training logs...\n")
        comparison = compare_logs(args.beauty_dir, args.toys_dir)
        results['comparison'] = comparison
        
        # Print summary
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80 + "\n")
        
        for dataset in ['Beauty', 'Toys']:
            if dataset in comparison:
                analysis = comparison[dataset]['analysis']
                print(f"{dataset}:")
                if 'best_recall' in analysis:
                    print(f"  Best Recall@10: {analysis['best_recall']:.4f} (epoch {analysis['best_epoch']})")
                if 'overfitting_signal' in analysis:
                    print(f"  Overfitting signal: {analysis['overfitting_signal']:.4f}")
                if 'improvement_trend' in analysis:
                    print(f"  Improvement trend: {analysis['improvement_trend']:.4f}")
                if 'stability_std' in analysis:
                    print(f"  Stability (last 5 epochs std): {analysis['stability_std']:.4f}")
                if 'avg_alignment_loss' in analysis:
                    print(f"  Avg alignment loss: {analysis['avg_alignment_loss']:.4f}")
                print()
        
    elif args.log_dir:
        log_path = Path(args.log_dir)
        
        log_files = list(log_path.glob('*.log'))
        if not log_files:
            log_files = list(log_path.glob('**/*.log'))
        
        if log_files:
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            print(f"Analyzing: {latest_log}\n")
            
            metrics = parse_log_file(latest_log)
            analysis = analyze_convergence(metrics)
            
            results['log_file'] = str(latest_log)
            results['metrics'] = {k: v for k, v in metrics.items() if v}
            results['analysis'] = analysis
            
            # Print summary
            print("\nAnalysis Summary:")
            print("-" * 80)
            for key, value in analysis.items():
                print(f"  {key}: {value}")
        else:
            print(f"No log files found in {log_path}")
    
    # Save results
    if results:
        output_file = args.output or 'training_log_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*80)
        print(f"✓ Analysis saved to: {output_file}")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
