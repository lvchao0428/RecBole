#!/usr/bin/env python3
"""
plot_sensitivity.py - Generate sensitivity analysis visualization for the paper

This script creates a 2x2 subplot figure showing the effect of hyperparameter 
variations on key metrics (HR@10, NDCG@10, MRR@10, HR_new@10).

Usage:
    python paper_sigir/scripts/plot_sensitivity.py

Output:
    paper_sigir/figures/sensitivity.pdf
    paper_sigir/figures/sensitivity.png
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9

# Sensitivity analysis data from 0111.csv experiments
# Baseline: λ=0.10, τ=0.05, cold=2.5, infer=1.5
# All values in percentage (%)

data = {
    'lambda': {
        'param': r'$\lambda$ (Alignment Weight)',
        'values': [0.05, 0.10, 0.15],
        'x_label': r'$\lambda$',
        'HR@10': [6.82, 6.92, 6.93],
        'NDCG@10': [4.45, 4.51, 4.48],
        'MRR@10': [3.72, 3.76, 3.72],
        'HR_new@10': [1.98, 1.99, 2.04],
        'baseline_idx': 1,
    },
    'tau': {
        'param': r'$\tau$ (Temperature)',
        'values': [0.03, 0.05, 0.10],
        'x_label': r'$\tau$',
        'HR@10': [6.94, 6.92, 6.90],
        'NDCG@10': [4.50, 4.51, 4.50],
        'MRR@10': [3.75, 3.76, 3.76],
        'HR_new@10': [2.01, 1.99, 2.01],
        'baseline_idx': 1,
    },
    'cold_boost': {
        'param': 'Cold-start Boost',
        'values': [1.5, 2.5, 3.0],
        'x_label': 'cold_boost',
        'HR@10': [6.90, 6.92, 6.89],
        'NDCG@10': [4.46, 4.51, 4.47],
        'MRR@10': [3.71, 3.76, 3.72],
        'HR_new@10': [1.95, 1.99, 2.02],
        'baseline_idx': 1,
    },
    'infer_boost': {
        'param': 'Inference Boost',
        'values': [0.5, 1.0, 1.5],  # 1.0 pending, will use interpolation
        'x_label': 'infer_boost',
        'HR@10': [6.70, None, 6.92],  # None = pending experiment
        'NDCG@10': [4.40, None, 4.51],
        'MRR@10': [3.69, None, 3.76],
        'HR_new@10': [1.95, None, 1.99],
        'baseline_idx': 2,
    },
}

# Colors for different metrics
colors = {
    'HR@10': '#2E86AB',      # Blue
    'NDCG@10': '#28A745',    # Green
    'MRR@10': '#FD7E14',     # Orange
    'HR_new@10': '#DC3545',  # Red
}

markers = {
    'HR@10': 'o',
    'NDCG@10': 's',
    'MRR@10': '^',
    'HR_new@10': 'D',
}

def interpolate_missing(values):
    """Linear interpolation for missing values."""
    values = list(values)
    for i, v in enumerate(values):
        if v is None:
            # Find nearest non-None values
            left = right = None
            for j in range(i-1, -1, -1):
                if values[j] is not None:
                    left = (j, values[j])
                    break
            for j in range(i+1, len(values)):
                if values[j] is not None:
                    right = (j, values[j])
                    break
            if left and right:
                # Linear interpolation
                values[i] = left[1] + (right[1] - left[1]) * (i - left[0]) / (right[0] - left[0])
            elif left:
                values[i] = left[1]
            elif right:
                values[i] = right[1]
    return values


def create_sensitivity_plot():
    """Create 2x2 sensitivity analysis plot."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    param_order = ['lambda', 'tau', 'cold_boost', 'infer_boost']
    
    for ax_idx, param_key in enumerate(param_order):
        ax = axes[ax_idx]
        param_data = data[param_key]
        x = param_data['values']
        baseline_idx = param_data['baseline_idx']
        
        # Plot each metric
        for metric in ['HR@10', 'MRR@10', 'HR_new@10']:
            y = interpolate_missing(param_data[metric])
            line, = ax.plot(x, y, 
                           marker=markers[metric], 
                           color=colors[metric],
                           linewidth=2,
                           markersize=8,
                           label=metric,
                           markeredgecolor='white',
                           markeredgewidth=1.5)
            
            # Mark baseline point
            ax.scatter([x[baseline_idx]], [y[baseline_idx]], 
                      s=150, facecolors='none', edgecolors=colors[metric], 
                      linewidths=2, zorder=5)
        
        ax.set_xlabel(param_data['x_label'], fontweight='bold')
        ax.set_ylabel('Metric Value (%)', fontweight='bold')
        ax.set_title(f'({chr(97+ax_idx)}) {param_data["param"]}', fontweight='bold', pad=10)
        
        # Set x-axis ticks
        ax.set_xticks(x)
        
        # Grid styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Legend only in first subplot
        if ax_idx == 0:
            ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
    
    # Add note about infer_boost interpolation
    fig.text(0.5, 0.01, 
             'Note: infer=1.0 point in (d) is interpolated; baseline configuration circled',
             ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figures
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    fig.savefig(output_dir / 'sensitivity.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'sensitivity.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'sensitivity.pdf'}")
    print(f"✅ Saved: {output_dir / 'sensitivity.png'}")
    
    plt.show()
    return fig


def create_scale_law_plot():
    """Create Scale Law verification plot."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Data from Standard config (Toys)
    models = ['7B', '14B', '32B']
    x = [7, 14, 32]
    
    # Overall metrics
    hr_overall = [6.67, 6.71, 6.80]
    ndcg_overall = [4.39, 4.44, 4.42]
    
    # Cold-start metrics
    hr_new = [2.01, 2.02, 2.08]
    ndcg_new = [1.45, 1.44, 1.45]
    
    # Plot 1: Overall metrics
    ax1 = axes[0]
    ax1.plot(x, hr_overall, 'o-', color=colors['HR@10'], linewidth=2, markersize=10, label='HR@10')
    ax1.plot(x, ndcg_overall, 's-', color=colors['NDCG@10'], linewidth=2, markersize=10, label='NDCG@10')
    ax1.set_xlabel('Model Size (B)', fontweight='bold')
    ax1.set_ylabel('Metric Value (%)', fontweight='bold')
    ax1.set_title('(a) Overall Metrics', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Cold-start metrics
    ax2 = axes[1]
    ax2.plot(x, hr_new, 'o-', color=colors['HR_new@10'], linewidth=2, markersize=10, label='HR_new@10')
    ax2.plot(x, ndcg_new, 's-', color='#6C757D', linewidth=2, markersize=10, label='NDCG_new@10')
    ax2.set_xlabel('Model Size (B)', fontweight='bold')
    ax2.set_ylabel('Metric Value (%)', fontweight='bold')
    ax2.set_title('(b) Cold-start (new) Metrics', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save figures
    output_dir = Path(__file__).parent.parent / 'figures'
    fig.savefig(output_dir / 'scale_law.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'scale_law.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'scale_law.pdf'}")
    print(f"✅ Saved: {output_dir / 'scale_law.png'}")
    
    plt.show()
    return fig


def create_trade_off_plot():
    """Create HR vs MRR trade-off visualization."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data points from ablation experiments (Toys)
    # Format: (HR@10, MRR@10, label, color, marker)
    points = [
        (6.92, 3.76, 'Full Model', '#2E86AB', 'o'),
        (6.97, 3.78, '−SENet', '#28A745', 's'),
        (7.45, 3.18, '−Cross', '#FD7E14', '^'),
        (7.43, 3.19, '−SENet −Cross', '#DC3545', 'D'),
    ]
    
    for hr, mrr, label, color, marker in points:
        ax.scatter(hr, mrr, s=200, c=color, marker=marker, label=label, 
                  edgecolors='white', linewidths=2, zorder=5)
        ax.annotate(label, (hr, mrr), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, ha='left')
    
    # Draw trade-off arrow
    ax.annotate('', xy=(7.4, 3.2), xytext=(7.0, 3.7),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))
    ax.text(7.25, 3.5, 'Trade-off\nDirection', fontsize=9, ha='center', 
            color='gray', style='italic')
    
    ax.set_xlabel('HR@10 (%)', fontweight='bold', fontsize=12)
    ax.set_ylabel('MRR@10 (%)', fontweight='bold', fontsize=12)
    ax.set_title('HR vs MRR Trade-off (Toys, Ablation Study)', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figures
    output_dir = Path(__file__).parent.parent / 'figures'
    fig.savefig(output_dir / 'hr_mrr_tradeoff.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'hr_mrr_tradeoff.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'hr_mrr_tradeoff.pdf'}")
    print(f"✅ Saved: {output_dir / 'hr_mrr_tradeoff.png'}")
    
    plt.show()
    return fig


if __name__ == '__main__':
    print("=" * 60)
    print("Generating Sensitivity Analysis Figures")
    print("=" * 60)
    
    print("\n1. Creating sensitivity analysis plot...")
    create_sensitivity_plot()
    
    print("\n2. Creating Scale Law verification plot...")
    create_scale_law_plot()
    
    print("\n3. Creating HR-MRR trade-off plot...")
    create_trade_off_plot()
    
    print("\n" + "=" * 60)
    print("✅ All figures generated successfully!")
    print("=" * 60)
