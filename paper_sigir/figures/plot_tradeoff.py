#!/usr/bin/env python3
"""
HR–MRR / HR–NDCG Trade-off Scatter Plots for SIGIR Paper

Generates two publication-ready scatter plots:
  1) HR@10 vs MRR@10   → tradeoff_mrr.pdf / tradeoff_mrr.png
  2) HR@10 vs NDCG@10  → tradeoff_ndcg.pdf / tradeoff_ndcg.png

Usage:
    python plot_tradeoff.py
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# =============================================================================
# Data: Ablation results on Toys&Games (from ablation0309.txt)
# =============================================================================

ablation_data = {
    'Full (SE+Cross)': {'hr': 6.58, 'ndcg': 4.40, 'mrr': 3.72,
                        'marker': 'o', 'color': '#2E86AB'},
    '−SE':             {'hr': 6.62, 'ndcg': 4.39, 'mrr': 3.70,
                        'marker': 's', 'color': '#A23B72'},
    '−Cross':          {'hr': 7.22, 'ndcg': 4.23, 'mrr': 3.31,
                        'marker': '^', 'color': '#F18F01'},
    '−SE−Cross':       {'hr': 7.23, 'ndcg': 4.24, 'mrr': 3.31,
                        'marker': 'D', 'color': '#C73E1D'},
}

whiten_data = {
    'w/ Whiten':  {'hr': 6.58, 'ndcg': 4.40, 'mrr': 3.72,
                   'marker': 'o', 'color': '#2E86AB'},
    'w/o Whiten': {'hr': 6.46, 'ndcg': 4.34, 'mrr': 3.68,
                   'marker': 'v', 'color': '#3B7A57'},
}

# =============================================================================
# Plotting helper
# =============================================================================

def plot_tradeoff(y_key, y_label, title_suffix, fname_suffix):
    """Draw one HR-vs-{y_key} scatter plot and save PDF+PNG."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for label, d in ablation_data.items():
        ax.scatter(d['hr'], d[y_key],
                   marker=d['marker'], c=d['color'],
                   s=120, label=label,
                   edgecolors='black', linewidths=0.8, zorder=3)

    wd = whiten_data['w/o Whiten']
    ax.scatter(wd['hr'], wd[y_key],
               marker=wd['marker'], c=wd['color'],
               s=120, label='w/o Whiten',
               edgecolors='black', linewidths=0.8, zorder=3)

    full = ablation_data['Full (SE+Cross)']
    cross = ablation_data['−Cross']
    ax.annotate('',
                xy=(cross['hr'], cross[y_key]),
                xytext=(full['hr'], full[y_key]),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'),
                zorder=1)

    mid_x = (full['hr'] + cross['hr']) / 2
    mid_y = (full[y_key] + cross[y_key]) / 2
    ax.text(mid_x, mid_y + 0.04,
            f'HR↑ {y_label.split("@")[0]}↓\n(trade-off)',
            fontsize=9, ha='center', va='bottom',
            color='gray', style='italic')

    ax.set_xlabel('HR@10 (%)', fontweight='bold')
    ax.set_ylabel(f'{y_label} (%)', fontweight='bold')
    ax.set_title(f'HR–{title_suffix} Trade-off on Toys&Games (7B)',
                 fontweight='bold')

    all_x = [d['hr'] for d in ablation_data.values()] + [wd['hr']]
    all_y = [d[y_key] for d in ablation_data.values()] + [wd[y_key]]
    pad_x = (max(all_x) - min(all_x)) * 0.25
    pad_y = (max(all_y) - min(all_y)) * 0.25
    ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
    ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)

    ax.legend(loc='lower left', framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.tight_layout()

    pdf_name = f'tradeoff_{fname_suffix}.pdf'
    png_name = f'tradeoff_{fname_suffix}.png'
    plt.savefig(pdf_name, format='pdf', bbox_inches='tight')
    plt.savefig(png_name, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_name} and {png_name}")

    print(f"\nData points ({y_label}):")
    print("-" * 55)
    for label, d in ablation_data.items():
        print(f"  {label:20s}: HR@10={d['hr']:.2f}%, {y_label}={d[y_key]:.2f}%")
    print(f"  {'w/o Whiten':20s}: HR@10={wd['hr']:.2f}%, {y_label}={wd[y_key]:.2f}%")

    plt.show()
    return fig

# =============================================================================
# Generate both plots
# =============================================================================

fig_mrr  = plot_tradeoff('mrr',  'MRR@10',  'MRR',  'mrr')
fig_ndcg = plot_tradeoff('ndcg', 'NDCG@10', 'NDCG', 'ndcg')
