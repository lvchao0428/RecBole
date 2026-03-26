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
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
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
                        'marker': 'o', 'color': '#1a1a1a', 'facecolor': '#1a1a1a'},
    '\u2212SE':             {'hr': 6.62, 'ndcg': 4.39, 'mrr': 3.70,
                        'marker': 's', 'color': '#666666', 'facecolor': '#666666'},
    '\u2212Cross':          {'hr': 7.22, 'ndcg': 4.23, 'mrr': 3.31,
                        'marker': '^', 'color': '#1a1a1a', 'facecolor': 'white'},
    '\u2212SE\u2212Cross':       {'hr': 7.23, 'ndcg': 4.24, 'mrr': 3.31,
                        'marker': 'D', 'color': '#666666', 'facecolor': 'white'},
}

whiten_data = {
    'w/ Whiten':  {'hr': 6.58, 'ndcg': 4.40, 'mrr': 3.72,
                   'marker': 'o', 'color': '#1a1a1a', 'facecolor': '#1a1a1a'},
    'w/o Whiten': {'hr': 6.46, 'ndcg': 4.34, 'mrr': 3.68,
                   'marker': 'v', 'color': '#999999', 'facecolor': '#999999'},
}

# =============================================================================
# Plotting helper
# =============================================================================

def plot_tradeoff(y_key, y_label, title_suffix, fname_suffix):
    """Draw one HR-vs-{y_key} scatter plot and save PDF+PNG."""
    fig, ax = plt.subplots(figsize=(3.8, 3.8))

    for label, d in ablation_data.items():
        ax.scatter(d['hr'], d[y_key],
                   marker=d['marker'], c=d.get('facecolor', d['color']),
                   s=120, label=label,
                   edgecolors=d['color'], linewidths=1.3, zorder=3)

    wd = whiten_data['w/o Whiten']
    ax.scatter(wd['hr'], wd[y_key],
               marker=wd['marker'], c=wd.get('facecolor', wd['color']),
               s=120, label='w/o Whiten',
               edgecolors=wd['color'], linewidths=1.3, zorder=3)

    full = ablation_data['Full (SE+Cross)']
    cross = ablation_data['\u2212Cross']
    ax.annotate('',
                xy=(cross['hr'], cross[y_key]),
                xytext=(full['hr'], full[y_key]),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.8, ls='--'),
                zorder=1)

    mid_x = (full['hr'] + cross['hr']) / 2
    mid_y = (full[y_key] + cross[y_key]) / 2
    ax.text(mid_x, mid_y + 0.04,
            f'HR$\\uparrow$ {y_label.split("@")[0]}$\\downarrow$\n(trade-off)',
            fontsize=9, ha='center', va='bottom',
            color='gray', style='italic')

    ax.set_xlabel('HR@10 (%)', fontweight='bold')
    ax.set_ylabel(f'{y_label} (%)', fontweight='bold')
    ax.set_title(f'HR\u2013{title_suffix} Trade-off on Toys&Games (7B)',
                 fontweight='bold', pad=6)
    ax.tick_params(axis='both', which='major', width=1.0, colors='#111111')

    all_x = [d['hr'] for d in ablation_data.values()] + [wd['hr']]
    all_y = [d[y_key] for d in ablation_data.values()] + [wd[y_key]]
    pad_x = (max(all_x) - min(all_x)) * 0.25
    pad_y = (max(all_y) - min(all_y)) * 0.25
    ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
    ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)

    ax.set_aspect('auto')

    leg = ax.legend(
        loc='lower left',
        ncol=2,
        framealpha=0.92,
        edgecolor='#cccccc',
        fancybox=False,
        markerscale=0.9,
        columnspacing=0.6,
        handletextpad=0.3,
        labelspacing=0.35,
        borderpad=0.4,
        fontsize=9,
    )
    leg.get_frame().set_linewidth(0.6)

    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.4)
    fig.tight_layout(pad=0.5)

    pdf_name = f'tradeoff_{fname_suffix}.pdf'
    png_name = f'tradeoff_{fname_suffix}.png'
    fig.savefig(pdf_name, format='pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(png_name, format='png', bbox_inches='tight', pad_inches=0.05, dpi=300)
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
