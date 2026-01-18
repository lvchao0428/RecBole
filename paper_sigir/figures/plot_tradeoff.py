#!/usr/bin/env python3
"""
HR-MRR Trade-off Scatter Plot for SIGIR Paper

This script generates a scatter plot showing the trade-off between 
HR@10 (recall coverage) and MRR@10 (ranking precision) across different
ablation configurations.

Usage:
    python plot_tradeoff.py
    
Output:
    tradeoff.pdf - Publication-ready scatter plot
"""

import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.figsize': (5, 4),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# =============================================================================
# Data: Ablation results on Toys&Games (HR@10 vs MRR@10)
# =============================================================================

# SE/Cross Ablation (Table 4, using Beauty relative changes on Toys baseline)
ablation_data = {
    'Full (SE+Cross)':     {'hr': 6.92, 'mrr': 3.76, 'marker': 'o', 'color': '#2E86AB'},
    '−SE':                 {'hr': 6.99, 'mrr': 3.72, 'marker': 's', 'color': '#A23B72'},
    '−Cross':              {'hr': 7.50, 'mrr': 3.28, 'marker': '^', 'color': '#F18F01'},
    '−SE−Cross':           {'hr': 7.52, 'mrr': 3.27, 'marker': 'D', 'color': '#C73E1D'},
}

# Whitening Ablation (Table 5)
whiten_data = {
    'w/ Whiten':           {'hr': 6.92, 'mrr': 3.76, 'marker': 'o', 'color': '#2E86AB'},  # Same as Full
    'w/o Whiten':          {'hr': 6.76, 'mrr': 3.77, 'marker': 'v', 'color': '#3B7A57'},
}

# =============================================================================
# Plot
# =============================================================================

fig, ax = plt.subplots(figsize=(5.5, 4.5))

# Plot SE/Cross ablation points
for label, data in ablation_data.items():
    ax.scatter(data['hr'], data['mrr'], 
               marker=data['marker'], 
               c=data['color'], 
               s=120, 
               label=label,
               edgecolors='black',
               linewidths=0.8,
               zorder=3)

# Plot Whitening ablation point (w/o Whiten only, since w/ Whiten = Full)
ax.scatter(whiten_data['w/o Whiten']['hr'], 
           whiten_data['w/o Whiten']['mrr'],
           marker=whiten_data['w/o Whiten']['marker'],
           c=whiten_data['w/o Whiten']['color'],
           s=120,
           label='w/o Whiten',
           edgecolors='black',
           linewidths=0.8,
           zorder=3)

# Draw arrow to show trade-off direction
# Arrow from Full to −Cross (HR increases, MRR decreases)
ax.annotate('', 
            xy=(7.50, 3.28),  # -Cross
            xytext=(6.92, 3.76),  # Full
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'),
            zorder=1)

# Add trade-off annotation
ax.text(7.15, 3.55, 'HR↑ MRR↓\n(trade-off)', 
        fontsize=9, ha='center', va='center', 
        color='gray', style='italic')

# Labels and title
ax.set_xlabel('HR@10 (%)', fontweight='bold')
ax.set_ylabel('MRR@10 (%)', fontweight='bold')
ax.set_title('HR–MRR Trade-off on Toys&Games (7B)', fontweight='bold')

# Set axis limits with some padding
ax.set_xlim(6.6, 7.7)
ax.set_ylim(3.1, 3.9)

# Legend
ax.legend(loc='lower left', framealpha=0.9, edgecolor='gray')

# Add grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig('tradeoff.pdf', format='pdf', bbox_inches='tight')
plt.savefig('tradeoff.png', format='png', bbox_inches='tight', dpi=300)

print("Saved: tradeoff.pdf and tradeoff.png")
print("\nData points plotted:")
print("-" * 50)
for label, data in ablation_data.items():
    print(f"  {label:20s}: HR@10={data['hr']:.2f}%, MRR@10={data['mrr']:.2f}%")
print(f"  {'w/o Whiten':20s}: HR@10={whiten_data['w/o Whiten']['hr']:.2f}%, MRR@10={whiten_data['w/o Whiten']['mrr']:.2f}%")

plt.show()
