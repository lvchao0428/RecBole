#!/usr/bin/env python3
"""
plot_sensitivity.py - Generate sensitivity analysis figures for the paper.

The sensitivity plot now reads raw experiment results directly from
`paper_sigir/sentitivity` and extracts four hyperparameter groups:
alignment weight, cold_text_boost, infer boost, and temperature.
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
import re

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

DEFAULT_DATA_FILE = Path(__file__).parent.parent / 'sentitivity'

PARAMETER_CONFIG = {
    'lambda': {
        'param': r'$\lambda$ (Alignment Weight)',
        'x_label': r'$\lambda$',
        'baseline_value': 0.10,
        'pattern': re.compile(r'\+\s*align\s*([0-9.]+)', re.IGNORECASE),
    },
    'cold_boost': {
        'param': 'Cold-start Boost',
        'x_label': 'cold_text_boost',
        'baseline_value': 3.0,
        'pattern': re.compile(r'\+\s*cold_text_boost\s*([0-9.]+)', re.IGNORECASE),
    },
    'infer_boost': {
        'param': 'Inference Boost',
        'x_label': 'infer_boost',
        'baseline_value': 0.6,
        'pattern': re.compile(r'\+\s*infer\s*boost\s*([0-9.]+)', re.IGNORECASE),
    },
    'tau': {
        'param': r'$\tau$ (Temperature)',
        'x_label': r'$\tau$',
        'baseline_value': 0.05,
        'pattern': re.compile(r'\+\s*temperature\s*([0-9.]+)', re.IGNORECASE),
    },
}

METRIC_COLUMNS = {
    'HR@10': 'hit@10',
    'NDCG@10': 'ndcg@10',
    'MRR@10': 'mrr@10',
    'HR_new@10': 'Hit_new@10',
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

def _get_header_indices(header_tokens):
    """Build first-occurrence index map for header columns."""
    idx_map = {}
    for idx, col_name in enumerate(header_tokens):
        col_name = col_name.strip()
        if col_name and col_name not in idx_map:
            idx_map[col_name] = idx
    return idx_map


def _match_parameter(row_name):
    """Match row name to a sensitivity hyperparameter group."""
    for param_key, config in PARAMETER_CONFIG.items():
        matched = config['pattern'].search(row_name)
        if matched:
            return param_key, float(matched.group(1))
    return None, None


def _format_tick_labels(values):
    """Format x ticks to compact decimal text."""
    labels = []
    for v in values:
        if float(v).is_integer():
            labels.append(str(int(v)))
        else:
            labels.append(f'{v:.2f}'.rstrip('0').rstrip('.'))
    return labels


def load_sensitivity_data(data_file=DEFAULT_DATA_FILE):
    """
    Parse raw sensitivity file and return plotting-ready data.

    Returns:
        dict: keyed by parameter group (`lambda`, `tau`, `cold_boost`, `infer_boost`).
    """
    lines = Path(data_file).read_text(encoding='utf-8').splitlines()
    header_tokens = None

    for line in lines:
        if '训练脚本' in line and 'hit@10' in line and 'Hit_new@10' in line:
            header_tokens = [token.strip() for token in line.split('\t')]
            break

    if header_tokens is None:
        raise ValueError(f'Cannot find valid header row in {data_file}')

    header_idx = _get_header_indices(header_tokens)
    required_cols = list(METRIC_COLUMNS.values())
    missing_cols = [col for col in required_cols if col not in header_idx]
    if missing_cols:
        raise ValueError(f'Missing metric columns in {data_file}: {missing_cols}')

    grouped_points = {param_key: [] for param_key in PARAMETER_CONFIG}

    for line in lines:
        stripped = line.strip()
        if not stripped or not stripped.startswith('multi-view'):
            continue

        tokens = [token.strip() for token in line.split('\t')]
        row_name = tokens[0]
        param_key, param_value = _match_parameter(row_name)
        if param_key is None:
            continue

        metric_values = {}
        try:
            for metric_name, col_name in METRIC_COLUMNS.items():
                col_idx = header_idx[col_name]
                metric_values[metric_name] = float(tokens[col_idx]) * 100.0
        except (IndexError, ValueError):
            continue

        grouped_points[param_key].append((param_value, metric_values))

    plot_data = {}
    for param_key, config in PARAMETER_CONFIG.items():
        points = sorted(grouped_points[param_key], key=lambda item: item[0])
        if not points:
            continue

        x_values = [p[0] for p in points]
        metric_series = {
            metric_name: [p[1][metric_name] for p in points]
            for metric_name in METRIC_COLUMNS
        }

        baseline_value = config['baseline_value']
        baseline_idx = next(
            (i for i, value in enumerate(x_values) if abs(value - baseline_value) < 1e-9),
            int(np.argmin(np.abs(np.array(x_values) - baseline_value))),
        )

        plot_data[param_key] = {
            'param': config['param'],
            'x_label': config['x_label'],
            'values': x_values,
            'baseline_idx': baseline_idx,
            **metric_series,
        }

    return plot_data


def create_sensitivity_plot(data_file=DEFAULT_DATA_FILE, show=False):
    """Create 2x2 sensitivity analysis plot from raw sensitivity data file."""
    data = load_sensitivity_data(data_file)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    param_order = ['lambda', 'tau', 'cold_boost', 'infer_boost']
    
    for ax_idx, param_key in enumerate(param_order):
        ax = axes[ax_idx]
        param_data = data[param_key]
        x = param_data['values']
        baseline_idx = param_data['baseline_idx']
        
        # Plot each metric
        for metric in ['HR@10', 'NDCG@10', 'MRR@10', 'HR_new@10']:
            y = param_data[metric]
            ax.plot(x, y,
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
        ax.set_xticklabels(_format_tick_labels(x))
        
        # Grid styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Legend only in first subplot
        if ax_idx == 0:
            ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
    
    # Add note for data source and baseline marker
    fig.text(0.5, 0.01, 
             f'Raw data source: {Path(data_file).name}; baseline configuration circled',
             ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figures
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    fig.savefig(output_dir / 'sensitivity.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'sensitivity.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'sensitivity.pdf'}")
    print(f"✅ Saved: {output_dir / 'sensitivity.png'}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
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
    parser = argparse.ArgumentParser(description='Generate paper figures.')
    parser.add_argument(
        '--data-file',
        type=str,
        default=str(DEFAULT_DATA_FILE),
        help='Path to sensitivity raw data file (default: paper_sigir/sentitivity).',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display figures interactively.',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Also generate legacy scale-law and trade-off figures.',
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Generating Sensitivity Analysis Figures")
    print("=" * 60)

    print("\n1. Creating sensitivity analysis plot...")
    create_sensitivity_plot(data_file=args.data_file, show=args.show)

    if args.all:
        print("\n2. Creating Scale Law verification plot...")
        create_scale_law_plot()

        print("\n3. Creating HR-MRR trade-off plot...")
        create_trade_off_plot()

    print("\n" + "=" * 60)
    print("✅ Figure generation completed!")
    print("=" * 60)
