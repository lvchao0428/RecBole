#!/usr/bin/env python3
"""
plot_sensitivity.py - Generate sensitivity analysis figures for the paper.

Reads raw experiment results from paper_sigir/sensitivity0309.txt and extracts
four hyperparameter groups: alignment weight, cold_text_boost, infer boost, temperature.

Modes:
  - tradeoff: Hit vs NDCG/MRR dual-axis (seesaw 跷跷板) per hyperparameter
  - pareto:   Scatter (Hit, NDCG) with Pareto frontier (帕累托边缘)
  - full:     Both tradeoff and pareto figures
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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

DEFAULT_DATA_FILE = Path(__file__).parent.parent / 'sensitivity0309.txt'

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


def _compute_pareto_frontier(x_vals, y_vals, labels=None):
    """
    Compute Pareto frontier for maximization of both x and y.
    Returns indices of Pareto-optimal points, sorted by x ascending.
    """
    pts = np.column_stack([np.asarray(x_vals), np.asarray(y_vals)])
    n = len(pts)
    pareto_idx = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has both >= and at least one strictly >
            if pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1]:
                if pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1]:
                    dominated = True
                    break
        if not dominated:
            pareto_idx.append(i)
    # Sort by x for drawing the frontier line
    pareto_idx = sorted(pareto_idx, key=lambda i: pts[i, 0])
    return pareto_idx


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


def load_all_sensitivity_points(data_file=DEFAULT_DATA_FILE):
    """
    Load all (Hit@10, NDCG@10, MRR@10, param_name) for Pareto / trade-off plots.
    Returns list of dicts: {hr, ndcg, mrr, param_key, param_value, row_name}.
    """
    data = load_sensitivity_data(data_file)
    all_pts = []
    for param_key, param_data in data.items():
        n = len(param_data['values'])
        for i in range(n):
            all_pts.append({
                'hr': param_data['HR@10'][i],
                'ndcg': param_data['NDCG@10'][i],
                'mrr': param_data['MRR@10'][i],
                'param_key': param_key,
                'param_value': param_data['values'][i],
                'label': f"{param_data['x_label']}={_format_tick_labels([param_data['values'][i]])[0]}",
            })
    return all_pts


def create_sensitivity_tradeoff_plot(
    data_file=DEFAULT_DATA_FILE,
    second_metric='NDCG@10',
    show=False,
):
    """
    Create 2x2 sensitivity plot: Hit@10 vs NDCG@10 (or MRR@10) dual-axis.
    Shows the seesaw (跷跷板) phenomenon: when one metric rises, the other may fall.
    """
    data = load_sensitivity_data(data_file)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    param_order = ['lambda', 'tau', 'cold_boost', 'infer_boost']
    color_hr = colors['HR@10']
    color_second = colors.get(second_metric, colors['NDCG@10'])

    for ax_idx, param_key in enumerate(param_order):
        ax = axes[ax_idx]
        param_data = data[param_key]
        x = param_data['values']
        baseline_idx = param_data['baseline_idx']
        y1 = param_data['HR@10']
        y2 = param_data[second_metric]

        ax2 = ax.twinx()
        line1, = ax.plot(x, y1, 'o-', color=color_hr, markersize=8, linewidth=2, label='HR@10')
        ax.set_ylabel('HR@10 (%)', color=color_hr)
        ax.tick_params(axis='y', labelcolor=color_hr)

        line2, = ax2.plot(x, y2, 's--', color=color_second, markersize=8, linewidth=2, label=second_metric)
        ax2.set_ylabel(f'{second_metric} (%)', color=color_second)
        ax2.tick_params(axis='y', labelcolor=color_second)

        # Mark baseline
        ax.axvline(x=x[baseline_idx], color='gray', linestyle=':', alpha=0.7, linewidth=1.2)

        ax.set_xlabel(param_data['x_label'], fontweight='bold')
        ax.set_title(f'({chr(97+ax_idx)}) {param_data["param"]}', fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(_format_tick_labels(x))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    fig.legend([line1, line2], ['HR@10 (Overall)', second_metric],
               loc='upper center', ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.text(0.5, 0.01,
             f'HR vs {second_metric} trade-off (seesaw); baseline config marked with vertical line',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])

    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    suffix = 'ndcg' if 'NDCG' in second_metric else 'mrr'
    fig.savefig(output_dir / f'sensitivity_tradeoff_{suffix}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'sensitivity_tradeoff_{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: sensitivity_tradeoff_{suffix}.pdf/.png")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def create_pareto_plot(data_file=DEFAULT_DATA_FILE, metric_pair='ndcg', show=False):
    """
    Create Pareto frontier scatter: HR@10 vs NDCG@10 (or MRR@10).

    Design goals for readability:
    - Each hyperparameter group is a different color; each setting is a distinct marker.
    - Baseline config per group is shown with a solid ring around the point.
    - Pareto-frontier points are highlighted with a bold edge + dashed staircase line.
    - Shaded region inside the frontier communicates the dominated area.
    - Annotations on Pareto-frontier points show which hyperparam/value they correspond to.
    - A trade-off arrow and legend quadrant note explain the HR↑ vs ranking-metric↓ tension.
    """
    data = load_sensitivity_data(data_file)

    param_colors = {
        'lambda':     '#2E86AB',  # blue
        'tau':        '#28A745',  # green
        'cold_boost': '#FD7E14',  # orange
        'infer_boost':'#9B59B6',  # purple
    }
    param_markers = {
        'lambda':     'o',
        'tau':        's',
        'cold_boost': '^',
        'infer_boost':'D',
    }
    param_labels = {
        'lambda':     r'$\lambda$ (align weight)',
        'tau':        r'$\tau$ (temperature)',
        'cold_boost': 'cold_text_boost',
        'infer_boost':'infer_boost',
    }

    second_col = 'ndcg' if metric_pair == 'ndcg' else 'mrr'
    metric_key  = 'NDCG@10' if metric_pair == 'ndcg' else 'MRR@10'
    xlabel, ylabel = 'HR@10 (%)', f'{metric_key} (%)'

    # Collect all points, remembering baseline membership
    all_hr, all_other, all_param, all_val, all_baseline = [], [], [], [], []
    for param_key, param_data in data.items():
        for i, (x_val, hr_val, oth_val) in enumerate(
            zip(param_data['values'], param_data['HR@10'], param_data[metric_key])
        ):
            all_hr.append(hr_val)
            all_other.append(oth_val)
            all_param.append(param_key)
            all_val.append(x_val)
            all_baseline.append(i == param_data['baseline_idx'])

    # Compute Pareto frontier
    pareto_idx = _compute_pareto_frontier(all_hr, all_other)
    pareto_set = set(pareto_idx)

    fig, ax = plt.subplots(figsize=(8, 6))

    # --- Shaded dominated region ---
    sorted_pf = sorted(pareto_idx, key=lambda i: all_hr[i])
    pf_hr  = [all_hr[i]    for i in sorted_pf]
    pf_oth = [all_other[i] for i in sorted_pf]
    # Staircase fill: extend to axis limits
    stair_x = [pf_hr[0]]
    stair_y = [pf_oth[0]]
    for xi, yi in zip(pf_hr[1:], pf_oth[1:]):
        stair_x.append(xi)
        stair_y.append(stair_y[-1])  # horizontal step
        stair_x.append(xi)
        stair_y.append(yi)           # vertical step
    ax.fill_between(
        stair_x + [stair_x[-1], stair_x[0]],
        stair_y + [min(pf_oth) - 1, min(pf_oth) - 1],
        alpha=0.07, color='gray', zorder=0,
        label='Dominated region',
    )

    # --- All non-Pareto points, by group ---
    legend_handles = {}
    for i in range(len(all_hr)):
        pk = all_param[i]
        is_pareto = i in pareto_set
        is_baseline = all_baseline[i]
        c = param_colors[pk]
        m = param_markers[pk]

        # Base scatter
        ax.scatter(
            all_hr[i], all_other[i],
            c=c, marker=m, s=70 if not is_pareto else 0,  # Pareto drawn later
            alpha=0.55 if not is_baseline else 0.9,
            edgecolors='white', linewidths=0.8,
            zorder=2,
        )

        # Baseline ring
        if is_baseline:
            ax.scatter(
                all_hr[i], all_other[i],
                facecolors='none', edgecolors=c, marker=m,
                s=200, linewidths=2.0, zorder=4,
            )

        if pk not in legend_handles:
            legend_handles[pk] = mlines.Line2D(
                [], [], color=c, marker=m, linestyle='None',
                markersize=8, label=param_labels[pk],
            )

    # --- Pareto points (highlighted) ---
    for i in sorted_pf:
        pk = all_param[i]
        c = param_colors[pk]
        m = param_markers[pk]
        ax.scatter(
            all_hr[i], all_other[i],
            c=c, marker=m, s=130,
            edgecolors='black', linewidths=1.8, zorder=5,
        )
        # Annotation: show hyperparam=value
        label_txt = _format_tick_labels([all_val[i]])[0]
        ax.annotate(
            f"{param_labels[pk].split('(')[-1].rstrip(')')}={label_txt}",
            (all_hr[i], all_other[i]),
            xytext=(6, 4), textcoords='offset points',
            fontsize=7.5, color='black',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.65, ec='none'),
            zorder=6,
        )

    # --- Pareto staircase line ---
    ax.step(
        pf_hr + [pf_hr[-1] + 0.05],
        [pf_oth[0]] + pf_oth,
        where='post', color='#DC3545', linewidth=2.0, linestyle='--',
        alpha=0.85, zorder=3, label='Pareto frontier',
    )

    # --- Baseline marker in legend ---
    legend_handles['_baseline'] = mlines.Line2D(
        [], [], color='gray', marker='o', linestyle='None',
        markersize=10, markerfacecolor='none', markeredgewidth=2.0,
        label='Baseline config (per group)',
    )
    legend_handles['_pareto'] = mlines.Line2D(
        [], [], color='#DC3545', linestyle='--', linewidth=2,
        label='Pareto frontier',
    )
    legend_handles['_dominated'] = mpatches.Patch(
        facecolor='gray', alpha=0.2, label='Dominated region',
    )

    # --- Trade-off arrow annotation ---
    ax_xlim = (min(all_hr) - 0.03, max(all_hr) + 0.08)
    ax_ylim = (min(all_other) - 0.02, max(all_other) + 0.06)
    ax.set_xlim(*ax_xlim)
    ax.set_ylim(*ax_ylim)
    arrow_x = ax_xlim[1] - 0.07
    arrow_y_hi = ax_ylim[1] - 0.01
    arrow_y_lo = ax_ylim[0] + 0.02
    ax.annotate(
        '', xy=(arrow_x, arrow_y_lo), xytext=(arrow_x, arrow_y_hi),
        arrowprops=dict(arrowstyle='->', color='#888', lw=1.5),
    )
    ax.annotate(
        '', xy=(arrow_x + 0.07, arrow_y_hi), xytext=(arrow_x, arrow_y_hi),
        arrowprops=dict(arrowstyle='->', color='#888', lw=1.5),
    )
    ax.text(arrow_x + 0.025, arrow_y_hi - 0.005, 'HR↑', fontsize=8, color='#555')
    ax.text(arrow_x - 0.005, arrow_y_hi - 0.02,
            f'{metric_key.split("@")[0]}↑', fontsize=8, color='#555', ha='right')
    ax.text(
        arrow_x - 0.01, (arrow_y_hi + arrow_y_lo) / 2,
        'trade-\noff', fontsize=7.5, color='#888', style='italic', ha='center',
    )

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(
        f'HR@10 vs {metric_key} Trade-off: Pareto Frontier across Hyperparameter Configs',
        fontweight='bold', pad=10,
    )
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    ordered_handles = (
        list(legend_handles[pk] for pk in param_colors if pk in legend_handles)
        + [legend_handles['_baseline'], legend_handles['_pareto'], legend_handles['_dominated']]
    )
    ax.legend(handles=ordered_handles, loc='lower left', framealpha=0.9,
              edgecolor='gray', fontsize=8.5)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    suffix = 'ndcg' if metric_pair == 'ndcg' else 'mrr'
    fig.savefig(output_dir / f'sensitivity_pareto_{suffix}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'sensitivity_pareto_{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: sensitivity_pareto_{suffix}.pdf/.png")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


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
    
    # Data points from ablation experiments (Toys, ablation0309.txt)
    # Format: (HR@10, MRR@10, label, color, marker)
    points = [
        (6.58, 3.72, 'Full (SE+Cross)', '#2E86AB', 'o'),
        (6.62, 3.70, '−SENet', '#28A745', 's'),
        (7.22, 3.31, '−Cross', '#FD7E14', '^'),
        (7.23, 3.31, '−SENet −Cross', '#DC3545', 'D'),
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
        help='Path to sensitivity raw data file (default: paper_sigir/sensitivity0309.txt).',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display figures interactively.',
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['tradeoff', 'pareto', 'full', 'legacy'],
        default='tradeoff',
        help='tradeoff: Hit vs NDCG/MRR seesaw; pareto: Pareto frontier; full: both; legacy: 4-metric plot.',
    )
    parser.add_argument(
        '--metric-pair',
        type=str,
        choices=['ndcg', 'mrr'],
        default='ndcg',
        help='Second metric for tradeoff/pareto: ndcg or mrr (default: ndcg).',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Also generate legacy scale-law and trade-off figures.',
    )
    args = parser.parse_args()

    second_metric = 'NDCG@10' if args.metric_pair == 'ndcg' else 'MRR@10'

    print("=" * 60)
    print("Generating Sensitivity Analysis Figures")
    print("=" * 60)

    if args.mode in ('tradeoff', 'full'):
        print("\n1. Creating Hit vs {} trade-off (seesaw) plot...".format(second_metric))
        create_sensitivity_tradeoff_plot(
            data_file=args.data_file,
            second_metric=second_metric,
            show=args.show,
        )

    if args.mode in ('pareto', 'full'):
        print("\n2. Creating Pareto frontier plot...")
        create_pareto_plot(
            data_file=args.data_file,
            metric_pair=args.metric_pair,
            show=args.show,
        )

    if args.mode == 'legacy':
        print("\n1. Creating legacy 4-metric sensitivity plot...")
        create_sensitivity_plot(data_file=args.data_file, show=args.show)

    if args.all:
        print("\n3. Creating Scale Law verification plot...")
        create_scale_law_plot()
        print("\n4. Creating HR-MRR trade-off plot (ablation)...")
        create_trade_off_plot()

    print("\n" + "=" * 60)
    print("✅ Figure generation completed!")
    print("=" * 60)
