#!/usr/bin/env python3
"""
Sensitivity Analysis Plot for MV-Align Paper (figures/ wrapper)

Delegates to scripts/plot_sensitivity.py which reads from sensitivity0309.txt.
Generates Hit vs NDCG/MRR trade-off (seesaw 跷跷板) and Pareto frontier (帕累托边缘).

Usage:
    python plot_sensitivity.py [--mode tradeoff|pareto|full] [--metric-pair ndcg|mrr]
    
Output (default mode=tradeoff):
    sensitivity_tradeoff_ndcg.pdf/.png  (Hit vs NDCG dual-axis per hyperparam)
    sensitivity_pareto_ndcg.pdf/.png    (if --mode pareto or full)
"""
import sys
from pathlib import Path

# Add scripts dir to path so we can import plot_sensitivity
_scripts_dir = Path(__file__).resolve().parent.parent / 'scripts'
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

# Change to figures dir so outputs go there
import os
os.chdir(Path(__file__).resolve().parent)

from plot_sensitivity import (
    create_sensitivity_tradeoff_plot,
    create_pareto_plot,
    create_sensitivity_plot,
    DEFAULT_DATA_FILE,
)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-file', default=str(DEFAULT_DATA_FILE))
    p.add_argument('--show', action='store_true')
    p.add_argument('--mode', choices=['tradeoff', 'pareto', 'full', 'legacy'], default='full')
    p.add_argument('--metric-pair', choices=['ndcg', 'mrr'], default='ndcg')
    args = p.parse_args()

    second = 'NDCG@10' if args.metric_pair == 'ndcg' else 'MRR@10'
    if args.mode in ('tradeoff', 'full'):
        create_sensitivity_tradeoff_plot(args.data_file, second_metric=second, show=args.show)
    if args.mode in ('pareto', 'full'):
        create_pareto_plot(args.data_file, metric_pair=args.metric_pair, show=args.show)
    if args.mode == 'legacy':
        create_sensitivity_plot(args.data_file, show=args.show)
