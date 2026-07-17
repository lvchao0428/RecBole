#!/usr/bin/env python3
"""
Leave-One-View-Out Analysis

消融实验：每次去掉一个 view，对比 MV 完整模型，判断各 view 的边际贡献。
对应老师 0717 建议："做 leave-one-view-out，判断到底是 view redundancy 还是容量不足"

实现方式：修改模型加载时的 text_view_indices 配置，重新推理。

Usage:
    python scripts/analyze_leave_one_view_out.py \
        --ckpt saved/ps_beauty_mv_nc_noboost_seed2025 \
        --dataset Amazon_Beauty \
        --output paper_recsys/leave_one_view_out_beauty.md
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


VIEW_NAMES = ["Description", "Function", "Audience", "Style"]


def evaluate_with_views(model, test_data, config, view_indices):
    """Evaluate model with subset of views."""
    original_indices = getattr(model, 'text_view_indices', list(range(4)))
    model.text_view_indices = view_indices

    from recbole.evaluator import Evaluator
    evaluator = Evaluator(config)

    model.eval()
    device = config["device"]

    all_results = []
    for batch in test_data:
        batch = batch.to(device)
        with torch.no_grad():
            scores = model.full_sort_predict(batch)
        all_results.append((batch, scores))

    model.text_view_indices = original_indices
    return all_results


def compute_mrr_at_k(results, dataset, k=10):
    """Compute MRR@K from prediction results."""
    n_items = dataset.item_num
    mrr_sum = 0.0
    count = 0

    for batch, scores in results:
        scores = scores.view(-1, n_items)
        target_items = batch['item_id']

        for i in range(scores.size(0)):
            user_scores = scores[i]
            target = target_items[i].item()

            sorted_indices = torch.argsort(user_scores, descending=True)
            rank = int((sorted_indices == target).nonzero(as_tuple=True)[0].item()) + 1

            if rank <= k:
                mrr_sum += 1.0 / rank
            count += 1

    return mrr_sum / count if count > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="Amazon_Beauty")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    from recbole.quick_start import load_data_and_model

    ckpt_path = Path(args.ckpt)
    pth_files = list(ckpt_path.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth file in {args.ckpt}")

    print("=" * 60)
    print("LEAVE-ONE-VIEW-OUT ANALYSIS")
    print("=" * 60)

    print(f"\nLoading checkpoint: {pth_files[0]}")
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(str(pth_files[0]))
    model.eval()

    print(f"\nDataset: {args.dataset}, Test users: ~{sum(1 for _ in test_data)}")

    all_views = list(range(4))

    print("\nEvaluating full model (all 4 views)...")
    results_full = evaluate_with_views(model, test_data, config, all_views)
    mrr_full = compute_mrr_at_k(results_full, dataset)
    print(f"  Full MV MRR@10 = {mrr_full:.6f}")

    lines = ["# Leave-One-View-Out Analysis\n"]
    lines.append(f"Checkpoint: `{args.ckpt}`\n")
    lines.append(f"| Config | Views Used | MRR@10 | Δ vs Full | Contribution |")
    lines.append(f"|--------|-----------|:------:|:---------:|:------------:|")
    lines.append(f"| **Full (4 views)** | 0,1,2,3 | **{mrr_full:.6f}** | — | — |")

    view_results = {}
    for drop_idx in range(4):
        remaining = [v for v in all_views if v != drop_idx]
        print(f"\nEvaluating without view_{drop_idx} ({VIEW_NAMES[drop_idx]})...")
        results = evaluate_with_views(model, test_data, config, remaining)
        mrr = compute_mrr_at_k(results, dataset)
        delta = mrr - mrr_full
        contribution = -delta
        view_results[drop_idx] = {"mrr": mrr, "delta": delta, "contribution": contribution}
        print(f"  MRR@10 = {mrr:.6f} (Δ = {delta:+.6f})")

        remaining_str = ",".join(str(v) for v in remaining)
        lines.append(f"| Drop view_{drop_idx} ({VIEW_NAMES[drop_idx]}) | {remaining_str} | "
                     f"{mrr:.6f} | {delta:+.6f} | {contribution:+.6f} |")

    lines.append("\n## View Contribution Ranking\n")
    sorted_views = sorted(view_results.items(), key=lambda x: x[1]["contribution"], reverse=True)
    lines.append("| Rank | View | Name | Contribution (drop hurts by) |")
    lines.append("|:----:|:----:|------|:----------------------------:|")
    for rank, (idx, data) in enumerate(sorted_views, 1):
        lines.append(f"| {rank} | view_{idx} | {VIEW_NAMES[idx]} | {data['contribution']:+.6f} |")

    lines.append("\n## Interpretation\n")
    max_contrib = sorted_views[0][1]["contribution"]
    min_contrib = sorted_views[-1][1]["contribution"]
    if max_contrib - min_contrib < 0.0005:
        lines.append("- All views have similar contribution → **high redundancy** (consistent with SVD diagnosis)")
    else:
        lines.append(f"- Most important: view_{sorted_views[0][0]} ({VIEW_NAMES[sorted_views[0][0]]})")
        lines.append(f"- Least important: view_{sorted_views[-1][0]} ({VIEW_NAMES[sorted_views[-1][0]]})")

    report = "\n".join(lines)
    output_file = args.output or f"paper_recsys/leave_one_view_out_{args.dataset.replace('Amazon_', '').lower()}.md"
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()
