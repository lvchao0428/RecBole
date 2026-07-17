#!/usr/bin/env python3
"""
Rank Transition Analysis Tool

分析 Cross 对 target item rank 的影响：
- 构建 rank-transition matrix（1 / 2-5 / 6-10 / 11-50 / >50）
- 统计 score entropy, Top-1~Top-10 margin, Top-K head/tail 占比
- 分 item-frequency bucket 输出

对应老师 0717 建议中的 per-user target rank transition 分析。

Usage:
    python scripts/analyze_rank_transition.py \
        --ckpt_nocross saved/ps_beauty_llm_nc_noboost_seed2025 \
        --ckpt_cross saved/ps_beauty_llm_cross_noboost_seed2025 \
        --dataset Amazon_Beauty \
        --output paper_recsys/rank_transition_beauty_llm.md
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RANK_BINS = [(1, 1), (2, 5), (6, 10), (11, 50), (51, float('inf'))]
BIN_LABELS = ["1", "2-5", "6-10", "11-50", ">50"]


def rank_to_bin(rank: int) -> int:
    for i, (lo, hi) in enumerate(RANK_BINS):
        if lo <= rank <= hi:
            return i
    return len(RANK_BINS) - 1


def score_entropy(scores: np.ndarray) -> float:
    """Normalized score entropy over Top-K items."""
    scores = scores - scores.max()
    probs = np.exp(scores) / np.exp(scores).sum()
    probs = probs[probs > 1e-12]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(scores))
    return float(entropy / max_entropy) if max_entropy > 0 else 0


def top_margin(scores_sorted: np.ndarray) -> dict:
    """Margin between Top-1 and Top-2, Top-1 and Top-10."""
    return {
        "top1_top2_margin": float(scores_sorted[0] - scores_sorted[1]) if len(scores_sorted) > 1 else 0,
        "top1_top10_margin": float(scores_sorted[0] - scores_sorted[9]) if len(scores_sorted) > 9 else 0,
    }


def item_bucket(count: int) -> str:
    if count == 0:
        return "unseen"
    elif count < 3:
        return "low"
    elif count < 10:
        return "mid"
    else:
        return "head"


def load_model_and_predict(ckpt_dir: str):
    """Load checkpoint and get predictions for test set."""
    from recbole.quick_start import load_data_and_model

    ckpt_path = Path(ckpt_dir)
    pth_files = list(ckpt_path.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth file in {ckpt_dir}")

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(str(pth_files[0]))
    model.eval()

    return config, model, dataset, test_data


def compute_ranks_and_scores(model, test_data, dataset, config, max_users=2000):
    """Compute target item ranks and score distributions for each test user."""
    from recbole.data.interaction import Interaction

    results = []
    device = config["device"]
    n_items = dataset.item_num

    count = 0
    for batch in test_data:
        if count >= max_users:
            break

        batch = batch.to(device)
        with torch.no_grad():
            scores = model.full_sort_predict(batch)

        scores = scores.view(-1, n_items)
        target_items = batch['item_id']

        for i in range(scores.size(0)):
            if count >= max_users:
                break
            user_scores = scores[i].cpu().numpy()
            target = target_items[i].item()

            sorted_indices = np.argsort(-user_scores)
            rank = int(np.where(sorted_indices == target)[0][0]) + 1

            top_k_scores = np.sort(user_scores)[::-1][:50]
            ent = score_entropy(top_k_scores[:10])
            margins = top_margin(top_k_scores)

            head_in_topk = 0
            results.append({
                "target_item": target,
                "rank": rank,
                "score_entropy_top10": ent,
                **margins,
                "top10_scores": top_k_scores[:10].tolist(),
            })
            count += 1

    return results


def build_transition_matrix(results_nc, results_cross):
    """Build rank transition matrix: rows=no-Cross bins, cols=+Cross bins."""
    n_bins = len(BIN_LABELS)
    matrix = np.zeros((n_bins, n_bins), dtype=int)

    for nc, cr in zip(results_nc, results_cross):
        bin_nc = rank_to_bin(nc["rank"])
        bin_cr = rank_to_bin(cr["rank"])
        matrix[bin_nc][bin_cr] += 1

    return matrix


def format_report(matrix, results_nc, results_cross, item_train_counts=None):
    """Generate markdown report."""
    lines = []
    lines.append("# Rank Transition Analysis\n")

    n_bins = len(BIN_LABELS)
    total = matrix.sum()

    lines.append("## Rank Transition Matrix (no-Cross → +Cross)\n")
    lines.append("| no-Cross \\ +Cross | " + " | ".join(BIN_LABELS) + " | Total |")
    lines.append("|" + "---|" * (n_bins + 2))
    for i in range(n_bins):
        row_total = matrix[i].sum()
        cells = [f"{matrix[i][j]}" for j in range(n_bins)]
        lines.append(f"| **{BIN_LABELS[i]}** | " + " | ".join(cells) + f" | {row_total} |")

    improved = sum(1 for nc, cr in zip(results_nc, results_cross) if cr["rank"] < nc["rank"])
    degraded = sum(1 for nc, cr in zip(results_nc, results_cross) if cr["rank"] > nc["rank"])
    unchanged = sum(1 for nc, cr in zip(results_nc, results_cross) if cr["rank"] == nc["rank"])

    lines.append(f"\n**Summary**: Improved {improved} ({improved/total:.1%}), "
                 f"Degraded {degraded} ({degraded/total:.1%}), "
                 f"Unchanged {unchanged} ({unchanged/total:.1%})")

    lines.append("\n## Score Distribution Comparison\n")
    nc_entropy = np.mean([r["score_entropy_top10"] for r in results_nc])
    cr_entropy = np.mean([r["score_entropy_top10"] for r in results_cross])
    nc_margin = np.mean([r["top1_top10_margin"] for r in results_nc])
    cr_margin = np.mean([r["top1_top10_margin"] for r in results_cross])

    lines.append(f"| Metric | no-Cross | +Cross | Δ |")
    lines.append(f"|--------|:--------:|:------:|:--:|")
    lines.append(f"| Score entropy (Top-10) | {nc_entropy:.4f} | {cr_entropy:.4f} | {cr_entropy-nc_entropy:+.4f} |")
    lines.append(f"| Top1-Top10 margin | {nc_margin:.4f} | {cr_margin:.4f} | {cr_margin-nc_margin:+.4f} |")

    if item_train_counts:
        lines.append("\n## By Item-Frequency Bucket\n")
        lines.append("| Bucket | #Samples | NC→CR Improved | NC→CR Degraded | Mean Rank NC | Mean Rank CR |")
        lines.append("|--------|:--------:|:--------------:|:--------------:|:------------:|:------------:|")

        bucket_data = defaultdict(list)
        for nc, cr in zip(results_nc, results_cross):
            cnt = item_train_counts.get(nc["target_item"], 0)
            b = item_bucket(cnt)
            bucket_data[b].append((nc["rank"], cr["rank"]))

        for b in ["head", "mid", "low", "unseen"]:
            pairs = bucket_data.get(b, [])
            if not pairs:
                continue
            n = len(pairs)
            imp = sum(1 for nc_r, cr_r in pairs if cr_r < nc_r)
            deg = sum(1 for nc_r, cr_r in pairs if cr_r > nc_r)
            mean_nc = np.mean([p[0] for p in pairs])
            mean_cr = np.mean([p[1] for p in pairs])
            lines.append(f"| {b} | {n} | {imp} ({imp/n:.0%}) | {deg} ({deg/n:.0%}) | {mean_nc:.1f} | {mean_cr:.1f} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_nocross", type=str, required=True)
    parser.add_argument("--ckpt_cross", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="Amazon_Beauty")
    parser.add_argument("--max_users", type=int, default=2000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("RANK TRANSITION ANALYSIS")
    print("=" * 70)

    print(f"\nLoading no-Cross checkpoint: {args.ckpt_nocross}")
    config_nc, model_nc, dataset_nc, test_nc = load_model_and_predict(args.ckpt_nocross)

    print(f"Loading +Cross checkpoint: {args.ckpt_cross}")
    config_cr, model_cr, dataset_cr, test_cr = load_model_and_predict(args.ckpt_cross)

    print(f"\nComputing ranks (no-Cross, max {args.max_users} users)...")
    results_nc = compute_ranks_and_scores(model_nc, test_nc, dataset_nc, config_nc, args.max_users)

    print(f"Computing ranks (+Cross, max {args.max_users} users)...")
    results_cross = compute_ranks_and_scores(model_cr, test_cr, dataset_cr, config_cr, args.max_users)

    n = min(len(results_nc), len(results_cross))
    results_nc = results_nc[:n]
    results_cross = results_cross[:n]

    print(f"\nBuilding transition matrix ({n} users)...")
    matrix = build_transition_matrix(results_nc, results_cross)

    report = format_report(matrix, results_nc, results_cross)

    output_file = args.output or f"paper_recsys/rank_transition_{args.dataset.replace('Amazon_', '').lower()}.md"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {output_file}")
    print("\n" + report)


if __name__ == "__main__":
    main()
