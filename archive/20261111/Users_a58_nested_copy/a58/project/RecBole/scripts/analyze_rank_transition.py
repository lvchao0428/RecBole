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


RANK_BINS = [(1, 1), (2, 5), (6, 10), (11, 50), (51, float("inf"))]
BIN_LABELS = ["1", "2-5", "6-10", "11-50", ">50"]
TOPK_LIST = (5, 10, 20)


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
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def top_margin(scores_sorted: np.ndarray) -> dict:
    """Margin between Top-1 and Top-2, Top-1 and Top-10."""
    return {
        "top1_top2_margin": float(scores_sorted[0] - scores_sorted[1]) if len(scores_sorted) > 1 else 0.0,
        "top1_top10_margin": float(scores_sorted[0] - scores_sorted[9]) if len(scores_sorted) > 9 else 0.0,
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
    pth_files = sorted(ckpt_path.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not pth_files:
        raise FileNotFoundError(f"No .pth file in {ckpt_dir}")
    # Prefer Phase-B (usually last / larger) checkpoint
    ckpt_file = pth_files[-1]

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(str(ckpt_file))
    model.eval()

    return config, model, dataset, train_data, test_data


def get_train_item_counts(train_data, dataset, config) -> dict:
    """Map internal item_id → train interaction count (PAD=0 skipped)."""
    iid_field = config["ITEM_ID_FIELD"]
    items = train_data._dataset.inter_feat[iid_field].numpy()
    counts = defaultdict(int)
    for iid in items:
        if iid > 0:
            counts[int(iid)] += 1
    return dict(counts)


def compute_ranks_and_scores(model, test_data, dataset, config, item_train_counts, max_users=2000):
    """Compute target ranks, score stats, and Top-K popularity shares."""
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
        target_items = batch["item_id"]

        for i in range(scores.size(0)):
            if count >= max_users:
                break
            user_scores = scores[i].cpu().numpy()
            target = int(target_items[i].item())

            # mask PAD
            user_scores[0] = -np.inf
            sorted_indices = np.argsort(-user_scores)
            rank = int(np.where(sorted_indices == target)[0][0]) + 1

            top_k_scores = user_scores[sorted_indices[:50]]
            ent = score_entropy(top_k_scores[:10])
            margins = top_margin(top_k_scores)

            topk_share = {}
            for k in TOPK_LIST:
                top_items = sorted_indices[:k]
                bucket_counts = defaultdict(int)
                for iid in top_items:
                    b = item_bucket(item_train_counts.get(int(iid), 0))
                    bucket_counts[b] += 1
                for b in ("head", "mid", "low", "unseen"):
                    topk_share[f"top{k}_{b}_share"] = bucket_counts[b] / float(k)

            results.append({
                "target_item": target,
                "rank": rank,
                "score_entropy_top10": ent,
                **margins,
                **topk_share,
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


def _mean_key(results, key):
    vals = [r[key] for r in results if key in r]
    return float(np.mean(vals)) if vals else 0.0


def format_report(matrix, results_nc, results_cross, item_train_counts=None):
    """Generate markdown report."""
    lines = []
    lines.append("# Rank Transition Analysis\n")

    n_bins = len(BIN_LABELS)
    total = int(matrix.sum())
    if total == 0:
        return "# Rank Transition Analysis\n\n(no samples)\n"

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

    lines.append(
        f"\n**Summary**: Improved {improved} ({improved / total:.1%}), "
        f"Degraded {degraded} ({degraded / total:.1%}), "
        f"Unchanged {unchanged} ({unchanged / total:.1%})"
    )

    lines.append("\n## Score Distribution Comparison\n")
    nc_entropy = _mean_key(results_nc, "score_entropy_top10")
    cr_entropy = _mean_key(results_cross, "score_entropy_top10")
    nc_margin = _mean_key(results_nc, "top1_top10_margin")
    cr_margin = _mean_key(results_cross, "top1_top10_margin")

    lines.append("| Metric | no-Cross | +Cross | Δ |")
    lines.append("|--------|:--------:|:------:|:--:|")
    lines.append(
        f"| Score entropy (Top-10) | {nc_entropy:.4f} | {cr_entropy:.4f} | {cr_entropy - nc_entropy:+.4f} |"
    )
    lines.append(
        f"| Top1-Top10 margin | {nc_margin:.4f} | {cr_margin:.4f} | {cr_margin - nc_margin:+.4f} |"
    )

    lines.append("\n## Top-K Head / Tail Share (train-frequency buckets)\n")
    lines.append(
        "Fraction of recommended items in Top-K that fall into each train-frequency bucket "
        "(head≥10, mid 3–9, low 1–2, unseen=0). Higher head share ⇒ more popularity bias.\n"
    )
    lines.append("| K | Bucket | no-Cross | +Cross | Δ |")
    lines.append("|:-:|:------:|:--------:|:------:|:--:|")
    for k in TOPK_LIST:
        for b in ("head", "mid", "low", "unseen"):
            key = f"top{k}_{b}_share"
            nc = _mean_key(results_nc, key)
            cr = _mean_key(results_cross, key)
            lines.append(f"| {k} | {b} | {nc:.3f} | {cr:.3f} | {cr - nc:+.3f} |")

    if item_train_counts:
        lines.append("\n## By Target Item-Frequency Bucket\n")
        lines.append(
            "| Bucket | #Samples | NC→CR Improved | NC→CR Degraded | Mean Rank NC | Mean Rank CR |"
        )
        lines.append(
            "|--------|:--------:|:--------------:|:--------------:|:------------:|:------------:|"
        )

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
            lines.append(
                f"| {b} | {n} | {imp} ({imp / n:.0%}) | {deg} ({deg / n:.0%}) | {mean_nc:.1f} | {mean_cr:.1f} |"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_nocross", type=str, required=True)
    parser.add_argument("--ckpt_cross", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="Amazon_Beauty")
    parser.add_argument("--max_users", type=int, default=2000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--json_out", type=str, default=None,
                        help="Optional JSON dump of aggregate Top-K shares")
    args = parser.parse_args()

    print("=" * 70)
    print("RANK TRANSITION + TOP-K HEAD/TAIL SHARE")
    print("=" * 70)

    print(f"\nLoading no-Cross checkpoint: {args.ckpt_nocross}")
    config_nc, model_nc, dataset_nc, train_nc, test_nc = load_model_and_predict(args.ckpt_nocross)
    item_counts = get_train_item_counts(train_nc, dataset_nc, config_nc)
    print(f"  train items with count>0: {len(item_counts)}")

    print(f"Loading +Cross checkpoint: {args.ckpt_cross}")
    config_cr, model_cr, dataset_cr, train_cr, test_cr = load_model_and_predict(args.ckpt_cross)

    print(f"\nComputing ranks (no-Cross, max {args.max_users} users)...")
    results_nc = compute_ranks_and_scores(
        model_nc, test_nc, dataset_nc, config_nc, item_counts, args.max_users
    )

    print(f"Computing ranks (+Cross, max {args.max_users} users)...")
    results_cross = compute_ranks_and_scores(
        model_cr, test_cr, dataset_cr, config_cr, item_counts, args.max_users
    )

    n = min(len(results_nc), len(results_cross))
    results_nc = results_nc[:n]
    results_cross = results_cross[:n]

    print(f"\nBuilding transition matrix ({n} users)...")
    matrix = build_transition_matrix(results_nc, results_cross)

    report = format_report(matrix, results_nc, results_cross, item_counts)

    output_file = args.output or f"paper_recsys/rank_transition_{args.dataset.replace('Amazon_', '').lower()}.md"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(report)

    json_out = args.json_out or str(Path(output_file).with_suffix(".json"))
    summary = {
        "n_users": n,
        "ckpt_nocross": args.ckpt_nocross,
        "ckpt_cross": args.ckpt_cross,
        "topk_share": {},
    }
    for k in TOPK_LIST:
        summary["topk_share"][f"top{k}"] = {}
        for b in ("head", "mid", "low", "unseen"):
            key = f"top{k}_{b}_share"
            summary["topk_share"][f"top{k}"][b] = {
                "nocross": _mean_key(results_nc, key),
                "cross": _mean_key(results_cross, key),
                "delta": _mean_key(results_cross, key) - _mean_key(results_nc, key),
            }
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nReport saved to: {output_file}")
    print(f"JSON saved to:   {json_out}")
    print("\n" + report)


if __name__ == "__main__":
    main()
