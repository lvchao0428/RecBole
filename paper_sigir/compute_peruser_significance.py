# -*- coding: utf-8 -*-
"""
Per-user paired t-test for the main table (Table 2).

Usage:
    python compute_peruser_significance.py \
        --model_a  saved/peruser/beauty_tfidf_llm.npy \
        --model_b  saved/peruser/beauty_mv_7b.npy \
        --label_a  "TF-IDF+LLM" \
        --label_b  "MV-Align(7B)"

Each .npy file is a rec.topk tensor of shape (n_users, max_topk + 1) saved
by RecBole's trainer.evaluate(save_peruser_topk_path=...).

The script computes per-user Hit@k, NDCG@k, MRR@k from the raw hit
indicator matrix and runs a two-sided paired t-test between two models
on the *same* set of test users.

Output:
    - Per-metric: mean_A, mean_B, delta, t-stat, p-value, significance
    - Footnote-ready summary for paper
"""

import argparse
import numpy as np
from scipy import stats


def load_topk(path):
    """Load rec.topk tensor: (n_users, max_topk + 1)."""
    arr = np.load(path)
    max_k = arr.shape[1] - 1
    pos_index = arr[:, :max_k].astype(np.float64)
    pos_len = arr[:, -1].astype(np.float64)
    return pos_index, pos_len


def peruser_hit(pos_index, k):
    """Per-user Hit@k: 1 if any hit in top-k positions."""
    cumhits = np.cumsum(pos_index[:, :k], axis=1)
    return (cumhits[:, k - 1] > 0).astype(np.float64)


def peruser_mrr(pos_index, k):
    """Per-user MRR@k: 1/rank of first relevant item, 0 if none."""
    topk = pos_index[:, :k]
    has_hit = topk.any(axis=1)
    first_hit = topk.argmax(axis=1)  # index of first 1 (or 0 if all-zero)
    mrr = np.where(has_hit, 1.0 / (first_hit + 1), 0.0)
    return mrr


def peruser_ndcg(pos_index, pos_len, k):
    """Per-user NDCG@k."""
    topk = pos_index[:, :k]
    ranks = np.arange(1, k + 1, dtype=np.float64)
    dcg = np.sum(topk / np.log2(ranks + 1), axis=1)

    idcg_len = np.minimum(pos_len, k).astype(int)
    idcg = np.zeros(len(pos_len), dtype=np.float64)
    for i, l in enumerate(idcg_len):
        if l > 0:
            idcg[i] = np.sum(1.0 / np.log2(np.arange(1, l + 1, dtype=np.float64) + 1))

    return np.where(idcg > 0, dcg / idcg, 0.0)


def paired_ttest(vals_a, vals_b):
    """Two-sided paired t-test. Returns (t_stat, p_value)."""
    t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
    return t_stat, p_val


def sig_label(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def main():
    parser = argparse.ArgumentParser(
        description="Per-user paired t-test between two models"
    )
    parser.add_argument("--model_a", required=True, help="Path to baseline rec.topk .npy")
    parser.add_argument("--model_b", required=True, help="Path to proposed rec.topk .npy")
    parser.add_argument("--label_a", default="Baseline")
    parser.add_argument("--label_b", default="Proposed")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20],
                        help="Top-K cutoffs to evaluate")
    args = parser.parse_args()

    pos_index_a, pos_len_a = load_topk(args.model_a)
    pos_index_b, pos_len_b = load_topk(args.model_b)

    n_a, n_b = pos_index_a.shape[0], pos_index_b.shape[0]
    if n_a != n_b:
        raise ValueError(
            f"User count mismatch: {args.label_a} has {n_a}, "
            f"{args.label_b} has {n_b}. "
            "Both models must be evaluated on the same test set."
        )

    print(f"{'=' * 80}")
    print(f"Per-user paired t-test:  {args.label_a}  vs  {args.label_b}")
    print(f"Number of test users: {n_a}")
    print(f"{'=' * 80}\n")

    metrics = [("Hit", peruser_hit), ("NDCG", peruser_ndcg), ("MRR", peruser_mrr)]

    for k in args.k:
        print(f"--- K = {k} ---")
        print(f"  {'Metric':<12} {'Mean_A':>8} {'Mean_B':>8} {'Delta':>8} "
              f"{'t-stat':>10} {'p-value':>12}  Sig")
        print(f"  {'-' * 72}")

        for name, fn in metrics:
            if name == "NDCG":
                va = fn(pos_index_a, pos_len_a, k)
                vb = fn(pos_index_b, pos_len_b, k)
            else:
                va = fn(pos_index_a, k)
                vb = fn(pos_index_b, k)

            mean_a = va.mean()
            mean_b = vb.mean()
            delta = mean_b - mean_a
            t, p = paired_ttest(vb, va)

            print(f"  {name+'@'+str(k):<12} {mean_a*100:>7.2f}% {mean_b*100:>7.2f}% "
                  f"{delta*100:>+7.3f}% {t:>10.3f} {p:>12.2e}  {sig_label(p)}")
        print()

    print("Footnote for paper:")
    print('  "$^{*}$ $p{<}0.05$, $^{**}$ $p{<}0.01$; '
          'paired $t$-test on per-user metric differences '
          f'({n_a} test users), '
          f'{args.label_b} vs.\\ {args.label_a}."')


if __name__ == "__main__":
    main()
