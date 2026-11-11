#!/usr/bin/env python3
"""
View Redundancy Analysis (Pairwise Cosine + CKA)

深度分析 multi-view 的冗余度，对应老师 0717 建议：
"先做 pairwise cosine/CKA、奇异值谱或 effective rank、leave-one-view-out，
 判断到底是 view redundancy 还是容量不足"

Usage:
    python scripts/analyze_view_redundancy.py --dataset Amazon_Beauty
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VIEW_NAMES = ["Description", "Function", "Audience", "Style"]


def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear Centered Kernel Alignment between two representations."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    hsic_xy = np.linalg.norm(X.T @ Y, ord='fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord='fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, ord='fro') ** 2

    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12))


def pairwise_cosine_stats(X: np.ndarray, Y: np.ndarray, n_samples: int = 2000) -> dict:
    """Per-item cosine similarity between two views."""
    n = min(X.shape[0], Y.shape[0], n_samples)
    X_sub = X[:n].astype(np.float32)
    Y_sub = Y[:n].astype(np.float32)

    x_norm = X_sub / (np.linalg.norm(X_sub, axis=1, keepdims=True) + 1e-8)
    y_norm = Y_sub / (np.linalg.norm(Y_sub, axis=1, keepdims=True) + 1e-8)

    cos_sims = np.sum(x_norm * y_norm, axis=1)

    return {
        "mean": float(np.mean(cos_sims)),
        "std": float(np.std(cos_sims)),
        "median": float(np.median(cos_sims)),
        "q25": float(np.percentile(cos_sims, 25)),
        "q75": float(np.percentile(cos_sims, 75)),
    }


def compute_effective_rank(emb: np.ndarray) -> float:
    """Effective rank via Shannon entropy of singular values."""
    if emb.dtype == np.float16:
        emb = emb.astype(np.float32)
    emb = emb - emb.mean(axis=0)
    n = min(emb.shape[0], 5000)
    if emb.shape[0] > n:
        emb = emb[np.random.choice(emb.shape[0], n, replace=False)]

    _, S, _ = np.linalg.svd(emb, full_matrices=False)
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    p = S / S.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Amazon_Beauty")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.base_dir is None:
        args.base_dir = str(ROOT / "dataset")

    ds_path = Path(args.base_dir) / args.dataset

    views_dir = ds_path / "qwen2.5_7b_4views_ts"
    if not views_dir.exists():
        views_dir = ds_path / "qwen2.5_7b_4views"
    if not views_dir.exists():
        print(f"ERROR: Views directory not found in {ds_path}")
        sys.exit(1)

    views = []
    for i in range(4):
        vp = views_dir / f"view_{i}.npy"
        if vp.exists():
            v = np.load(vp)
            if v.dtype == np.float16:
                v = v.astype(np.float32)
            views.append(v)
            print(f"Loaded view_{i} ({VIEW_NAMES[i]}): {v.shape}")
        else:
            print(f"WARNING: view_{i}.npy not found")

    tf_path = ds_path / "item_text_emb.base.ts.npy"
    if not tf_path.exists():
        tf_path = ds_path / "item_text_emb.base.npy"
    tf_emb = np.load(tf_path).astype(np.float32) if tf_path.exists() else None
    if tf_emb is not None:
        print(f"Loaded TF-IDF base: {tf_emb.shape}")

    results = {"dataset": args.dataset, "n_views": len(views)}

    print("\n" + "=" * 70)
    print("PAIRWISE COSINE SIMILARITY (per-item)")
    print("=" * 70)
    print(f"\n{'Pair':<35} {'Mean':>8} {'Std':>8} {'Median':>8}")
    print("-" * 65)

    cos_matrix = {}
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            stats = pairwise_cosine_stats(views[i], views[j])
            key = f"view_{i}_vs_view_{j}"
            cos_matrix[key] = stats
            label = f"{VIEW_NAMES[i]} vs {VIEW_NAMES[j]}"
            print(f"{label:<35} {stats['mean']:>8.4f} {stats['std']:>8.4f} {stats['median']:>8.4f}")

    if tf_emb is not None and tf_emb.shape[1] == views[0].shape[1]:
        for i in range(len(views)):
            stats = pairwise_cosine_stats(tf_emb, views[i])
            key = f"tfidf_vs_view_{i}"
            cos_matrix[key] = stats
            label = f"TF-IDF vs {VIEW_NAMES[i]}"
            print(f"{label:<35} {stats['mean']:>8.4f} {stats['std']:>8.4f} {stats['median']:>8.4f}")
    elif tf_emb is not None:
        print(f"\n  (Skipping TF-IDF vs view cosine: dim mismatch {tf_emb.shape[1]} vs {views[0].shape[1]})")

    results["pairwise_cosine"] = cos_matrix

    print("\n" + "=" * 70)
    print("LINEAR CKA (representation similarity)")
    print("=" * 70)
    print(f"\n{'Pair':<35} {'CKA':>8}")
    print("-" * 45)

    cka_matrix = {}
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            cka = linear_CKA(views[i], views[j])
            key = f"view_{i}_vs_view_{j}"
            cka_matrix[key] = cka
            label = f"{VIEW_NAMES[i]} vs {VIEW_NAMES[j]}"
            print(f"{label:<35} {cka:>8.4f}")

    if tf_emb is not None:
        for i in range(len(views)):
            n = min(tf_emb.shape[0], views[i].shape[0], 5000)
            cka = linear_CKA(tf_emb[:n], views[i][:n])
            key = f"tfidf_vs_view_{i}"
            cka_matrix[key] = cka
            label = f"TF-IDF vs {VIEW_NAMES[i]}"
            print(f"{label:<35} {cka:>8.4f}")

    results["cka"] = cka_matrix

    print("\n" + "=" * 70)
    print("EFFECTIVE RANK (per-view and combinations)")
    print("=" * 70)

    np.random.seed(42)
    rank_results = {}
    print(f"\n{'Config':<30} {'Dim':>5} {'Eff Rank':>10} {'Ratio':>7}")
    print("-" * 55)

    for i, v in enumerate(views):
        er = compute_effective_rank(v)
        key = f"view_{i}"
        rank_results[key] = {"eff_rank": er, "dim": v.shape[1], "ratio": er / v.shape[1]}
        print(f"{VIEW_NAMES[i]:<30} {v.shape[1]:>5} {er:>10.1f} {er/v.shape[1]:>7.3f}")

    if len(views) == 4:
        concat_all = np.concatenate(views, axis=1)
        er = compute_effective_rank(concat_all)
        rank_results["concat_4views"] = {"eff_rank": er, "dim": concat_all.shape[1], "ratio": er / concat_all.shape[1]}
        print(f"{'Concat 4 views':<30} {concat_all.shape[1]:>5} {er:>10.1f} {er/concat_all.shape[1]:>7.3f}")

        for drop_i in range(4):
            remaining = [views[j] for j in range(4) if j != drop_i]
            concat_3 = np.concatenate(remaining, axis=1)
            er = compute_effective_rank(concat_3)
            key = f"drop_view_{drop_i}"
            rank_results[key] = {"eff_rank": er, "dim": concat_3.shape[1], "ratio": er / concat_3.shape[1]}
            print(f"{'Drop '+VIEW_NAMES[drop_i]:<30} {concat_3.shape[1]:>5} {er:>10.1f} {er/concat_3.shape[1]:>7.3f}")

    if tf_emb is not None:
        er = compute_effective_rank(tf_emb)
        rank_results["tfidf"] = {"eff_rank": er, "dim": tf_emb.shape[1], "ratio": er / tf_emb.shape[1]}
        print(f"{'TF-IDF':<30} {tf_emb.shape[1]:>5} {er:>10.1f} {er/tf_emb.shape[1]:>7.3f}")

        if len(views) == 4:
            full = np.concatenate([tf_emb] + views, axis=1)
            er = compute_effective_rank(full)
            rank_results["tfidf_plus_4views"] = {"eff_rank": er, "dim": full.shape[1], "ratio": er / full.shape[1]}
            print(f"{'TF-IDF + 4 views':<30} {full.shape[1]:>5} {er:>10.1f} {er/full.shape[1]:>7.3f}")

    results["effective_rank"] = rank_results

    print("\n" + "=" * 70)
    print("REDUNDANCY DIAGNOSIS")
    print("=" * 70)

    high_cos_pairs = [(k, v["mean"]) for k, v in cos_matrix.items() if v["mean"] > 0.3 and "tfidf" not in k]
    high_cka_pairs = [(k, v) for k, v in cka_matrix.items() if v > 0.3 and "tfidf" not in k]

    if high_cos_pairs or high_cka_pairs:
        print("\n  REDUNDANCY DETECTED:")
        for k, v in high_cos_pairs:
            print(f"    - {k}: cosine={v:.4f} (>0.3)")
        for k, v in high_cka_pairs:
            print(f"    - {k}: CKA={v:.4f} (>0.3)")
        print("\n  Conclusion: Prompt-based multi-view contains significant redundancy.")
        print("  The 4 views from the same title+encoder are partially semantic restatements.")
        results["diagnosis"] = "REDUNDANT"
    else:
        print("\n  NO SIGNIFICANT REDUNDANCY detected between views.")
        results["diagnosis"] = "COMPLEMENTARY"

    concat_ratio = rank_results.get("concat_4views", {}).get("ratio", 1.0)
    if concat_ratio < 0.7:
        print(f"  DIMENSION COLLAPSE: 4-view concat effective rank ratio = {concat_ratio:.3f} (<0.7)")
        results["collapse"] = True
    else:
        results["collapse"] = False

    output_file = args.output or f"view_redundancy_{args.dataset.replace('Amazon_', '').lower()}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
