#!/usr/bin/env python3
"""
CKA with 3 independent item subsamples — report mean ± std.

Addresses advisor's 0803 requirement: "CKA用3次独立item子样本报告均值和方差"

Usage:
  python scripts/cka_subsample.py --dataset Amazon_Beauty
  python scripts/cka_subsample.py --dataset Amazon_Toys_and_Games
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VIEW_NAMES = ["Description", "Function", "Audience", "Style"]
N_TRIALS = 3
SUBSAMPLE_SIZE = 3000


def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, ord='fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord='fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, ord='fro') ** 2
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12))


def compute_effective_rank(emb: np.ndarray) -> float:
    if emb.dtype == np.float16:
        emb = emb.astype(np.float32)
    emb = emb - emb.mean(axis=0)
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
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    parser.add_argument("--subsample_size", type=int, default=SUBSAMPLE_SIZE)
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
        v = np.load(vp).astype(np.float32)
        views.append(v)
        print(f"Loaded view_{i} ({VIEW_NAMES[i]}): {v.shape}")

    tf_path = ds_path / "item_text_emb.base.ts.npy"
    if not tf_path.exists():
        tf_path = ds_path / "item_text_emb.base.npy"
    tf_emb = np.load(tf_path).astype(np.float32) if tf_path.exists() else None
    if tf_emb is not None:
        print(f"Loaded TF-IDF: {tf_emb.shape}")

    n_items = views[0].shape[0]
    n_sub = min(args.subsample_size, n_items)

    print(f"\n{'='*70}")
    print(f"CKA with {args.n_trials} independent subsamples of {n_sub} items")
    print(f"{'='*70}")

    results = {
        "dataset": args.dataset,
        "n_items_total": n_items,
        "subsample_size": n_sub,
        "n_trials": args.n_trials,
        "view_pairs": {},
        "tfidf_pairs": {},
        "effective_rank": {},
    }

    # View-view CKA
    print(f"\n{'Pair':<35} {'CKA values':<30} {'mean±std'}")
    print("-" * 80)

    for i in range(4):
        for j in range(i + 1, 4):
            cka_vals = []
            for trial in range(args.n_trials):
                rng = np.random.default_rng(seed=1000 + trial * 7)
                idx = rng.choice(n_items, size=n_sub, replace=False)
                cka = linear_CKA(views[i][idx], views[j][idx])
                cka_vals.append(cka)
            mean_cka = np.mean(cka_vals)
            std_cka = np.std(cka_vals)
            key = f"{VIEW_NAMES[i]} vs {VIEW_NAMES[j]}"
            results["view_pairs"][key] = {
                "values": cka_vals,
                "mean": float(mean_cka),
                "std": float(std_cka),
            }
            vals_str = ", ".join(f"{v:.4f}" for v in cka_vals)
            print(f"{key:<35} [{vals_str}]  {mean_cka:.4f}±{std_cka:.4f}")

    # TF-IDF vs views CKA
    if tf_emb is not None:
        print()
        for i in range(4):
            cka_vals = []
            for trial in range(args.n_trials):
                rng = np.random.default_rng(seed=2000 + trial * 7)
                idx = rng.choice(min(n_items, tf_emb.shape[0]), size=n_sub, replace=False)
                cka = linear_CKA(tf_emb[idx], views[i][idx])
                cka_vals.append(cka)
            mean_cka = np.mean(cka_vals)
            std_cka = np.std(cka_vals)
            key = f"TF-IDF vs {VIEW_NAMES[i]}"
            results["tfidf_pairs"][key] = {
                "values": cka_vals,
                "mean": float(mean_cka),
                "std": float(std_cka),
            }
            vals_str = ", ".join(f"{v:.4f}" for v in cka_vals)
            print(f"{key:<35} [{vals_str}]  {mean_cka:.4f}±{std_cka:.4f}")

    # Effective rank with subsamples
    print(f"\n{'='*70}")
    print(f"Effective Rank with {args.n_trials} subsamples")
    print(f"{'='*70}")
    print(f"\n{'Config':<30} {'Dim':>5} {'ER values':<25} {'mean±std'}")
    print("-" * 80)

    configs = [("Single view (avg)", views[0].shape[1])]
    for trial in range(args.n_trials):
        rng = np.random.default_rng(seed=3000 + trial * 7)
        idx = rng.choice(n_items, size=n_sub, replace=False)

        # Individual views (avg across views for this subsample)
        if trial == 0:
            view_ers = {f"view_{i}": [] for i in range(4)}

        for i in range(4):
            er = compute_effective_rank(views[i][idx])
            view_ers[f"view_{i}"].append(er)

    for i in range(4):
        ers = view_ers[f"view_{i}"]
        mean_er = np.mean(ers)
        std_er = np.std(ers)
        key = VIEW_NAMES[i]
        results["effective_rank"][key] = {
            "values": [float(e) for e in ers],
            "mean": float(mean_er),
            "std": float(std_er),
            "dim": views[i].shape[1],
            "ratio_mean": float(mean_er / views[i].shape[1]),
        }
        vals_str = ", ".join(f"{v:.1f}" for v in ers)
        print(f"{key:<30} {views[i].shape[1]:>5} [{vals_str}]  {mean_er:.1f}±{std_er:.1f}")

    # Concat 4 views
    concat_ers = []
    for trial in range(args.n_trials):
        rng = np.random.default_rng(seed=4000 + trial * 7)
        idx = rng.choice(n_items, size=n_sub, replace=False)
        concat = np.concatenate([v[idx] for v in views], axis=1)
        er = compute_effective_rank(concat)
        concat_ers.append(er)

    dim_concat = views[0].shape[1] * 4
    mean_er = np.mean(concat_ers)
    std_er = np.std(concat_ers)
    results["effective_rank"]["concat_4views"] = {
        "values": [float(e) for e in concat_ers],
        "mean": float(mean_er),
        "std": float(std_er),
        "dim": dim_concat,
        "ratio_mean": float(mean_er / dim_concat),
    }
    vals_str = ", ".join(f"{v:.1f}" for v in concat_ers)
    print(f"{'Concat 4 views':<30} {dim_concat:>5} [{vals_str}]  {mean_er:.1f}±{std_er:.1f}")

    # TF-IDF
    if tf_emb is not None:
        tf_ers = []
        for trial in range(args.n_trials):
            rng = np.random.default_rng(seed=5000 + trial * 7)
            idx = rng.choice(min(n_items, tf_emb.shape[0]), size=n_sub, replace=False)
            er = compute_effective_rank(tf_emb[idx])
            tf_ers.append(er)
        dim_tf = tf_emb.shape[1]
        mean_er = np.mean(tf_ers)
        std_er = np.std(tf_ers)
        results["effective_rank"]["tfidf"] = {
            "values": [float(e) for e in tf_ers],
            "mean": float(mean_er),
            "std": float(std_er),
            "dim": dim_tf,
            "ratio_mean": float(mean_er / dim_tf),
        }
        vals_str = ", ".join(f"{v:.1f}" for v in tf_ers)
        print(f"{'TF-IDF':<30} {dim_tf:>5} [{vals_str}]  {mean_er:.1f}±{std_er:.1f}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    view_cka_means = [v["mean"] for v in results["view_pairs"].values()]
    tfidf_cka_means = [v["mean"] for v in results["tfidf_pairs"].values()] if tf_emb is not None else []
    print(f"  Inter-view CKA range: {min(view_cka_means):.4f} - {max(view_cka_means):.4f}")
    if tfidf_cka_means:
        print(f"  TF-IDF vs LLM CKA range: {min(tfidf_cka_means):.4f} - {max(tfidf_cka_means):.4f}")
    concat_info = results["effective_rank"].get("concat_4views", {})
    if concat_info:
        print(f"  Concat eff rank / dim = {concat_info['ratio_mean']:.3f}")

    output_file = args.output or f"cka_subsample_{args.dataset.replace('Amazon_', '').lower()}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
