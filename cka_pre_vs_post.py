#!/usr/bin/env python3
"""
CKA redundancy audit: compare inter-view CKA/ER at two preprocessing stages:
  1. no_whiten (SVD-projected, centered, NOT whitened) — 64-d per view
  2. ts (SVD-projected, centered + whitened + TS-cutoff) — 64-d per view

If CKA is already high at stage 1, redundancy is NOT caused by whitening.
Since both stages share SVD projection from 4096-d, remaining redundancy
must originate from prompt semantic paraphrasing in the raw Qwen space.

Output: paper_recsys/cka_pre_vs_post_{dataset}.json
"""
import json
import sys
from pathlib import Path

import numpy as np

N_TRIALS = 3
SUBSAMPLE_SIZE = 3000


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy + 1e-12))


def effective_rank(X: np.ndarray) -> float:
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def analyze_stage(view_paths: list[Path], n_items: int, label: str):
    views = [np.load(str(p), mmap_mode="r") for p in view_paths]
    n_views = len(views)
    dim = views[0].shape[1]

    view_names = ["Description", "Function", "Audience", "Style"]
    pair_names = []
    for i in range(n_views):
        for j in range(i + 1, n_views):
            pair_names.append(f"{view_names[i]} vs {view_names[j]}")

    cka_results = {pn: [] for pn in pair_names}
    er_concat_results = []
    er_single_results = {view_names[i]: [] for i in range(n_views)}

    for trial in range(N_TRIALS):
        rng = np.random.RandomState(42 + trial * 7)
        idx = rng.choice(n_items, size=min(SUBSAMPLE_SIZE, n_items), replace=False)
        idx.sort()

        sub_views = [np.array(v[idx], dtype=np.float32) for v in views]

        pair_idx = 0
        for i in range(n_views):
            for j in range(i + 1, n_views):
                c = linear_cka(sub_views[i], sub_views[j])
                cka_results[pair_names[pair_idx]].append(c)
                pair_idx += 1

        concat = np.concatenate(sub_views, axis=1)
        er_concat_results.append(effective_rank(concat))

        for i in range(n_views):
            er_single_results[view_names[i]].append(effective_rank(sub_views[i]))

    result = {
        "stage": label,
        "dim_per_view": dim,
        "n_trials": N_TRIALS,
        "subsample_size": SUBSAMPLE_SIZE,
        "cka_pairs": {},
        "effective_rank": {},
    }

    for pn, vals in cka_results.items():
        result["cka_pairs"][pn] = {
            "values": vals,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    result["effective_rank"]["concat_4views"] = {
        "values": er_concat_results,
        "mean": float(np.mean(er_concat_results)),
        "std": float(np.std(er_concat_results)),
        "dim": dim * n_views,
        "ratio_mean": float(np.mean(er_concat_results)) / (dim * n_views),
    }

    for vn, vals in er_single_results.items():
        result["effective_rank"][vn] = {
            "values": vals,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "dim": dim,
            "ratio_mean": float(np.mean(vals)) / dim,
        }

    return result


def main():
    datasets = {
        "Amazon_Beauty": 259205,
        "Amazon_Toys_and_Games": 336073,
    }

    for ds_name, n_items in datasets.items():
        base = Path("dataset") / ds_name
        out_path = Path("paper_recsys") / f"cka_pre_vs_post_{ds_name.lower().replace('amazon_','').replace('_and_','_')}.json"

        # Stage 1: no_whiten (SVD + center, no whiten)
        no_whiten_dir = base / "qwen2.5_7b_4views_no_whiten"
        # Stage 2: ts (SVD + center + whiten + TS-cutoff)
        ts_dir = base / "qwen2.5_7b_4views_ts"

        stages = []
        for label, vdir in [("no_whiten", no_whiten_dir), ("ts_whitened", ts_dir)]:
            view_paths = [vdir / f"view_{i}.npy" for i in range(4)]
            if all(p.exists() for p in view_paths):
                print(f"[{ds_name}] Analyzing stage={label} from {vdir}")
                stages.append(analyze_stage(view_paths, n_items, label))
            else:
                print(f"[{ds_name}] SKIP stage={label}: views not found in {vdir}")

        if stages:
            output = {
                "dataset": ds_name,
                "n_items": n_items,
                "stages": stages,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"  -> Wrote {out_path}")


if __name__ == "__main__":
    main()
