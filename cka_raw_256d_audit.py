#!/usr/bin/env python3
"""
CKA/ER audit on the highest-dimensional available pre-whitening embeddings (256-d).

This provides evidence that inter-view redundancy is NOT an artifact of whitening
or low-rank projection to 64-d. The 256-d embeddings are:
  - Post-SVD (reduced from raw 4096-d Qwen hidden states)
  - Centered but NOT whitened (no ZCA/PCA decorrelation applied)

If pairwise CKA among 4 views is already high at 256-d (pre-whitening),
it strongly indicates redundancy originates in the Qwen encoder itself,
not from the SVD/whitening pipeline.

Also computes CKA between the 256-d single-view (base) and 256-d multiview
(concatenated all views then re-SVD'd) embeddings.

Output: paper_recsys/cka_raw_256d_{dataset}.json
"""
import json
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


def main():
    ROOT = Path(__file__).resolve().parent.parent
    datasets = {
        "Amazon_Beauty": 259205,
        "Amazon_Toys_and_Games": 336073,
    }

    for ds_name, n_items in datasets.items():
        base = ROOT / "dataset" / ds_name
        out_path = ROOT / "paper_recsys" / f"cka_raw_256d_{ds_name.lower().replace('amazon_','').replace('_and_','_')}.json"

        # 256-d pre-whitening embeddings
        base_emb_path = base / "item_text_emb.qwen2.5_7b.base.no_whiten.npy"
        mv_emb_path = base / "item_text_emb.qwen2.5_7b.multiview.no_whiten.npy"

        if not base_emb_path.exists() or not mv_emb_path.exists():
            print(f"[{ds_name}] SKIP: no_whiten 256-d files not found")
            continue

        print(f"[{ds_name}] Loading 256-d no_whiten embeddings...")
        base_emb = np.load(str(base_emb_path), mmap_mode="r")
        mv_emb = np.load(str(mv_emb_path), mmap_mode="r")
        print(f"  base: {base_emb.shape}, multiview: {mv_emb.shape}")

        # Also try to load per-view 256-d if available (from center_only dir)
        center_only_dir = base / "qwen2.5_7b_4views_center_only"
        has_per_view = all((center_only_dir / f"view_{i}.npy").exists() for i in range(4))

        cka_base_mv_trials = []
        er_base_trials = []
        er_mv_trials = []
        per_view_cka = {}
        per_view_er = {}

        for trial in range(N_TRIALS):
            rng = np.random.RandomState(42 + trial * 7)
            idx = rng.choice(n_items, size=min(SUBSAMPLE_SIZE, n_items), replace=False)
            idx.sort()

            sub_base = np.array(base_emb[idx], dtype=np.float32)
            sub_mv = np.array(mv_emb[idx], dtype=np.float32)

            cka_base_mv_trials.append(linear_cka(sub_base, sub_mv))
            er_base_trials.append(effective_rank(sub_base))
            er_mv_trials.append(effective_rank(sub_mv))

            if has_per_view:
                views_256 = []
                for i in range(4):
                    v = np.load(str(center_only_dir / f"view_{i}.npy"), mmap_mode="r")
                    views_256.append(np.array(v[idx], dtype=np.float32))

                view_names = ["Description", "Function", "Audience", "Style"]
                for i in range(4):
                    for j in range(i + 1, 4):
                        pair_key = f"{view_names[i]} vs {view_names[j]}"
                        if pair_key not in per_view_cka:
                            per_view_cka[pair_key] = []
                        per_view_cka[pair_key].append(linear_cka(views_256[i], views_256[j]))

                for i in range(4):
                    vn = view_names[i]
                    if vn not in per_view_er:
                        per_view_er[vn] = []
                    per_view_er[vn].append(effective_rank(views_256[i]))

                concat_4 = np.concatenate(views_256, axis=1)
                if "concat_4views" not in per_view_er:
                    per_view_er["concat_4views"] = []
                per_view_er["concat_4views"].append(effective_rank(concat_4))

        result = {
            "dataset": ds_name,
            "n_items": n_items,
            "dim": 256 if has_per_view else int(base_emb.shape[1]),
            "stage": "post-SVD_256d_pre-whitening",
            "n_trials": N_TRIALS,
            "subsample_size": SUBSAMPLE_SIZE,
            "cka_base_vs_multiview": {
                "mean": float(np.mean(cka_base_mv_trials)),
                "std": float(np.std(cka_base_mv_trials)),
                "values": cka_base_mv_trials,
            },
            "effective_rank_base_256d": {
                "mean": float(np.mean(er_base_trials)),
                "std": float(np.std(er_base_trials)),
                "ratio": float(np.mean(er_base_trials)) / base_emb.shape[1],
            },
            "effective_rank_multiview_256d": {
                "mean": float(np.mean(er_mv_trials)),
                "std": float(np.std(er_mv_trials)),
                "ratio": float(np.mean(er_mv_trials)) / mv_emb.shape[1],
            },
        }

        if per_view_cka:
            result["per_view_pairwise_cka_256d"] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
                for k, v in per_view_cka.items()
            }
            result["per_view_effective_rank_256d"] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                    "dim": 256 if k != "concat_4views" else 1024,
                    "ratio": float(np.mean(v)) / (256 if k != "concat_4views" else 1024)}
                for k, v in per_view_er.items()
            }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  -> Wrote {out_path}")
        print(f"  CKA(base vs mv) = {result['cka_base_vs_multiview']['mean']:.4f}")
        print(f"  ER(base 256-d) = {result['effective_rank_base_256d']['mean']:.1f} / 256 = {result['effective_rank_base_256d']['ratio']:.3f}")
        print(f"  ER(mv 256-d) = {result['effective_rank_multiview_256d']['mean']:.1f} / 256 = {result['effective_rank_multiview_256d']['ratio']:.3f}")


if __name__ == "__main__":
    main()
