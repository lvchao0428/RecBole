#!/usr/bin/env python3
"""
Compare LLM embedding quality across datasets (Beauty vs Toys).

Focus:
- LLM basic stats (mean/std, feature variance spread)
- Feature-space structure (pairwise cosine spread, effective rank)
- Alignment between merged multi-view embedding and single LLM

Usage (defaults to Beauty+Toys in repo ./dataset):
  python tools/compare_llm_quality.py

Custom base dir or datasets:
  python tools/compare_llm_quality.py \
    --base-dir /path/to/dataset \
    --datasets Amazon_Beauty Amazon_Toys_and_Games \
    --sample-size 2000 \
    --output compare_llm_quality.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from numpy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity


def default_base_dir() -> Path:
    """Resolve repo-root/dataset by default."""
    return Path(__file__).resolve().parents[1] / "dataset"


def load_embeddings(dataset: str, base_dir: Path) -> Dict[str, np.ndarray]:
    """Load LLM and merged multi-view embeddings (float32)."""
    ds_dir = base_dir / dataset
    llm_path = ds_dir / "item_text_emb.qwen3.base.npy"
    mv_path = ds_dir / "item_text_emb.qwen3.multiview.npy"

    embeddings: Dict[str, np.ndarray] = {}

    if llm_path.exists():
        emb = np.load(llm_path)
        embeddings["llm"] = emb.astype(np.float32, copy=False)
    else:
        raise FileNotFoundError(f"Missing LLM embedding: {llm_path}")

    if mv_path.exists():
        emb = np.load(mv_path)
        embeddings["multiview_merged"] = emb.astype(np.float32, copy=False)

    return embeddings


def basic_stats(emb: np.ndarray) -> Dict[str, float]:
    feature_var = np.var(emb, axis=0)
    norms = np.linalg.norm(emb, axis=1)
    return {
        "shape": list(emb.shape),
        "mean": float(np.mean(emb)),
        "std": float(np.std(emb)),
        "min": float(np.min(emb)),
        "max": float(np.max(emb)),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "sparsity": float(np.mean(np.abs(emb) < 1e-6)),
        "feature_var_mean": float(np.mean(feature_var)),
        "feature_var_std": float(np.std(feature_var)),
        "feature_var_min": float(np.min(feature_var)),
        "feature_var_max": float(np.max(feature_var)),
    }


def feature_space(emb: np.ndarray, sample_size: int = 2000) -> Dict[str, float]:
    n = min(len(emb), sample_size)
    sampled = emb[:n].astype(np.float32, copy=False)

    cos = cosine_similarity(sampled)
    mask = ~np.eye(cos.shape[0], dtype=bool)
    cos_vals = cos[mask]
    cos_vals = cos_vals[~np.isnan(cos_vals) & ~np.isinf(cos_vals)]

    metrics: Dict[str, float] = {}
    if len(cos_vals) > 0:
        metrics.update(
            {
                "mean_pairwise_cosine": float(np.mean(cos_vals)),
                "std_pairwise_cosine": float(np.std(cos_vals)),
                "min_pairwise_cosine": float(np.min(cos_vals)),
                "max_pairwise_cosine": float(np.max(cos_vals)),
                "median_pairwise_cosine": float(np.median(cos_vals)),
            }
        )
    else:
        metrics.update(
            {
                "mean_pairwise_cosine": None,
                "std_pairwise_cosine": None,
                "min_pairwise_cosine": None,
                "max_pairwise_cosine": None,
                "median_pairwise_cosine": None,
            }
        )

    # Effective rank via participation ratio of singular values
    centered = sampled - sampled.mean(axis=0, keepdims=True)
    try:
        _, s, _ = svd(centered, full_matrices=False)
        s = s[s > 1e-10]
        if len(s) > 0:
            p = s / s.sum()
            effective_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
            metrics["effective_rank"] = effective_rank
            metrics["top_10_singular_values"] = s[:10].tolist()
            metrics["singular_value_decay"] = float(s[10] / s[0]) if len(s) > 10 else 0.0
        else:
            metrics["effective_rank"] = None
    except Exception as e:  # pragma: no cover - diagnostic fallback
        metrics["effective_rank"] = None
        metrics["svd_error"] = str(e)

    return metrics


def llm_vs_multiview(
    llm: np.ndarray, mv: np.ndarray, sample_size: int = 2000
) -> Optional[Dict[str, float]]:
    if mv is None or mv.shape[1] != llm.shape[1]:
        return None

    n = min(len(llm), len(mv), sample_size)
    llm_s = llm[:n]
    mv_s = mv[:n]
    # cosine_similarity expects shape (n_samples, n_features)
    sims = np.sum(llm_s * mv_s, axis=1) / (
        np.linalg.norm(llm_s, axis=1) * np.linalg.norm(mv_s, axis=1) + 1e-12
    )
    sims = sims[~np.isnan(sims) & ~np.isinf(sims)]
    if len(sims) == 0:
        return None
    return {
        "mean_cosine": float(np.mean(sims)),
        "std_cosine": float(np.std(sims)),
        "min_cosine": float(np.min(sims)),
        "max_cosine": float(np.max(sims)),
        "sample_size": int(len(sims)),
    }


def analyze_dataset(
    dataset: str, base_dir: Path, sample_size: int
) -> Dict[str, Dict]:
    embs = load_embeddings(dataset, base_dir)

    llm_stats = basic_stats(embs["llm"])
    llm_fs = feature_space(embs["llm"], sample_size)

    mv_stats = mv_fs = llm_mv = None
    if "multiview_merged" in embs:
        mv_stats = basic_stats(embs["multiview_merged"])
        mv_fs = feature_space(embs["multiview_merged"], sample_size)
        llm_mv = llm_vs_multiview(embs["llm"], embs["multiview_merged"], sample_size)

    return {
        "dataset": dataset,
        "paths": {
            "llm": str((base_dir / dataset / "item_text_emb.qwen3.base.npy").resolve()),
            "multiview_merged": str(
                (base_dir / dataset / "item_text_emb.qwen3.multiview.npy").resolve()
            )
            if "multiview_merged" in embs
            else None,
        },
        "llm": {"basic_stats": llm_stats, "feature_space": llm_fs},
        "multiview_merged": {
            "basic_stats": mv_stats,
            "feature_space": mv_fs,
        },
        "llm_vs_multiview": llm_mv,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM embedding quality across datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Amazon_Beauty", "Amazon_Toys_and_Games"],
        help="Dataset names under base-dir (default: Beauty & Toys).",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(default_base_dir()),
        help="Directory containing dataset folders.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Samples used for cosine/SVD computations.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="compare_llm_quality.json",
        help="Where to save comparison JSON.",
    )

    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    results = {}
    for ds in args.datasets:
        print(f"Analyzing {ds} ...")
        try:
            results[ds] = analyze_dataset(ds, base_dir, args.sample_size)
            llm_fs = results[ds]["llm"]["feature_space"]
            print(
                f"  LLM effective_rank={llm_fs.get('effective_rank')}, "
                f"mean_pairwise_cosine={llm_fs.get('mean_pairwise_cosine')}"
            )
            if results[ds]["llm_vs_multiview"]:
                mv = results[ds]["llm_vs_multiview"]
                print(
                    f"  MV vs LLM mean_cosine={mv['mean_cosine']:.4f} "
                    f"(n={mv['sample_size']})"
                )
        except Exception as e:  # pragma: no cover - diagnostic flow
            results[ds] = {"error": str(e)}
            print(f"  Error: {e}")

    # If two datasets, add a delta block for quick eyeballing
    if len(args.datasets) == 2:
        a, b = args.datasets
        if "error" not in results.get(a, {}) and "error" not in results.get(b, {}):
            delta = {}
            for key in [
                "mean_pairwise_cosine",
                "std_pairwise_cosine",
                "median_pairwise_cosine",
                "effective_rank",
            ]:
                delta[f"llm_{key}"] = (
                    results[a]["llm"]["feature_space"].get(key),
                    results[b]["llm"]["feature_space"].get(key),
                    None
                    if results[a]["llm"]["feature_space"].get(key) is None
                    or results[b]["llm"]["feature_space"].get(key) is None
                    else results[b]["llm"]["feature_space"][key]
                    - results[a]["llm"]["feature_space"][key],
                )
            results["delta"] = delta

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
