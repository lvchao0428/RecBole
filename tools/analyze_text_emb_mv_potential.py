#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze text embedding decorrelation and multi-view potential for a RecBole dataset.

Metrics:
- inter-item cosine similarity (TF-IDF / LLM single / each MV view)
- pairwise view diversity (like quick_emb_stats)
- whiten diagonal strength (post-whiten correlation off-diagonal)
- concat vs single-view redundancy

Usage:
  python tools/analyze_text_emb_mv_potential.py --dataset-dir dataset/book-crossing
  python tools/analyze_text_emb_mv_potential.py --dataset-dir dataset/Amazon_Beauty --compare
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _emb_stats(emb: np.ndarray, name: str, n_sample: int = 500, seed: int = 42) -> Dict:
    from sklearn.metrics.pairwise import cosine_similarity

    emb = np.asarray(emb, dtype=np.float64)
    n = emb.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(1, n)  # skip PAD
    if len(idx) > n_sample:
        idx = rng.choice(idx, n_sample, replace=False)
    sub = emb[idx]
    norms = np.linalg.norm(sub, axis=1)
    sim = cosine_similarity(sub)
    mask = ~np.eye(sim.shape[0], dtype=bool)
    vals = sim[mask]
    return {
        "name": name,
        "shape": list(emb.shape),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "inter_item_sim_mean": float(vals.mean()),
        "inter_item_sim_std": float(vals.std()),
        "inter_item_sim_p90": float(np.percentile(vals, 90)),
        "high_sim_ratio_gt0.9": float((vals > 0.9).mean()),
        "low_sim_ratio_lt0.3": float((vals < 0.3).mean()),
    }


def _view_diversity(views: Dict[str, np.ndarray], n_sample: int = 500) -> Dict:
    from scipy.spatial.distance import cosine

    names = sorted(views.keys())
    pairwise = {}
    rng = np.random.default_rng(42)
    n = min(v.shape[0] for v in views.values())
    idx = np.arange(1, n)
    if len(idx) > n_sample:
        idx = rng.choice(idx, n_sample, replace=False)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sims = []
            for k in idx:
                va = np.asarray(views[a][k], dtype=np.float64)
                vb = np.asarray(views[b][k], dtype=np.float64)
                if np.linalg.norm(va) < 1e-9 or np.linalg.norm(vb) < 1e-9:
                    continue
                sims.append(1 - cosine(va, vb))
            pairwise[f"{a}_vs_{b}"] = float(np.mean(sims)) if sims else float("nan")
    avg = float(np.nanmean(list(pairwise.values())))
    return {
        "pairwise": pairwise,
        "avg_cross_view_cosine": avg,
        "diversity_score": float(1 - avg),
    }


def _whiten_diag(stats_path: str) -> Optional[Dict]:
    if not os.path.isfile(stats_path):
        return None
    z = np.load(stats_path)
    key = "whiten" if "whiten" in z.files else (
        "whitening_matrix" if "whitening_matrix" in z.files else (
            "whiten_matrix" if "whiten_matrix" in z.files else None
        )
    )
    if key is None:
        return {"keys": list(z.files)}
    w = np.asarray(z[key], dtype=np.float64)
    if w.ndim != 2:
        return None
    d = np.diag(w)
    off = w.copy()
    np.fill_diagonal(off, 0)
    return {
        "whiten_diag_mean": float(np.abs(d).mean()),
        "whiten_offdiag_mean": float(np.abs(off).mean()),
        "whiten_offdiag_ratio": float(np.abs(off).mean() / (np.abs(d).mean() + 1e-12)),
    }


def _concat_vs_single_redundancy(
    single: np.ndarray, views: Dict[str, np.ndarray], n_sample: int = 300
) -> Dict:
    """How much concat(multiview) adds beyond single LLM view (sampled items)."""
    from scipy.spatial.distance import cosine

    concat = np.hstack([views[k] for k in sorted(views.keys())])
    rng = np.random.default_rng(42)
    n = min(single.shape[0], concat.shape[0])
    idx = np.arange(1, n)
    if len(idx) > n_sample:
        idx = rng.choice(idx, n_sample, replace=False)
    sims = []
    for k in idx:
        a = np.asarray(single[k], dtype=np.float64)
        b = np.asarray(concat[k], dtype=np.float64)
        if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
            continue
        sims.append(1 - cosine(a, b))
    mean_sim = float(np.mean(sims)) if sims else float("nan")
    return {
        "single_vs_concat_cosine": mean_sim,
        "concat_incremental_diversity": float(1 - mean_sim),
    }


def analyze_dataset_dir(ddir: str, label: Optional[str] = None) -> Dict:
    ddir = str(ddir)
    label = label or Path(ddir).name
    out: Dict = {"label": label, "dataset_dir": ddir, "embeddings": {}, "whiten": {}, "multiview": {}}

    paths = {
        "tfidf": "item_text_emb.base.npy",
        "llm_single": "item_text_emb.qwen3.base.npy",
        "llm_multiview_concat": "item_text_emb.qwen3.multiview.npy",
    }
    for key, fn in paths.items():
        p = os.path.join(ddir, fn)
        if os.path.isfile(p):
            out["embeddings"][key] = _emb_stats(np.load(p, mmap_mode="r"), key)

    for key, fn in [
        ("tfidf", "item_text_emb.base_whiten_stats.npz"),
        ("llm_single", "item_text_emb.qwen3.base_whiten_stats.npz"),
        ("llm_mv", "item_text_emb.qwen3.multiview_whiten_stats.npz"),
    ]:
        ws = _whiten_diag(os.path.join(ddir, fn))
        if ws:
            out["whiten"][key] = ws

    mv_dir = os.path.join(ddir, "qwen3_4views")
    views = {}
    if os.path.isdir(mv_dir):
        for i in range(8):
            vp = os.path.join(mv_dir, f"view_{i}.npy")
            if os.path.isfile(vp):
                views[f"view_{i}"] = np.load(vp, mmap_mode="r")
        if views:
            out["embeddings"]["views"] = {k: _emb_stats(v, k, n_sample=300) for k, v in views.items()}
            out["multiview"]["view_diversity"] = _view_diversity(views)
            if "llm_single" in out["embeddings"]:
                single = np.load(os.path.join(ddir, "item_text_emb.qwen3.base.npy"), mmap_mode="r")
                out["multiview"]["single_vs_concat"] = _concat_vs_single_redundancy(single, views)

    # MV advantage heuristic
    vd = out["multiview"].get("view_diversity", {})
    div = vd.get("diversity_score")
    wh = out["whiten"].get("llm_mv", {})
    off_ratio = wh.get("whiten_offdiag_ratio", 1.0) if wh else 1.0
    if div is not None and div >= 0.06 and off_ratio < 0.15:
        mv_adv = "likely"
    elif div is not None and div >= 0.04:
        mv_adv = "moderate"
    else:
        mv_adv = "unlikely"
    out["mv_advantage_heuristic"] = mv_adv
    out["mv_advantage_note"] = (
        "diversity_score>=0.06 and whiten offdiag low → views decorrelated enough for MV gain"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", action="append", required=True)
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    rows = []
    for i, d in enumerate(args.dataset_dir):
        lab = args.label[i] if i < len(args.label) else None
        print(f"Analyzing {d} ...", file=sys.stderr)
        rows.append(analyze_dataset_dir(d, lab))

    out = {"datasets": rows}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.md:
        lines = [
            "# 文本 embedding 去相关 / MV 潜力分析",
            "",
            "| Dataset | TF-IDF sim | LLM sim | view diversity | whiten offdiag | MV 优势 |",
            "|---------|------------|---------|----------------|----------------|---------|",
        ]
        for r in rows:
            tf = r["embeddings"].get("tfidf", {})
            ll = r["embeddings"].get("llm_single", {})
            vd = r["multiview"].get("view_diversity", {})
            wh = r["whiten"].get("llm_mv", r["whiten"].get("llm_single", {}))
            lines.append(
                f"| {r['label']} | {tf.get('inter_item_sim_mean', float('nan')):.3f} | "
                f"{ll.get('inter_item_sim_mean', float('nan')):.3f} | "
                f"{vd.get('diversity_score', float('nan')):.3f} | "
                f"{wh.get('whiten_offdiag_ratio', float('nan')):.3f} | {r['mv_advantage_heuristic']} |"
            )
        if len(rows) >= 2:
            lines += ["", "## Pairwise view cosine (lower = more diverse)"]
            for r in rows:
                vd = r["multiview"].get("view_diversity", {})
                if vd.get("pairwise"):
                    lines.append(f"\n**{r['label']}**:")
                    for k, v in sorted(vd["pairwise"].items()):
                        lines.append(f"- {k}: {v:.3f}")
        Path(args.md).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
