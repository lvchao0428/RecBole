#!/usr/bin/env python3
"""
Embedding Collapse Diagnosis Tool

Analyzes whether multi-view text embeddings suffer from dimensionality collapse
compared to single-view LLM embeddings, explaining the MV < LLM phenomenon.

Mode A: Raw npy analysis (no checkpoint needed)
Mode B: Checkpoint-based projected embedding analysis

Usage:
    python scripts/diagnose_collapse.py --mode A --dataset Amazon_Beauty
    python scripts/diagnose_collapse.py --mode B --dataset Amazon_Beauty \
        --llm_ckpt saved/ps_beauty_llm_nc_noboost_seed2025 \
        --mv_ckpt saved/ps_beauty_mv_nc_noboost_seed2025
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from numpy.linalg import svd


def compute_effective_rank(emb: np.ndarray, center: bool = True) -> dict:
    """Compute effective rank via SVD and Shannon entropy."""
    if emb.dtype == np.float16:
        emb = emb.astype(np.float32)

    if center:
        emb = emb - emb.mean(axis=0)

    n_samples = min(emb.shape[0], 5000)
    if emb.shape[0] > n_samples:
        idx = np.random.choice(emb.shape[0], n_samples, replace=False)
        emb = emb[idx]

    _, S, _ = svd(emb, full_matrices=False)
    S = S[S > 1e-10]

    if len(S) == 0:
        return {"effective_rank": 0, "max_rank": emb.shape[1], "ratio": 0}

    p = S / S.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    eff_rank = np.exp(entropy)
    max_rank = min(emb.shape)

    top10_energy = (S[:10] ** 2).sum() / (S ** 2).sum()
    top50_energy = (S[:50] ** 2).sum() / (S ** 2).sum()

    return {
        "effective_rank": float(eff_rank),
        "max_rank": int(max_rank),
        "ratio": float(eff_rank / max_rank),
        "top10_sv": S[:10].tolist(),
        "sv_decay_10": float(S[9] / S[0]) if len(S) >= 10 else 0,
        "sv_decay_50": float(S[49] / S[0]) if len(S) >= 50 else 0,
        "top10_energy": float(top10_energy),
        "top50_energy": float(top50_energy),
        "num_sv": int(len(S)),
        "singular_values": S.tolist(),
    }


def compute_uniformity(emb: np.ndarray, n_pairs: int = 5000) -> float:
    """Uniformity: log E[e^{-2||x-y||^2}] over random pairs."""
    if emb.dtype == np.float16:
        emb = emb.astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    emb_norm = emb / norms

    n = emb_norm.shape[0]
    idx_i = np.random.choice(n, n_pairs, replace=True)
    idx_j = np.random.choice(n, n_pairs, replace=True)

    diff = emb_norm[idx_i] - emb_norm[idx_j]
    sq_dist = (diff ** 2).sum(axis=1)
    uniformity = np.log(np.mean(np.exp(-2 * sq_dist)) + 1e-12)
    return float(uniformity)


def compute_pairwise_cosine(emb: np.ndarray, n_samples: int = 2000) -> dict:
    """Pairwise cosine similarity statistics."""
    if emb.dtype == np.float16:
        emb = emb.astype(np.float32)
    n = min(emb.shape[0], n_samples)
    emb_sub = emb[:n]
    norms = np.linalg.norm(emb_sub, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    emb_normed = emb_sub / norms
    sim_matrix = emb_normed @ emb_normed.T
    mask = ~np.eye(n, dtype=bool)
    sims = sim_matrix[mask]
    return {
        "mean_cosine": float(np.mean(sims)),
        "std_cosine": float(np.std(sims)),
        "median_cosine": float(np.median(sims)),
    }


def run_mode_a(dataset: str, base_dir: str):
    """Mode A: Analyze raw npy files without checkpoint."""
    ds_path = Path(base_dir) / dataset
    results = {"mode": "A", "dataset": dataset, "layers": {}}

    embeddings = {}

    tfidf_path = ds_path / "item_text_emb.base.ts.npy"
    if not tfidf_path.exists():
        tfidf_path = ds_path / "item_text_emb.base.npy"
    if tfidf_path.exists():
        embeddings["TF-IDF_base"] = np.load(tfidf_path)
        print(f"  Loaded TF-IDF base: {embeddings['TF-IDF_base'].shape}")

    llm_path = ds_path / "item_text_emb.qwen2.5_7b.base.ts.npy"
    if not llm_path.exists():
        llm_path = ds_path / "item_text_emb.qwen2.5_7b.base.npy"
    if llm_path.exists():
        embeddings["LLM_base"] = np.load(llm_path)
        print(f"  Loaded LLM base: {embeddings['LLM_base'].shape}")

    views_dir = ds_path / "qwen2.5_7b_4views_ts"
    if not views_dir.exists():
        views_dir = ds_path / "qwen2.5_7b_4views"
    if views_dir.exists():
        view_parts = []
        for i in range(4):
            vp = views_dir / f"view_{i}.npy"
            if vp.exists():
                v = np.load(vp)
                embeddings[f"view_{i}"] = v
                view_parts.append(v.astype(np.float32) if v.dtype == np.float16 else v)
                print(f"  Loaded view_{i}: {v.shape}")
        if view_parts:
            embeddings["MV_views_concat"] = np.concatenate(view_parts, axis=1)
            print(f"  Concat views: {embeddings['MV_views_concat'].shape}")

    if "TF-IDF_base" in embeddings and "MV_views_concat" in embeddings:
        tf = embeddings["TF-IDF_base"]
        if tf.dtype == np.float16:
            tf = tf.astype(np.float32)
        mv_concat = embeddings["MV_views_concat"]
        embeddings["MV_full_concat"] = np.concatenate([tf, mv_concat], axis=1)
        print(f"  Full MV concat (base+views): {embeddings['MV_full_concat'].shape}")

    if "TF-IDF_base" in embeddings and "LLM_base" in embeddings:
        tf = embeddings["TF-IDF_base"]
        llm = embeddings["LLM_base"]
        if tf.dtype == np.float16:
            tf = tf.astype(np.float32)
        if llm.dtype == np.float16:
            llm = llm.astype(np.float32)
        embeddings["LLM_full_concat"] = np.concatenate([tf, llm], axis=1)
        print(f"  Full LLM concat (base+llm): {embeddings['LLM_full_concat'].shape}")

    print("\n" + "=" * 80)
    print("EFFECTIVE RANK ANALYSIS")
    print("=" * 80)

    header = f"{'Layer':<25} {'Dim':>5} {'Eff Rank':>10} {'Ratio':>7} {'Top10 E':>8} {'Uniformity':>11} {'Mean Cos':>9}"
    print(header)
    print("-" * 80)

    for name, emb in embeddings.items():
        np.random.seed(42)
        rank_info = compute_effective_rank(emb)
        uniformity = compute_uniformity(emb)
        cos_info = compute_pairwise_cosine(emb)

        results["layers"][name] = {
            **rank_info,
            "uniformity": uniformity,
            **cos_info,
            "shape": list(emb.shape),
        }

        print(f"{name:<25} {emb.shape[1]:>5} {rank_info['effective_rank']:>10.1f} "
              f"{rank_info['ratio']:>7.3f} {rank_info['top10_energy']:>8.3f} "
              f"{uniformity:>11.4f} {cos_info['mean_cosine']:>9.4f}")

    if "LLM_base" in results["layers"] and "MV_views_concat" in results["layers"]:
        llm_rank = results["layers"]["LLM_base"]["effective_rank"]
        mv_rank = results["layers"]["MV_views_concat"]["effective_rank"]
        print(f"\n{'=' * 80}")
        print(f"KEY COMPARISON: MV views effective rank / LLM effective rank = {mv_rank/llm_rank:.3f}")
        if mv_rank / llm_rank < 0.7:
            print("  >>> COLLAPSE DETECTED: MV views have significantly lower effective rank")
        elif mv_rank / llm_rank < 0.9:
            print("  >>> MILD COLLAPSE: MV views have moderately lower effective rank")
        else:
            print("  >>> NO COLLAPSE: MV views have similar effective rank to LLM")

    if "MV_full_concat" in results["layers"] and "LLM_full_concat" in results["layers"]:
        mv_fc = results["layers"]["MV_full_concat"]["effective_rank"]
        llm_fc = results["layers"]["LLM_full_concat"]["effective_rank"]
        print(f"\n  Full concat comparison: MV {mv_fc:.1f} vs LLM {llm_fc:.1f} (ratio={mv_fc/llm_fc:.3f})")

    print("\n" + "=" * 80)
    print("VIEW REDUNDANCY ANALYSIS")
    print("=" * 80)

    view_names = [k for k in embeddings if k.startswith("view_")]
    if len(view_names) >= 2:
        print(f"\n{'Pair':<25} {'Mean Cosine':>12}")
        print("-" * 40)
        for i in range(len(view_names)):
            for j in range(i + 1, len(view_names)):
                vi = embeddings[view_names[i]].astype(np.float32)
                vj = embeddings[view_names[j]].astype(np.float32)
                n = min(2000, vi.shape[0])
                vi_n = vi[:n] / (np.linalg.norm(vi[:n], axis=1, keepdims=True) + 1e-8)
                vj_n = vj[:n] / (np.linalg.norm(vj[:n], axis=1, keepdims=True) + 1e-8)
                cos_sim = np.mean(np.sum(vi_n * vj_n, axis=1))
                print(f"{view_names[i]} vs {view_names[j]:<10} {cos_sim:>12.4f}")
                results.setdefault("view_redundancy", {})[f"{view_names[i]}_vs_{view_names[j]}"] = float(cos_sim)

    return results


def run_mode_b(dataset: str, base_dir: str, llm_ckpt: str, mv_ckpt: str):
    """Mode B: Load checkpoints and compare projected embeddings."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch
    from recbole.quick_start import load_data_and_model

    results = {"mode": "B", "dataset": dataset, "layers": {}}

    print("\n" + "=" * 80)
    print("MODE B: CHECKPOINT-BASED ANALYSIS")
    print("=" * 80)

    ckpt_dir = Path(llm_ckpt)
    pth_files = list(ckpt_dir.glob("*.pth"))
    if not pth_files:
        print(f"  ERROR: No .pth file in {llm_ckpt}")
        return results
    llm_pth = str(pth_files[0])

    ckpt_dir = Path(mv_ckpt)
    pth_files = list(ckpt_dir.glob("*.pth"))
    if not pth_files:
        print(f"  ERROR: No .pth file in {mv_ckpt}")
        return results
    mv_pth = str(pth_files[0])

    print(f"\n  Loading LLM checkpoint: {llm_pth}")
    config_l, model_l, dataset_l, _, _, _ = load_data_and_model(llm_pth)
    model_l.eval()

    print(f"  Loading MV checkpoint: {mv_pth}")
    config_m, model_m, dataset_m, _, _, _ = load_data_and_model(mv_pth)
    model_m.eval()

    device = config_l["device"]
    n_items = min(model_l.n_items, 10000)
    ids = torch.arange(1, n_items, device=device)

    with torch.no_grad():
        llm_raw = model_l._gather_text_raw(ids)
        llm_proj = model_l._project_text(llm_raw)
        llm_fused = model_l._get_fused_item_embeddings(ids)
        llm_id = model_l.item_embedding(ids)

        mv_raw = model_m._gather_mv_text_raw(ids)
        mv_proj = model_m._project_mv_text(mv_raw)
        mv_fused = model_m._get_fused_item_embeddings(ids)
        mv_id = model_m.item_embedding(ids)

    layers = {
        "LLM_raw_text": llm_raw.cpu().numpy(),
        "LLM_projected": llm_proj.cpu().numpy(),
        "LLM_fused_item": llm_fused.cpu().numpy(),
        "LLM_id_emb": llm_id.cpu().numpy(),
        "MV_raw_text": mv_raw.cpu().numpy(),
        "MV_projected": mv_proj.cpu().numpy(),
        "MV_fused_item": mv_fused.cpu().numpy(),
        "MV_id_emb": mv_id.cpu().numpy(),
    }

    print(f"\n{'Layer':<20} {'Dim':>5} {'Eff Rank':>10} {'Ratio':>7} {'Top10 E':>8} {'Uniformity':>11}")
    print("-" * 70)

    for name, emb in layers.items():
        np.random.seed(42)
        rank_info = compute_effective_rank(emb)
        uniformity = compute_uniformity(emb)
        results["layers"][name] = {**rank_info, "uniformity": uniformity, "shape": list(emb.shape)}
        print(f"{name:<20} {emb.shape[1]:>5} {rank_info['effective_rank']:>10.1f} "
              f"{rank_info['ratio']:>7.3f} {rank_info['top10_energy']:>8.3f} {uniformity:>11.4f}")

    print(f"\n{'=' * 70}")
    print("LAYER-BY-LAYER MV vs LLM COMPARISON")
    print(f"{'=' * 70}")
    comparisons = [
        ("Raw text", "LLM_raw_text", "MV_raw_text"),
        ("Projected", "LLM_projected", "MV_projected"),
        ("Fused item", "LLM_fused_item", "MV_fused_item"),
    ]
    print(f"{'Layer':<15} {'LLM Rank':>10} {'MV Rank':>10} {'MV/LLM':>8} {'Collapse?':>10}")
    print("-" * 55)
    for label, llm_key, mv_key in comparisons:
        lr = results["layers"][llm_key]["effective_rank"]
        mr = results["layers"][mv_key]["effective_rank"]
        ratio = mr / lr if lr > 0 else 0
        collapse = "YES" if ratio < 0.7 else ("Mild" if ratio < 0.85 else "No")
        print(f"{label:<15} {lr:>10.1f} {mr:>10.1f} {ratio:>8.3f} {collapse:>10}")

    gate_alpha_l = torch.sigmoid(model_l.text_gate_param).item() if hasattr(model_l, 'text_gate_param') else None
    gate_alpha_m = torch.sigmoid(model_m.text_gate_param).item() if hasattr(model_m, 'text_gate_param') else None
    if gate_alpha_l is not None:
        tw_l = getattr(model_l, 'text_weight', 1.0)
        tw_m = getattr(model_m, 'text_weight', 1.0)
        print(f"\n  Gate alpha: LLM={gate_alpha_l:.4f} (eff={gate_alpha_l*tw_l:.4f}), "
              f"MV={gate_alpha_m:.4f} (eff={gate_alpha_m*tw_m:.4f})")
        results["gate"] = {"llm_alpha": gate_alpha_l, "mv_alpha": gate_alpha_m}

    return results


def main():
    parser = argparse.ArgumentParser(description="Embedding Collapse Diagnosis")
    parser.add_argument("--mode", type=str, choices=["A", "B", "AB"], default="A")
    parser.add_argument("--dataset", type=str, default="Amazon_Beauty")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--llm_ckpt", type=str, default=None)
    parser.add_argument("--mv_ckpt", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.base_dir is None:
        repo_root = Path(__file__).resolve().parents[1]
        args.base_dir = str(repo_root / "dataset")

    print("=" * 80)
    print(f"EMBEDDING COLLAPSE DIAGNOSIS — {args.dataset}")
    print(f"Mode: {args.mode}")
    print("=" * 80)

    all_results = {}

    if args.mode in ("A", "AB"):
        print("\n>>> Running Mode A (raw npy analysis)...")
        all_results["mode_a"] = run_mode_a(args.dataset, args.base_dir)

    if args.mode in ("B", "AB"):
        if not args.llm_ckpt or not args.mv_ckpt:
            print("\n  ERROR: Mode B requires --llm_ckpt and --mv_ckpt")
            sys.exit(1)
        print("\n>>> Running Mode B (checkpoint analysis)...")
        all_results["mode_b"] = run_mode_b(args.dataset, args.base_dir,
                                           args.llm_ckpt, args.mv_ckpt)

    output_file = args.output
    if output_file is None:
        output_file = f"collapse_diagnosis_{args.dataset.replace('Amazon_', '').lower()}.json"

    sv_keys_to_remove = []
    for mode_key, mode_data in all_results.items():
        if "layers" in mode_data:
            for layer_name, layer_data in mode_data["layers"].items():
                if "singular_values" in layer_data:
                    sv_keys_to_remove.append((mode_key, layer_name))

    for mode_key, layer_name in sv_keys_to_remove:
        del all_results[mode_key]["layers"][layer_name]["singular_values"]

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
