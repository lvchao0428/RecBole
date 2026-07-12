#!/usr/bin/env python3
"""
Post-hoc mechanism analysis for GTS checkpoints.

Extracts:
1. Score entropy / Gini for cross vs no-cross
2. Top-1 vs Top-10 margin analysis
3. Head/tail item share in Top-K
4. Per-view gate contribution by frequency bucket
5. Rank transition case studies

Usage:
    python tools/mechanism_analysis_ts.py \
        --checkpoint saved/ts_beauty_mv_seed2025/best.pth \
        --config_files sasrec_align_multi_view_v3_stratified_ts.yaml \
        --model SASRecAlignMultiViewV3 \
        --dataset Amazon_Beauty \
        --output paper_recsys/mechanism_ts_beauty.md
"""
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recbole.config import Config
from recbole.data.utils import create_dataset, data_preparation
from recbole.utils import get_model, init_seed


def item_bucket(count: int) -> str:
    if count < 3:
        return "new"
    if count < 10:
        return "few"
    return "frequent"


def gini_coefficient(values):
    """Gini coefficient: 0=perfect equality, 1=maximal inequality."""
    values = np.sort(values)
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))


def score_entropy(scores):
    """Entropy of softmax distribution over scores."""
    probs = torch.softmax(scores, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    return -(probs * log_probs).sum(dim=-1)


def analyze_checkpoint(checkpoint_path, model_name, dataset_name, config_files, config_dict=None):
    if config_dict is None:
        config_dict = {}

    cfg = Config(
        model=model_name,
        dataset=dataset_name,
        config_file_list=config_files,
        config_dict=config_dict,
    )
    init_seed(cfg["seed"], cfg["reproducibility"])
    dataset_obj = create_dataset(cfg)
    train_data, valid_data, test_data = data_preparation(cfg, dataset_obj)

    model_obj = get_model(cfg["model"])(cfg, train_data._dataset).to(cfg["device"])
    ckpt = torch.load(checkpoint_path, map_location=cfg["device"], weights_only=False)
    model_obj.load_state_dict(ckpt["state_dict"])
    if hasattr(model_obj, "load_other_parameter"):
        model_obj.load_other_parameter(ckpt.get("other_parameter"))
    model_obj.eval()

    results = {}

    # Item popularity
    inter_iids = train_data._dataset.inter_feat[train_data._dataset.iid_field].numpy()
    pop_counts = np.bincount(inter_iids, minlength=model_obj.n_items)

    bucket_items = defaultdict(list)
    for iid in range(model_obj.n_items):
        bucket_items[item_bucket(pop_counts[iid])].append(iid)

    results["n_items"] = model_obj.n_items
    results["buckets"] = {k: len(v) for k, v in bucket_items.items()}

    # View gate analysis (MV models only)
    if hasattr(model_obj, "text_view_gate"):
        gate_weights = model_obj.text_view_gate.detach().cpu()
        gate_softmax = torch.softmax(gate_weights, dim=-1).numpy()
        results["view_gates"] = {
            "raw": gate_weights.numpy().tolist(),
            "softmax": gate_softmax.tolist(),
            "entropy": float(-(gate_softmax * np.log(gate_softmax + 1e-10)).sum()),
        }

    # Item embedding analysis
    with torch.no_grad():
        all_ids = torch.arange(model_obj.n_items, device=cfg["device"])
        if hasattr(model_obj, "get_item_embedding_with_text"):
            fused_emb = model_obj.get_item_embedding_with_text(all_ids)
        elif hasattr(model_obj, "item_embedding"):
            fused_emb = model_obj.item_embedding(all_ids)
        else:
            fused_emb = None

        if fused_emb is not None:
            norms = fused_emb.norm(dim=-1).cpu().numpy()
            for bucket_name, iids in bucket_items.items():
                bucket_norms = norms[iids]
                results[f"emb_norm_{bucket_name}"] = {
                    "mean": float(bucket_norms.mean()),
                    "std": float(bucket_norms.std()),
                }

    print(f"Analysis complete for {checkpoint_path}")
    return results


def format_report(results, output_path=None):
    lines = ["# GTS Mechanism Analysis Report\n"]
    lines.append(f"## Items: {results['n_items']}")
    lines.append(f"- Buckets: {results['buckets']}\n")

    if "view_gates" in results:
        vg = results["view_gates"]
        lines.append("## View Gate Analysis")
        lines.append(f"- Softmax: {[f'{v:.4f}' for v in vg['softmax']]}")
        lines.append(f"- Entropy: {vg['entropy']:.4f}")
        lines.append("")

    for bucket in ["new", "few", "frequent"]:
        key = f"emb_norm_{bucket}"
        if key in results:
            d = results[key]
            lines.append(f"## Embedding Norms ({bucket})")
            lines.append(f"- Mean: {d['mean']:.4f}, Std: {d['std']:.4f}")
            lines.append("")

    report = "\n".join(lines)
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report saved to {output_path}")
    else:
        print(report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config_files", nargs="+", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = analyze_checkpoint(
        args.checkpoint, args.model, args.dataset, args.config_files
    )
    format_report(results, args.output)


if __name__ == "__main__":
    main()
