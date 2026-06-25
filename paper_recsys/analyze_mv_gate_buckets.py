#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0617 mechanism analysis (2): view/gate contribution by item frequency bucket.

Loads an MV checkpoint and reports global view gates plus per-bucket mean
weighted-view L2 norms (proxy for view contribution on new/few/frequent items).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from recbole.config import Config
from recbole.data.utils import create_dataset
from recbole.utils import get_model, init_seed


def item_bucket(count: int) -> str:
    if count < 3:
        return "new"
    if count < 10:
        return "few"
    return "frequent"


def load_model(checkpoint: str, model: str, dataset: str, config_files: list[str], config_dict: dict):
    cfg = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_files,
        config_dict=config_dict,
    )
    init_seed(cfg["seed"], cfg["reproducibility"])
    dataset_obj = create_dataset(cfg)
    model_obj = get_model(cfg["model"])(cfg, dataset_obj).to(cfg["device"])
    ckpt = torch.load(checkpoint, map_location=cfg["device"], weights_only=False)
    model_obj.load_state_dict(ckpt["state_dict"])
    model_obj.load_other_parameter(ckpt.get("other_parameter"))
    model_obj.eval()
    return cfg, model_obj, dataset_obj


def compute_bucket_stats(model, item_counts: torch.Tensor, num_views: int) -> dict:
    gates = torch.sigmoid(model.text_view_gate_params).detach().cpu()
    buckets = {"new": [], "few": [], "frequent": []}
    for item_id in range(1, item_counts.shape[0]):
        cnt = int(item_counts[item_id].item())
        buckets[item_bucket(cnt)].append(item_id)

    stats = {"global_gates": [float(gates[i]) for i in range(num_views)]}
    view_norms_by_bucket = {}
    with torch.no_grad():
        for bname, ids in buckets.items():
            if not ids:
                view_norms_by_bucket[bname] = [0.0] * num_views
                continue
            id_tensor = torch.tensor(ids, device=model.item_embedding.weight.device)
            view_stack = model._gather_text_views(id_tensor)
            weighted = []
            for vi in range(num_views):
                w = gates[vi].to(view_stack.device)
                norm = (w * view_stack[:, vi, :]).norm(dim=-1).mean().item()
                weighted.append(norm)
            view_norms_by_bucket[bname] = weighted
    stats["weighted_view_l2_mean"] = view_norms_by_bucket
    stats["bucket_sizes"] = {k: len(v) for k, v in buckets.items()}
    return stats


def format_report(stats: dict, checkpoint: str) -> str:
    lines = [
        "MV view gate / bucket analysis (0617 mechanism #2)",
        f"checkpoint: {checkpoint}",
        "",
        "Global view gates (sigmoid):",
    ]
    for i, g in enumerate(stats["global_gates"]):
        lines.append(f"  view_{i}: {g:.4f}")
    lines.append("")
    lines.append("Mean weighted-view L2 by bucket:")
    lines.append(f"{'bucket':<10}  {'n_items':>8}  " + "  ".join(f"v{i}" for i in range(len(stats["global_gates"]))))
    for bname in ("new", "few", "frequent"):
        norms = stats["weighted_view_l2_mean"][bname]
        n = stats["bucket_sizes"][bname]
        row = "  ".join(f"{x:.4f}" for x in norms)
        lines.append(f"{bname:<10}  {n:>8}  {row}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="saved/peruser_runs/beauty_mv_7b/SASRecAlignMultiViewV3-Mar-11-2026_16-42-34.pth",
    )
    parser.add_argument("--model", default="SASRecAlignMultiViewV3")
    parser.add_argument("--dataset", default="Amazon_Beauty")
    parser.add_argument(
        "--config_files",
        default="sasrec_align_multi_view_v3_stratified.yaml",
    )
    parser.add_argument(
        "--output",
        default="paper_recsys/mechanism_gate_buckets_beauty.txt",
    )
    args = parser.parse_args()

    config_files = [x for x in args.config_files.split() if x]
    config_dict = {
        "align_weight": 0.1,
        "cold_text_boost": 3.0,
        "infer_boost": 0.6,
        "cold_threshold": 10,
    }
    _, model, dataset = load_model(
        args.checkpoint, args.model, args.dataset, config_files, config_dict
    )
    item_counts = model.item_popularity
    stats = compute_bucket_stats(model, item_counts, model.num_text_views)
    report = format_report(stats, args.checkpoint)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
