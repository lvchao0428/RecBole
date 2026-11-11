#!/usr/bin/env python3
"""
Parameter count + inference cost table.

Report: #params, text feature dim, inference time per batch for each model variant.

Usage:
  python scripts/model_cost_table.py --dataset Amazon_Beauty
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pick_pth(ckpt_dir: Path) -> Path:
    pths = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not pths:
        raise FileNotFoundError(f"No .pth in {ckpt_dir}")
    return pths[-1]


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


@torch.no_grad()
def measure_inference_time(model, data_loader, config, n_batches: int = 10):
    device = config["device"]
    model.eval()
    times = []

    for i, batched in enumerate(data_loader):
        if i >= n_batches:
            break
        interaction, history_index, positive_u, positive_i = batched
        interaction = interaction.to(device)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        scores = model.full_sort_predict(interaction)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()

        batch_size = len(positive_u)
        times.append((t1 - t0, batch_size))
        torch.cuda.empty_cache()

    total_time = sum(t for t, _ in times)
    total_samples = sum(b for _, b in times)
    return {
        "total_time_s": total_time,
        "total_samples": total_samples,
        "ms_per_sample": (total_time / max(total_samples, 1)) * 1000,
        "n_batches": len(times),
    }


def run_one(ckpt_dir: Path, gpu_id: str, n_batches: int):
    from recbole.quick_start import load_data_and_model

    pth = pick_pth(ckpt_dir)
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(str(pth))
    if gpu_id is not None:
        config["device"] = f"cuda:{gpu_id}" if str(gpu_id).isdigit() else gpu_id
        model = model.to(config["device"])

    params = count_params(model)
    timing = measure_inference_time(model, test_data, config, n_batches)

    # Determine text feature dimension
    text_dim = 0
    if hasattr(model, "text_emb_dim"):
        text_dim = model.text_emb_dim
    elif hasattr(model, "text_embedding"):
        text_dim = model.text_embedding.weight.shape[1] if hasattr(model.text_embedding, 'weight') else 0
    elif hasattr(model, "n_views"):
        text_dim = getattr(model, "view_dim", 0) * getattr(model, "n_views", 4)

    model_name = config["model"]
    variant = ckpt_dir.name

    row = {
        "tag": variant,
        "model": model_name,
        "dataset": config["dataset"],
        "params_total": params["total"],
        "params_trainable": params["trainable"],
        "params_M": f"{params['total'] / 1e6:.2f}",
        "text_feature_dim": text_dim,
        "ms_per_sample": f"{timing['ms_per_sample']:.2f}",
        "samples_measured": timing["total_samples"],
    }

    print(f"  {variant}: {params['total']/1e6:.2f}M params, "
          f"text_dim={text_dim}, {timing['ms_per_sample']:.2f}ms/sample")

    del model
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_list", type=str, required=True)
    ap.add_argument("--out", type=str, default="logs/model_cost_table.csv")
    ap.add_argument("--gpu_id", type=str, default="0")
    ap.add_argument("--n_batches", type=int, default=10)
    args = ap.parse_args()

    dirs: list[Path] = []
    for line in Path(args.ckpt_list).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            dirs.append(Path(line))
    if not dirs:
        raise SystemExit("Empty ckpt_list")

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    import csv
    rows = []
    for d in dirs:
        d = d if d.is_absolute() else ROOT / d
        if not d.exists():
            print(f"SKIP missing {d}")
            continue
        try:
            rows.append(run_one(d, args.gpu_id, args.n_batches))
        except Exception as e:
            print(f"FAIL {d}: {e}")

    if rows:
        all_keys = []
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=all_keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {out} ({len(rows)} rows)")

    # Print summary table
    print(f"\n{'Model':<40} {'Params(M)':>10} {'TextDim':>8} {'ms/sample':>10}")
    print("-" * 72)
    for r in rows:
        print(f"{r['tag']:<40} {r['params_M']:>10} {r.get('text_feature_dim','?'):>8} {r['ms_per_sample']:>10}")


if __name__ == "__main__":
    main()
