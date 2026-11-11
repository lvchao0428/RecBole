#!/usr/bin/env python3
"""
Leave-One-View-Out Analysis (fixed).

Previous bug: the script set ``model.text_view_indices``, but the forward path
reads ``self.active_buffer_positions`` fixed at ``__init__``. That made every
drop-view Δ exactly 0.

Correct approach for a *trained* full-MV checkpoint: keep projection dims
unchanged and **zero-mask** the dropped view's embedding buffer during eval.

Usage:
    python scripts/analyze_leave_one_view_out.py \
        --ckpt saved/sg_beauty_mv_lr5e4_do0.3_seed42 \
        --dataset Amazon_Beauty \
        --output paper_recsys/leave_one_view_out_beauty.md
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


VIEW_NAMES = ["Description", "Function", "Audience", "Style"]


def pick_pth(ckpt: Path) -> Path:
    if ckpt.is_file() and ckpt.suffix == ".pth":
        return ckpt
    pths = sorted(ckpt.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not pths:
        raise FileNotFoundError(f"No .pth in {ckpt}")
    return pths[-1]


def _view_buffers(model):
    """Return list of (buffer_name, tensor) for active MV views."""
    names = getattr(model, "text_view_buffer_names", None)
    positions = getattr(model, "active_buffer_positions", None)
    if not names or positions is None:
        raise RuntimeError("Model is not a multi-view MV model (missing view buffers)")
    out = []
    for pos in positions:
        name = names[pos]
        out.append((name, getattr(model, name)))
    return out


class ViewMask:
    """Temporarily zero selected active views (by active index 0..n_active-1)."""

    def __init__(self, model, drop_active_indices: set[int]):
        self.model = model
        self.drop = drop_active_indices
        self._backup: list[tuple[str, torch.Tensor]] = []

    def __enter__(self):
        bufs = _view_buffers(self.model)
        for i, (name, tensor) in enumerate(bufs):
            if i in self.drop:
                self._backup.append((name, tensor.detach().clone()))
                tensor.zero_()
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, data in self._backup:
            getattr(self.model, name).copy_(data)
        self._backup.clear()
        return False


@torch.no_grad()
def evaluate_mrr(model, test_data, config, max_users: int | None = None, ks=(10,)):
    device = config["device"]
    n_items = test_data._dataset.item_num
    model.eval()

    mrr = {k: [] for k in ks}
    n = 0
    for batched in test_data:
        interaction, history_index, positive_u, positive_i = batched
        interaction = interaction.to(device)
        scores = model.full_sort_predict(interaction)
        scores = scores.view(-1, n_items)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        scores = scores.cpu()

        for i in range(len(positive_u)):
            if max_users is not None and n >= max_users:
                break
            row = int(positive_u[i].item())
            tgt = int(positive_i[i].item())
            target_score = scores[row, tgt]
            rank = int((scores[row] > target_score).sum().item()) + 1
            for k in ks:
                mrr[k].append((1.0 / rank) if rank <= k else 0.0)
            n += 1
        if max_users is not None and n >= max_users:
            break
        torch.cuda.empty_cache()

    return {f"mrr@{k}": (float(np.mean(v)) if v else 0.0) for k, v in mrr.items()}, n


def _eval_gts_last_warm(model, test_data, config, train_data, min_train_inter: int):
    """Same Last/warm filter as scripts/eval_gts_last.py (main-table protocol)."""
    from scripts.eval_gts_last import eval_gts_last, train_item_set

    train_dataset = train_data._dataset
    uid_field = train_dataset.uid_field
    uid_counts = Counter(int(u) for u in train_dataset.inter_feat[uid_field].numpy())
    warm_items = train_item_set(train_dataset)
    results = eval_gts_last(
        model,
        test_data,
        config,
        warm_items,
        min_train_inter=min_train_inter,
        train_uid_counts=uid_counts,
    )
    return results["warm"]["mrr@10"], results["n_warm"], results["n_all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="Amazon_Beauty")
    ap.add_argument("--output", type=str, default="")
    ap.add_argument("--max_users", type=int, default=0, help="0 = all test users")
    ap.add_argument("--gpu_id", type=str, default="0")
    ap.add_argument(
        "--min_train_interactions",
        type=int,
        default=5,
        help="Rebuild eval loader with the training-time history filter",
    )
    ap.add_argument(
        "--protocol",
        choices=["all", "gts_last_warm"],
        default="all",
        help="all = RecBole default test loader (GTS-All). "
             "gts_last_warm = last target per user, warm MRR@10 (main table).",
    )
    ap.add_argument(
        "--csv",
        type=str,
        default="",
        help="Append rows to this CSV (created with header if missing).",
    )
    args = ap.parse_args()

    from recbole.data import create_dataset
    from recbole.quick_start import load_data_and_model
    from scripts.two_phase_train import _data_preparation_with_filter

    pth = pick_pth(Path(args.ckpt))
    print("=" * 60)
    print("LEAVE-ONE-VIEW-OUT (zero-mask active views)")
    print("=" * 60)
    print(f"Loading: {pth}")
    print(f"Protocol: {args.protocol}")

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(str(pth))
    if args.gpu_id is not None:
        config["device"] = f"cuda:{args.gpu_id}" if str(args.gpu_id).isdigit() else args.gpu_id
        model = model.to(config["device"])
    # GTS-Last warm matches eval_gts_last.py: do not rebuild the loader; filter inside eval.
    if args.protocol == "all" and args.min_train_interactions > 0:
        filtered_dataset = create_dataset(config)
        if filtered_dataset.item_num != dataset.item_num:
            raise RuntimeError(
                "Reloaded dataset item mapping differs from checkpoint dataset: "
                f"{filtered_dataset.item_num} != {dataset.item_num}"
            )
        train_data, valid_data, test_data = _data_preparation_with_filter(
            config,
            filtered_dataset,
            min_train_interactions=args.min_train_interactions,
        )
    model.eval()

    active = list(getattr(model, "active_view_indices", list(range(4))))
    n_active = len(active)
    if n_active < 2:
        raise RuntimeError(f"Need ≥2 active views for LOVO, got {active}")

    max_users = args.max_users if args.max_users > 0 else None
    print(
        f"Dataset={args.dataset} active_views={active} max_users={max_users or 'all'} "
        f"min_train_interactions={args.min_train_interactions}"
    )

    def run_eval():
        if args.protocol == "gts_last_warm":
            mrr, n_warm, n_all = _eval_gts_last_warm(
                model, test_data, config, train_data, args.min_train_interactions
            )
            return mrr, n_warm, n_all
        full_m, n_eval = evaluate_mrr(model, test_data, config, max_users)
        return full_m["mrr@10"], n_eval, n_eval

    print("\nFull model (all active views)...")
    with ViewMask(model, set()):
        mrr_full, n_eval, n_all = run_eval()
    print(f"  Full MRR@10 = {mrr_full:.6f} (n_warm={n_eval}, n_all={n_all})")

    proto_line = (
        f"Protocol: `{args.protocol}`, `min_train_interactions={args.min_train_interactions}`"
    )
    lines = [
        "# Leave-One-View-Out Analysis\n",
        f"Checkpoint: `{args.ckpt}`  \n",
        f"Method: **zero-mask** dropped view buffer (projection dims unchanged)  \n",
        f"{proto_line}  \n",
        f"Active view indices: `{active}`  ·  eval users (warm): {n_eval}  ·  all-last: {n_all}\n",
        "| Config | Views Used | MRR@10 | Δ vs Full | Contribution |",
        "|--------|-----------|:------:|:---------:|:------------:|",
        f"| **Full ({n_active} views)** | {','.join(map(str, active))} | **{mrr_full:.6f}** | — | — |",
    ]

    view_results = {}
    for drop_i, view_idx in enumerate(active):
        remaining = [v for j, v in enumerate(active) if j != drop_i]
        name = VIEW_NAMES[view_idx] if view_idx < len(VIEW_NAMES) else f"view_{view_idx}"
        print(f"\nDrop view_{view_idx} ({name})...")
        with ViewMask(model, {drop_i}):
            mrr, _, _ = run_eval()
        delta = mrr - mrr_full
        contrib = -delta
        view_results[view_idx] = {"mrr": mrr, "delta": delta, "contribution": contrib, "name": name}
        print(f"  MRR@10 = {mrr:.6f} (Δ = {delta:+.6f})")
        lines.append(
            f"| Drop view_{view_idx} ({name}) | {','.join(map(str, remaining))} | "
            f"{mrr:.6f} | {delta:+.6f} | {contrib:+.6f} |"
        )

    lines.append("\n## View Contribution Ranking\n")
    lines.append("| Rank | View | Name | Contribution (drop hurts by) |")
    lines.append("|:----:|:----:|------|:----------------------------:|")
    ranked = sorted(view_results.items(), key=lambda x: x[1]["contribution"], reverse=True)
    for rank, (idx, data) in enumerate(ranked, 1):
        lines.append(f"| {rank} | view_{idx} | {data['name']} | {data['contribution']:+.6f} |")

    lines.append("\n## Interpretation\n")
    spreads = [d["contribution"] for _, d in ranked]
    if max(spreads) - min(spreads) < 0.0005 and max(abs(x) for x in spreads) < 0.0005:
        lines.append(
            "- Dropping any single view barely changes MRR → **high redundancy** "
            "(consistent with high inter-view CKA and low concat effective rank)."
        )
    elif max(spreads) - min(spreads) < 0.0005:
        lines.append("- All views have similar contribution → **redundant / interchangeable**.")
    else:
        lines.append(f"- Most important: view_{ranked[0][0]} ({ranked[0][1]['name']})")
        lines.append(f"- Least important: view_{ranked[-1][0]} ({ranked[-1][1]['name']})")

    out = Path(args.output) if args.output else ROOT / "paper_recsys" / (
        f"leave_one_view_out_{args.dataset.replace('Amazon_', '').lower()}.md"
    )
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\nReport saved to: {out}")

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = ROOT / csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        seed_m = re.search(r"seed(\d+)", Path(args.ckpt).name)
        seed = seed_m.group(1) if seed_m else ""
        rows = [
            {
                "dataset": args.dataset,
                "seed": seed,
                "ckpt": Path(args.ckpt).name,
                "protocol": args.protocol,
                "config": "full",
                "view": "all",
                "mrr10_warm": f"{mrr_full:.6f}",
                "delta": "0",
                "n_warm": n_eval,
                "n_all": n_all,
            }
        ]
        for idx, data in view_results.items():
            rows.append(
                {
                    "dataset": args.dataset,
                    "seed": seed,
                    "ckpt": Path(args.ckpt).name,
                    "protocol": args.protocol,
                    "config": f"drop_{idx}",
                    "view": data["name"],
                    "mrr10_warm": f"{data['mrr']:.6f}",
                    "delta": f"{data['delta']:+.6f}",
                    "n_warm": n_eval,
                    "n_all": n_all,
                }
            )
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(rows)
        print(f"CSV appended: {csv_path}")


if __name__ == "__main__":
    main()
