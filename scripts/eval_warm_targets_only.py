#!/usr/bin/env python3
"""
Warm-target-only re-evaluation (0723 G1).

For each checkpoint: report all-target vs warm-only MRR/NDCG/HR@10.
Warm = test target item appeared in the train catalog.

Important: checkpoint reload does not persist the custom
``min_train_interactions`` filter used by ``two_phase_train.py``. This script
therefore rebuilds the dataloaders with the same filter before evaluating.
Warm-target metrics rank over the train catalog by default.

Usage:
  python scripts/eval_warm_targets_only.py --ckpt_dir saved/sg_beauty_tf_lr5e4_do0.3_seed42
  python scripts/eval_warm_targets_only.py --ckpt_list logs/warm_eval_ckpt_list.txt --out logs/warm_only_eval.csv
"""
from __future__ import annotations

import argparse
import csv
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
    return pths[-1]  # newest = usually Phase-B / final


def train_item_set(train_data) -> set[int]:
    """Internal item ids that appear in train interactions."""
    ds = train_data._dataset
    item_field = ds.iid_field
    # RecBole inter feat after mapping
    if hasattr(ds, "inter_feat") and item_field in ds.inter_feat:
        arr = ds.inter_feat[item_field].numpy()
        return set(int(x) for x in arr if int(x) > 0)
    # Fallback: item_num covers padding 0
    return set(range(1, ds.item_num))


def metrics_from_scores(
    user_scores: torch.Tensor,
    target: int,
    ks=(5, 10, 20),
    candidate_mask: torch.Tensor | None = None,
):
    """user_scores: [n_items]; return dict of mrr@k, ndcg@k, hit@k."""
    target_score = user_scores[target]
    if candidate_mask is not None:
        if not bool(candidate_mask[target]):
            raise ValueError(f"Warm target {target} is absent from candidate mask")
        candidate_scores = user_scores[candidate_mask]
    else:
        candidate_scores = user_scores
    # Sequential full-sort has one target per row. Counting scores above the
    # target is equivalent to sorting, but avoids an O(n_items log n_items)
    # argsort for every row.
    rank = int((candidate_scores > target_score).sum().item()) + 1
    out = {"rank": rank}
    for k in ks:
        hit = 1.0 if rank <= k else 0.0
        out[f"hit@{k}"] = hit
        out[f"mrr@{k}"] = (1.0 / rank) if rank <= k else 0.0
        out[f"ndcg@{k}"] = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
    return out


@torch.no_grad()
def eval_split(
    model,
    data_loader,
    config,
    warm_items: set[int],
    max_users: int | None = None,
    warm_candidates_only: bool = True,
):
    device = config["device"]
    n_items = data_loader._dataset.item_num
    model.eval()

    all_m = {k: [] for k in ("mrr@10", "ndcg@10", "hit@10")}
    warm_m = {k: [] for k in ("mrr@10", "ndcg@10", "hit@10")}
    n_all = n_warm = n_unseen = 0
    warm_candidate_mask = None
    if warm_candidates_only:
        warm_candidate_mask = torch.zeros(n_items, dtype=torch.bool)
        warm_candidate_mask[list(warm_items)] = True

    for batched in data_loader:
        interaction, history_index, positive_u, positive_i = batched
        interaction = interaction.to(device)
        scores = model.full_sort_predict(interaction)
        scores = scores.view(-1, n_items)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        scores = scores.cpu()

        for i in range(len(positive_u)):
            if max_users is not None and n_all >= max_users:
                break
            row = int(positive_u[i].item())
            tgt = int(positive_i[i].item())
            m = metrics_from_scores(scores[row], tgt)
            for k in all_m:
                all_m[k].append(m[k])
            n_all += 1
            if tgt in warm_items:
                warm_metrics = metrics_from_scores(
                    scores[row],
                    tgt,
                    candidate_mask=warm_candidate_mask,
                )
                for k in warm_m:
                    warm_m[k].append(warm_metrics[k])
                n_warm += 1
            else:
                n_unseen += 1
        if max_users is not None and n_all >= max_users:
            break
        torch.cuda.empty_cache()

    def avg(d):
        return {k: (float(np.mean(v)) if v else 0.0) for k, v in d.items()}

    return {
        "n_all": n_all,
        "n_warm": n_warm,
        "n_unseen": n_unseen,
        "unseen_share": n_unseen / max(n_all, 1),
        "all": avg(all_m),
        "warm": avg(warm_m),
    }


def run_one(
    ckpt_dir: Path,
    max_users: int | None,
    gpu_id: str,
    min_train_interactions: int,
    warm_candidates_only: bool,
    do_valid: bool,
):
    from recbole.data import create_dataset
    from recbole.quick_start import load_data_and_model
    from scripts.two_phase_train import _data_preparation_with_filter

    pth = pick_pth(ckpt_dir)
    print(f"\n=== {ckpt_dir.name} ← {pth.name} ===")
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(str(pth))
    if gpu_id is not None:
        config["device"] = f"cuda:{gpu_id}" if str(gpu_id).isdigit() else gpu_id
        model = model.to(config["device"])

    if min_train_interactions > 0:
        filtered_dataset = create_dataset(config)
        if filtered_dataset.item_num != dataset.item_num:
            raise RuntimeError(
                "Reloaded dataset item mapping differs from checkpoint dataset: "
                f"{filtered_dataset.item_num} != {dataset.item_num}"
            )
        train_data, valid_data, test_data = _data_preparation_with_filter(
            config,
            filtered_dataset,
            min_train_interactions=min_train_interactions,
        )
        dataset = filtered_dataset

    warm = train_item_set(train_data)
    print(
        f"train catalog size={len(warm)}, item_num={dataset.item_num}, "
        f"min_train_interactions={min_train_interactions}"
    )

    test_res = eval_split(
        model,
        test_data,
        config,
        warm,
        max_users,
        warm_candidates_only=warm_candidates_only,
    )
    valid_res = None
    if do_valid:
        valid_res = eval_split(
            model,
            valid_data,
            config,
            warm,
            max_users,
            warm_candidates_only=warm_candidates_only,
        )

    row = {
        "tag": ckpt_dir.name,
        "pth": pth.name,
        "dataset": str(config["dataset"]),
        "model": str(config["model"]),
        "min_train_interactions": min_train_interactions,
        "warm_candidate_scope": "train_catalog" if warm_candidates_only else "full_catalog",
        "test_n_all": test_res["n_all"],
        "test_n_warm": test_res["n_warm"],
        "test_unseen_share": f"{test_res['unseen_share']:.4f}",
        "test_mrr@10_all": f"{test_res['all']['mrr@10']:.6f}",
        "test_mrr@10_warm": f"{test_res['warm']['mrr@10']:.6f}",
        "test_ndcg@10_all": f"{test_res['all']['ndcg@10']:.6f}",
        "test_ndcg@10_warm": f"{test_res['warm']['ndcg@10']:.6f}",
        "test_hit@10_all": f"{test_res['all']['hit@10']:.6f}",
        "test_hit@10_warm": f"{test_res['warm']['hit@10']:.6f}",
        "valid_mrr@10_all": (
            f"{valid_res['all']['mrr@10']:.6f}" if valid_res else ""
        ),
        "valid_mrr@10_warm": (
            f"{valid_res['warm']['mrr@10']:.6f}" if valid_res else ""
        ),
    }
    print(
        f"  test all MRR={row['test_mrr@10_all']} warm={row['test_mrr@10_warm']} "
        f"unseen={row['test_unseen_share']}"
    )
    del model
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, default="")
    ap.add_argument("--ckpt_list", type=str, default="")
    ap.add_argument("--out", type=str, default="logs/warm_only_eval.csv")
    ap.add_argument("--max_users", type=int, default=0, help="0 = all users")
    ap.add_argument("--gpu_id", type=str, default="0")
    ap.add_argument(
        "--min_train_interactions",
        type=int,
        default=5,
        help="Rebuild eval loaders with the training-time history filter",
    )
    ap.add_argument(
        "--warm_full_catalog",
        action="store_true",
        help="Keep the full item catalog for warm metrics (default: train catalog only)",
    )
    ap.add_argument(
        "--skip_valid",
        action="store_true",
        help="Evaluate test only (recommended for final main-table refresh)",
    )
    args = ap.parse_args()

    dirs: list[Path] = []
    if args.ckpt_dir:
        dirs.append(Path(args.ckpt_dir))
    if args.ckpt_list:
        for line in Path(args.ckpt_list).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                dirs.append(Path(line))
    if not dirs:
        raise SystemExit("Provide --ckpt_dir or --ckpt_list")

    max_users = args.max_users if args.max_users > 0 else None
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    def write_rows(rows_to_write):
        if not rows_to_write:
            return
        keys = sorted({k for row in rows_to_write for k in row})
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows_to_write)

    rows = []
    for index, d in enumerate(dirs, 1):
        d = d if d.is_absolute() else ROOT / d
        started = time.monotonic()
        print(f"\n[PROGRESS {index}/{len(dirs)}] {d.name}")
        if not d.exists():
            print(f"SKIP missing {d}")
            rows.append(
                {
                    "tag": d.name,
                    "error": "missing",
                    "elapsed_seconds": f"{time.monotonic() - started:.1f}",
                }
            )
            write_rows(rows)
            continue
        try:
            row = run_one(
                d,
                max_users,
                args.gpu_id,
                args.min_train_interactions,
                warm_candidates_only=not args.warm_full_catalog,
                do_valid=not args.skip_valid,
            )
            row["elapsed_seconds"] = f"{time.monotonic() - started:.1f}"
            rows.append(row)
        except Exception as e:
            print(f"FAIL {d}: {e}")
            rows.append(
                {
                    "tag": d.name,
                    "error": str(e),
                    "elapsed_seconds": f"{time.monotonic() - started:.1f}",
                }
            )
        write_rows(rows)
        elapsed_min = float(rows[-1]["elapsed_seconds"]) / 60
        print(f"[PROGRESS {index}/{len(dirs)}] complete in {elapsed_min:.1f} min")

    print(f"\nWrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
