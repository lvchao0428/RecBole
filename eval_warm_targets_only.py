#!/usr/bin/env python3
"""
Warm-target-only re-evaluation (0723 G1).

For each checkpoint: report all-target vs warm-only MRR/NDCG/HR@10.
Warm = test target item appeared in the train catalog.

Usage:
  python scripts/eval_warm_targets_only.py --ckpt_dir saved/sg_beauty_tf_lr5e4_do0.3_seed42
  python scripts/eval_warm_targets_only.py --ckpt_list logs/warm_eval_ckpt_list.txt --out logs/warm_only_eval.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
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


def metrics_from_scores(user_scores: torch.Tensor, target: int, ks=(5, 10, 20)):
    """user_scores: [n_items]; return dict of mrr@k, ndcg@k, hit@k."""
    ordered = torch.argsort(user_scores, descending=True)
    rank = int((ordered == target).nonzero(as_tuple=True)[0].item()) + 1
    out = {"rank": rank}
    for k in ks:
        hit = 1.0 if rank <= k else 0.0
        out[f"hit@{k}"] = hit
        out[f"mrr@{k}"] = (1.0 / rank) if rank <= k else 0.0
        out[f"ndcg@{k}"] = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
    return out


@torch.no_grad()
def eval_split(model, data_loader, config, warm_items: set[int], max_users: int | None = None):
    device = config["device"]
    n_items = data_loader._dataset.item_num
    model.eval()

    all_m = {k: [] for k in ("mrr@10", "ndcg@10", "hit@10")}
    warm_m = {k: [] for k in ("mrr@10", "ndcg@10", "hit@10")}
    n_all = n_warm = n_unseen = 0

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
                for k in warm_m:
                    warm_m[k].append(m[k])
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


def run_one(ckpt_dir: Path, max_users: int | None, gpu_id: str):
    from recbole.quick_start import load_data_and_model

    pth = pick_pth(ckpt_dir)
    print(f"\n=== {ckpt_dir.name} ← {pth.name} ===")
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(str(pth))
    if gpu_id is not None:
        config["device"] = f"cuda:{gpu_id}" if str(gpu_id).isdigit() else gpu_id
        model = model.to(config["device"])

    warm = train_item_set(train_data)
    print(f"train catalog size={len(warm)}, item_num={dataset.item_num}")

    test_res = eval_split(model, test_data, config, warm, max_users)
    valid_res = eval_split(model, valid_data, config, warm, max_users)

    row = {
        "tag": ckpt_dir.name,
        "pth": pth.name,
        "dataset": str(config["dataset"]),
        "model": str(config["model"]),
        "test_n_all": test_res["n_all"],
        "test_n_warm": test_res["n_warm"],
        "test_unseen_share": f"{test_res['unseen_share']:.4f}",
        "test_mrr@10_all": f"{test_res['all']['mrr@10']:.6f}",
        "test_mrr@10_warm": f"{test_res['warm']['mrr@10']:.6f}",
        "test_ndcg@10_all": f"{test_res['all']['ndcg@10']:.6f}",
        "test_ndcg@10_warm": f"{test_res['warm']['ndcg@10']:.6f}",
        "test_hit@10_all": f"{test_res['all']['hit@10']:.6f}",
        "test_hit@10_warm": f"{test_res['warm']['hit@10']:.6f}",
        "valid_mrr@10_all": f"{valid_res['all']['mrr@10']:.6f}",
        "valid_mrr@10_warm": f"{valid_res['warm']['mrr@10']:.6f}",
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

    rows = []
    for d in dirs:
        d = d if d.is_absolute() else ROOT / d
        if not d.exists():
            print(f"SKIP missing {d}")
            continue
        try:
            rows.append(run_one(d, max_users, args.gpu_id))
        except Exception as e:
            print(f"FAIL {d}: {e}")
            rows.append({"tag": d.name, "error": str(e)})

    if rows:
        keys = sorted({k for r in rows for k in r})
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
