#!/usr/bin/env python3
"""
GTS-Random evaluation: re-evaluate existing checkpoints using ONE randomly
sampled test interaction per user (instead of all successive test targets).

Protocol notes (aligned with 0803 advisor guidance):
- Training unchanged (Successive-All augmentation for SASRec training).
- Evaluation: among each user's post-cutoff test targets, sample exactly one
  with a fixed per-user RNG seed for reproducibility.
- Also reports warm-target metrics (target in train catalog).

Usage:
  python scripts/eval_gts_random.py --ckpt_dir saved/sg_beauty_tf_lr5e4_do0.3_seed42
  python scripts/eval_gts_random.py \\
      --ckpt_list logs/gts_last_ckpt_list.txt \\
      --out logs/gts_random_eval.csv \\
      --sample_seed 42
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
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


def train_item_set(train_dataset) -> set[int]:
    item_field = train_dataset.iid_field
    if hasattr(train_dataset, "inter_feat") and item_field in train_dataset.inter_feat:
        arr = train_dataset.inter_feat[item_field].numpy()
        return set(int(x) for x in arr if int(x) > 0)
    return set(range(1, train_dataset.item_num))


def metrics_from_scores(user_scores: torch.Tensor, target: int, ks=(5, 10, 20)):
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
def eval_gts_random(
    model,
    data_loader,
    config,
    warm_items: set[int] | None,
    min_train_inter: int,
    train_uid_counts: dict,
    sample_seed: int,
):
    """Evaluate with GTS-Random: collect all valid test samples per user, then
    pick exactly one with a deterministic RNG (seed = sample_seed + uid).
    """
    device = config["device"]
    n_items = data_loader._dataset.item_num
    uid_field = data_loader._dataset.uid_field
    time_field = data_loader._dataset.time_field
    model.eval()

    all_uids = data_loader._dataset.inter_feat[uid_field].numpy()
    all_times = data_loader._dataset.inter_feat[time_field].numpy()

    # uid -> list[(timestamp, metrics, target)]
    user_cands: dict[int, list] = defaultdict(list)
    global_idx = 0

    for batched in data_loader:
        interaction, history_index, positive_u, positive_i = batched
        interaction = interaction.to(device)
        scores = model.full_sort_predict(interaction)
        scores = scores.view(-1, n_items)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        scores = scores.cpu()

        batch_size = scores.shape[0]
        for i in range(len(positive_u)):
            row_in_batch = int(positive_u[i].item())
            tgt = int(positive_i[i].item())
            dataset_idx = global_idx + row_in_batch
            if dataset_idx >= len(all_uids):
                continue

            uid = int(all_uids[dataset_idx])
            ts = float(all_times[dataset_idx])

            if min_train_inter > 0 and train_uid_counts.get(uid, 0) < min_train_inter:
                continue

            m = metrics_from_scores(scores[row_in_batch], tgt)
            user_cands[uid].append((ts, m, tgt))

        global_idx += batch_size
        torch.cuda.empty_cache()

    all_m = {k: [] for k in ("mrr@10", "ndcg@10", "hit@10")}
    warm_m = {k: [] for k in ("mrr@10", "ndcg@10", "hit@10")}
    n_all = n_warm = 0
    n_cands = []

    for uid, cands in user_cands.items():
        if not cands:
            continue
        n_cands.append(len(cands))
        # Per-user deterministic draw (order-invariant across user iteration)
        rng = np.random.RandomState((int(sample_seed) + int(uid) * 10007) % (2**31 - 1))
        # Sort by timestamp then target for stable indexing before draw
        cands_sorted = sorted(cands, key=lambda x: (x[0], x[2]))
        ts, m, tgt = cands_sorted[int(rng.randint(0, len(cands_sorted)))]

        for k in all_m:
            all_m[k].append(m[k])
        n_all += 1
        if warm_items is not None and tgt in warm_items:
            for k in warm_m:
                warm_m[k].append(m[k])
            n_warm += 1

    def avg(d):
        return {k: (float(np.mean(v)) if v else 0.0) for k, v in d.items()}

    return {
        "n_all": n_all,
        "n_warm": n_warm,
        "mean_cands_per_user": float(np.mean(n_cands)) if n_cands else 0.0,
        "all": avg(all_m),
        "warm": avg(warm_m),
    }


def run_one(ckpt_dir: Path, gpu_id: str, min_train_inter: int, sample_seed: int,
            warm_scope: str = "train_catalog"):
    from recbole.quick_start import load_data_and_model

    pth = pick_pth(ckpt_dir)
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_dir.name} <- {pth.name}")
    print(f"{'='*60}")

    t0 = time.time()
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(str(pth))
    if gpu_id is not None:
        config["device"] = f"cuda:{gpu_id}" if str(gpu_id).isdigit() else gpu_id
        model = model.to(config["device"])

    train_dataset = train_data._dataset
    uid_field = dataset.uid_field

    n_test = len(test_data._dataset.inter_feat[uid_field])
    print(f"  Test samples (All mode): {n_test}")

    train_uids = train_dataset.inter_feat[uid_field].numpy()
    uid_counts = Counter(int(u) for u in train_uids)
    warm_items = train_item_set(train_dataset)
    print(f"  Train catalog={len(warm_items)} items | sample_seed={sample_seed}")

    results = eval_gts_random(
        model, test_data, config, warm_items,
        min_train_inter=min_train_inter,
        train_uid_counts=uid_counts,
        sample_seed=sample_seed,
    )
    elapsed = time.time() - t0

    print(f"  GTS-Random MRR@10: all={results['all']['mrr@10']:.6f} "
          f"warm={results['warm']['mrr@10']:.6f}")
    print(f"  n_eval={results['n_all']} users (warm={results['n_warm']}) "
          f"mean_cands/user={results['mean_cands_per_user']:.2f}")
    print(f"  elapsed={elapsed:.1f}s")

    row = {
        "tag": ckpt_dir.name,
        "pth": pth.name,
        "dataset": str(config["dataset"]),
        "model": str(config["model"]),
        "min_train_interactions": min_train_inter,
        "eval_mode": "GTS-Random",
        "sample_seed": sample_seed,
        "n_users": results["n_all"],
        "n_warm": results["n_warm"],
        "mean_cands_per_user": f"{results['mean_cands_per_user']:.4f}",
        "test_mrr@10_all": f"{results['all']['mrr@10']:.6f}",
        "test_mrr@10_warm": f"{results['warm']['mrr@10']:.6f}",
        "test_ndcg@10_all": f"{results['all']['ndcg@10']:.6f}",
        "test_ndcg@10_warm": f"{results['warm']['ndcg@10']:.6f}",
        "test_hit@10_all": f"{results['all']['hit@10']:.6f}",
        "test_hit@10_warm": f"{results['warm']['hit@10']:.6f}",
        "elapsed_seconds": f"{elapsed:.1f}",
        "warm_candidate_scope": warm_scope,
    }

    del model
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, default="")
    ap.add_argument("--ckpt_list", type=str, default="")
    ap.add_argument("--out", type=str, default="logs/gts_random_eval.csv")
    ap.add_argument("--min_train_interactions", type=int, default=5)
    ap.add_argument("--sample_seed", type=int, default=42,
                    help="Base seed for per-user Random draws")
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

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resume-friendly: load existing rows by tag
    existing = {}
    if out.exists():
        with out.open() as fh:
            for r in csv.DictReader(fh):
                if r.get("tag") and "error" not in r:
                    existing[r["tag"]] = r
        print(f"Resume: {len(existing)} existing rows in {out}")

    rows = list(existing.values())
    done_tags = set(existing)

    for d in dirs:
        d = d if d.is_absolute() else ROOT / d
        if d.name in done_tags:
            print(f"SKIP done {d.name}")
            continue
        if not d.exists():
            print(f"SKIP missing {d}")
            continue
        try:
            row = run_one(d, args.gpu_id, args.min_train_interactions, args.sample_seed)
            rows.append(row)
            done_tags.add(d.name)
            # Incremental write
            all_keys = []
            for r in rows:
                for k in r:
                    if k not in all_keys:
                        all_keys.append(k)
            with out.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=all_keys)
                w.writeheader()
                w.writerows(rows)
            print(f"  checkpoint → {out} ({len(rows)} rows)")
        except Exception as e:
            import traceback
            print(f"FAIL {d}: {e}")
            traceback.print_exc()
            rows.append({"tag": d.name, "error": str(e)})

    print(f"\nWrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
