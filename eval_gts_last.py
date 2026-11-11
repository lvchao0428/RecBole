#!/usr/bin/env python3
"""
GTS-Last evaluation: re-evaluate existing checkpoints using only the LAST
test interaction per user (instead of all test interactions).

This addresses the reviewer concern about GTS-All vs GTS-Last:
- Training remains unchanged (all augmented steps used as training signal)
- Evaluation only considers the final test interaction per user (post-hoc filter)
- Warm metrics (default): target in train catalog AND rank among train-catalog
  candidates only (matches 0803 protocol lock).

Usage:
  python scripts/eval_gts_last.py --ckpt_dir saved/sg_beauty_tf_lr5e4_do0.3_seed42
  python scripts/eval_gts_last.py --ckpt_list logs/gts_last_ckpt_list.txt --out logs/gts_last_eval.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pick_pth(ckpt_dir: Path) -> Path:
    pths = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not pths:
        raise FileNotFoundError(f"No .pth in {ckpt_dir}")
    return pths[-1]


def train_item_set(train_dataset) -> set[int]:
    """Internal item ids that appear in train interactions."""
    item_field = train_dataset.iid_field
    if hasattr(train_dataset, "inter_feat") and item_field in train_dataset.inter_feat:
        arr = train_dataset.inter_feat[item_field].numpy()
        return set(int(x) for x in arr if int(x) > 0)
    return set(range(1, train_dataset.item_num))


def metrics_from_scores(
    user_scores: torch.Tensor,
    target: int,
    ks=(5, 10, 20),
    candidate_mask: torch.Tensor | None = None,
):
    """user_scores: [n_items]; optionally rank only within candidate_mask."""
    target_score = user_scores[target]
    if candidate_mask is not None:
        if not bool(candidate_mask[target]):
            raise ValueError(f"Warm target {target} is absent from candidate mask")
        candidate_scores = user_scores[candidate_mask]
    else:
        candidate_scores = user_scores
    rank = int((candidate_scores > target_score).sum().item()) + 1
    out = {"rank": rank}
    for k in ks:
        hit = 1.0 if rank <= k else 0.0
        out[f"hit@{k}"] = hit
        out[f"mrr@{k}"] = (1.0 / rank) if rank <= k else 0.0
        out[f"ndcg@{k}"] = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
    return out


@torch.no_grad()
def eval_gts_last(
    model,
    data_loader,
    config,
    warm_items: set[int] | None,
    min_train_inter: int,
    train_uid_counts: dict,
    warm_candidates_only: bool = True,
):
    """Evaluate with GTS-Last: for each user, only keep the metrics of their
    LAST test sample (highest timestamp).

    All-target metrics use the full item catalog.
    Warm-target metrics (default) rank only within the train catalog.
    """
    device = config["device"]
    n_items = data_loader._dataset.item_num
    uid_field = data_loader._dataset.uid_field
    time_field = data_loader._dataset.time_field
    model.eval()

    all_uids = data_loader._dataset.inter_feat[uid_field].numpy()
    all_times = data_loader._dataset.inter_feat[time_field].numpy()

    warm_candidate_mask = None
    if warm_candidates_only and warm_items is not None:
        warm_candidate_mask = torch.zeros(n_items, dtype=torch.bool)
        warm_candidate_mask[list(warm_items)] = True

    # user_results: uid -> (timestamp, metrics_all, metrics_warm|None, target)
    user_results = {}
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

            row_scores = scores[row_in_batch]
            m_all = metrics_from_scores(row_scores, tgt)
            m_warm = None
            if warm_items is not None and tgt in warm_items:
                m_warm = metrics_from_scores(
                    row_scores, tgt, candidate_mask=warm_candidate_mask
                )

            if uid not in user_results or ts > user_results[uid][0]:
                user_results[uid] = (ts, m_all, m_warm, tgt)

        global_idx += batch_size
        torch.cuda.empty_cache()

    all_m = {k: [] for k in ("mrr@10", "ndcg@10", "hit@10")}
    warm_m = {k: [] for k in ("mrr@10", "ndcg@10", "hit@10")}
    n_all = n_warm = 0

    for uid, (ts, m_all, m_warm, tgt) in user_results.items():
        for k in all_m:
            all_m[k].append(m_all[k])
        n_all += 1
        if m_warm is not None:
            for k in warm_m:
                warm_m[k].append(m_warm[k])
            n_warm += 1

    def avg(d):
        return {k: (float(np.mean(v)) if v else 0.0) for k, v in d.items()}

    return {
        "n_all": n_all,
        "n_warm": n_warm,
        "all": avg(all_m),
        "warm": avg(warm_m),
    }


def run_one(
    ckpt_dir: Path,
    gpu_id: str,
    min_train_inter: int,
    warm_candidates_only: bool = True,
):
    from recbole.quick_start import load_data_and_model

    warm_scope = "train_catalog" if warm_candidates_only else "full_catalog"
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
    print(f"  Train catalog={len(warm_items)} items | warm_candidate_scope={warm_scope}")

    results = eval_gts_last(
        model,
        test_data,
        config,
        warm_items,
        min_train_inter=min_train_inter,
        train_uid_counts=uid_counts,
        warm_candidates_only=warm_candidates_only,
    )
    elapsed = time.time() - t0

    print(f"  GTS-Last MRR@10: all={results['all']['mrr@10']:.6f} "
          f"warm={results['warm']['mrr@10']:.6f}")
    print(f"  n_eval={results['n_all']} users (warm={results['n_warm']})")
    print(f"  elapsed={elapsed:.1f}s")

    row = {
        "tag": ckpt_dir.name,
        "pth": pth.name,
        "dataset": str(config["dataset"]),
        "model": str(config["model"]),
        "min_train_interactions": min_train_inter,
        "eval_mode": "GTS-Last",
        "n_users": results["n_all"],
        "n_warm": results["n_warm"],
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
    ap.add_argument("--out", type=str, default="logs/gts_last_eval.csv")
    ap.add_argument("--min_train_interactions", type=int, default=5)
    ap.add_argument("--gpu_id", type=str, default="0")
    ap.add_argument(
        "--warm_full_catalog",
        action="store_true",
        help="Rank warm targets against the full item catalog (ablation). "
             "Default ranks warm targets within the train catalog only.",
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

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    warm_candidates_only = not args.warm_full_catalog
    rows = []
    for d in dirs:
        d = d if d.is_absolute() else ROOT / d
        if not d.exists():
            print(f"SKIP missing {d}")
            continue
        try:
            rows.append(
                run_one(d, args.gpu_id, args.min_train_interactions, warm_candidates_only)
            )
        except Exception as e:
            import traceback
            print(f"FAIL {d}: {e}")
            traceback.print_exc()
            rows.append({"tag": d.name, "error": str(e)})

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


if __name__ == "__main__":
    main()
