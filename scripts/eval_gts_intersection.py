#!/usr/bin/env python3
"""Last-vs-Random intersection-user sanity check (advisor 0820, item 3).

Concern: because warm filtering happens AFTER target selection, GTS-Last and
GTS-Random may evaluate different eligible-user cohorts, so the Last/Random
ranking flip could partly be a cohort-composition effect rather than a pure
target-selection effect.

Check: in a single forward pass per checkpoint we compute, for every user,
(a) the Last target and (b) the Random target for each sampling seed, with
per-sample metrics.  We then restrict BOTH protocols to the intersection of
their warm-user sets (users whose Last target AND Random target are both in
the train catalog) and re-aggregate.  If the flip persists on the identical
cohort, it is purely target selection.

Warm metrics follow the paper protocol: target in train catalog, ranked
against train-catalog candidates only.  (Full-catalog warm numbers are also
recorded for cross-checking older GTS-Last CSVs.)

Usage:
  python scripts/eval_gts_intersection.py \
      --ckpt_list logs/gts_last_ckpt_list.txt \
      --out logs/gts_intersection_eval.csv \
      --sample_seeds 42,123,456
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
    if ckpt_dir.is_file() and ckpt_dir.suffix == ".pth":
        return ckpt_dir
    pths = list(ckpt_dir.glob("*.pth"))
    if not pths:
        raise FileNotFoundError(f"No .pth in {ckpt_dir}")
    # Phase-B is larger than Phase-A; prefer size over mtime.
    return max(pths, key=lambda p: p.stat().st_size)


def train_item_set(train_dataset) -> set[int]:
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
def collect_per_user_candidates(
    model,
    data_loader,
    config,
    warm_items: set[int],
    min_train_inter: int,
    train_uid_counts: dict,
):
    """One forward pass; per user, per test sample: (ts, tgt, metrics).

    Candidates are stored in dataset iteration order so that the Last
    reduction can replicate eval_gts_last.py exactly (ts > current wins,
    first occurrence wins timestamp ties).
    """
    device = config["device"]
    n_items = data_loader._dataset.item_num
    uid_field = data_loader._dataset.uid_field
    time_field = data_loader._dataset.time_field
    model.eval()

    all_uids = data_loader._dataset.inter_feat[uid_field].numpy()
    all_times = data_loader._dataset.inter_feat[time_field].numpy()

    warm_candidate_mask = torch.zeros(n_items, dtype=torch.bool)
    warm_candidate_mask[list(warm_items)] = True

    # uid -> list[(ts, tgt, m_full, m_warm_masked|None)]
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

            row_scores = scores[row_in_batch]
            m_full = metrics_from_scores(row_scores, tgt)
            m_warm = None
            if tgt in warm_items:
                m_warm = metrics_from_scores(
                    row_scores, tgt, candidate_mask=warm_candidate_mask
                )
            user_cands[uid].append((ts, tgt, m_full, m_warm))

        global_idx += batch_size
        torch.cuda.empty_cache()

    return user_cands


def pick_last(cands):
    """Replicates eval_gts_last.py: iterate in dataset order, replace on ts >."""
    best = None
    for c in cands:
        if best is None or c[0] > best[0]:
            best = c
    return best


def pick_random(cands, sample_seed: int, uid: int):
    """Replicates eval_gts_random.py RNG and ordering."""
    rng = np.random.RandomState((int(sample_seed) + int(uid) * 10007) % (2**31 - 1))
    cands_sorted = sorted(cands, key=lambda x: (x[0], x[1]))
    return cands_sorted[int(rng.randint(0, len(cands_sorted)))]


def mean_of(selection, key="mrr@10", which="warm"):
    """selection: list of (ts, tgt, m_full, m_warm)."""
    vals = []
    for _, _, m_full, m_warm in selection:
        m = m_warm if which == "warm" else m_full
        if m is not None:
            vals.append(m[key])
    return float(np.mean(vals)) if vals else 0.0


def run_one(ckpt_dir: Path, gpu_id: str, min_train_inter: int, sample_seeds: list[int]):
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
    train_uids = train_dataset.inter_feat[uid_field].numpy()
    uid_counts = Counter(int(u) for u in train_uids)
    warm_items = train_item_set(train_dataset)
    print(f"  Train catalog={len(warm_items)} items")

    user_cands = collect_per_user_candidates(
        model, test_data, config, warm_items,
        min_train_inter=min_train_inter, train_uid_counts=uid_counts,
    )
    n_eligible = len(user_cands)
    print(f"  Eligible users (min{min_train_inter}): {n_eligible}")

    # Per-user Last selection (shared across seeds)
    last_sel = {uid: pick_last(c) for uid, c in user_cands.items()}
    warm_last = {uid for uid, s in last_sel.items() if s[3] is not None}

    rows = []
    for seed in sample_seeds:
        rand_sel = {uid: pick_random(c, seed, uid) for uid, c in user_cands.items()}
        warm_rand = {uid for uid, s in rand_sel.items() if s[3] is not None}
        inter = warm_last & warm_rand

        def agg(sel, uids, which="warm"):
            picked = [sel[u] for u in uids]
            return {
                k: mean_of(picked, key=k, which=which)
                for k in ("mrr@10", "ndcg@10", "hit@10")
            }

        # Full warm cohorts (reproduce published numbers)
        last_full_cohort = agg(last_sel, warm_last)
        rand_full_cohort = agg(rand_sel, warm_rand)
        # Intersection cohort
        last_inter = agg(last_sel, inter)
        rand_inter = agg(rand_sel, inter)
        # Cross-check: Last warm on full catalog scope (older gts_last CSVs)
        last_warm_fullcat = float(np.mean(
            [last_sel[u][2]["mrr@10"] for u in warm_last]
        )) if warm_last else 0.0

        print(
            f"  seed={seed}: |W_last|={len(warm_last)} |W_rand|={len(warm_rand)} "
            f"|inter|={len(inter)} ({len(inter)/max(n_eligible,1):.1%} of eligible)"
        )
        print(
            f"    warm MRR@10  Last full-cohort={last_full_cohort['mrr@10']:.6f} "
            f"Random full-cohort={rand_full_cohort['mrr@10']:.6f}"
        )
        print(
            f"    warm MRR@10  Last intersection={last_inter['mrr@10']:.6f} "
            f"Random intersection={rand_inter['mrr@10']:.6f}"
        )

        rows.append({
            "tag": ckpt_dir.name,
            "pth": pth.name,
            "dataset": str(config["dataset"]),
            "model": str(config["model"]),
            "min_train_interactions": min_train_inter,
            "sample_seed": seed,
            "n_eligible": n_eligible,
            "n_warm_last": len(warm_last),
            "n_warm_random": len(warm_rand),
            "n_intersection": len(inter),
            "last_mrr@10_warm_cohort": f"{last_full_cohort['mrr@10']:.6f}",
            "random_mrr@10_warm_cohort": f"{rand_full_cohort['mrr@10']:.6f}",
            "last_mrr@10_intersection": f"{last_inter['mrr@10']:.6f}",
            "random_mrr@10_intersection": f"{rand_inter['mrr@10']:.6f}",
            "last_ndcg@10_intersection": f"{last_inter['ndcg@10']:.6f}",
            "random_ndcg@10_intersection": f"{rand_inter['ndcg@10']:.6f}",
            "last_hit@10_intersection": f"{last_inter['hit@10']:.6f}",
            "random_hit@10_intersection": f"{rand_inter['hit@10']:.6f}",
            "last_mrr@10_warm_fullcatalog_scope": f"{last_warm_fullcat:.6f}",
            "elapsed_seconds": f"{time.time() - t0:.1f}",
        })

    del model
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, default="")
    ap.add_argument("--ckpt_list", type=str, default="")
    ap.add_argument("--out", type=str, default="logs/gts_intersection_eval.csv")
    ap.add_argument("--min_train_interactions", type=int, default=5)
    ap.add_argument("--sample_seeds", type=str, default="42,123,456")
    ap.add_argument("--gpu_id", type=str, default="0")
    args = ap.parse_args()

    seeds = [int(s) for s in args.sample_seeds.split(",") if s.strip()]

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

    existing = {}
    if out.exists():
        with out.open() as fh:
            for r in csv.DictReader(fh):
                if r.get("tag") and "error" not in r:
                    existing[(r["tag"], r.get("sample_seed", ""))] = r
        print(f"Resume: {len(existing)} existing rows in {out}")

    rows = list(existing.values())
    done_tags = {t for t, _ in existing}

    for d in dirs:
        d = d if d.is_absolute() else ROOT / d
        if d.name in done_tags:
            print(f"SKIP done {d.name}")
            continue
        if not d.exists():
            print(f"SKIP missing {d}")
            continue
        try:
            rows.extend(run_one(d, args.gpu_id, args.min_train_interactions, seeds))
            all_keys = []
            for r in rows:
                for k in r:
                    if k not in all_keys:
                        all_keys.append(k)
            with out.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=all_keys)
                w.writeheader()
                w.writerows(rows)
            print(f"  checkpoint -> {out} ({len(rows)} rows)")
        except Exception as e:
            import traceback
            print(f"FAIL {d}: {e}")
            traceback.print_exc()
            rows.append({"tag": d.name, "error": str(e)})

    print(f"\nWrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
