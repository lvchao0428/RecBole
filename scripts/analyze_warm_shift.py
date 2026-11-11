#!/usr/bin/env python3
"""Warm-cohort distribution shift (advisor 0820, item 2).

The main tables score only warm targets (per-user last interaction in the
window, target inside the train catalog, user has >=5 train interactions).
tab:shift / tab:bucket previously used all window interactions, which mixes
unseen targets that never enter warm MRR.  This script recomputes the
statistics inside the warm evaluation cohort, per-user LAST target, so they
can directly speak to warm-MRR degradation.

Usage: python scripts/analyze_warm_shift.py --dataset Amazon_Beauty
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_inter(ds_path: Path):
    inter_file = next(f for f in ds_path.iterdir() if f.suffix == ".inter")
    with open(inter_file) as fh:
        header = fh.readline().strip().split("\t")
        uid_c = next(i for i, h in enumerate(header) if "user" in h.lower() and "id" in h.lower())
        iid_c = next(i for i, h in enumerate(header) if "item" in h.lower() and "id" in h.lower())
        ts_c = next(i for i, h in enumerate(header) if "timestamp" in h.lower())
        data = []
        for line in fh:
            p = line.rstrip("\n").split("\t")
            data.append((p[uid_c], p[iid_c], float(p[ts_c])))
    return data


def bucket(count):
    if count == 0:
        return "unseen"
    if count < 3:
        return "low"
    if count < 10:
        return "mid"
    return "head"


def window_stats(window, train_counts, train_items, eligible, first_seen, cutoff_ts):
    """Per-user LAST target in the window, restricted to eligible users.

    Item age = seconds from the item's first train-period appearance to the
    training cutoff. Defined only for warm targets (in the train catalog).
    """
    last = {}
    for uid, iid, ts in window:
        if uid in eligible and (uid not in last or ts >= last[uid][1]):
            last[uid] = (iid, ts)
    targets = [iid for iid, _ in last.values()]
    warm = [iid for iid in targets if iid in train_items]
    pops = [train_counts[iid] for iid in warm]
    ages = [cutoff_ts - first_seen[iid] for iid in warm]
    bks = Counter(bucket(train_counts.get(iid, 0)) for iid in targets)
    n = len(targets)
    warm_bks = Counter(bucket(train_counts[iid]) for iid in warm)
    wn = max(len(warm), 1)
    age_med = float(np.median(ages)) if ages else 0.0
    age_mean = float(np.mean(ages)) if ages else 0.0
    return {
        "eligible_users_with_target": n,
        "warm_users": len(warm),
        "warm_share": len(warm) / max(n, 1),
        "warm_pop_mean": float(np.mean(pops)) if pops else 0.0,
        "warm_pop_median": float(np.median(pops)) if pops else 0.0,
        "warm_age_mean_sec": age_mean,
        "warm_age_median_sec": age_med,
        "warm_age_mean_days": age_mean / 86400.0,
        "warm_age_median_days": age_med / 86400.0,
        "alltarget_bucket": {k: bks.get(k, 0) / max(n, 1) for k in ["head", "mid", "low", "unseen"]},
        "warm_bucket": {k: warm_bks.get(k, 0) / wn for k in ["head", "mid", "low"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Amazon_Beauty")
    ap.add_argument("--min_history", type=int, default=5)
    args = ap.parse_args()

    data = sorted(load_inter(ROOT / "dataset" / args.dataset), key=lambda x: x[2])
    n = len(data)
    train = data[: int(n * 0.8)]
    valid = data[int(n * 0.8) : int(n * 0.9)]
    test = data[int(n * 0.9) :]

    train_counts = Counter(iid for _, iid, _ in train)
    train_items = set(train_counts)
    hist = Counter(uid for uid, _, _ in train)
    eligible = {u for u, c in hist.items() if c >= args.min_history}
    first_seen = {}
    for _, iid, ts in train:
        if iid not in first_seen:
            first_seen[iid] = ts
    cutoff_ts = train[-1][2] if train else 0.0

    out = {
        "dataset": args.dataset,
        "protocol": "Main TS[0.8,0.1,0.1], per-user last target, min5, warm=target in train catalog",
        "age_def": "cutoff_ts - first_train_timestamp of the warm last-target item (seconds/days)",
        "valid": window_stats(valid, train_counts, train_items, eligible, first_seen, cutoff_ts),
        "test": window_stats(test, train_counts, train_items, eligible, first_seen, cutoff_ts),
    }

    of = ROOT / f"warm_shift_{args.dataset.replace('Amazon_', '').lower()}.json"
    of.write_text(json.dumps(out, indent=2))
    paper = ROOT / "paper_recsys" / of.name
    paper.write_text(of.read_text())
    print(json.dumps(out, indent=2))
    print(f"saved -> {of}")
    print(f"saved -> {paper}")


if __name__ == "__main__":
    main()
