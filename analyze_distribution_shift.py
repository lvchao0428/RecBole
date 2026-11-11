#!/usr/bin/env python3
"""
Valid/Test Distribution Shift Analysis

按照老师 0717 建议：检查 valid/test 两个时间窗口的分布差异，
包括 item popularity, catalog overlap, target item age, 用户历史长度, 文本分布。

Usage:
    python scripts/analyze_distribution_shift.py --dataset Amazon_Beauty \
        --base_dir dataset
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_inter_data(dataset_path: Path):
    """Load .inter file and parse columns."""
    inter_file = None
    for f in dataset_path.iterdir():
        if f.suffix == '.inter':
            inter_file = f
            break
    if inter_file is None:
        raise FileNotFoundError(f"No .inter file in {dataset_path}")

    with open(inter_file, 'r') as fh:
        header = fh.readline().strip().split('\t')
        rows = []
        for line in fh:
            parts = line.strip().split('\t')
            rows.append(parts)

    uid_col = next(i for i, h in enumerate(header) if 'user' in h.lower() and 'id' in h.lower())
    iid_col = next(i for i, h in enumerate(header) if 'item' in h.lower() and 'id' in h.lower())
    ts_col = next((i for i, h in enumerate(header) if 'timestamp' in h.lower()), None)

    data = []
    for row in rows:
        uid = row[uid_col]
        iid = row[iid_col]
        ts = float(row[ts_col]) if ts_col is not None else 0
        data.append((uid, iid, ts))

    return data


def analyze_split(data, train_cutoff_pct=0.8, valid_cutoff_pct=0.9, min_history=5):
    """Split data by global timestamp and analyze distribution shift."""
    data_sorted = sorted(data, key=lambda x: x[2])
    n = len(data_sorted)
    train_end = int(n * train_cutoff_pct)
    valid_end = int(n * valid_cutoff_pct)

    train_data = data_sorted[:train_end]
    valid_data = data_sorted[train_end:valid_end]
    test_data = data_sorted[valid_end:]

    results = {"split_sizes": {"train": len(train_data), "valid": len(valid_data), "test": len(test_data)}}

    # --- Item popularity in train ---
    train_item_counts = Counter(iid for _, iid, _ in train_data)
    train_items = set(train_item_counts.keys())

    # --- Catalog overlap ---
    valid_items = set(iid for _, iid, _ in valid_data)
    test_items = set(iid for _, iid, _ in test_data)

    valid_new = valid_items - train_items
    test_new = test_items - train_items
    test_vs_valid_new = test_items - train_items - valid_items

    results["catalog"] = {
        "train_items": len(train_items),
        "valid_items": len(valid_items),
        "test_items": len(test_items),
        "valid_new_items": len(valid_new),
        "valid_new_ratio": len(valid_new) / max(len(valid_items), 1),
        "test_new_items": len(test_new),
        "test_new_ratio": len(test_new) / max(len(test_items), 1),
        "test_exclusive_new": len(test_vs_valid_new),
    }

    # --- Target item age (train cutoff 前首次出现距 cutoff 的时间) ---
    item_first_seen = {}
    for _, iid, ts in train_data:
        if iid not in item_first_seen:
            item_first_seen[iid] = ts

    train_cutoff_ts = train_data[-1][2] if train_data else 0

    valid_target_ages = []
    test_target_ages = []
    for _, iid, _ in valid_data:
        if iid in item_first_seen:
            valid_target_ages.append(train_cutoff_ts - item_first_seen[iid])
    for _, iid, _ in test_data:
        if iid in item_first_seen:
            test_target_ages.append(train_cutoff_ts - item_first_seen[iid])

    if valid_target_ages:
        results["target_item_age"] = {
            "valid_median": float(np.median(valid_target_ages)),
            "valid_mean": float(np.mean(valid_target_ages)),
            "test_median": float(np.median(test_target_ages)) if test_target_ages else 0,
            "test_mean": float(np.mean(test_target_ages)) if test_target_ages else 0,
        }

    # --- Target item train popularity ---
    valid_target_pop = [train_item_counts.get(iid, 0) for _, iid, _ in valid_data]
    test_target_pop = [train_item_counts.get(iid, 0) for _, iid, _ in test_data]

    results["target_popularity"] = {
        "valid_mean": float(np.mean(valid_target_pop)),
        "valid_median": float(np.median(valid_target_pop)),
        "test_mean": float(np.mean(test_target_pop)),
        "test_median": float(np.median(test_target_pop)),
        "valid_zero_count": sum(1 for x in valid_target_pop if x == 0),
        "test_zero_count": sum(1 for x in test_target_pop if x == 0),
    }

    # --- User history length (up to train cutoff) ---
    user_train_history = Counter(uid for uid, _, _ in train_data)
    eligible_users = {uid for uid, cnt in user_train_history.items() if cnt >= min_history}

    valid_users = set(uid for uid, _, _ in valid_data) & eligible_users
    test_users = set(uid for uid, _, _ in test_data) & eligible_users

    valid_hist_lens = [user_train_history[uid] for uid in valid_users]
    test_hist_lens = [user_train_history[uid] for uid in test_users]

    results["user_history"] = {
        "eligible_users": len(eligible_users),
        "valid_users": len(valid_users),
        "test_users": len(test_users),
        "valid_hist_median": float(np.median(valid_hist_lens)) if valid_hist_lens else 0,
        "valid_hist_mean": float(np.mean(valid_hist_lens)) if valid_hist_lens else 0,
        "test_hist_median": float(np.median(test_hist_lens)) if test_hist_lens else 0,
        "test_hist_mean": float(np.mean(test_hist_lens)) if test_hist_lens else 0,
    }

    # --- Item frequency bucket distribution in targets ---
    def bucket(count):
        if count == 0:
            return "unseen"
        elif count < 3:
            return "low"
        elif count < 10:
            return "mid"
        else:
            return "head"

    valid_buckets = Counter(bucket(train_item_counts.get(iid, 0)) for _, iid, _ in valid_data)
    test_buckets = Counter(bucket(train_item_counts.get(iid, 0)) for _, iid, _ in test_data)

    results["target_bucket_dist"] = {
        "valid": {k: v / max(sum(valid_buckets.values()), 1) for k, v in sorted(valid_buckets.items())},
        "test": {k: v / max(sum(test_buckets.values()), 1) for k, v in sorted(test_buckets.items())},
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Amazon_Beauty")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.base_dir is None:
        args.base_dir = str(ROOT / "dataset")

    ds_path = Path(args.base_dir) / args.dataset
    print(f"Loading data from {ds_path}...")
    data = load_inter_data(ds_path)
    print(f"  Total interactions: {len(data)}")

    print("\nAnalyzing distribution shift...")
    results = analyze_split(data)

    print("\n" + "=" * 70)
    print("DISTRIBUTION SHIFT ANALYSIS")
    print("=" * 70)

    print(f"\nSplit sizes: train={results['split_sizes']['train']}, "
          f"valid={results['split_sizes']['valid']}, test={results['split_sizes']['test']}")

    cat = results["catalog"]
    print(f"\n--- Catalog ---")
    print(f"  Train items: {cat['train_items']}")
    print(f"  Valid new items: {cat['valid_new_items']} ({cat['valid_new_ratio']:.1%})")
    print(f"  Test new items: {cat['test_new_items']} ({cat['test_new_ratio']:.1%})")

    pop = results["target_popularity"]
    print(f"\n--- Target Item Popularity (train count) ---")
    print(f"  Valid: mean={pop['valid_mean']:.1f}, median={pop['valid_median']:.0f}, zero={pop['valid_zero_count']}")
    print(f"  Test:  mean={pop['test_mean']:.1f}, median={pop['test_median']:.0f}, zero={pop['test_zero_count']}")

    uh = results["user_history"]
    print(f"\n--- User History Length ---")
    print(f"  Eligible users (≥5 train): {uh['eligible_users']}")
    print(f"  Valid users: {uh['valid_users']}, hist median={uh['valid_hist_median']:.0f}")
    print(f"  Test users: {uh['test_users']}, hist median={uh['test_hist_median']:.0f}")

    bd = results["target_bucket_dist"]
    print(f"\n--- Target Bucket Distribution ---")
    print(f"  {'Bucket':<10} {'Valid':>8} {'Test':>8} {'Shift':>8}")
    for b in ["unseen", "low", "mid", "head"]:
        v = bd["valid"].get(b, 0)
        t = bd["test"].get(b, 0)
        print(f"  {b:<10} {v:>8.1%} {t:>8.1%} {t-v:>+8.1%}")

    output_file = args.output or f"distribution_shift_{args.dataset.replace('Amazon_', '').lower()}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
