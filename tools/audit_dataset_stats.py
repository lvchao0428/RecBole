#!/usr/bin/env python3
"""
Dataset audit script for WSDM submission.
Verifies 5-core filtering, timestamp ranges, sequence length distribution,
and proposes GTS 80/10/10 cutoff dates.

Usage:
    python tools/audit_dataset_stats.py --data_dir dataset/
    python tools/audit_dataset_stats.py --data_dir dataset/ --datasets Amazon_Beauty Amazon_Toys_and_Games
"""
import argparse
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd


def load_inter(dataset_dir):
    """Load .inter file from a RecBole dataset directory."""
    inter_files = [f for f in os.listdir(dataset_dir) if f.endswith(".inter")]
    if not inter_files:
        raise FileNotFoundError(f"No .inter file found in {dataset_dir}")
    path = os.path.join(dataset_dir, inter_files[0])
    with open(path, "r") as f:
        header = f.readline().strip()
    cols = []
    dtypes = {}
    for field_type in header.split("\t"):
        field, ftype = field_type.split(":")
        cols.append(field)
        if ftype == "token":
            dtypes[field] = str
        elif ftype == "float":
            dtypes[field] = float
    df = pd.read_csv(path, sep="\t", skiprows=1, names=cols, dtype=dtypes)
    return df, path


def audit_dataset(dataset_dir, dataset_name):
    """Run full audit on one dataset."""
    print(f"\n{'='*70}")
    print(f"  AUDIT: {dataset_name}")
    print(f"  Path:  {dataset_dir}")
    print(f"{'='*70}")

    df, inter_path = load_inter(dataset_dir)
    print(f"\nLoaded {inter_path}: {len(df):,} rows, columns: {list(df.columns)}")

    uid_col = "user_id"
    iid_col = "item_id"
    time_col = "timestamp"

    for c in [uid_col, iid_col]:
        if c not in df.columns:
            print(f"  WARNING: column '{c}' not found, skipping audit")
            return

    n_users = df[uid_col].nunique()
    n_items = df[iid_col].nunique()
    n_inters = len(df)
    density = n_inters / (n_users * n_items) * 100

    print(f"\n--- Basic Statistics ---")
    print(f"  #Users:        {n_users:>12,}")
    print(f"  #Items:        {n_items:>12,}")
    print(f"  #Interactions: {n_inters:>12,}")
    print(f"  Density:       {density:>12.6f}%")
    print(f"  Avg inter/user:{n_inters/n_users:>12.2f}")
    print(f"  Avg inter/item:{n_inters/n_items:>12.2f}")

    # Per-user interaction counts
    user_counts = df[uid_col].value_counts()
    # Per-item interaction counts
    item_counts = df[iid_col].value_counts()

    print(f"\n--- Per-User Interaction Distribution ---")
    print(f"  Min:    {user_counts.min()}")
    print(f"  Max:    {user_counts.max()}")
    print(f"  Mean:   {user_counts.mean():.2f}")
    print(f"  Median: {user_counts.median():.1f}")
    for k in [1, 2, 3, 4, 5, 10, 20, 50]:
        n = (user_counts >= k).sum()
        pct = n / n_users * 100
        print(f"  Users with >= {k:>2} interactions: {n:>10,} ({pct:5.1f}%)")

    print(f"\n--- Per-Item Interaction Distribution ---")
    print(f"  Min:    {item_counts.min()}")
    print(f"  Max:    {item_counts.max()}")
    print(f"  Mean:   {item_counts.mean():.2f}")
    print(f"  Median: {item_counts.median():.1f}")
    for k in [1, 2, 3, 5, 10, 50]:
        n = (item_counts >= k).sum()
        pct = n / n_items * 100
        print(f"  Items with >= {k:>2} interactions: {n:>10,} ({pct:5.1f}%)")

    # 5-core verification
    print(f"\n--- 5-Core Filtering Verification ---")
    user_min = user_counts.min()
    item_min = item_counts.min()
    is_user_5core = user_min >= 5
    is_item_5core = item_min >= 5
    print(f"  Min user interactions: {user_min}")
    print(f"  Min item interactions: {item_min}")
    print(f"  User 5-core: {'YES' if is_user_5core else 'NO'}")
    print(f"  Item 5-core: {'YES' if is_item_5core else 'NO'}")
    if is_user_5core and is_item_5core:
        print(f"  ==> Dataset IS 5-core filtered")
    else:
        print(f"  ==> Dataset is NOT 5-core filtered!")
        if not is_user_5core:
            n_under5 = (user_counts < 5).sum()
            print(f"      {n_under5:,} users ({n_under5/n_users*100:.1f}%) have <5 interactions")
        if not is_item_5core:
            n_under5 = (item_counts < 5).sum()
            print(f"      {n_under5:,} items ({n_under5/n_items*100:.1f}%) have <5 interactions")

    # Sequence length analysis (for sequential recommendation fitness)
    print(f"\n--- Sequence Length Analysis (users with >=2 interactions) ---")
    users_ge2 = (user_counts >= 2).sum()
    users_ge3 = (user_counts >= 3).sum()
    users_ge5 = (user_counts >= 5).sum()
    print(f"  Users with >=2 (can form 1 train sample): {users_ge2:,} ({users_ge2/n_users*100:.1f}%)")
    print(f"  Users with >=3 (train+valid+test):        {users_ge3:,} ({users_ge3/n_users*100:.1f}%)")
    print(f"  Users with >=5 (meaningful sequence):     {users_ge5:,} ({users_ge5/n_users*100:.1f}%)")

    # Timestamp analysis
    if time_col in df.columns:
        timestamps = df[time_col].values
        ts_min, ts_max = timestamps.min(), timestamps.max()
        print(f"\n--- Timestamp Analysis ---")
        try:
            dt_min = datetime.fromtimestamp(ts_min)
            dt_max = datetime.fromtimestamp(ts_max)
            print(f"  Earliest: {ts_min:.0f} ({dt_min.strftime('%Y-%m-%d %H:%M:%S')})")
            print(f"  Latest:   {ts_max:.0f} ({dt_max.strftime('%Y-%m-%d %H:%M:%S')})")
            print(f"  Span:     {(ts_max - ts_min) / 86400:.0f} days")
        except (OSError, ValueError):
            print(f"  Earliest: {ts_min}")
            print(f"  Latest:   {ts_max}")

        # GTS cutoff proposals (80/10/10)
        p80 = np.percentile(timestamps, 80)
        p90 = np.percentile(timestamps, 90)
        print(f"\n--- Proposed GTS Cutoffs (80/10/10) ---")
        try:
            dt_p80 = datetime.fromtimestamp(p80)
            dt_p90 = datetime.fromtimestamp(p90)
            print(f"  Train cutoff  (80th pct): {p80:.0f} ({dt_p80.strftime('%Y-%m-%d')})")
            print(f"  Valid cutoff  (90th pct): {p90:.0f} ({dt_p90.strftime('%Y-%m-%d')})")
        except (OSError, ValueError):
            print(f"  Train cutoff  (80th pct): {p80}")
            print(f"  Valid cutoff  (90th pct): {p90}")

        train_mask = timestamps <= p80
        valid_mask = (timestamps > p80) & (timestamps <= p90)
        test_mask = timestamps > p90
        n_train = train_mask.sum()
        n_valid = valid_mask.sum()
        n_test = test_mask.sum()
        print(f"  Train inters: {n_train:>10,} ({n_train/n_inters*100:.1f}%)")
        print(f"  Valid inters: {n_valid:>10,} ({n_valid/n_inters*100:.1f}%)")
        print(f"  Test  inters: {n_test:>10,} ({n_test/n_inters*100:.1f}%)")

        # Users with history before cutoff AND test interactions
        train_users = set(df[uid_col][train_mask].unique())
        test_users = set(df[uid_col][test_mask].unique())
        test_users_with_history = train_users & test_users
        print(f"\n  Test users (total):              {len(test_users):>10,}")
        print(f"  Test users with train history:   {len(test_users_with_history):>10,}")

        # Users with >=5 train interactions AND test interactions
        train_user_counts = df[train_mask].groupby(uid_col).size()
        users_5plus_train = set(train_user_counts[train_user_counts >= 5].index)
        test_users_5plus = users_5plus_train & test_users
        print(f"  Test users with >=5 train inters:{len(test_users_5plus):>10,}")
    else:
        print(f"\n  No timestamp column found, skipping temporal analysis")

    # Popularity strata (as used in paper)
    print(f"\n--- Item Popularity Strata ---")
    new_items = (item_counts < 3).sum()
    few_items = ((item_counts >= 3) & (item_counts < 10)).sum()
    freq_items = (item_counts >= 10).sum()
    print(f"  New  [1,3):    {new_items:>10,} ({new_items/n_items*100:.1f}%)")
    print(f"  Few  [3,10):   {few_items:>10,} ({few_items/n_items*100:.1f}%)")
    print(f"  Freq [10,inf): {freq_items:>10,} ({freq_items/n_items*100:.1f}%)")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Audit RecBole datasets")
    parser.add_argument("--data_dir", default="dataset/", help="Root dataset directory")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Dataset names to audit (default: auto-detect)")
    args = parser.parse_args()

    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = []
        for name in sorted(os.listdir(args.data_dir)):
            d = os.path.join(args.data_dir, name)
            if os.path.isdir(d) and any(f.endswith(".inter") for f in os.listdir(d)):
                dataset_names.append(name)

    if not dataset_names:
        print(f"No datasets found in {args.data_dir}")
        sys.exit(1)

    print(f"Auditing {len(dataset_names)} datasets in {args.data_dir}")
    for name in dataset_names:
        dataset_dir = os.path.join(args.data_dir, name)
        if os.path.isdir(dataset_dir):
            audit_dataset(dataset_dir, name)
        else:
            print(f"WARNING: {dataset_dir} not found, skipping")


if __name__ == "__main__":
    main()
