#!/usr/bin/env python3
"""
Warm-target / unseen-catalog audit (0723 G1).

Reports GTS cutoff stats, unseen target share, catalog turnover, and
recommends warm-target-only main-table protocol.

Usage:
  python scripts/audit_warm_targets.py --dataset Amazon_Beauty
  python scripts/audit_warm_targets.py --dataset Amazon_Toys_and_Games --out paper_recsys/warm_target_audit_toys.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_inter(dataset_path: Path):
    inter_file = next(dataset_path.glob("*.inter"), None)
    if inter_file is None:
        raise FileNotFoundError(f"No .inter in {dataset_path}")
    with inter_file.open() as fh:
        header = fh.readline().strip().split("\t")
        rows = [line.strip().split("\t") for line in fh if line.strip()]
    uid_col = next(i for i, h in enumerate(header) if "user" in h.lower() and "id" in h.lower())
    iid_col = next(i for i, h in enumerate(header) if "item" in h.lower() and "id" in h.lower())
    ts_col = next(i for i, h in enumerate(header) if "timestamp" in h.lower())
    return [(r[uid_col], r[iid_col], float(r[ts_col])) for r in rows]


def analyze(data, train_pct=0.8, valid_pct=0.9, min_history=5):
    data = sorted(data, key=lambda x: x[2])
    n = len(data)
    t_cut = data[int(n * train_pct) - 1][2]
    v_cut = data[int(n * valid_pct) - 1][2]
    train = [x for x in data if x[2] <= t_cut]
    valid = [x for x in data if t_cut < x[2] <= v_cut]
    test = [x for x in data if x[2] > v_cut]

    train_hist = Counter(u for u, _, _ in train)
    eligible = {u for u, c in train_hist.items() if c >= min_history}

    def last_targets(split):
        by_user = {}
        for u, i, ts in split:
            if u in eligible:
                by_user[u] = (i, ts)
        return list(by_user.values())

    train_counts = Counter(i for _, i, _ in train)
    train_items = set(train_counts)
    valid_tg = last_targets(valid)
    test_tg = last_targets(test)

    def unseen_share(targets):
        if not targets:
            return 0.0, 0, 0
        unseen = sum(1 for i, _ in targets if i not in train_items)
        return unseen / len(targets), unseen, len(targets)

    v_share, v_unseen, v_n = unseen_share(valid_tg)
    t_share, t_unseen, t_n = unseen_share(test_tg)

    test_items = {i for i, _ in test_tg}
    turnover = len(test_items - train_items) / max(1, len(test_items))

    pops_v = [train_counts.get(i, 0) for i, _ in valid_tg]
    pops_t = [train_counts.get(i, 0) for i, _ in test_tg]

    def pct(arr, p):
        return float(np.percentile(arr, p)) if arr else 0.0

    return {
        "n_inter": n,
        "train_cutoff_ts": t_cut,
        "valid_cutoff_ts": v_cut,
        "train_cutoff_date": datetime.fromtimestamp(t_cut, tz=timezone.utc).strftime("%Y-%m-%d"),
        "valid_cutoff_date": datetime.fromtimestamp(v_cut, tz=timezone.utc).strftime("%Y-%m-%d"),
        "n_train_items": len(train_items),
        "n_eligible_users": len(eligible),
        "valid_unseen_share": v_share,
        "valid_unseen_n": v_unseen,
        "valid_targets_n": v_n,
        "test_unseen_share": t_share,
        "test_unseen_n": t_unseen,
        "test_targets_n": t_n,
        "test_catalog_turnover": turnover,
        "valid_pop_mean": float(np.mean(pops_v)) if pops_v else 0.0,
        "test_pop_mean": float(np.mean(pops_t)) if pops_t else 0.0,
        "valid_pop_p50": pct(pops_v, 50),
        "test_pop_p50": pct(pops_t, 50),
        "protocol": {
            "split": f"TS [{train_pct}, {valid_pct-train_pct:.1f}, {1-valid_pct:.1f}]",
            "main_table": "warm targets only (item in train catalog)",
            "unseen": "report separately; compare only if content-only scoring supported",
            "candidates": "full ranking over train warm catalog (recommended)",
            "min_train_interactions": min_history,
        },
    }


def to_md(dataset: str, r: dict) -> str:
    return "\n".join([
        f"# Warm-target Audit — {dataset}",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> Script: `scripts/audit_warm_targets.py`",
        "",
        "## Cutoffs",
        "",
        "| | |",
        "|---|---|",
        f"| Train cutoff | {r['train_cutoff_date']} (ts={r['train_cutoff_ts']}) |",
        f"| Valid cutoff | {r['valid_cutoff_date']} (ts={r['valid_cutoff_ts']}) |",
        f"| Train catalog size | {r['n_train_items']} |",
        f"| Eligible users (hist≥{r['protocol']['min_train_interactions']}) | {r['n_eligible_users']} |",
        "",
        "## Unseen target share",
        "",
        "| Split | Unseen share | Unseen / Total |",
        "|-------|:------------:|:--------------:|",
        f"| Valid | {r['valid_unseen_share']:.1%} | {r['valid_unseen_n']} / {r['valid_targets_n']} |",
        f"| Test | **{r['test_unseen_share']:.1%}** | {r['test_unseen_n']} / {r['test_targets_n']} |",
        "",
        f"Test catalog turnover (unseen items among test targets): **{r['test_catalog_turnover']:.1%}**",
        "",
        "## Target popularity (train count)",
        "",
        "| Split | mean | p50 |",
        "|-------|:----:|:---:|",
        f"| Valid | {r['valid_pop_mean']:.1f} | {r['valid_pop_p50']:.1f} |",
        f"| Test | {r['test_pop_mean']:.1f} | {r['test_pop_p50']:.1f} |",
        "",
        "## Protocol recommendation (0723)",
        "",
        "- Main table: **warm targets only** (item appeared in train before cutoff).",
        "- Unseen targets: separate table; do **not** compare ID-only vs text models unless content-only scoring is enabled.",
        "- Candidate catalog: train warm items (full ranking).",
        "- Paper Methods must state: GTS cutoff, target selection, candidate catalog, cold filtering, valid/test construction.",
        "",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Amazon_Beauty")
    ap.add_argument("--base_dir", default="dataset")
    ap.add_argument("--train_pct", type=float, default=0.8)
    ap.add_argument("--valid_pct", type=float, default=0.9)
    ap.add_argument("--min_history", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ds_path = ROOT / args.base_dir / args.dataset
    data = load_inter(ds_path)
    result = analyze(data, args.train_pct, args.valid_pct, args.min_history)

    out = Path(args.out) if args.out else ROOT / "paper_recsys" / f"warm_target_audit_{args.dataset.lower()}.md"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_md(args.dataset, result))
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {json_path}")
    print(f"test_unseen_share={result['test_unseen_share']:.4f}")


if __name__ == "__main__":
    main()
