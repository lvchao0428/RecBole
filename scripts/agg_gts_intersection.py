#!/usr/bin/env python3
"""Aggregate gts_intersection_eval.csv into paper-style numbers.

For each dataset x family: average over sampling seeds within each training
seed, then report mean +- std across the 3 training seeds, for warm MRR@10
under Last and Random restricted to the intersection cohort (and the full
warm cohorts for reference).
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FAM_ORDER = ["id", "tf", "llm", "mv"]
FAM_LABEL = {"id": "ID", "tf": "TF", "llm": "TF+LLM", "mv": "TF+MV"}


def parse_tag(tag):
    m = re.match(r"wrap_(beauty|toys|grocery)_(id|tf|llm|mv)_.*_seed(\d+)", tag)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/gts_intersection_eval.csv")
    # (ds, fam, train_seed) -> list of row dicts (one per sampling seed)
    groups = defaultdict(list)
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("error") or not r.get("tag"):
                continue
            parsed = parse_tag(r["tag"])
            if not parsed:
                continue
            groups[parsed].append(r)

    cols = [
        ("last_mrr@10_intersection", "Last∩"),
        ("random_mrr@10_intersection", "Rand∩"),
        ("last_mrr@10_warm_cohort", "LastFull"),
        ("random_mrr@10_warm_cohort", "RandFull"),
    ]

    # (ds, fam) -> {col: [per-training-seed mean over sampling seeds]}
    agg = defaultdict(lambda: defaultdict(list))
    nint = defaultdict(list)
    for (ds, fam, tseed), rows in sorted(groups.items()):
        for col, _ in cols:
            agg[(ds, fam)][col].append(np.mean([float(r[col]) for r in rows]))
        nint[(ds, fam)].extend(int(r["n_intersection"]) for r in rows)

    print(f"{'dataset':<10} {'family':<8} " +
          " ".join(f"{lab:>16}" for _, lab in cols) + f" {'|inter|':>8} {'n_tseed':>7}")
    for ds in ["beauty", "toys", "grocery"]:
        for fam in FAM_ORDER:
            key = (ds, fam)
            if key not in agg:
                continue
            cells = []
            for col, _ in cols:
                v = agg[key][col]
                cells.append(f"{np.mean(v):.4f}±{np.std(v):.4f}")
            print(f"{ds:<10} {FAM_LABEL[fam]:<8} " +
                  " ".join(f"{c:>16}" for c in cells) +
                  f" {int(np.mean(nint[key])):>8} {len(agg[key][cols[0][0]]):>7}")
        print()


if __name__ == "__main__":
    main()
