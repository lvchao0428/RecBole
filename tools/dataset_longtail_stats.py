#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long-tail / popularity concentration stats for RecBole atomic .inter datasets.

Reads one or more dataset directories (e.g. dataset/Amazon_Beauty-example or
dataset/Amazon_Beauty on server), finds a single *.inter file, aggregates
user/item interaction counts, and prints comparable metrics across datasets.

Usage:
  python tools/dataset_longtail_stats.py \\
    dataset/Amazon_Beauty-example \\
    dataset/Amazon_Toys_and_Games-example \\
    dataset/book-crossing-example

  python tools/dataset_longtail_stats.py --json out.json dataset/Amazon_Beauty

  # On server (dirs without -example), same order as --name:
  python tools/dataset_longtail_stats.py \\
    dataset/Amazon_Beauty dataset/Amazon_Toys_and_Games dataset/book-crossing \\
    --name Beauty --name Toys --name BookCrossing --json longtail.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _find_inter_file(dataset_dir: str) -> str:
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Not a directory: {dataset_dir}")
    paths = sorted(glob(os.path.join(dataset_dir, "*.inter")))
    if not paths:
        raise FileNotFoundError(f"No .inter file under {dataset_dir}")
    if len(paths) > 1:
        names = [os.path.basename(p) for p in paths]
        raise ValueError(
            f"Multiple .inter files in {dataset_dir}: {names}; "
            "use a directory with exactly one .inter file."
        )
    return paths[0]


def _field_col(cols: List[str], base: str) -> Optional[str]:
    for c in cols:
        if c.split(":")[0].strip() == base:
            return c
    return None


def _gini(counts: np.ndarray) -> float:
    """Gini of nonnegative counts; 0 = equal, ->1 = maximally concentrated."""
    x = np.asarray(counts, dtype=np.float64)
    x = x[x > 0]
    n = x.size
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    x = np.sort(x)
    s = x.sum()
    if s <= 0:
        return float("nan")
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((np.sum((2 * idx - n - 1) * x)) / (n * s))


def _normalized_entropy(counts: np.ndarray) -> float:
    """H(p) / log(n) with n = #items with positive support; in [0,1] if full support."""
    x = np.asarray(counts, dtype=np.float64)
    x = x[x > 0]
    n = x.size
    if n <= 1:
        return float("nan")
    p = x / x.sum()
    h = -np.sum(p * np.log(p + 1e-30))
    return float(h / math.log(n))


def _hhi(counts: np.ndarray) -> float:
    """Herfindahl on item share of interactions; in (1/n_items, 1]."""
    x = np.asarray(counts, dtype=np.float64)
    s = x.sum()
    if s <= 0:
        return float("nan")
    p = x / s
    return float(np.sum(p * p))


def _top_r_share(counts: np.ndarray, frac_items: float) -> float:
    """Fraction of all interactions covered by the top `frac_items` fraction of items (by degree)."""
    x = np.asarray(counts, dtype=np.float64)
    n = x.size
    if n == 0 or x.sum() <= 0:
        return float("nan")
    k = max(1, int(math.floor(frac_items * n)))
    x_sorted = np.sort(x)[::-1]
    return float(x_sorted[:k].sum() / x.sum())


def _percentile_stats(counts: np.ndarray) -> Dict[str, float]:
    x = np.asarray(counts, dtype=np.float64)
    if x.size == 0:
        return {k: float("nan") for k in ("p50", "p90", "p95", "p99")}
    return {
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
    }


def collect_counts(
    inter_path: str, chunksize: int = 1_000_000
) -> Tuple[Counter, Counter, int]:
    header = pd.read_csv(inter_path, sep="\t", nrows=0)
    cols = list(header.columns)
    ucol = _field_col(cols, "user_id")
    icol = _field_col(cols, "item_id")
    if ucol is None or icol is None:
        raise ValueError(
            f"{inter_path}: need user_id and item_id fields in header, got {cols}"
        )
    usecols = [ucol, icol]
    user_c: Counter = Counter()
    item_c: Counter = Counter()
    n_rows = 0
    for chunk in pd.read_csv(
        inter_path,
        sep="\t",
        usecols=usecols,
        chunksize=chunksize,
        dtype=str,
        low_memory=False,
    ):
        n_rows += len(chunk)
        user_c.update(chunk[ucol].astype(str))
        item_c.update(chunk[icol].astype(str))
    return user_c, item_c, n_rows


def summarize(name: str, inter_path: str, chunksize: int) -> Dict:
    user_c, item_c, n_rows = collect_counts(inter_path, chunksize=chunksize)
    uc = np.array(list(user_c.values()), dtype=np.int64)
    ic = np.array(list(item_c.values()), dtype=np.int64)
    n_users, n_items = len(user_c), len(item_c)
    tot_u, tot_i = int(uc.sum()), int(ic.sum())
    assert tot_u == tot_i == n_rows

    density = n_rows / (n_users * n_items) if n_users and n_items else 0.0

    item_ge1 = int(np.sum(ic == 1))
    item_le2 = int(np.sum(ic <= 2))
    item_lt3 = int(np.sum(ic < 3))
    idp = _percentile_stats(ic)

    return {
        "dataset_label": name,
        "inter_file": os.path.abspath(inter_path),
        "n_interactions": n_rows,
        "n_users": n_users,
        "n_items": n_items,
        "density": density,
        "item_degree_mean": float(ic.mean()) if n_items else float("nan"),
        "item_degree_median": float(np.median(ic)) if n_items else float("nan"),
        "item_degree_max": int(ic.max()) if n_items else 0,
        "item_degree_std": float(ic.std()) if n_items else float("nan"),
        "item_degree_p50": idp["p50"],
        "item_degree_p90": idp["p90"],
        "item_degree_p95": idp["p95"],
        "item_degree_p99": idp["p99"],
        "user_degree_mean": float(uc.mean()) if n_users else float("nan"),
        "user_degree_median": float(np.median(uc)) if n_users else float("nan"),
        "user_degree_max": int(uc.max()) if n_users else 0,
        "item_gini": _gini(ic),
        "user_gini": _gini(uc),
        "item_entropy_norm": _normalized_entropy(ic),
        "user_entropy_norm": _normalized_entropy(uc),
        "item_hhi": _hhi(ic),
        "user_hhi": _hhi(uc),
        "item_share_interactions_top1pct_items": _top_r_share(ic, 0.01),
        "item_share_interactions_top5pct_items": _top_r_share(ic, 0.05),
        "item_share_interactions_top10pct_items": _top_r_share(ic, 0.10),
        "item_share_interactions_top20pct_items": _top_r_share(ic, 0.20),
        "pct_items_deg_eq_1": float(item_ge1 / n_items * 100) if n_items else float("nan"),
        "pct_items_deg_le_2": float(item_le2 / n_items * 100) if n_items else float("nan"),
        "pct_items_deg_lt_3": float(item_lt3 / n_items * 100) if n_items else float("nan"),
    }


def _fmt_pct(x: float) -> str:
    if x != x:
        return "nan"
    return f"{x * 100:.2f}%"


def _fmt_float(x: float, nd: int = 4) -> str:
    if x != x:
        return "nan"
    return f"{x:.{nd}f}"


def print_table(rows: List[Dict]) -> None:
    columns = [
        ("label", "dataset", str),
        ("n_interactions", "|inter|", lambda x: str(x)),
        ("n_users", "|users|", lambda x: str(x)),
        ("n_items", "|items|", lambda x: str(x)),
        ("density", "density", lambda x: _fmt_float(x, 8)),
        ("item_gini", "item Gini", lambda x: _fmt_float(x, 4)),
        ("user_gini", "user Gini", lambda x: _fmt_float(x, 4)),
        ("item_hhi", "item HHI", lambda x: _fmt_float(x, 6)),
        ("item_entropy_norm", "item H/log n", lambda x: _fmt_float(x, 4)),
        ("item_share_interactions_top1pct_items", "top 1% items share", _fmt_pct),
        ("item_share_interactions_top5pct_items", "top 5% items share", _fmt_pct),
        ("item_share_interactions_top10pct_items", "top 10% items share", _fmt_pct),
        ("pct_items_deg_lt_3", "% items deg<3", lambda x: _fmt_float(x, 2) + "%"),
        ("item_degree_median", "median item deg", lambda x: _fmt_float(x, 2)),
        ("item_degree_p99", "p99 item deg", lambda x: _fmt_float(x, 2)),
    ]

    headers = [c[1] for c in columns]
    table = []
    for r in rows:
        out = []
        for key, _, conv in columns:
            if key == "label":
                out.append(conv(r.get("dataset_label", "")))
            else:
                out.append(conv(r[key]))
        table.append(out)

    widths = [max(len(h), max(len(row[i]) for row in table)) for i, h in enumerate(headers)]
    sep = "  "
    head = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(head)
    print(sep.join("-" * w for w in widths))
    for row in table:
        print(sep.join(row[i].ljust(widths[i]) for i in range(len(row))))


def main() -> None:
    p = argparse.ArgumentParser(description="Long-tail stats for RecBole .inter datasets.")
    p.add_argument(
        "dataset_dirs",
        nargs="*",
        help="Paths to dataset folders each containing exactly one .inter file.",
    )
    p.add_argument(
        "--name",
        action="append",
        default=None,
        metavar="LABEL",
        help="Row label for the corresponding dataset dir (use once per dir, same order). "
        "Example: dir1 dir2 --name Beauty --name Toys",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Rows per chunk when reading large .inter files.",
    )
    p.add_argument("--json", type=str, default=None, help="Write full metrics to this JSON path.")
    args = p.parse_args()

    dirs = args.dataset_dirs
    if not dirs:
        p.print_help()
        sys.exit(1)

    names = args.name
    if names is not None and len(names) != len(dirs):
        print(
            "error: number of --name must match number of dataset dirs",
            file=sys.stderr,
        )
        sys.exit(2)

    results: List[Dict] = []
    for i, d in enumerate(dirs):
        d = os.path.abspath(d)
        label = names[i] if names else os.path.basename(d.rstrip(os.sep))
        inter = _find_inter_file(d)
        results.append(summarize(label, inter, chunksize=args.chunksize))

    print_table(results)
    print(
        "\nNotes: top k%% items share = fraction of all interactions on items in the "
        "top k%% of items when ranked by interaction count (head of the tail). "
        "item Gini / HHI / H/log n measure item popularity skew (higher Gini & HHI "
        "=> more concentrated). % items deg<3 matches popularity stratum 'new' "
        "if you use [1,3) on training counts.",
        file=sys.stderr,
    )

    if args.json:
        out_path = os.path.abspath(args.json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
