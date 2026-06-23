#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified item/user strata + long-tail comparison for RecBole .inter datasets.

Supports reading from a dataset directory or directly from a ProcessedDatasets .zip
(Amazon_ratings layout: Amazon_*.inter inside zip).

Usage:
  python tools/dataset_strata_compare.py \\
    --zip /path/Amazon_Beauty.zip --name Beauty \\
    --zip /path/Amazon_Toys_and_Games.zip --name Toys \\
    --inter /path/Food/Food.inter --name Food \\
    --json out.json
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _field_col(cols: List[str], base: str) -> Optional[str]:
    for c in cols:
        if str(c).split(":")[0].strip() == base:
            return c
    return None


def _gini(counts: np.ndarray) -> float:
    x = np.asarray(counts, dtype=np.float64)
    x = x[x > 0]
    n = x.size
    if n <= 1:
        return 0.0
    x = np.sort(x)
    s = x.sum()
    if s <= 0:
        return float("nan")
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((np.sum((2 * idx - n - 1) * x)) / (n * s))


def _top_r_share(counts: np.ndarray, frac: float) -> float:
    x = np.asarray(counts, dtype=np.float64)
    n = x.size
    if n == 0 or x.sum() <= 0:
        return float("nan")
    k = max(1, int(math.floor(frac * n)))
    return float(np.sort(x)[::-1][:k].sum() / x.sum())


def _read_inter_from_zip(zip_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        inter_names = [n for n in zf.namelist() if n.endswith(".inter")]
        if not inter_names:
            raise FileNotFoundError(f"No .inter in zip: {zip_path}")
        if len(inter_names) > 1:
            inter_names.sort(key=len)
        name = inter_names[0]
        with zf.open(name) as f:
            return pd.read_csv(f, sep="\t")


def _read_inter(path: str) -> pd.DataFrame:
    if path.lower().endswith(".zip"):
        return _read_inter_from_zip(path)
    return pd.read_csv(path, sep="\t")


def analyze_inter(df: pd.DataFrame, label: str, source: str) -> Dict:
    cols = list(df.columns)
    ucol = _field_col(cols, "user_id")
    icol = _field_col(cols, "item_id")
    if ucol is None or icol is None:
        raise ValueError(f"{label}: missing user_id/item_id in {cols}")

    n_inter = len(df)
    item_deg = df.groupby(icol).size()
    user_deg = df.groupby(ucol).size()
    n_items = len(item_deg)
    n_users = len(user_deg)

    new_m = (item_deg >= 1) & (item_deg < 3)
    few_m = (item_deg >= 3) & (item_deg < 10)
    freq_m = item_deg >= 10

    new_ids = set(item_deg[new_m].index)
    few_ids = set(item_deg[few_m].index)
    freq_ids = set(item_deg[freq_m].index)

    inter_new = df[df[icol].isin(new_ids)].shape[0]
    inter_few = df[df[icol].isin(few_ids)].shape[0]
    inter_freq = df[df[icol].isin(freq_ids)].shape[0]

    ud = user_deg.values.astype(np.float64)
    idg = item_deg.values.astype(np.float64)

    # Users whose test interactions would mostly hit cold items: users with low personal seq
    pct_users_le3 = float((ud <= 3).sum() / n_users * 100)
    pct_users_le5 = float((ud <= 5).sum() / n_users * 100)

    # Per-stratum avg item degree (how "supported" items are within stratum)
    strata_item_deg_mean = {
        "new": float(item_deg[new_m].mean()) if new_m.any() else 0.0,
        "few": float(item_deg[few_m].mean()) if few_m.any() else 0.0,
        "freq": float(item_deg[freq_m].mean()) if freq_m.any() else 0.0,
    }

    # Interactions per user on items from each stratum (user exposure to cold catalog)
    user_new_touch = df[df[icol].isin(new_ids)].groupby(ucol).size()
    user_few_touch = df[df[icol].isin(few_ids)].groupby(ucol).size()
    user_freq_touch = df[df[icol].isin(freq_ids)].groupby(ucol).size()
    pct_users_touch_new = float(user_new_touch.index.nunique() / n_users * 100)
    pct_users_touch_few = float(user_few_touch.index.nunique() / n_users * 100)
    pct_users_touch_freq = float(user_freq_touch.index.nunique() / n_users * 100)

    return {
        "label": label,
        "source": source,
        "n_inter": int(n_inter),
        "n_users": int(n_users),
        "n_items": int(n_items),
        "density_pct": float(n_inter / (n_users * n_items) * 100),
        "avg_seq_len": float(n_inter / n_users),
        "avg_item_deg": float(n_inter / n_items),
        "item_gini": _gini(idg),
        "user_gini": _gini(ud),
        "top1pct_item_inter_share_pct": _top_r_share(idg, 0.01) * 100,
        "top5pct_item_inter_share_pct": _top_r_share(idg, 0.05) * 100,
        "top10pct_item_inter_share_pct": _top_r_share(idg, 0.10) * 100,
        "item_strata_pct": {
            "new": float(new_m.sum() / n_items * 100),
            "few": float(few_m.sum() / n_items * 100),
            "freq": float(freq_m.sum() / n_items * 100),
        },
        "inter_strata_pct": {
            "new": float(inter_new / n_inter * 100),
            "few": float(inter_few / n_inter * 100),
            "freq": float(inter_freq / n_inter * 100),
        },
        "strata_item_deg_mean": strata_item_deg_mean,
        "user_degree": {
            "mean": float(ud.mean()),
            "median": float(np.median(ud)),
            "p90": float(np.percentile(ud, 90)),
            "p99": float(np.percentile(ud, 99)),
            "max": int(ud.max()),
            "pct_le3": pct_users_le3,
            "pct_le5": pct_users_le5,
        },
        "item_degree": {
            "median": float(np.median(idg)),
            "p90": float(np.percentile(idg, 90)),
            "p99": float(np.percentile(idg, 99)),
            "max": int(idg.max()),
        },
        "pct_users_touch_strata_item": {
            "new": pct_users_touch_new,
            "few": pct_users_touch_few,
            "freq": pct_users_touch_freq,
        },
        "cold_inter_share_pct": float((inter_new + inter_few) / n_inter * 100),
        "freq_inter_share_pct": float(inter_freq / n_inter * 100),
    }


def _l1_dist(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


def rank_vs_beauty(rows: List[Dict], ref_label: str = "Beauty") -> List[Dict]:
    ref = next((r for r in rows if r["label"] == ref_label), None)
    if ref is None:
        return rows
    for r in rows:
        r["dist_item_strata_l1_vs_beauty"] = _l1_dist(
            r["item_strata_pct"], ref["item_strata_pct"]
        )
        r["dist_inter_strata_l1_vs_beauty"] = _l1_dist(
            r["inter_strata_pct"], ref["inter_strata_pct"]
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", action="append", default=[], help="Path to dataset .zip")
    p.add_argument("--name", action="append", default=[], help="Label for preceding --zip")
    p.add_argument("--inter", action="append", default=[], help="Path to .inter file")
    p.add_argument("--inter-name", action="append", default=[], help="Label for --inter")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    sources: List[Tuple[str, str]] = []
    if len(args.name) != len(args.zip):
        sys.exit("Each --zip needs one --name")
    if len(args.inter_name) != len(args.inter):
        sys.exit("Each --inter needs one --inter-name")
    for z, n in zip(args.zip, args.name):
        sources.append((n, z))
    for path, n in zip(args.inter, args.inter_name):
        sources.append((n, path))

    rows = []
    for label, src in sources:
        print(f"Analyzing {label} ...", file=sys.stderr)
        df = _read_inter(src)
        rows.append(analyze_inter(df, label, src))

    rows = rank_vs_beauty(rows)
    rows.sort(key=lambda r: r.get("dist_item_strata_l1_vs_beauty", 999))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.json}", file=sys.stderr)

    # compact table
    hdr = (
        f"{'Dataset':<18} {'|U|':>8} {'|I|':>8} {'|R|':>10} "
        f"{'AvgSeq':>6} {'I-new%':>6} {'I-few%':>6} {'I-fr%':>6} "
        f"{'R-new%':>6} {'R-few%':>6} {'R-fr%':>6} "
        f"{'coldR%':>7} {'top5%I':>7} {'Δitem':>6} {'Δinter':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        isp = r["item_strata_pct"]
        rsp = r["inter_strata_pct"]
        print(
            f"{r['label']:<18} {r['n_users']:>8,} {r['n_items']:>8,} {r['n_inter']:>10,} "
            f"{r['avg_seq_len']:>6.2f} "
            f"{isp['new']:>6.1f} {isp['few']:>6.1f} {isp['freq']:>6.1f} "
            f"{rsp['new']:>6.1f} {rsp['few']:>6.1f} {rsp['freq']:>6.1f} "
            f"{r['cold_inter_share_pct']:>7.1f} "
            f"{r['top5pct_item_inter_share_pct']:>7.1f} "
            f"{r.get('dist_item_strata_l1_vs_beauty', float('nan')):>6.1f} "
            f"{r.get('dist_inter_strata_l1_vs_beauty', float('nan')):>6.1f}"
        )


if __name__ == "__main__":
    main()
