#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan all RecBole datasets under ProcessedDatasets + project dataset/, compute strata stats.

Usage (5090):
  python tools/dataset_strata_scan_all.py \\
    --processed-root /home/charlie/project/RecSysDatasets/RecBole/ProcessedDatasets \\
    --project-dataset /home/charlie/project/RecBole/dataset \\
    --json paper_recsys/dataset_strata_all_20260623.json \\
    --md paper_recsys/dataset_strata_all_20260623.md \\
    --min-inter 5000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# reuse core logic
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.dataset_strata_compare import analyze_inter, _read_inter  # noqa: E402


def _is_example_path(path: str) -> bool:
    p = path.lower()
    return "example" in p or "/__macosx/" in p


def _normalize_key(name: str) -> str:
    n = name.replace(".inter", "").replace(".zip", "")
    n = re.sub(r"[-_]?example$", "", n, flags=re.I)
    n = re.sub(r"^amazon_", "Amazon_", n)
    return n.lower()


def _is_amazon(label: str, source: str) -> bool:
    s = (label + " " + source).lower()
    return "amazon" in s or "/amazon_ratings/" in s.lower() or "/amazon2018/" in s.lower()


def _domain_family(label: str, source: str) -> str:
    if _is_amazon(label, source):
        return "Amazon"
    l = label.lower()
    if "yelp" in l:
        return "Yelp"
    if "book" in l and "cross" in l:
        return "Book-Crossing"
    if "food" in l:
        return "Food"
    if "mind" in l:
        return "MIND"
    if "ml-1m" in l or "ml_1m" in l:
        return "MovieLens"
    if "ml-100k" in l or "ml_100k" in l:
        return "MovieLens-100k"
    if "douban" in l:
        return "Douban"
    if "goodreads" in l:
        return "GoodReads"
    if "beer" in l:
        return "BeerAdvocate"
    if "lastfm" in l or "lfm" in l:
        return "LastFM"
    if "gowalla" in l:
        return "Gowalla"
    if "alibaba" in l or "ifashion" in l:
        return "Alibaba"
    return "Other"


def _inter_uncompressed_bytes(path: str) -> int:
    """Return uncompressed .inter size in bytes (0 if unknown)."""
    if path.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(path, "r") as zf:
                inter = [n for n in zf.namelist() if n.endswith(".inter") and not n.startswith("__")]
                if not inter:
                    return 0
                return zf.getinfo(inter[0]).file_size
        except Exception:
            return 0
    try:
        return os.path.getsize(path)
    except Exception:
        return 0


def _quick_inter_rows(path: str) -> int:
    """Fast row count for .inter; for zip read member size heuristic."""
    nbytes = _inter_uncompressed_bytes(path)
    if nbytes <= 0:
        return 0
    return max(0, nbytes // 40)


def discover_sources(processed_root: str, project_dataset: str) -> List[Tuple[str, str, int]]:
    """Return list of (label, path, priority). Higher priority wins dedup."""
    found: Dict[str, Tuple[str, str, int]] = {}

    def add(label: str, path: str, priority: int) -> None:
        if _is_example_path(path):
            return
        key = _normalize_key(label)
        prev = found.get(key)
        if prev is None or priority > prev[2]:
            found[key] = (label, path, priority)

    pr = Path(processed_root)
    if pr.is_dir():
        for zp in sorted(pr.rglob("*.zip")):
            if _is_example_path(str(zp)):
                continue
            name = zp.stem
            if name.lower().endswith("-example") or name.lower().endswith("_example"):
                continue
            add(name, str(zp), 10)
        for ip in sorted(pr.rglob("*.inter")):
            if _is_example_path(str(ip)):
                continue
            add(ip.stem, str(ip), 20)

    pd = Path(project_dataset)
    if pd.is_dir():
        for ip in sorted(pd.rglob("*.inter")):
            if _is_example_path(str(ip)):
                continue
            add(ip.stem, str(ip), 30)

    out = [(v[0], v[1], v[2]) for v in found.values()]
    out.sort(key=lambda x: x[0].lower())
    return out


def rank_vs_beauty(rows: List[Dict], ref_label: str = "Amazon_Beauty") -> None:
    ref = None
    for r in rows:
        if r["label"] == ref_label or _normalize_key(r["label"]) == _normalize_key(ref_label):
            ref = r
            break
    if ref is None:
        for r in rows:
            if "beauty" in r["label"].lower() and "luxury" not in r["label"].lower():
                ref = r
                break
    if ref is None:
        return

    def l1(a: Dict, b: Dict) -> float:
        keys = sorted(set(a) | set(b))
        return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)

    for r in rows:
        r["domain"] = _domain_family(r["label"], r["source"])
        r["is_amazon"] = _is_amazon(r["label"], r["source"])
        r["dist_item_strata_l1_vs_beauty"] = l1(r["item_strata_pct"], ref["item_strata_pct"])
        r["dist_inter_strata_l1_vs_beauty"] = l1(r["inter_strata_pct"], ref["inter_strata_pct"])
        r["dist_total_l1_vs_beauty"] = (
            r["dist_item_strata_l1_vs_beauty"] + r["dist_inter_strata_l1_vs_beauty"]
        )


def write_markdown(rows: List[Dict], path: str, ref: Dict) -> None:
    lines = [
        "# 全数据集分层分布扫描（vs Beauty 参照）",
        "",
        f"> 生成: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 数据集数量: **{len(rows)}**（去重后，排除 example 子集）",
        f"> 参照: **{ref['label']}** — R-freq={ref['inter_strata_pct']['freq']:.1f}%, "
        f"AvgSeq={ref['avg_seq_len']:.2f}, coldR={ref['cold_inter_share_pct']:.1f}%",
        "",
        "## 非 Amazon 域 — 按与 Beauty 总 L1 距离排序",
        "",
        "| Rank | Dataset | Domain | #U | #I | #R | AvgSeq | I-fr% | R-fr% | coldR% | Δtotal |",
        "|------|---------|--------|-----|-----|------|--------|-------|-------|--------|--------|",
    ]
    non_amz = sorted([r for r in rows if not r["is_amazon"]], key=lambda x: x["dist_total_l1_vs_beauty"])
    for i, r in enumerate(non_amz, 1):
        isp = r["item_strata_pct"]
        rsp = r["inter_strata_pct"]
        lines.append(
            f"| {i} | {r['label']} | {r['domain']} | {r['n_users']:,} | {r['n_items']:,} | "
            f"{r['n_inter']:,} | {r['avg_seq_len']:.2f} | {isp['freq']:.1f} | "
            f"{rsp['freq']:.1f} | {r['cold_inter_share_pct']:.1f} | {r['dist_total_l1_vs_beauty']:.1f} |"
        )

    lines += [
        "",
        "## Amazon 域 — Top 15 最接近 Beauty（inter 分布）",
        "",
        "| Rank | Dataset | AvgSeq | I-new/few/fr | R-new/few/fr | coldR% | Δtotal |",
        "|------|---------|--------|--------------|--------------|--------|--------|",
    ]
    amz = sorted([r for r in rows if r["is_amazon"]], key=lambda x: x["dist_total_l1_vs_beauty"])[:15]
    for i, r in enumerate(amz, 1):
        isp = r["item_strata_pct"]
        rsp = r["inter_strata_pct"]
        lines.append(
            f"| {i} | {r['label']} | {r['avg_seq_len']:.2f} | "
            f"{isp['new']:.0f}/{isp['few']:.0f}/{isp['freq']:.0f} | "
            f"{rsp['new']:.0f}/{rsp['few']:.0f}/{rsp['freq']:.0f} | "
            f"{r['cold_inter_share_pct']:.1f} | {r['dist_total_l1_vs_beauty']:.1f} |"
        )

    lines += [
        "",
        "## 全量表（按 Δtotal 排序）",
        "",
        "<details><summary>展开全部</summary>",
        "",
        "| Dataset | Amazon? | #U | #I | #R | AvgSeq | R-fr% | coldR% | Δitem | Δinter | Δtotal |",
        "|---------|---------|-----|-----|------|--------|-------|--------|-------|--------|--------|",
    ]
    for r in sorted(rows, key=lambda x: x["dist_total_l1_vs_beauty"]):
        lines.append(
            f"| {r['label']} | {'Y' if r['is_amazon'] else 'N'} | {r['n_users']:,} | {r['n_items']:,} | "
            f"{r['n_inter']:,} | {r['avg_seq_len']:.2f} | {r['inter_strata_pct']['freq']:.1f} | "
            f"{r['cold_inter_share_pct']:.1f} | {r['dist_item_strata_l1_vs_beauty']:.1f} | "
            f"{r['dist_inter_strata_l1_vs_beauty']:.1f} | {r['dist_total_l1_vs_beauty']:.1f} |"
        )
    lines += ["", "</details>", ""]

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _save_checkpoint(
    json_path: Path,
    rows: List[Dict],
    errors: List[Dict],
    sources: List,
    t0: float,
    skipped: List[Dict],
) -> None:
    rank_vs_beauty(rows)
    ref = next(
        (r for r in rows if "beauty" in r["label"].lower() and "luxury" not in r["label"].lower()),
        rows[0] if rows else None,
    )
    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
        "n_scanned": len(sources),
        "n_ok": len(rows),
        "n_errors": len(errors),
        "n_skipped": len(skipped),
        "reference": ref["label"] if ref else None,
        "skipped": skipped,
        "errors": errors,
        "datasets": sorted(rows, key=lambda r: r.get("dist_total_l1_vs_beauty", 9999)),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", required=True)
    ap.add_argument("--project-dataset", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--min-inter", type=int, default=5000)
    ap.add_argument(
        "--max-inter-mb",
        type=int,
        default=2048,
        help="Skip datasets whose .inter exceeds this size (avoids OOM on Twitch-full etc.)",
    )
    ap.add_argument("--skip", action="append", default=[], help="Dataset label to skip")
    ap.add_argument(
        "--skip-amazon2018-dup",
        action="store_true",
        default=True,
        help="Skip Amazon2018 zips if same name exists in Amazon_ratings",
    )
    args = ap.parse_args()

    sources = discover_sources(args.processed_root, args.project_dataset)
    print(f"Discovered {len(sources)} unique dataset sources", file=sys.stderr)

    ratings_names = {
        _normalize_key(Path(p).stem)
        for _, p, _ in sources
        if "/Amazon_ratings/" in p.replace("\\", "/")
    }
    skip_labels = {_normalize_key(s) for s in args.skip}

    json_path = Path(args.json)
    rows: List[Dict] = []
    errors: List[Dict] = []
    skipped: List[Dict] = []
    done_labels: set = set()
    if json_path.is_file():
        try:
            prev = json.loads(json_path.read_text(encoding="utf-8"))
            rows = prev.get("datasets", [])
            errors = prev.get("errors", [])
            skipped = prev.get("skipped", [])
            done_labels = {_normalize_key(r["label"]) for r in rows}
            done_labels |= {_normalize_key(e["label"]) for e in errors}
            done_labels |= {_normalize_key(s["label"]) for s in skipped}
            print(f"Resume: {len(done_labels)} datasets already processed", file=sys.stderr)
        except Exception as e:
            print(f"Could not resume from {json_path}: {e}", file=sys.stderr)

    t0 = time.time()
    max_bytes = args.max_inter_mb * 1024 * 1024

    for i, (label, path, _) in enumerate(sources):
        nkey = _normalize_key(label)
        if nkey in done_labels:
            continue
        if nkey in skip_labels:
            skipped.append({"label": label, "source": path, "reason": "user skip"})
            done_labels.add(nkey)
            continue
        nbytes = _inter_uncompressed_bytes(path)
        if nbytes > max_bytes:
            reason = f"inter size {nbytes / 1024 / 1024:.0f}MB > max {args.max_inter_mb}MB"
            skipped.append({"label": label, "source": path, "reason": reason})
            print(f"[{i+1}/{len(sources)}] SKIP {label} ({reason})", file=sys.stderr)
            done_labels.add(nkey)
            _save_checkpoint(json_path, rows, errors, sources, t0, skipped)
            continue
        est = max(0, nbytes // 40) if nbytes else _quick_inter_rows(path)
        if est < args.min_inter:
            skipped.append({"label": label, "source": path, "reason": f"est rows {est}"})
            print(f"[{i+1}/{len(sources)}] SKIP {label} (est rows {est})", file=sys.stderr)
            done_labels.add(nkey)
            continue
        if args.skip_amazon2018_dup and "/Amazon2018/" in path.replace("\\", "/"):
            if _normalize_key(label) in ratings_names:
                skipped.append({"label": label, "source": path, "reason": "Amazon2018 dup"})
                print(f"[{i+1}/{len(sources)}] SKIP dup Amazon2018 {label}", file=sys.stderr)
                done_labels.add(nkey)
                continue
        print(f"[{i+1}/{len(sources)}] {label} ...", file=sys.stderr, flush=True)
        try:
            df = _read_inter(path)
            if len(df) < args.min_inter:
                skipped.append({"label": label, "source": path, "reason": f"only {len(df)} rows"})
                print(f"  skip: only {len(df)} rows", file=sys.stderr)
            else:
                rows.append(analyze_inter(df, label, path))
            done_labels.add(nkey)
        except Exception as e:
            errors.append({"label": label, "source": path, "error": str(e)})
            done_labels.add(nkey)
            print(f"  ERROR: {e}", file=sys.stderr)
        _save_checkpoint(json_path, rows, errors, sources, t0, skipped)

    rank_vs_beauty(rows)
    rows.sort(key=lambda r: r.get("dist_total_l1_vs_beauty", 9999))

    ref = next(
        (r for r in rows if "beauty" in r["label"].lower() and "luxury" not in r["label"].lower()),
        rows[0] if rows else None,
    )

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
        "n_scanned": len(sources),
        "n_ok": len(rows),
        "n_errors": len(errors),
        "n_skipped": len(skipped),
        "reference": ref["label"] if ref else None,
        "skipped": skipped,
        "errors": errors,
        "datasets": rows,
    }

    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {json_path}", file=sys.stderr)

    if ref:
        write_markdown(rows, args.md, ref)
        print(f"Wrote {args.md}", file=sys.stderr)

    print(f"Done: {len(rows)} datasets in {out['elapsed_sec']}s", file=sys.stderr)


if __name__ == "__main__":
    main()
