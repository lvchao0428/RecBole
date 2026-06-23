#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strata eval feasibility + text diversity audit for candidate datasets.

Combines interaction strata (new/few/freq) with text-side metrics that proxy
whether multi-view can differ from single-view (field redundancy, TF-IDF diversity).

Usage:
  python tools/dataset_strata_text_audit.py \\
    --zip /path/BeerAdvocate.zip --name BeerAdvocate \\
    --primary name --secondary style \\
    --ref-json paper_recsys/dataset_strata_all_20260623.json \\
    --json out.json --md out.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.dataset_strata_compare import (  # noqa: E402
    _field_col,
    _read_inter,
    analyze_inter,
)


def _read_item(path: str) -> pd.DataFrame:
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if n.endswith(".item") and "__" not in n]
            if not names:
                raise FileNotFoundError(f"No .item in {path}")
            names.sort(key=len)
            with zf.open(names[0]) as f:
                return pd.read_csv(f, sep="\t")
    p = Path(path)
    if p.suffix == ".inter":
        ip = p.parent / (p.stem + ".item")
    elif p.suffix == ".zip":
        raise FileNotFoundError(path)
    else:
        ip = p
    if not ip.exists():
        raise FileNotFoundError(ip)
    return pd.read_csv(ip, sep="\t")


def _col(df: pd.DataFrame, base: str) -> Optional[str]:
    return _field_col(list(df.columns), base)


def _seq_to_str(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    s = str(val).strip()
    if s in ("", "nan", "None"):
        return ""
    return s


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower())


def _strata_masks(item_deg: pd.Series) -> Dict[str, pd.Series]:
    return {
        "new": (item_deg >= 1) & (item_deg < 3),
        "few": (item_deg >= 3) & (item_deg < 10),
        "freq": item_deg >= 10,
    }


def _eval_feasibility(inter_stats: Dict, item_deg: pd.Series) -> Dict:
    """Estimate whether new/few strata have enough mass for stratified eval."""
    n_inter = inter_stats["n_inter"]
    n_items = inter_stats["n_items"]
    n_users = inter_stats["n_users"]
    rsp = inter_stats["inter_strata_pct"]
    isp = inter_stats["item_strata_pct"]

    masks = _strata_masks(item_deg)
    inter_counts = {
        k: int((item_deg[masks[k]] * 0).sum()) for k in masks
    }
    # recompute from inter_stats
    for k in ("new", "few", "freq"):
        inter_counts[k] = int(round(n_inter * rsp[k] / 100))

    item_counts = {k: int(masks[k].sum()) for k in masks}

    # Rough test-size proxy: one held-out interaction per user (leave-one-out style)
    est_test_per_stratum = {k: int(round(n_users * rsp[k] / 100)) for k in rsp}

    def grade(r_pct: float, est_n: int) -> str:
        if r_pct < 3 or est_n < 500:
            return "unlikely"  # too few test hits
        if r_pct < 8 or est_n < 3000:
            return "marginal"
        if r_pct < 15 or est_n < 10000:
            return "moderate"
        return "good"

    strata_eval = {}
    for k in ("new", "few", "freq"):
        r_pct = rsp[k]
        est_n = est_test_per_stratum[k]
        strata_eval[k] = {
            "item_count": item_counts[k],
            "item_pct": isp[k],
            "inter_count": inter_counts[k],
            "inter_pct": r_pct,
            "est_test_hits": est_n,
            "eval_feasibility": grade(r_pct, est_n),
        }

    cold_r = rsp["new"] + rsp["few"]
    cold_est = est_test_per_stratum["new"] + est_test_per_stratum["few"]
    return {
        "strata": strata_eval,
        "cold_inter_share_pct": cold_r,
        "est_cold_test_hits": cold_est,
        "overall_eval_note": (
            "good" if cold_r >= 20 and cold_est >= 8000
            else "marginal" if cold_r >= 8 and cold_est >= 2000
            else "unlikely"
        ),
    }


def _text_basic(texts: List[str]) -> Dict:
    lens = [len(t) for t in texts]
    wcounts = [len(_tokenize_words(t)) for t in texts]
    nonempty = [t for t in texts if len(t.strip()) >= 2]
    return {
        "n_items": len(texts),
        "nonempty_ratio": len(nonempty) / max(len(texts), 1),
        "char_len_mean": float(np.mean(lens)) if lens else 0,
        "char_len_median": float(np.median(lens)) if lens else 0,
        "word_count_mean": float(np.mean(wcounts)) if wcounts else 0,
        "word_count_median": float(np.median(wcounts)) if wcounts else 0,
    }


def _vocab_stats(texts: List[str]) -> Dict:
    all_toks: List[str] = []
    for t in texts:
        all_toks.extend(_tokenize_words(t))
    if not all_toks:
        return {"unique_tokens": 0, "total_tokens": 0, "ttr": 0, "top50_coverage": 0}
    from collections import Counter

    freq = Counter(all_toks)
    total = len(all_toks)
    top50 = sum(c for _, c in freq.most_common(50))
    return {
        "unique_tokens": len(freq),
        "total_tokens": total,
        "ttr": len(freq) / total,
        "top50_coverage": top50 / total,
    }


def _tfidf_pairwise_sim(texts: List[str], max_items: int = 3000, random_state: int = 42) -> Dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    idx = np.arange(len(texts))
    if len(texts) > max_items:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(texts), max_items, replace=False)
    sub = [texts[i] if texts[i].strip() else " " for i in idx]
    if len(sub) < 10:
        return {"n_sampled": len(sub), "mean_cosine": None, "p90_cosine": None, "high_sim_ratio_gt0.9": None}

    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=50000)
    X = vec.fit_transform(sub)
    sim = cosine_similarity(X)
    mask = ~np.eye(sim.shape[0], dtype=bool)
    vals = sim[mask]
    return {
        "n_sampled": len(sub),
        "vocab_size": len(vec.vocabulary_),
        "mean_cosine": float(np.mean(vals)),
        "std_cosine": float(np.std(vals)),
        "p90_cosine": float(np.percentile(vals, 90)),
        "high_sim_ratio_gt0.9": float(np.mean(vals > 0.9)),
        "low_sim_ratio_lt0.3": float(np.mean(vals < 0.3)),
    }


def _field_cross_diversity(text_a: List[str], text_b: List[str], max_items: int = 3000) -> Dict:
    """Per-item complementarity between two text fields (lower cosine = more complementary)."""
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    n = min(len(text_a), len(text_b))
    if n < 10:
        return {"mean_cross_field_cosine": None, "diversity_score": None}
    idx = np.arange(n)
    if n > max_items:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_items, replace=False)
    a = [text_a[i] if text_a[i].strip() else " " for i in idx]
    b = [text_b[i] if text_b[i].strip() else " " for i in idx]
    hv = HashingVectorizer(analyzer="char", ngram_range=(2, 4), n_features=8192, alternate_sign=False)
    Xa = normalize(hv.transform(a))
    Xb = normalize(hv.transform(b))
    sims = np.array([cosine_similarity(Xa[i], Xb[i])[0, 0] for i in range(Xa.shape[0])])
    mean_sim = float(np.mean(sims))
    return {
        "mean_cross_field_cosine": mean_sim,
        "diversity_score": float(1 - mean_sim),
        "p90_cross_field_cosine": float(np.percentile(sims, 90)),
        "n_sampled": len(sims),
    }


def _category_entropy(categories: List[str]) -> Dict:
    from collections import Counter

    cats = []
    for c in categories:
        s = _seq_to_str(c)
        if not s:
            continue
        # split RecBole token_seq style: 'A', 'B' or comma-separated
        parts = re.split(r"[,']+", s)
        cats.extend(p.strip().lower() for p in parts if p.strip())
    if not cats:
        return {"n_unique_cats": 0, "entropy_bits": 0, "top5_coverage": 0}
    freq = Counter(cats)
    total = sum(freq.values())
    probs = np.array([c / total for c in freq.values()])
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    top5 = sum(c for _, c in freq.most_common(5)) / total
    return {
        "n_unique_cats": len(freq),
        "entropy_bits": entropy,
        "top5_coverage": float(top5),
    }


def _text_by_stratum(
    item_df: pd.DataFrame,
    item_deg: pd.Series,
    icol: str,
    primary: str,
    secondary: Optional[str],
) -> Dict:
    pcol = _col(item_df, primary)
    if pcol is None:
        raise ValueError(f"primary field {primary!r} not in {list(item_df.columns)}")
    scol = _col(item_df, secondary) if secondary else None

    item_df = item_df.copy()
    item_df["_iid"] = item_df[icol].astype(str)
    deg_map = item_deg.to_dict()
    item_df["_deg"] = item_df["_iid"].map(deg_map).fillna(0).astype(int)

    def stratum(d):
        if d < 3:
            return "new"
        if d < 10:
            return "few"
        return "freq"

    item_df["_stratum"] = item_df["_deg"].map(stratum)
    primary_texts = item_df[pcol].map(_seq_to_str).tolist()
    secondary_texts = item_df[scol].map(_seq_to_str).tolist() if scol else None

    out = {
        "primary_field": primary,
        "secondary_field": secondary,
        "combined": {},
        "by_stratum": {},
    }
    combined = [
        (p + (" " + s if s else "")).strip() for p, s in zip(primary_texts, secondary_texts or [""] * len(primary_texts))
    ]
    out["combined"]["basic"] = _text_basic(combined)
    out["combined"]["vocab"] = _vocab_stats(combined)
    out["combined"]["tfidf_sim"] = _tfidf_pairwise_sim(combined)
    if secondary_texts:
        out["combined"]["cross_field"] = _field_cross_diversity(primary_texts, secondary_texts)
        if secondary in ("categories", "style", "tags"):
            out["combined"]["category_entropy"] = _category_entropy(secondary_texts)

    for st in ("new", "few", "freq"):
        sub = item_df[item_df["_stratum"] == st]
        if sub.empty:
            continue
        texts = sub[pcol].map(_seq_to_str).tolist()
        out["by_stratum"][st] = {
            "n_items": len(sub),
            "basic": _text_basic(texts),
            "vocab": _vocab_stats(texts),
        }
    return out


def analyze_dataset(
    label: str,
    inter_path: str,
    primary: str,
    secondary: Optional[str],
    item_path: Optional[str] = None,
) -> Dict:
    df = _read_inter(inter_path)
    inter_stats = analyze_inter(df, label, inter_path)

    icol = _field_col(list(df.columns), "item_id")
    item_deg = df.groupby(icol).size()
    eval_feas = _eval_feasibility(inter_stats, item_deg)

    item_src = item_path or inter_path
    item_df = _read_item(item_src)
    iid_item = _col(item_df, "item_id")
    if iid_item is None:
        raise ValueError(f"{label}: no item_id in item file")

    text_stats = _text_by_stratum(item_df, item_deg, iid_item, primary, secondary)

    # MV potential heuristic vs single concatenated view
    cf = text_stats["combined"].get("cross_field", {})
    tfidf = text_stats["combined"].get("tfidf_sim", {})
    div_score = cf.get("diversity_score")
    mean_sim = tfidf.get("mean_cosine")
    if div_score is not None and div_score >= 0.35 and mean_sim is not None and mean_sim < 0.25:
        mv_potential = "high"
    elif div_score is not None and div_score >= 0.20:
        mv_potential = "moderate"
    else:
        mv_potential = "low"

    return {
        "label": label,
        "inter_path": inter_path,
        "inter": inter_stats,
        "eval_feasibility": eval_feas,
        "text": text_stats,
        "mv_potential_heuristic": mv_potential,
    }


def _load_ref(path: str, ref_label: str = "Amazon_Beauty") -> Optional[Dict]:
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    datasets = data.get("datasets", data)
    for r in datasets:
        if r["label"] == ref_label or ref_label.lower() in r["label"].lower():
            return r
    return None


def write_md(rows: List[Dict], ref: Optional[Dict], path: str) -> None:
    lines = [
        "# Beer / Yelp 分层可评估性 + 文本多样性审计",
        "",
        "> 分层: new=1–2, few=3–9, freq≥10 次交互（全量 .inter item degree）",
        "> eval 可行性: 按「每用户 1 条 test、test item  strata 占比≈R-strata%」粗估",
        "",
    ]
    if ref:
        rsp = ref["inter_strata_pct"]
        lines.append(
            f"> 参照 **{ref['label']}**: R-new={rsp['new']:.1f}% R-few={rsp['few']:.1f}% "
            f"R-fr={rsp['freq']:.1f}% coldR={ref['cold_inter_share_pct']:.1f}%"
        )
        lines.append("")

    lines += [
        "## 1. 分层与 eval 可行性（能否看出 new/few 效果）",
        "",
        "| Dataset | R-new% | R-few% | R-fr% | est.test new | est.test few | new 评级 | few 评级 | cold 总体 |",
        "|---------|--------|--------|-------|--------------|--------------|----------|----------|-----------|",
    ]
    for r in rows:
        ev = r["eval_feasibility"]
        s = ev["strata"]
        lines.append(
            f"| {r['label']} | {s['new']['inter_pct']:.1f} | {s['few']['inter_pct']:.1f} | "
            f"{s['freq']['inter_pct']:.1f} | {s['new']['est_test_hits']:,} | {s['few']['est_test_hits']:,} | "
            f"{s['new']['eval_feasibility']} | {s['few']['eval_feasibility']} | {ev['overall_eval_note']} |"
        )

    lines += [
        "",
        "**评级**: `good`≥15% R-strata & ≥10K est.test; `moderate`≥8%; `marginal`≥3%; else `unlikely`",
        "",
        "## 2. 文本多样性（MV vs single-view 潜力）",
        "",
        "| Dataset | 主字段 | 副字段 | 词数均值 | TTR | TF-IDF均相似度 | 跨字段多样性* | MV潜力 |",
        "|---------|--------|--------|----------|-----|----------------|---------------|--------|",
    ]
    for r in rows:
        t = r["text"]["combined"]
        b = t["basic"]
        v = t["vocab"]
        tf = t.get("tfidf_sim", {})
        cf = t.get("cross_field", {})
        lines.append(
            f"| {r['label']} | {r['text']['primary_field']} | {r['text']['secondary_field'] or '-'} | "
            f"{b['word_count_mean']:.1f} | {v['ttr']:.3f} | "
            f"{tf.get('mean_cosine', float('nan')):.3f} | "
            f"{cf.get('diversity_score', float('nan')):.3f} | {r['mv_potential_heuristic']} |"
        )
    lines.append("")
    lines.append("*跨字段多样性 = 1 − mean_cosine(TF-IDF(主字段), TF-IDF(副字段))，越高表示两路文本越互补")

    if ref:
        lines += ["", "## 3. 与 Beauty 对比", ""]
        rr = ref["inter_strata_pct"]
        lines.append(f"| 指标 | Beauty | " + " | ".join(r["label"] for r in rows) + " |")
        lines.append("|" + "---|" * (len(rows) + 2))
        for key, fmt, src in [
            ("R-new%", lambda r: r["eval_feasibility"]["strata"]["new"]["inter_pct"], "inter"),
            ("R-few%", lambda r: r["eval_feasibility"]["strata"]["few"]["inter_pct"], "inter"),
            ("coldR%", lambda r: r["eval_feasibility"]["cold_inter_share_pct"], "inter"),
            ("AvgSeq", lambda r: r["inter"]["avg_seq_len"], "inter"),
            ("词数均值", lambda r: r["text"]["combined"]["basic"]["word_count_mean"], "text"),
            ("TTR", lambda r: r["text"]["combined"]["vocab"]["ttr"], "text"),
        ]:
            vals = " | ".join(fmt(r).__format__(".1f") if isinstance(fmt(r), float) else f"{fmt(r):.1f}" for r in rows)
            ref_v = {
                "R-new%": rr["new"],
                "R-few%": rr["few"],
                "coldR%": ref["cold_inter_share_pct"],
                "AvgSeq": ref["avg_seq_len"],
            }.get(key, "")
            if key in ("词数均值", "TTR"):
                ref_v = "-"
            lines.append(f"| {key} | {ref_v} | {vals} |")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="append", default=[])
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--inter", action="append", default=[])
    ap.add_argument("--inter-name", action="append", default=[])
    ap.add_argument("--primary", action="append", default=[], help="Primary text field per dataset")
    ap.add_argument("--secondary", action="append", default=[], help="Secondary field (optional)")
    ap.add_argument("--ref-json", default=None)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    args = ap.parse_args()

    specs: List[Tuple[str, str, str, Optional[str]]] = []
    for z, n in zip(args.zip, args.name):
        specs.append((n, z, "", None))
    for p, n in zip(args.inter, args.inter_name):
        specs.append((n, p, "", None))

    if len(specs) != len(args.primary):
        sys.exit("Need --primary for each dataset")

    rows = []
    for i, (label, path, _, _) in enumerate(specs):
        sec = args.secondary[i] if i < len(args.secondary) and args.secondary[i] else None
        print(f"Analyzing {label} ...", file=sys.stderr)
        rows.append(analyze_dataset(label, path, args.primary[i], sec))

    ref = _load_ref(args.ref_json) if args.ref_json else None
    out = {"reference": ref["label"] if ref else None, "datasets": rows}
    jp = Path(args.json)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(rows, ref, args.md)
    print(f"Wrote {args.json} and {args.md}", file=sys.stderr)


if __name__ == "__main__":
    main()
