#!/usr/bin/env python3
"""Build paper main tables from GTS-Last / GTS-Random / (optional) LOO CSVs.

Outputs:
  paper_recsys/main_table_gts_last_random.md
  paper_recsys/main_table_gts_last_random.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fam(tag: str) -> str:
    t = tag.lower()
    if "unisrec" in t:
        return "UniSRec"
    if "_id_" in t:
        return "ID"
    if "_tf_" in t:
        return "TF"
    if "_llm_" in t:
        return "LLM"
    if "_mv_" in t:
        return "MV"
    return "?"


def agg(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = df.copy()
    df["fam"] = df["tag"].map(fam)
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    rows = []
    for (ds, f), g in df.groupby(["dataset", "fam"]):
        rows.append({
            "dataset": ds,
            "family": f,
            "mean": g[metric].mean(),
            "std": g[metric].std(ddof=1) if len(g) > 1 else 0.0,
            "n": len(g),
        })
    return pd.DataFrame(rows)


def fmt(mean, std):
    return f"{mean:.4f}±{std:.4f}"


def pivot_block(a: pd.DataFrame, datasets, families):
    out = []
    for ds in datasets:
        row = {"dataset": ds}
        for f in families:
            sub = a[(a.dataset == ds) & (a.family == f)]
            if len(sub) == 0:
                row[f] = "—"
            else:
                row[f] = fmt(sub.iloc[0]["mean"], sub.iloc[0]["std"])
        out.append(row)
    return pd.DataFrame(out)


def load_loo_means(path: Path):
    """Optional LOO csv with columns varying; best-effort."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Expected from prior wrap: tag/model + mrr
    cols = {c.lower(): c for c in df.columns}
    mrr_col = None
    for k in cols:
        if "mrr" in k and "10" in k:
            mrr_col = cols[k]
            break
    if mrr_col is None:
        return None
    tag_col = cols.get("tag") or cols.get("run") or cols.get("model")
    if tag_col is None:
        return None
    df["fam"] = df[tag_col].astype(str).map(fam)
    # detect dataset from filename
    ds = "Amazon_Beauty" if "beauty" in path.name.lower() else "Amazon_Toys_and_Games"
    g = df.groupby("fam")[mrr_col].agg(["mean", "std", "count"])
    rows = []
    for f, r in g.iterrows():
        rows.append({"dataset": ds, "family": f, "mean": r["mean"],
                     "std": r["std"] if r["count"] > 1 else 0.0, "n": int(r["count"])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", default="paper_recsys/gts_last_eval.csv")
    ap.add_argument("--random", default="paper_recsys/gts_random_eval.csv")
    ap.add_argument("--out_md", default="paper_recsys/main_table_gts_last_random.md")
    ap.add_argument("--out_csv", default="paper_recsys/main_table_gts_last_random.csv")
    ap.add_argument("--metric_all", default="test_mrr@10_all")
    ap.add_argument("--metric_warm", default="test_mrr@10_warm")
    args = ap.parse_args()

    families = ["ID", "TF", "LLM", "MV", "UniSRec"]
    datasets = ["Amazon_Beauty", "Amazon_Toys_and_Games"]
    short = {"Amazon_Beauty": "Beauty", "Amazon_Toys_and_Games": "Toys"}

    last_path = ROOT / args.last
    rand_path = ROOT / args.random
    last = pd.read_csv(last_path)
    has_random = rand_path.exists() and rand_path.stat().st_size > 50
    rand = pd.read_csv(rand_path) if has_random else None

    blocks = {}
    long_rows = []

    for mode_name, df in [("GTS-Last", last)] + ([("GTS-Random", rand)] if rand is not None else []):
        for metric, scope in [(args.metric_all, "all"), (args.metric_warm, "warm")]:
            a = agg(df, metric)
            key = f"{mode_name}/{scope}"
            blocks[key] = pivot_block(a, datasets, families)
            for _, r in a.iterrows():
                long_rows.append({
                    "eval_mode": mode_name,
                    "scope": scope,
                    "dataset": r["dataset"],
                    "family": r["family"],
                    "mrr@10_mean": r["mean"],
                    "mrr@10_std": r["std"],
                    "n_seeds": r["n"],
                })

    # Markdown
    lines = []
    lines.append("# Main Tables — GTS-Last / GTS-Random (MRR@10)")
    lines.append("")
    lines.append("> Protocol lock (0803): primary = **Last / Random**, warm preferred for main claims;")
    lines.append("> GTS-All and Clean LOO are contrast / appendix only.")
    lines.append(">")
    lines.append(f"> Last source: `{args.last}`")
    if has_random:
        lines.append(f"> Random source: `{args.random}`")
    else:
        lines.append("> Random source: **PENDING** (run `run_5090_gts_random.sh`)")
    lines.append("")
    lines.append("min_train_interactions=5; 3 seeds {42,2024,2026}; mean±sample-std.")
    lines.append("")

    for mode_name in (["GTS-Last", "GTS-Random"] if has_random else ["GTS-Last"]):
        lines.append(f"## {mode_name}")
        lines.append("")
        for scope, title in [("all", "All-user (one target / user)"),
                             ("warm", "Warm targets only (target ∈ train catalog)")]:
            key = f"{mode_name}/{scope}"
            lines.append(f"### {title}")
            lines.append("")
            header = "| Dataset | " + " | ".join(families) + " |"
            sep = "|---|---:|---:|---:|---:|---:|"
            lines.append(header)
            lines.append(sep)
            pv = blocks[key]
            for _, r in pv.iterrows():
                cells = [short.get(r["dataset"], r["dataset"])] + [r[f] for f in families]
                # bold best among ID/TF/LLM/MV (not UniSRec) by parsing mean
                means = []
                for f in families:
                    try:
                        means.append(float(r[f].split("±")[0]))
                    except Exception:
                        means.append(-1)
                best_i = int(np.argmax(means[:4]))  # among ID..MV
                for i, f in enumerate(families):
                    if i == best_i and means[i] >= 0:
                        cells[i + 1] = f"**{cells[i + 1]}**"
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

        # Ranking summary
        lines.append(f"### Ranking summary ({mode_name}, warm)")
        lines.append("")
        pv = blocks[f"{mode_name}/warm"]
        for _, r in pv.iterrows():
            scored = []
            for f in families:
                try:
                    scored.append((f, float(r[f].split("±")[0])))
                except Exception:
                    pass
            scored.sort(key=lambda x: -x[1])
            order = " > ".join(f"{f}({v:.4f})" for f, v in scored)
            lines.append(f"- **{short.get(r['dataset'], r['dataset'])}**: {order}")
        lines.append("")

    if not has_random:
        lines.append("## GTS-Random")
        lines.append("")
        lines.append("_Not available yet. Re-run this script after `logs/gts_random_eval.csv` is ready._")
        lines.append("")

    # Contrast note vs All (hardcoded known means from training logs)
    lines.append("## Contrast: GTS-All (appendix reference, not main)")
    lines.append("")
    lines.append("| Dataset | ID | TF | LLM | MV | UniSRec |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append("| Beauty | 0.0106±0.0005 | **0.0165±0.0008** | 0.0161±0.0005 | 0.0160±0.0008 | 0.0123±0.0001 |")
    lines.append("| Toys | 0.0070±0.0001 | **0.0113±0.0003** | 0.0107±0.0004 | 0.0109±0.0006 | 0.0082±0.0001 |")
    lines.append("")
    lines.append("Clean LOO (contrast): Beauty TF 0.0251 / LLM 0.0254 / MV 0.0253; "
                 "Toys TF 0.0247 / LLM≈MV 0.0258.")
    lines.append("")

    out_md = ROOT / args.out_md
    out_md.write_text("\n".join(lines) + "\n")
    out_csv = ROOT / args.out_csv
    pd.DataFrame(long_rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_md}")
    print(f"Wrote {out_csv}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
