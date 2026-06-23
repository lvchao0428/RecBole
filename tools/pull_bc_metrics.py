#!/usr/bin/env python3
"""Extract book-crossing metrics from run_metrics/*.txt on 5090."""
import json
import glob
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get_test(d):
    r = d.get("results", {})
    pbr = r.get("phase_b_result") or {}
    return pbr.get("test_result") or r.get("test_result") or {}


def infer_config(ckpt: str, model: str, cfg: dict) -> str:
    if "baseline_v3" in ckpt:
        return "ID (MultiViewV3)"
    if "baseline_60ep" in ckpt or "baseline_book_crossing" in ckpt:
        return "ID (SASRecAlign)"
    if "multiview" in ckpt.lower():
        return "MV-Align"
    if "tfidf_llm" in ckpt:
        return "TF-IDF+LLM"
    if "tfidf" in ckpt:
        return "TF-IDF"
    ta = cfg.get("text_align", {})
    if ta.get("use_llm"):
        return "TF-IDF+LLM"
    if "MultiView" in model:
        return "MV-Align"
    if model == "SASRecAlignV3":
        return "TF-IDF"
    if model == "SASRecAlign":
        return "ID (SASRecAlign)"
    return model


def pct(x):
    return round(x * 100, 4) if x is not None else None


def main():
    metrics_dir = ROOT / "run_metrics"
    if len(sys.argv) > 1:
        metrics_dir = Path(sys.argv[1])

    rows = []
    for fp in sorted(metrics_dir.glob("*.txt")):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        if d.get("dataset") != "book-crossing":
            continue
        if d.get("label") != "Phase-B":
            continue
        tr = get_test(d)
        if not tr:
            continue
        ckpt = d.get("results", {}).get("phase_b_result", {}).get("saved_model_file", "")
        m = re.search(r"seed(\d+|42)", ckpt)
        seed = m.group(1) if m else "-"
        tag = "phaseb50" if "phaseb50" in ckpt else "phaseb40"
        rows.append(
            {
                "ts": d["timestamp"],
                "config": infer_config(ckpt, d.get("model", ""), d.get("config_groups", {})),
                "seed": seed,
                "tag": tag,
                "mrr10": pct(tr.get("mrr@10")),
                "rec10": pct(tr.get("recall@10")),
                "hr10": pct(tr.get("hit@10")),
                "mrr_new": pct(tr.get("MRR_new@10")),
                "mrr_few": pct(tr.get("MRR_few@10")),
                "mrr_freq": pct(tr.get("MRR_frequent@10")),
                "ckpt": ckpt,
            }
        )

    rows.sort(key=lambda x: (x["config"], x["seed"], x["ts"]))
    print(
        "ts\tconfig\tseed\ttag\tMRR@10\tRec@10\tHR@10\tMRR_new\tMRR_few\tMRR_freq\tcheckpoint"
    )
    for r in rows:
        ck = r["ckpt"].split("/")[1] if r["ckpt"] else ""
        print(
            f"{r['ts']}\t{r['config']}\t{r['seed']}\t{r['tag']}\t"
            f"{r['mrr10']}\t{r['rec10']}\t{r['hr10']}\t{r['mrr_new']}\t{r['mrr_few']}\t{r['mrr_freq']}\t{ck}"
        )


if __name__ == "__main__":
    main()
