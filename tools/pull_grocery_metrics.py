#!/usr/bin/env python3
"""Extract Grocery metrics from run_metrics/*.txt."""
import json
import glob
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "Amazon_Grocery_and_Gourmet_Food"


def get_test(d):
    r = d.get("results", {})
    pbr = r.get("phase_b_result") or {}
    return pbr.get("test_result") or r.get("test_result") or {}


def infer_config(ckpt: str, model: str) -> str:
    if "baseline_grocery" in ckpt or "baseline" in ckpt and model == "SASRecAlign":
        return "ID (SASRecAlign)"
    if "multiview" in ckpt.lower():
        return "MV-Align"
    if "tfidf_llm" in ckpt:
        return "TF-IDF+LLM"
    if "tfidf" in ckpt:
        return "TF-IDF"
    if "MultiView" in model:
        return "MV-Align"
    if model == "SASRecAlignV3":
        return "TF-IDF+LLM" if "llm" in ckpt else "TF-IDF"
    return model


def scan_dir(metrics_dir: Path, label_prefix: str = ""):
    rows = []
    for fp in sorted(metrics_dir.glob("*.txt")):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        if d.get("dataset") != DATASET:
            continue
        tr = get_test(d)
        if not tr or tr.get("mrr@10") is None:
            continue
        ckpt = d.get("results", {}).get("phase_b_result", {}).get("saved_model_file", "")
        if not ckpt and d.get("label") == "Phase-A":
            ckpt = "baseline"
        m = re.search(r"seed(\d+|42)", ckpt + fp.name)
        seed = m.group(1) if m else "-"
        rows.append(
            {
                "source": label_prefix or metrics_dir.name,
                "ts": d.get("timestamp", ""),
                "label": d.get("label", ""),
                "config": infer_config(ckpt, d.get("model", "")),
                "seed": seed,
                "mrr10": round(tr["mrr@10"] * 100, 4),
                "rec10": round(tr.get("recall@10", 0) * 100, 4),
                "mrr_new": round(tr.get("MRR_new@10", 0) * 100, 4),
                "mrr_few": round(tr.get("MRR_few@10", 0) * 100, 4),
                "mrr_freq": round(tr.get("MRR_frequent@10", 0) * 100, 4),
            }
        )
    return rows


def main():
    dirs = [ROOT / "run_metrics"]
    log10 = ROOT / "run_metrics_log10"
    if log10.is_dir():
        dirs.append(log10)
    if len(sys.argv) > 1:
        dirs = [Path(sys.argv[1])]

    all_rows = []
    for d in dirs:
        all_rows.extend(scan_dir(d, d.name))

    print("source\tts\tlabel\tconfig\tseed\tMRR@10\tRec@10\tMRR_new\tMRR_few\tMRR_freq")
    for r in sorted(all_rows, key=lambda x: (x["config"], x["seed"], x["ts"])):
        print(
            f"{r['source']}\t{r['ts']}\t{r['label']}\t{r['config']}\t{r['seed']}\t"
            f"{r['mrr10']}\t{r['rec10']}\t{r['mrr_new']}\t{r['mrr_few']}\t{r['mrr_freq']}"
        )


if __name__ == "__main__":
    main()
