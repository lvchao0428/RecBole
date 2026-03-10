# -*- coding: utf-8 -*-
"""
Compute mean +/- std over 5 seeds for the main table (Table 2),
and run paired t-tests (MV-Align 7B vs strongest baseline).

Seeds: 42, 2023, 2024, 2025, 2026

Row layout per seed file (tab-separated):
  Row 0:  seed label
  Row 1:  header
  Row 2:  sasrec base50ep  | beauty | ... | metrics...
  Row 3:  tfidf            |        | ... | metrics...
  Row 4:  tfidf + llm      |        | ... | metrics...
  Row 5:  multi-view 7b    |        | ... | metrics...
  Row 6:  multi-view 14b   |        | ... | metrics...
  Row 7-8: blank
  Row 9:  sasrec 50ep      | toy    | ... | metrics...
  Row 10: tfidf            |        | ... | metrics...
  Row 11: tfidf + llm      |        | ... | metrics...
  Row 12: multi-view 7b    |        | ... | metrics...
  Row 13: multi-view 14b   |        | ... | metrics...

Metric columns (0-indexed from field[5], i.e. first numeric field):
  0:recall@5  1:recall@10  2:recall@20
  3:mrr@5     4:mrr@10     5:mrr@20
  6:ndcg@5    7:ndcg@10    8:ndcg@20
  9:hit@5    10:hit@10    11:hit@20
  12:prec@5  13:prec@10   14:prec@20
  15-17: Recall_new @5/10/20
  18-20: Recall_few @5/10/20
  21-23: Recall_frequent @5/10/20
  24-26: NDCG_new @5/10/20
  27-29: NDCG_few @5/10/20
  30-32: NDCG_frequent @5/10/20
  33-35: MRR_new @5/10/20
  36-38: MRR_few @5/10/20
  39-41: MRR_frequent @5/10/20
  42-44: Hit_new @5/10/20
  45-47: Hit_few @5/10/20
  48-50: Hit_frequent @5/10/20
"""

import numpy as np
from scipy import stats
import os

METRIC_COL = {
    "HR@10":       10,   # hit@10
    "NDCG@10":      7,   # ndcg@10
    "MRR@10":       4,   # mrr@10
    "HR_new@10":   43,   # Hit_new@10
    "NDCG_new@10": 25,   # NDCG_new@10
    "MRR_new@10":  34,   # MRR_new@10
    "HR_few@10":   46,   # Hit_few@10
    "NDCG_few@10": 28,   # NDCG_few@10
    "MRR_few@10":  37,   # MRR_few@10
    "HR_freq@10":  49,   # Hit_frequent@10
    "NDCG_freq@10":31,   # NDCG_frequent@10
    "MRR_freq@10": 40,   # MRR_frequent@10
}

MODEL_NAMES = [
    r"\base (ID-only, 50ep)",
    r"\base + \tfidf",
    r"\base + \tfidf + \llm (7B)",
    r"\model (7B)",
]

BEAUTY_ROWS = [2, 3, 4, 5]   # line indices in the file
TOYS_ROWS   = [9, 10, 11, 12]

SEED_FILES = {
    42:   "seed42.txt",
    2024: "seed2024.txt",
    2025: "seed2025.txt",
    2026: "seed2026.txt",
}

METRIC_START_FIELD = 5  # first numeric value is at tab-field index 5


def parse_seed_file(filepath):
    """Return {dataset: {model_idx: [float, ...] of metric values}}"""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    def extract_row(line):
        parts = line.rstrip("\n").split("\t")
        raw = parts[METRIC_START_FIELD:]
        vals = []
        for v in raw:
            v = v.strip().rstrip("%")
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(None)
        return vals

    result = {}
    for dataset, row_indices in [("Beauty", BEAUTY_ROWS), ("Toys", TOYS_ROWS)]:
        models = {}
        for mi, ri in enumerate(row_indices):
            models[mi] = extract_row(lines[ri])
        result[dataset] = models
    return result


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    all_seeds = {}
    for seed, fname in sorted(SEED_FILES.items()):
        fpath = os.path.join(script_dir, fname)
        all_seeds[seed] = parse_seed_file(fpath)

    seeds_list = sorted(all_seeds.keys())
    print("Seeds: %s\n" % seeds_list)

    metric_order = ["HR@10", "NDCG@10", "MRR@10",
                    "HR_new@10", "NDCG_new@10", "MRR_new@10",
                    "HR_few@10", "NDCG_few@10", "MRR_few@10",
                    "HR_freq@10", "NDCG_freq@10", "MRR_freq@10"]

    for dataset in ["Beauty", "Toys"]:
        print("=" * 100)
        print("  Dataset: Amazon %s" % dataset)
        print("=" * 100)

        # Gather per-model, per-metric arrays across seeds
        model_data = {}
        for mi in range(4):
            model_data[mi] = {}
            for mname, mcol in METRIC_COL.items():
                vals = []
                for seed in seeds_list:
                    row = all_seeds[seed][dataset][mi]
                    v = row[mcol] if mcol < len(row) else None
                    vals.append(v)
                model_data[mi][mname] = vals

        # ---- Per-seed raw values ----
        header = "%-26s" % "Model"
        for mname in metric_order:
            header += " %13s" % mname
        print("\n" + header)
        print("-" * 106)

        for mi in range(4):
            print("  %s" % MODEL_NAMES[mi])
            for si, seed in enumerate(seeds_list):
                line = "    seed=%-5d" % seed
                for mname in metric_order:
                    v = model_data[mi][mname][si]
                    if v is not None:
                        line += " %12.2f%%" % (v * 100)
                    else:
                        line += " %13s" % "N/A"
                print(line)

        # ---- Mean +/- Std ----
        print("\n" + ("--- Mean +/- Std (5 seeds) ---").center(106))
        header2 = "%-26s" % "Model"
        for mname in metric_order:
            header2 += " %13s" % mname
        print(header2)
        print("-" * 106)

        means = {}
        stds = {}
        for mi in range(4):
            means[mi] = {}
            stds[mi] = {}
            line = "  %-24s" % MODEL_NAMES[mi]
            for mname in metric_order:
                arr = np.array([v for v in model_data[mi][mname] if v is not None])
                m = arr.mean() * 100
                s = arr.std(ddof=1) * 100
                means[mi][mname] = m
                stds[mi][mname] = s
                line += " %5.2f+/-%-5.2f" % (m, s)
            print(line)

        # ---- Paired t-tests ----
        overall_metrics = ["HR@10", "NDCG@10", "MRR@10"]

        # MV-Align(7B) vs TF-IDF+LLM (strongest baseline)
        print("\n--- Paired t-test: MV-Align(7B) [model 3] vs TF-IDF+LLM [model 2] ---")
        print("  %-15s %10s %12s %12s" % ("Metric", "t-stat", "p-value", "Sig"))
        print("  " + "-" * 55)
        for mname in overall_metrics:
            mv = np.array(model_data[3][mname])
            bl = np.array(model_data[2][mname])
            t_stat, p_val = stats.ttest_rel(mv, bl)
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = "n.s."
            print("  %-15s %10.4f %12.6f %12s" % (mname, t_stat, p_val, sig))

        # MV-Align(7B) vs TF-IDF
        print("\n--- Paired t-test: MV-Align(7B) [model 3] vs TF-IDF [model 1] ---")
        print("  %-15s %10s %12s %12s" % ("Metric", "t-stat", "p-value", "Sig"))
        print("  " + "-" * 55)
        for mname in overall_metrics:
            mv = np.array(model_data[3][mname])
            bl = np.array(model_data[1][mname])
            t_stat, p_val = stats.ttest_rel(mv, bl)
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = "n.s."
            print("  %-15s %10.4f %12.6f %12s" % (mname, t_stat, p_val, sig))

        # MV-Align(7B) vs ID-only
        print("\n--- Paired t-test: MV-Align(7B) [model 3] vs ID-only [model 0] ---")
        print("  %-15s %10s %12s %12s" % ("Metric", "t-stat", "p-value", "Sig"))
        print("  " + "-" * 55)
        for mname in overall_metrics:
            mv = np.array(model_data[3][mname])
            bl = np.array(model_data[0][mname])
            t_stat, p_val = stats.ttest_rel(mv, bl)
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = "n.s."
            print("  %-15s %10.4f %12.6f %12s" % (mname, t_stat, p_val, sig))

        # ---- LaTeX-ready rows ----
        print("\n--- LaTeX Table Rows (for copy-paste) ---")
        for mi in range(4):
            parts = []
            for mname in metric_order:
                m = means[mi][mname]
                s = stds[mi][mname]
                parts.append("%.2f$\\pm$%.2f" % (m, s))
            row_str = " & ".join(parts)
            print("  %s & %s \\\\" % (MODEL_NAMES[mi], row_str))

        print()


if __name__ == "__main__":
    main()
