#!/usr/bin/env python3
"""Compute main-table mean +- std across seeds (2025/2024/42).

Input files:
  - seed2025 file in the same tabular format as main_table_seed2025.txt
  - combined seed2024+seed42 file in the same format as main_table_seed2024_seed42.txt

Output:
  - LaTeX-ready table rows for `main_original.tex` main table.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple


MODEL_ALIASES = {
    "sasrec + base50ep": "id_only",
    "sasrec + 50ep": "id_only",
    "tfidf": "tfidf",
    "tfidf + llm": "tfidf_llm",
    "multi-view 7b": "mv7b",
}


LATEX_MODEL_NAMES = {
    "id_only": r"\base (ID-only)",
    "tfidf": r"\base + \tfidf",
    "tfidf_llm": r"\base + \tfidf + \llm (7B)",
    "mv7b": r"\model (7B)",
}


DATASET_NAMES = {"beauty": "Amazon Beauty", "toy": "Amazon Toys\\&Games"}


TARGET_METRICS = [
    ("hit@10", "HR@10"),
    ("ndcg@10", "NDCG@10"),
    ("mrr@10", "MRR@10"),
    ("Recall_new@10", "HR@10"),
    ("NDCG_new@10", "NDCG@10"),
    ("MRR_new@10", "MRR@10"),
    ("Recall_few@10", "HR@10"),
    ("MRR_few@10", "MRR@10"),
    ("Recall_frequent@10", "HR@10"),
    ("MRR_frequent@10", "MRR@10"),
]


def first_col_index(header_cells: List[str], col_name: str) -> int:
    for idx, cell in enumerate(header_cells):
        if cell.strip() == col_name:
            return idx
    raise ValueError(f"Column not found in header: {col_name}")


def safe_float(cell: str) -> float:
    cell = cell.strip()
    if not cell:
        raise ValueError("Empty numeric cell")
    return float(cell)


def parse_seed_file(
    path: Path,
    fixed_seed: str | None = None,
    has_seed_markers: bool = False,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Parse into: seed -> dataset -> model_key -> metric_name -> value(percentage)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    result: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    header_cells: List[str] | None = None
    col_idx: Dict[str, int] = {}

    current_seed = fixed_seed
    current_dataset = None

    for raw in lines:
        line = raw.strip("\n")
        if not line.strip():
            continue

        lower = line.lower().strip()
        if has_seed_markers and lower.startswith("seed "):
            current_seed = line.split()[-1]
            current_dataset = None
            continue

        cells = line.split("\t")

        # Header line starts with first cell empty and second cell "数据集"
        if "数据集" in cells:
            header_cells = cells
            col_idx = {m: first_col_index(header_cells, m) for m, _ in TARGET_METRICS}
            dataset_col = first_col_index(header_cells, "数据集")
            continue

        if header_cells is None:
            continue
        if current_seed is None:
            raise ValueError(f"Cannot infer seed for line in {path}: {line}")

        model_raw = cells[0].strip().lower() if cells else ""
        if model_raw not in MODEL_ALIASES:
            continue

        model_key = MODEL_ALIASES[model_raw]
        dataset_val = cells[dataset_col].strip().lower() if len(cells) > dataset_col else ""
        if dataset_val in DATASET_NAMES:
            current_dataset = dataset_val
        if current_dataset is None:
            raise ValueError(f"Dataset missing for model row: {line}")

        result.setdefault(current_seed, {}).setdefault(current_dataset, {}).setdefault(model_key, {})
        for metric_col, _ in TARGET_METRICS:
            idx = col_idx[metric_col]
            if len(cells) <= idx:
                raise ValueError(f"Missing column {metric_col} in line: {line}")
            val = safe_float(cells[idx]) * 100.0
            result[current_seed][current_dataset][model_key][metric_col] = val

    return result


def merge_nested_dict(
    a: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    b: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    out = dict(a)
    for seed, seed_data in b.items():
        out.setdefault(seed, {})
        for dataset, ds_data in seed_data.items():
            out[seed].setdefault(dataset, {})
            for model_key, model_data in ds_data.items():
                out[seed][dataset].setdefault(model_key, {})
                out[seed][dataset][model_key].update(model_data)
    return out


def mean_std(values: List[float], ddof: int) -> Tuple[float, float]:
    if len(values) == 0:
        raise ValueError("No values provided")
    if len(values) == 1:
        return values[0], 0.0
    if ddof == 0:
        mu = mean(values)
        var = sum((x - mu) ** 2 for x in values) / len(values)
        return mu, math.sqrt(var)
    if ddof == 1:
        return mean(values), stdev(values)
    raise ValueError("Only ddof 0 or 1 is supported")


def build_latex_rows(
    all_data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    seed_order: List[str],
    ddof: int,
) -> str:
    dataset_order = ["beauty", "toy"]
    model_order = ["id_only", "tfidf", "tfidf_llm", "mv7b"]

    lines: List[str] = []
    lines.append("% Auto-generated by compute_main_table_mean_std.py")
    lines.append(f"% seeds = {', '.join(seed_order)}, ddof = {ddof}")
    lines.append("")

    for ds in dataset_order:
        lines.append(f"% {DATASET_NAMES[ds]}")
        for mk in model_order:
            row_cells = [LATEX_MODEL_NAMES[mk]]
            for metric_col, _display in TARGET_METRICS:
                vals = [all_data[s][ds][mk][metric_col] for s in seed_order]
                mu, sd = mean_std(vals, ddof=ddof)
                row_cells.append(f"{mu:.2f} $\\pm$ {sd:.2f}")
            lines.append(" & ".join(row_cells) + r" \\")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute main table mean ± std across seeds")
    parser.add_argument(
        "--seed2025",
        default="paper_sigir/main_table_seed2025.txt",
        help="Path to seed 2025 table file",
    )
    parser.add_argument(
        "--seed2024_42",
        default="paper_sigir/main_table_seed2024_seed42.txt",
        help="Path to combined seed 2024 and seed 42 table file",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        choices=[0, 1],
        default=1,
        help="Std ddof: 1(sample std, default) or 0(population std)",
    )
    args = parser.parse_args()

    seed2025_data = parse_seed_file(Path(args.seed2025), fixed_seed="2025", has_seed_markers=False)
    seed2024_42_data = parse_seed_file(
        Path(args.seed2024_42), fixed_seed=None, has_seed_markers=True
    )
    all_data = merge_nested_dict(seed2025_data, seed2024_42_data)

    seed_order = ["2025", "2024", "42"]
    for seed in seed_order:
        if seed not in all_data:
            raise ValueError(f"Missing seed in parsed data: {seed}")

    output = build_latex_rows(all_data, seed_order=seed_order, ddof=args.ddof)
    print(output)


if __name__ == "__main__":
    main()
