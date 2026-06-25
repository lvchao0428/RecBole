#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract Coverage_new@10 (and related) from seed files and run_metrics JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_seed_coverage(seed_dir: Path) -> list[dict]:
    rows = []
    for seed_file in sorted(seed_dir.glob("seed*.txt")):
        m = re.search(r"seed(\d+)", seed_file.stem)
        if not m:
            continue
        seed = int(m.group(1))
        lines = seed_file.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) < 3:
            continue
        header = [h.strip() for h in lines[1].split("\t") if h.strip()]
        idx = {name: i for i, name in enumerate(header)}

        def row_metrics(line: str, dataset: str, config: str) -> dict | None:
            parts = line.split("\t")
            if len(parts) < len(header):
                return None
            out = {"seed": seed, "dataset": dataset, "config": config, "source": seed_file.name}
            for key in (
                "Coverage_new@10",
                "Coverage_few@10",
                "Coverage_frequent@10",
            ):
                if key in idx:
                    val = parts[idx[key]].strip().replace("%", "")
                    try:
                        out[key] = float(val)
                    except ValueError:
                        out[key] = val
            return out

        # Beauty rows 2-5, Toys rows 9-12 (0-based line index in file)
        beauty_configs = ["ID-only", "TF-IDF", "TF-IDF+LLM", "MV-Align(7B)"]
        for i, cfg in enumerate(beauty_configs):
            line_idx = 2 + i
            if line_idx < len(lines):
                r = row_metrics(lines[line_idx], "beauty", cfg)
                if r:
                    rows.append(r)

        toys_configs = ["ID-only", "TF-IDF", "TF-IDF+LLM", "MV-Align(7B)"]
        for i, cfg in enumerate(toys_configs):
            line_idx = 9 + i
            if line_idx < len(lines):
                r = row_metrics(lines[line_idx], "toys", cfg)
                if r:
                    rows.append(r)
    return rows


def parse_run_metrics(run_metrics_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(run_metrics_dir.glob("*.txt")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        test = data.get("test_result") or data.get("phase_b_result", {}).get("test_result")
        if not test:
            continue
        row = {
            "source": path.name,
            "dataset": data.get("dataset") or data.get("config", {}).get("dataset"),
            "model": data.get("model"),
            "seed": data.get("seed"),
            "variant": data.get("variant_features") or data.get("variant_label"),
        }
        for key in test:
            if key.startswith("Coverage_"):
                row[key] = test[key]
        if any(k.startswith("Coverage_") for k in row):
            rows.append(row)
    return rows


def format_table(seed_rows: list[dict], metric_rows: list[dict]) -> str:
    lines = [
        "Coverage extraction (auto-generated)",
        "=" * 72,
        "",
        "## Beauty/Toys from seed files (Coverage_new@10)",
        f"{'seed':>6}  {'dataset':<8}  {'config':<16}  {'Cov_new@10':>12}",
        "-" * 52,
    ]
    for r in sorted(seed_rows, key=lambda x: (x["dataset"], x["config"], x["seed"])):
        val = r.get("Coverage_new@10", "NA")
        lines.append(
            f"{r['seed']:>6}  {r['dataset']:<8}  {r['config']:<16}  {val!s:>12}"
        )

    lines.extend(["", "## Grocery/BC/others from run_metrics", "-" * 52])
    for r in metric_rows:
        cov = r.get("Coverage_new@10", "NA")
        lines.append(
            f"{r.get('source', '?')[:40]:<40}  seed={r.get('seed')}  Coverage_new@10={cov}"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_dir", default="paper_recsys")
    parser.add_argument("--run_metrics_dir", default="run_metrics")
    parser.add_argument(
        "--output", default="paper_recsys/coverage_table_20260625.txt"
    )
    args = parser.parse_args()

    seed_rows = parse_seed_coverage(Path(args.seed_dir))
    metric_rows = parse_run_metrics(Path(args.run_metrics_dir))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(format_table(seed_rows, metric_rows), encoding="utf-8")
    print(f"Wrote {out} ({len(seed_rows)} seed rows, {len(metric_rows)} run_metrics rows)")


if __name__ == "__main__":
    main()
