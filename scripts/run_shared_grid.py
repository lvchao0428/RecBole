#!/usr/bin/env python3
"""
Shared Small Grid Search Runner

按照老师 0717 建议：只开一个很小且共享的 valid 网格，
lr/dropout/weight_decay 各 2-3 个值，所有模型同预算调参。
绝对不根据 test 反复选配置。

Usage:
    python scripts/run_shared_grid.py \
        --models tfidf llm mv \
        --dataset Amazon_Beauty \
        --seed 2025 \
        --gpu 0
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


GRID = {
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "attn_dropout_prob": [0.1, 0.3, 0.5],
    "weight_decay": [0.0, 1e-5, 1e-4],
}

MODEL_CONFIGS = {
    "tfidf": {
        "yaml": "sasrec_align_base_stratified_v3_ts.yaml",
        "model": "SASRecAlignV3",
    },
    "llm": {
        "yaml": "sasrec_align_qwen3_stratified_v3_ts.yaml",
        "model": "SASRecAlignV3",
    },
    "mv": {
        "yaml": "sasrec_align_multi_view_v3_stratified_ts.yaml",
        "model": "SASRecAlignMultiViewV3",
    },
}


def generate_grid_configs(base_yaml: str, model_name: str, dataset: str, seed: int):
    """Generate all grid search configurations."""
    configs = []
    for lr, dropout, wd in itertools.product(
        GRID["learning_rate"], GRID["attn_dropout_prob"], GRID["weight_decay"]
    ):
        config_id = f"lr{lr}_do{dropout}_wd{wd}"
        configs.append({
            "id": config_id,
            "yaml": base_yaml,
            "model": model_name,
            "dataset": dataset,
            "seed": seed,
            "overrides": {
                "learning_rate": lr,
                "attn_dropout_prob": dropout,
                "hidden_dropout_prob": dropout,
                "weight_decay": wd,
            },
        })
    return configs


def run_single_config(config: dict, gpu: int, python_path: str, dry_run: bool = False):
    """Run a single grid configuration."""
    override_args = []
    for k, v in config["overrides"].items():
        override_args.extend([f"--{k}={v}"])

    cmd = [
        python_path, str(ROOT / "scripts" / "two_phase_train.py"),
        "--model", config["model"],
        "--dataset", config["dataset"],
        "--config_files", str(ROOT / config["yaml"]),
        "--seed", str(config["seed"]),
        *override_args,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    log_name = f"grid_{config['dataset'].replace('Amazon_', '').lower()}_{config['model'].lower()}_{config['id']}_seed{config['seed']}.log"
    log_path = ROOT / "logs" / log_name

    if dry_run:
        return {"id": config["id"], "cmd": " ".join(cmd), "log": str(log_path)}

    print(f"  Running {config['id']}... ", end="", flush=True)
    with open(log_path, 'w') as log_file:
        result = subprocess.run(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    return {"id": config["id"], "exit_code": result.returncode, "log": str(log_path)}


def extract_valid_mrr(log_path: str) -> float:
    """Extract best valid MRR@10 from log file."""
    best_mrr = 0.0
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if 'valid' in line.lower() and 'mrr@10' in line.lower():
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if 'mrr@10' in p.lower() and i + 1 < len(parts):
                            try:
                                val = float(parts[i + 1].strip(','))
                                best_mrr = max(best_mrr, val)
                            except ValueError:
                                pass
    except FileNotFoundError:
        pass
    return best_mrr


def main():
    parser = argparse.ArgumentParser(description="Shared Small Grid Search")
    parser.add_argument("--models", nargs="+", default=["tfidf", "llm", "mv"],
                       choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--dataset", type=str, default="Amazon_Beauty")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("SHARED SMALL GRID SEARCH")
    print(f"Models: {args.models}")
    print(f"Grid: {len(GRID['learning_rate'])} × {len(GRID['attn_dropout_prob'])} × {len(GRID['weight_decay'])} = "
          f"{len(GRID['learning_rate']) * len(GRID['attn_dropout_prob']) * len(GRID['weight_decay'])} configs per model")
    print(f"Total runs: {len(args.models) * 27}")
    print("=" * 70)

    all_results = {}

    for model_key in args.models:
        mc = MODEL_CONFIGS[model_key]
        print(f"\n{'='*40}")
        print(f"Model: {model_key} ({mc['model']})")
        print(f"{'='*40}")

        configs = generate_grid_configs(mc["yaml"], mc["model"], args.dataset, args.seed)

        if args.dry_run:
            for c in configs[:3]:
                r = run_single_config(c, args.gpu, args.python, dry_run=True)
                print(f"  [DRY] {r['id']}: {r['cmd'][:100]}...")
            print(f"  ... ({len(configs)} total)")
            continue

        model_results = []
        for c in configs:
            r = run_single_config(c, args.gpu, args.python)
            valid_mrr = extract_valid_mrr(r["log"])
            r["valid_mrr"] = valid_mrr
            model_results.append(r)
            print(f"valid MRR@10={valid_mrr:.6f}")

        model_results.sort(key=lambda x: x["valid_mrr"], reverse=True)
        all_results[model_key] = model_results

        print(f"\n  Top-3 configs for {model_key}:")
        for i, r in enumerate(model_results[:3], 1):
            print(f"    {i}. {r['id']}: valid MRR={r['valid_mrr']:.6f}")

    if not args.dry_run:
        output_file = args.output or f"grid_results_{args.dataset.replace('Amazon_', '').lower()}_seed{args.seed}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")

        print("\n" + "=" * 70)
        print("GRID SEARCH SUMMARY (Best per model, selected by VALID only)")
        print("=" * 70)
        print(f"{'Model':<10} {'Best Config':<25} {'Valid MRR@10':>12}")
        print("-" * 50)
        for model_key, results in all_results.items():
            if results:
                best = results[0]
                print(f"{model_key:<10} {best['id']:<25} {best['valid_mrr']:>12.6f}")


if __name__ == "__main__":
    main()
