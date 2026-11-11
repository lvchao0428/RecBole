#!/usr/bin/env python3
"""
Test-only (and optional valid) re-evaluation for existing checkpoints.

Used to fix UniSRec GTS runs that trained with --only_phase_a but never
logged a true test_result (CSV previously stored last-epoch valid MRR).

Usage:
    python scripts/eval_ckpt_test.py --ckpt_dir saved/unisrec_beauty_gts_seed42
    python scripts/eval_ckpt_test.py --ckpt_list logs/unisrec_eval_list.txt \
        --out logs/unisrec_gts_test_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pick_pth(ckpt_dir: Path) -> Path:
    if ckpt_dir.is_file() and ckpt_dir.suffix == ".pth":
        return ckpt_dir
    pths = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not pths:
        raise FileNotFoundError(f"No .pth in {ckpt_dir}")
    return pths[-1]


def _metric(d, key="mrr@10"):
    if not d:
        return ""
    if key in d:
        return f"{float(d[key]):.6f}"
    # RecBole sometimes uses capitalized keys
    for k, v in d.items():
        if str(k).lower() == key.lower():
            return f"{float(v):.6f}"
    return ""


def run_one(ckpt_dir: Path, gpu_id: str, do_valid: bool):
    from recbole.quick_start import load_data_and_model
    from recbole.trainer import Trainer

    pth = pick_pth(ckpt_dir)
    print(f"\n=== {ckpt_dir.name} ← {pth.name} ===")
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(str(pth))
    if gpu_id is not None and str(gpu_id) != "":
        config["device"] = f"cuda:{gpu_id}" if str(gpu_id).isdigit() else gpu_id
        model = model.to(config["device"])

    trainer = Trainer(config, model)
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
    valid_result = None
    if do_valid:
        valid_result = trainer.evaluate(valid_data, load_best_model=False, show_progress=False)

    row = {
        "tag": ckpt_dir.name if ckpt_dir.is_dir() else ckpt_dir.stem,
        "pth": pth.name,
        "dataset": str(config["dataset"]),
        "model": str(config["model"]),
        "seed": str(config["seed"]) if "seed" in config.final_config_dict else "",
        "test_mrr@10": _metric(test_result, "mrr@10"),
        "test_ndcg@10": _metric(test_result, "ndcg@10"),
        "test_hit@10": _metric(test_result, "hit@10"),
        "valid_mrr@10": _metric(valid_result, "mrr@10") if valid_result else "",
    }
    print(
        f"  test MRR@10={row['test_mrr@10']} NDCG@10={row['test_ndcg@10']} "
        f"Hit@10={row['test_hit@10']} valid_mrr={row['valid_mrr@10'] or '—'}"
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, default="")
    ap.add_argument("--ckpt_list", type=str, default="")
    ap.add_argument("--out", type=str, default="logs/ckpt_test_eval.csv")
    ap.add_argument("--gpu_id", type=str, default="0")
    ap.add_argument("--also_valid", action="store_true", help="Also eval valid split")
    args = ap.parse_args()

    dirs: list[Path] = []
    if args.ckpt_dir:
        dirs.append(Path(args.ckpt_dir))
    if args.ckpt_list:
        for line in Path(args.ckpt_list).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                dirs.append(Path(line))
    if not dirs:
        raise SystemExit("Provide --ckpt_dir or --ckpt_list")

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in dirs:
        d = d if d.is_absolute() else ROOT / d
        if not d.exists():
            print(f"SKIP missing {d}")
            rows.append({"tag": d.name, "error": "missing"})
            continue
        try:
            rows.append(run_one(d, args.gpu_id, args.also_valid))
        except Exception as e:
            print(f"FAIL {d}: {e}")
            rows.append({"tag": d.name, "error": str(e)})

    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
