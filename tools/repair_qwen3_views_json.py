#!/usr/bin/env python3
"""Rebuild views.json for qwen3_4views split dirs when .npy files exist but metadata is missing.

Usage (project root):
  python tools/repair_qwen3_views_json.py --split_dir dataset/book-crossing/qwen3_4views
  python tools/repair_qwen3_views_json.py --split_dir dataset/book-crossing/qwen3_4views --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Must match tools/build_item_text_emb_qwen3_hf.py PROMPT_PRESETS["multiview"]
MULTIVIEW_PROMPTS = [
    "Identify the item: [TITLE] {text}",
    "What are the main functions and features of [TITLE] {text}?",
    "Who is the target audience or user group for [TITLE] {text}?",
    "Categorize the item [TITLE] {text} and describe its context.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--split_dir",
        required=True,
        help="Directory with view_{i}.npy (e.g. dataset/book-crossing/qwen3_4views)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print metadata only; do not write views.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = args.split_dir
    if not os.path.isdir(split_dir):
        print(f"ERROR: split_dir not found: {split_dir}", file=sys.stderr)
        sys.exit(1)

    meta_path = os.path.join(split_dir, "views.json")
    view_files = sorted(
        f for f in os.listdir(split_dir) if f.startswith("view_") and f.endswith(".npy")
    )
    if not view_files:
        print(f"ERROR: no view_*.npy in {split_dir}", file=sys.stderr)
        sys.exit(1)

    prompts = []
    num_items = None
    for file_name in view_files:
        idx = int(file_name.replace("view_", "").replace(".npy", ""))
        path = os.path.join(split_dir, file_name)
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            print(f"ERROR: {path} is not 2D: {arr.shape}", file=sys.stderr)
            sys.exit(1)
        if num_items is None:
            num_items = int(arr.shape[0])
        elif arr.shape[0] != num_items:
            print(
                f"ERROR: row mismatch {path} {arr.shape[0]} vs {num_items}",
                file=sys.stderr,
            )
            sys.exit(1)
        prompt = (
            MULTIVIEW_PROMPTS[idx]
            if idx < len(MULTIVIEW_PROMPTS)
            else f"View {idx} prompt (repaired)"
        )
        prompts.append(
            {
                "index": idx,
                "prompt": prompt,
                "file": file_name,
                "vector_dim": int(arr.shape[1]),
            }
        )

    prompts.sort(key=lambda x: x["index"])
    meta = {
        "num_items": num_items,
        "num_prompts": len(prompts),
        "dtype": str(arr.dtype),
        "prompts": prompts,
        "repaired": True,
        "note": "Auto-rebuilt by tools/repair_qwen3_views_json.py",
    }

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    if args.dry_run:
        print(f"[dry-run] would write {meta_path}")
        return

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Wrote {meta_path} ({len(prompts)} views, num_items={num_items})")


if __name__ == "__main__":
    main()
