#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 .npy 嵌入矩阵裁剪为 RecBole 当前数据集的 n_items 行（含 PAD 行 0）。

用于修复「Qwen 向量比 n_items 多 1 行」这类问题：先确认 mapping CSV 多出的那一行
在末尾或确认为冗余，再执行裁剪。

用法（项目根目录）:
  python tools/trim_embedding_to_n_items.py \\
    --dataset book-crossing \\
    --config sasrec_book_crossing_plain.yaml \\
    --emb dataset/book-crossing/item_text_emb.qwen3.base.npy \\
    --emb dataset/book-crossing/item_text_emb.qwen3.multiview.npy \\
    --inplace

默认 dry-run；加 --inplace 会覆盖原文件（同目录下先备份为 <原文件名>.bak）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from recbole.config.configurator import Config
from recbole.data.utils import create_dataset


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="book-crossing")
    p.add_argument("--config", nargs="+", default=["sasrec_book_crossing_plain.yaml"])
    p.add_argument("--emb", action="append", required=True, help="Path to .npy (repeatable)")
    p.add_argument("--inplace", action="store_true", help="Overwrite; backup as .bak.npy")
    args = p.parse_args()

    os.chdir(ROOT_DIR)
    _argv = sys.argv
    sys.argv = [_argv[0]]
    try:
        cfg = Config(
            model="BPR",
            dataset=args.dataset,
            config_file_list=list(args.config),
        )
        dataset = create_dataset(cfg)
        n_items = dataset.num(dataset.iid_field)
    finally:
        sys.argv = _argv

    print(f"n_items={n_items} (dataset={args.dataset}, config={args.config})")
    rc = 0
    for rel in args.emb:
        path = rel if os.path.isabs(rel) else os.path.join(ROOT_DIR, rel)
        if not os.path.isfile(path):
            print(f"SKIP missing: {path}")
            rc = 1
            continue
        a = np.load(path, mmap_mode="r")
        if a.shape[0] == n_items:
            print(f"OK already {n_items} rows: {path}")
            continue
        if a.shape[0] < n_items:
            print(f"FAIL fewer rows than n_items: {a.shape} vs {n_items} -> {path}")
            rc = 1
            continue
        if a.shape[0] == n_items + 1:
            trimmed = np.asarray(a[:n_items], dtype=a.dtype)
            print(f"TRIM {a.shape} -> {trimmed.shape}  ({path})")
            if args.inplace:
                bak = path + ".bak"
                shutil.copy2(path, bak)
                np.save(path, trimmed)
                print(f"  saved; backup -> {bak}")
            else:
                print("  dry-run (pass --inplace to write)")
        else:
            print(f"FAIL unexpected shape {a.shape} (expected {n_items} or {n_items+1}): {path}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
